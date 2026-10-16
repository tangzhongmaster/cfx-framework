"""
Les backbones conformes : chacun produit hat_y(x) et sigma(x), calibres par
split-CQR. Interface commune (predict_interval) pour que le reste du
framework (margin.py, search.py) n'ait jamais a savoir quel backbone tourne
dessous -- exactement le point du papier (Section 4, "backbone-agnostic").
"""

import numpy as np
import torch
import torch.nn as nn

from device import DEVICE


class QuantileNet(nn.Module):
    """MLP -> quantiles bas/haut. Utilise pour raw_context ET spci (meme
    architecture, entree differente)."""

    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        out = self.net(x)
        lo, hi = out[:, 0], out[:, 1]
        # assure lo <= hi
        lo, hi = torch.minimum(lo, hi), torch.maximum(lo, hi)
        return lo, hi


def pinball_loss(pred, target, quantile):
    diff = target - pred
    return torch.maximum(quantile * diff, (quantile - 1) * diff).mean()


def train_quantile_net(X, y, alpha=0.1, epochs=200, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    model = QuantileNet(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    for _ in range(epochs):
        opt.zero_grad()
        lo, hi = model(Xt)
        loss = pinball_loss(lo, yt, alpha / 2) + pinball_loss(hi, yt, 1 - alpha / 2)
        loss.backward()
        opt.step()
    return model


def calibrate_split_cqr(model, X_calib, y_calib):
    """Retourne q_hat, la correction de conformite (split-CQR)."""
    with torch.no_grad():
        lo, hi = model(torch.tensor(X_calib, dtype=torch.float32, device=DEVICE))
    lo, hi = lo.cpu().numpy(), hi.cpu().numpy()
    scores = np.maximum(lo - y_calib, y_calib - hi)
    q_hat = np.quantile(scores, 0.9)  # 1 - alpha, alpha = 0.1
    return float(q_hat)


class Backbone:
    """Interface commune. predict_interval(x) -> (hat_y, sigma) tel que
    C_alpha(x) = [hat_y - q_{1-alpha} sigma, hat_y + q_{1-alpha} sigma]
    avec q_{1-alpha} := 1 (sigma absorbe deja la calibration ici, cf papier
    Section 3 -- w(x) = 2 q_{1-alpha} sigma(x))."""

    def predict_interval(self, x: torch.Tensor):
        raise NotImplementedError

    def predict_interval_np(self, x: np.ndarray):
        xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            hat_y, sigma = self.predict_interval(xt)
        return float(hat_y), float(sigma)


class RawContextBackbone(Backbone):
    """CQR standard : hat_y(X_t), sigma(X_t) directement sur le contexte brut."""

    def __init__(self, data_bundle):
        self.model = train_quantile_net(data_bundle.X_train, data_bundle.y_train)
        self.q_hat = calibrate_split_cqr(self.model, data_bundle.X_calib, data_bundle.y_calib)

    def predict_interval(self, x: torch.Tensor):
        x = x.to(DEVICE)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        lo, hi = self.model(x)
        hat_y = (lo + hi) / 2 + 0  # centre du reseau
        sigma = (hi - lo) / 2 + self.q_hat
        return hat_y.squeeze(), sigma.squeeze()


class SPCIBackbone(Backbone):
    """Backbone style SPCI : point forecaster sur le contexte brut, puis un
    second reseau quantile sur la fenetre d'erreurs passees (Section 5.1.5,
    eq:spci-backbone). Notion de contexte differente : le "x" que search.py
    manipule ici est la fenetre de residus, pas le signal brut."""

    def __init__(self, data_bundle, residual_window=10, seed=42):
        self.residual_window = residual_window
        torch.manual_seed(seed)
        # 1) point forecaster sur contexte brut (MSE)
        self.forecaster = nn.Sequential(
            nn.Linear(data_bundle.X_train.shape[1], 64), nn.ReLU(),
            nn.Linear(64, 1),
        ).to(DEVICE)
        opt = torch.optim.Adam(self.forecaster.parameters(), lr=1e-3)
        Xt = torch.tensor(data_bundle.X_train, dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(data_bundle.y_train, dtype=torch.float32, device=DEVICE)
        for _ in range(200):
            opt.zero_grad()
            pred = self.forecaster(Xt).squeeze()
            loss = ((pred - yt) ** 2).mean()
            loss.backward()
            opt.step()

        # 2) residus sur train, fenetres de residus -> quantiles du residu suivant
        # (.cpu() avant .numpy() : _make_residual_windows et train_quantile_net
        # travaillent en numpy, un tensor CUDA ne peut pas etre converti direct)
        with torch.no_grad():
            resid_train = (yt - self.forecaster(Xt).squeeze()).cpu().numpy()
        Xr, yr = self._make_residual_windows(resid_train)
        self.qnet = train_quantile_net(Xr, yr)

        with torch.no_grad():
            Xc = torch.tensor(data_bundle.X_calib, dtype=torch.float32, device=DEVICE)
            resid_calib = (torch.tensor(data_bundle.y_calib, dtype=torch.float32, device=DEVICE)
                           - self.forecaster(Xc).squeeze()).cpu().numpy()
        Xrc, yrc = self._make_residual_windows(resid_calib)
        self.q_hat = calibrate_split_cqr(self.qnet, Xrc, yrc)

    def _make_residual_windows(self, resid):
        w = self.residual_window
        X, y = [], []
        for i in range(w, len(resid)):
            X.append(resid[i - w:i])
            y.append(resid[i])
        return np.array(X), np.array(y)

    def predict_interval_from_residuals(self, resid_window: torch.Tensor, point_forecast):
        resid_window = resid_window.to(DEVICE)
        if resid_window.dim() == 1:
            resid_window = resid_window.unsqueeze(0)
        lo, hi = self.qnet(resid_window)
        # point_forecast : scalaire python (cas par-cas) OU tensor/array un
        # par ligne du batch (cas batche, voir select_cases/select_reference_
        # cases) -- dans les deux cas, le ramener explicitement sur DEVICE
        # pour que l'addition avec lo/hi (deja sur DEVICE) ne plante pas.
        if not torch.is_tensor(point_forecast):
            point_forecast = torch.as_tensor(point_forecast, dtype=torch.float32, device=DEVICE)
        else:
            point_forecast = point_forecast.to(DEVICE)
        hat_y = point_forecast + (lo + hi) / 2
        sigma = (hi - lo) / 2 + self.q_hat
        return hat_y.squeeze(), sigma.squeeze()


class SplitCPBackbone(Backbone):
    """Split CP standard : meme forecaster ponctuel (MLP, MSE) que les
    autres backbones, mais la demi-largeur q est CONSTANTE, calibree une
    fois sur le residu absolu du calibration set, independante du contexte
    x. C'est le controle theorique du papier (Section 'geometrie des
    intervalles') : w(x') = 2q pour tout x', donc dw/dx_t = 0 et, dans la
    case center_fixed_width_free (hat_y(x')=hat_y(X_t)), tout candidat x'
    respectant la contrainte de centre a EXACTEMENT la meme largeur que le
    factuel -- un contrefactuel center-fixed/width-free est donc
    structurellement impossible sous ce backbone, pas juste difficile a
    trouver. A verifier empiriquement (taux de succes = 0 sur cette case),
    pas seulement affirme."""

    def __init__(self, data_bundle, seed=42):
        torch.manual_seed(seed)
        self.forecaster = nn.Sequential(
            nn.Linear(data_bundle.X_train.shape[1], 64), nn.ReLU(),
            nn.Linear(64, 1),
        ).to(DEVICE)
        opt = torch.optim.Adam(self.forecaster.parameters(), lr=1e-3)
        Xt = torch.tensor(data_bundle.X_train, dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(data_bundle.y_train, dtype=torch.float32, device=DEVICE)
        for _ in range(200):
            opt.zero_grad()
            pred = self.forecaster(Xt).squeeze()
            loss = ((pred - yt) ** 2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            Xc = torch.tensor(data_bundle.X_calib, dtype=torch.float32, device=DEVICE)
            resid_abs = (torch.tensor(data_bundle.y_calib, dtype=torch.float32, device=DEVICE)
                         - self.forecaster(Xc).squeeze()).abs().cpu().numpy()
        # meme convention de calibration que calibrate_split_cqr (alpha=0.1) :
        # score = |y - hat_y|, q = quantile empirique (1-alpha) du score.
        self.q = float(np.quantile(resid_abs, 0.9))

    def predict_interval(self, x: torch.Tensor):
        x = x.to(DEVICE)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        hat_y = self.forecaster(x).squeeze()
        # sigma CONSTANT : ne depend pas de x, contrairement a tous les
        # autres backbones -- c'est tout le point de ce backbone.
        sigma = torch.full_like(hat_y, self.q)
        return hat_y, sigma


class AdaptiveRawContextBackbone(RawContextBackbone):
    """4e backbone : memes hat_y(x), sigma(x) de base que RawContextBackbone
    (CQR standard sur le contexte brut), mais q_hat n'est PAS fige apres la
    calibration -- il est mis a jour en ligne, pas a pas, par la regle ACI
    (Gibbs & Candes 2021, "Adaptive Conformal Inference Under Distribution
    Shift") :
        q_hat_{t+1} = q_hat_t + gamma * (alpha - err_t)
    avec err_t = 1 si y_t est tombe hors de l'intervalle NON corrige du
    reseau [lo(x_t), hi(x_t)] (avant d'ajouter q_hat), 0 sinon. C'est le
    seul backbone ou q_hat garde une trace du temps qui passe -- necessaire
    (et SEULEMENT necessaire) pour le mecanisme de reecriture causale :
    rejouer une sequence H_t', H_t'+1, ... vers l'avant et observer q_hat
    (donc l'intervalle) reagir en chaine aux valeurs rencontrees en chemin
    (voir replay_forward ci-dessous). Les 3 autres backbones n'ont rien a y
    gagner : leur q_hat est soit une constante par construction (Split CP),
    soit une fonction PURE du contexte courant (RawContext, SPCI) -- jamais
    une fonction de l'historique des mises a jour."""

    def __init__(self, data_bundle, alpha=0.1, gamma=0.01):
        super().__init__(data_bundle)
        self.alpha = alpha
        self.gamma = gamma
        self.q_hat_init = self.q_hat          # valeur calibree classique (split-CQR),
                                                 # conservee pour pouvoir reinitialiser
                                                 # une branche contrefactuelle
        self.q_hat_history = [self.q_hat]

    def reset_q_hat(self, value=None):
        """Reinitialise q_hat (typiquement avant de rejouer une nouvelle
        branche contrefactuelle depuis un point de remontee donne) -- sans
        ca, l'etat laisse par une branche precedente contaminerait la
        branche suivante."""
        self.q_hat = self.q_hat_init if value is None else value
        self.q_hat_history = [self.q_hat]

    def update(self, x: torch.Tensor, y_true: float):
        """Un pas ACI : observe (x, y_true) et met a jour self.q_hat.
        A appeler sequentiellement, UNE FOIS PAR PAS DE TEMPS, dans l'ordre
        temporel -- c'est cet ordre qui donne un sens au mot 'adaptatif'
        (contrairement a hat_y(x)/sigma(x) qui sont de simples evaluations
        de fonction, ca ne depend pas de l'ordre d'appel)."""
        with torch.no_grad():
            xi = x.to(DEVICE)
            xi = xi.unsqueeze(0) if xi.dim() == 1 else xi
            lo, hi = self.model(xi)
            lo, hi = lo.squeeze().item(), hi.squeeze().item()
        err = 1.0 if (y_true < lo or y_true > hi) else 0.0
        self.q_hat = max(0.0, self.q_hat + self.gamma * (self.alpha - err))
        self.q_hat_history.append(self.q_hat)
        return err


def replay_forward(backbone: "AdaptiveRawContextBackbone", X_seq, y_seq, q_hat_start=None):
    """Rejoue la sequence (X_seq, y_seq), DANS L'ORDRE, pas a pas, a partir
    de q_hat_start (la valeur de q_hat au moment de la remontee dans le
    passe -- None = repart de la valeur calibree initiale q_hat_init). A
    chaque pas : predit avec le q_hat COURANT (celui accumule par les pas
    precedents de CETTE sequence), puis observe y_true et met a jour q_hat
    pour le pas suivant -- c'est la chaine causale discutee : remonter a un
    pas donne, rejouer vers l'avant (eventuellement une sequence
    contrefactuelle, differente de l'historique reel a partir de ce point),
    et voir l'intervalle reagir en chaine a q_hat qui bouge a chaque pas.

    Fonction PURE vis-a-vis de l'etat a long terme du backbone : sauvegarde
    puis restaure q_hat et son historique autour de l'appel, pour qu'on
    puisse rejouer plusieurs branches contrefactuelles (meme point de
    depart, suites differentes) sans qu'une branche contamine la suivante.

    Retourne une liste de dicts {hat_y, sigma, q_hat, err}, un par pas de
    X_seq/y_seq, dans l'ordre (hat_y/sigma/q_hat sont les valeurs utilisees
    POUR PREDIRE ce pas, err est l'erreur observee a ce pas une fois y_true
    connu)."""
    saved_q_hat, saved_history = backbone.q_hat, backbone.q_hat_history
    backbone.q_hat = backbone.q_hat_init if q_hat_start is None else q_hat_start
    backbone.q_hat_history = [backbone.q_hat]

    trace = []
    try:
        for x_np, y_true in zip(X_seq, y_seq):
            xt = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                hat_y, sigma = backbone.predict_interval(xt)
            step = {"hat_y": float(hat_y), "sigma": float(sigma), "q_hat": backbone.q_hat}
            step["err"] = backbone.update(xt, float(y_true))
            trace.append(step)
    finally:
        backbone.q_hat, backbone.q_hat_history = saved_q_hat, saved_history

    return trace


class ForestBackbone(Backbone):
    """Foret de regression quantile -- non-differentiable, utilisee pour
    exposer le diagnostic 'why naive search fails' (Section 5.1.2)."""

    def __init__(self, data_bundle):
        from sklearn.ensemble import GradientBoostingRegressor
        self.lo_model = GradientBoostingRegressor(loss="quantile", alpha=0.05)
        self.hi_model = GradientBoostingRegressor(loss="quantile", alpha=0.95)
        self.lo_model.fit(data_bundle.X_train, data_bundle.y_train)
        self.hi_model.fit(data_bundle.X_train, data_bundle.y_train)
        lo_c = self.lo_model.predict(data_bundle.X_calib)
        hi_c = self.hi_model.predict(data_bundle.X_calib)
        scores = np.maximum(lo_c - data_bundle.y_calib, data_bundle.y_calib - hi_c)
        self.q_hat = float(np.quantile(scores, 0.9))

    def predict_interval_np(self, x: np.ndarray):
        x = x.reshape(1, -1)
        lo = self.lo_model.predict(x)[0]
        hi = self.hi_model.predict(x)[0]
        hat_y = (lo + hi) / 2
        sigma = (hi - lo) / 2 + self.q_hat
        return float(hat_y), float(sigma)

    def predict_interval(self, x):
        raise NotImplementedError("ForestBackbone n'est pas differentiable, utiliser predict_interval_np")


class SPCIBackboneAdapter(Backbone):
    """Adapte SPCIBackbone a l'interface generique predict_interval(x).
    Ici 'x' EST directement la fenetre de residus (pas le contexte brut) --
    c'est cette fenetre que search.py va perturber, puisque c'est elle qui
    pilote la largeur pour ce backbone (Section 5.1.5). Le point_forecast
    (issu du contexte brut, hors de portee de cette recherche) est fige a
    sa valeur factuelle : on explique la largeur via l'historique de
    residus, le mecanisme de centre reste un point fixe externe pour cette
    case de la grille, coherent avec eq:spci-backbone."""

    def __init__(self, spci_backbone: "SPCIBackbone", point_forecast: float):
        self.spci = spci_backbone
        self.point_forecast = point_forecast

    def predict_interval(self, x: torch.Tensor):
        x = x.to(DEVICE)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        hat_y, sigma = self.spci.predict_interval_from_residuals(x, self.point_forecast)
        return hat_y, sigma

    def predict_interval_np(self, x: np.ndarray):
        xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            hat_y, sigma = self.predict_interval(xt)
        return float(hat_y), float(sigma)


def build_test_series(X_test, L):
    """Reconstruit la SERIE de test (n+L-1, F) a partir des fenetres consecutives
    (pas 1, aplaties temps-majeur) : la fenetre de la ligne m est serie[m:m+L], sa
    cible est serie[m+L, 0]. Sert au REJEU : l'historique reel qui precede la
    fenetre modifiee. Retourne (serie, contigu) ; contigu=False si les fenetres
    ne se recouvrent pas comme attendu (alors le rejeu n'est pas fiable)."""
    X_test = np.asarray(X_test)
    F = X_test.shape[1] // L
    first = X_test[0].reshape(L, F)
    rest = X_test[1:, -F:]
    series = np.concatenate([first, rest], axis=0)
    W = X_test.reshape(len(X_test), L, F)
    contiguous = bool(np.allclose(W[1:, :-1, :], W[:-1, 1:, :], atol=1e-6)) if len(W) > 1 else True
    return series, contiguous


class SPCIReplayAdapter(Backbone):
    """SPCI avec REJEU CAUSAL : la variable contrefactuelle est la fenetre BRUTE
    x' (comme pour RawContext), pas la fenetre de residus. A chaque appel on
    recalcule, a partir de x' et de l'historique reel qui la precede :
      1) les R dernieres previsions f(fenetre_m) le long de x' (chaque fenetre est
         faite de valeurs reelles puis de valeurs modifiees),
      2) les residus e_m = cible_m - f(fenetre_m), la cible_m etant la valeur
         (modifiee) de x' au temps m+L,
      3) l'intervalle : qnet(fenetre de residus) puis centre f(x') + (lo+hi)/2,
         demi-largeur (hi-lo)/2 + q_hat.
    C'est exactement revenir en arriere au debut de la fenetre et rejouer :
    previsions, erreurs, fenetre de residus, intervalle. q_hat est une constante
    de calibration (pas d'etat qui evolue dans SPCI), donc rien d'autre a restaurer.
    Tout est differentiable en x'.

    real_hist : (R, F) tenseur, les R pas de temps REELS juste avant la fenetre."""

    def __init__(self, spci_backbone, real_hist, L):
        self.spci = spci_backbone
        self.hist = real_hist.to(DEVICE)
        self.L = L

    def predict_interval(self, x):
        x = x.to(DEVICE).reshape(-1)
        R, F = self.hist.shape
        L = self.L
        full = torch.cat([self.hist, x.reshape(L, F)], dim=0)                 # (R+L, F)
        wins = torch.stack([full[k:k + L].reshape(-1) for k in range(R + 1)])  # (R+1, L*F)
        fc = self.spci.forecaster(wins).reshape(-1)                            # (R+1,)
        resid = full[L:L + R, 0] - fc[:R]                                      # e_{r-R..r-1}
        lo, hi = self.spci.qnet(resid.unsqueeze(0))
        hat_y = fc[R] + (lo + hi) / 2
        sigma = (hi - lo) / 2 + self.spci.q_hat
        return hat_y.squeeze(), sigma.squeeze()

    def predict_interval_np(self, x):
        xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            hat_y, sigma = self.predict_interval(xt)
        return float(hat_y), float(sigma)


def spci_factual_state(spci_backbone, X_test, y_test):
    """(hat_y, sigma) FACTUELS de toutes les lignes de test, en un seul passage
    (meme resultat que SPCIReplayAdapter sur la fenetre factuelle). NaN pour les
    R premieres lignes (pas assez de residus reels)."""
    R = spci_backbone.residual_window
    X = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)
    n = len(y)
    hat_y = torch.full((n,), float("nan"), device=DEVICE)
    sigma = torch.full((n,), float("nan"), device=DEVICE)
    with torch.no_grad():
        fc = spci_backbone.forecaster(X).reshape(-1)
        resid = y - fc
        wins = resid.unfold(0, R, 1)[:n - R]                  # ligne r-R = resid[r-R:r]
        lo, hi = spci_backbone.qnet(wins)
        hat_y[R:] = fc[R:] + (lo + hi) / 2
        sigma[R:] = (hi - lo) / 2 + spci_backbone.q_hat
    return hat_y, sigma


def build_spci_case_data(spci_backbone: "SPCIBackbone", data_bundle):
    """Construit un DataBundle en espace RESIDUS (pas contexte brut), pour
    que run_experiment puisse traiter SPCI exactement comme les autres
    backbones -- chaque 'x' de ce bundle est deja une fenetre de residus.
    A utiliser a la place de data_bundle quand backbone == SPCI."""
    from data import DataBundle

    def to_residuals(X, y):
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            point_fc = spci_backbone.forecaster(Xt).squeeze().cpu().numpy()
        resid = y - point_fc
        w = spci_backbone.residual_window
        Xr, yr, fc_r = [], [], []
        for i in range(w, len(resid)):
            Xr.append(resid[i - w:i])
            yr.append(resid[i])
            fc_r.append(point_fc[i])
        return np.array(Xr), np.array(yr), np.array(fc_r)

    Xr_train, yr_train, _ = to_residuals(data_bundle.X_train, data_bundle.y_train)
    Xr_calib, yr_calib, _ = to_residuals(data_bundle.X_calib, data_bundle.y_calib)
    Xr_test, yr_test, fc_test = to_residuals(data_bundle.X_test, data_bundle.y_test)

    n_test = len(Xr_test)
    labels_test = data_bundle.labels_test[-n_test:] if len(data_bundle.labels_test) >= n_test else data_bundle.labels_test

    bundle = DataBundle(Xr_train, yr_train, Xr_calib, yr_calib, Xr_test, yr_test,
                         labels_test, data_bundle.mean, data_bundle.std)
    bundle.point_forecast_test = fc_test  # attache pour construire l'adapter par cas
    return bundle


def build_backbone(config, data_bundle):
    from config import Backbone
    if config.backbone == Backbone.RAW_CONTEXT:
        return RawContextBackbone(data_bundle)
    elif config.backbone == Backbone.SPCI:
        return SPCIBackbone(data_bundle)
    elif config.backbone == Backbone.SPLIT_CP:
        return SplitCPBackbone(data_bundle)
    elif config.backbone == Backbone.FOREST:
        return ForestBackbone(data_bundle)
    elif config.backbone == Backbone.TSFM:
        from tsfm_backbone import TSFMWeightedBackbone
        return TSFMWeightedBackbone(data_bundle, config.context_window)
    elif config.backbone == Backbone.ADAPTIVE:
        return AdaptiveRawContextBackbone(data_bundle, alpha=config.adaptive_alpha,
                                           gamma=config.adaptive_gamma)
    raise ValueError(f"backbone inconnu: {config.backbone}")
