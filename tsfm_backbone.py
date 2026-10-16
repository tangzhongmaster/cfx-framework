"""Backbone IBM : W1ACAS (Martinez Gil, O'Donncha, Gifford et al., ICLR 2026,
arXiv 2604.20122) sur Granite TTM zero-shot.

FIDELITE
--------
La partie "conformale" N'EST PAS reimplementee : on appelle directement
`tsfm_public.toolkit.conformal.AdaptiveWeightedConformalScoreWrapper` du paquet
`granite-tsfm` (le meme objet que `get_forecast_conformal_adaptive_online_score`
dans leur depot), avec les MEMES parametres que leur pipeline :
  weighting="uniform", conformal_weights_update=False, stride=1,
  n_batch_update=10, lr=1e-3, prior_past_weights_value="proximity",
  score = erreur absolue, fit() sur les ceil(1/alpha) premiers scores puis
  predict() pas a pas sur le reste.
(Leur w1acas.py passe la cle "n_epochs" mais la classe lit "epochs" : la valeur
par defaut de la classe (2) s'applique, ici aussi puisqu'on appelle la meme classe
avec le meme dictionnaire.)

Mecanisme (conformal.py) : a chaque pas, les poids de memoire w (en [0,1], un par
position de retard) sont mis a jour par AdamW pour que les p-values des
n_batch_update derniers scores soient uniformes (perte = W1 entre la loi
empirique des beta et U[0,1]), puis projetes (norme l1 >= ceil(1/alpha - 1)).
p-value du nouveau score s :   p = 1 - (sum_j w_j 1{s_j <= s} + 1) / (sum_j w_j + 1).
Alarme  <=>  p < alpha.

LIEN AVEC NOTRE MARGE
---------------------
A l'instant t, les scores passes s_j et les poids w sont des CONSTANTES par
rapport a x_t (ils dependent des erreurs passees). p ne depend de x_t que par le
score s_t = |y_t - y_hat(x_t)|. Donc
    alarme  <=>  s_t >= q_t,   q_t = plus petit s_j tel que (cumsum_w(s_j)+1)/(sum w+1) > 1-alpha,
c'est-a-dire un intervalle [y_hat - q_t, y_hat + q_t] de demi-largeur sigma = q_t,
marge m = q_t - |y_t - y_hat(x_t)|. L'equivalence est verifiee numeriquement au
chargement (accord alarme-par-q vs alarme-par-p).

CE QUI DIFFERE DE L'ARTICLE (a ecrire dans le papier)
  * horizon 1 uniquement (forecast_steps=[1]). Avec un seul horizon, l'agregation
    de Cauchy est l'identite (T = tan((1/2 - p) pi) -> 1 - F_Cauchy(T) = p), donc
    c'est exactement W1ACAS restreint a h=1. Les horizons d>1 s'appuient sur des
    contextes anterieurs a x_t : ils ne dependent pas de la perturbation de x_t.
  * un seul canal (la cible) est donne a TTM, univarie ; les autres canaux de MSL
    sont ignores par le forecaster.
  * alpha = 0.1 (celui de nos intervalles), pas 0.01 comme leur defaut.
  * le contexte de TTM = notre fenetre (50) ; si le modele choisi a un contexte
    plus long, la fenetre est completee a gauche par des zeros MASQUES.

REJEU CAUSAL (voir TSFMReplayAdapter)
------------------------------------
Le contrefactuel modifie la fenetre (l'historique), pas le seuil : on revient au
debut de la fenetre, on restaure l'etat de W1ACAS (scores passes, poids, AdamW) a cet
instant, puis on REJOUE les pas de la fenetre avec les valeurs modifiees (previsions
TTM, scores, mises a jour des poids) jusqu'a t. La largeur q_t n'est donc PAS constante
en x : elle depend de la fenetre via les scores rejoues et les poids appris. (Une
version anterieure gelait q_t sur le passe reel : ce n'etait pas le contrefactuel.)
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn

from device import DEVICE
from backbones import Backbone

TTM_CKPT = os.environ.get("CFX_TTM_CKPT", "ibm-granite/granite-timeseries-ttm-r2")
TTM_PRED_LEN = 15          # valeur par defaut de leur pipeline ; seul l'horizon 1 est utilise
_CHUNK = 2048
SNAP_EVERY = 25            # un instantane de l'etat W1ACAS tous les 25 pas (pour le rejeu)


# ----------------------------------------------------------------------
# Forecaster : Granite TTM, differentiable en x
# ----------------------------------------------------------------------
class TTMForecaster(nn.Module):
    def __init__(self, context_length, prediction_length=TTM_PRED_LEN, ckpt=TTM_CKPT):
        super().__init__()
        # get_model choisit la bonne branche (contexte/horizon) comme dans leur pipeline.
        from tsfm_public.toolkit import get_model
        # Plus court contexte TTM-r2 = 52. Notre fenetre (50) est plus courte : get_model
        # exige force_return="zeropad" (completion a gauche faite dans forward, avec masque)
        # et, pour ne pas prendre le modele au contexte le PLUS LONG (son defaut), on
        # demande le plus court (prefer_longer_context=False). Si la fenetre est >= 52, on
        # garde le comportement par defaut de leur pipeline (plus long contexte <= fenetre).
        short = context_length < 52
        model = get_model(ckpt, context_length=context_length, prediction_length=prediction_length,
                          force_return="zeropad" if short else None,
                          prefer_longer_context=not short)
        if model is None:
            raise RuntimeError(f"get_model n'a trouve aucun TTM pour context={context_length}, "
                               f"pred={prediction_length}")
        self.context = int(model.config.context_length)
        print(f"  [tsfm] TTM {ckpt}  context={self.context}  "
              f"pred_len={model.config.prediction_length}  (fenetre fournie={context_length})")
        self.model = model.eval().to(DEVICE)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, w):                        # w : (B, L) -> (B,) prevision a l'horizon 1
        B, L = w.shape
        mask = None
        if L < self.context:
            pad = torch.zeros(B, self.context - L, device=w.device, dtype=w.dtype)
            mask = torch.cat([torch.zeros_like(pad), torch.ones_like(w)], dim=1).unsqueeze(-1)
            w = torch.cat([pad, w], dim=1)
        elif L > self.context:
            w = w[:, -self.context:]
        pv = w.unsqueeze(-1)
        kw = {}
        if mask is not None:
            kw["past_observed_mask"] = mask
        # Les TTM-r2.1 a contexte court (52-16-ft) sont "frequency prefix tuned" : ils exigent
        # un freq_token. Frequence inconnue (indices entiers, comme leur notebook) -> "oov" = 0
        # (DEFAULT_FREQUENCY_MAPPING dans tsfm_public.toolkit.time_series_preprocessor).
        if getattr(self.model.config, "resolution_prefix_tuning", False):
            kw["freq_token"] = torch.zeros(B, dtype=torch.long, device=w.device)
        try:
            out = self.model(past_values=pv, **kw)
        except TypeError:                        # version sans past_observed_mask
            kw.pop("past_observed_mask", None)
            out = self.model(past_values=pv, **kw)
        return out.prediction_outputs[:, 0, 0]


# ----------------------------------------------------------------------
# W1ACAS en ligne (classe IBM) + extraction du seuil q_t
# ----------------------------------------------------------------------
def w1acas_threshold(obs, w, alpha):
    """Plus petit score observe s tel que beta(s) = (sum_{s_j<=s} w_j + 1)/(sum w + 1) > 1-alpha,
    i.e. p(s) = 1 - beta(s) < alpha. Alarme <=> s_t >= seuil."""
    order = np.argsort(obs, kind="stable")
    s, ww = obs[order], w[order]
    beta = (np.cumsum(ww) + 1.0) / (ww.sum() + 1.0)
    k = int(np.searchsorted(beta, 1.0 - alpha, side="right"))
    return float(s[min(k, len(s) - 1)])


def p_value_from_state(score, obs, w):
    return 1.0 - (float((w * (obs <= score)).sum()) + 1.0) / (float(w.sum()) + 1.0)


def _observed_state(awcs, n_batch):
    """Scores passes + poids EXACTEMENT tels que predict() les utilisera pour le
    prochain score (cf. conformal.py, AdaptiveWeightedConformalScoreWrapper.predict :
    largeur de fenetre ca = len(cal)+1-nb_eff, poids = weights_parameters[-ca:])."""
    cal = awcs.cal_scores.detach().cpu().numpy().astype(float)
    n_min_update = int(awcs.weights_critical_norm + n_batch)
    len_c = len(cal) + 1
    nb_eff = n_batch if len_c >= n_min_update else 1
    ca = len_c - nb_eff
    w = awcs.weights_parameters.detach().cpu().numpy().astype(float)[-ca:]
    return cal[-ca:], w


def run_w1acas_online(scores, alpha=0.1, n_batch_update=10, lr=1e-3, prior="proximity",
                      snap_every=SNAP_EVERY):
    """scores : (N,) erreurs absolues h=1 en ordre chronologique (calibration puis test).
    Retourne q (N,) [NaN pour les ceil(1/alpha) premiers], p (N,), l'ecart max entre
    la p-value de predict() et celle recalculee depuis l'etat lu (controle de coherence),
    et snaps : {t: etat de l'objet IBM AVANT de traiter le score t}, un tous les
    snap_every pas -- point de depart du REJEU (voir TSFMReplayAdapter)."""
    from tsfm_public.toolkit.conformal import AdaptiveWeightedConformalScoreWrapper
    scores = np.asarray(scores, dtype=float)
    N = len(scores)
    n0 = int(np.ceil(1.0 / alpha))
    window_size = int(min(max(n0 + 1, N), 5000))
    awcs = AdaptiveWeightedConformalScoreWrapper(
        false_alarm=alpha, window_size=window_size, weighting="uniform",
        weighting_params={"n_batch_update": n_batch_update, "conformal_weights_update": False,
                          "stride": 1, "lr": lr, "n_epochs": 1,
                          "prior_past_weights_value": prior})
    awcs.fit(scores[:n0], beta_prior=None)
    q = np.full(N, np.nan)
    p = np.full(N, np.nan)
    gap = 0.0
    snaps = {}
    for t in range(n0, N):
        if snap_every and (t - n0) % snap_every == 0:
            snaps[t] = copy.deepcopy(awcs)
        obs, w = _observed_state(awcs, n_batch_update)
        q[t] = w1acas_threshold(obs, w, alpha)
        p[t] = float(awcs.predict(scores[t:t + 1], beta_prior=None)[0])
        gap = max(gap, abs(p[t] - p_value_from_state(scores[t], obs, w)))
    return q, p, gap, snaps


# ----------------------------------------------------------------------
# Backbone
# ----------------------------------------------------------------------
class TSFMWeightedBackbone(Backbone):
    """Forecaster TTM + seuils q_t de W1ACAS pour chaque fenetre de test.
    Suppose que X_calib puis X_test sont des fenetres CONSECUTIVES (pas 1), dans
    l'ordre du temps (vrai pour tous les loaders : make_windows / make_windows_mv)."""

    def __init__(self, data_bundle, context_window, alpha=0.1, n_batch_update=10,
                 lr=1e-3, prior="proximity", forecaster=None):
        self.alpha = alpha
        self.L = context_window
        self.F = data_bundle.X_train.shape[1] // context_window
        self.forecaster = forecaster if forecaster is not None else TTMForecaster(context_window)

        Xc = torch.tensor(data_bundle.X_calib, dtype=torch.float32, device=DEVICE)
        yc = torch.tensor(data_bundle.y_calib, dtype=torch.float32, device=DEVICE)
        Xt = torch.tensor(data_bundle.X_test, dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(data_bundle.y_test, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            s_cal = (yc - self._forecast(Xc)).abs().cpu().numpy()
            s_test = (yt - self._forecast(Xt)).abs().cpu().numpy()
        n_cal = len(s_cal)
        assert n_cal >= int(np.ceil(1.0 / alpha)), "calibration trop courte pour W1ACAS"

        print(f"  [tsfm] W1ACAS en ligne sur {n_cal} (calib) + {len(s_test)} (test) scores ...")
        s_all = np.concatenate([s_cal, s_test])
        q, p, gap, snaps = run_w1acas_online(s_all, alpha=alpha, n_batch_update=n_batch_update,
                                              lr=lr, prior=prior)
        # --- pour le REJEU causal (TSFMReplayAdapter) ---
        self.n_cal, self.n_batch, self.n0 = n_cal, n_batch_update, int(np.ceil(1.0 / alpha))
        self.s_all, self.snaps = s_all, snaps
        self.n_test = len(s_test)
        W = np.asarray(data_bundle.X_test).reshape(self.n_test, self.L, self.F)[:, :, 0]
        self.series_ok = bool(np.allclose(W[1:, :-1], W[:-1, 1:], atol=1e-6)
                              and np.allclose(W[1:, -1], np.asarray(data_bundle.y_test)[:-1], atol=1e-6))
        self.series_t = torch.tensor(np.concatenate([W[0], np.asarray(data_bundle.y_test)]),
                                     dtype=torch.float32, device=DEVICE)   # canal cible, temps de test
        self.X_test_np = np.asarray(data_bundle.X_test)
        if not self.series_ok:
            print("  [tsfm] ATTENTION : les fenetres de test ne sont pas consecutives (pas 1) -- "
                  "le rejeu causal n'est pas fiable.")
        self.q_test = q[n_cal:]
        self.p_test = p[n_cal:]
        agree = float(np.mean((s_test >= self.q_test) == (self.p_test < alpha)))
        print(f"  [tsfm] controle : |p(predict) - p(etat)| max = {gap:.2e} ; "
              f"accord 'alarme par q_t' vs 'alarme par p<alpha' = {agree:.4f} ; "
              f"taux d'alarme test = {float(np.mean(self.p_test < alpha)):.3f} (alpha={alpha}) ; "
              f"q_t median = {np.median(self.q_test):.4f}")
        if gap > 1e-3 or agree < 0.999:
            print("  [tsfm] ATTENTION : l'equivalence q_t <-> p-value n'est pas verifiee, "
                  "ne pas utiliser ces resultats avant d'avoir regarde _observed_state.")

    def _forecast(self, x):
        x = x.to(DEVICE)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        w = x.reshape(x.shape[0], self.L, self.F)[:, :, 0]       # canal cible
        return torch.cat([self.forecaster(w[i:i + _CHUNK])
                          for i in range(0, w.shape[0], _CHUNK)], dim=0)

    def case_adapter(self, i=None):
        """i : indice de test -> adaptateur avec REJEU causal (la fenetre perturbee
        recalcule scores passes, poids et seuil q_t) ; None : tout le test, seuils
        FACTUELS (vecteur), uniquement pour la selection batchee des cas."""
        if i is None:
            return TSFMCaseAdapter(self, self.q_test)
        return TSFMReplayAdapter(self, int(i))

    def predict_interval(self, x):
        raise RuntimeError("q_t depend de l'instant : utiliser backbone.case_adapter(i)")


class TSFMCaseAdapter(Backbone):
    """Meme role que SPCIBackboneAdapter : fige le seuil q_t d'un cas (ou le vecteur
    de tous les seuils de test)."""

    def __init__(self, parent, q):
        self.parent = parent
        self.q = q

    def predict_interval(self, x):
        hat_y = self.parent._forecast(x)
        q = torch.as_tensor(self.q, dtype=hat_y.dtype, device=DEVICE)
        sigma = q.expand_as(hat_y) if q.dim() == 0 else q
        return hat_y.squeeze(), sigma.squeeze()


class TSFMReplayAdapter(Backbone):
    """W1ACAS avec REJEU CAUSAL. La variable contrefactuelle est la fenetre x' (L pas,
    canal cible pour TTM). Revenir en arriere au debut de la fenetre, c'est :
      1) restaurer l'etat de W1ACAS (scores passes, poids, AdamW) tel qu'il etait
         juste avant le premier pas de la fenetre ;
      2) rejouer les L pas de la fenetre avec les valeurs modifiees : pour chaque pas,
         prevision TTM sur la fenetre qui le precede (valeurs reelles puis modifiees),
         score |valeur - prevision|, puis mise a jour en ligne des poids (la vraie
         classe IBM) ;
      3) lire le seuil q_t de l'etat obtenu ; centre = prevision TTM sur x'.

    Deux chemins, selectionnes par x.requires_grad :
      * x SANS gradient (evaluations : marge factuelle, verifications des tours
        externes, resultat final) -> rejeu EXACT, y compris l'apprentissage des poids.
        C'est lui qui decide de la faisabilite et des chiffres rapportes ;
        il memorise (scores passes, poids) de ce rejeu.
      * x AVEC gradient (descente) -> meme calcul des scores et du seuil, mais avec les
        poids GELES a ceux du dernier rejeu exact (l'apprentissage des poids est
        constant par morceaux : gradient nul presque partout). Le gradient passe par
        la prevision et par le score selectionne pour q_t. Les poids sont rafraichis
        a chaque verification exacte (fin de chaque tour externe). Gradient de q_t :
        estimateur straight-through (valeur exacte, derivee du quantile lisse).
    Il faut i >= L (historique de test reel avant la fenetre)."""

    def __init__(self, parent, i):
        L = parent.L
        if not parent.series_ok:
            raise RuntimeError("fenetres de test non consecutives : rejeu impossible")
        if i < L:
            raise ValueError(f"cas {i} < L={L} : pas assez d'historique de test pour rejouer")
        self.p, self.i, self.L = parent, i, L
        self.hist = parent.series_t[i - L:i]                       # temps i-L .. i-1 (reels)
        t0 = parent.n_cal + i - L                                   # premier pas a rejouer (flux S)
        key = parent.n0 + ((t0 - parent.n0) // SNAP_EVERY) * SNAP_EVERY
        st = copy.deepcopy(parent.snaps[key])
        with torch.enable_grad():       # la mise a jour des poids d'IBM fait un backward()
            for m in range(key, t0):                                # pas reels jusqu'a t0
                st.predict(parent.s_all[m:m + 1], beta_prior=None)
        self.base = st
        self._frozen = None
        self._last = None
        x0 = torch.tensor(parent.X_test_np[i], dtype=torch.float32, device=DEVICE)
        hat0, q0 = self._exact(x0)
        self.gap0 = abs(float(q0) - float(parent.q_test[i]))
        if self.gap0 > 1e-4:
            print(f"  [tsfm] ATTENTION cas {i} : seuil rejoue {float(q0):.6f} != seuil du flux "
                  f"{float(parent.q_test[i]):.6f} (ecart {self.gap0:.2e}) ; rejeu a verifier")

    def _scores_and_center(self, x):
        xs = x.reshape(-1).reshape(self.L, self.p.F)[:, 0]          # canal cible, (L,)
        full = torch.cat([self.hist, xs])                           # (2L,)
        ctx = full.unfold(0, self.L, 1)                             # (L+1, L)
        fc = self.p.forecaster(ctx)                                 # (L+1,)
        s_mod = (xs - fc[:self.L]).abs()                            # scores des L pas de la fenetre
        return s_mod, fc[self.L]                                    # fc[L] = prevision de y_t

    def _exact(self, x):
        x = x.detach().to(DEVICE).reshape(-1)
        if self._last is not None and torch.equal(self._last[0], x):
            return self._last[1], self._last[2]
        with torch.no_grad():
            s_mod, hat_y = self._scores_and_center(x)
        st = copy.deepcopy(self.base)
        # Ce rejeu est souvent appele sous torch.no_grad() (verifications de la recherche) :
        # la mise a jour des poids de la classe IBM fait un backward(), il faut le reactiver.
        with torch.enable_grad():
            for sv in s_mod.double().cpu().numpy():
                st.predict(np.array([sv]), beta_prior=None)
        obs, w = _observed_state(st, self.p.n_batch)
        q = w1acas_threshold(obs, w, self.p.alpha)
        self._frozen = (torch.tensor(obs[:-self.L], dtype=torch.float32, device=DEVICE),
                        torch.tensor(w, dtype=torch.float32, device=DEVICE))
        q_t = torch.tensor(q, dtype=torch.float32, device=DEVICE)
        self._last = (x.clone(), hat_y, q_t)
        return hat_y, q_t

    def predict_interval(self, x):
        x = x.to(DEVICE)
        if x.numel() != self.L * self.p.F:
            raise ValueError("TSFMReplayAdapter : une seule fenetre a la fois")
        if not x.requires_grad:
            hat_y, q = self._exact(x)
            return hat_y.squeeze(), q.squeeze()
        if self._frozen is None:
            self._exact(x)
        head, w = self._frozen
        s_mod, hat_y = self._scores_and_center(x)
        obs = torch.cat([head, s_mod])
        order = torch.argsort(obs.detach())
        beta = (torch.cumsum(w[order], 0) + 1.0) / (w.sum() + 1.0)
        k = min(int((beta <= 1.0 - self.p.alpha).sum().item()), obs.shape[0] - 1)
        q0 = obs[order[k]].detach()
        # Estimateur "straight-through" : VALEUR = seuil exact q0 (poids gels), DERIVEE =
        # celle du quantile lisse  sum_k w_k sigmoid((q-s_k)/tau) = 1-alpha, soit
        #   dq/ds_j = w_j sigma'_j / sum_k w_k sigma'_k   (j dans la fenetre).
        # Sans cela, le gradient est nul tant que le quantile ne tombe pas sur un score de
        # la fenetre (la p-value d'IBM utilise un comparateur dur). tau = 10 % de q0.
        tau = 0.1 * q0.clamp_min(1e-6)
        sg = torch.sigmoid((q0 - obs.detach()) / tau)
        dsg = w * sg * (1.0 - sg) / tau
        c = dsg[-self.L:] / dsg.sum().clamp_min(1e-12)
        q = q0 + (c.detach() * (s_mod - s_mod.detach())).sum()
        return hat_y.squeeze(), q
