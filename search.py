"""
Recherche, reorganisee autour des 7 niveaux de config.py. Une seule methode
principale (niveaux 3-4-5 : decodeur flow + espace latent + lagrangien
augmente), et la parcimonie (niveau 7) est un choix EXPLICITE separe, pas
une 6e famille de recherche parmi d'autres.

run_search() dispatche uniquement sur config.sparsity :
  NONE   -> search_proposed (la methode, sans parcimonie)
  WINDOW -> search_window (fenetre contigue, seule option de parcimonie)

search_latent_cardinality reste definie plus bas : ce n'est plus une
option de config.sparsity (retiree, voir config.py), mais elle est
reutilisee comme PRIMITIVE interne par la cascade de recherche
(search_variants.py, Exp. 9) -- gardee pour cette seule raison.
"""

from dataclasses import dataclass
import time
import numpy as np
import torch
import pygad

from margin import compute_margin, fixed_channel_value, channel_violation, \
    target_violation, scenario_targets
from config import SparsityMethod
from device import DEVICE


@dataclass
class ResultRow:
    x_star: np.ndarray
    feasible: bool
    distance: float
    n_dims_changed: int
    m_obtained: float
    log_likelihood: float = None
    n_evals: int = 0
    runtime: float = 0.0   # secondes, mesure au point d'entree unique
                             # (run_search) -- pas besoin de l'ajouter dans
                             # chaque search_* individuellement


def _decode(decoder, z, y_t):
    return decoder.decode(z, y_t)


def _evaluate(backbone, decoder, z, y_t, x0_value, cell, epsilon, m_target, direction):
    x = _decode(decoder, z, y_t)
    m, hat_y, sigma = compute_margin(backbone, x, torch.tensor(y_t, dtype=torch.float32, device=DEVICE))
    g1 = target_violation(m, m_target, direction)
    g2 = channel_violation(backbone, x, x0_value, cell, epsilon)
    return x, m, g1, g2


def _augmented_lagrangian_core(backbone, decoder, x_factual, y_t, config, m0, z_init,
                                n_outer=8, n_inner=100, rho0=10.0, gamma=3.0):
    """Le coeur du lagrangien augmente (niveau 5), factorise pour etre
    reutilise a la fois par la recherche principale et par la
    re-optimisation apres chaque seuillage (option c de la parcimonie)."""
    x_factual = x_factual.to(DEVICE)
    m_target, direction = scenario_targets(m0, config.scenario)
    x0_value = fixed_channel_value(backbone, x_factual, config.cell)
    if x0_value is not None:
        x0_value = x0_value.detach()
    z = z_init.to(DEVICE).clone().requires_grad_(True)
    z0 = decoder.encode(x_factual, y_t).detach()

    lam1, lam2, rho = 0.0, 0.0, rho0
    for outer in range(n_outer):
        opt = torch.optim.Adam([z], lr=0.02)
        for _ in range(n_inner):
            opt.zero_grad()
            x, m, g1, g2 = _evaluate(backbone, decoder, z, y_t, x0_value,
                                      config.cell, config.epsilon, m_target, direction)
            # distance_space est verrouillee a LATENT (niveau 4, voir
            # config.py) : la branche "likelihood" (vraisemblance exacte
            # comme objectif de recherche) a ete retiree avec cette option.
            base = torch.norm(z - z0)
            penalty = (torch.clamp(lam1 + rho * g1, min=0) ** 2 - lam1 ** 2) / (2 * rho) \
                    + (torch.clamp(lam2 + rho * g2, min=0) ** 2 - lam2 ** 2) / (2 * rho)
            (base + penalty).backward()
            opt.step()
        with torch.no_grad():
            x, m, g1, g2 = _evaluate(backbone, decoder, z, y_t, x0_value,
                                      config.cell, config.epsilon, m_target, direction)
            g1v, g2v = g1.item(), g2.item()
        lam1 = max(0, lam1 + rho * g1v)
        lam2 = max(0, lam2 + rho * g2v)
        rho *= gamma

    with torch.no_grad():
        x, m, g1, g2 = _evaluate(backbone, decoder, z, y_t, x0_value,
                                  config.cell, config.epsilon, m_target, direction)
        feasible = g1.item() < 1e-4 and g2.item() < 1e-4
    return z.detach(), x, m.item(), feasible


def search_proposed(backbone, decoder, x_factual, y_t, config, m0,
                     n_outer=8, n_inner=100, rho0=10.0, gamma=3.0):
    """LA methode proposee : decodeur (niveau 3, flow), espace latent
    (niveau 4), lagrangien augmente (niveau 5). Sans parcimonie
    (config.sparsity == NONE) -- la parcimonie (niveau 7, fenetre contigue
    uniquement) passe par search_window, pas par cette fonction."""
    x_factual = x_factual.to(DEVICE)
    z0 = decoder.encode(x_factual, y_t).detach()
    z_star, x, m_val, feasible = _augmented_lagrangian_core(
        backbone, decoder, x_factual, y_t, config, m0, z0, n_outer, n_inner, rho0, gamma)

    log_lik = decoder.log_prob(x, y_t) if hasattr(decoder, "log_prob") else None
    log_lik = float(log_lik.item()) if log_lik is not None else None
    return ResultRow(x.cpu().numpy(), feasible, float(torch.norm(x - x_factual).item()),
                      int((torch.abs(x - x_factual) > 1e-6).sum().item()), m_val,
                      log_likelihood=log_lik)


def _masked_augmented_lagrangian(backbone, decoder, x_factual, y_t, config, m0,
                                  mask, n_outer=6, n_inner=60, rho0=10.0, gamma=3.0,
                                  channel_l1_lambda=0.0, n_features=1):
    """Lagrangien augmente ou seules les dimensions de x sous 'mask' sont
    autorisees a differer de x_factual -- le masque est applique A
    L'INTERIEUR de la boucle d'optimisation (x_masked = x_factual +
    mask*(D(z;y_t)-x_factual)), donc z est cherche pour satisfaire les
    contraintes SOUS ce masque precis, pas juste tronque apres coup.

    channel_l1_lambda/n_features (optionnel, n_features>1 seulement pour
    les datasets multivaries type MSL -- voir search_window) : ajoute une
    penalite de GROUPE L1 (norme L2,1, groupee par canal a travers les pas
    de temps libres, a la Sulem et al. 2022 eq. Sparse DPE) qui pousse vers
    peu de canaux distincts touches, EN PLUS du masque dur sur l'axe temps
    deja applique via 'mask'. Les deux axes (temps=masque dur, canal=
    penalite souple) sont donc cherches ENSEMBLE dans la meme descente, pas
    en deux etapes separees."""
    x_factual = x_factual.to(DEVICE)
    m_target, direction = scenario_targets(m0, config.scenario)
    x0_value = fixed_channel_value(backbone, x_factual, config.cell)
    if x0_value is not None:
        x0_value = x0_value.detach()
    z0 = decoder.encode(x_factual, y_t).detach()
    z = z0.clone().requires_grad_(True)
    mask_f = mask.to(DEVICE).float()
    dim = x_factual.shape[0]
    n_timesteps = dim // n_features if n_features > 1 else None

    def _channel_penalty(x_masked):
        if channel_l1_lambda <= 0 or n_features <= 1:
            return 0.0
        diff = (x_masked - x_factual).reshape(n_timesteps, n_features)
        return channel_l1_lambda * torch.sqrt((diff ** 2).sum(dim=0) + 1e-8).sum()

    lam1, lam2, rho = 0.0, 0.0, rho0
    for outer in range(n_outer):
        opt = torch.optim.Adam([z], lr=0.02)
        for _ in range(n_inner):
            opt.zero_grad()
            x_raw = decoder.decode(z, y_t)
            x_masked = x_factual + mask_f * (x_raw - x_factual)
            m, hat_y, sigma = compute_margin(backbone, x_masked, torch.tensor(y_t, dtype=torch.float32, device=DEVICE))
            g1 = target_violation(m, m_target, direction)
            g2 = channel_violation(backbone, x_masked, x0_value, config.cell, config.epsilon)
            base = torch.norm(z - z0) + _channel_penalty(x_masked)
            penalty = (torch.clamp(lam1 + rho * g1, min=0) ** 2 - lam1 ** 2) / (2 * rho) \
                    + (torch.clamp(lam2 + rho * g2, min=0) ** 2 - lam2 ** 2) / (2 * rho)
            loss = base + penalty
            loss.backward()
            opt.step()
        with torch.no_grad():
            x_raw = decoder.decode(z, y_t)
            x_masked = x_factual + mask_f * (x_raw - x_factual)
            m, hat_y, sigma = compute_margin(backbone, x_masked, torch.tensor(y_t, dtype=torch.float32, device=DEVICE))
            g1v = target_violation(m, m_target, direction).item()
            g2v = channel_violation(backbone, x_masked, x0_value, config.cell, config.epsilon).item()
        lam1 = max(0, lam1 + rho * g1v)
        lam2 = max(0, lam2 + rho * g2v)
        rho = min(rho * gamma, 1e5)

    with torch.no_grad():
        x_raw = decoder.decode(z, y_t)
        x_masked = x_factual + mask_f * (x_raw - x_factual)
        m, hat_y, sigma = compute_margin(backbone, x_masked, torch.tensor(y_t, dtype=torch.float32, device=DEVICE))
        g1 = target_violation(m, m_target, direction).item()
        g2 = channel_violation(backbone, x_masked, x0_value, config.cell, config.epsilon).item()
        feasible = g1 < 1e-4 and g2 < 1e-4

    return x_masked, feasible, m.item(), max(g1, 0.0) + max(g2, 0.0)


def search_latent_cardinality(backbone, decoder, x_factual, y_t, config, m0,
                                max_k=None, n_outer_trial=4, n_inner_trial=40):
    """N'est PLUS une option de config.sparsity (retiree, voir config.py
    niveau 7) -- gardee uniquement comme PRIMITIVE interne, reutilisee par
    la cascade de recherche (search_variants.py:run_cascade, Exp. 9, dernier
    etage). Croissance gloutonne du masque ; a CHAQUE essai on reoptimise z
    sous ce masque precis (_masked_augmented_lagrangian), pas juste un
    decoupage d'un x deja trouve sans contrainte -- corrige un bug ou la
    version precedente retournait quasi-toujours le point factuel inchange,
    les contraintes (surtout le canal, tolerance serree) n'etant jamais
    satisfaites par un masque applique apres coup sur une solution pleine
    dimension.

    BUG CORRIGE (voir discussion du 2026-10-04) : la version precedente ne
    faisait AVANCER free_dims (append) QUE quand un d etait feasible a ce
    tour -- si aucun d seul (k=1) n'etait feasible, k=2,3,...,max_k
    retestaient tous les MEMES masques a 1 dimension (trial_free = [] + [d]
    puisque free_dims restait vide), sans jamais former de masque a 2+
    dimensions. Sur un cas ou aucune dimension unique ne suffit, ca
    gaspillait max_k * dim appels a _masked_augmented_lagrangian (soit
    O(dim^2), potentiellement des heures sur MSL) pour finir par retourner
    infeasible de toute facon -- exactement le symptome d'un cas bloque
    bien plus longtemps que les autres dans la cascade (Exp. 9). Le fix
    fait maintenant progresser free_dims a CHAQUE tour (en gardant le
    meilleur d du tour, feasible en priorite, sinon celui qui minimise la
    violation residuelle g1+g2), donc le masque grandit reellement d'un
    tour a l'autre et peut devenir feasible a un k bien plus petit que
    dim -- change aussi le cout pire cas en pratique (la plupart des cas
    deviennent feasibles a k petit) meme si la complexite O(dim^2) du pire
    cas theorique (jamais feasible avant k=dim) reste inchangee."""
    x_factual = x_factual.to(DEVICE)
    dim = x_factual.shape[0]
    max_k = max_k or dim
    free_dims = []

    for k in range(1, max_k + 1):
        best_d, best_x, best_m, best_dist = None, None, None, float("inf")
        best_feasible = False
        best_violation = float("inf")
        for d in range(dim):
            if d in free_dims:
                continue
            trial_free = free_dims + [d]
            mask = torch.zeros(dim, dtype=torch.bool, device=DEVICE)
            mask[trial_free] = True
            x_masked, feasible, m_val, violation = _masked_augmented_lagrangian(
                backbone, decoder, x_factual, y_t, config, m0, mask,
                n_outer=n_outer_trial, n_inner=n_inner_trial)
            if feasible:
                dist = torch.norm(x_masked - x_factual).item()
                # parmi les d feasibles, garde le plus proche (distance minimale)
                if not best_feasible or dist < best_dist:
                    best_dist, best_d, best_x, best_m = dist, d, x_masked, m_val
                    best_feasible, best_violation = True, 0.0
            elif not best_feasible and violation < best_violation:
                # aucun d feasible pour l'instant : garde celui qui reduit le
                # plus la violation residuelle, pour que le masque progresse
                # quand meme vers une region feasible au lieu de stagner
                best_violation = violation
                best_d, best_x, best_m = d, x_masked, m_val
                best_dist = torch.norm(x_masked - x_factual).item()
        if best_d is None:
            # ne devrait arriver que si dim==len(free_dims) deja (plus aucun d
            # a tester) ; par securite on arrete plutot que de boucler a vide
            break
        free_dims.append(best_d)
        if best_feasible:
            return ResultRow(best_x.cpu().numpy(), True, best_dist, len(free_dims), best_m)

    return ResultRow(x_factual.cpu().numpy(), False, 0.0, 0, m0)


def search_window(backbone, decoder, x_factual, y_t, config, m0,
                   max_width=None, n_outer=12, n_inner=200):
    """Parcimonie, seule option retenue (niveau 7) : fenetre SUFFIXE
    (i, t] ancree a DROITE, pas une fenetre a position libre -- decision
    explicite (pas deduite seule, voir l'echange qui a precede) : x_factual
    est [y_{t-w}, ..., y_{t-1}] (ou w blocs de n_features pour MSL, voir
    data.py:load_msl -- make_windows_mv aplatit (w, n_features) en
    row-major, donc chaque bloc contigu de n_features dimensions = UN pas
    de temps, et le DERNIER bloc = y_{t-1}, le plus recent). Liberer les
    width DERNIERS pas de temps donne un sens causal direct a la largeur
    trouvee : ell* = t - i*, "jusqu'ou il a fallu remonter dans
    l'historique pour rendre le contrefactuel faisable" -- une fenetre a
    position arbitraire n'aurait pas cette lecture.

    Axe canal (n_features>1, MSL uniquement) : PAS un second masque dur --
    une penalite de groupe L1 sur les canaux (config.channel_l1_lambda),
    cherchee EN MEME TEMPS que le masque temporel dans
    _masked_augmented_lagrangian (voir sa docstring) -- a la Sulem et al.
    2022, masque factorise M = t x w, mais w est ici une penalite souple
    plutot qu'un sous-ensemble dur pre-decide.

    CAUSALITE. Cette fonction traite predict_interval(x') comme une evaluation
    DIRECTE et SUFFISANTE du nouveau contexte. C'est exact pour RawContext/SplitCP
    (fonctions pures de x). Pour SPCI et W1ACAS/TSFM, c'est le backbone lui-meme qui
    REJOUE la chaine (previsions -> residus/scores -> etat -> intervalle) a chaque
    appel a partir de l'historique reel qui precede la fenetre : voir
    backbones.SPCIReplayAdapter et tsfm_backbone.TSFMReplayAdapter. Le masque porte
    sur x', le rejeu est dans le backbone. ADAPTIVE (ACI) n'a pas d'adaptateur de
    rejeu branche ici (voir backbones.replay_forward)."""
    x_factual = x_factual.to(DEVICE)
    dim = x_factual.shape[0]
    n_features = max(1, dim // config.context_window)
    n_timesteps = dim // n_features
    max_width = max_width or n_timesteps
    channel_l1_lambda = getattr(config, "channel_l1_lambda", 0.0) if n_features > 1 else 0.0

    def _try(width):
        mask = torch.zeros(dim, dtype=torch.bool, device=DEVICE)
        mask[(n_timesteps - width) * n_features:] = True   # suffixe (i,t], ancre a droite
        x_masked, feasible, m_val, _v = _masked_augmented_lagrangian(
            backbone, decoder, x_factual, y_t, config, m0, mask,
            n_outer=n_outer, n_inner=n_inner,
            channel_l1_lambda=channel_l1_lambda, n_features=n_features)
        return x_masked, feasible, m_val

    # Budget large (12x200, rho jusqu'a 1e5) : diagnose_fail.py a montre que 6x100 s'arrete
    # avant que rho/lambda soient assez grands pour fermer les contraintes (marge ET plafond
    # de centre) sur les cas proches du seuil.
    # Recherche : (1) on teste d'abord la fenetre ENTIERE (si elle echoue, le cas est declare
    # infaisable sans tester les autres) ; (2) on balaie ensuite les longueurs 1, 2, 3, ... et
    # on s'arrete a la PREMIERE faisable, c'est-a-dire en remontant pas a pas depuis t. Pas de
    # dichotomie : l'optimisation etant non convexe, un echec a une longueur intermediaire ne
    # prouve pas que les plus courtes echouent, et une dichotomie peut alors sur-estimer ell*.
    # ell* reste un majorant du vrai minimum (budget fini), pas une garantie.
    x_best, ok, m_best = _try(max_width)
    if not ok:
        return ResultRow(x_factual.cpu().numpy(), False, 0.0, 0, m0)
    best_w = max_width
    for width in range(1, max_width):
        x_w, ok, m_w = _try(width)
        if ok:
            x_best, m_best, best_w = x_w, m_w, width
            break
    dist = torch.norm(x_best - x_factual).item()
    return ResultRow(x_best.cpu().numpy(), True, dist, best_w, m_best)


def run_search_diverse(backbone, decoder, x_factual, y_t, config, m0, X_calib=None,
                        max_attempts_multiplier=4):
    """Retourne une LISTE de ResultRow distincts (pas un seul), a la
    Sulem et al. (ensemble de contrefactuels divers) : relance run_search
    plusieurs fois depuis des redemarrages differents, ne garde que les
    resultats faisables et suffisamment differents les uns des autres
    (distance > config.min_diversity_dist), jusqu'a en avoir config.k_diverse
    ou avoir epuise le budget d'essais. Si config.k_diverse == 1, comportement
    d'origine : une liste a un seul element."""
    if config.k_diverse <= 1:
        return [run_search(backbone, decoder, x_factual, y_t, config, m0, X_calib)]

    accepted = []
    max_attempts = config.k_diverse * max_attempts_multiplier
    for attempt in range(max_attempts):
        if len(accepted) >= config.k_diverse:
            break
        # perturbe legerement z0 a chaque essai pour explorer differentes
        # solutions, sans changer le mecanisme de recherche lui-meme
        result = run_search(backbone, decoder, x_factual, y_t, config, m0, X_calib)
        if not result.feasible:
            continue
        x_candidate = torch.tensor(result.x_star, dtype=torch.float32)
        is_diverse = all(
            torch.norm(x_candidate - torch.tensor(prev.x_star, dtype=torch.float32)).item()
            > config.min_diversity_dist
            for prev in accepted
        )
        if is_diverse:
            accepted.append(result)

    if not accepted:
        # aucun succes divers trouve : retourne quand meme le dernier essai
        # (feasible ou non) pour ne pas cacher un echec total
        accepted = [result]
    return accepted


def run_search(backbone, decoder, x_factual, y_t, config, m0, X_calib=None):
    """Point d'entree unique -- dispatch sur config.sparsity (niveau 7),
    pas sur une liste de methodes concurrentes. Chronometre ici, une seule
    fois, plutot que dans chaque search_* : runtime mesure exactement ce
    que le reste du pipeline appelle, pas une notion de 'temps interne'
    definie differemment par methode."""
    t0 = time.perf_counter()
    if config.sparsity == SparsityMethod.NONE:
        result = search_proposed(backbone, decoder, x_factual, y_t, config, m0)
    elif config.sparsity == SparsityMethod.WINDOW:
        result = search_window(backbone, decoder, x_factual, y_t, config, m0)
    else:
        raise ValueError(config.sparsity)
    result.runtime = time.perf_counter() - t0
    return result
