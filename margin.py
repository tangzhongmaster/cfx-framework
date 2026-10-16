"""
La marge signee m(x'), la logique de la grille (quel canal est fixe), et
les 4 scenarios -- Section 4.1 et Definition 1 du papier, traduits en code.
Ce module est le coeur commun a TOUTE experience, quels que soient le
backbone, le decodeur ou la methode de recherche choisis.
"""

import torch
from config import Cell, Scenario
from device import DEVICE


def compute_margin(backbone, x, y_t):
    """m(x') = w(x')/2 - |y_t - hat_y(x')|, avec w(x') = 2*sigma(x').
    Retourne (m, hat_y, sigma) -- tous des tenseurs scalaires."""
    hat_y, sigma = backbone.predict_interval(x)
    width = 2 * sigma
    m = width / 2 - torch.abs(y_t - hat_y)
    return m, hat_y, sigma


def fixed_channel_value(backbone, x, cell: Cell):
    """h(x) pour la case donnee -- Proposition 1 s'applique quand h = centre.
    Ne detache PAS le resultat : appele aussi sur le candidat en cours
    d'optimisation (channel_violation), ou le gradient doit continuer a
    circuler. Pour une reference fixe (x0_value, calculee sur x_factual),
    detacher explicitement au point d'appel -- voir search.py."""
    hat_y, sigma = backbone.predict_interval(x)
    if cell == Cell.CENTER_FIXED_WIDTH_FREE:
        return hat_y
    elif cell == Cell.CENTER_FREE_WIDTH_FIXED:
        return 2 * sigma  # la largeur
    elif cell == Cell.BOTH_FREE:
        return None  # aucune contrainte de canal
    raise ValueError(cell)


def channel_violation(backbone, x, x0_value, cell: Cell, epsilon: float):
    """|h(x) - h0| - epsilon, clip a 0 (comme g_2 dans le lagrangien augmente).
    Retourne 0.0 si cell == BOTH_FREE (pas de contrainte)."""
    if cell == Cell.BOTH_FREE:
        # sur DEVICE : combinee ensuite (search.py) avec g1/base, qui sont
        # sur DEVICE -- un tensor CPU ici casserait l'addition si DEVICE=cuda.
        return torch.tensor(0.0, device=DEVICE)
    h = fixed_channel_value(backbone, x, cell)
    return torch.clamp(torch.abs(h - x0_value) - epsilon, min=0.0)


def scenario_targets(m0: float, scenario: Scenario):
    """Retourne (m_target, direction) pour le scenario demande, etant donne
    la marge factuelle m0 = m(X_t). direction='geq' ou 'leq' -> lequel des
    deux sens de l'inegalite g_1 utiliser (eq:bidirectional-constraint)."""
    if scenario == Scenario.ANOMALY_TO_NORMAL:
        assert m0 < 0, "anomaly_to_normal suppose m(X_t) < 0"
        return abs(m0) * 0.1 + 0.02, "geq"
    elif scenario == Scenario.NORMAL_TO_ANOMALY:
        assert m0 > 0, "normal_to_anomaly suppose m(X_t) > 0"
        return -(abs(m0) * 0.1 + 0.02), "leq"
    elif scenario == Scenario.NARROW_STAY_NORMAL:
        assert m0 > 0, "narrow_stay_normal suppose m(X_t) > 0 (deja normal)"
        # 'leq' (pas 'geq') : on veut forcer m a VRAIMENT descendre vers la
        # cible, pas juste rester au-dessus d'un seuil plus bas que m0 lui-
        # meme -- avec 'geq' le point factuel satisfaisait deja trivialement
        # la contrainte (m0 > m0*0.3), d'ou une distance nulle systematique.
        return m0 * 0.3, "leq"   # force a se rapprocher de zero, reste positif
    elif scenario == Scenario.WIDEN_STAY_ANOMALY:
        assert m0 < 0, "widen_stay_anomaly suppose m(X_t) < 0 (deja anormal)"
        return m0 * 1.5, "leq"   # encore plus negatif
    raise ValueError(scenario)


def target_violation(m, m_target, direction):
    """g_1(z), clip a 0 -- Section 4.3, eq:bidirectional-constraint."""
    if direction == "geq":
        return torch.clamp(m_target - m, min=0.0)
    elif direction == "leq":
        return torch.clamp(m - m_target, min=0.0)
    raise ValueError(direction)


def check_proposition_1(backbone, x, y_t, delta=0.01):
    """Verification numerique de la Proposition 1 : deplacer hat_y de delta
    (a largeur fixe) doit augmenter m de exactement delta. Utilise pour les
    tests de non-regression, pas dans le chemin critique des experiences."""
    m0, hat_y0, sigma0 = compute_margin(backbone, x, y_t)
    # perturbation synthetique : on ne peut pas bouger hat_y directement sans
    # changer x, donc ce check se fait au niveau de la fonction m elle-meme
    m_perturbed = sigma0 - torch.abs(y_t - (hat_y0 + delta))
    predicted = m0 + delta if (y_t - hat_y0) > 0 else m0 - delta
    return torch.allclose(m_perturbed, predicted, atol=1e-5)
