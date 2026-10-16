"""
Verification explicite de la Definition 1 (contrefactuel valide) sur un
ResultRow -- pas une confiance implicite dans ce que search.py a retourne.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CriteriaReport:
    validity: bool
    channel_preservation: bool
    plausibility: bool          # toujours True si le decodeur a ete utilise (par construction)
    minimality_gap: float        # distance obtenue - distance minimale connue (0 si non comparee)
    sparsity: int
    all_valid: bool


def check_definition_1(result, backbone, decoder, x_factual, y_t, config, m0):
    from margin import compute_margin, fixed_channel_value, scenario_targets
    from device import DEVICE
    import torch

    m_target, direction = scenario_targets(m0, config.scenario)
    x_factual = x_factual.to(DEVICE)
    x_star_t = torch.tensor(result.x_star, dtype=torch.float32, device=DEVICE)
    m, hat_y, sigma = compute_margin(backbone, x_star_t, torch.tensor(y_t, dtype=torch.float32, device=DEVICE))

    validity = (m.item() >= m_target - 1e-4) if direction == "geq" else (m.item() <= m_target + 1e-4)

    x0_value = fixed_channel_value(backbone, x_factual, config.cell)
    if config.cell.value == "both_free":
        channel_ok = True
    else:
        h_star = fixed_channel_value(backbone, x_star_t, config.cell)
        channel_ok = abs(h_star.item() - x0_value.item()) <= config.epsilon + 1e-4

    plausible = not isinstance(decoder, type(None))  # True des lors qu'un decodeur borne l'espace

    sparsity = int(np.sum(np.abs(result.x_star - x_factual.cpu().numpy()) > 1e-6))

    all_valid = validity and channel_ok and plausible
    return CriteriaReport(validity, channel_ok, plausible, 0.0, sparsity, all_valid)
