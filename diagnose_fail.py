"""Diagnostic des cas MSL en echec (exp9). Rejoue la recherche latente a
masque COMPLET (meme boucle que search._masked_augmented_lagrangian) et
affiche, a chaque tour externe : marge m, cible, |dyhat|, violation de
centre g2, largeur, lambda, rho, distance.

Usage (dossier cfx_framework, meme environnement que protocol.py) :
    python diagnose_fail.py 2406 6150 outer=12 inner=200
"""
import sys
import torch
import protocol as P
from config import ExperimentConfig, Dataset, Backbone, Cell, Decoder
from margin import compute_margin, fixed_channel_value, channel_violation, \
    scenario_targets, target_violation
from device import DEVICE

args = sys.argv[1:]
N_OUTER = int(next((a.split("=")[1] for a in args if a.startswith("outer=")), 6))
N_INNER = int(next((a.split("=")[1] for a in args if a.startswith("inner=")), 100))
cases = [int(a) for a in args if a.isdigit()] or [2406, 6150]
config = ExperimentConfig(dataset=Dataset.MSL, backbone=Backbone.RAW_CONTEXT,
                          cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                          msl_channel="T-1", n_cases=40)
setup = P.build_setup_generic(config, verbose=False)
decoder = setup["decoder"]
db = setup["data_bundle"]

for i in cases:
    bb = P.active_backbone_for_case(setup, config, i)
    x_f = torch.tensor(db.X_test[i], dtype=torch.float32, device=DEVICE)
    y_t = float(db.y_test[i])
    yt = torch.tensor(y_t, dtype=torch.float32, device=DEVICE)
    m0, yh0, s0 = compute_margin(bb, x_f, yt)
    m_target, direction = scenario_targets(m0.item(), config.scenario)
    x0v = fixed_channel_value(bb, x_f, config.cell)
    x0v = x0v.detach() if x0v is not None else None
    print(f"\n=== idx {i} : m0={m0.item():.4f} cible={m_target} dir={direction} "
          f"yhat={yh0.item():.4f} w={2*s0.item():.4f} y={y_t:.4f} eps={config.epsilon} ===")

    z0 = decoder.encode(x_f, y_t).detach()
    z = z0.clone().requires_grad_(True)
    lam1, lam2, rho = 0.0, 0.0, 10.0
    for outer in range(N_OUTER):
        opt = torch.optim.Adam([z], lr=0.02)
        for _ in range(N_INNER):
            opt.zero_grad()
            x = decoder.decode(z, y_t)
            m, yh, s = compute_margin(bb, x, yt)
            g1 = target_violation(m, m_target, direction)
            g2 = channel_violation(bb, x, x0v, config.cell, config.epsilon)
            pen = (torch.clamp(lam1 + rho * g1, min=0) ** 2 - lam1 ** 2) / (2 * rho) \
                + (torch.clamp(lam2 + rho * g2, min=0) ** 2 - lam2 ** 2) / (2 * rho)
            (torch.norm(z - z0) + pen).backward()
            opt.step()
        with torch.no_grad():
            x = decoder.decode(z, y_t)
            m, yh, s = compute_margin(bb, x, yt)
            g1 = target_violation(m, m_target, direction).item()
            g2 = channel_violation(bb, x, x0v, config.cell, config.epsilon).item()
        print(f"  tour {outer}: m={m.item():+.4f} g1={g1:+.4f} g2={g2:+.4f} "
              f"|dyhat|={abs(yh.item()-yh0.item()):.4f} w={2*s.item():.4f} "
              f"lam1={lam1:.3f} lam2={lam2:.3f} rho={rho:.0f} "
              f"|z-z0|={torch.norm(z-z0).item():.3f} |x-xf|={torch.norm(x-x_f).item():.3f}")
        lam1 = max(0, lam1 + rho * g1)
        lam2 = max(0, lam2 + rho * g2)
        rho = min(rho * 3.0, 1e5)
    print(f"  => faisable (g<1e-4) : {g1 < 1e-4 and g2 < 1e-4}")
