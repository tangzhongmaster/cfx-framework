"""Tests du REJEU causal (SPCI et W1ACAS/TSFM). A lancer AVANT toute relance.

    python test_replay.py spci nab
    python test_replay.py spci msl
    python test_replay.py tsfm nab
    python test_replay.py tsfm msl

Ce que chaque test doit afficher (OK = ecart < seuil) :
  T1  fenetre factuelle : le rejeu redonne EXACTEMENT l'intervalle factuel
      (SPCI : meme (hat_y, sigma) que le calcul direct ; TSFM : meme q_t que le flux
      W1ACAS d'origine).
  T2  gradient : autograd == difference finie sur la prevision (et sur la largeur
      quand elle depend de la fenetre).
  T3  perturber la fenetre change la LARGEUR (sinon le rejeu ne fait rien).
"""
import sys
import numpy as np
import torch
import protocol as P
from config import ExperimentConfig, Dataset, Backbone, Cell, Decoder
from device import DEVICE

which = sys.argv[1] if len(sys.argv) > 1 else "spci"
ds = Dataset.MSL if (len(sys.argv) > 2 and sys.argv[2] == "msl") else Dataset.NAB
bb = Backbone.SPCI if which == "spci" else Backbone.TSFM
extra = dict(msl_channel="T-1") if ds == Dataset.MSL else {}
cfg = ExperimentConfig(dataset=ds, backbone=bb, cell=Cell.BOTH_FREE, decoder=Decoder.FLOW,
                       n_cases=10, **extra)
setup = P.build_setup_generic(cfg, verbose=False)
db = setup["data_bundle"]
L = cfg.context_window
cases = [60, 200, 777, 1500]
cases = [c for c in cases if c < len(db.X_test)]
torch.manual_seed(0)


def check(name, err, tol):
    print(f"  {name}: {err:.2e}  {'OK' if err < tol else 'ECART TROP GRAND'}")


print(f"=== {which} / {ds.value} (L={L}) ===")
for i in cases:
    ad = P.active_backbone_for_case(setup, cfg, i)
    x = torch.tensor(db.X_test[i], dtype=torch.float32, device=DEVICE)
    y = float(db.y_test[i])
    hy, sg = ad.predict_interval(x)
    print(f"\n-- cas {i} : y_hat={hy.item():.4f} sigma={sg.item():.4f}")

    # T1 : equivalence avec le calcul factuel direct
    if which == "spci":
        from backbones import spci_factual_state
        hy_all, sg_all = spci_factual_state(setup["spci_raw"], db.X_test, db.y_test)
        check("T1 |y_hat rejoue - direct|", abs(hy.item() - hy_all[i].item()), 1e-4)
        check("T1 |sigma rejoue - direct|", abs(sg.item() - sg_all[i].item()), 1e-4)
    else:
        check("T1 |q rejoue - q du flux|", abs(sg.item() - float(setup["backbone"].q_test[i])), 1e-4)

    # T2 : gradient (autograd vs difference finie) sur la prevision
    xg = x.clone().requires_grad_(True)
    hyg, sgg = ad.predict_interval(xg)
    hyg.backward()
    g = xg.grad.clone()
    j = int(torch.argmax(g.abs()))
    eps = 1e-3
    xp, xm = x.clone(), x.clone()
    xp[j] += eps
    xm[j] -= eps
    fd = (ad.predict_interval(xp)[0] - ad.predict_interval(xm)[0]).item() / (2 * eps)
    check("T2 gradient y_hat (auto vs diff. finie)", abs(g[j].item() - fd) / (abs(fd) + 1e-8), 1e-2)

    # T3 : la largeur depend de la fenetre
    noise = 0.5 * torch.randn_like(x) * x.std()
    sg2 = ad.predict_interval(x + noise)[1].item()
    print(f"  T3 sigma factuel={sg.item():.4f}  sigma apres perturbation={sg2:.4f}  "
          f"{'LARGEUR VARIE' if abs(sg2 - sg.item()) > 1e-6 else 'largeur constante (!)'}")

    # T2b : gradient de la largeur (chemin differentiable), sur la perturbation
    xg2 = (x + noise).clone().requires_grad_(True)
    ad.predict_interval((x + noise))            # rafraichit les poids gels (rejeu exact)
    _, sgg2 = ad.predict_interval(xg2)
    if sgg2.requires_grad:
        sgg2.backward()
        g2 = xg2.grad
        print(f"  T2b |grad sigma|_max = {g2.abs().max().item():.3e} "
              f"({'dependance en x' if g2.abs().max().item() > 0 else 'seuil pris hors fenetre : gradient nul'})")
        # compare le chemin differentiable au rejeu exact sur la meme fenetre
        check("T2b |sigma differentiable - exact|", abs(sgg2.item() - ad.predict_interval((x + noise))[1].item()), 1e-4)
print("\nFin des tests.")
