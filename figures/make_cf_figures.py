"""Figure qualitative : un cas NAB, factuel vs contrefactuel dans 3 reglages.

A lancer dans le dossier cfx_framework :
    python make_cf_figures.py 622          # calcule (lent) + sauvegarde cf_case_622.npz + figure
    python make_cf_figures.py 622 --replot # redessine depuis le npz, sans recalcul

Ligne du haut : historique (axe y symlog pour voir a la fois les petites et les grandes
modifications), gris = factuel, bleu = contrefactuel, zone grisee = fenetre modifiee.
Ligne du bas : zoom sur l'intervalle a t : barre noire = factuel, bleue = contrefactuel,
point = centre y_hat, etoile = y_t (jamais modifiee).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

case = int(sys.argv[1]) if len(sys.argv) > 1 else 622
replot = "--replot" in sys.argv
npz = f"cf_case_{case}.npz"
titles = ["(a) center and width free", "(b) center preserved", "(c) center preserved + window"]

if not (replot and os.path.exists(npz)):
    import torch
    import protocol as P
    from config import ExperimentConfig, Dataset, Backbone, Cell, Decoder, SparsityMethod
    from margin import compute_margin
    from search import run_search_diverse
    from device import DEVICE
    settings = [(Cell.BOTH_FREE, SparsityMethod.NONE),
                (Cell.CENTER_FIXED_WIDTH_FREE, SparsityMethod.NONE),
                (Cell.CENTER_FIXED_WIDTH_FREE, SparsityMethod.WINDOW)]
    out = {}
    for k, (cell, spars) in enumerate(settings):
        config = ExperimentConfig(dataset=Dataset.NAB, backbone=Backbone.RAW_CONTEXT, cell=cell,
                                  decoder=Decoder.FLOW, sparsity=spars, n_cases=40)
        setup = P.build_setup_generic(config, verbose=False)
        db = setup["data_bundle"]
        bb = P.active_backbone_for_case(setup, config, case)
        x_f = torch.tensor(db.X_test[case], dtype=torch.float32, device=DEVICE)
        y_t = float(db.y_test[case])
        yt = torch.tensor(y_t, device=DEVICE)
        m0, yh0, s0 = compute_margin(bb, x_f, yt)
        res = run_search_diverse(bb, setup["decoder"], x_f, y_t, config, m0.item(),
                                 X_calib=db.X_calib)[0]
        x_s = torch.tensor(res.x_star, dtype=torch.float32, device=DEVICE)
        _, yh1, s1 = compute_margin(bb, x_s, yt)
        out[f"xf{k}"] = x_f.detach().cpu().numpy().ravel()
        out[f"xs{k}"] = x_s.detach().cpu().numpy().ravel()
        out[f"v{k}"] = np.array([y_t, yh0.item(), s0.item(), yh1.item(), s1.item(), float(res.feasible)])
    np.savez(npz, **out)
d = np.load(npz)

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), gridspec_kw={"height_ratios": [1.5, 1]})
for k in range(3):
    xf, xs, v = d[f"xf{k}"], d[f"xs{k}"], d[f"v{k}"]
    y_t, yh0, s0, yh1, s1, feas = v
    L = len(xf)
    lag = np.arange(L, 0, -1)                       # L (oldest) ... 1 (just before t)
    ax = axes[0, k]
    changed = np.where(np.abs(xs - xf) > 1e-6)[0]
    if len(changed):
        ax.axvspan(lag[changed].max() + .5, lag[changed].min() - .5, color="#0072B2", alpha=.10, lw=0)
    ax.plot(lag, xf, color="#777777", lw=1.0, label="factual")
    ax.plot(lag, xs, color="#0072B2", lw=1.2, label="counterfactual")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.invert_xaxis()
    ax.set_title(titles[k] + ("" if feas else " (infeasible)"), fontsize=8)
    ax.set_xlabel("time steps before $t$", fontsize=8)
    ax.tick_params(labelsize=7)
    if k == 0:
        ax.set_ylabel("normalized value (symlog)", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    bx = axes[1, k]
    bx.errorbar([0], [yh0], yerr=[s0], color="black", capsize=4, fmt="o", ms=4, lw=1.5, label="factual")
    bx.errorbar([1], [yh1], yerr=[s1], color="#0072B2", capsize=4, fmt="o", ms=4, lw=1.5, label="counterfactual")
    bx.axhline(y_t, color="#D55E00", lw=1.0, ls="--")
    bx.plot([0.5], [y_t], marker="*", color="#D55E00", ms=10, ls="none", label="$y_t$")
    lo = min(yh0 - s0, yh1 - s1, y_t); hi = max(yh0 + s0, yh1 + s1, y_t); pad = .15 * (hi - lo + 1e-9)
    bx.set_ylim(lo - pad, hi + pad); bx.set_xlim(-.6, 1.6)
    bx.set_xticks([0, 1]); bx.set_xticklabels(["factual", "counterf."], fontsize=7)
    bx.tick_params(labelsize=7)
    if k == 0:
        bx.set_ylabel("interval at $t$", fontsize=8)
        bx.legend(fontsize=7, frameon=False, loc="best")
plt.tight_layout()
plt.savefig(f"fig6_counterfactual_nab_{case}.pdf")
plt.savefig(f"fig6_counterfactual_nab_{case}.png", dpi=150)
print("ecrit : fig6_counterfactual_nab_%d.pdf" % case)
