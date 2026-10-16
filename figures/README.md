# Figures of the paper

Every figure is produced by a script in this folder from the result files
`../results_exp*.csv` (already included, one per experiment). Run each script
from this folder, e.g. `python make_fig_epsilon.py`. Only `numpy`, `pandas` and
`matplotlib` are needed, except for the counterfactual example (last row).
The output file names are given below. They do not follow the figure numbers of
the paper, so use this table to match them.

| Paper figure | Output file | Script | Data |
|---|---|---|---|
| Center shift vs. margin (RawContext and SPCI, MSL and NAB) | `fig2_center_shift.pdf` | `make_fig_center_shift.py` | `../results_exp6.csv` (Exp. 6) |
| Temporal window scan (sparsity, MSL and NAB) | `fig4_temporal_window.pdf` | `make_fig_temporal_window.py` | `../results_exp10.csv`, `../results_exp11.csv` (Exp. 10, 11) |
| Sensitivity to the center-preserving tolerance epsilon | `fig3_epsilon.pdf` | `make_fig_epsilon.py` | `../results_exp7.csv` (Exp. 7) |
| Synthetic experiments (AR and regime change) | `fig5_synthetic.pdf` | `make_fig_synthetic.py` | `../results_exp13.csv`, `../results_exp14.csv` (Exp. 13, 14) |
| Counterfactual example on NAB (case 622) | `fig6_counterfactual_nab.pdf` | `make_cf_figures.py` | `cf_case_622.npz` (stored arrays) |

## Counterfactual example

`cf_case_622.npz` stores, for each of the three cases displayed, the factual
window, the counterfactual window and the associated values.

- `python make_cf_figures.py 622 --replot` redraws the figure from the stored
  arrays. This is instant and needs no model.
- `python make_cf_figures.py 622` recomputes the counterfactuals first. This
  needs the framework dependencies (`../requirements.txt`), the NAB data
  (`NAB_PATH`, see `../README.md`) and takes several minutes.

## Other result files

All tables of the paper are computed from `../results_exp*.csv`. The mapping
between experiments and tables is given in `../README.md`.
