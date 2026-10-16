"""
Test rapide a lancer en premier des que torch est disponible (pas
executable dans le sandbox qui a produit ces changements). Verifie que le
controle theorique Split CP tient, mais PAS via feasibility_rate==0 (faux
en toute rigueur, voir plus bas) -- via Delta_w == 0 pour TOUS les cas
faisables, ce qui est la vraie prediction theorique.

Pourquoi pas feasibility_rate==0 : la contrainte "centre fixe" (Cell.
CENTER_FIXED_WIDTH_FREE) est une TOLERANCE |hat_y(x')-hat_y(X_t)| <= epsilon
(defaut 0.05, voir margin.py:channel_violation), pas une egalite exacte.
Avec w strictement constant (Split CP), un cas dont |m0| est deja plus
petit que ce qu'un deplacement de centre <= epsilon peut compenser sera
trouve "faisable" -- le centre a fait tout le travail, DANS la marge de
tolerance epsilon, la largeur n'a jamais bouge. Ce n'est pas un echec du
controle theorique, c'est une consequence attendue de epsilon > 0.
Verifie ici par Delta_w, pas par un comptage de succes/echec.

Usage : python smoke_test.py
"""

import numpy as np
from config import ExperimentConfig, Dataset, Backbone, Cell, Decoder
from run_experiment import run_experiment

print("=== 1/3 : SplitCPBackbone sur synthetic_ar (n_cases=8, epsilon=0.05 defaut) ===")
cfg = ExperimentConfig(dataset=Dataset.SYNTHETIC_AR, backbone=Backbone.SPLIT_CP,
                        cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                        n_cases=8)
df, raw_cases = run_experiment(cfg, make_figures=False, return_cases=True)
print(df[["case_idx", "m0", "feasible", "distance"]])

feasible_cases = [c for c in raw_cases if c["feasible"]]
print(f"\n{len(feasible_cases)}/{len(raw_cases)} cas faisables.")
if feasible_cases:
    delta_w = np.array([c["delta_w"] for c in feasible_cases])
    delta_y = np.array([c["delta_y_hat"] for c in feasible_cases])
    print(f"Delta_w  (doit etre ~0, largeur constante) : {delta_w}")
    print(f"Delta_y_hat (doit etre <= epsilon=0.05)    : {delta_y}")
    assert np.allclose(delta_w, 0.0, atol=1e-4), \
        f"ECHEC THEORIQUE : Delta_w non nul alors que sigma est constant ! {delta_w}"
    assert (np.abs(delta_y) <= cfg.epsilon + 1e-4).all(), \
        f"ECHEC : le centre a bouge plus que epsilon ! {delta_y}"
    print("OK : tous les succes (s'il y en a) viennent du centre, DANS la tolerance "
          "epsilon -- Delta_w est exactement nul partout, comme predit par la theorie.")
else:
    print("OK (trivialement) : aucun cas faisable, rien a verifier sur Delta_w.")

print("\n=== 2/3 : meme chose, epsilon quasi nul (0.0001) -- doit donner feasibility~0 ===")
cfg_tight = ExperimentConfig(dataset=Dataset.SYNTHETIC_AR, backbone=Backbone.SPLIT_CP,
                              cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                              n_cases=8, epsilon=1e-4)
df_tight = run_experiment(cfg_tight, make_figures=False)
print(df_tight[["case_idx", "m0", "feasible"]])
print("-> attendu cette fois : feasibility_rate ~ 0 (centre quasi totalement fige, "
      "largeur constante -- plus rien pour faire bouger la marge)\n")

print("=== 3/3 : grille de decomposition complete (3 backbones x 3 cells, n_cases=5) ===")
from experiment_channel_decomposition import run_decomposition_grid, check_split_cp_width_invariant
per_case, summary = run_decomposition_grid(Dataset.SYNTHETIC_AR, n_cases=5)
print(summary.to_string(index=False))
check_split_cp_width_invariant(per_case)  # leve une AssertionError explicite si ca casse
print("\n-> attendu : ligne (split_cp, center_fixed_width_free) a median_delta_w ~ 0 "
      "(et median_width_ratio ~ 0), feasibility_rate pouvant etre > 0 -- c'est normal, "
      "voir l'explication en tete de fichier. Les autres backbones devraient avoir "
      "un median_width_ratio notablement > 0.")

print("\nTout s'est execute sans lever d'exception.")
