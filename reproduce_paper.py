"""
Reproduit chaque table/figure du papier, organise autour des 7 niveaux de
config.py. La toute premiere entree, "proposed_method", est LA methode --
un seul run qui donne LE resultat qu'on pointe du doigt. Tout le reste est
soit un benchmark explicite (niveau par niveau), soit le diagnostic
historique (annexe).

Usage :
    python reproduce_paper.py --table proposed_method
    python reproduce_paper.py --list
    python reproduce_paper.py --all
"""

import argparse
from config import (ExperimentConfig, Dataset, Backbone, Cell, Decoder,
                     DistanceSpace, SparsityMethod, Scenario, PROPOSED_METHOD)
from run_experiment import run_experiment

# Retire de ce fichier (voir config.py, niveaux 3/4/7 verrouilles) : les
# ablations qui comparaient flow vs VAE/flow_ar/diffusion (niveau 3),
# latent vs likelihood (niveau 4), et les 4 methodes de parcimonie
# (niveau 7) -- ces alternatives n'existent plus dans le code, il n'y a
# plus rien a comparer. Seule la fenetre contigue (WINDOW) reste comme
# option de parcimonie.

PAPER_EXPERIMENTS = {

    # ========================================================
    # LA METHODE -- un seul run, un seul resultat, pas noye
    # ========================================================
    "proposed_method": (
        "LA methode proposee : niveaux 1-2-3-4-5 fixes, sans parcimonie (niveau 7 = NONE)",
        [PROPOSED_METHOD]
    ),

    "proposed_method_sparsity": (
        "Methode proposee + parcimonie : fenetre contigue, p(x|y,hors-fenetre)",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT,
                           sparsity=SparsityMethod.WINDOW,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=15)]
    ),

    "proposed_method_diverse": (
        "Methode proposee, ensemble divers (3 contrefactuels par cas, a la Sulem et al.)",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT,
                           sparsity=SparsityMethod.NONE,
                           k_diverse=3, min_diversity_dist=0.5,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=10)]
    ),

    # ========================================================
    # NIVEAU 1 -- backbone : check de generalite
    # ========================================================
    "level2_backbone_spci": (
        "Niveau 2 : meme methode, backbone SPCI au lieu de contexte brut",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.SPCI,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT, sparsity=SparsityMethod.NONE,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=40)]
    ),

    "level2_backbone_nab": (
        "Niveau 1 : meme methode, domaine NAB au lieu de MIT-BIH",
        [ExperimentConfig(dataset=Dataset.NAB, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT, sparsity=SparsityMethod.NONE,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=40)]
    ),

    # ========================================================
    # NIVEAU 2 -- case de la grille
    # ========================================================
    "level1_grid_free_center": (
        "Niveau 1 : case Wachter classique (centre libre, largeur fixe)",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FREE_WIDTH_FIXED, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT, sparsity=SparsityMethod.NONE,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=15)]
    ),

    "level1_grid_both_free": (
        "Niveau 1 : ablation, les deux canaux libres",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.BOTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT, sparsity=SparsityMethod.NONE,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=15)]
    ),

    # ========================================================
    # Les 4 scenarios, avec LA methode fixee (pas un backbone de recherche a part)
    # ========================================================
    "four_scenarios": (
        "Les 4 scenarios de marge, avec la methode proposee (niveaux 1-5 fixes)",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT, sparsity=SparsityMethod.NONE,
                           scenario=scenario, n_cases=5)
         for scenario in Scenario]
    ),

    "four_scenarios_nab": (
        "Les 4 scenarios de marge, sur NAB (jamais teste hors MIT-BIH avant)",
        [ExperimentConfig(dataset=Dataset.NAB, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT, sparsity=SparsityMethod.NONE,
                           scenario=scenario, n_cases=5)
         for scenario in Scenario]
    ),

    # ========================================================
    # Parcimonie (niveau 7) x domaine : WINDOW (seule option), sur NAB
    # ========================================================
    "sparsity_nab": (
        "Parcimonie (fenetre contigue), sur NAB",
        [ExperimentConfig(dataset=Dataset.NAB, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT, sparsity=SparsityMethod.WINDOW,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=15)]
    ),

    # ========================================================
    # Parcimonie (niveau 7) x backbone SPCI : RETIREE. WINDOW suppose que
    # predict_interval(x') est une evaluation directe suffisante (vrai pour
    # RAW_CONTEXT/SPLIT_CP, voir search.py:search_window) -- pas pour SPCI,
    # dont la largeur depend d'un etat (fenetre de residus) construit une
    # fois pour toutes sur l'historique REEL, jamais rejoue. Liberer les
    # derniers pas de temps sans rejouer prevision->residu serait
    # causalement incoherent. A reintroduire seulement si un mecanisme de
    # rejeu (a la backbones.py:replay_forward) est ajoute pour SPCI.
    # ========================================================

    "proposed_method_nab": (
        "LA methode proposee, meme config, sur NAB (le pendant NAB de proposed_method)",
        [ExperimentConfig(dataset=Dataset.NAB, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT, sparsity=SparsityMethod.NONE,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=40)]
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=list(PAPER_EXPERIMENTS.keys()), default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list or (not args.table and not args.all):
        print("Tables disponibles (la premiere est LA methode) :\n")
        for key, (desc, _) in PAPER_EXPERIMENTS.items():
            print(f"  {key:32s} {desc}")
        return

    keys = list(PAPER_EXPERIMENTS.keys()) if args.all else [args.table]
    for key in keys:
        desc, configs = PAPER_EXPERIMENTS[key]
        print(f"\n{'=' * 60}\n{key} : {desc}\n{'=' * 60}")
        for config in configs:
            df = run_experiment(config, fig_dir=f"figures/{key}")
            df.to_csv(f"results_{key}_{config.label()}.csv", index=False)


if __name__ == "__main__":
    main()
