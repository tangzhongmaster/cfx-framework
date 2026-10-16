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

PAPER_EXPERIMENTS = {

    # ========================================================
    # LA METHODE -- un seul run, un seul resultat, pas noye
    # ========================================================
    "proposed_method": (
        "LA methode proposee : niveaux 1-2-3-4-5 fixes, sans parcimonie (niveau 7 = NONE)",
        [PROPOSED_METHOD]
    ),

    "proposed_method_sparsity_a": (
        "Methode proposee + parcimonie (a) : cardinalite input, retrieval reel (Maalej)",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT,
                           sparsity=SparsityMethod.MAALEJ_INPUT_CARDINALITY,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=15)]
    ),

    "proposed_method_sparsity_b": (
        "Methode proposee + parcimonie (b) : cardinalite en espace latent",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT,
                           sparsity=SparsityMethod.LATENT_CARDINALITY,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=15)]
    ),

    "proposed_method_sparsity_c": (
        "Methode proposee + parcimonie (c) : penalite L1 directement dans la recherche latente",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT,
                           sparsity=SparsityMethod.LATENT_L1_PENALTY,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=15)]
    ),

    "proposed_method_sparsity_d": (
        "Methode proposee + parcimonie (d) : fenetre contiguee, p(x|y,hors-fenetre)",
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
    # NIVEAU 3 -- decodeur : benchmark explicite des alternatives
    # ========================================================
    "level3_decoder_benchmark": (
        "Niveau 3 : flow (retenu) vs VAE vs flow autoregressif vs diffusion, memes 15 cas",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=dec,
                           distance_space=DistanceSpace.LATENT, sparsity=SparsityMethod.NONE,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=15)
         for dec in [Decoder.FLOW, Decoder.VAE, Decoder.FLOW_AR]]
        # DIFFUSION exclu ici : n'utilise pas la recherche latente standard,
        # cf. la piste "sparsity_b" qui l'exploite via l'inpainting natif
    ),

    # ========================================================
    # NIVEAU 4 -- espace de recherche : latent vs vraisemblance
    # ========================================================
    "level4_distance_benchmark": (
        "Niveau 4 : latent (retenu) vs vraisemblance, meme decodeur (flow), 15 cas",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=ds, sparsity=SparsityMethod.NONE,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=15)
         for ds in [DistanceSpace.LATENT, DistanceSpace.LIKELIHOOD]]
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
    # Niveau 3 x niveau 4 : le decodeur croise avec le choix de distance
    # (le benchmark decodeur ci-dessus ne testait que latent ; ici on
    # verifie si le classement VAE/flow/flow_ar tient aussi sous vraisemblance)
    # ========================================================
    "level3_decoder_benchmark_likelihood": (
        "Niveau 3 x 4 : flow vs VAE vs flow autoregressif, sous vraisemblance plutot que latent",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=dec,
                           distance_space=DistanceSpace.LIKELIHOOD, sparsity=SparsityMethod.NONE,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=15)
         for dec in [Decoder.FLOW, Decoder.VAE, Decoder.FLOW_AR]]
    ),

    # ========================================================
    # Niveau 2 x niveau 4 : le choix de distance croise avec le backbone
    # (jamais verifie sur SPCI avant, seulement sur contexte brut)
    # ========================================================
    "level2_distance_benchmark_spci": (
        "Niveau 2 x 4 : latent vs vraisemblance, backbone SPCI au lieu de contexte brut",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.SPCI,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=ds, sparsity=SparsityMethod.NONE,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=15)
         for ds in [DistanceSpace.LATENT, DistanceSpace.LIKELIHOOD]]
    ),

    # ========================================================
    # Parcimonie (niveau 7) x domaine : les 4 options, jamais testees sur NAB
    # ========================================================
    "sparsity_all_nab": (
        "Les 4 options de parcimonie (a,b,c,d), sur NAB",
        [ExperimentConfig(dataset=Dataset.NAB, backbone=Backbone.RAW_CONTEXT,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT, sparsity=sp,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=15)
         for sp in [SparsityMethod.MAALEJ_INPUT_CARDINALITY, SparsityMethod.LATENT_CARDINALITY,
                    SparsityMethod.LATENT_L1_PENALTY, SparsityMethod.WINDOW]]
    ),

    # ========================================================
    # Parcimonie (niveau 7) x backbone : les 4 options, jamais testees sur SPCI
    # ========================================================
    "sparsity_all_spci": (
        "Les 4 options de parcimonie (a,b,c,d), backbone SPCI",
        [ExperimentConfig(dataset=Dataset.MITBIH, backbone=Backbone.SPCI,
                           cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                           distance_space=DistanceSpace.LATENT, sparsity=sp,
                           scenario=Scenario.ANOMALY_TO_NORMAL, n_cases=15)
         for sp in [SparsityMethod.MAALEJ_INPUT_CARDINALITY, SparsityMethod.LATENT_CARDINALITY,
                    SparsityMethod.LATENT_L1_PENALTY, SparsityMethod.WINDOW]]
    ),

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
