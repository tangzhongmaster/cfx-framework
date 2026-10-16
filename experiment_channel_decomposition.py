"""
RQ2 (geometrie conforme) + RQ3 (centre vs largeur) en une seule grille :
pour chaque backbone (Split CP / CQR-RawContext / SPCI) et chaque Cell
(BOTH_FREE = Configuration A, CENTER_FIXED_WIDTH_FREE = Configuration B,
CENTER_FREE_WIDTH_FIXED = Configuration C), on lance run_experiment.

Deux niveaux de sortie, distincts expres :
  - un DataFrame PAR CAS (m0, m*, Delta_y_hat, Delta_w, largeur factuelle/
    contrefactuelle, violation de la contrainte du canal fixe, feasible,
    distance, runtime) -- les resultats bruts, rien d'agrege, pour pouvoir
    les inspecter avant toute conclusion ;
  - un DataFrame RESUME (medianes/taux par cellule backbone x cell), calcule
    a partir du premier, jamais l'inverse.

Rappel (pour ne pas re-sur-interpreter ces chiffres) : pour Split CP,
Delta_w == 0 est une PROPRIETE STRUCTURELLE du backbone (w constant par
construction), verifiee ici, pas decouverte. Ce qui est a determiner
empiriquement est : (a) est-ce que des cas sont trouves faisables dans la
cellule center-fixed malgre ca (via le canal centre, DANS la tolerance
epsilon -- colonne center_constraint_violated doit rester False pour ces
cas), et (b) comment Delta_w se comporte pour les deux autres backbones,
dont la largeur n'est PAS structurellement constante.

Ce script ne cree aucune nouvelle mecanique de recherche -- il reutilise
run_experiment() tel quel.

Usage :
    python experiment_channel_decomposition.py --dataset mitbih --n_cases 15
    python experiment_channel_decomposition.py --dataset synthetic_regime \
        --n_cases 15 --synthetic_phi_1 0.7 --synthetic_phi_2 0.3
"""

import argparse
import numpy as np
import pandas as pd

from config import ExperimentConfig, Dataset, Backbone, Cell, Decoder, DistanceSpace
from run_experiment import run_experiment, build_setup


def _width_contribution_ratio(delta_y_hat, delta_w, eps=1e-8):
    num = np.abs(delta_w)
    den = np.abs(delta_y_hat) + np.abs(delta_w) + eps
    return num / den


def run_decomposition_grid(dataset: Dataset, n_cases: int = 15, decoder=Decoder.FLOW,
                            distance_space=DistanceSpace.LATENT, seed: int = 42,
                            extra_config_kwargs: dict = None, verbose=True):
    """Boucle (backbone x cell). setup (donnees+backbone+decodeur) est
    reconstruit une fois PAR BACKBONE (pas par cell), et reutilise pour les
    3 cells de ce backbone -- comparaison sur le meme backbone entraine,
    coherent avec la mise en garde de build_setup dans run_experiment.py.

    Retourne (per_case_df, summary_df) -- le detail d'abord, le resume
    ensuite, jamais l'inverse."""
    extra_config_kwargs = extra_config_kwargs or {}
    backbones = [Backbone.SPLIT_CP, Backbone.RAW_CONTEXT, Backbone.SPCI]
    cells = [Cell.BOTH_FREE, Cell.CENTER_FIXED_WIDTH_FREE, Cell.CENTER_FREE_WIDTH_FIXED]

    per_case_rows = []
    for bb in backbones:
        setup = None
        for cell in cells:
            config = ExperimentConfig(
                dataset=dataset, backbone=bb, cell=cell, decoder=decoder,
                distance_space=distance_space, n_cases=n_cases, seed=seed,
                **extra_config_kwargs,
            )
            if verbose:
                print(f"\n=== backbone={bb.value} cell={cell.value} ===")
            if setup is None:
                setup = build_setup(config, verbose=verbose)
            df, raw_cases = run_experiment(config, verbose=verbose, return_cases=True,
                                            setup=setup, make_figures=False)

            raw_by_idx = {c["case_idx"]: c for c in raw_cases}
            for _, row in df.iterrows():
                case_idx = int(row["case_idx"])
                raw = raw_by_idx.get(case_idx, {})
                width_factual = 2 * raw.get("sigma_factual", np.nan) if raw else np.nan
                hat_ys = raw.get("hat_y_stars", [np.nan])
                sigmas = raw.get("sigma_stars", [np.nan])
                width_star = 2 * sigmas[0] if sigmas else np.nan

                per_case_rows.append({
                    "backbone": bb.value, "cell": cell.value, "case_idx": case_idx,
                    "m0": row.get("m0", np.nan),
                    "m_star": row.get("m_obtained", np.nan),
                    "delta_y_hat": raw.get("delta_y_hat", np.nan),
                    "delta_w": raw.get("delta_w", np.nan),
                    "width_factual": width_factual,
                    "width_star": width_star,
                    # violation de la contrainte du canal designe fixe par
                    # cette cell -- PAS juste "channel_preservation" tel
                    # quel (qui est deja un bool valide/invalide, cf.
                    # criteria.py) ; on le renomme ici explicitement comme
                    # une violation pour que son sens soit sans ambiguite
                    # dans ce tableau.
                    "center_constraint_violated": not bool(row.get("channel_preservation", True)),
                    "feasible": bool(row.get("feasible", False)),
                    "distance": row.get("distance", np.nan),
                    "n_dims_changed": row.get("n_dims_changed", np.nan),
                    "runtime": row.get("runtime", np.nan),
                    "error": row.get("error", None),
                })

    per_case_df = pd.DataFrame(per_case_rows)
    summary_df = _summarize(per_case_df)
    return per_case_df, summary_df


def _summarize(per_case_df: pd.DataFrame) -> pd.DataFrame:
    """Calcule le resume A PARTIR du detail -- jamais l'inverse, pour ne
    pas pouvoir accidentellement agreger des chiffres qu'on n'a pas sous
    la main par ailleurs."""
    rows = []
    for (bb, cell), g in per_case_df.groupby(["backbone", "cell"], sort=False):
        feasible = g[g["feasible"]]
        if len(feasible) > 0:
            ratio = _width_contribution_ratio(feasible["delta_y_hat"].values,
                                               feasible["delta_w"].values)
            median_dy, median_dw = feasible["delta_y_hat"].median(), feasible["delta_w"].median()
            median_ratio = float(np.median(ratio))
            median_distance = feasible["distance"].median()
        else:
            median_dy = median_dw = median_ratio = median_distance = np.nan
        rows.append({
            "backbone": bb, "cell": cell, "n_cases": len(g),
            "feasibility_rate": g["feasible"].mean(),
            "n_center_constraint_violated": int(g["center_constraint_violated"].sum()),
            "median_distance": median_distance,
            "median_delta_y_hat": median_dy, "median_delta_w": median_dw,
            "median_width_ratio": median_ratio,
            "median_runtime": g["runtime"].median(),
        })
    # ordre stable : backbone puis cell, dans l'ordre defini dans la grille
    order_bb = {"split_cp": 0, "raw_context": 1, "spci": 2}
    order_cell = {"both_free": 0, "center_fixed_width_free": 1, "center_free_width_fixed": 2}
    return pd.DataFrame(rows).sort_values(
        by=["backbone", "cell"],
        key=lambda s: s.map(order_bb) if s.name == "backbone" else s.map(order_cell)
    ).reset_index(drop=True)


def check_split_cp_width_invariant(per_case_df: pd.DataFrame, tol: float = 1e-4) -> bool:
    """Assertion simple demandee : |Delta_w| < tol pour TOUS les cas
    faisables de Split CP, quelle que soit la cell -- pas une assertion sur
    le taux de faisabilite. Leve une AssertionError explicite si ca casse,
    ne retourne pas juste False en silence."""
    mask = (per_case_df["backbone"] == "split_cp") & (per_case_df["feasible"])
    bad = per_case_df[mask & (per_case_df["delta_w"].abs() >= tol)]
    if len(bad) > 0:
        raise AssertionError(
            f"Split CP : {len(bad)} cas faisable(s) avec |Delta_w| >= {tol} -- "
            f"ca ne devrait structurellement jamais arriver (w constant par construction).\n"
            f"{bad[['cell', 'case_idx', 'm0', 'delta_w', 'delta_y_hat']].to_string(index=False)}"
        )
    return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=[d.value for d in Dataset], default="mitbih")
    p.add_argument("--n_cases", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic_phi_1", type=float, default=None)
    p.add_argument("--synthetic_phi_2", type=float, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extra = {}
    if args.synthetic_phi_1 is not None:
        extra["synthetic_phi_1"] = args.synthetic_phi_1
    if args.synthetic_phi_2 is not None:
        extra["synthetic_phi_2"] = args.synthetic_phi_2

    per_case_df, summary_df = run_decomposition_grid(Dataset(args.dataset), n_cases=args.n_cases,
                                                       seed=args.seed, extra_config_kwargs=extra)

    print("\n" + "=" * 70)
    print("RESUME (calcule a partir du detail ci-dessous, pas l'inverse) :")
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("Verification structurelle Split CP (|Delta_w| < 1e-4 sur tous les cas faisables) :")
    try:
        check_split_cp_width_invariant(per_case_df)
        print("OK.")
    except AssertionError as e:
        print(f"ECHEC :\n{e}")

    base = args.out or f"decomposition_{args.dataset}"
    per_case_df.to_csv(f"{base}_raw.csv", index=False)
    summary_df.to_csv(f"{base}_summary.csv", index=False)
    print(f"\nDetail par cas -> {base}_raw.csv")
    print(f"Resume          -> {base}_summary.csv")
