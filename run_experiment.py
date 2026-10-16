"""
Point d'entree unique. Choisir une experience = construire un
ExperimentConfig et appeler run_experiment(config). Chaque table/figure du
papier correspond a un config precis -- voir reproduce_paper.py pour la
liste complete.

Usage direct :
    python run_experiment.py --dataset mitbih --backbone raw_context \
        --cell center_fixed_width_free --decoder flow --search augmented_lagrangian \
        --scenario anomaly_to_normal --n_cases 15
"""

import argparse
import numpy as np
import torch
import pandas as pd

from config import (ExperimentConfig, Dataset, Backbone, Cell, Decoder,
                     DistanceSpace, ConstraintMethod, SparsityMethod, Scenario)
from data import load_dataset
from backbones import build_backbone, SPCIBackboneAdapter
from decoders import build_decoder, train_decoder
from margin import compute_margin
from search import run_search, run_search_diverse
from criteria import check_definition_1
from setup_cache import (setup_cache_key, get_memory_setup, store_memory_setup,
                          load_decoder_weights, save_decoder_weights)
from device import DEVICE


def select_cases(data_bundle, backbone, config, n_cases, verbose=True, point_forecasts=None):
    """Choisit n_cases parmi les points ou le MODELE LUI-MEME calcule un
    m0 compatible avec le scenario demande (pas le label AAMI brut -- le
    papier selectionne parmi ce que le modele detecte, le label sert
    seulement de contexte/stratification, pas de critere de selection).
    Stratifie par difficulte |m(X_t)|.

    point_forecasts : tableau optionnel, un point_forecast REEL par cas de
    test (necessaire pour SPCI, ou 'backbone' seul ne suffit pas a calculer
    m0 correctement -- il lui faut aussi le point_forecast externe de CE
    cas precis, pas un placeholder fixe).

    m0 est calcule en UN SEUL appel batche a compute_margin sur tout
    X_test (predict_interval de tous les backbones utilises ici accepte
    deja un batch complet, voir backbones.py), au lieu d'une boucle Python
    point par point -- meme resultat, sans le cout de milliers de petits
    appels tensoriels. Pour SPCI, un seul SPCIBackboneAdapter est construit
    avec le VECTEUR complet point_forecasts (au lieu d'un objet par case
    avec un point_forecast scalaire)."""
    want_negative = config.scenario in (Scenario.ANOMALY_TO_NORMAL, Scenario.WIDEN_STAY_ANOMALY)

    X_all = torch.tensor(data_bundle.X_test, dtype=torch.float32, device=DEVICE)
    y_all = torch.tensor(data_bundle.y_test, dtype=torch.float32, device=DEVICE)
    if point_forecasts is not None:
        from backbones import SPCIBackboneAdapter
        fc_all = torch.tensor(point_forecasts, dtype=torch.float32, device=DEVICE)
        active = SPCIBackboneAdapter(backbone.spci, fc_all)
    else:
        active = backbone
    with torch.no_grad():
        m0_true, _, _ = compute_margin(active, X_all, y_all)
    m0_true = m0_true.cpu().numpy()

    mask = (m0_true < 0) if want_negative else (m0_true > 0)
    idx = np.where(mask)[0]

    if verbose:
        n_aami_pos = (data_bundle.labels_test == 1).sum()
        print(f"  [select_cases] AAMI dit anormal: {n_aami_pos} pts | "
              f"modele dit anormal (m0<0): {(m0_true < 0).sum()} pts | "
              f"accord: {((data_bundle.labels_test == 1) & (m0_true < 0)).sum()} pts | "
              f"candidats pour ce scenario: {len(idx)} pts")

    if len(idx) == 0:
        return np.array([], dtype=int)

    order = idx[np.argsort(np.abs(m0_true[idx]))]
    if len(order) <= n_cases:
        return order
    positions = np.linspace(0, len(order) - 1, n_cases).astype(int)
    return order[positions]


def build_setup(config: ExperimentConfig, verbose=True):
    """Construit ce qui est CHER et PARTAGEABLE entre plusieurs methodes de
    recherche testees sur la meme table : chargement des donnees (reseau/
    disque), entrainement du backbone, entrainement du decodeur. A calculer
    UNE SEULE FOIS par (dataset, backbone, decodeur), jamais une fois par
    methode -- sinon chaque methode compare sur un probleme different (voir
    README) et, en prime, chaque appel retente un chargement reseau fragile.

    Memoise en memoire (voir setup_cache.py) : un appel ulterieur avec un
    config qui ne differe que par cell/epsilon/sparsity/scenario/n_cases/...
    renvoie le MEME objet setup immediatement, sans reconstruire/reentrainer
    quoi que ce soit. Optionnellement, les poids du decodeur peuvent aussi
    etre persistes sur disque entre deux lancements separes, voir
    setup_cache.set_disk_cache / variable d'environnement CFX_DISK_CACHE."""
    key = setup_cache_key(config)
    cached = get_memory_setup(key)
    if cached is not None:
        if verbose:
            print(f"[build_setup] {config.dataset.value} / {config.backbone.value} / "
                  f"{config.decoder.value} -- reutilise (cache memoire, rien reconstruit)")
        return cached

    if verbose:
        print(f"[build_setup] {config.dataset.value} / {config.backbone.value} / {config.decoder.value}")

    data_bundle = load_dataset(config)
    backbone = build_backbone(config, data_bundle)
    spci_raw = None

    if config.backbone == Backbone.SPCI:
        from backbones import build_spci_case_data
        data_bundle = build_spci_case_data(backbone, data_bundle)
        spci_raw = backbone

    dim = data_bundle.X_train.shape[1]
    decoder = build_decoder(config, data_bundle.X_train, dim)
    if not load_decoder_weights(key, decoder, verbose=verbose):
        decoder = train_decoder(decoder, data_bundle.X_train, data_bundle.y_train)
        save_decoder_weights(key, decoder)

    setup = {"data_bundle": data_bundle, "backbone": backbone, "decoder": decoder, "spci_raw": spci_raw}
    store_memory_setup(key, setup)
    return setup


def run_experiment(config: ExperimentConfig, verbose=True, return_cases=False, setup=None,
                    make_figures=True, fig_dir="."):
    """Si setup est fourni (dict de build_setup), le reutilise au lieu de
    tout recharger/reentrainer -- c'est ce qu'il faut faire des qu'on compare
    plusieurs methodes sur le meme (dataset, backbone, decodeur).

    make_figures=True (par defaut) : genere automatiquement une galerie et
    (si assez de succes) une figure de confirmation de l'identite
    margin/centre a la fin du run, sauvegardees dans fig_dir. Passer
    make_figures=False pour desactiver (ex: usage interne dans une
    boucle ou les figures seraient redondantes).

    Retourne toujours (df, raw_cases) si return_cases=True, sinon juste df
    -- mais les cas bruts sont TOUJOURS collectes en interne (necessaire
    pour make_figures), retourner=False evite juste de les exposer."""
    if verbose:
        print(f"=== {config.label()} ===")

    if setup is None:
        setup = build_setup(config, verbose=verbose)
    data_bundle = setup["data_bundle"]
    backbone = setup["backbone"]
    decoder = setup["decoder"]
    spci_raw = setup["spci_raw"]

    # Pour SPCI, predict_interval a besoin d'un point_forecast par cas -- on
    # passe le tableau REEL des point_forecasts a select_cases (pas un
    # placeholder unique) pour que le m0 utilise pour la SELECTION soit
    # le meme que celui utilise ensuite pour la recherche.
    if config.backbone == Backbone.SPCI:
        # petit wrapper juste pour porter .spci jusqu'a select_cases
        class _SPCICarrier:
            def __init__(self, spci):
                self.spci = spci
        case_idx = select_cases(data_bundle, _SPCICarrier(spci_raw), config, config.n_cases,
                                 point_forecasts=data_bundle.point_forecast_test)
    else:
        case_idx = select_cases(data_bundle, backbone, config, config.n_cases)
    if verbose:
        print(f"{len(case_idx)} cas selectionnes")

    rows = []
    raw_cases = []
    n_errors = 0
    for i in case_idx:
        x_factual = torch.tensor(data_bundle.X_test[i], dtype=torch.float32, device=DEVICE)
        y_t = float(data_bundle.y_test[i])

        if config.backbone == Backbone.SPCI:
            point_fc = float(data_bundle.point_forecast_test[i])
            active_backbone = SPCIBackboneAdapter(spci_raw, point_fc)
        else:
            active_backbone = backbone

        m0, hat_y0, sigma0 = compute_margin(active_backbone, x_factual, torch.tensor(y_t, device=DEVICE))
        m0 = m0.item()
        hat_y0 = hat_y0.item()
        sigma0 = sigma0.item()

        try:
            results = run_search_diverse(active_backbone, decoder, x_factual, y_t, config, m0,
                                          X_calib=data_bundle.X_calib)
            # le df de synthese (tables, comparaisons) s'appuie sur le
            # premier resultat -- le meilleur trouve si k_diverse==1,
            # sinon le premier de l'ensemble divers, coherent avec le
            # comportement d'origine pour ne rien casser en aval.
            result = results[0]
            report = check_definition_1(result, active_backbone, decoder, x_factual, y_t, config, m0)
            rows.append({
                "case_idx": int(i), "m0": m0, "feasible": result.feasible,
                "distance": result.distance, "n_dims_changed": result.n_dims_changed,
                "m_obtained": result.m_obtained, "log_likelihood": result.log_likelihood,
                "runtime": result.runtime,
                "definition_1_valid": report.all_valid,
                "validity": report.validity, "channel_preservation": report.channel_preservation,
                "n_diverse_found": len(results),
            })
            # raw_cases stocke TOUS les contrefactuels trouves pour ce cas,
            # pas seulement le premier -- plotting.py les affiche tous.
            x_stars, n_dims_list, feasibles, hat_ys, sigmas = [], [], [], [], []
            for r in results:
                x_star_t = torch.tensor(r.x_star, dtype=torch.float32, device=DEVICE)
                hat_y_star, sigma_star = active_backbone.predict_interval(x_star_t)
                x_stars.append(r.x_star)
                n_dims_list.append(r.n_dims_changed)
                feasibles.append(r.feasible)
                hat_ys.append(hat_y_star.item())
                sigmas.append(sigma_star.item())
            raw_cases.append({
                "case_idx": int(i), "x_factual": x_factual.cpu().numpy(),
                "x_star": results[0].x_star,          # retro-compatible (1er element)
                "x_stars": x_stars,                    # tous les contrefactuels
                "n_dims_changed_list": n_dims_list,
                "feasibles_list": feasibles,
                "y_t": y_t,                             # NOUVEAU : la vraie valeur jugee
                "hat_y_factual": hat_y0, "sigma_factual": sigma0,   # NOUVEAU : intervalle factuel
                "hat_y_stars": hat_ys, "sigma_stars": sigmas,        # NOUVEAU : intervalle(s) contrefactuel(s)
                "m0": m0, "feasible": result.feasible, "n_dims_changed": result.n_dims_changed,
                "center_shift": abs(hat_ys[0] - hat_y0),
                # decomposition centre/largeur (RQ3) : signees, pas en
                # valeur absolue, pour pouvoir distinguer la direction du
                # mouvement, pas seulement son ampleur.
                "delta_y_hat": hat_ys[0] - hat_y0,
                "delta_w": 2 * sigmas[0] - 2 * sigma0,
                "runtime": result.runtime,
            })
        except Exception as e:
            n_errors += 1
            rows.append({"case_idx": int(i), "m0": m0, "feasible": False, "error": str(e)})

    df = pd.DataFrame(rows)
    if verbose and len(df) > 0:
        n_feasible = df["feasible"].sum() if "feasible" in df else 0
        print(f"Succes : {n_feasible}/{len(df)}" + (f"  ({n_errors} erreurs internes)" if n_errors else ""))
        if "distance" in df and df["feasible"].any():
            print(f"Distance mediane (succes) : {df[df['feasible']]['distance'].median():.4f}")
        if n_errors > 0 and "error" in df:
            print(f"  Exemples d'erreur : {df[df['error'].notna()]['error'].unique()[:3]}")

        failed = df[~df["feasible"]] if "feasible" in df else df
        if len(failed) > 0:
            failed_sorted = failed.reindex(failed["m0"].abs().sort_values().index)
            print(f"  Cas echoues ({len(failed)}), tries par |m0| croissant :")
            print(f"    m0 = {[round(v, 3) for v in failed_sorted['m0'].tolist()]}")
            m0_success = df[df["feasible"]]["m0"].abs() if "feasible" in df and df["feasible"].any() else None
            if m0_success is not None and len(m0_success) > 0:
                print(f"    (pour comparaison, |m0| des succes : min={m0_success.min():.3f}, "
                      f"median={m0_success.median():.3f}, max={m0_success.max():.3f})")

    if make_figures and len(raw_cases) > 0:
        import os
        from plotting import plot_gallery, plot_identity_confirmation
        os.makedirs(fig_dir, exist_ok=True)
        label = config.label()
        try:
            gallery_path = os.path.join(fig_dir, f"gallery_{label}.png")
            plot_gallery(raw_cases, gallery_path)
            if verbose:
                print(f"  Figure -> {gallery_path}")
        except Exception as e:
            if verbose:
                print(f"  (galerie non generee : {e})")
        n_feasible_cases = sum(1 for c in raw_cases if c["feasible"])
        # La figure d'identite margin/centre n'a de sens que si le centre
        # peut reellement bouger : sur center_fixed_width_free, le
        # deplacement du centre est CONTRAINT a rester sous epsilon par
        # construction, donc plat par definition -- generer cette figure
        # sur ce genre de run donnerait une pente ~0 qui ressemble a une
        # refutation de la Proposition 1 alors que c'est juste la
        # contrainte qui fait son travail.
        center_is_free = config.cell.value in ("center_free_width_fixed", "both_free")
        if n_feasible_cases >= 5 and center_is_free:
            try:
                identity_path = os.path.join(fig_dir, f"identity_{label}.png")
                cases_df = pd.DataFrame(raw_cases)
                slope, r, p = plot_identity_confirmation(cases_df, identity_path)
                if verbose:
                    print(f"  Figure -> {identity_path} (pente={slope:.3f}, r={r:.3f})")
            except Exception as e:
                if verbose:
                    print(f"  (figure identite non generee : {e})")
        elif n_feasible_cases >= 5 and verbose:
            print(f"  (figure identite non generee : centre contraint sur cette case, "
                  f"pas de sens a mesurer son deplacement -- cf. cell=both_free ou "
                  f"center_free_width_fixed pour cette figure)")

    if return_cases:
        return df, raw_cases
    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=[d.value for d in Dataset], default="mitbih")
    p.add_argument("--backbone", choices=[b.value for b in Backbone], default="raw_context")
    p.add_argument("--cell", choices=[c.value for c in Cell], default="center_fixed_width_free")
    p.add_argument("--decoder", choices=[d.value for d in Decoder], default="flow")
    p.add_argument("--distance_space", choices=[d.value for d in DistanceSpace], default="latent")
    p.add_argument("--sparsity", choices=[s.value for s in SparsityMethod], default="none")
    p.add_argument("--scenario", choices=[s.value for s in Scenario], default="anomaly_to_normal")
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--n_cases", type=int, default=15)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = ExperimentConfig(
        dataset=Dataset(args.dataset), backbone=Backbone(args.backbone),
        cell=Cell(args.cell), decoder=Decoder(args.decoder),
        distance_space=DistanceSpace(args.distance_space),
        sparsity=SparsityMethod(args.sparsity), scenario=Scenario(args.scenario),
        epsilon=args.epsilon, n_cases=args.n_cases,
    )
    df = run_experiment(config)
    out_path = args.out or f"results_{config.label()}.csv"
    df.to_csv(out_path, index=False)
    print(f"Resultats -> {out_path}")
