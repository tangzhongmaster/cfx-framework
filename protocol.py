"""
Protocole experimental FERME (15 experiences, 0-14), fixe par l'utilisateur
apres revue complete -- remplace reproduce_paper.py (19 cles / ~35 configs,
voir reproduce_paper_legacy.py pour l'ancienne version, conservee pour
reference, plus utilisee).

REGLES ABSOLUES (section T du protocole, a respecter partout dans ce
fichier) :
  - ne pas ajouter d'experience/dataset/backbone/optimiseur non liste ici ;
  - memes cas entre methodes comparees ; seeds fixes (config.seed) ;
  - ne jamais remplacer les metriques par cas par un score global unique ;
  - documenter les cas exclus et la raison exacte (champ exclusion_reason) ;
  - si un resultat est inattendu, s'arreter et signaler -- pas de nouvelle
    branche scientifique inventee pour l'expliquer.

DECISIONS EXPLICITES VERROUILLEES (pas deduites seul, validees par
l'utilisateur -- voir l'echange qui a precede ce fichier) :
  1. MSL remplace MIT-BIH dans le benchmark principal. Datasets reels =
     MSL + NAB + UCR (+ synthetic_ar, synthetic_regime pour le controle
     mecanistique, RQ7).
  2. MSL : H_t MULTIVARIE (toutes les features du canal), y_t SCALAIRE
     (1ere feature, convention telemanom) -- voir data.py:load_msl.
  3. UCR : loader ecrit (data.py:load_ucr), univarie par construction,
     AUCUNE serie par defaut -- doit etre fournie explicitement partout
     ou UCR est utilise ci-dessous (voir UCR_SERIES_NAME en tete de
     fichier, a renseigner avant de lancer quoi que ce soit touchant UCR).
  4. Exp. 8/9 (search methods, cascade) : les noms "multi-start/annealing/
     genetic/hybrid/cascade" n'existaient QUE dans l'ancien code pre-
     refonte, en CONTEXTE BRUT -- reconstruits ici en espace LATENT
     (search_variants.py), nouveau code, pas un portage. A relire avant de
     presenter Exp. 8/9 comme "deja implemente" dans le papier.

NOTE DE DESIGN NON TRANCHEE PAR L'UTILISATEUR (a valider avant de faire
confiance a Exp. 2/3) : "memes 40 cas pour toutes les configurations" a
l'interieur d'une experience qui croise plusieurs BACKBONES (Exp. 2, 3)
est ambigu -- des backbones differents calculent des m0 differents sur le
MEME point, donc une selection independante par backbone donnerait en
general des ENSEMBLES d'indices differents (c'est exactement le bug #1 du
README, deja corrige dans l'autre sens : selection sur le vrai m0 du
modele, pas sur un label fixe). Choix fait ICI, a auditer explicitement :
les indices sont selectionnes UNE FOIS via le backbone de reference
(RAW_CONTEXT, "backbone principal" du niveau 2), puis REUTILISES tels
quels pour les autres backbones de la meme experience. Quand un cas n'a
pas le bon signe de m0 pour un AUTRE backbone (desaccord entre backbones
sur ce qui est anormal), il est EXCLU explicitement (excluded=True,
exclusion_reason renseignee), jamais recalcule ni cache. Cette convention
doit etre revue avant publication -- ce n'est pas neutre scientifiquement.
"""

import os
import time
import numpy as np
import pandas as pd
import torch

from config import (ExperimentConfig, Dataset, Backbone, Cell, Decoder,
                     DistanceSpace, SparsityMethod, Scenario)
from data import load_dataset
from backbones import (build_backbone, build_spci_case_data, SPCIBackboneAdapter,
                       SPCIReplayAdapter, build_test_series, spci_factual_state)
from decoders import build_decoder, train_decoder
from margin import compute_margin
from search import run_search_diverse
import search_variants as sv
from setup_cache import (setup_cache_key, get_memory_setup, store_memory_setup,
                          load_decoder_weights, save_decoder_weights)
from device import DEVICE


# ======================================================================
# A renseigner avant de lancer toute experience touchant UCR (Exp. 4, 5) --
# aucune valeur par defaut n'est fournie par data.py:load_ucr, par design.
# Les deux peuvent aussi etre fixes depuis l'exterieur (notebook Colab, shell)
# via les variables d'environnement UCR_SERIES_NAME / UCR_PATH -- pratique
# pour ne pas avoir a editer ce fichier a chaque fois.
# ======================================================================
UCR_SERIES_NAME = os.environ.get("UCR_SERIES_NAME")   # ex: "InternalBleeding5" -- DOIT identifier UN SEUL fichier
UCR_PATH = os.environ.get("UCR_PATH")                   # dossier contenant les .txt de l'archive

CASE_COLUMNS = [
    "experiment", "dataset", "backbone", "cell", "case_idx",
    "m0", "m_star", "feasible",
    "y_hat_factual", "y_hat_star", "delta_y_hat",
    "w_factual", "w_star", "delta_w",
    "abs_delta_y_hat", "abs_delta_w",
    "center_constraint_violated", "width_constraint_violated",
    "distance", "n_dims_changed", "runtime", "log_likelihood",
    "excluded", "exclusion_reason",
]


# ======================================================================
# MOTEUR PARTAGE
# ======================================================================

def build_setup_generic(config: ExperimentConfig, verbose=True):
    """Meme logique que run_experiment.build_setup (donnees + backbone +
    decodeur, une fois par (dataset,backbone,decodeur), jamais par
    cell/epsilon/sparsity/methode de recherche) -- duplique ici pour que
    protocol.py n'ait pas de dependance circulaire sur run_experiment.py.

    Memoise via setup_cache.py (cache PARTAGE avec run_experiment.py, meme
    cle) : exp1, le setup de reference d'exp2/exp3, exp6, exp7, exp8, exp9
    et exp10/exp11 retombent presque tous sur (MSL ou NAB, RAW_CONTEXT,
    FLOW) -- avant ce cache, chacun reconstruisait ET reentrainait (1000
    epoques) depuis zero. Maintenant, le premier qui passe construit, tous
    les suivants reutilisent tel quel."""
    key = setup_cache_key(config)
    # Cle MEMOIRE : le setup contient le data_bundle (segment de TEST inclus), qui depend de
    # magnitude / phi_2 / n_test. Ces champs restent hors de `key` (poids du decodeur, sur
    # disque : train/calib n'en dependent pas) mais DOIVENT etre dans la cle memoire, sinon
    # exp13/exp14 reutilisent le test de la premiere valeur pour toutes les autres.
    mem_key = key + (config.synthetic_anomaly_magnitude, config.synthetic_phi_2,
                     config.synthetic_n_test)
    cached = get_memory_setup(mem_key)
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
    test_series = None
    if config.backbone == Backbone.SPCI:
        # REJEU : SPCI travaille maintenant sur la fenetre BRUTE (comme RawContext).
        # Les residus sont recalcules a chaque evaluation par SPCIReplayAdapter, on
        # ne construit donc plus de bundle "en espace de residus".
        spci_raw = backbone
        test_series, contiguous = build_test_series(data_bundle.X_test, config.context_window)
        if not contiguous:
            print("  [spci] ATTENTION : fenetres de test non consecutives, rejeu non fiable.")
    dim = data_bundle.X_train.shape[1]
    decoder = build_decoder(config, data_bundle.X_train, dim)
    if not load_decoder_weights(key, decoder, verbose=verbose):
        decoder = train_decoder(decoder, data_bundle.X_train, data_bundle.y_train)
        save_decoder_weights(key, decoder)

    setup = {"data_bundle": data_bundle, "backbone": backbone, "decoder": decoder, "spci_raw": spci_raw,
             "test_series": test_series}
    store_memory_setup(mem_key, setup)
    return setup


def active_backbone_for_case(setup, config, i):
    if config.backbone == Backbone.SPCI:
        spci = setup["spci_raw"]
        R, L = spci.residual_window, config.context_window
        V = setup["test_series"]                       # (n+L-1, F), serie de test reelle
        hist = V[max(0, i - R):i]
        if len(hist) < R:                              # tout debut du test : on complete a gauche
            first = V[:1] if len(hist) == 0 else hist[:1]
            hist = np.concatenate([np.repeat(first, R - len(hist), axis=0), hist], axis=0)
        return SPCIReplayAdapter(spci, torch.tensor(hist, dtype=torch.float32), L)
    if config.backbone == Backbone.TSFM:
        return setup["backbone"].case_adapter(i)     # W1ACAS rejoue sur la fenetre perturbee
    return setup["backbone"]


def want_negative_for_scenario(scenario):
    return scenario in (Scenario.ANOMALY_TO_NORMAL, Scenario.WIDEN_STAY_ANOMALY)


def select_reference_cases(setup, config, n_cases, verbose=True):
    """Selection stratifiee par |m0|, calculee avec LE backbone de ce
    setup -- identique a run_experiment.select_cases. Utilisee seule pour
    une selection standard (un seul backbone), ou comme REFERENCE partagee
    entre backbones (voir note de design en tete de fichier).

    m0 est calcule en UN SEUL appel batche a compute_margin sur tout
    X_test, au lieu d'une boucle Python appelant active_backbone_for_case
    + compute_margin point par point (c'etait le gros du cout de
    select_reference_cases : des milliers de petits appels tensoriels avec
    overhead Python a chaque fois, voir le "5191 candidats" observe sur
    MSL). predict_interval(x) de tous les backbones utilises ici
    (RawContext, SplitCP, Adaptive, SPCI via SPCIBackboneAdapter) accepte
    deja un batch complet -- voir backbones.py -- donc le resultat est
    EXACTEMENT le meme, juste sans la boucle. Pour SPCI, un seul
    SPCIBackboneAdapter est construit avec le VECTEUR complet de
    point_forecast_test (au lieu d'un objet par case avec un scalaire),
    predict_interval_from_residuals() le diffuse (broadcasting) sur le
    batch de fenetres de residus exactement comme il le faisait deja un
    par un. active_backbone_for_case reste utilisee ailleurs (run_one_case,
    ligne ~176/529) pour le cas par-cas, inchangee."""
    data_bundle = setup["data_bundle"]
    want_negative = want_negative_for_scenario(config.scenario)

    X_all = torch.tensor(data_bundle.X_test, dtype=torch.float32, device=DEVICE)
    y_all = torch.tensor(data_bundle.y_test, dtype=torch.float32, device=DEVICE)
    if config.backbone == Backbone.SPCI:
        # marge factuelle de toutes les lignes (NaN pour les R premieres : pas assez
        # de residus reels) -- meme valeur que SPCIReplayAdapter sur la fenetre factuelle
        hat_all, sig_all = spci_factual_state(setup["spci_raw"], data_bundle.X_test, data_bundle.y_test)
        m0_all = (sig_all - (y_all - hat_all).abs()).cpu().numpy()
    else:
        if config.backbone == Backbone.TSFM:
            active = setup["backbone"].case_adapter(None)  # vecteur des q_t factuels
        else:
            active = setup["backbone"]
        with torch.no_grad():
            m0_all, _, _ = compute_margin(active, X_all, y_all)
        m0_all = m0_all.cpu().numpy()

    mask = (m0_all < 0) if want_negative else (m0_all > 0)
    if config.backbone == Backbone.TSFM:
        # le rejeu a besoin de L pas de test reel avant la fenetre : on exclut les L premiers cas
        mask = mask & (np.arange(len(m0_all)) >= setup["backbone"].L)
    idx = np.where(mask)[0]
    if verbose:
        print(f"  [select_reference_cases] {config.dataset.value}/{config.backbone.value} : "
              f"{len(idx)} candidats pour {config.scenario.value}")
    if len(idx) == 0:
        return np.array([], dtype=int)
    order = idx[np.argsort(np.abs(m0_all[idx]))]
    if len(order) <= n_cases:
        return order
    positions = np.linspace(0, len(order) - 1, n_cases).astype(int)
    return order[positions]


def run_one_case(setup, config, i, experiment_name, search_fn=None):
    """Coeur partage par TOUTES les experiences de ce fichier. search_fn,
    si fourni, remplace run_search_diverse standard (utilise par Exp. 8/9
    pour appeler les variantes de search_variants.py directement) --
    signature attendue : search_fn(backbone, decoder, x_factual, y_t,
    config, m0) -> ResultRow (une seule, pas une liste diverse)."""
    data_bundle = setup["data_bundle"]
    decoder = setup["decoder"]
    active_backbone = active_backbone_for_case(setup, config, i)

    x_factual = torch.tensor(data_bundle.X_test[i], dtype=torch.float32, device=DEVICE)
    y_t = float(data_bundle.y_test[i])
    m0_t, hat_y0_t, sigma0_t = compute_margin(active_backbone, x_factual, torch.tensor(y_t, device=DEVICE))
    m0, hat_y0, sigma0 = m0_t.item(), hat_y0_t.item(), sigma0_t.item()

    base_row = dict(experiment=experiment_name, dataset=config.dataset.value,
                     backbone=config.backbone.value, cell=config.cell.value, case_idx=int(i),
                     m0=m0, y_hat_factual=hat_y0, w_factual=2 * sigma0)

    want_negative = want_negative_for_scenario(config.scenario)
    sign_ok = (m0 < 0) if want_negative else (m0 > 0)
    if not sign_ok:
        base_row.update(
            m_star=None, feasible=False, y_hat_star=None, delta_y_hat=None,
            w_star=None, delta_w=None, abs_delta_y_hat=None, abs_delta_w=None,
            center_constraint_violated=None, width_constraint_violated=None,
            distance=None, n_dims_changed=None, runtime=0.0, log_likelihood=None,
            excluded=True,
            exclusion_reason=(f"m0={m0:.4f} de signe incompatible avec le scenario "
                               f"{config.scenario.value} pour ce backbone (cas selectionne via "
                               f"un backbone de reference partage -- voir note de design, en-tete "
                               f"de protocol.py)"))
        return base_row

    t0 = time.perf_counter()
    try:
        if search_fn is not None:
            # Les variantes de search_variants.py (Exp. 8/9) ne mesurent PAS
            # leur propre runtime (ResultRow.runtime y reste a 0.0 par
            # defaut, contrairement a search.run_search qui le chronometre
            # lui-meme) -- mesure ici, au point d'appel, et ECRASE
            # explicitement le champ ci-dessous plutot que de laisser
            # passer un 0.0 silencieux dans la colonne runtime.
            result = search_fn(active_backbone, decoder, x_factual, y_t, config, m0)
            measured_runtime = time.perf_counter() - t0
            result.runtime = measured_runtime
        else:
            results = run_search_diverse(active_backbone, decoder, x_factual, y_t, config, m0,
                                          X_calib=data_bundle.X_calib)
            result = results[0]
    except Exception as e:
        base_row.update(
            m_star=None, feasible=False, y_hat_star=None, delta_y_hat=None,
            w_star=None, delta_w=None, abs_delta_y_hat=None, abs_delta_w=None,
            center_constraint_violated=None, width_constraint_violated=None,
            distance=None, n_dims_changed=None, runtime=time.perf_counter() - t0,
            log_likelihood=None, excluded=True,
            exclusion_reason=f"erreur interne pendant la recherche : {type(e).__name__}: {e}")
        return base_row

    x_star_t = torch.tensor(result.x_star, dtype=torch.float32, device=DEVICE)
    hat_y_star_t, sigma_star_t = active_backbone.predict_interval(x_star_t)
    hat_y_star, sigma_star = hat_y_star_t.item(), sigma_star_t.item()
    w_star = 2 * sigma_star
    w_factual = 2 * sigma0
    delta_y_hat = hat_y_star - hat_y0
    delta_w = w_star - w_factual

    # Pas de champ epsilon_w separe dans ExperimentConfig : la seule
    # tolerance implementee (config.epsilon) est celle deja appliquee par
    # margin.py:channel_violation au canal designe fixe par la cell. On
    # l'utilise ICI pour les DEUX verifications generiques (centre ET
    # largeur), que la cell contraigne ce canal ou non -- documente
    # explicitement, pas une tolerance "largeur" distincte inventee.
    base_row.update(
        m_star=result.m_obtained, feasible=bool(result.feasible),
        y_hat_star=hat_y_star, delta_y_hat=delta_y_hat,
        w_star=w_star, delta_w=delta_w,
        abs_delta_y_hat=abs(delta_y_hat), abs_delta_w=abs(delta_w),
        center_constraint_violated=abs(delta_y_hat) > config.epsilon + 1e-4,
        width_constraint_violated=abs(delta_w) > config.epsilon + 1e-4,
        distance=result.distance, n_dims_changed=result.n_dims_changed,
        runtime=result.runtime, log_likelihood=result.log_likelihood,
        excluded=False, exclusion_reason=None,
    )
    return base_row


def run_standard_experiment(config, experiment_name, search_fn=None, verbose=True):
    """Une selection de cas (via le backbone propre a `config`), une boucle
    run_one_case -- le cas standard (pas de partage d'indices entre
    backbones)."""
    setup = build_setup_generic(config, verbose=verbose)
    case_idx = select_reference_cases(setup, config, config.n_cases, verbose=verbose)
    rows = [run_one_case(setup, config, i, experiment_name, search_fn=search_fn) for i in case_idx]
    return pd.DataFrame(rows, columns=CASE_COLUMNS), setup, case_idx


def _df(rows):
    return pd.DataFrame(rows, columns=CASE_COLUMNS)


# ======================================================================
# EXPERIENCE 0 -- PIPELINE SANITY (pas un resultat scientifique)
# ======================================================================

def exp0_pipeline_sanity(n_cases=5, ucr_series_name=None, verbose=True):
    """Pour chaque (dataset reel/synthetique) x (backbone), cell B par
    defaut : verifie que la chaine entiere tourne (entrainement,
    calibration, intervalle, selection, m0, recherche, m_star, deltas,
    contraintes, distance, runtime). N'est jamais presente comme une
    contribution -- ne modifie aucun design scientifique a partir de ce
    resultat (section D du protocole)."""
    ucr_series_name = ucr_series_name or UCR_SERIES_NAME
    datasets = [Dataset.MSL, Dataset.NAB, Dataset.SYNTHETIC_AR, Dataset.SYNTHETIC_REGIME]
    if ucr_series_name is not None:
        datasets.append(Dataset.UCR)
    elif verbose:
        print("  [exp0] UCR_SERIES_NAME non fourni -- UCR exclu de la sanity check "
              "(aucune serie par defaut, par design, voir data.py:load_ucr)")

    backbones = [Backbone.SPLIT_CP, Backbone.RAW_CONTEXT, Backbone.SPCI]
    all_rows = []
    for ds in datasets:
        for bb in backbones:
            kwargs = dict(dataset=ds, backbone=bb, cell=Cell.CENTER_FIXED_WIDTH_FREE,
                           decoder=Decoder.FLOW, n_cases=n_cases)
            if ds == Dataset.UCR:
                kwargs.update(ucr_series_name=ucr_series_name, ucr_path=UCR_PATH)
            config = ExperimentConfig(**kwargs)
            try:
                df, _, _ = run_standard_experiment(config, "exp0_pipeline_sanity", verbose=verbose)
                all_rows.append(df)
            except Exception as e:
                if verbose:
                    print(f"  [exp0] ECHEC {ds.value}/{bb.value} : {type(e).__name__}: {e}")
                all_rows.append(_df([dict(experiment="exp0_pipeline_sanity", dataset=ds.value,
                                           backbone=bb.value, cell=Cell.CENTER_FIXED_WIDTH_FREE.value,
                                           case_idx=-1, m0=None, excluded=True,
                                           exclusion_reason=f"echec pipeline: {type(e).__name__}: {e}")]))
    return pd.concat(all_rows, ignore_index=True) if all_rows else _df([])


# ======================================================================
# EXPERIENCE 1 -- SPLIT CP STRUCTURAL CONTROL
# ======================================================================

def exp1_split_cp_control(n_cases=40, verbose=True):
    """Delta_w == 0 (precision numerique) pour tous les cas FAISABLES de
    Split CP, cell B -- propriete STRUCTURELLE verifiee, pas decouverte.
    NE PAS utiliser feasibility_rate==0 comme assertion (un cas proche de
    la frontiere peut etre faisable via le centre, dans la tolerance
    epsilon -- voir smoke_test.py)."""
    config = ExperimentConfig(dataset=Dataset.MSL, backbone=Backbone.SPLIT_CP,
                               cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                               n_cases=n_cases)
    df, _, _ = run_standard_experiment(config, "exp1_split_cp_control", verbose=verbose)
    feasible = df[df["feasible"] == True]
    if verbose and len(feasible) > 0:
        print(f"  max|Delta_w|={feasible['abs_delta_w'].max():.6f}  "
              f"median|Delta_w|={feasible['abs_delta_w'].median():.6f}  "
              f"max|Delta_y_hat|={feasible['abs_delta_y_hat'].max():.6f}  "
              f"feasibility={len(feasible)}/{len(df)}")
    return df


# ======================================================================
# EXPERIENCES 2/3 -- CHANNEL DECOMPOSITION (3 backbones x 3 cells)
# ======================================================================

def _channel_decomposition(dataset: Dataset, experiment_name, n_cases=40,
                            msl_channel="T-1", verbose=True):
    backbones = [Backbone.SPLIT_CP, Backbone.RAW_CONTEXT, Backbone.SPCI]
    cells = [Cell.BOTH_FREE, Cell.CENTER_FIXED_WIDTH_FREE, Cell.CENTER_FREE_WIDTH_FIXED]
    extra = dict(msl_channel=msl_channel) if dataset == Dataset.MSL else {}

    ref_config = ExperimentConfig(dataset=dataset, backbone=Backbone.RAW_CONTEXT,
                                   cell=cells[0], decoder=Decoder.FLOW, n_cases=n_cases, **extra)
    ref_setup = build_setup_generic(ref_config, verbose=verbose)
    case_idx = select_reference_cases(ref_setup, ref_config, n_cases, verbose=verbose)

    rows = []
    for bb in backbones:
        setup = ref_setup if bb == Backbone.RAW_CONTEXT else build_setup_generic(
            ExperimentConfig(dataset=dataset, backbone=bb, cell=cells[0], decoder=Decoder.FLOW,
                              n_cases=n_cases, **extra), verbose=verbose)
        for cell in cells:
            config = ExperimentConfig(dataset=dataset, backbone=bb, cell=cell,
                                       decoder=Decoder.FLOW, n_cases=n_cases, **extra)
            if verbose:
                print(f"\n=== {experiment_name} backbone={bb.value} cell={cell.value} ===")
            for i in case_idx:
                rows.append(run_one_case(setup, config, i, experiment_name))
    return _df(rows)


def exp2_channel_decomposition_msl(n_cases=40, msl_channel="T-1", verbose=True):
    return _channel_decomposition(Dataset.MSL, "exp2_channel_decomposition_msl",
                                   n_cases=n_cases, msl_channel=msl_channel, verbose=verbose)


def exp3_channel_decomposition_nab(n_cases=40, verbose=True):
    return _channel_decomposition(Dataset.NAB, "exp3_channel_decomposition_nab",
                                   n_cases=n_cases, verbose=verbose)


# ======================================================================
# EXPERIENCE 4 -- GENERALISATION (cell B, backbones Raw/SPCI, 3 datasets)
# ======================================================================

def exp4_generalization(n_cases=40, msl_channel="T-1", ucr_series_name=None, verbose=True):
    """exp4 fait partie de EXPERIMENT_ORDER (--all) : contrairement a exp5
    (hors ordre par defaut, cf. plus bas), on ne peut pas se permettre de
    lever une ValueError ici si UCR_SERIES_NAME n'est pas fourni -- ca
    ferait echouer tout `protocol.py --all` a chaque fois. Meme degradation
    gracieuse que exp0_pipeline_sanity : MSL+NAB tournent toujours, UCR est
    ajoute seulement si une serie est explicitement fournie."""
    ucr_series_name = ucr_series_name or UCR_SERIES_NAME
    datasets = [(Dataset.MSL, dict(msl_channel=msl_channel)), (Dataset.NAB, {})]
    if ucr_series_name is not None:
        datasets.append((Dataset.UCR, dict(ucr_series_name=ucr_series_name, ucr_path=UCR_PATH)))
    elif verbose:
        print("  [exp4] UCR_SERIES_NAME non fourni -- UCR exclu de exp4_generalization "
              "(aucune serie par defaut, par design, voir data.py:load_ucr). "
              "Pour l'inclure : protocol.py --exp exp4 apres avoir renseigne "
              "UCR_SERIES_NAME en tete de protocol.py.")

    all_rows = []
    for bb in [Backbone.RAW_CONTEXT, Backbone.SPCI]:
        for ds, extra in datasets:
            config = ExperimentConfig(dataset=ds, backbone=bb, cell=Cell.CENTER_FIXED_WIDTH_FREE,
                                       decoder=Decoder.FLOW, distance_space=DistanceSpace.LATENT,
                                       n_cases=n_cases, **extra)
            df, _, _ = run_standard_experiment(config, "exp4_generalization", verbose=verbose)
            all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)


# ======================================================================
# EXPERIENCE 5 -- UCR, CONTROLE STRUCTUREL DU CONTEXTE
# ======================================================================

def exp5_ucr_control(n_cases=40, ucr_series_name=None, verbose=True):
    """Documente explicitement (section I du protocole) : longueur de
    l'historique avant l'anomalie, position de l'anomalie, m0, feasibility,
    distance -- pas une nouvelle methode UCR-specifique."""
    ucr_series_name = ucr_series_name or UCR_SERIES_NAME
    if ucr_series_name is None:
        raise ValueError("exp5_ucr_control requiert ucr_series_name explicitement.")
    config = ExperimentConfig(dataset=Dataset.UCR, backbone=Backbone.RAW_CONTEXT,
                               cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                               ucr_series_name=ucr_series_name, ucr_path=UCR_PATH, n_cases=n_cases)
    df, setup, _ = run_standard_experiment(config, "exp5_ucr_control", verbose=verbose)
    bundle = setup["data_bundle"]
    if verbose:
        print(f"  serie={bundle.ucr_filename}  train_end={bundle.ucr_train_end}  "
              f"anomaly=[{bundle.ucr_anomaly_start},{bundle.ucr_anomaly_end}]  "
              f"historique_avant_anomalie={bundle.ucr_history_before_anomaly}")
    df["ucr_filename"] = bundle.ucr_filename
    df["ucr_history_before_anomaly"] = bundle.ucr_history_before_anomaly
    return df


# ======================================================================
# EXPERIENCE 6 -- CENTER-SHIFT IDENTITY (cell A, both_free)
# ======================================================================

def exp6_center_shift_identity(n_cases=40, msl_channel="T-1", verbose=True):
    """|Delta_y_hat| = a|m0| + b, regression sur les cas FAISABLES de cell
    A (both_free), pour (MSL,NAB) x (RawContext,SPCI). Pente attendue
    proche de 1 QUAND le centre est le mecanisme utilise -- ne pas
    presenter comme une loi universelle si le backbone differe (SPCI
    route la largeur via l'historique de residus, rien ne garantit la
    meme pente)."""
    all_rows = []
    regressions = []
    for ds, extra in [(Dataset.MSL, dict(msl_channel=msl_channel)), (Dataset.NAB, {})]:
        for bb in [Backbone.RAW_CONTEXT, Backbone.SPCI]:
            config = ExperimentConfig(dataset=ds, backbone=bb, cell=Cell.BOTH_FREE,
                                       decoder=Decoder.FLOW, n_cases=n_cases, **extra)
            df, _, _ = run_standard_experiment(config, "exp6_center_shift_identity", verbose=verbose)
            all_rows.append(df)
            feas = df[(df["feasible"] == True)]
            if len(feas) >= 3:
                x = feas["m0"].abs().values
                y = feas["abs_delta_y_hat"].values
                a, b = np.polyfit(x, y, 1)
                resid = y - (a * x + b)
                ss_res, ss_tot = float(np.sum(resid ** 2)), float(np.sum((y - y.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
                regressions.append(dict(dataset=ds.value, backbone=bb.value, slope=a, intercept=b,
                                         r2=r2, n_feasible=len(feas)))
                if verbose:
                    print(f"  [exp6] {ds.value}/{bb.value} : pente={a:.4f} intercept={b:.4f} "
                          f"R2={r2:.4f} (n={len(feas)})")
            elif verbose:
                print(f"  [exp6] {ds.value}/{bb.value} : trop peu de cas faisables pour regresser")
    cases_df = pd.concat(all_rows, ignore_index=True)
    reg_df = pd.DataFrame(regressions)
    return cases_df, reg_df


# ======================================================================
# EXPERIENCE 7 -- SENSIBILITE A EPSILON
# ======================================================================

def exp7_epsilon_sensitivity(n_cases=40, msl_channel="T-1",
                              epsilons=(0.0001, 0.01, 0.05, 0.10), verbose=True):
    """MEME 40 cas pour toutes les valeurs d'epsilon -- selectionnes UNE
    fois (epsilon ne joue aucun role dans le calcul de m0, donc les
    selectionner a epsilon=0.05 puis les reutiliser est strictement
    equivalent a les reselectionner a chaque epsilon, mais l'ecrit comme
    une garantie explicite plutot qu'une coincidence numerique)."""
    base_config = ExperimentConfig(dataset=Dataset.MSL, backbone=Backbone.RAW_CONTEXT,
                                    cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                                    msl_channel=msl_channel, n_cases=n_cases, epsilon=0.05)
    setup = build_setup_generic(base_config, verbose=verbose)
    case_idx = select_reference_cases(setup, base_config, n_cases, verbose=verbose)

    rows = []
    for eps in epsilons:
        config = ExperimentConfig(dataset=Dataset.MSL, backbone=Backbone.RAW_CONTEXT,
                                   cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                                   msl_channel=msl_channel, n_cases=n_cases, epsilon=eps)
        if verbose:
            print(f"\n=== exp7_epsilon_sensitivity epsilon={eps} ===")
        for i in case_idx:
            rows.append(run_one_case(setup, config, i, "exp7_epsilon_sensitivity"))
    df = _df(rows)
    df["epsilon"] = sum(([eps] * len(case_idx) for eps in epsilons), [])
    return df


# ======================================================================
# EXPERIENCE 8 -- COMPARAISON DES SEARCH METHODS (voir search_variants.py)
# ======================================================================

def exp8_search_methods(n_cases=40, msl_channel="T-1", verbose=True):
    config_ref = ExperimentConfig(dataset=Dataset.MSL, backbone=Backbone.RAW_CONTEXT,
                                   cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                                   msl_channel=msl_channel, n_cases=n_cases)
    setup = build_setup_generic(config_ref, verbose=verbose)
    case_idx = select_reference_cases(setup, config_ref, n_cases, verbose=verbose)

    rows = []
    for name, fn in sv.SEARCH_METHODS.items():
        if verbose:
            print(f"\n=== exp8_search_methods methode={name} ===")
        for i in case_idx:
            row = run_one_case(setup, config_ref, i, "exp8_search_methods", search_fn=fn)
            row["search_method"] = name
            rows.append(row)
    df = _df([{k: v for k, v in r.items() if k != "search_method"} for r in rows])
    df["search_method"] = [r["search_method"] for r in rows]
    return df


# ======================================================================
# EXPERIENCE 9 -- CASCADE FINALE (Hybrid -> Multi-start -> Annealing -> Cardinalite)
# ======================================================================

def exp9_cascade(n_cases=40, msl_channel="T-1", verbose=True,
                  checkpoint_path="results_exp9_checkpoint.csv", resume=True):
    """checkpoint_path : chaque cas termine est ajoute (append) a ce CSV au
    fur et a mesure, pas seulement a la toute fin -- exp9 etant la plus
    lente (cascade a 4 etages par cas), une deconnexion Colab en cours de
    route ne doit pas faire perdre les cas deja calcules. resume=True :
    si ce fichier existe deja, les cas (dataset, case_idx) qu'il contient
    sont relus et sautes, pour reprendre exactement ou on s'est arrete au
    lieu de tout relancer depuis zero."""
    done_keys = set()
    if resume and os.path.exists(checkpoint_path):
        prev = pd.read_csv(checkpoint_path)
        done_keys = set(zip(prev["dataset"], prev["case_idx"]))
        if verbose:
            print(f"  [exp9] checkpoint trouve ({checkpoint_path}) : "
                  f"{len(done_keys)} cas deja calcules, ils seront sautes.")

    header_written = os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0
    rows = list(prev.to_dict("records")) if done_keys else []

    for ds, extra in [(Dataset.MSL, dict(msl_channel=msl_channel)), (Dataset.NAB, {})]:
        config = ExperimentConfig(dataset=ds, backbone=Backbone.RAW_CONTEXT,
                                   cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                                   n_cases=n_cases, **extra)
        setup = build_setup_generic(config, verbose=verbose)
        case_idx = select_reference_cases(setup, config, n_cases, verbose=verbose)
        n_total = len(case_idx)
        if verbose:
            print(f"  [exp9] {ds.value} : cascade sur {n_total} cas -- chaque cas "
                  f"peut enchainer jusqu'a 4 recherches (hybride/GA -> multistart -> "
                  f"annealing -> cardinalite) si les precedentes echouent, donc c'est "
                  f"normalement l'experience la plus lente du protocole. Progression "
                  f"affichee cas par cas ci-dessous pour eviter l'impression de blocage.")
        for j, i in enumerate(case_idx, start=1):
            if (ds.value, i) in done_keys:
                if verbose:
                    print(f"    [exp9] {ds.value} cas {j}/{n_total} (idx={i}) : deja fait (checkpoint), saute")
                continue
            data_bundle = setup["data_bundle"]
            active_backbone = active_backbone_for_case(setup, config, i)
            x_factual = torch.tensor(data_bundle.X_test[i], dtype=torch.float32, device=DEVICE)
            y_t = float(data_bundle.y_test[i])
            m0 = compute_margin(active_backbone, x_factual, torch.tensor(y_t, device=DEVICE))[0].item()
            want_negative = want_negative_for_scenario(config.scenario)
            sign_ok = (m0 < 0) if want_negative else (m0 > 0)
            if not sign_ok:
                # cas documente explicitement comme exclu (meme regle que
                # partout ailleurs), PAS silencieusement sautee -- la
                # cascade (couteuse) n'est simplement jamais invoquee pour
                # un cas qu'on sait d'avance hors scenario.
                row = run_one_case(setup, config, i, "exp9_cascade")
                row["stage_succeeded"] = None
                if verbose:
                    print(f"    [exp9] {ds.value} cas {j}/{n_total} (idx={i}) : hors scenario, cascade sautee")
            else:
                cascade_result = sv.run_cascade(active_backbone, setup["decoder"], x_factual, y_t,
                                                 config, m0)
                row = run_one_case(setup, config, i, "exp9_cascade",
                                    search_fn=lambda *a, **kw: cascade_result.result)
                row["stage_succeeded"] = cascade_result.stage
                row["runtime"] = cascade_result.total_runtime
                if verbose:
                    print(f"    [exp9] {ds.value} cas {j}/{n_total} (idx={i}) : "
                          f"etage='{cascade_result.stage}' en {cascade_result.total_runtime:.1f}s")
            rows.append(row)
            pd.DataFrame([row]).to_csv(checkpoint_path, mode="a", index=False,
                                        header=not header_written)
            header_written = True
    return _df([{k: v for k, v in r.items() if k != "stage_succeeded"} for r in rows]).assign(
        stage_succeeded=[r["stage_succeeded"] for r in rows])


# ======================================================================
# EXPERIENCES 10/11 -- SPARSITE (fenetre contigue, seule option retenue --
# Maalej/cardinalite-latente/L1 retirees du code, voir config.py niveau 7 ;
# cette experience ne compare donc plus plusieurs methodes entre elles,
# elle verifie juste que WINDOW tourne correctement sur chaque dataset)
# ======================================================================

def _sparsity_experiment(dataset: Dataset, experiment_name, n_cases=15, verbose=True):
    methods = [SparsityMethod.WINDOW]
    base_config = ExperimentConfig(dataset=dataset, backbone=Backbone.RAW_CONTEXT,
                                    cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                                    sparsity=SparsityMethod.NONE, n_cases=n_cases)
    setup = build_setup_generic(base_config, verbose=verbose)
    case_idx = select_reference_cases(setup, base_config, n_cases, verbose=verbose)

    rows = []
    for sp in methods:
        config = ExperimentConfig(dataset=dataset, backbone=Backbone.RAW_CONTEXT,
                                   cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                                   sparsity=sp, n_cases=n_cases)
        if verbose:
            print(f"\n=== {experiment_name} sparsity={sp.value} ===")
        for i in case_idx:
            rows.append(run_one_case(setup, config, i, experiment_name))
    return _df(rows)


def exp10_sparsity_msl(n_cases=15, verbose=True):
    return _sparsity_experiment(Dataset.MSL, "exp10_sparsity_msl", n_cases=n_cases, verbose=verbose)


def exp11_sparsity_nab(n_cases=15, verbose=True):
    return _sparsity_experiment(Dataset.NAB, "exp11_sparsity_nab", n_cases=n_cases, verbose=verbose)


# ======================================================================
# EXPERIENCE 12 -- RETIREE : comparait DistanceSpace.LATENT vs LIKELIHOOD
# (niveau 4). LIKELIHOOD n'existe plus (voir config.py, decision verrouillee
# : distance euclidienne latente, seule option) -- il n'y a plus rien a
# comparer. Numero d'experience laisse VACANT plutot que renumerote, pour
# ne pas decaler silencieusement exp13/exp14 par rapport a toute reference
# externe deja faite a ces numeros.
# ======================================================================
# EXPERIENCE 13 -- SYNTHETIC AR (severite de l'anomalie)
# ======================================================================

def exp13_synthetic_ar(n_cases=40, magnitudes=(0.5, 1, 2, 3, 4), verbose=True):
    rows = []
    for mag in magnitudes:
        config = ExperimentConfig(dataset=Dataset.SYNTHETIC_AR, backbone=Backbone.RAW_CONTEXT,
                                   cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                                   synthetic_anomaly_magnitude=mag, n_cases=n_cases)
        if verbose:
            print(f"\n=== exp13_synthetic_ar magnitude={mag} ===")
        df, _, _ = run_standard_experiment(config, "exp13_synthetic_ar", verbose=verbose)
        df["anomaly_magnitude"] = mag
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ======================================================================
# EXPERIENCE 14 -- SYNTHETIC REGIME SHIFT (D_regime = |phi_1-phi_2|)
# ======================================================================

def exp14_synthetic_regime(n_cases=40, d_regimes=(0.0, 0.2, 0.4, 0.6), phi_1=0.7, verbose=True):
    rows = []
    for d in d_regimes:
        phi_2 = phi_1 - d
        config = ExperimentConfig(dataset=Dataset.SYNTHETIC_REGIME, backbone=Backbone.RAW_CONTEXT,
                                   cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                                   synthetic_phi_1=phi_1, synthetic_phi_2=phi_2, n_cases=n_cases)
        if verbose:
            print(f"\n=== exp14_synthetic_regime D_regime={d} (phi_1={phi_1}, phi_2={phi_2}) ===")
        df, _, _ = run_standard_experiment(config, "exp14_synthetic_regime", verbose=verbose)
        df["d_regime"] = d
        rows.append(df)
    return pd.concat(rows, ignore_index=True)



# ======================================================================
# EXPERIENCE 15 -- BASELINE IBM (foundation model + conformal ponderee)
# ======================================================================

def exp15_tsfm_baseline(n_cases=40, msl_channel="T-1", verbose=True,
                         checkpoint_path="results_exp15_checkpoint.csv", resume=True):
    """Compare NOTRE backbone principal (RAW_CONTEXT) au backbone de l'article
    IBM (TSFMWeightedBackbone : Granite TTM zero-shot + quantile pondere
    Wasserstein, cf. tsfm_backbone.py) sur les 3 cellules (chaque backbone
    choisit ses propres cas, comme exp4). Hors
    EXPERIMENT_ORDER : requiert le telechargement du modele HF (GPU conseille).
    TSFM est evalue avec REJEU causal (la fenetre perturbee recalcule scores,
    poids et seuil W1ACAS, voir tsfm_backbone.TSFMReplayAdapter) : la largeur n'est
    plus constante en x. Checkpoint/reprise
    par (dataset, backbone, cell, case_idx)."""
    done_keys, rows = set(), []
    if resume and os.path.exists(checkpoint_path):
        prev = pd.read_csv(checkpoint_path)
        done_keys = set(zip(prev["dataset"], prev["backbone"], prev["cell"], prev["case_idx"]))
        rows = list(prev.to_dict("records"))
        if verbose:
            print(f"  [exp15] checkpoint : {len(done_keys)} lignes deja calculees, sautees.")
    header_written = os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0

    cells = [Cell.BOTH_FREE, Cell.CENTER_FIXED_WIDTH_FREE, Cell.CENTER_FREE_WIDTH_FIXED]
    for ds, extra in [(Dataset.MSL, dict(msl_channel=msl_channel)), (Dataset.NAB, {})]:
        for bb in [Backbone.RAW_CONTEXT, Backbone.TSFM]:
            # Chaque backbone choisit SES cas (stratifies par |m0|, comme exp4) : les
            # deux detecteurs ne signalent pas les memes points, un jeu commun
            # exclurait la plupart des cas TSFM.
            config0 = ExperimentConfig(dataset=ds, backbone=bb, cell=cells[0], decoder=Decoder.FLOW,
                                        n_cases=n_cases, **extra)
            setup = build_setup_generic(config0, verbose=verbose)
            case_idx = select_reference_cases(setup, config0, n_cases, verbose=verbose)
            for cell in cells:
                config = ExperimentConfig(dataset=ds, backbone=bb, cell=cell, decoder=Decoder.FLOW,
                                           n_cases=n_cases, **extra)
                if verbose:
                    print(f"\n=== exp15 {ds.value} backbone={bb.value} cell={cell.value} ===")
                for j, i in enumerate(case_idx, start=1):
                    if (ds.value, bb.value, cell.value, int(i)) in done_keys:
                        continue
                    row = run_one_case(setup, config, i, "exp15_tsfm_baseline")
                    rows.append(row)
                    pd.DataFrame([row]).to_csv(checkpoint_path, mode="a", index=False,
                                                header=not header_written)
                    header_written = True
                    if verbose and j % 10 == 0:
                        print(f"    {j}/{len(case_idx)}")
    return _df(rows)

# ======================================================================
# ORDRE D'EXECUTION OBLIGATOIRE (section S du protocole)
# ======================================================================

EXPERIMENT_ORDER = [
    ("exp0", exp0_pipeline_sanity),
    ("exp1", exp1_split_cp_control),
    ("exp2", exp2_channel_decomposition_msl),
    ("exp3", exp3_channel_decomposition_nab),
    ("exp4", exp4_generalization),
    ("exp6", exp6_center_shift_identity),
    ("exp7", exp7_epsilon_sensitivity),
    ("exp8", exp8_search_methods),
    ("exp9", exp9_cascade),
    ("exp10", exp10_sparsity_msl),
    ("exp11", exp11_sparsity_nab),
    # exp12 retiree (voir le bloc de commentaire ci-dessus, DistanceSpace.
    # LIKELIHOOD n'existe plus) -- numero laisse vacant.
    ("exp13", exp13_synthetic_ar),
    ("exp14", exp14_synthetic_regime),
    # exp5 (UCR) hors ordre par defaut : requiert UCR_SERIES_NAME, inseree
    # manuellement une fois l'archive disponible et la serie choisie.
]

EXTRA_EXPERIMENTS = {"exp5": exp5_ucr_control, "exp15": exp15_tsfm_baseline}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", choices=[k for k, _ in EXPERIMENT_ORDER] + list(EXTRA_EXPERIMENTS), default=None)
    parser.add_argument("--all", action="store_true",
                         help="lance tout dans l'ordre obligatoire -- PAS lance par defaut "
                              "(section S : ne pas lancer les 14 experiences immediatement)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list or (not args.exp and not args.all):
        print("Experiences disponibles (ordre d'execution obligatoire, section S) :\n")
        for key, fn in EXPERIMENT_ORDER:
            print(f"  {key:8s} {fn.__name__}")
        print("  exp5     exp5_ucr_control  (hors ordre -- requiert UCR_SERIES_NAME)")
        print("  exp15    exp15_tsfm_baseline  (hors ordre -- baseline IBM, telecharge Granite TTM)")
        return

    targets = EXPERIMENT_ORDER if args.all else [
        (args.exp, EXTRA_EXPERIMENTS[args.exp] if args.exp in EXTRA_EXPERIMENTS
         else dict(EXPERIMENT_ORDER)[args.exp])
    ]
    for key, fn in targets:
        print(f"\n{'=' * 70}\n{key} : {fn.__name__}\n{'=' * 70}")
        result = fn()
        dfs = result if isinstance(result, tuple) else (result,)
        for n, df in enumerate(dfs):
            suffix = "" if n == 0 else f"_{n}"
            out = f"results_{key}{suffix}.csv"
            df.to_csv(out, index=False)
            print(f"-> {out} ({len(df)} lignes)")


if __name__ == "__main__":
    main()
