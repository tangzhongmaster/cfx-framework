"""
Chargement des deux jeux de donnees, derriere une interface commune.

Chaque loader retourne un DataBundle : contextes/cibles/labels pour
train/calib/test, deja normalises, prets a etre consommes par n'importe
quel backbone (Section 5.1.1 du papier).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class DataBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    labels_test: np.ndarray   # 1 = anomalie labellisee (verite terrain), 0 = normal
    mean: float
    std: float


def load_mitbih(context_window: int = 10, local_dir: str = None) -> DataBundle:
    """MIT-BIH Arrhythmia Database (PhysioNet). Records 100/103 = train,
    101 = calibration, 207/208 = test (decalage inter-patients deliberre,
    Section 5.1.1).

    local_dir : dossier local contenant les fichiers .hea/.dat/.atr deja
    telecharges (ex: via `wfdb.dl_database('mitdb', 'chemin')` une seule
    fois). Si None, essaie d'abord './mitdb' localement, puis retombe sur
    le streaming distant PhysioNet (lent, et fragile aux erreurs serveur
    comme les 502 -- a n'utiliser qu'en dernier recours)."""
    import os
    import wfdb

    if local_dir is None:
        local_dir = "./mitdb" if os.path.isdir("./mitdb") else None

    def extract_rr(record_name):
        if local_dir is not None and os.path.exists(os.path.join(local_dir, record_name + ".hea")):
            record = wfdb.rdrecord(os.path.join(local_dir, record_name))
            ann = wfdb.rdann(os.path.join(local_dir, record_name), "atr")
        else:
            record = wfdb.rdrecord(record_name, pn_dir="mitdb")
            ann = wfdb.rdann(record_name, "atr", pn_dir="mitdb")
        fs = record.fs
        rpeaks = ann.sample
        rr = np.diff(rpeaks) / fs  # secondes
        symbols = ann.symbol[1:]   # un symbole par intervalle RR
        is_vpc = np.array([s == "V" for s in symbols], dtype=int)
        return rr, is_vpc

    def make_windows(rr, is_vpc, w):
        X, y, labels = [], [], []
        for i in range(w, len(rr)):
            X.append(rr[i - w:i])
            y.append(rr[i])
            labels.append(is_vpc[i])
        return np.array(X), np.array(y), np.array(labels)

    rr_100, vpc_100 = extract_rr("100")
    rr_103, vpc_103 = extract_rr("103")
    rr_101, vpc_101 = extract_rr("101")
    rr_207, vpc_207 = extract_rr("207")
    rr_208, vpc_208 = extract_rr("208")

    mean = np.mean(np.concatenate([rr_100, rr_103]))
    std = np.std(np.concatenate([rr_100, rr_103]))

    def norm(rr):
        return (rr - mean) / std

    X_train_a, y_train_a, _ = make_windows(norm(rr_100), vpc_100, context_window)
    X_train_b, y_train_b, _ = make_windows(norm(rr_103), vpc_103, context_window)
    X_train = np.concatenate([X_train_a, X_train_b])
    y_train = np.concatenate([y_train_a, y_train_b])

    X_calib, y_calib, _ = make_windows(norm(rr_101), vpc_101, context_window)

    X_test_a, y_test_a, lab_a = make_windows(norm(rr_207), vpc_207, context_window)
    X_test_b, y_test_b, lab_b = make_windows(norm(rr_208), vpc_208, context_window)
    X_test = np.concatenate([X_test_a, X_test_b])
    y_test = np.concatenate([y_test_a, y_test_b])
    labels_test = np.concatenate([lab_a, lab_b])

    return DataBundle(X_train, y_train, X_calib, y_calib, X_test, y_test,
                       labels_test, mean, std)


def load_nab(context_window: int = 50, nab_path: str = None) -> DataBundle:
    """NAB realAWSCloudwatch EC2 CPU utilization. Decoupage TEMPOREL
    (50/20/30 %), pas inter-entite (Section 5.1.1).

    nab_path : dossier racine du depot NAB (contient data/ et labels/). Si
    None, cherche automatiquement dans quelques emplacements usuels avant
    d'abandonner avec un message explicite plutot qu'un traceback pandas cru."""
    import pandas as pd
    import json
    import os

    rel = os.path.join("data", "realAWSCloudwatch", "ec2_cpu_utilization_24ae8d.csv")

    if nab_path is None:
        env_path = os.environ.get("NAB_PATH")
        candidates = [env_path] if env_path else []
        candidates += [
            ".", "./NAB", "../NAB", os.path.expanduser("~/NAB"),
            os.path.expanduser("~/cqr/NAB"), "./data/NAB",
        ]
        nab_path = next((c for c in candidates if c and os.path.exists(os.path.join(c, rel))), None)
        if nab_path is None:
            raise FileNotFoundError(
                "NAB introuvable. Cherche dans : " + ", ".join(candidates) + ".\n"
                "Soit clone-le : git clone https://github.com/numenta/NAB.git\n"
                "puis relance depuis un dossier qui le contient, soit passe le chemin "
                "explicitement : load_nab(nab_path='/chemin/vers/NAB') ou, via le framework, "
                "modifie context_window dans ExperimentConfig et appelle load_nab directement "
                "avec nab_path, ou fixe la variable d'environnement NAB_PATH."
            )

    csv_path = os.path.join(nab_path, "data", "realAWSCloudwatch",
                             "ec2_cpu_utilization_24ae8d.csv")
    labels_path = os.path.join(nab_path, "labels", "combined_windows.json")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"NAB_PATH='{nab_path}' ne contient pas {rel}. "
            f"Verifie le chemin ou clone https://github.com/numenta/NAB.git"
        )

    df = pd.read_csv(csv_path)
    values = df["value"].values.astype(float)
    timestamps = df["timestamp"].values

    with open(labels_path) as f:
        all_windows = json.load(f)
    windows = all_windows.get("realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv", [])

    is_anomaly = np.zeros(len(values), dtype=int)
    ts = pd.to_datetime(timestamps)
    for start, end in windows:
        mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
        is_anomaly[mask] = 1

    mean, std = values.mean(), values.std()
    values_n = (values - mean) / std

    def make_windows(vals, labs, w, stride=1):
        X, y, labels = [], [], []
        for i in range(w, len(vals), stride):
            X.append(vals[i - w:i])
            y.append(vals[i])
            labels.append(labs[i])
        return np.array(X), np.array(y), np.array(labels)

    X, y, labels = make_windows(values_n, is_anomaly, context_window)
    n = len(X)
    n_train = int(0.5 * n)
    n_calib = int(0.7 * n)

    X_train, y_train = X[:n_train][labels[:n_train] == 0], y[:n_train][labels[:n_train] == 0]
    X_calib, y_calib = X[n_train:n_calib][labels[n_train:n_calib] == 0], y[n_train:n_calib][labels[n_train:n_calib] == 0]
    X_test, y_test, labels_test = X[n_calib:], y[n_calib:], labels[n_calib:]

    return DataBundle(X_train, y_train, X_calib, y_calib, X_test, y_test,
                       labels_test, mean, std)


def load_msl(context_window: int = 50, channel: str = "T-1", msl_path: str = None,
             calib_fraction: float = 0.3) -> DataBundle:
    """NASA MSL (Mars Science Laboratory), benchmark telemanom (Hundman et
    al.). UN SEUL CANAL a la fois, mais DESORMAIS MULTIVARIE : decision
    scientifique explicite (pas deduite seul) -- H_t (l'historique que le
    contrefactuel modifie) garde TOUTES les colonnes du .npy du canal
    (n_timesteps, n_inputs), tandis que y_t (la cible couverte par
    l'intervalle, fixe par le contrefactuel) reste SCALAIRE = la 1ere
    colonne, convention telemanom elle-meme ("the telemetry values being
    predicted ... must be the first feature"). Consequence : H_t ->
    X_t(H_t) reste un vecteur plat generique pour tout le reste du
    framework (backbones.py, decoders.py, margin.py, criteria.py sont deja
    agnostiques a la dimension de X_t -- aucun de ces fichiers n'a besoin
    de changer), seule la taille de la fenetre aplatie passe de
    context_window a context_window * n_features. La marge reste
    m(H_t') = w/2 - |y_t - hat_y(H_t')| (Definition 1) avec y_t scalaire :
    PAS de marge/intervalle multivarie ici (option ecartee explicitement,
    pas celle choisie).

    Structure locale attendue (convention telemanom) :
        <msl_path>/data/train/<channel>.npy
        <msl_path>/data/test/<channel>.npy
        <msl_path>/labeled_anomalies.csv  (colonnes chan_id, anomaly_sequences --
            nom reel de la colonne dans le CSV telemanom/Kaggle, verifie sur le
            fichier brut du repo khundman/telemanom ; ne pas confondre avec
            "channel_id", qui n'existe pas dans ce fichier)

    Pas de calibration set separe fourni par le benchmark -- contrairement
    a MIT-BIH (patient dedie) ou NAB (tranche temporelle dediee), on
    decoupe donc le train.npy lui-meme en deux blocs TEMPORELS disjoints
    (train/calib), calib_fraction de la fin du train -- jamais les memes
    points des deux cotes, coherent avec la regle deja actee pour MIT-BIH
    (ne jamais calibrer sur les residus d'entrainement).

    msl_path : dossier racine (voir structure ci-dessus). Si None, cherche
    dans quelques emplacements usuels avant d'abandonner avec un message
    explicite (comme load_nab) -- obtention des donnees : soit
    `kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl`,
    soit les instructions du repo https://github.com/khundman/telemanom."""
    import os
    import ast
    import pandas as pd

    if msl_path is None:
        env_path = os.environ.get("MSL_PATH")
        candidates = [env_path] if env_path else []
        candidates += [".", "./telemanom", "../telemanom", os.path.expanduser("~/telemanom"),
                       "./data/MSL", "./nasa-anomaly-detection-dataset-smap-msl"]
        msl_path = next(
            (c for c in candidates if c and os.path.exists(os.path.join(c, "labeled_anomalies.csv"))),
            None)
        if msl_path is None:
            raise FileNotFoundError(
                "MSL introuvable. Cherche dans : " + ", ".join(candidates) + ".\n"
                "Obtention : soit `kaggle datasets download -d "
                "patrickfleith/nasa-anomaly-detection-dataset-smap-msl` (puis dezippe),\n"
                "soit suis les instructions de https://github.com/khundman/telemanom\n"
                "Structure attendue : <msl_path>/data/train/<channel>.npy, "
                "<msl_path>/data/test/<channel>.npy, <msl_path>/labeled_anomalies.csv\n"
                "Sinon, passe le chemin explicitement : load_msl(msl_path='/chemin/vers/telemanom') "
                "ou fixe la variable d'environnement MSL_PATH."
            )

    # La structure interne varie selon la source : le repo telemanom donne
    # <msl_path>/data/{train,test}/, alors que le zip Kaggle
    # (patrickfleith/...) ajoute un niveau en plus,
    # <msl_path>/data/data/{train,test}/ -- on essaie les deux plutot que
    # d'en imposer une et de faire deplacer des fichiers a la main.
    candidate_roots = [os.path.join(msl_path, "data"), os.path.join(msl_path, "data", "data")]
    data_root = next((r for r in candidate_roots
                       if os.path.exists(os.path.join(r, "train", f"{channel}.npy"))), None)
    if data_root is None:
        raise FileNotFoundError(
            f"MSL_PATH='{msl_path}' : canal '{channel}' introuvable sous aucune des structures "
            f"testees ({' ni '.join(os.path.join(r, 'train', channel + '.npy') for r in candidate_roots)}). "
            f"Verifie le nom du canal (voir chan_id dans labeled_anomalies.csv) et la structure "
            f"reelle du dossier (fais `find {msl_path} -maxdepth 4 -name '*.npy' | head`)."
        )
    train_fp = os.path.join(data_root, "train", f"{channel}.npy")
    test_fp = os.path.join(data_root, "test", f"{channel}.npy")
    labels_fp = os.path.join(msl_path, "labeled_anomalies.csv")
    for fp in (train_fp, test_fp, labels_fp):
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"MSL_PATH='{msl_path}' : fichier manquant {fp}. "
                f"Verifie le nom du canal (ex: 'T-1', 'C-1', ... -- voir chan_id "
                f"dans labeled_anomalies.csv) et la structure du dossier."
            )

    train_raw = np.load(train_fp).astype(float)   # (n_timesteps, n_features) -- TOUTES les colonnes
    test_raw = np.load(test_fp).astype(float)
    if train_raw.ndim == 1:
        train_raw = train_raw[:, None]
        test_raw = test_raw[:, None]

    anomalies_df = pd.read_csv(labels_fp)
    # Nom reel de la colonne dans le CSV telemanom/Kaggle : "chan_id" (verifie
    # sur le fichier brut du repo khundman/telemanom -- pas "channel_id").
    chan_col = "chan_id" if "chan_id" in anomalies_df.columns else "channel_id"
    row = anomalies_df[anomalies_df[chan_col] == channel]
    if len(row) == 0:
        raise ValueError(f"Canal '{channel}' absent de {labels_fp}. "
                          f"Canaux disponibles : {sorted(anomalies_df[chan_col].unique())}")
    sequences = ast.literal_eval(row.iloc[0]["anomaly_sequences"])  # ex: [[100,200],[500,520]]

    is_anomaly_test = np.zeros(len(test_raw), dtype=int)
    for start, end in sequences:
        is_anomaly_test[start:end + 1] = 1

    n_train_total = len(train_raw)
    n_calib = int(n_train_total * calib_fraction)
    n_train = n_train_total - n_calib
    train_part, calib_part = train_raw[:n_train], train_raw[n_train:]

    # Normalisation PAR FEATURE (chaque colonne de H_t a sa propre echelle),
    # calculee sur train uniquement, jamais sur calib/test -- meme regle
    # que partout ailleurs dans ce fichier. feat_mean/feat_std sont des
    # VECTEURS (n_features,), pas des scalaires, contrairement aux autres
    # loaders univaries de ce fichier.
    feat_mean = train_part.mean(axis=0)
    feat_std = train_part.std(axis=0)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)  # feature constante sur le train -- evite /0

    def norm(v):
        return (v - feat_mean) / feat_std

    def make_windows_mv(vals, labs, w):
        """vals: (n, n_features). Fenetre d'historique (w, n_features)
        APLATIE en un vecteur (w*n_features,) -- interface generique
        identique a tous les autres loaders (backbones/decodeur ne
        distinguent pas une fenetre multivariee aplatie d'un vecteur de
        features ordinaire). y = SCALAIRE = vals[i, 0] (1ere feature,
        convention telemanom) : c'est y_t, fixe par le contrefactuel, pas
        modifie par H_t -> H_t'."""
        X, Y, L = [], [], []
        for i in range(w, len(vals)):
            X.append(vals[i - w:i].reshape(-1))
            Y.append(vals[i, 0])
            L.append(labs[i])
        return np.array(X), np.array(Y), np.array(L)

    X_train, y_train, _ = make_windows_mv(norm(train_part), np.zeros(n_train, dtype=int), context_window)
    X_calib, y_calib, _ = make_windows_mv(norm(calib_part), np.zeros(n_calib, dtype=int), context_window)
    X_test, y_test, labels_test = make_windows_mv(norm(test_raw), is_anomaly_test, context_window)

    # DataBundle.mean/std restent SCALAIRES par contrat (dataclass), donc
    # on y range ceux de la cible (1ere feature) uniquement -- ils ne sont
    # de toute facon utilises nulle part en aval pour de la denormalisation
    # (verifie : seul backbones.py les reporte tels quels vers le bundle
    # SPCI, rien ne les relit pour un calcul reel).
    return DataBundle(X_train, y_train, X_calib, y_calib, X_test, y_test,
                       labels_test, float(feat_mean[0]), float(feat_std[0]))


def load_ucr(context_window: int = 50, series_name: str = None, ucr_path: str = None,
             calib_fraction: float = 0.3) -> DataBundle:
    """UCR Anomaly Archive (Wu & Keogh). Une serie a la fois, UNIVARIEE par
    construction (toutes les series de cette archive sont univariees --
    pas de decision multivariee a prendre ici, contrairement a MSL), meme
    formalisme scalaire que NAB (Definition 1, margin.py inchange).

    Role dans le protocole (Exp. 5) : verifier le comportement du
    contrefactuel quand l'anomalie est precedee d'une LONGUE dynamique
    normale -- pas une nouvelle methode, une validation externe.

    Convention de nommage de l'archive (Wu & Keogh 2021) :
        <id>_UCR_Anomaly_<nom>_<train_end>_<anomaly_start>_<anomaly_end>.txt
    un seul fichier texte, une valeur par ligne, serie ENTIERE (train+test
    concatenes) -- train_end/anomaly_start/anomaly_end sont des POSITIONS
    (1-indexees dans la doc d'origine, converties ici en indices 0-indexes)
    dans cette serie entiere, pas des longueurs independantes.

    ucr_path : dossier contenant les .txt de l'archive. Si None, cherche
    dans quelques emplacements usuels (voir candidates ci-dessous) avant
    d'abandonner avec un message explicite -- obtention des donnees :
    https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/ ou le
    depot https://github.com/TheDatumOrg/TSB-UAD (archive UCR incluse).

    series_name : sous-chaine (ou nom exact) du fichier a charger -- doit
    identifier EXACTEMENT un fichier, sinon erreur explicite listant les
    options. Pas de valeur par defaut : contrairement a MSL (canal "T-1"
    par defaut), choisir une serie UCR est une decision experimentale a
    assumer explicitement (Exp. 5 : "ne pas selectionner artificiellement
    les cas faciles"), jamais un choix implicite du loader."""
    import os

    if ucr_path is None:
        env_path = os.environ.get("UCR_PATH")
        candidates = [env_path] if env_path else []
        candidates += [".", "./UCR_Anomaly_FullData", "../UCR_Anomaly_FullData",
                       os.path.expanduser("~/UCR_Anomaly_FullData"), "./data/UCR"]
        ucr_path = next((c for c in candidates if c and os.path.isdir(c)), None)
        if ucr_path is None:
            raise FileNotFoundError(
                "UCR Anomaly Archive introuvable. Cherche dans : " + ", ".join(candidates) + ".\n"
                "Obtention : https://github.com/TheDatumOrg/TSB-UAD (archive UCR incluse) ou "
                "la page de l'archive originale (Wu & Keogh).\n"
                "Sinon, passe le chemin explicitement : load_ucr(ucr_path='/chemin/vers/UCR') "
                "ou fixe la variable d'environnement UCR_PATH."
            )

    files = sorted(f for f in os.listdir(ucr_path) if f.endswith(".txt"))
    if not files:
        raise FileNotFoundError(f"UCR_PATH='{ucr_path}' ne contient aucun fichier .txt.")

    if series_name is None:
        raise ValueError(
            "series_name requis (aucune valeur par defaut pour UCR -- voir docstring). "
            f"Fichiers disponibles ({len(files)} au total, 10 premiers) : "
            + ", ".join(files[:10])
        )

    matches = [f for f in files if series_name in f]
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Aucun fichier UCR ne contient '{series_name}' dans {ucr_path}. "
            f"Fichiers disponibles (10 premiers sur {len(files)}) : " + ", ".join(files[:10])
        )
    if len(matches) > 1:
        raise ValueError(
            f"'{series_name}' correspond a {len(matches)} fichiers, pas un seul : "
            + ", ".join(matches[:10]) + " -- precise davantage series_name."
        )
    filename = matches[0]

    stem = filename[:-4]  # retire ".txt"
    parts = stem.split("_")
    try:
        anomaly_end = int(parts[-1])
        anomaly_start = int(parts[-2])
        train_end = int(parts[-3])
    except (ValueError, IndexError):
        raise ValueError(
            f"Nom de fichier '{filename}' ne suit pas la convention UCR "
            f"<id>_UCR_Anomaly_<nom>_<train_end>_<anomaly_start>_<anomaly_end>.txt -- "
            f"impossible d'en extraire les 3 entiers finaux."
        )

    values = np.loadtxt(os.path.join(ucr_path, filename)).astype(float).reshape(-1)
    n = len(values)
    train_end = min(train_end, n - 1)

    is_anomaly_full = np.zeros(n, dtype=int)
    is_anomaly_full[max(0, anomaly_start - 1):min(n, anomaly_end)] = 1
    if is_anomaly_full[:train_end].any():
        raise ValueError(
            f"'{filename}' : la plage d'anomalie declaree ({anomaly_start}-{anomaly_end}) "
            f"chevauche le segment train declare (0-{train_end}) -- fichier incoherent avec "
            f"la convention UCR, verifie train_end dans le nom de fichier."
        )

    train_full = values[:train_end]
    test_full = values[train_end:]
    labels_test_full = is_anomaly_full[train_end:]

    n_train_total = len(train_full)
    n_calib = int(n_train_total * calib_fraction)
    n_train = n_train_total - n_calib
    train_part, calib_part = train_full[:n_train], train_full[n_train:]

    mean, std = train_part.mean(), train_part.std()
    if std < 1e-8:
        std = 1.0

    def norm(v):
        return (v - mean) / std

    def make_windows(vals, labs, w):
        X, Y, L = [], [], []
        for i in range(w, len(vals)):
            X.append(vals[i - w:i])
            Y.append(vals[i])
            L.append(labs[i])
        return np.array(X), np.array(Y), np.array(L)

    X_train, y_train, _ = make_windows(norm(train_part), np.zeros(n_train, dtype=int), context_window)
    X_calib, y_calib, _ = make_windows(norm(calib_part), np.zeros(n_calib, dtype=int), context_window)
    X_test, y_test, labels_test = make_windows(norm(test_full), labels_test_full, context_window)

    bundle = DataBundle(X_train, y_train, X_calib, y_calib, X_test, y_test,
                         labels_test, mean, std)
    # Metadonnees a documenter explicitement pour l'Exp. 5 (section I du
    # protocole : "longueur de l'historique avant anomalie, position de
    # l'anomalie") -- pas des champs du dataclass DataBundle (qui reste
    # commun a tous les loaders), attachees dynamiquement, lues par le
    # script d'experience, pas par le reste du framework.
    bundle.ucr_filename = filename
    bundle.ucr_train_end = train_end
    bundle.ucr_anomaly_start = anomaly_start
    bundle.ucr_anomaly_end = anomaly_end
    bundle.ucr_history_before_anomaly = anomaly_start - train_end
    return bundle


def _make_ar1_series(phi, sigma, n, seed, burn_in=200, y0=0.0):
    """AR(1) standard : y_t = phi*y_{t-1} + eps_t, eps_t ~ N(0, sigma^2).
    burn_in points generes et jetes pour atteindre le regime stationnaire
    avant de garder les n points retournes."""
    rng = np.random.default_rng(seed)
    total = n + burn_in
    y = np.zeros(total)
    y[0] = y0
    eps = rng.normal(0, sigma, size=total)
    for t in range(1, total):
        y[t] = phi * y[t - 1] + eps[t]
    return y[burn_in:]


def load_synthetic_ar(context_window: int = 20, phi: float = 0.7, sigma: float = 1.0,
                       anomaly_magnitude: float = 4.0, n_train: int = 1500,
                       n_calib: int = 800, n_test: int = 1500, seed: int = 42) -> DataBundle:
    """Processus AR(1) mono-regime totalement controle (RQ7, cas le plus
    simple). anomaly_magnitude : les points de test sont, une fois sur
    deux (indices pairs), perturbes par un choc additif de
    +/- anomaly_magnitude*sigma AVANT d'entrer dans la fenetre de contexte
    du point suivant -- donc l'anomalie observee a l'instant t est le
    resultat d'un historique recent deja perturbe, pas un outlier ponctuel
    isole, coherent avec le cadre H_t -> x_t(H_t) -> C_alpha(H_t) du
    papier (une perturbation ponctuelle non precedee d'un historique
    modifie ne testerait rien sur la recherche contrefactuelle historique).
    Pas de notion de 'regime requis' ici (voir load_synthetic_regime pour
    la version qui sert directement a D_CF vs D_regime) -- ce loader sert
    de cas de base controle pour RQ1 (faisabilite) avant de passer a la
    version a deux regimes."""
    rng = np.random.default_rng(seed)
    total_n = n_train + n_calib + n_test
    y = _make_ar1_series(phi, sigma, total_n, seed)

    # injection d'anomalies dans le seul segment de test, une fois sur
    # deux, par un choc additif qui se propage dans l'AR (pas juste au
    # point lui-meme) -- sinon la fenetre de contexte precedant le point
    # signale ne porterait aucune trace de la perturbation.
    test_start = n_train + n_calib
    is_anomaly = np.zeros(total_n, dtype=int)
    for t in range(test_start, total_n):
        if (t - test_start) % 2 == 0 and rng.random() < 0.5:
            shock = anomaly_magnitude * sigma * rng.choice([-1, 1])
            y[t] += shock
            is_anomaly[t] = 1
            # propage le choc (attenue par phi) dans les points suivants,
            # comme le ferait une vraie dynamique AR avec un historique
            # modifie, jusqu'a extinction numerique
            decay = shock
            for k in range(t + 1, min(t + context_window, total_n)):
                decay *= phi
                y[k] += decay

    mean, std = y[:n_train].mean(), y[:n_train].std()
    y_n = (y - mean) / std

    def make_windows(vals, labs, w):
        X, Y, L = [], [], []
        for i in range(w, len(vals)):
            X.append(vals[i - w:i])
            Y.append(vals[i])
            L.append(labs[i])
        return np.array(X), np.array(Y), np.array(L)

    X, Yv, L = make_windows(y_n, is_anomaly, context_window)
    n_tr = n_train - context_window
    n_ca = n_tr + n_calib

    X_train, y_train = X[:n_tr], Yv[:n_tr]
    X_calib, y_calib = X[n_tr:n_ca], Yv[n_tr:n_ca]
    X_test, y_test, labels_test = X[n_ca:], Yv[n_ca:], L[n_ca:]

    return DataBundle(X_train, y_train, X_calib, y_calib, X_test, y_test,
                       labels_test, mean, std)


def load_synthetic_regime(context_window: int = 20, phi_1: float = 0.7, phi_2: float = 0.7,
                           sigma: float = 1.0, n_train: int = 1500, n_calib: int = 800,
                           n_test: int = 1500, seed: int = 42) -> DataBundle:
    """Processus a changement de regime (RQ7, workhorse) :
        y_t = phi_1 y_{t-1} + eps_t   pour t avant le regime 2,
        y_t = phi_2 y_{t-1} + eps_t   une fois dans le regime 2.

    Le train et le calibrage sont tires ENTIEREMENT du regime 1 (phi_1) --
    le detecteur 'apprend' une seule dynamique, exactement comme un
    detecteur entraine sur des donnees normales dans le cadre du papier.
    Le test alterne des blocs regime 1 / regime 2, de sorte que les points
    du regime 2 sont naturellement juges anormaux par un backbone calibre
    sur le regime 1 (pas besoin d'injection manuelle d'anomalie) : leur
    historique recent EST le regime 2, differe structurellement du regime
    1 que le modele attend.

    D_regime := |phi_1 - phi_2| est le knob experimental : on construit
    plusieurs DataBundle avec des valeurs croissantes de |phi_1-phi_2| (a
    phi_1 fixe) et on verifie si la distance contrefactuelle mediane D_CF
    (obtenue par run_experiment sur chaque DataBundle) croit avec
    D_regime -- c'est directement le test de l'hypothese RQ7."""
    rng = np.random.default_rng(seed)
    block = 50  # longueur d'un bloc regime avant alternance, dans le test

    y_train = _make_ar1_series(phi_1, sigma, n_train, seed)
    y_calib = _make_ar1_series(phi_1, sigma, n_calib, seed + 1, y0=y_train[-1])

    # test : alternance de blocs regime_1/regime_2, en repartant a chaque
    # bloc de la derniere valeur du bloc precedent (continuite de la
    # trajectoire, pas de saut artificiel entre blocs)
    y_test_parts, regime_test_parts = [], []
    y_prev = y_calib[-1]
    n_blocks = int(np.ceil(n_test / block))
    for b in range(n_blocks):
        in_regime_2 = (b % 2 == 1)
        phi_b = phi_2 if in_regime_2 else phi_1
        seg = _make_ar1_series(phi_b, sigma, block, seed + 10 + b, y0=y_prev)
        y_test_parts.append(seg)
        regime_test_parts.append(np.full(block, int(in_regime_2)))
        y_prev = seg[-1]
    y_test = np.concatenate(y_test_parts)[:n_test]
    regime_test = np.concatenate(regime_test_parts)[:n_test]

    mean, std = y_train.mean(), y_train.std()

    def norm(v):
        return (v - mean) / std

    def make_windows(vals, labs, w):
        X, Y, L = [], [], []
        for i in range(w, len(vals)):
            X.append(vals[i - w:i])
            Y.append(vals[i])
            L.append(labs[i])
        return np.array(X), np.array(Y), np.array(L)

    # train/calib : label toujours 0 (regime 1 par construction)
    X_train, Y_train, _ = make_windows(norm(y_train), np.zeros(n_train, dtype=int), context_window)
    X_calib, Y_calib, _ = make_windows(norm(y_calib), np.zeros(n_calib, dtype=int), context_window)
    X_test, Y_test, labels_test = make_windows(norm(y_test), regime_test, context_window)

    return DataBundle(X_train, Y_train, X_calib, Y_calib, X_test, Y_test,
                       labels_test, mean, std)


def load_dataset(config):
    """Prend le config entier (pas seulement context_window) : les jeux
    synthetiques ont besoin de plus de parametres que les loaders
    reels."""
    from config import Dataset
    dataset, w = config.dataset, config.context_window
    if dataset == Dataset.MITBIH:
        return load_mitbih(w)
    elif dataset == Dataset.NAB:
        return load_nab(w)
    elif dataset == Dataset.MSL:
        return load_msl(context_window=w, channel=config.msl_channel, msl_path=config.msl_path)
    elif dataset == Dataset.UCR:
        return load_ucr(context_window=w, series_name=config.ucr_series_name, ucr_path=config.ucr_path)
    elif dataset == Dataset.SYNTHETIC_AR:
        return load_synthetic_ar(
            context_window=w, phi=config.synthetic_phi, sigma=config.synthetic_sigma,
            anomaly_magnitude=config.synthetic_anomaly_magnitude,
            n_train=config.synthetic_n_train, n_calib=config.synthetic_n_calib,
            n_test=config.synthetic_n_test, seed=config.seed)
    elif dataset == Dataset.SYNTHETIC_REGIME:
        return load_synthetic_regime(
            context_window=w, phi_1=config.synthetic_phi_1, phi_2=config.synthetic_phi_2,
            sigma=config.synthetic_sigma, n_train=config.synthetic_n_train,
            n_calib=config.synthetic_n_calib, n_test=config.synthetic_n_test, seed=config.seed)
    raise ValueError(f"dataset inconnu: {dataset}")
