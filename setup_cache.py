"""
Cache partage pour build_setup (run_experiment.py) / build_setup_generic
(protocol.py) -- evite de reconstruire le data_bundle/backbone et surtout
de RE-ENTRAINER le decodeur flow (1000 epoques full-batch, voir
decoders.py:train_decoder) a chaque experience qui retombe sur le meme
(dataset, backbone, decodeur, ...).

Pourquoi ca compte : au sein d'un seul `reproduce_paper.py --all` ou
`protocol.py --all`, un meme triplet (dataset=MSL, backbone=RAW_CONTEXT,
decoder=FLOW) est aujourd'hui reconstruit et reentraine depuis zero dans
exp1, le setup de reference d'exp2, exp6, exp7, exp8, exp9 et exp10 --
6-7 fois le meme entrainement de 1000 epoques pour un resultat identique
a chaque fois (build_setup(_generic) ne depend PAS de cell/epsilon/
sparsity/scenario, voir leur docstring d'origine).

Cle de cache : UNIQUEMENT ce qui influence data_bundle/backbone/decoder.
Ne doit JAMAIS inclure cell/epsilon/sparsity/scenario/constraint_method/
k_diverse/min_diversity_dist/n_cases/channel_l1_lambda -- ces niveaux ne
changent rien a ce que build_setup(_generic) construit, les melanger dans
la cle romprait le partage qu'on cherche a obtenir.

Deux niveaux de cache :
  - MEMOIRE (toujours actif) : un dict Python, valide pour la duree du
    process en cours. C'est lui qui resout l'essentiel du probleme
    ci-dessus, sans aucun risque (rien n'est persiste, rien ne peut
    devenir perime entre deux lancements).
  - DISQUE (off par defaut, voir set_disk_cache/CFX_DISK_CACHE) : persiste
    UNIQUEMENT les poids du decodeur entraine (torch.save/load_state_dict),
    pas le data_bundle ni l'objet backbone -- trop specifiques pour etre
    serialises sans risque de decalage silencieux. A activer seulement si
    tu sais que X_train/y_train sont stables entre deux lancements separes
    (deux jours differents, meme machine) : aucune verification au-dela de
    la cle ci-dessous n'est faite (pas de hash des donnees).
"""

import os
import hashlib
import torch

MEMORY_CACHE = {}
DISK_CACHE_DIR = "cache_decoders"

# Active par defaut si la variable d'environnement CFX_DISK_CACHE=1 est
# positionnee ; sinon utiliser set_disk_cache(True) explicitement.
_disk_cache_enabled = os.environ.get("CFX_DISK_CACHE", "0") == "1"


def set_disk_cache(enabled: bool):
    """Active/desactive le cache disque des poids du decodeur a partir
    de maintenant (n'affecte pas le cache memoire, toujours actif)."""
    global _disk_cache_enabled
    _disk_cache_enabled = bool(enabled)


def setup_cache_key(config):
    """Cle stable et hashable : seulement les champs qui influencent
    data_bundle/backbone/decoder. Deux ExperimentConfig qui ne different
    que par cell/epsilon/sparsity/scenario/n_cases/... partagent la meme
    cle, par design -- c'est le but.

    synthetic_anomaly_magnitude, synthetic_phi_2 et synthetic_n_test sont
    DELIBEREMENT absents de cette cle : verifie sur data.py (load_synthetic_
    ar, load_synthetic_regime) que X_train/y_train/X_calib/y_calib ne
    dependent JAMAIS de ces trois champs -- magnitude et n_test n'affectent
    que le segment de TEST de load_synthetic_ar, et phi_2 n'intervient que
    dans les blocs de regime 2 du TEST de load_synthetic_regime (train/calib
    sont tires entierement de phi_1, voir sa docstring). Les inclure dans la
    cle forcait exp13 (5 magnitudes) et exp14 (4 d_regime) a reentrainer le
    meme backbone+decodeur 5x/4x pour rien -- observe en pratique (5 lignes
    '[build_setup] synthetic_ar / raw_context / flow' au lieu d'une seule)."""
    key = [
        config.dataset.value, config.backbone.value, config.decoder.value,
        config.context_window, config.seed,
        config.msl_channel, config.msl_path,
        config.ucr_series_name, config.ucr_path,
        config.synthetic_phi, config.synthetic_phi_1,
        config.synthetic_sigma,
        config.synthetic_n_train, config.synthetic_n_calib,
    ]
    if config.backbone.value == "adaptive":
        key += [config.adaptive_gamma, config.adaptive_alpha]
    return tuple(key)


def get_memory_setup(key):
    return MEMORY_CACHE.get(key)


def store_memory_setup(key, setup):
    MEMORY_CACHE[key] = setup


def clear_memory_cache():
    """Utile en test/notebook pour forcer une reconstruction propre."""
    MEMORY_CACHE.clear()


def _decoder_disk_path(key):
    h = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:20]
    return os.path.join(DISK_CACHE_DIR, f"{h}.pt")


def load_decoder_weights(key, decoder, verbose=True):
    """Tente de charger des poids deja entraines depuis le disque, IN
    PLACE dans `decoder`. Retourne True si charge (l'appelant doit SAUTER
    train_decoder), False sinon (l'appelant doit entrainer puis appeler
    save_decoder_weights)."""
    if not _disk_cache_enabled:
        return False
    path = _decoder_disk_path(key)
    if not os.path.exists(path):
        return False
    try:
        decoder.load_state_dict(torch.load(path, map_location="cpu"))
        if verbose:
            print(f"  [cache disque] decodeur charge depuis {path} (entrainement evite)")
        return True
    except Exception as e:
        # Architecture incompatible (ex: dim a change) ou fichier corrompu
        # -- on retombe silencieusement sur un reentrainement plutot que
        # de planter.
        if verbose:
            print(f"  [cache disque] poids incompatibles ({type(e).__name__}: {e}), reentrainement")
        return False


def save_decoder_weights(key, decoder):
    if not _disk_cache_enabled:
        return
    os.makedirs(DISK_CACHE_DIR, exist_ok=True)
    torch.save(decoder.state_dict(), _decoder_disk_path(key))
