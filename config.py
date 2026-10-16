"""
Configuration de l'espace complet des experiences, organisee en 7 NIVEAUX
EXPLICITES. Chaque niveau est un choix independant ; un ExperimentConfig en
fixe un pour chaque, et PROPOSED_METHOD (en bas de ce fichier) donne LA
config retenue pour la methode proposee -- pas noyee dans un tas d'options
au meme niveau.

======================================================================
NIVEAU 1 -- CASE DE LA GRILLE : quel canal (centre, largeur, les deux)
porte l'explication. Choix retenu : centre fixe / largeur libre, tenu
FIXE dans les experiences (pas balaye) sauf mention explicite contraire.

NIVEAU 2 -- BACKBONE : comment l'intervalle conforme C_alpha(X_t) est
construit. Choix retenu : contexte brut (backbone principal). SPCI et
foret sont des checks de generalite, pas des alternatives concurrentes.

NIVEAU 3 -- DECODEUR : comment on modelise p(x | y) pour chercher dedans.
Choix retenu et VERROUILLE : le flow de normalisation (RealNVP conditionnel),
seule option -- VAE / flow autoregressif / diffusion ne sont plus des
alternatives testables ici, retires du code (voir discussion ci-dessous
pour la justification du choix).

NIVEAU 4 -- ESPACE DE RECHERCHE : on ne cherche JAMAIS en contexte brut
comme approche generale. Choix retenu et VERROUILLE : la distance euclidienne
dans l'espace latent du flow, seule option -- la vraisemblance exacte comme
objectif de recherche n'est plus une alternative testable ici, retiree du
code.

NIVEAU 5 -- APPLICATION DE CONTRAINTE : lagrangien augmente directement,
pas de variante a poids fixe -- ce n'est pas principiel, on ne la garde
pas comme option de la methode proposee.

NIVEAU 6 -- (fusionne avec le niveau 5 : l'algorithme d'optimisation
*est* le lagrangien augmente, il n'y a pas de sous-choix supplementaire)

NIVEAU 7 -- PARCIMONIE : choix retenu et VERROUILLE : la fenetre contigue
(option (d) ci-dessous), seule option -- retrieval reel (Maalej) et
cardinalite libre en espace latent (avec ou sans penalite L1) ne sont plus
des alternatives testables ici, retirees du code (NONE reste pour le cas
"pas de contrainte de parcimonie du tout").
======================================================================

NIVEAU 3, pourquoi le flow (pour memoire, decision deja prise -- ce n'est
plus un benchmark a rejouer) : ce qu'on veut, en clair (pas de jargon),
c'est modeliser p(x | y), la distribution des contextes compatibles avec
l'observation y, pour pouvoir chercher dedans plutot que de verifier la
plausibilite apres coup. Le flow de normalisation donne p(x|y) EXACTEMENT,
et le passage entre le contexte et l'espace de recherche est exactement
reversible dans les deux sens, sans approximation -- l'outil le plus
direct qui donne cette garantie, sans la structure supplementaire d'un
flow autoregressif ni le renoncement a l'evaluation exacte d'un VAE ou
d'une diffusion. VAE / flow autoregressif / diffusion ne sont plus dans le
code (voir decoders.py) : ce n'etaient que des alternatives de benchmark,
jamais le choix retenu.
"""

from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# NIVEAU 1 -- Case de la grille
# ============================================================
class Cell(str, Enum):
    CENTER_FIXED_WIDTH_FREE = "center_fixed_width_free"   # retenue
    CENTER_FREE_WIDTH_FIXED = "center_free_width_fixed"   # Wachter classique
    BOTH_FREE = "both_free"                                 # ablation


# ============================================================
# NIVEAU 2 -- Backbone
# ============================================================
class Backbone(str, Enum):
    RAW_CONTEXT = "raw_context"    # backbone principal, retenu (= CQR)
    SPCI = "spci"                   # 2e geometrie : largeur fonction de
                                       # l'historique des residus
    SPLIT_CP = "split_cp"           # 3e geometrie : largeur CONSTANTE,
                                       # independante du contexte -- controle
                                       # theorique (dw/dx=0)
    FOREST = "forest"                # non-differentiable, sert uniquement
                                       # au diagnostic (Appendice)
    TSFM = "tsfm"                    # baseline IBM (W1-ACAS, arXiv 2604.20122) :
                                       # forecaster pre-entraine (Granite TTM,
                                       # zero-shot) + quantile pondere par
                                       # Wasserstein ; largeur constante en x,
                                       # centre = forecaster (voir tsfm_backbone.py)
    ADAPTIVE = "adaptive"            # 4e geometrie : memes hat_y(x), sigma(x)
                                       # que RAW_CONTEXT, mais q_hat n'est PLUS
                                       # calibre une fois pour toutes -- il est
                                       # mis a jour en ligne, pas a pas, par la
                                       # regle ACI (Gibbs & Candes 2021). Seul
                                       # backbone ou q_hat garde une trace du
                                       # temps qui passe ; necessaire (et
                                       # seulement necessaire) pour rejouer une
                                       # sequence H_t', H_t'+1, ... vers l'avant
                                       # et voir l'intervalle reagir en chaine
                                       # (mecanisme de reecriture causale, voir
                                       # backbones.py:replay_forward). Les 3
                                       # autres backbones n'ont rien a y gagner.


# ============================================================
# NIVEAU 3 -- Decodeur (modeliser p(x|y))
# ============================================================
class Decoder(str, Enum):
    FLOW = "flow"                  # seule option, verrouillee (voir discussion
                                       # ci-dessus) -- VAE/FLOW_AR/DIFFUSION/BOX
                                       # retires du code (anciennes alternatives
                                       # de benchmark, plus testees)


# ============================================================
# NIVEAU 4 -- Espace de recherche
# ============================================================
class DistanceSpace(str, Enum):
    LATENT = "latent"              # seule option, verrouillee : ||z - z0||
    # LIKELIHOOD retiree du code (ancienne alternative de benchmark, plus
    # testee). CONTEXT n'a jamais ete une option generale : la recherche en
    # contexte brut n'existe que dans la piste retrieval (niveau 7, retiree
    # elle aussi -- voir SparsityMethod).


# ============================================================
# NIVEAU 5 -- Application de contrainte
# ============================================================
class ConstraintMethod(str, Enum):
    AUGMENTED_LAGRANGIAN = "augmented_lagrangian"   # seule option retenue,
                                                       # principielle


# ============================================================
# NIVEAU 7 -- Parcimonie
# ============================================================
class SparsityMethod(str, Enum):
    NONE = "none"                                   # pas de contrainte de parcimonie
    WINDOW = "window"
    # Seule option de parcimonie, verrouillee -- DEUX axes, pas un (decision
    # explicite apres relecture de Sulem et al. 2022, "Diverse Counterfactual
    # Explanations for Anomaly Detection in Time Series", arXiv:2203.11103,
    # qui factorise deja M(w,t) = t x w pour les series multivariees) :
    #
    #   - axe TEMPS : fenetre SUFFIXE (i, t] ancree a DROITE (pas une
    #     position libre) -- x_factual = [y_{t-w},...,y_{t-1}] (ou un bloc
    #     de n_features par pas de temps pour MSL, voir data.py:load_msl),
    #     on libere les k DERNIERS pas de temps, masque DUR applique PENDANT
    #     l'optimisation (search.py:search_window/_masked_augmented_
    #     lagrangian). Largeur croissante jusqu'a faisabilite -> ell* = t-i*
    #     a une lecture causale directe ("jusqu'ou remonter dans
    #     l'historique"), ce qu'une fenetre a position arbitraire n'aurait
    #     pas.
    #
    #   - axe CANAL (datasets multivaries, MSL -- n_features>1 seulement) :
    #     PAS un second masque dur, une penalite de GROUPE L1 (channel_l1_
    #     lambda) sur l'axe canal, cherchee EN MEME TEMPS que le masque
    #     temps dans la meme descente -- pousse vers peu de canaux
    #     distincts touches, sans sous-ensemble pre-decide. Pour les
    #     datasets univaries (NAB/UCR/MITBIH/synthetic), n_features=1 :
    #     cet axe n'existe pas, channel_l1_lambda est ignoree.
    #
    # ATTENTION CAUSALITE (voir search.py:search_window pour le detail) :
    # liberer les derniers pas de temps est causalement CORRECT (rien
    # d'autre a recalculer) seulement pour un backbone dont hat_y(x)/
    # sigma(x) sont des fonctions PURES du contexte (RAW_CONTEXT, SPLIT_CP).
    # Pour un backbone a etat dependant de l'historique (SPCI, ADAPTIVE), il
    # faudrait rejouer la chaine prevision->residu->etat pas a pas depuis i
    # (voir backbones.py:replay_forward) -- pas implemente ici, donc WINDOW
    # ne doit etre utilisee qu'avec RAW_CONTEXT/SPLIT_CP pour l'instant.
    #
    # Retrouvailles pour memoire : retrieval reel (Maalej, cardinalite
    # input) et cardinalite libre en espace latent (avec ou sans penalite
    # L1 generale, pas specifique a l'axe canal) etaient d'anciennes
    # alternatives de benchmark -- retirees du code. search_latent_
    # cardinality reste dans search.py comme PRIMITIVE interne reutilisee
    # par la cascade de recherche (search_variants.py, Exp. 9), pas comme
    # option de ce niveau.


class Scenario(str, Enum):
    ANOMALY_TO_NORMAL = "anomaly_to_normal"
    NORMAL_TO_ANOMALY = "normal_to_anomaly"
    NARROW_STAY_NORMAL = "narrow_stay_normal"
    WIDEN_STAY_ANOMALY = "widen_stay_anomaly"


class Dataset(str, Enum):
    MITBIH = "mitbih"                        # HORS benchmark principal depuis
                                                # le protocole ferme (MSL l'a
                                                # remplace) -- garde pour
                                                # l'appendice/diagnostic
                                                # historique uniquement.
    NAB = "nab"
    MSL = "msl"                              # NASA telemetry, UN CANAL a la
                                                # fois (voir data.py). DECISION
                                                # EXPLICITE (pas deduite) :
                                                # H_t MULTIVARIE (toutes les
                                                # features du canal), y_t
                                                # reste SCALAIRE (1ere
                                                # feature, convention
                                                # telemanom) -- backbones.py/
                                                # margin.py/criteria.py
                                                # restent scalaires, seule la
                                                # dimension aplatie de X_t
                                                # change.
    UCR = "ucr"                              # UCR Anomaly Archive, UNE SERIE
                                                # a la fois (voir data.py) --
                                                # univarie par construction,
                                                # meme formalisme que NAB.
    SYNTHETIC_AR = "synthetic_ar"           # RQ7, bloc controle -- AR(1)
                                               # mono-regime, magnitude
                                               # d'anomalie controlee
    SYNTHETIC_REGIME = "synthetic_regime"   # RQ7, bloc controle -- deux
                                               # regimes phi_1/phi_2,
                                               # D_regime=|phi_1-phi_2| est
                                               # le knob experimental


@dataclass
class ExperimentConfig:
    dataset: Dataset = Dataset.MITBIH
    cell: Cell = Cell.CENTER_FIXED_WIDTH_FREE               # niveau 1
    backbone: Backbone = Backbone.RAW_CONTEXT             # niveau 2
    decoder: Decoder = Decoder.FLOW                          # niveau 3
    distance_space: DistanceSpace = DistanceSpace.LATENT     # niveau 4
    constraint_method: ConstraintMethod = ConstraintMethod.AUGMENTED_LAGRANGIAN  # niveau 5
    sparsity: SparsityMethod = SparsityMethod.NONE            # niveau 7
    scenario: Scenario = Scenario.ANOMALY_TO_NORMAL

    epsilon: float = 0.05
    lambda_penalty: float = 2000.0     # utilise seulement si un jour on
                                          # reintroduit une penalite fixe
                                          # (diagnostic historique)
    mu_sparsity: float = 0.1            # echelle du seuil de seuillage doux
                                          # (niveau 7, option c) -- pas un
                                          # poids de penalite, un vrai pas
                                          # proximal (seuil ~ mu_sparsity*lr)
    k_diverse: int = 1                   # nombre de contrefactuels distincts
                                          # a retourner par cas (1 = comportement
                                          # d'origine, un seul). >1 active la
                                          # recherche de diversite (Sulem et al.)
    min_diversity_dist: float = 0.5      # distance minimale entre deux
                                          # contrefactuels acceptes pour qu'ils
                                          # comptent comme genuinement distincts
    context_window: int = None
    n_cases: int = 40
    seed: int = 42

    # ------------------------------------------------------------
    # Backbone.ADAPTIVE uniquement : parametres de la regle ACI
    # (q_hat_{t+1} = q_hat_t + adaptive_gamma * (adaptive_alpha - err_t)).
    # adaptive_alpha doit rester coherent avec le alpha=0.1 deja code en dur
    # dans calibrate_split_cqr (quantile 0.9) -- meme definition de la
    # couverture cible, juste rendue mobile dans le temps.
    # ------------------------------------------------------------
    adaptive_gamma: float = 0.01
    adaptive_alpha: float = 0.1

    # ------------------------------------------------------------
    # SparsityMethod.WINDOW, datasets MULTIVARIES uniquement (MSL) :
    # force de la penalite de groupe L1 sur l'axe CANAL (voir search.py
    # :_masked_augmented_lagrangian/_channel_penalty et search_window).
    # Ignoree pour les datasets univaries (NAB/UCR/MITBIH/synthetic,
    # n_features=1 -- il n'y a pas d'axe canal a penaliser).
    # ------------------------------------------------------------
    channel_l1_lambda: float = 0.1

    # ------------------------------------------------------------
    # Parametres des jeux de donnees synthetiques (RQ7, bloc controle).
    # Ignores pour MITBIH/NAB. synthetic_phi_1/2 et synthetic_n sont les
    # seuls a varier reellement d'une etude a l'autre : D_regime :=
    # |phi_1 - phi_2| est LE knob experimental de l'etude de monotonie
    # D_CF vs D_regime (voir data.py:load_synthetic_regime).
    # ------------------------------------------------------------
    synthetic_phi: float = 0.7          # SYNTHETIC_AR uniquement
    synthetic_phi_1: float = 0.7        # SYNTHETIC_REGIME : regime factuel
    synthetic_phi_2: float = 0.7        # SYNTHETIC_REGIME : second regime
                                           # (fait varier D_regime)
    synthetic_sigma: float = 1.0
    synthetic_anomaly_magnitude: float = 4.0   # SYNTHETIC_AR : magnitude de
                                                  # l'anomalie injectee, en
                                                  # multiples de synthetic_sigma
    synthetic_n_train: int = 1500
    synthetic_n_calib: int = 800
    synthetic_n_test: int = 1500

    # MSL : un seul canal a la fois (voir data.py:load_msl) -- H_t
    # MULTIVARIE, y_t SCALAIRE (decision explicite, voir Dataset.MSL et
    # data.py). margin.py/criteria.py restent scalaires.
    msl_channel: str = "T-1"
    msl_path: str = None

    # UCR Anomaly Archive : une seule serie a la fois (voir data.py:load_ucr)
    # -- univarie par construction. Pas de valeur par defaut pour
    # ucr_series_name : contrairement a msl_channel, choisir une serie UCR
    # est une decision experimentale a assumer explicitement par appel
    # (Exp. 5 du protocole), jamais un choix implicite.
    ucr_series_name: str = None
    ucr_path: str = None

    def __post_init__(self):
        if self.context_window is None:
            if self.dataset == Dataset.MITBIH:
                self.context_window = 10
            elif self.dataset in (Dataset.SYNTHETIC_AR, Dataset.SYNTHETIC_REGIME):
                self.context_window = 20
            else:
                self.context_window = 50   # NAB, MSL, UCR

    def fixed_channel(self) -> str:
        return {
            Cell.CENTER_FIXED_WIDTH_FREE: "center",
            Cell.CENTER_FREE_WIDTH_FIXED: "width",
            Cell.BOTH_FREE: "none",
        }[self.cell]

    def label(self) -> str:
        base = (f"{self.dataset.value}_{self.backbone.value}_{self.cell.value}"
                f"_{self.decoder.value}_{self.distance_space.value}"
                f"_{self.sparsity.value}_{self.scenario.value}")
        if self.dataset == Dataset.SYNTHETIC_REGIME:
            base += f"_phi1{self.synthetic_phi_1}_phi2{self.synthetic_phi_2}"
        elif self.dataset == Dataset.MSL:
            base += f"_chan{self.msl_channel}"
        elif self.dataset == Dataset.UCR:
            base += f"_series{self.ucr_series_name}"
        return base


# ============================================================
# LA config retenue pour "la methode proposee" -- un seul objet, pas noye
# dans une liste d'options au meme niveau. C'est CELLE-LA qu'on pointe
# quand on dit "notre methode", tout le reste est diagnostic ou benchmark.
# Dataset MSL (pas MIT-BIH) : decision explicite du protocole ferme --
# MSL remplace MIT-BIH dans le benchmark principal. N'est plus utilise par
# le protocole experimental ferme (voir protocol.py, qui fixe chaque
# config experience par experience) ; garde ici comme reference ponctuelle
# pour un usage ad hoc hors protocole.
# ============================================================
PROPOSED_METHOD = ExperimentConfig(
    dataset=Dataset.MSL,
    backbone=Backbone.RAW_CONTEXT,
    cell=Cell.CENTER_FIXED_WIDTH_FREE,
    decoder=Decoder.FLOW,
    distance_space=DistanceSpace.LATENT,
    constraint_method=ConstraintMethod.AUGMENTED_LAGRANGIAN,
    sparsity=SparsityMethod.NONE,   # toujours le vrai trou : aucune des deux
                                       # options de parcimonie n'est cochee
                                       # par defaut, a choisir explicitement
    scenario=Scenario.ANOMALY_TO_NORMAL,
)
