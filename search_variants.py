"""
Variantes de recherche pour les Experiences 8 (comparaison d'optimiseurs)
et 9 (cascade) du protocole experimental ferme.

DECISION EXPLICITE (pas deduite seule, validee par l'utilisateur) : les
noms "multi-start / annealing / genetic algorithm / Hybrid / cascade"
existaient dans l'ancien code pre-refonte (/mnt/user-data/uploads/outputs/
*.py : mitbih_full_experiment.py, diagnose_gradient_failure.py,
cardinality_constrained_search.py, counterfactual_search_ga.py,
real_cascade_and_diagnosis.py), mais operaient TOUJOURS directement en
CONTEXTE BRUT (gradient sur x, ou GA sur x), avec une penalite fixe ou
progressive -- un paradigme que le framework actuel a explicitement
abandonne au niveau 4 (config.py : "on ne cherche JAMAIS en contexte brut
comme approche generale").

Ce module ne porte donc PAS ces anciens scripts. Il reconstruit les memes
INTENTIONS (demarrage multiple, refroidissement progressif de la
penalite, recherche black-box, combinaison recherche-globale +
raffinement-local) comme des variantes de LA methode principale
(search_proposed : decodeur + espace latent + lagrangien augmente),
c'est-a-dire en perturbant/schedulant la recherche en z, jamais en
revenant chercher dans x. C'est du nouveau code, pas un "deja
implemente" -- a traiter comme tel dans toute description du protocole.

Les 4 methodes + la cascade :
    search_adam_single   -- le coeur augmente-lagrangien standard, un seul
                             demarrage depuis z0 = encode(x_factual).
                             Strictement equivalent a search_proposed sans
                             le post-traitement de parcimonie (sparsity).
    search_multistart     -- n_restarts demarrages z0 + bruit (plusieurs
                             echelles), garde le MEILLEUR succes (plus
                             petite distance) ; equivalent latent de
                             search_wide_restarts (contexte brut, ancien
                             code).
    search_annealing      -- un seul demarrage, mais un schedule de
                             penalite BEAUCOUP plus progressif (plus
                             d'etages exterieurs, gamma plus petit) que le
                             schedule standard -- expose comme variante
                             explicite, pas invente a partir de rien : le
                             mecanisme (rho *= gamma a chaque etage) est
                             deja celui de _augmented_lagrangian_core,
                             seuls les hyperparametres de schedule changent.
    search_genetic_latent -- algorithme genetique (pygad) sur z (genes
                             continus bornes autour de z0), boite noire
                             (pas de gradient) -- equivalent latent de
                             search_counterfactual_ga (contexte brut,
                             ancien code).
    search_hybrid_latent  -- GA latent d'abord (trouve une region
                             faisable), puis raffinement par gradient
                             (lagrangien augmente) depuis cette solution,
                             pas depuis z0 -- equivalent latent de
                             search_counterfactual_hybrid (ancien code).
    run_cascade           -- Hybrid -> Multi-start -> Annealing ->
                             Fenetre suffixe (i, t] (search_window,
                             ancree a droite : on remonte i en arriere
                             depuis t jusqu'a trouver un contrefactuel
                             faisable). CHAQUE etage n'etant tente que si
                             le precedent a echoue (cout de calcul reel,
                             pas la somme de tout lancer independamment).
                             Le dernier etage etait auparavant la
                             cardinalite latente (coordonnees arbitraires,
                             sans ordre temporel, O(dim) optimisations par
                             tour) : remplace par la fenetre, qui est la
                             parcimonie retenue dans le papier et coute au
                             plus n_timesteps optimisations.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import pygad

from margin import compute_margin, fixed_channel_value, channel_violation, \
    target_violation, scenario_targets
from search import ResultRow, _augmented_lagrangian_core, _evaluate, search_window
from device import DEVICE


def search_adam_single(backbone, decoder, x_factual, y_t, config, m0,
                        n_outer=8, n_inner=100, rho0=10.0, gamma=3.0):
    """Baseline de comparaison (Exp. 8) : UN SEUL demarrage, schedule
    standard -- litteralement le coeur de search_proposed, expose sous un
    nom explicite pour la table de comparaison d'optimiseurs plutot que
    noye dans le chemin "methode principale"."""
    x_factual = x_factual.to(DEVICE)
    z0 = decoder.encode(x_factual, y_t).detach()
    z_star, x, m_val, feasible = _augmented_lagrangian_core(
        backbone, decoder, x_factual, y_t, config, m0, z0, n_outer, n_inner, rho0, gamma)
    return ResultRow(x.cpu().numpy(), feasible, float(torch.norm(x - x_factual).item()),
                      int((torch.abs(x - x_factual) > 1e-6).sum().item()), m_val)


def search_multistart(backbone, decoder, x_factual, y_t, config, m0,
                       n_restarts=8, sigmas=(0.0, 0.3, 1.0, 2.0, 3.0, 5.0),
                       n_outer=6, n_inner=80, rho0=10.0, gamma=3.0, seed=None):
    """Equivalent latent de search_wide_restarts (ancien code, contexte
    brut) : n_restarts demarrages z0 + bruit gaussien a plusieurs echelles
    (sigma=0 = demarrage standard inclus), garde le succes de plus petite
    distance. Pas de raffinement croise entre restarts -- chaque restart
    est un run independant du meme coeur augmente-lagrangien."""
    x_factual = x_factual.to(DEVICE)
    rng = np.random.default_rng(seed if seed is not None else config.seed)
    z0 = decoder.encode(x_factual, y_t).detach()
    best = None
    for r in range(n_restarts):
        sigma = 0.0 if r == 0 else float(rng.choice(sigmas))
        noise = torch.tensor(rng.normal(0, sigma, size=z0.shape), dtype=torch.float32, device=DEVICE)
        z_init = z0 + noise
        z_star, x, m_val, feasible = _augmented_lagrangian_core(
            backbone, decoder, x_factual, y_t, config, m0, z_init, n_outer, n_inner, rho0, gamma)
        if feasible:
            dist = float(torch.norm(x - x_factual).item())
            if best is None or dist < best[0]:
                best = (dist, x, m_val)
    if best is None:
        # aucun restart faisable : retourne le dernier essai (non faisable),
        # pour ne jamais cacher un echec total derriere un retour vide.
        return ResultRow(x.cpu().numpy(), False, float(torch.norm(x - x_factual).item()),
                          int((torch.abs(x - x_factual) > 1e-6).sum().item()), m_val)
    dist, x, m_val = best
    return ResultRow(x.cpu().numpy(), True, dist,
                      int((torch.abs(x - x_factual) > 1e-6).sum().item()), m_val)


def search_annealing(backbone, decoder, x_factual, y_t, config, m0,
                      n_outer=20, n_inner=60, rho0=1.0, gamma=1.5):
    """Equivalent latent de search_annealed_penalty (ancien code, contexte
    brut) : UN SEUL demarrage depuis z0, mais un schedule de penalite
    beaucoup plus progressif que le standard (rho0 plus petit, gamma plus
    petit, plus d'etages exterieurs) -- meme mecanisme que
    _augmented_lagrangian_core (rho *= gamma a chaque etage exterieur),
    seuls les hyperparametres de schedule distinguent cette variante."""
    x_factual = x_factual.to(DEVICE)
    z0 = decoder.encode(x_factual, y_t).detach()
    z_star, x, m_val, feasible = _augmented_lagrangian_core(
        backbone, decoder, x_factual, y_t, config, m0, z0, n_outer, n_inner, rho0, gamma)
    return ResultRow(x.cpu().numpy(), feasible, float(torch.norm(x - x_factual).item()),
                      int((torch.abs(x - x_factual) > 1e-6).sum().item()), m_val)


def _ga_fitness_latent(backbone, decoder, x_factual, y_t, config, m0, z0):
    """Fabrique la fonction de fitness pygad pour une recherche GA en z --
    boite noire (pas de gradient utilise), penalise violation de cible +
    violation de canal + distance, comme _ga_fitness en contexte brut dans
    l'ancien counterfactual_search_ga.py, mais evaluee apres decodage."""
    x_factual = x_factual.to(DEVICE)
    m_target, direction = scenario_targets(m0, config.scenario)
    x0_value = fixed_channel_value(backbone, x_factual, config.cell)
    if x0_value is not None:
        x0_value = x0_value.detach()
    dim = z0.shape[0]

    def fitness_func(ga_instance, solution, idx):
        # 'solution' vient toujours de pygad sous forme de numpy array CPU,
        # quel que soit DEVICE -- le ramener explicitement sur DEVICE avant
        # de le passer au decodeur/backbone (potentiellement sur GPU).
        z = torch.tensor(solution, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            x, m, g1, g2 = _evaluate(backbone, decoder, z, y_t, x0_value,
                                      config.cell, config.epsilon, m_target, direction)
            dist = torch.norm(x - x_factual).item()
        return -(config.lambda_penalty * (g1.item() ** 2 + g2.item() ** 2) + dist)

    return fitness_func, dim


def search_genetic_latent(backbone, decoder, x_factual, y_t, config, m0,
                           search_radius=3.0, num_generations=150, sol_per_pop=24):
    """Equivalent latent de search_counterfactual_ga (ancien code, contexte
    brut) : pygad sur z, genes continus bornes [z0-r, z0+r] (r=search_radius,
    coherent avec le prior N(0,1) du flow -- z0 est generalement proche de
    0, +/-3 couvre la quasi-totalite de la masse). Boite noire : aucun
    gradient utilise pendant la recherche GA elle-meme."""
    x_factual = x_factual.to(DEVICE)
    # pygad travaille en numpy pur (CPU) -- z0/gene_space doivent donc
    # transiter par le CPU ici, independamment de DEVICE.
    z0 = decoder.encode(x_factual, y_t).detach().cpu().numpy()
    fitness_func, dim = _ga_fitness_latent(backbone, decoder, x_factual, y_t, config, m0,
                                            torch.tensor(z0, dtype=torch.float32, device=DEVICE))
    gene_space = [{"low": float(v - search_radius), "high": float(v + search_radius)} for v in z0]

    ga = pygad.GA(num_generations=num_generations, num_parents_mating=8, sol_per_pop=sol_per_pop,
                  num_genes=dim, gene_space=gene_space, fitness_func=fitness_func,
                  mutation_probability=0.2, crossover_type="uniform", keep_elitism=2,
                  stop_criteria="saturate_20", random_seed=config.seed, suppress_warnings=True)
    ga.run()
    best_solution, _, _ = ga.best_solution()
    z_star = torch.tensor(best_solution, dtype=torch.float32, device=DEVICE)

    m_target, direction = scenario_targets(m0, config.scenario)
    x0_value = fixed_channel_value(backbone, x_factual, config.cell)
    if x0_value is not None:
        x0_value = x0_value.detach()
    with torch.no_grad():
        x, m, g1, g2 = _evaluate(backbone, decoder, z_star, y_t, x0_value,
                                  config.cell, config.epsilon, m_target, direction)
        feasible = g1.item() < 1e-4 and g2.item() < 1e-4

    return ResultRow(x.cpu().numpy(), feasible, float(torch.norm(x - x_factual).item()),
                      int((torch.abs(x - x_factual) > 1e-6).sum().item()), m.item()), z_star


def search_hybrid_latent(backbone, decoder, x_factual, y_t, config, m0,
                          ga_generations=100, n_outer_refine=4, n_inner_refine=80,
                          debug=False):
    """Equivalent latent de search_counterfactual_hybrid (ancien code,
    contexte brut) : GA latent d'abord (trouve une region faisable, budget
    reduit), PUIS raffinement par gradient (lagrangien augmente) depuis la
    solution GA -- pas depuis z0 -- pour la resserrer. Le raffinement
    garde la solution GA si le gradient n'ameliore pas la distance (meme
    regle que l'ancien hybride, cf. README : "le hybride garde toujours la
    solution AG si le raffinement n'aide pas").

    debug=True : imprime pourquoi chaque appel retombe (ou pas) sur le GA --
    ajoute pour diagnostiquer le fait que exp8 donne des stats identiques
    pour genetic et hybrid (6/40, meme distance/|dyhat|/|dw|, meme runtime a
    0.2s pres) : soit le GA echoue systematiquement avant meme d'essayer le
    raffinement, soit le raffinement (4x80=320 pas) n'a jamais un budget
    suffisant pour ameliorer une solution GA deja feasible (8x100=800 pas
    pour la recherche principale, a titre de comparaison)."""
    x_factual = x_factual.to(DEVICE)
    ga_result, z_ga = search_genetic_latent(backbone, decoder, x_factual, y_t, config, m0,
                                             num_generations=ga_generations)
    if not ga_result.feasible:
        if debug:
            print(f"      [hybrid-debug] GA infeasible -- raffinement jamais tente, "
                  f"retour direct de ga_result (dist={ga_result.distance:.4f})")
        return ga_result

    z_star, x, m_val, feasible = _augmented_lagrangian_core(
        backbone, decoder, x_factual, y_t, config, m0, z_ga,
        n_outer_refine, n_inner_refine, rho0=10.0, gamma=3.0)
    dist_refined = float(torch.norm(x - x_factual).item())

    if debug:
        print(f"      [hybrid-debug] GA feasible dist={ga_result.distance:.4f} | "
              f"raffinement feasible={feasible} dist={dist_refined:.4f} | "
              f"garde={'raffinement' if (feasible and dist_refined < ga_result.distance) else 'GA'}")

    if feasible and dist_refined < ga_result.distance:
        return ResultRow(x.cpu().numpy(), True, dist_refined,
                          int((torch.abs(x - x_factual) > 1e-6).sum().item()), m_val)
    return ga_result  # le raffinement n'a pas aide (ou a casse la faisabilite) -- garde le GA


@dataclass
class CascadeResult:
    result: "ResultRow"
    stage: str
    n_stages_tried: int
    total_runtime: float


def run_cascade(backbone, decoder, x_factual, y_t, config, m0):
    """Experience 9 : cascade sequentielle Hybrid -> Multi-start ->
    Annealing -> Fenetre suffixe (i, t], chaque etage tente SEULEMENT si le
    precedent a echoue -- mesure le vrai cout de calcul de la cascade,
    pas la somme de tout lancer independamment.

    Dernier etage = search_window (fenetre ancree a droite, ell* = t - i*) :
    remplace search_latent_cardinality (coordonnees arbitraires, hors
    recit temporel du papier, et O(dim) optimisations par tour : c'est elle
    qui bloquait l'exp. 9 sur certains cas). NB : n_dims_changed du resultat
    de cet etage est la LARGEUR de la fenetre en pas de temps, pas un nombre
    de dimensions comme aux etages 1-3 (ResultRow.n_dims_changed y compte
    les coordonnees differentes de x_factual)."""
    t0 = time.perf_counter()

    result = search_hybrid_latent(backbone, decoder, x_factual, y_t, config, m0)
    if result.feasible:
        return CascadeResult(result, "Hybride (GA latent + raffinement)", 1,
                              time.perf_counter() - t0)

    result = search_multistart(backbone, decoder, x_factual, y_t, config, m0)
    if result.feasible:
        return CascadeResult(result, "Multi-start latent", 2, time.perf_counter() - t0)

    result = search_annealing(backbone, decoder, x_factual, y_t, config, m0)
    if result.feasible:
        return CascadeResult(result, "Annealing latent", 3, time.perf_counter() - t0)

    result = search_window(backbone, decoder, x_factual, y_t, config, m0)
    if result.feasible:
        return CascadeResult(result, "Fenetre suffixe (i, t]", 4,
                              time.perf_counter() - t0)

    return CascadeResult(result, "Echec total", 4, time.perf_counter() - t0)


SEARCH_METHODS = {
    "adam_single": search_adam_single,
    "multistart": search_multistart,
    "annealing": search_annealing,
    "genetic": lambda *a, **kw: search_genetic_latent(*a, **kw)[0],  # jette z_star, garde l'interface commune
    "hybrid": search_hybrid_latent,
}
