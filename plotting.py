"""
Genere les figures du papier a partir des DataFrames produits par
run_experiment(). Ne duplique pas la logique de recherche -- prend en
entree ce que run_experiment() a deja calcule.

Fonctions :
  - plot_counterfactual_example : Figure 1-type, une trace contexte avant/apres
  - plot_identity_confirmation  : scatter |m0| vs deplacement du centre (Prop. 1)
  - plot_methods_comparison     : barres taux de succes / distance mediane par methode
  - plot_gallery                : grille de sous-figures contexte avant/apres, plusieurs cas
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_counterfactual_example(x_factual, x_star, y_t, backbone, out_path,
                                  title=None):
    """Une trace : contexte factuel vs contrefactuel, dimension par dimension."""
    dim = len(x_factual)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    steps = np.arange(1, dim + 1)
    ax.plot(steps, x_factual, "o-", color="#333333", label="Factuel", markersize=5)
    ax.plot(steps, x_star, "^--", color="#2c7fb8", label="Contrefactuel", markersize=5)
    ax.set_xlabel("Pas passé (contexte)")
    ax.set_ylabel("Valeur normalisée")
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gallery(cases, out_path, n_cols=1, max_cases=6):
    """cases : liste de dicts {x_factual, x_star, m0, feasible, n_dims_changed}.
    Si un cas contient 'x_stars' (liste), affiche TOUS les contrefactuels
    de ce cas superposes, pas seulement le premier -- utilise avec
    run_search_diverse / config.k_diverse > 1. Retro-compatible : un cas
    sans 'x_stars' n'affiche que 'x_star' comme avant.

    Si le cas contient y_t, hat_y_factual, sigma_factual, hat_y_stars,
    sigma_stars : affiche EN PLUS, au pas w+1, le vrai mecanisme qui decide
    le verdict -- la prediction ponctuelle, la bande [hat_y +- sigma], et
    le point y_t reellement observe, avant (factuel) et apres (chaque
    contrefactuel) -- pas seulement le changement de contexte qui le
    cause, mais le changement de verdict lui-meme. Retro-compatible : sans
    ces champs, seul le contexte est trace comme avant.

    Reproduit le format des galeries NAB/MIT-BIH du papier (une ligne par cas,
    triees par |m0| croissant)."""
    cases = sorted(cases, key=lambda c: abs(c["m0"]))[:max_cases]
    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.4 * n))
    if n == 1:
        axes = [axes]
    palette = ["#2c7fb8", "#e6550d", "#31a354", "#756bb1", "#e7298a"]
    has_interval_info = all(k in cases[0] for k in
                             ["y_t", "hat_y_factual", "sigma_factual"]) if cases else False

    for ax, c in zip(axes, cases):
        w = len(c["x_factual"])
        steps = np.arange(1, w + 1)
        ax.plot(steps, c["x_factual"], "o-", color="#333333", label="Factuel", markersize=4, zorder=10)

        x_stars = c.get("x_stars", [c["x_star"]])
        feasibles = c.get("feasibles_list", [c["feasible"]])
        n_dims_list = c.get("n_dims_changed_list", [c.get("n_dims_changed", "?")])

        for k, (x_star_k, feas_k) in enumerate(zip(x_stars, feasibles)):
            color = palette[k % len(palette)] if feas_k else "#999999"
            label = f"Contrefactuel {k+1}" if len(x_stars) > 1 else "Contrefactuel"
            ax.plot(steps, x_star_k, "^--", color=color, label=label, markersize=4, alpha=0.85)

        if has_interval_info and "hat_y_factual" in c:
            pred_step = w + 1
            y_t = c["y_t"]
            hat_y_f, sigma_f = c["hat_y_factual"], c["sigma_factual"]
            # intervalle factuel : point + bande, en noir, legerement a gauche
            ax.errorbar([pred_step - 0.08], [hat_y_f], yerr=[sigma_f], fmt="o",
                        color="#333333", capsize=4, markersize=5, elinewidth=2,
                        label="Intervalle factuel", zorder=9)
            # intervalle(s) contrefactuel(s) : meme code couleur que les courbes
            hat_y_stars = c.get("hat_y_stars", [])
            sigma_stars = c.get("sigma_stars", [])
            for k, (hy, sg, feas_k) in enumerate(zip(hat_y_stars, sigma_stars, feasibles)):
                color = palette[k % len(palette)] if feas_k else "#999999"
                offset = 0.08 + 0.12 * k
                ax.errorbar([pred_step + offset], [hy], yerr=[sg], fmt="^",
                            color=color, capsize=4, markersize=5, elinewidth=2, alpha=0.85, zorder=9)
            # le vrai y_t observe : fixe, jamais touche par la recherche
            ax.scatter([pred_step], [y_t], marker="*", s=140, color="gold",
                       edgecolor="black", linewidth=0.8, zorder=11, label="$y_t$ (reel)")
            ax.axvline(pred_step - 0.5, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
            ax.set_xticks(list(steps) + [pred_step])
            ax.set_xticklabels([str(s) for s in steps] + ["pred."])

        status = "faisable" if c["feasible"] else "infaisable"
        dims_str = ",".join(str(d) for d in n_dims_list) if len(x_stars) > 1 else str(n_dims_list[0])
        n_found = len(x_stars)
        title = f"m0={c['m0']:.3f} -> {status}  (dims changées={dims_str}"
        title += f", {n_found} contrefactuels)" if n_found > 1 else ")"
        ax.set_title(title, fontsize=9)
        if not has_interval_info:
            ax.set_xticks([])
        ax.legend(fontsize=6, loc="upper right", ncol=min(3, max(1, len(x_stars))))
    axes[-1].set_xlabel("Pas passé (contexte)" + ("  +  point de prédiction" if has_interval_info else ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_identity_confirmation(df, out_path, title="Identité margin/centre"):
    """df doit avoir les colonnes m0 et center_shift (a calculer en amont :
    |hat_y(x_star) - hat_y(x_factual)|). Reproduit le scatter + regression
    lineaire qui confirme la Proposition 1."""
    from scipy import stats
    sub = df.dropna(subset=["m0", "center_shift"])
    x, y = sub["m0"].abs().values, sub["center_shift"].values
    slope, intercept, r, p, se = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, alpha=0.6, color="#2c7fb8")
    xs = np.linspace(0, x.max(), 50)
    ax.plot(xs, slope * xs + intercept, "r--",
            label=f"pente={slope:.3f}, r={r:.3f}")
    ax.plot(xs, xs, "k:", alpha=0.4, label="identité (pente=1)")
    ax.set_xlabel("|m(X_t)|")
    ax.set_ylabel("Déplacement du centre observé")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return slope, r, p


def plot_methods_comparison(results_by_method: dict, out_path):
    """results_by_method : {nom_methode: DataFrame}. Barres taux de succes
    (gauche) et distance mediane parmi les succes (droite), reproduit
    Figure methods_comparison du papier."""
    names = list(results_by_method.keys())
    success_rates = [df["feasible"].mean() * 100 for df in results_by_method.values()]
    median_dist = [df[df["feasible"]]["distance"].median() if df["feasible"].any() else np.nan
                   for df in results_by_method.values()]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, success_rates, color="#2c7fb8")
    axes[0].set_ylabel("Taux de succès (%)")
    axes[0].set_xticklabels(names, rotation=45, ha="right")

    axes[1].bar(names, median_dist, color="#2c7fb8")
    axes[1].set_ylabel("Distance médiane (succès)")
    axes[1].set_yscale("log")
    axes[1].set_xticklabels(names, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_objective_tradeoff(objective_results: dict, out_path):
    """objective_results : {nom_objectif: DataFrame avec colonnes feasible,
    distance, log_likelihood}. Reproduit le point 7 du retour ChatGPT :
    feasibility / distance / likelihood ne bougent pas ensemble."""
    names = list(objective_results.keys())
    feas = [df["feasible"].mean() * 100 for df in objective_results.values()]
    dist = [df[df["feasible"]]["distance"].median() for df in objective_results.values()]
    ll = [df[df["feasible"]]["log_likelihood"].median() for df in objective_results.values()]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, vals, ylabel in zip(axes, [feas, dist, ll],
                                 ["Faisabilité (%)", "Distance médiane", "Log-vraisemblance médiane"]):
        ax.bar(names, vals, color="#2c7fb8")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(names, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
