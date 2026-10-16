"""Diagnostic ponctuel pour la question levee sur exp8 : genetic et hybrid
donnent des stats identiques (6/40, meme distance/|dyhat|/|dw|, runtimes
quasi identiques). Ce script ne fait PAS partie du protocole ferme -- il
sert juste a compter, sur les memes 40 cas MSL que exp8, dans quelle
proportion search_hybrid_latent (search_variants.py) retombe sur le
resultat du GA plutot que de garder le raffinement par gradient, et pour
quelle raison (GA deja infeasible vs raffinement n'a pas aide).

Usage (depuis le dossier cfx_framework, meme environnement que protocol.py) :
    python diagnose_hybrid.py
"""
import protocol as P
import search_variants as sv
from config import ExperimentConfig, Dataset, Backbone, Cell, Decoder

config_ref = ExperimentConfig(dataset=Dataset.MSL, backbone=Backbone.RAW_CONTEXT,
                               cell=Cell.CENTER_FIXED_WIDTH_FREE, decoder=Decoder.FLOW,
                               msl_channel="T-1", n_cases=40)
setup = P.build_setup_generic(config_ref, verbose=False)
case_idx = P.select_reference_cases(setup, config_ref, 40, verbose=False)

counts = {"ga_infeasible": 0, "refinement_lost": 0, "refinement_won": 0, "excluded": 0}

for j, i in enumerate(case_idx, start=1):
    data_bundle = setup["data_bundle"]
    active_backbone = P.active_backbone_for_case(setup, config_ref, i)
    x_factual = P.torch.tensor(data_bundle.X_test[i], dtype=P.torch.float32, device=P.DEVICE)
    y_t = float(data_bundle.y_test[i])
    m0 = P.compute_margin(active_backbone, x_factual, P.torch.tensor(y_t, device=P.DEVICE))[0].item()
    want_negative = P.want_negative_for_scenario(config_ref.scenario)
    sign_ok = (m0 < 0) if want_negative else (m0 > 0)
    if not sign_ok:
        counts["excluded"] += 1
        continue

    print(f"--- cas {j}/40 (idx={i}) ---")
    ga_result, _ = sv.search_genetic_latent(active_backbone, setup["decoder"], x_factual, y_t,
                                              config_ref, m0)
    if not ga_result.feasible:
        counts["ga_infeasible"] += 1
        print(f"    GA infeasible (dist={ga_result.distance:.4f}) -- raffinement jamais tente")
        continue

    result = sv.search_hybrid_latent(active_backbone, setup["decoder"], x_factual, y_t,
                                      config_ref, m0, debug=True)
    if result.distance == ga_result.distance and result.feasible == ga_result.feasible:
        counts["refinement_lost"] += 1
    else:
        counts["refinement_won"] += 1

print("\n=== resume sur 40 cas MSL ===")
for k, v in counts.items():
    print(f"  {k}: {v}")
print("\nSi 'refinement_won' == 0, le raffinement par gradient ne sert jamais ici -- "
      "confirmant l'hypothese que son budget (4x80=320 pas) est insuffisant face a "
      "celui de la recherche principale (8x100=800 pas). Dans ce cas, 'hybrid' == "
      "'genetic' est attendu et PAS un bug de selection -- mais ca vaut la peine "
      "d'augmenter n_outer_refine/n_inner_refine dans search_hybrid_latent avant de "
      "publier une comparaison hybrid vs genetic.")
