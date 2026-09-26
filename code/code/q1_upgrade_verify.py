#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent artifact checks for q1_upgrade_validation.py outputs.

AI 辅助信息：OpenAI Codex（GPT-5 系列），OpenAI，2026-09-24。
本脚本不导入问题一评价函数；只用标准库、pandas/scipy/sklearn复算结果工件。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TUTORIAL = ROOT / "老师教程_F题" / "问题一"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    checks: dict[str, object] = {}
    manifest = pd.read_csv(TUTORIAL / "manifest.csv")
    checks["tutorial_23_files"] = len(manifest) == 23 and all((TUTORIAL / f"{i:02d}.png").is_file() for i in range(1, 24))
    checks["tutorial_hashes_match"] = all(sha256(TUTORIAL / row.file) == row.sha256 for row in manifest.itertuples())

    semantic = pd.read_csv(RESULTS / "q1_upgrade_semantic_manifest.csv")
    checks["semantic_positions_complete"] = semantic.semantic_position.tolist() == list(range(1, 26))
    checks["dsir_not_quality"] = set(semantic[semantic.source_field.str.startswith("dsir_")].role) == {"auxiliary"}
    checks["length_not_quality"] = set(semantic[semantic.source_field.isin([
        "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_mean_word_length"])].role) == {"structure"}

    pred = pd.read_csv(RESULTS / "q1_upgrade_predictions.csv.gz")
    bench = pd.read_csv(RESULTS / "q1_upgrade_model_benchmark.csv")
    maxdiff = 0.0
    for (model, dataset), part in pred.groupby(["model", "dataset"]):
        per_domain = []
        for _, d in part.groupby("domain"):
            y, p = d.observed_loss.to_numpy(float), d.predicted_loss.to_numpy(float)
            per_domain.append((mean_squared_error(y, p) ** 0.5, mean_absolute_error(y, p),
                               r2_score(y, p), spearmanr(y, p).statistic))
        calc = {
            "rmse_micro": mean_squared_error(part.observed_loss, part.predicted_loss) ** 0.5,
            "rmse_macro": np.mean([x[0] for x in per_domain]),
            "mae_macro": np.mean([x[1] for x in per_domain]),
            "r2_macro": np.mean([x[2] for x in per_domain]),
            "spearman_macro": np.nanmean([x[3] for x in per_domain]),
        }
        row = bench[(bench.model == model) & (bench.dataset == dataset)].iloc[0]
        maxdiff = max(maxdiff, *(abs(calc[k] - row[k]) for k in calc))
    checks["external_metrics_max_abs_diff"] = maxdiff
    checks["external_metrics_match"] = maxdiff < 1e-10

    summary = json.loads((RESULTS / "q1_upgrade_summary.json").read_text(encoding="utf-8"))
    decision = summary["mixture_candidate_decision"]
    cv = bench[bench.dataset.eq("A4_A5_oof")].sort_values(["rmse_micro", "spearman_macro"], ascending=[True, False])
    checks["cv_selection_reproduced"] = decision["cv_selected_model"] == cv.iloc[0].model
    checks["selected_model_confirmed_on_a6"] = bool(decision["promote_cv_selected_model"])
    checks["active_model_set_is_frozen"] = set(bench.model) == {
        "raw_share_ridge", "ilr_shared_alpha", "log_shift_ridge", "quadratic_ilr_ridge"}
    checks["benchmark_has_16_rows"] = len(bench) == 16
    alpha_audit = pd.read_csv(RESULTS / "q1_upgrade_domain_alpha.csv")
    checks["per_domain_alpha_removed"] = (len(alpha_audit) == 1 and
        alpha_audit.iloc[0].status == "deprecated_not_in_active_dual_model")
    checks["conflict_rate_bounded"] = 0 <= summary["conflict"]["a1_observed_rate_conditional_threshold"] <= 1

    calibration = pd.read_csv(RESULTS / "q1_upgrade_conflict_calibration_comparison.csv")
    checks["conditional_conflict_subset_of_global"] = bool(calibration.conditional_is_subset_of_global.all())
    checks["conflict_calibration_has_all_groups"] = len(calibration) == 9

    effects = pd.read_csv(RESULTS / "q1_dual_model_domain_effects.csv")
    wide = effects.pivot(index=["validation_domain", "increased_domain"], columns="model",
                         values="mean_delta_loss").reset_index()
    logv = wide.log_shift_ridge.to_numpy(float); quadv = wide.quadratic_ilr_ridge.to_numpy(float)
    sign = np.sign(logv) == np.sign(quadv)
    material = (np.abs(logv) >= .02) | (np.abs(quadv) >= .02)
    consistency = pd.read_csv(RESULTS / "q1_dual_model_consistency.csv")
    overall = consistency[consistency.scope.eq("overall")].iloc[0]
    checks["dual_effect_cell_count"] = len(wide) == 17 * 13
    checks["dual_overall_rank_match"] = abs(spearmanr(logv, quadv).statistic - overall.effect_rank_spearman) < 1e-12
    checks["dual_overall_sign_match"] = abs(sign.mean() - overall.sign_agreement_all) < 1e-12
    checks["dual_material_sign_match"] = abs(sign[material].mean() - overall.sign_agreement_material) < 1e-12

    interactions = pd.read_csv(RESULTS / "q1_quadratic_ilr_interactions.csv")
    checks["quadratic_interaction_cells_complete"] = len(interactions) == 136 * 13
    checks["quadratic_interaction_ci_ordered"] = bool(
        (interactions.bootstrap_mean_ci_low <= interactions.bootstrap_mean_ci_high).all())

    figure_contract = pd.read_csv(RESULTS / "q1_upgrade_figure_contract.csv")
    archived_figures = ROOT / "_tmp" / "freeze_archive" / "20260926" / "figures_root"
    checks["eleven_png_and_pdf_pairs"] = len(figure_contract) == 11 and all(
        (FIGURES / f"{stem}.{ext}").is_file() or (archived_figures / f"{stem}.{ext}").is_file()
        for stem in figure_contract.figure for ext in ("png", "pdf"))

    checks = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in checks.items()}
    passed = bool(all(v is True or (k == "external_metrics_max_abs_diff" and float(v) < 1e-10)
                      for k, v in checks.items()))
    output = {"passed": passed, "checks": checks}
    (RESULTS / "q1_upgrade_independent_verification.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
