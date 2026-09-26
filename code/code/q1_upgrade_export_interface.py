#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the reviewed Q1 dual-model interface for Q2 comparison.

AI 辅助信息：OpenAI Codex（GPT-5 系列），OpenAI，2026-09-24。
该接口标记为本分支工程冻结版；三人合并及人工表面效度复核前不得称最终定稿。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def main() -> None:
    summary = json.loads((R / "q1_upgrade_summary.json").read_text(encoding="utf-8"))
    bench = pd.read_csv(R / "q1_upgrade_model_benchmark.csv")
    policy = summary["mixture_candidate_decision"]["frozen_dual_model_policy"]
    selected = policy["primary_prediction"]
    structural = policy["structural_interaction"]
    datasets = ["A4_A5_oof", "A6_A7_1m", "A8_A9_60m", "A10_A11_1b"]
    primary_rows = bench[(bench.model == selected) & bench.dataset.isin(datasets)]
    structural_rows = bench[(bench.model == structural) & bench.dataset.isin(datasets)]
    consistency = pd.read_csv(R / "q1_dual_model_consistency.csv")
    overall_consistency = consistency[consistency.scope.eq("overall")].iloc[0].to_dict()
    interface = {
        "status": "q1_engineering_frozen_pending_three_member_merge_and_human_face_validity",
        "quality": {
            "representation": "four_facets",
            "facets": ["content_value", "language_quality", "cleanliness", "reasoning_professional"],
            "scalar_q": "provisional_definition_based_huber_mean_not_independent_ground_truth",
            "human_validation": "two-rater blinded descriptive face-validity check pending",
            "excluded": ["DSIR", "word_count", "num_sentences", "mean_word_length"],
            "q2_rule": "calibrate scalar Q with B6-B8 under non-circular validation",
        },
        "primary_prediction_model": {
            "model": selected,
            "formula": "L_h=b_h+sum_j beta_hj*log(p_j+epsilon)",
            "epsilon": 1e-4,
            "alpha": 1.0,
            "parameter_file": "results/q1_upgrade_selected_parameters.csv",
            "effect_file": "results/q1_upgrade_selected_domain_effects.csv",
            "metrics": primary_rows.to_dict("records"),
            "claim_scope": "source-internal prediction and ranking; substitution effects are noncausal model scenarios",
        },
        "structural_interaction_model": {
            "model": structural,
            "formula": "L_h=b_h+gamma_h^T z(p)+z(p)^T B_h z(p), where z is an ILR coordinate",
            "parameter_file": "results/q1_upgrade_quadratic_parameters.csv",
            "raw_domain_interaction_file": "results/q1_quadratic_ilr_interactions.csv",
            "effect_file": "results/q1_dual_model_domain_effects.csv",
            "metrics": structural_rows.to_dict("records"),
            "interpretation_rule": "ILR coefficients are basis-dependent; raw-domain combinations use closed scenario contrasts",
            "claim_scope": "structural sensitivity and noncausal interaction scenarios, not the primary point predictor",
        },
        "dual_model_consistency": {
            "file": "results/q1_dual_model_consistency.csv",
            "overall": overall_consistency,
            "rule": "disclose disagreement rather than force structural and predictive conclusions to coincide",
        },
        "transparent_baseline": "linear ILR-Ridge retained only as the compositional-geometry baseline",
        "removed_model": "per-domain-alpha ILR removed because prior improvement was below 0.1%",
        "transfer_to_q2": [
            "four facets and domain uncertainty",
            "source-internal mixture ranking/effects",
            "model parameters and applicable scale",
            "cross-scale failure evidence",
        ],
        "do_not_transfer": [
            "A-group absolute Loss as if directly compatible with B",
            "Q as a physical treatment variable",
            "10B/70B estimated Loss as real holdout",
            "substitution effects as causal or optimal allocation",
        ],
    }
    (R / "q1_upgrade_interface_to_q2.json").write_text(json.dumps(interface, ensure_ascii=False, indent=2), encoding="utf-8")
    print(R / "q1_upgrade_interface_to_q2.json")


if __name__ == "__main__":
    main()
