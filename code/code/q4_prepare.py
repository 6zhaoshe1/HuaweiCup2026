# -*- coding: utf-8 -*-
"""Question 4 frozen cohort construction and C8 task reconstruction.

AI assistance: OpenAI Codex, 2026-09-25.

Inputs are official read-only attachment C files and the frozen Q4 preflight
selection.  Outputs are machine-readable modeling tables; no model fitting is
performed here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
C_DIR = (
    ROOT
    / "第二十三届中国研究生数学建模竞赛 - 中文题目"
    / "中文题目"
    / "F题"
    / "real_attachments"
    / "C_efficiency_evolution"
)
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SCORES = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
RAW_MAP = {
    "IFEval": "IFEval Raw",
    "BBH": "BBH Raw",
    "MATH Lvl 5": "MATH Lvl 5 Raw",
    "GPQA": "GPQA Raw",
    "MUSR": "MUSR Raw",
    "MMLU-PRO": "MMLU-PRO Raw",
}

# Frozen before fitting.  Strict means weights are on the hub, immutable SHA
# is present, record is not flagged, and the license is in this explicit
# reproducibility-oriented whitelist.  It is not inferred from non-nullness.
STRICT_LICENSES = [
    "apache-2.0",
    "mit",
    "gpl-3.0",
    "cc-by-sa-4.0",
    "cc-by-4.0",
    "wtfpl",
    "afl-3.0",
    "bsd-3-clause-clear",
    "osl-3.0",
    "creativeml-openrail-m",
    "bigscience-bloom-rail-1.0",
    "bigcode-openrail-m",
    "openrail",
    "bigscience-openrail-m",
]
WIDE_EXCLUDED_LICENSES = ["", "unknown", "other"]

TYPE_MAP = {
    "🟢 pretrained": "pretrained",
    "🟩 continuously pretrained": "pretrained",
    "💬 chat models (RLHF, DPO, IFT, ...)": "chat_finetuned",
    "🤝 base merges and moerges": "derived",
    "🔶 fine-tuned on domain-specific datasets": "derived",
    "🌸 multimodal": "other",
    "❓ other": "other",
}

GROUPS = {
    "BBH": "leaderboard_bbh",
    "GPQA": "leaderboard_gpqa",
    "MATH Lvl 5": "leaderboard_math_hard",
    "MUSR": "leaderboard_musr",
}


def norm_model(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = re.sub(r"^(pretrained=|hf:|hf-)", "", text)
    text = text.replace("__", "/").replace("\\", "/")
    return re.sub(r"[^a-z0-9]+", "", text)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def first_metric(metrics: dict) -> tuple[str, float]:
    candidates = [
        (str(k), float(v))
        for k, v in metrics.items()
        if k != "alias" and "stderr" not in str(k) and isinstance(v, (int, float))
    ]
    if not candidates:
        return "", np.nan
    return candidates[0]


def build_cohorts() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    c9_path = C_DIR / "data" / "train-00000-of-00001.parquet"
    raw = pd.read_parquet(c9_path, engine="pyarrow").copy()
    raw["model_name"] = raw["fullname"].astype(str)
    raw["model_key"] = raw["model_name"].map(norm_model)
    raw["submission_date"] = pd.to_datetime(raw["Submission Date"], errors="coerce")
    raw["params_B"] = pd.to_numeric(raw["#Params (B)"], errors="coerce")
    raw["stratum"] = raw["Type"].map(TYPE_MAP).fillna("other")
    raw["sha_clean"] = raw["Model sha"].fillna("").astype(str).str.strip()
    raw["base_model_clean"] = raw["Base Model"].fillna("").astype(str).str.strip()
    raw["license_clean"] = raw["Hub License"].fillna("").astype(str).str.strip()
    raw["strict_open"] = (
        raw["Available on the hub"].fillna(False).astype(bool)
        & ~raw["Flagged"].fillna(False).astype(bool)
        & raw["sha_clean"].ne("")
        & raw["license_clean"].isin(STRICT_LICENSES)
    )
    raw["wide_open"] = (
        raw["Available on the hub"].fillna(False).astype(bool)
        & ~raw["Flagged"].fillna(False).astype(bool)
        & raw["sha_clean"].ne("")
        & ~raw["license_clean"].isin(WIDE_EXCLUDED_LICENSES)
    )
    raw["entity_id"] = np.where(
        raw["sha_clean"].ne(""),
        "sha:" + raw["sha_clean"],
        "fallback:" + raw["model_key"] + "|" + raw["Submission Date"].fillna("").astype(str),
    )
    usable_base = ~raw["base_model_clean"].isin(["", "Removed"])
    raw["family_id"] = np.where(
        usable_base,
        "base:" + raw["base_model_clean"].map(norm_model),
        "self:" + raw["model_key"],
    )
    raw["core_type"] = raw["stratum"].isin(["pretrained", "chat_finetuned"])
    raw["valid_core_fields"] = (
        raw["submission_date"].notna()
        & raw["params_B"].gt(0)
        & raw[SCORES].notna().all(axis=1)
    )

    # Earliest leaderboard appearance for each immutable SHA/entity.  Ties are
    # deterministic by eval_name.  We do not select the highest score.
    ordered = raw.sort_values(["entity_id", "submission_date", "eval_name"], na_position="last")
    entity = ordered.groupby("entity_id", sort=False, as_index=False).head(1).copy()
    entity["dedup_rule"] = "earliest_submission_then_eval_name; never select by score"
    entity["score_equal"] = entity[SCORES].mean(axis=1)
    clipped = np.clip(entity["score_equal"].to_numpy(float), 0.05, 99.95)
    entity["score_logit"] = np.log(clipped / (100.0 - clipped))

    duplicates = (
        raw.groupby("entity_id", as_index=False)
        .agg(
            records=("entity_id", "size"),
            first_date=("submission_date", "min"),
            last_date=("submission_date", "max"),
            score_min=("Average ⬆️", "min"),
            score_max=("Average ⬆️", "max"),
            names=("model_name", "nunique"),
        )
    )
    duplicates = duplicates[duplicates["records"].gt(1)].copy()

    summary_rows = []
    for scope, mask in {
        "all_rows": pd.Series(True, index=raw.index),
        "strict_rows": raw["strict_open"],
        "wide_rows": raw["wide_open"],
    }.items():
        for stratum in ["all", "pretrained", "chat_finetuned", "derived", "other"]:
            m = mask if stratum == "all" else mask & raw["stratum"].eq(stratum)
            dates = raw.loc[m, "submission_date"]
            summary_rows.append(
                {
                    "scope": scope,
                    "unit": "row",
                    "stratum": stratum,
                    "n": int(m.sum()),
                    "date_min": str(dates.min().date()) if dates.notna().any() else "",
                    "date_max": str(dates.max().date()) if dates.notna().any() else "",
                    "unique_entities": int(raw.loc[m, "entity_id"].nunique()),
                    "unique_families": int(raw.loc[m, "family_id"].nunique()),
                }
            )
    for scope, mask in {
        "strict_entities": entity["strict_open"],
        "wide_entities": entity["wide_open"],
    }.items():
        for stratum in ["all", "pretrained", "chat_finetuned", "derived", "other"]:
            m = mask if stratum == "all" else mask & entity["stratum"].eq(stratum)
            dates = entity.loc[m, "submission_date"]
            summary_rows.append(
                {
                    "scope": scope,
                    "unit": "entity",
                    "stratum": stratum,
                    "n": int(m.sum()),
                    "date_min": str(dates.min().date()) if dates.notna().any() else "",
                    "date_max": str(dates.max().date()) if dates.notna().any() else "",
                    "unique_entities": int(entity.loc[m, "entity_id"].nunique()),
                    "unique_families": int(entity.loc[m, "family_id"].nunique()),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        RESULTS / "q4_cohort_summary.csv", index=False, encoding="utf-8-sig"
    )
    duplicates.to_csv(
        RESULTS / "q4_entity_duplicates.csv", index=False, encoding="utf-8-sig"
    )
    entity.to_csv(RESULTS / "q4_entity_cohort.csv.gz", index=False, compression="gzip")

    contract = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "competition": "2026 Huawei Cup 23rd China Graduate Mathematical Modeling Competition",
        "problem": "F-Q4 technical evolution and frontier forecast",
        "user_role": "competition team",
        "target_reader": "competition judges",
        "time_axis_main": "leaderboard submission date",
        "prediction_origin": str(entity["submission_date"].max().date()),
        "prediction_end_12m": str((entity["submission_date"].max() + pd.DateOffset(months=12)).date()),
        "prediction_end_24m": str((entity["submission_date"].max() + pd.DateOffset(months=24)).date()),
        "strict_open_rule": {
            "available_on_hub": True,
            "not_flagged": True,
            "nonblank_model_sha": True,
            "license_whitelist": STRICT_LICENSES,
        },
        "wide_open_rule": {
            "available_on_hub": True,
            "not_flagged": True,
            "nonblank_model_sha": True,
            "excluded_license_values": WIDE_EXCLUDED_LICENSES,
        },
        "entity_rule": "Model SHA; blank SHA falls back to normalized fullname plus submission date and is excluded from strict/wide",
        "dedup_rule": "earliest submission per entity; deterministic eval_name tie break; never choose by score",
        "main_types": ["pretrained", "chat_finetuned"],
        "derived_role": "separate extension only",
        "c8_damage_rule": "skip all four directories containing an invalid official JSON",
        "evidence_levels": {
            "C1_C8_C9": "real observational leaderboard records",
            "C3": "mixed source sensitivity only",
            "C4_compute": "observational and partly estimated metadata",
            "C5_C6_medium": "mixed validation-set bridge sensitivity",
            "C5_C6_high": "local same-model/same-validation bridge",
            "Q2_Q3": "frozen model outputs/conditional scenarios, not real future models",
        },
        "input_hashes": {
            "C9": sha256_file(C_DIR / "data" / "train-00000-of-00001.parquet"),
            "C8_selection": sha256_file(RESULTS / "q4_preflight_c8_latest_valid.csv"),
        },
    }
    (RESULTS / "q4_filter_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return raw, entity, contract


def prepare_c8(entity: pd.DataFrame) -> dict:
    selected = pd.read_csv(RESULTS / "q4_preflight_c8_latest_valid.csv")
    entity_by_key = (
        entity.sort_values(["model_key", "submission_date"])
        .drop_duplicates("model_key", keep="first")
        .set_index("model_key")
    )
    family_rows: list[dict] = []
    task_rows: list[dict] = []
    skipped = Counter()

    for _, row in selected.iterrows():
        path = ROOT / str(row["relative_path"]).replace("/", "\\")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skipped["unexpected_parse_failure"] += 1
            continue
        model_name = str(payload.get("model_name", row.get("model_name", "")))
        key = norm_model(model_name)
        if key not in entity_by_key.index:
            skipped["no_c9_match"] += 1
            continue
        meta = entity_by_key.loc[key]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]
        results = payload.get("results", {}) or {}
        n_samples = payload.get("n-samples", {}) or {}
        group_subtasks = payload.get("group_subtasks", {}) or {}

        reconstructed: dict[str, float] = {}
        direct_raw: dict[str, float] = {}
        for label, group in GROUPS.items():
            vals, weights = [], []
            for task in group_subtasks.get(group, []):
                if task not in results:
                    continue
                metric, value = first_metric(results[task])
                eff = n_samples.get(task, {})
                weight = eff.get("effective") if isinstance(eff, dict) else None
                if np.isfinite(value) and isinstance(weight, (int, float)) and weight > 0:
                    vals.append(value)
                    weights.append(float(weight))
                task_rows.append(
                    {
                        "model_name": model_name,
                        "model_key": key,
                        "entity_id": meta["entity_id"],
                        "family_id": meta["family_id"],
                        "submission_date": meta["submission_date"],
                        "params_B": meta["params_B"],
                        "stratum": meta["stratum"],
                        "strict_open": bool(meta["strict_open"]),
                        "wide_open": bool(meta["wide_open"]),
                        "task_family": label,
                        "task": task,
                        "metric": metric,
                        "score_raw": value,
                        "effective_n": weight,
                        "eval_signature": row["eval_signature"],
                    }
                )
            reconstructed[label] = float(np.average(vals, weights=weights)) if vals else np.nan
            _, direct_raw[label] = first_metric(results.get(group, {}))

        # IFEval official raw is the mean of prompt- and instruction-level
        # strict accuracy; MMLU-Pro uses accuracy directly.
        ifeval = results.get("leaderboard_ifeval", {}) or {}
        strict_metrics = [
            ifeval.get("prompt_level_strict_acc,none"),
            ifeval.get("inst_level_strict_acc,none"),
        ]
        reconstructed["IFEval"] = (
            float(np.mean(strict_metrics)) if all(isinstance(v, (int, float)) for v in strict_metrics) else np.nan
        )
        direct_raw["IFEval"] = reconstructed["IFEval"]
        _, mmlu = first_metric(results.get("leaderboard_mmlu_pro", {}))
        reconstructed["MMLU-PRO"] = mmlu
        direct_raw["MMLU-PRO"] = mmlu
        for label, task, value, metric in [
            ("IFEval", "leaderboard_ifeval", reconstructed["IFEval"], "mean_strict_accuracy"),
            ("MMLU-PRO", "leaderboard_mmlu_pro", mmlu, "acc,none"),
        ]:
            sample_info = n_samples.get(task, {})
            effective = sample_info.get("effective") if isinstance(sample_info, dict) else None
            task_rows.append(
                {
                    "model_name": model_name,
                    "model_key": key,
                    "entity_id": meta["entity_id"],
                    "family_id": meta["family_id"],
                    "submission_date": meta["submission_date"],
                    "params_B": meta["params_B"],
                    "stratum": meta["stratum"],
                    "strict_open": bool(meta["strict_open"]),
                    "wide_open": bool(meta["wide_open"]),
                    "task_family": label,
                    "task": task,
                    "metric": metric,
                    "score_raw": value,
                    "effective_n": effective,
                    "eval_signature": row["eval_signature"],
                }
            )

        family_row = {
            "model_name": model_name,
            "model_key": key,
            "entity_id": meta["entity_id"],
            "family_id": meta["family_id"],
            "submission_date": meta["submission_date"],
            "params_B": meta["params_B"],
            "stratum": meta["stratum"],
            "strict_open": bool(meta["strict_open"]),
            "wide_open": bool(meta["wide_open"]),
            "eval_signature": row["eval_signature"],
        }
        for label in SCORES:
            family_row[f"{label}_leaf_reconstructed_raw"] = reconstructed[label]
            family_row[f"{label}_json_group_raw"] = direct_raw[label]
            family_row[f"{label}_c9_raw"] = meta[RAW_MAP[label]]
            family_row[f"{label}_official_adjusted"] = meta[label]
        family_rows.append(family_row)

    families = pd.DataFrame(family_rows)
    tasks = pd.DataFrame(task_rows)
    families.to_csv(RESULTS / "q4_c8_family_reconstruction.csv.gz", index=False, compression="gzip")
    tasks.to_csv(RESULTS / "q4_c8_task_long.csv.gz", index=False, compression="gzip")

    metrics = []
    for label in SCORES:
        leaf = pd.to_numeric(families[f"{label}_leaf_reconstructed_raw"], errors="coerce")
        direct = pd.to_numeric(families[f"{label}_json_group_raw"], errors="coerce")
        c9raw = pd.to_numeric(families[f"{label}_c9_raw"], errors="coerce")
        for comparison, left, right in [
            ("leaf_vs_json_group", leaf, direct),
            ("json_group_vs_c9_raw", direct, c9raw),
            ("leaf_vs_c9_raw", leaf, c9raw),
        ]:
            mask = left.notna() & right.notna()
            diff = (left[mask] - right[mask]).abs()
            metrics.append(
                {
                    "task_family": label,
                    "comparison": comparison,
                    "n": int(mask.sum()),
                    "mae": float(diff.mean()) if len(diff) else np.nan,
                    "max_abs_error": float(diff.max()) if len(diff) else np.nan,
                    "spearman": float(left[mask].corr(right[mask], method="spearman")) if mask.sum() > 2 else np.nan,
                }
            )
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(
        RESULTS / "q4_c8_reconstruction_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "selected_model_directories": int(len(selected)),
        "matched_family_rows": int(len(families)),
        "task_long_rows": int(len(tasks)),
        "unique_tasks": int(tasks["task"].nunique()),
        "unique_task_families": int(tasks["task_family"].nunique()),
        "skipped": dict(skipped),
        "all_four_weighted_groups_max_abs_error": float(
            metrics_df.loc[
                metrics_df["comparison"].eq("leaf_vs_json_group")
                & metrics_df["task_family"].isin(list(GROUPS)),
                "max_abs_error",
            ].max()
        ),
    }
    return summary


def main() -> None:
    _, entity, contract = build_cohorts()
    c8_summary = prepare_c8(entity)
    output = {
        "status": "complete",
        "contract": contract,
        "c8": c8_summary,
    }
    (RESULTS / "q4_prepare_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
