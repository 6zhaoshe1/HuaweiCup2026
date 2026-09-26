# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；运行时模型版本未独立核验；开发机构：OpenAI；日期：2026-09-23。
"""F题队伍确认前审计：22个质量字段结构、范围与A/B Loss口径证据。

安全边界：只读取完整 F题.zip 中的数据附件，不读取或解析受污染的《数据说明.pdf》。
"""

from __future__ import annotations

import csv
import io
import json
import lzma
import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
F_ZIP = ROOT / "第二十三届中国研究生数学建模竞赛 - 中文题目" / "中文题目" / "F题.zip"
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "code" / "outputs"

QUALITY_FIELDS = [
    "fineweb_edu",
    "fluency_en",
    "modernbert_cleanliness",
    "modernbert_readability",
    "modernbert_reasoning",
    "modernbert_professionalism",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "qurater",
    "ad_en",
    "rps_doc_word_count",
    "rps_doc_num_sentences",
    "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction",
    "rps_doc_mean_word_length",
]

JSONL_PATHS = {
    "A1": "real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz",
    "A2": "real_attachments/A_data_value/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz",
    "A3": "real_attachments/A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz",
}

LOSS_PATHS = {
    "A5": "real_attachments/A_data_value/regmix_tables/train_pile_loss_1m.csv",
    "A7": "real_attachments/A_data_value/regmix_tables/test_pile_loss_1m.csv",
    "A9": "real_attachments/A_data_value/regmix_tables/test_pile_loss_60m.csv",
    "A11": "real_attachments/A_data_value/regmix_tables/test_pile_loss_1B.csv",
    "A13": "real_attachments/A_data_value/regmix_tables/est_pile_loss_10b.csv",
    "A15": "real_attachments/A_data_value/regmix_tables/est_pile_loss_70b.csv",
    "B1": "real_attachments/B_scaling_laws/pythia_training_log_existing.csv",
    "B2": "real_attachments/B_scaling_laws/cerebras_training_log.csv",
    "B4": "real_attachments/B_scaling_laws/scaling_baseline.csv",
    "B5": "real_attachments/B_scaling_laws/published_scaling_data.csv",
    "B6": "real_attachments/B_scaling_laws/supplementary_NQ_experiment.csv",
    "B7": "real_attachments/B_scaling_laws/supplementary_NQ_experiment_expanded.csv",
    "B8": "real_attachments/B_scaling_laws/supplementary_NQ_experiment_large.csv",
    "B10": "real_attachments/B_scaling_laws/supplementary_large_baseline.csv",
}


@dataclass
class Stat:
    count: int = 0
    missing: int = 0
    min_value: float = math.inf
    max_value: float = -math.inf
    list_lengths: Counter[int] = field(default_factory=Counter)
    pos_min: dict[int, float] = field(default_factory=dict)
    pos_max: dict[int, float] = field(default_factory=dict)
    pos_count: Counter[int] = field(default_factory=Counter)
    argmax_counts: Counter[int] = field(default_factory=Counter)

    def add(self, value: Any) -> None:
        if value is None:
            self.missing += 1
            return
        if isinstance(value, list):
            self.count += 1
            self.list_lengths[len(value)] += 1
            numeric_values: list[float] = []
            numeric_positions: list[int] = []
            for i, item in enumerate(value):
                if isinstance(item, bool) or not isinstance(item, (int, float)) or not np.isfinite(item):
                    continue
                x = float(item)
                numeric_values.append(x)
                numeric_positions.append(i)
                self.pos_min[i] = min(self.pos_min.get(i, x), x)
                self.pos_max[i] = max(self.pos_max.get(i, x), x)
                self.pos_count[i] += 1
            if numeric_values and len(numeric_values) == len(value):
                self.argmax_counts[int(np.argmax(np.asarray(numeric_values)))] += 1
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            self.missing += 1
            return
        x = float(value)
        self.count += 1
        self.min_value = min(self.min_value, x)
        self.max_value = max(self.max_value, x)


def stream_jsonl(zf: zipfile.ZipFile, name: str):
    with zf.open(name) as compressed, lzma.LZMAFile(compressed) as source:
        for raw in source:
            if raw.strip():
                yield json.loads(raw)


def read_csv(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as source:
        return pd.read_csv(source, low_memory=False)


def finite_or_blank(x: float) -> str | float:
    return x if np.isfinite(x) else ""


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    log_path = OUTPUTS / "01_confirmation_audit.log"
    quality_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    combined: dict[str, Stat] = {name: Stat() for name in QUALITY_FIELDS}
    source_stats: dict[str, dict[str, Stat]] = {}

    with log_path.open("w", encoding="utf-8") as log, zipfile.ZipFile(F_ZIP) as zf:
        log.write(f"start={datetime.now().astimezone().isoformat()}\n")
        log.write(f"source={F_ZIP}\n")

        for dataset_id, name in JSONL_PATHS.items():
            stats = {field: Stat() for field in QUALITY_FIELDS}
            records = 0
            for obj in stream_jsonl(zf, name):
                records += 1
                for field_name in QUALITY_FIELDS:
                    value = obj.get(field_name)
                    stats[field_name].add(value)
                    combined[field_name].add(value)
            source_stats[dataset_id] = stats
            log.write(f"{dataset_id}: records={records}\n")

        for dataset_id, stats in list(source_stats.items()) + [("A1+A2+A3", combined)]:
            for order, field_name in enumerate(QUALITY_FIELDS, 1):
                stat = stats[field_name]
                if stat.list_lengths:
                    for pos in sorted(stat.pos_count):
                        quality_rows.append({
                            "dataset": dataset_id,
                            "metric_order": order,
                            "metric": field_name,
                            "storage_type": "list",
                            "list_lengths": json.dumps(dict(sorted(stat.list_lengths.items())), ensure_ascii=False),
                            "position": pos,
                            "observed_count": stat.pos_count[pos],
                            "missing_or_nonnumeric": stat.missing,
                            "observed_min": stat.pos_min[pos],
                            "observed_max": stat.pos_max[pos],
                            "argmax_counts": json.dumps(dict(sorted(stat.argmax_counts.items())), ensure_ascii=False),
                        })
                else:
                    quality_rows.append({
                        "dataset": dataset_id,
                        "metric_order": order,
                        "metric": field_name,
                        "storage_type": "scalar",
                        "list_lengths": "",
                        "position": "",
                        "observed_count": stat.count,
                        "missing_or_nonnumeric": stat.missing,
                        "observed_min": finite_or_blank(stat.min_value),
                        "observed_max": finite_or_blank(stat.max_value),
                        "argmax_counts": "",
                    })

        for dataset_id, name in LOSS_PATHS.items():
            df = read_csv(zf, name)
            loss_cols = [c for c in df.columns if "loss" in str(c).lower()]
            values = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in loss_cols], ignore_index=True)
            values = values[np.isfinite(values)]
            ppl_exp_max_abs = ""
            ppl_exp_max_rel = ""
            if "val_loss" in df.columns and "ppl" in df.columns:
                val = pd.to_numeric(df["val_loss"], errors="coerce")
                ppl = pd.to_numeric(df["ppl"], errors="coerce")
                mask = np.isfinite(val) & np.isfinite(ppl)
                if mask.any():
                    expected = np.exp(val[mask].to_numpy())
                    actual = ppl[mask].to_numpy()
                    ppl_exp_max_abs = float(np.max(np.abs(actual - expected)))
                    ppl_exp_max_rel = float(np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-12)))
            loss_rows.append({
                "dataset": dataset_id,
                "archive_path": name,
                "rows": len(df),
                "loss_columns": json.dumps(loss_cols, ensure_ascii=False),
                "loss_column_count": len(loss_cols),
                "loss_min": float(values.min()) if len(values) else "",
                "loss_max": float(values.max()) if len(values) else "",
                "run_id_count": int(df["run_id"].nunique()) if "run_id" in df.columns else "",
                "family_count": int(df["family"].nunique()) if "family" in df.columns else "",
                "tokenizer_field_present": any("token" in str(c).lower() and "d_tokens" not in str(c).lower() and "batch_tokens" not in str(c).lower() for c in df.columns),
                "validation_dataset_field_present": any("valid" in str(c).lower() and "loss" not in str(c).lower() for c in df.columns),
                "loss_definition_field_present": any("definition" in str(c).lower() or "loss_base" in str(c).lower() for c in df.columns),
                "sequence_length_field_present": any("sequence" in str(c).lower() or "context" in str(c).lower() for c in df.columns),
                "ppl_equals_exp_val_loss_max_abs_error": ppl_exp_max_abs,
                "ppl_equals_exp_val_loss_max_relative_error": ppl_exp_max_rel,
            })

    quality_path = RESULTS / "06_quality_metric_profile.csv"
    loss_path = RESULTS / "07_loss_compatibility_checks.csv"
    pd.DataFrame(quality_rows).to_csv(quality_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(loss_rows).to_csv(loss_path, index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(F_ZIP),
        "safety_boundary": "只读F题.zip数据附件；未读取或解析原始数据说明PDF",
        "quality_fields_expected": 22,
        "quality_fields_profiled": len(QUALITY_FIELDS),
        "quality_datasets": {k: JSONL_PATHS[k] for k in JSONL_PATHS},
        "loss_tables_profiled": len(LOSS_PATHS),
        "outputs": [str(quality_path.relative_to(ROOT)), str(loss_path.relative_to(ROOT)), str(log_path.relative_to(ROOT))],
    }
    (RESULTS / "08_confirmation_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"done={datetime.now().astimezone().isoformat()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
