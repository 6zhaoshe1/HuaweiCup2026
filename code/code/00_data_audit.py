"""F题只读数据审计。

AI辅助信息：OpenAI Codex（运行时后端版本未单独核验），OpenAI，2026-09-23。
用途：资料清单、结构统计、缺失/重复/重叠检查；不进行核心模型拟合或参数寻优。
输入保持只读，所有输出仅写入 results/、code/outputs/ 与 _tmp/。
"""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import io
import json
import lzma
import re
import sys
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(r"E:\读研\26届数学建模比赛")
F_ZIP = ROOT / "第二十三届中国研究生数学建模竞赛 - 中文题目" / "中文题目" / "F题.zip"
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "code" / "outputs"
TMP = ROOT / "_tmp" / "f_data_audit"


LOGICAL = {
    "A1": ("real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz", "真实", "问题一：质量主样本", "训练/描述/质量评分；不得把同文本跨折"),
    "A2": ("real_attachments/A_data_value/slimpajama_quality_extended/arxiv_*.jsonl.xz", "真实", "问题一：必用扩展质量信号", "训练/验证；与A1按文本或来源分组去泄漏"),
    "A3": ("real_attachments/A_data_value/slimpajama_quality_extended/github_*.jsonl.xz", "真实", "问题一：必用扩展质量信号", "训练/验证；与A1按文本或来源分组去泄漏"),
    "A4": ("real_attachments/A_data_value/regmix_tables/train_mixture_1m.csv", "真实", "问题一：1M训练配比", "训练"),
    "A5": ("real_attachments/A_data_value/regmix_tables/train_pile_loss_1m.csv", "真实", "问题一：1M训练Loss", "训练；按index与A4连接"),
    "A6": ("real_attachments/A_data_value/regmix_tables/test_mixture_1m.csv", "真实", "问题一：1M检验配比", "检验；不得用于选择模型"),
    "A7": ("real_attachments/A_data_value/regmix_tables/test_pile_loss_1m.csv", "真实", "问题一：1M检验Loss", "检验；按index与A6连接"),
    "A8": ("real_attachments/A_data_value/regmix_tables/test_mixture_60m.csv", "真实", "问题一：60M检验配比", "检验/跨尺度验证"),
    "A9": ("real_attachments/A_data_value/regmix_tables/test_pile_loss_60m.csv", "真实", "问题一：60M检验Loss", "检验/跨尺度验证"),
    "A10": ("real_attachments/A_data_value/regmix_tables/test_mixture_1B.csv", "真实", "问题一：1B检验配比", "检验/跨尺度验证"),
    "A11": ("real_attachments/A_data_value/regmix_tables/test_pile_loss_1B.csv", "真实", "问题一：1B检验Loss", "检验/跨尺度验证"),
    "A12": ("real_attachments/A_data_value/regmix_tables/est_mixture_10b.csv", "训练集子集", "问题一：10B外推配比", "外推稳健性辅助；不是独立真实实验"),
    "A13": ("real_attachments/A_data_value/regmix_tables/est_pile_loss_10b.csv", "外推估算", "问题一：10B外推Loss", "外推稳健性辅助；不得作真实观测"),
    "A14": ("real_attachments/A_data_value/regmix_tables/est_mixture_70b.csv", "训练集子集", "问题一：70B外推配比", "外推稳健性辅助；不是独立真实实验"),
    "A15": ("real_attachments/A_data_value/regmix_tables/est_pile_loss_70b.csv", "外推估算", "问题一：70B外推Loss", "外推稳健性辅助；不得作真实观测"),
    "A16": ("real_attachments/A_data_value/domain_mapping_guide.csv", "参考资料", "问题一：17域到质量域映射", "辅助建立映射；需人工核验假设"),
    "A17": ("real_attachments/A_data_value/regmix_domain_summary.csv", "真实", "问题一：域摘要", "辅助/描述"),
    "A18": ("real_attachments/A_data_value/regmix_domain_sample.jsonl.xz", "真实-可选辅助", "问题一：域原始文本样例", "可选映射核验；不得替代A1-A3全量"),
    "B1": ("real_attachments/B_scaling_laws/pythia_training_log_existing.csv", "真实", "问题二：主拟合训练轨迹", "主拟合；按模型族/检查点分组验证"),
    "B2": ("real_attachments/B_scaling_laws/cerebras_training_log.csv", "半合成", "问题二：族外验证", "外部/族外验证；必须标注半合成"),
    "B3": ("real_attachments/B_scaling_laws/training_trajectories/*.csv", "插值", "问题二：轨迹验证", "轨迹验证；不得当原始检查点"),
    "B4": ("real_attachments/B_scaling_laws/scaling_baseline.csv", "真实", "问题二：跨族收敛基准", "验证/基线"),
    "B5": ("real_attachments/B_scaling_laws/published_scaling_data.csv", "真实", "问题二：文献基准", "文献验证；核对定义"),
    "B6": ("real_attachments/B_scaling_laws/supplementary_NQ_experiment.csv", "半合成", "问题二：质量Q基础", "结构识别/敏感性；不得称真实实验"),
    "B7": ("real_attachments/B_scaling_laws/supplementary_NQ_experiment_expanded.csv", "半合成", "问题二：质量Q扩展", "结构识别/敏感性；不得称真实实验"),
    "B8": ("real_attachments/B_scaling_laws/supplementary_NQ_experiment_large.csv", "半合成-含外推", "问题二：质量Q大规模", "大规模敏感性；不得称真实实验"),
    "B9": ("real_attachments/B_scaling_laws/supplementary_large_models.csv", "真实", "问题二：大模型元数据", "外推边界/元数据"),
    "B10": ("real_attachments/B_scaling_laws/supplementary_large_baseline.csv", "估算", "问题二：大模型Loss", "外推辅助；不得称真实观测"),
    "B11": ("real_attachments/B_scaling_laws/open_model_family_metadata.csv", "真实", "问题二：模型族元数据", "分组/辅助"),
    "B12": ("real_attachments/B_scaling_laws/pythia_checkpoint_index.csv", "真实", "问题二：检查点索引", "检查点连接/辅助"),
    "C1": ("real_attachments/C_efficiency_evolution/leaderboard_cleaned.csv", "真实", "问题四：核心评测表", "主分析/回测"),
    "C2": ("real_attachments/C_efficiency_evolution/leaderboard_enhanced.csv", "真实", "问题四：增强评测表", "可替代C1；避免与C1重复计样本"),
    "C3": ("real_attachments/C_efficiency_evolution/leaderboard_extended_timeseries.csv", "混合来源", "问题四：历史时序", "时序分析；需标明2019-2025历史模拟部分"),
    "C4": ("real_attachments/C_efficiency_evolution/epoch_all_ai_models.csv", "真实", "问题四：宏观元数据", "架构/算力/日期辅助；文本字段需清洗"),
    "C5": ("real_attachments/C_efficiency_evolution/loss_benchmark_bridge.csv", "混合来源", "问题四：Loss-Benchmark桥接", "可比性筛选后建桥"),
    "C6": ("real_attachments/C_efficiency_evolution/loss_benchmark_bridge_expanded.csv", "混合来源", "问题四：扩展桥接", "主桥接候选；需标注来源"),
    "C7": ("real_attachments/C_efficiency_evolution/model_architecture_metadata.csv", "真实", "问题三/四：架构元数据", "上下文情景与架构辅助"),
    "C8": ("real_attachments/C_efficiency_evolution/detailed_results/*.json", "真实-含4个截断", "问题四：逐任务评测", "至少一项逐任务聚合；坏文件跳过并记录"),
    "C9": ("real_attachments/C_efficiency_evolution/data/*.parquet", "真实", "问题四：Leaderboard原始Parquet", "与C1等价，避免重复计样本"),
    "C10": ("real_attachments/C_efficiency_evolution/pythia*_eval_details/README.md", "说明资料", "问题四：评测说明", "只作字段/任务解释，不替代C8"),
}


def ensure_dirs() -> None:
    for p in (RESULTS, OUTPUTS, TMP):
        p.mkdir(parents=True, exist_ok=True)


def log(msg: str, fp: io.TextIOBase) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{stamp}] {msg}"
    print(line)
    fp.write(line + "\n")
    fp.flush()


def sha256_stream(src: Any, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    while True:
        b = src.read(chunk)
        if not b:
            break
        h.update(b)
    return h.hexdigest()


def logical_id(path: str) -> str:
    matches = [k for k, (pat, *_rest) in LOGICAL.items() if fnmatch.fnmatchcase(path, pat)]
    return ";".join(matches) if matches else "UNMAPPED"


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def safe_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and np.isfinite(v):
        return float(v)
    return None


def profile_dataframe(df: pd.DataFrame, path: str, lid: str) -> dict[str, Any]:
    numeric_ranges: dict[str, dict[str, float]] = {}
    unique_counts: dict[str, int] = {}
    for col in df.columns:
        s = df[col]
        unique_counts[str(col)] = int(s.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors="coerce")
            finite = vals[np.isfinite(vals)]
            if len(finite):
                numeric_ranges[str(col)] = {"min": float(finite.min()), "max": float(finite.max())}
    return {
        "logical_id": lid,
        "archive_path": path,
        "format": Path(path).suffix.lower().lstrip("."),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": json_compact([str(x) for x in df.columns]),
        "dtypes": json_compact({str(k): str(v) for k, v in df.dtypes.items()}),
        "missing_by_column": json_compact({str(k): int(v) for k, v in df.isna().sum().items()}),
        "unique_by_column": json_compact(unique_counts),
        "numeric_ranges": json_compact(numeric_ranges),
        "duplicate_rows": int(df.duplicated().sum()),
        "parse_status": "ok",
        "parse_note": "",
    }


def iter_jsonl_xz(zf: zipfile.ZipFile, name: str) -> Iterable[tuple[int, dict[str, Any], bytes]]:
    with zf.open(name) as compressed:
        with lzma.LZMAFile(compressed) as dec:
            for n, raw in enumerate(dec, 1):
                if not raw.strip():
                    continue
                yield n, json.loads(raw), raw


def profile_jsonl(zf: zipfile.ZipFile, name: str, lid: str) -> tuple[dict[str, Any], set[str], set[str]]:
    rows = 0
    keys: set[str] = set()
    missing_null = Counter()
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    numeric_min: dict[str, float] = {}
    numeric_max: dict[str, float] = {}
    line_hashes: set[str] = set()
    duplicate_rows = 0
    content_hashes: set[str] = set()
    source_ids: set[str] = set()
    for _n, obj, raw in iter_jsonl_xz(zf, name):
        rows += 1
        rh = hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest()
        if rh in line_hashes:
            duplicate_rows += 1
        else:
            line_hashes.add(rh)
        if not isinstance(obj, dict):
            type_counts["__record__"][type(obj).__name__] += 1
            continue
        keys.update(obj)
        for k, v in obj.items():
            type_counts[k][type(v).__name__] += 1
            if v is None:
                missing_null[k] += 1
            fv = safe_float(v)
            if fv is not None:
                numeric_min[k] = min(numeric_min.get(k, fv), fv)
                numeric_max[k] = max(numeric_max.get(k, fv), fv)
        text = obj.get("content", obj.get("text"))
        if isinstance(text, str):
            content_hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
        sid = obj.get("id") or obj.get("_source_path") or obj.get("source_path")
        if sid is not None:
            source_ids.add(str(sid))
    profile = {
        "logical_id": lid,
        "archive_path": name,
        "format": "jsonl.xz",
        "rows": rows,
        "columns": len(keys),
        "column_names": json_compact(sorted(keys)),
        "dtypes": json_compact({k: dict(v) for k, v in sorted(type_counts.items())}),
        "missing_by_column": json_compact({k: int(missing_null[k]) for k in sorted(keys)}),
        "unique_by_column": json_compact({"content_hash": len(content_hashes), "source_id": len(source_ids)}),
        "numeric_ranges": json_compact({k: {"min": numeric_min[k], "max": numeric_max[k]} for k in sorted(numeric_min)}),
        "duplicate_rows": duplicate_rows,
        "parse_status": "ok",
        "parse_note": "全量流式读取；未以抽样替代",
    }
    return profile, content_hashes, source_ids


def read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as src:
        return pd.read_csv(src, low_memory=False)


def normalize_model(s: pd.Series) -> set[str]:
    return {
        re.sub(r"[^a-z0-9]+", "", str(x).lower())
        for x in s.dropna().astype(str)
        if str(x).strip()
    }


def detect_model_col(df: pd.DataFrame) -> str | None:
    candidates = ["Model", "model", "model_name", "Model Name", "name", "model_id"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "model" in str(c).lower() and df[c].dtype == object:
            return str(c)
    return None


def main() -> int:
    ensure_dirs()
    log_path = OUTPUTS / "00_data_audit.log"
    with log_path.open("w", encoding="utf-8") as log_fp:
        log(f"START zip={F_ZIP}", log_fp)
        file_rows: list[dict[str, Any]] = []
        table_rows: list[dict[str, Any]] = []
        c8_rows: list[dict[str, Any]] = []
        overlap_rows: list[dict[str, Any]] = []
        dfs: dict[str, pd.DataFrame] = {}
        text_hash_sets: dict[str, set[str]] = {}
        source_id_sets: dict[str, set[str]] = {}
        parse_errors: list[str] = []

        with zipfile.ZipFile(F_ZIP) as zf:
            infos = [x for x in zf.infolist() if not x.is_dir()]
            log(f"archive file entries={len(infos)}", log_fp)
            for idx, info in enumerate(infos, 1):
                name = info.filename.replace("\\", "/")
                lid = logical_id(name)
                with zf.open(info) as src:
                    digest = sha256_stream(src)
                file_rows.append({
                    "archive_path": name,
                    "logical_id": lid,
                    "format": Path(name).suffix.lower().lstrip("."),
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "sha256": digest,
                    "source_zip": str(F_ZIP),
                })
                if idx % 250 == 0:
                    log(f"hashed {idx}/{len(infos)} files", log_fp)

            csv_names = [x.filename for x in infos if x.filename.lower().endswith(".csv")]
            for name in csv_names:
                lid = logical_id(name)
                try:
                    df = read_csv_from_zip(zf, name)
                    dfs[lid + "|" + name] = df
                    table_rows.append(profile_dataframe(df, name, lid))
                except Exception as exc:
                    parse_errors.append(f"{name}: {exc}")
                    table_rows.append({"logical_id": lid, "archive_path": name, "format": "csv", "rows": "", "columns": "", "column_names": "[]", "dtypes": "{}", "missing_by_column": "{}", "unique_by_column": "{}", "numeric_ranges": "{}", "duplicate_rows": "", "parse_status": "error", "parse_note": repr(exc)})
            log(f"profiled csv files={len(csv_names)}", log_fp)

            xz_names = [x.filename for x in infos if x.filename.lower().endswith(".xz")]
            for name in xz_names:
                lid = logical_id(name)
                try:
                    prof, text_hashes, source_ids = profile_jsonl(zf, name, lid)
                    table_rows.append(prof)
                    text_hash_sets[lid] = text_hashes
                    source_id_sets[lid] = source_ids
                    log(f"streamed {lid} rows={prof['rows']}", log_fp)
                except Exception as exc:
                    parse_errors.append(f"{name}: {exc}")
                    table_rows.append({"logical_id": lid, "archive_path": name, "format": "jsonl.xz", "rows": "", "columns": "", "column_names": "[]", "dtypes": "{}", "missing_by_column": "{}", "unique_by_column": "{}", "numeric_ranges": "{}", "duplicate_rows": "", "parse_status": "error", "parse_note": repr(exc)})

            c8_names = [x.filename for x in infos if fnmatch.fnmatchcase(x.filename, LOGICAL["C8"][0])]
            valid_model_dirs: set[str] = set()
            all_model_dirs: set[str] = set()
            c8_hashes: Counter[str] = Counter()
            for n, name in enumerate(c8_names, 1):
                raw = zf.read(name)
                digest = hashlib.sha256(raw).hexdigest()
                c8_hashes[digest] += 1
                rel = name.split("/detailed_results/", 1)[1]
                model_dir = rel.split("/", 1)[0]
                all_model_dirs.add(model_dir)
                row = {"archive_path": name, "model_dir": model_dir, "bytes": len(raw), "sha256": digest, "parse_status": "ok", "parse_error": "", "top_level_type": "", "top_level_keys": "[]", "item_count": ""}
                try:
                    obj = json.loads(raw)
                    valid_model_dirs.add(model_dir)
                    row["top_level_type"] = type(obj).__name__
                    if isinstance(obj, dict):
                        row["top_level_keys"] = json_compact(sorted(obj.keys()))
                        row["item_count"] = len(obj)
                    elif isinstance(obj, list):
                        row["item_count"] = len(obj)
                except Exception as exc:
                    row["parse_status"] = "truncated_or_invalid"
                    row["parse_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
                c8_rows.append(row)
                if n % 400 == 0:
                    log(f"parsed C8 {n}/{len(c8_names)}", log_fp)
            log(f"C8 json={len(c8_names)} valid={sum(r['parse_status']=='ok' for r in c8_rows)} dirs={len(all_model_dirs)} valid_dirs={len(valid_model_dirs)}", log_fp)

            parquet_names = [x.filename for x in infos if x.filename.lower().endswith(".parquet")]
            for name in parquet_names:
                lid = logical_id(name)
                tmp_file = TMP / Path(name).name
                with zf.open(name) as src, tmp_file.open("wb") as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
                try:
                    df = pd.read_parquet(tmp_file)
                    table_rows.append(profile_dataframe(df, name, lid))
                    dfs[lid + "|" + name] = df
                except Exception as exc:
                    table_rows.append({"logical_id": lid, "archive_path": name, "format": "parquet", "rows": 4576, "columns": 36, "column_names": "[]", "dtypes": "{}", "missing_by_column": "{}", "unique_by_column": "{}", "numeric_ranges": "{}", "duplicate_rows": "", "parse_status": "dependency_blocked", "parse_note": "截图可见说明给出4576行×36列；本机无pyarrow/fastparquet且镜像安装失败，列级审计待补"})
                    parse_errors.append(f"{name}: parquet dependency unavailable: {exc}")

        # A1/A2/A3：A2/A3不含原文，只能用id核对；A1同时保留文本哈希。
        for a, b in (("A1", "A2"), ("A1", "A3"), ("A2", "A3")):
            sa, sb = source_id_sets.get(a, set()), source_id_sets.get(b, set())
            overlap_rows.append({"check": f"{a}-{b} record-id overlap", "left_count": len(sa), "right_count": len(sb), "overlap_count": len(sa & sb), "status": "risk" if sa & sb else "pass", "interpretation": "A2/A3不含原文，按id全量核对；同id不得跨独立验证折"})

        # RegMix 配比/Loss 的index配对及训练-检验-外推重叠。
        def get_df(lid: str) -> pd.DataFrame | None:
            for key, df in dfs.items():
                if key.startswith(lid + "|"):
                    return df
            return None

        for mix, loss in (("A4", "A5"), ("A6", "A7"), ("A8", "A9"), ("A10", "A11"), ("A12", "A13"), ("A14", "A15")):
            dm, dl = get_df(mix), get_df(loss)
            if dm is not None and dl is not None and "index" in dm and "index" in dl:
                sm, sl = set(dm["index"].dropna()), set(dl["index"].dropna())
                overlap_rows.append({"check": f"{mix}-{loss} index pairing", "left_count": len(sm), "right_count": len(sl), "overlap_count": len(sm & sl), "status": "pass" if sm == sl else "risk", "interpretation": "配比表与Loss表应按index一一对应"})
        split_ids = {k: set(get_df(k)["index"].dropna()) for k in ("A4", "A6", "A8", "A10", "A12", "A14") if get_df(k) is not None and "index" in get_df(k)}
        for left in ("A4",):
            for right in ("A6", "A8", "A10", "A12", "A14"):
                if left in split_ids and right in split_ids:
                    inter = split_ids[left] & split_ids[right]
                    overlap_rows.append({"check": f"{left}-{right} index-label overlap", "left_count": len(split_ids[left]), "right_count": len(split_ids[right]), "overlap_count": len(inter), "status": "label_reuse_not_evidence", "interpretation": "index是各表局部编号，重号本身不证明样本重叠；以17维配比向量复核"})

        def mixture_hashes(lid: str) -> tuple[set[str], dict[str, float]]:
            df = get_df(lid)
            if df is None:
                return set(), {"min_sum": float("nan"), "max_sum": float("nan"), "violations": -1}
            cols = [c for c in df.columns if c != "index"]
            x = df[cols].apply(pd.to_numeric, errors="coerce")
            hashes = {
                hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).hexdigest()
                for row in x.round(12).to_numpy()
            }
            sums = x.sum(axis=1)
            return hashes, {
                "min_sum": float(sums.min()),
                "max_sum": float(sums.max()),
                "violations": int((np.abs(sums - 1.0) > 1e-6).sum()),
            }

        mixture_sets: dict[str, set[str]] = {}
        for lid in ("A4", "A6", "A8", "A10", "A12", "A14"):
            hashes, simplex = mixture_hashes(lid)
            mixture_sets[lid] = hashes
            overlap_rows.append({"check": f"{lid} simplex row-sum", "left_count": len(hashes), "right_count": "", "overlap_count": simplex["violations"], "status": "pass" if simplex["violations"] == 0 else "risk", "interpretation": f"17域配比和应为1；min={simplex['min_sum']:.12g}, max={simplex['max_sum']:.12g}"})
        for a, b in (("A4", "A6"), ("A4", "A8"), ("A4", "A10"), ("A4", "A12"), ("A4", "A14"), ("A6", "A8"), ("A12", "A14")):
            inter = mixture_sets[a] & mixture_sets[b]
            expected = (a, b) in (("A6", "A8"), ("A12", "A14"))
            overlap_rows.append({"check": f"{a}-{b} exact mixture-vector overlap", "left_count": len(mixture_sets[a]), "right_count": len(mixture_sets[b]), "overlap_count": len(inter), "status": "expected_scale_pair" if expected else ("risk" if inter else "pass"), "interpretation": "A6/A8与A12/A14按题面是相同配比在不同模型尺度的成对数据；其他跨训练/检验重复需隔离"})

        # C表模型重叠为连接所需，但切分必须按模型族/模型实体分组。
        csets: dict[str, set[str]] = {}
        for lid in ("C1", "C2", "C3", "C4", "C5", "C6", "C7"):
            df = get_df(lid)
            if df is None:
                continue
            col = detect_model_col(df)
            if col:
                csets[lid] = normalize_model(df[col])
        for a, b in (("C1", "C2"), ("C1", "C3"), ("C1", "C4"), ("C5", "C6"), ("C1", "C7")):
            if a in csets and b in csets:
                inter = csets[a] & csets[b]
                overlap_rows.append({"check": f"{a}-{b} normalized model overlap", "left_count": len(csets[a]), "right_count": len(csets[b]), "overlap_count": len(inter), "status": "expected_join_but_split_risk", "interpretation": "跨表连接所需；建模切分时同模型/同族不得跨折，C1与C2/C9不得重复计样本"})

        logical_rows = []
        for lid, (pat, nature, role, allowed) in LOGICAL.items():
            matches = [r for r in file_rows if lid in r["logical_id"].split(";")]
            logical_rows.append({
                "logical_id": lid,
                "path_pattern": pat,
                "actual_file_count": len(matches),
                "actual_bytes": sum(int(r["bytes"]) for r in matches),
                "format": ";".join(sorted({r["format"] for r in matches})),
                "source_nature": nature,
                "problem_role": role,
                "allowed_use": allowed,
                "presence_status": "present" if matches else "missing",
            })

        pd.DataFrame(file_rows).sort_values("archive_path").to_csv(RESULTS / "00_file_inventory.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(logical_rows).to_csv(RESULTS / "01_logical_attachment_status.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(table_rows).sort_values(["logical_id", "archive_path"]).to_csv(RESULTS / "02_tabular_profile.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(c8_rows).sort_values("archive_path").to_csv(RESULTS / "03_c8_json_profile.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(overlap_rows).to_csv(RESULTS / "04_overlap_checks.csv", index=False, encoding="utf-8-sig")

        summary = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_zip": str(F_ZIP),
            "source_zip_sha256": hashlib.sha256(F_ZIP.read_bytes()).hexdigest(),
            "file_entries": len(file_rows),
            "mapped_file_entries": sum(r["logical_id"] != "UNMAPPED" for r in file_rows),
            "unmapped_paths": [r["archive_path"] for r in file_rows if r["logical_id"] == "UNMAPPED"],
            "logical_ids_present": sum(r["presence_status"] == "present" for r in logical_rows),
            "logical_ids_total": len(logical_rows),
            "csv_profiles": sum(r["format"] == "csv" for r in table_rows),
            "xz_profiles": sum(r["format"] == "jsonl.xz" for r in table_rows),
            "parquet_profiles": sum(r["format"] == "parquet" for r in table_rows),
            "c8_json_files": len(c8_rows),
            "c8_valid_json": sum(r["parse_status"] == "ok" for r in c8_rows),
            "c8_invalid_json": sum(r["parse_status"] != "ok" for r in c8_rows),
            "c8_model_dirs": len({r["model_dir"] for r in c8_rows}),
            "c8_dirs_with_valid_json": len({r["model_dir"] for r in c8_rows if r["parse_status"] == "ok"}),
            "c8_duplicate_content_hash_groups": sum(v > 1 for v in Counter(r["sha256"] for r in c8_rows).values()),
            "parse_errors_or_dependency_blocks": parse_errors,
            "outputs": [
                "results/00_file_inventory.csv",
                "results/01_logical_attachment_status.csv",
                "results/02_tabular_profile.csv",
                "results/03_c8_json_profile.csv",
                "results/04_overlap_checks.csv",
                "code/outputs/00_data_audit.log",
            ],
        }
        (RESULTS / "00_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"DONE outputs={len(summary['outputs'])} parse_blocks={len(parse_errors)}", log_fp)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
