#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026 研赛 F 题问题二：广义标度律、质量效应与配比桥接。

AI 辅助信息：OpenAI Codex（GPT-5 系列），OpenAI，2026-09-24。
参赛队必须复核模型选择、数学推导与论文表述。

安全与证据边界
--------------
1. 只读 F题.zip 中正式附件；不读取受污染的《数据说明》PDF。
2. 不读取或复用老师/DeepSeek 的结果文件；所有正式数值均由本脚本重算。
3. B6 是 B7 的真子集，任何拟合均不得把二者同时当作独立样本。
4. A、B 的绝对 Loss 不直接拼接；A 组只向 B 组传递无量纲相对配比指数。
5. B8 外推段与 B10 仅作压力测试，不参与模型选择。
"""

from __future__ import annotations

import io
import json
import math
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear, brentq
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from q2_predict import equivalent_n


SEED = 42
RNG = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "第二十三届中国研究生数学建模竞赛 - 中文题目" / "中文题目" / "F题.zip"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOG_DIR = ROOT / "code" / "outputs"
for directory in (RESULTS, FIGURES, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "_模板" / "scripts"))
from mpl_cn import plt, save_fig, panel_label  # noqa: E402

LOG_PATH = LOG_DIR / "q2_modeling.log"


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


B_PREFIX = "real_attachments/B_scaling_laws/"
A_MEMBERS = {
    "A4": "real_attachments/A_data_value/regmix_tables/train_mixture_1m.csv",
    "A5": "real_attachments/A_data_value/regmix_tables/train_pile_loss_1m.csv",
    "A6": "real_attachments/A_data_value/regmix_tables/test_mixture_1m.csv",
    "A7": "real_attachments/A_data_value/regmix_tables/test_pile_loss_1m.csv",
    "A8": "real_attachments/A_data_value/regmix_tables/test_mixture_60m.csv",
    "A9": "real_attachments/A_data_value/regmix_tables/test_pile_loss_60m.csv",
    "A10": "real_attachments/A_data_value/regmix_tables/test_mixture_1B.csv",
    "A11": "real_attachments/A_data_value/regmix_tables/test_pile_loss_1B.csv",
}
B_FILES = {
    "B1": "pythia_training_log_existing.csv",
    "B2": "cerebras_training_log.csv",
    "B4": "scaling_baseline.csv",
    "B5": "published_scaling_data.csv",
    "B6": "supplementary_NQ_experiment.csv",
    "B7": "supplementary_NQ_experiment_expanded.csv",
    "B8": "supplementary_NQ_experiment_large.csv",
    "B10": "supplementary_large_baseline.csv",
}


def read_zip_csv(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as handle:
        return pd.read_csv(handle)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    rho = spearmanr(y, pred).statistic if len(y) > 2 else np.nan
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "spearman": float(rho),
        "bias": float(np.mean(pred - y)),
        "log_rmse": float(np.sqrt(np.mean((np.log(pred) - np.log(y)) ** 2))),
    }


def classic_predict(theta: np.ndarray, n: np.ndarray, d: np.ndarray) -> np.ndarray:
    e, a0, alpha, b0, beta = np.asarray(theta, float)
    return e + a0 * np.power(n, -alpha) + b0 * np.power(d, -beta)


def fit_classic(n: np.ndarray, d: np.ndarray, y: np.ndarray, objective: str,
                seed: int = SEED, starts: int = 8) -> tuple[np.ndarray, dict]:
    n, d, y = map(lambda x: np.asarray(x, float), (n, d, y))
    lo = np.array([0.05, 1e-4, 0.02, 1e-4, 0.02])
    hi = np.array([max(0.051, y.min() * 0.999), 5000.0, 1.5, 5000.0, 1.5])
    rng = np.random.default_rng(seed)
    candidates = [np.array([min(1.7, hi[0] * .9), 400., .34, 400., .28])]
    for _ in range(starts - 1):
        candidates.append(np.array([
            rng.uniform(lo[0], hi[0]),
            10 ** rng.uniform(-1, 3), rng.uniform(.08, .8),
            10 ** rng.uniform(-1, 3), rng.uniform(.08, .8),
        ]))

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = classic_predict(theta, n, d)
        if objective == "log_huber":
            return np.log(pred) - np.log(y)
        return pred - y

    best = None
    for x0 in candidates:
        x0 = np.clip(x0, lo * 1.001, hi * .999)
        f_scale = 0.01 if objective == "log_huber" else max(0.01, .1 * np.std(y))
        res = least_squares(residual, x0, bounds=(lo, hi), loss="huber",
                            f_scale=f_scale, max_nfev=30000, xtol=1e-11,
                            ftol=1e-11, gtol=1e-11)
        # Select restarts by the objective actually optimized, not raw SSE.
        score = float(res.cost)
        if best is None or score < best[0]:
            best = (score, res)
    assert best is not None
    res = best[1]
    jtj = res.jac.T @ res.jac
    cond = float(np.linalg.cond(jtj)) if np.all(np.isfinite(jtj)) else np.inf
    return res.x, {"success": bool(res.success), "cost": float(res.cost),
                   "optimality": float(res.optimality), "jacobian_condition": cond,
                   "nfev": int(res.nfev), "active_bounds": int(np.count_nonzero(res.active_mask)),
                   "restart_selection": "minimum_Huber_cost"}


def classical_validation(b1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, preds = [], []
    n = b1["N_params_B"].to_numpy(float)
    d = b1["D_tokens_B"].to_numpy(float)
    y = b1["val_loss"].to_numpy(float)
    group = b1["N_params_B"].round(9).astype(str).to_numpy()
    objectives = ["raw_huber", "log_huber"]
    for objective in objectives:
        # Leave-one-model-size-out: the independent unit is a full trajectory.
        oof = np.full(len(b1), np.nan)
        for k, g in enumerate(np.unique(group)):
            te = group == g
            th, _ = fit_classic(n[~te], d[~te], y[~te], objective, seed=SEED + k)
            oof[te] = classic_predict(th, n[te], d[te])
        m = metrics(y, oof)
        rows.append({"model": objective, "split": "leave_one_size_out", **m,
                     "evidence_level": "real_holdout"})
        preds.extend({"row_id": int(i), "model": objective, "split": "leave_one_size_out",
                      "N_params_B": n[i], "D_tokens_B": d[i], "observed": y[i],
                      "predicted": oof[i], "group_id": group[i],
                      "evidence_level": "real_holdout"} for i in range(len(y)))

        # Late-token blocked holdout: fit early 75% within each trajectory.
        train = np.zeros(len(b1), dtype=bool)
        for g in np.unique(group):
            ix = np.flatnonzero(group == g)
            cutoff = np.quantile(d[ix], .75)
            train[ix[d[ix] <= cutoff]] = True
        th, _ = fit_classic(n[train], d[train], y[train], objective, seed=SEED + 100)
        pr = classic_predict(th, n[~train], d[~train])
        m = metrics(y[~train], pr)
        rows.append({"model": objective, "split": "late_token_block", **m,
                     "evidence_level": "real_holdout"})
        for i, p in zip(np.flatnonzero(~train), pr):
            preds.append({"row_id": int(i), "model": objective, "split": "late_token_block",
                          "N_params_B": n[i], "D_tokens_B": d[i], "observed": y[i],
                          "predicted": p, "group_id": group[i],
                          "evidence_level": "real_holdout"})

        # Random-row CV is retained only to quantify leakage optimism.
        random_oof = np.full(len(b1), np.nan)
        for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(n)):
            th, _ = fit_classic(n[tr], d[tr], y[tr], objective, seed=SEED + 200 + k)
            random_oof[te] = classic_predict(th, n[te], d[te])
        m = metrics(y, random_oof)
        rows.append({"model": objective, "split": "random_row_diagnostic_only", **m,
                     "evidence_level": "optimistic_diagnostic"})

    score = pd.DataFrame(rows)
    # Revision protocol: primary task is unseen scale; late token is secondary.
    primary = score[score["split"] == "leave_one_size_out"].copy()
    rank = primary.groupby("model")["rmse"].mean().sort_values()
    selected = str(rank.index[0])
    theta, diag = fit_classic(n, d, y, selected, seed=SEED + 999, starts=12)
    return score, pd.DataFrame(preds), {"selected": selected, "theta": theta, "diagnostics": diag}


def source_calibration(name: str, df: pd.DataFrame, theta: np.ndarray) -> tuple[list[dict], pd.DataFrame]:
    n = df["N_params_B"].to_numpy(float)
    d = df["D_tokens_B"].to_numpy(float)
    y = df["val_loss"].to_numpy(float)
    _, a0, alpha, b0, beta = theta
    r = a0 * n ** (-alpha) + b0 * d ** (-beta)
    rows = []
    details = df.copy()
    details["evidence_level"] = {"B2":"semi_synthetic", "B4":"real_observational",
                                 "B5":"real_observational", "B10":"extrapolated"}[name]
    details["B1_reducible_shape"] = r
    details["B1_absolute_prediction"] = classic_predict(theta, n, d)
    rows.append({"dataset": name, "test": "uncalibrated_absolute", **metrics(y, details["B1_absolute_prediction"]),
                 "calibration_E": theta[0], "calibration_scale": 1.0,
                 "evidence_level": "cross_source_stress_test"})

    # Shape compatibility after using the target source to estimate only E_s and s_s.
    x = np.column_stack([np.ones(len(r)), r])
    sol = lsq_linear(x, y, bounds=([0., 0.], [np.inf, np.inf])).x
    calibrated = x @ sol
    details["source_calibrated_prediction"] = calibrated
    rows.append({"dataset": name, "test": "target_affine_shape_fit", **metrics(y, calibrated),
                 "calibration_E": sol[0], "calibration_scale": sol[1],
                 "evidence_level": "in_sample_shape_diagnostic"})

    # B2 trajectories allow an early-token calibration and a genuinely later-token test.
    if name == "B2":
        run = df["run_id"].astype(str).to_numpy()
        early = np.zeros(len(df), bool)
        for g in np.unique(run):
            ix = np.flatnonzero(run == g)
            cutoff = np.quantile(d[ix], .70)
            early[ix[d[ix] <= cutoff]] = True
        sol2 = lsq_linear(x[early], y[early], bounds=([0., 0.], [np.inf, np.inf])).x
        late_pred = x[~early] @ sol2
        details["early_calibrated_late_prediction"] = np.nan
        details.loc[~early, "early_calibrated_late_prediction"] = late_pred
        rows.append({"dataset": name, "test": "early_calibrate_late_predict", **metrics(y[~early], late_pred),
                     "calibration_E": sol2[0], "calibration_scale": sol2[1],
                     "evidence_level": "semi_synthetic_holdout"})
    return rows, details


def q_predict(kind: str, pars: np.ndarray, base: np.ndarray,
              n: np.ndarray, d: np.ndarray, q: np.ndarray) -> np.ndarray:
    e, a0, alpha, b0, beta = base
    rn = a0 * n ** (-alpha)
    rd = b0 * d ** (-beta)
    u = np.clip(1.0 - q, 0.0, None)
    if kind == "none":
        return e + rn + rd
    if kind == "additive":
        return e + rn + rd + pars[0] * u
    if kind == "reducible_linear":
        return e + (rn + rd) * (1.0 + pars[0] * u)
    if kind == "reducible_power":
        return e + (rn + rd) * (1.0 + pars[0] * np.power(u, pars[1]))
    if kind == "effective_data":
        return e + rn + b0 * np.power(d * np.exp(pars[0] * (q - 1.0)), -beta)
    if kind == "n_power":
        return e + rn * np.power(np.clip(q, 1e-3, None), -pars[0]) + rd
    raise KeyError(kind)


Q_BOUNDS = {
    "none": (np.array([]), np.array([]), np.array([])),
    "additive": (np.array([.1]), np.array([0.]), np.array([20.])),
    "reducible_linear": (np.array([.5]), np.array([0.]), np.array([20.])),
    "reducible_power": (np.array([.5, 1.]), np.array([0., .1]), np.array([20., 4.])),
    "effective_data": (np.array([1.]), np.array([0.]), np.array([20.])),
    "n_power": (np.array([.3]), np.array([0.]), np.array([5.])),
}


def fit_q(kind: str, df: pd.DataFrame, base: np.ndarray) -> tuple[np.ndarray, dict]:
    x0, lo, hi = Q_BOUNDS[kind]
    if kind == "none":
        return np.array([]), {"success": True, "jacobian_condition": np.nan, "nfev": 0}
    n = df["N_params_B"].to_numpy(float)
    d = df["D_tokens_B"].to_numpy(float)
    q = df["Q_score"].to_numpy(float)
    y = df["val_loss"].to_numpy(float)

    def resid(p):
        return np.log(q_predict(kind, p, base, n, d, q)) - np.log(y)

    res = least_squares(resid, x0, bounds=(lo, hi), loss="huber", f_scale=.01,
                        max_nfev=15000, xtol=1e-11, ftol=1e-11, gtol=1e-11)
    jtj = res.jac.T @ res.jac
    cond = float(np.linalg.cond(jtj)) if jtj.size else np.nan
    return res.x, {"success": bool(res.success), "jacobian_condition": cond, "nfev": int(res.nfev)}


def q_model_comparison(b6: pd.DataFrame, b7: pd.DataFrame, b8: pd.DataFrame,
                       base: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    kinds = list(Q_BOUNDS)
    groups = b7[["N_params_B", "D_tokens_B"]].astype(str).agg("|".join, axis=1).to_numpy()
    n, d, q, y = (b7[c].to_numpy(float) for c in ["N_params_B", "D_tokens_B", "Q_score", "val_loss"])
    rows, pred_rows = [], []
    splitter = GroupKFold(n_splits=5)
    for kind in kinds:
        oof = np.full(len(b7), np.nan)
        fold_pars = []
        for fold, (tr, te) in enumerate(splitter.split(b7, groups=groups)):
            pars, _ = fit_q(kind, b7.iloc[tr], base)
            oof[te] = q_predict(kind, pars, base, n[te], d[te], q[te])
            fold_pars.append(pars.tolist())
        m = metrics(y, oof)
        rows.append({"model": kind, "split": "grouped_ND_5fold", **m,
                     "parameter_count_q": len(Q_BOUNDS[kind][0]),
                     "fold_parameters_json": json.dumps(fold_pars),
                     "evidence_level": "semi_synthetic_holdout"})
        pred_rows.extend({"model": kind, "split": "grouped_ND_5fold", "row_id": int(i),
                          "N_params_B": n[i], "D_tokens_B": d[i], "Q_score": q[i],
                          "observed": y[i], "predicted": oof[i], "group_id": groups[i],
                          "evidence_level": "semi_synthetic_holdout"} for i in range(len(b7)))

        # Train B6, test only new Q levels/rows in B7 (never double count them).
        keys6 = set(map(tuple, b6[["N_params_B", "D_tokens_B", "Q_score"]].round(10).to_numpy()))
        new = np.array([tuple(v) not in keys6 for v in b7[["N_params_B", "D_tokens_B", "Q_score"]].round(10).to_numpy()])
        pars6, _ = fit_q(kind, b6, base)
        pr = q_predict(kind, pars6, base, n[new], d[new], q[new])
        rows.append({"model": kind, "split": "B6_to_B7_new_Q_rows", **metrics(y[new], pr),
                     "parameter_count_q": len(Q_BOUNDS[kind][0]),
                     "fold_parameters_json": json.dumps([pars6.tolist()]),
                     "evidence_level": "semi_synthetic_interpolation_holdout"})

    score = pd.DataFrame(rows)
    primary = score[score["split"] == "grouped_ND_5fold"].sort_values(["rmse", "mae"])
    selected = str(primary.iloc[0]["model"])
    pars, diag = fit_q(selected, b7, base)

    # B8 is never used for selection: calibrated and extrapolated portions are separate pressure tests.
    b8_rows = []
    for dtype, sub in b8.groupby("data_type", dropna=False):
        pred = q_predict(selected, pars, base, sub["N_params_B"].to_numpy(float),
                         sub["D_tokens_B"].to_numpy(float), sub["Q_score"].to_numpy(float))
        b8_rows.append({"model": selected, "split": f"B8_{dtype}", **metrics(sub["val_loss"], pred),
                        "parameter_count_q": len(pars), "fold_parameters_json": json.dumps([pars.tolist()]),
                        "evidence_level": "semi_synthetic_pressure_test" if dtype == "calibrated" else "extrapolated"})
    score = pd.concat([score, pd.DataFrame(b8_rows)], ignore_index=True)
    return score, pd.DataFrame(pred_rows), {"selected": selected, "pars": pars, "diagnostics": diag}, primary


def bootstrap_q(b7: pd.DataFrame, base: np.ndarray, kind: str, reps: int = 300) -> pd.DataFrame:
    groups = b7[["N_params_B", "D_tokens_B"]].drop_duplicates().reset_index(drop=True)
    out = []
    for b in range(reps):
        take = RNG.integers(0, len(groups), len(groups))
        pieces = []
        for new_id, j in enumerate(take):
            key = groups.iloc[j]
            sub = b7[(b7["N_params_B"] == key["N_params_B"]) & (b7["D_tokens_B"] == key["D_tokens_B"])].copy()
            sub["boot_group"] = new_id
            pieces.append(sub)
        sample = pd.concat(pieces, ignore_index=True)
        try:
            pars, diag = fit_q(kind, sample, base)
            row = {"bootstrap_id": b, "success": diag["success"], "jacobian_condition": diag["jacobian_condition"]}
            for j, value in enumerate(pars):
                row[f"q_parameter_{j}"] = value
        except Exception:
            row = {"bootstrap_id": b, "success": False, "jacobian_condition": np.nan}
        out.append(row)
    return pd.DataFrame(out).assign(evidence_level="semi_synthetic")


def join_a_pair(mix: pd.DataFrame, loss: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    candidates = [c for c in mix.columns if c in loss.columns and ("id" in c.lower() or "index" in c.lower())]
    if not candidates:
        raise ValueError("A 组配方表与 Loss 表没有明确公共关联键，禁止按行位置对齐")
    key = candidates[0]
    if mix[key].duplicated().any() or loss[key].duplicated().any():
        raise ValueError(f"关联键 {key} 非唯一")
    merged = mix.merge(loss, on=key, how="inner", validate="one_to_one", suffixes=("", "_loss"))
    if len(merged) != len(mix) or len(merged) != len(loss):
        raise ValueError(f"关联键 {key} 未实现全匹配: {len(mix)}/{len(loss)}/{len(merged)}")
    pcols = [c for c in mix.columns if c.startswith("train_the_pile_")]
    lcols = [c for c in loss.columns if c.startswith("metric/the_pile_") and c.endswith("_val_loss")]
    if len(pcols) != 17 or len(lcols) != 13:
        raise ValueError(f"A 组字段数异常：配比={len(pcols)}，Loss={len(lcols)}")
    return merged, pcols, lcols


def p_bridge(a_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    params = pd.read_csv(RESULTS / "q1_upgrade_selected_parameters.csv")
    delta = float(params["delta"].dropna().iloc[0])
    feature_rows = params[params["feature"].str.startswith("log(")].copy()
    domain_names = [x[len("log("):-len("+delta)")] for x in feature_rows["feature"].unique()]
    outputs = params["output_domain"].unique().tolist()
    coef = np.zeros((len(outputs), len(domain_names)))
    intercept = np.zeros(len(outputs))
    for i, out in enumerate(outputs):
        sub = params[params["output_domain"] == out]
        intercept[i] = float(sub.loc[sub["feature"] == "intercept", "coefficient"].iloc[0])
        for j, dom in enumerate(domain_names):
            coef[i, j] = float(sub.loc[sub["feature"] == f"log({dom}+delta)", "coefficient"].iloc[0])

    merged4, pcols, lcols = join_a_pair(a_tables["A4"], a_tables["A5"])
    pmap = {c.replace("train_the_pile_", ""): c for c in pcols}
    if set(domain_names) != set(pmap):
        raise ValueError(f"Q1 参数领域与 A4 配方领域不一致: params={domain_names}, data={list(pmap)}")

    def predict_index(df: pd.DataFrame, ref_stats=None):
        p = df[[pmap[d] for d in domain_names]].to_numpy(float)
        p = p / p.sum(axis=1, keepdims=True)
        yhat = intercept + np.log(p + delta) @ coef.T
        if ref_stats is None:
            mu = yhat.mean(axis=0)
            sd = yhat.std(axis=0, ddof=1)
            sd[sd <= 1e-12] = 1.
            raw = ((yhat - mu) / sd).mean(axis=1)
            center, scale = raw.mean(), raw.std(ddof=1)
            stats = (mu, sd, center, scale)
        else:
            mu, sd, center, scale = ref_stats
            raw = ((yhat - mu) / sd).mean(axis=1)
            stats = ref_stats
        # Larger R_A means predicted lower loss / better mixture.
        r_a = -(raw - stats[2]) / stats[3]
        return r_a, yhat, stats

    r4, _, stats = predict_index(merged4)
    bridge_rows = []
    pred_rows = []
    for mix_key, loss_key, label, level in [
        ("A4", "A5", "1M_train", "real_observational"),
        ("A6", "A7", "1M_holdout", "real_holdout"),
        ("A8", "A9", "60M_holdout", "real_holdout"),
        ("A10", "A11", "1B_holdout", "real_holdout"),
    ]:
        merged, pc, lc = join_a_pair(a_tables[mix_key], a_tables[loss_key])
        r_a, _, _ = predict_index(merged, stats)
        obs = merged[lc].to_numpy(float)
        obs_z = ((obs - obs.mean(axis=0)) / np.where(obs.std(axis=0, ddof=1) > 0,
                                                     obs.std(axis=0, ddof=1), 1.)).mean(axis=1)
        rho = float(spearmanr(r_a, -obs_z).statistic)
        # y_z = c - eta * R_A; eta is identifiable within A but not transferable to B.
        x = np.column_stack([np.ones(len(r_a)), -r_a])
        sol = np.linalg.lstsq(x, obs_z, rcond=None)[0]
        fit = x @ sol
        bridge_rows.append({"scale": label, "n": len(merged), "spearman_p_index_vs_better_loss": rho,
                            "eta_A_within_source": sol[1], "standardized_loss_rmse": float(np.sqrt(np.mean((fit-obs_z)**2))),
                            "evidence_level": level,
                            "interpretation": "A-source relative effect only; not an identified B-source coefficient"})
        pred_rows.extend({"scale": label, "row_id": int(i), "R_A": r_a[i],
                          "observed_standardized_mean_loss": obs_z[i], "predicted_standardized_mean_loss": fit[i],
                          "evidence_level": level} for i in range(len(r_a)))
    return pd.DataFrame(bridge_rows), pd.DataFrame(pred_rows), {
        "delta": delta, "domain_names": domain_names,
        "reference": "A4 predicted-output mean/SD, followed by aggregate mean/SD; NOT the prediction at mean p",
        "coefficients": coef.tolist(), "intercepts": intercept.tolist(),
        "output_mean": stats[0].tolist(), "output_sd": stats[1].tolist(),
        "index_center": float(stats[2]), "index_scale": float(stats[3]),
        "formula": "psi_p(p;eta_p)=exp(-eta_p*R_A(p)); eta_p is not identified by B data",
    }


def marginal_table(base: np.ndarray, q_kind: str, q_pars: np.ndarray,
                   q_boot: pd.DataFrame) -> pd.DataFrame:
    if q_kind != "reducible_linear":
        raise ValueError("Analytic substitution requires reducible_linear; revise formula if selected model changes")
    e, a0, alpha, b0, beta = base
    rows = []
    pcols = [c for c in q_boot if c.startswith("q_parameter_")]
    boot_pars = q_boot.loc[q_boot["success"], pcols].to_numpy(float) if pcols else np.empty((0, 0))
    for n in [0.1, 1., 10.]:
        for d in [10., 100., 300.]:
            for q in [.3, .5, .7, .9]:
                pred = float(q_predict(q_kind, q_pars, base, np.array([n]), np.array([d]), np.array([q]))[0])
                h = 1e-4
                dN = (q_predict(q_kind, q_pars, base, np.array([n*(1+h)]), np.array([d]), np.array([q]))[0] -
                      q_predict(q_kind, q_pars, base, np.array([n*(1-h)]), np.array([d]), np.array([q]))[0]) / (2*h*n)
                dD = (q_predict(q_kind, q_pars, base, np.array([n]), np.array([d*(1+h)]), np.array([q]))[0] -
                      q_predict(q_kind, q_pars, base, np.array([n]), np.array([d*(1-h)]), np.array([q]))[0]) / (2*h*d)
                dq = (q_predict(q_kind, q_pars, base, np.array([n]), np.array([d]), np.array([min(.999, q+h)]))[0] -
                      q_predict(q_kind, q_pars, base, np.array([n]), np.array([d]), np.array([max(.001, q-h)]))[0]) / (2*h)
                q2 = q + .1
                target = float(q_predict(q_kind, q_pars, base, np.array([n]), np.array([d]), np.array([q2]))[0])

                eq = equivalent_n(base, q_pars[0], n, d, q, q2)
                neq = eq["N_equivalent"] if eq["N_equivalent"] is not None else np.nan
                boot_loss, boot_mult = [], []
                for bp in boot_pars:
                    bp = np.asarray(bp, float)
                    boot_loss.append(float(q_predict(q_kind, bp, base, np.array([n]), np.array([d]), np.array([q]))[0]))
                    btarget = float(q_predict(q_kind, bp, base, np.array([n]), np.array([d]), np.array([q2]))[0])
                    beq = equivalent_n(base, bp[0], n, d, q, q2)
                    bneq = beq["N_equivalent"] if beq["N_equivalent"] is not None else np.nan
                    boot_mult.append(bneq/n if np.isfinite(bneq) else np.nan)
                boot_loss = np.asarray(boot_loss, float); boot_mult = np.asarray(boot_mult, float)
                rows.append({"N_params_B": n, "D_tokens_B": d, "Q_score": q, "predicted_loss": pred,
                             "dL_dN": dN, "dL_dD": dD, "dL_dQ": dq,
                             "elasticity_N": -dN*n/pred, "elasticity_D": -dD*d/pred,
                             "elasticity_Q": -dq*q/pred, "Q_plus_0p1": q2,
                             "equivalent_N_params_B": neq,
                             "analytic_solution_status": eq["status"],
                             "equivalent_N_multiplier": neq/n if np.isfinite(neq) else np.nan,
                             "predicted_loss_qboot_ci2p5": np.nanquantile(boot_loss, .025) if len(boot_loss) else np.nan,
                             "predicted_loss_qboot_ci97p5": np.nanquantile(boot_loss, .975) if len(boot_loss) else np.nan,
                             "equivalent_N_multiplier_qboot_ci2p5": np.nanquantile(boot_mult, .025) if len(boot_mult) else np.nan,
                             "equivalent_N_multiplier_qboot_ci97p5": np.nanquantile(boot_mult, .975) if len(boot_mult) else np.nan,
                             "within_Q_calibration_range": q2 <= 1.0,
                             "uncertainty_scope": "B7 (N,D)-group bootstrap of Q parameter only; excludes structural/source uncertainty",
                             "evidence_level": "semi_synthetic_model_scenario"})
    return pd.DataFrame(rows)


def make_figures(b1: pd.DataFrame, classic_pred: pd.DataFrame, classic_score: pd.DataFrame,
                 source_summary: pd.DataFrame, q_score: pd.DataFrame, q_pred: pd.DataFrame,
                 q_selected: dict, base: np.ndarray, marginal: pd.DataFrame,
                 p_summary: pd.DataFrame, p_detail: pd.DataFrame) -> pd.DataFrame:
    contracts = []
    # 1: Real learning trajectories.
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for n, sub in b1.groupby("N_params_B"):
        ax.plot(sub["D_tokens_B"], sub["val_loss"], label=f"{n:g}B")
    ax.set_xscale("log"); ax.set_xlabel("训练 Token 数 D（十亿）"); ax.set_ylabel("验证 Loss")
    ax.legend(ncol=2, title="参数量 N", fontsize=8)
    save_fig(fig, FIGURES / "q2_01_b1_trajectories.pdf", also_png=True); plt.close(fig)
    contracts.append({"figure": "q2_01_b1_trajectories.pdf", "result_source": "B1 raw",
                      "assertion": "B1 的独立单位是完整规模轨迹，检查点不是独立样本"})

    # 2: honest splits, observed vs predicted.
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    selected = classic_score[classic_score["split"] == "leave_one_size_out"].sort_values("rmse").iloc[0]["model"]
    for ax, split, lab in zip(axes, ["leave_one_size_out", "late_token_block"], ["留一规模", "后段 Token"]):
        sub = classic_pred[(classic_pred["model"] == selected) & (classic_pred["split"] == split)]
        ax.scatter(sub["observed"], sub["predicted"], s=10, alpha=.45)
        lo = min(sub["observed"].min(), sub["predicted"].min()); hi = max(sub["observed"].max(), sub["predicted"].max())
        ax.plot([lo, hi], [lo, hi], "--", color="#555555")
        ax.set_xlabel("观测 Loss"); ax.set_ylabel("预测 Loss"); ax.text(.03, .95, lab, transform=ax.transAxes, va="top")
    save_fig(fig, FIGURES / "q2_02_classic_holdout.pdf", also_png=True); plt.close(fig)
    contracts.append({"figure": "q2_02_classic_holdout.pdf", "result_source": "q2_classic_predictions.csv.gz",
                      "assertion": "经典律必须在按规模和时间阻断的划分上评价"})

    # 3: validation score and source transfer.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    piv = classic_score.pivot(index="model", columns="split", values="rmse")
    piv = piv.rename(index={"log_huber":"对数 Huber", "raw_huber":"原始 Huber"},
                     columns={"late_token_block":"后段 Token", "leave_one_size_out":"留一规模",
                              "random_row_diagnostic_only":"随机行（诊断）"})
    piv.plot(kind="bar", ax=axes[0]); axes[0].set_ylabel("RMSE"); axes[0].set_xlabel(""); axes[0].tick_params(axis="x", rotation=0); axes[0].legend(title="划分")
    ss = source_summary.pivot(index="dataset", columns="test", values="rmse")
    ss = ss.rename(columns={"early_calibrate_late_predict":"早期校准→后段预测",
                            "target_affine_shape_fit":"目标源样本内拟合", "uncalibrated_absolute":"未校准绝对预测"})
    ss.plot(kind="bar", ax=axes[1]); axes[1].set_yscale("log"); axes[1].set_ylabel("RMSE（对数轴）"); axes[1].set_xlabel(""); axes[1].tick_params(axis="x", rotation=0); axes[1].legend(title="跨源检验")
    save_fig(fig, FIGURES / "q2_03_validation_and_source_shift.pdf", also_png=True); plt.close(fig)
    contracts.append({"figure": "q2_03_validation_and_source_shift.pdf", "result_source": "q2_classic_validation.csv;q2_source_calibration_summary.csv",
                      "assertion": "随机行划分乐观；跨源绝对 Loss 需校准而非直拼"})

    # 4: Q model comparison.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sub = q_score[q_score["split"].isin(["grouped_ND_5fold", "B6_to_B7_new_Q_rows"])]
    pv = sub.pivot(index="model", columns="split", values="rmse").sort_values("grouped_ND_5fold")
    pv = pv.rename(index={"reducible_linear":"可约项线性", "reducible_power":"可约项幂次", "additive":"加性修正",
                          "n_power":"N 项幂次", "effective_data":"有效数据量", "none":"无 Q"},
                   columns={"B6_to_B7_new_Q_rows":"新增 Q 档", "grouped_ND_5fold":"(N,D) 分组五折"})
    pv.plot(kind="bar", ax=ax); ax.set_ylabel("RMSE"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=25); ax.legend(title="验证")
    save_fig(fig, FIGURES / "q2_04_quality_model_comparison.pdf", also_png=True); plt.close(fig)
    contracts.append({"figure": "q2_04_quality_model_comparison.pdf", "result_source": "q2_quality_model_comparison.csv",
                      "assertion": "质量函数在统一分组划分上比较，B6 与 B7 不重复计数"})

    # 5: Q OOF calibration by quality level.
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.0))
    sub = q_pred[q_pred["model"] == q_selected["selected"]]
    axes[0].scatter(sub["observed"], sub["predicted"], c=sub["Q_score"], cmap="viridis", s=15, alpha=.65)
    lo = min(sub["observed"].min(), sub["predicted"].min()); hi=max(sub["observed"].max(),sub["predicted"].max())
    axes[0].plot([lo,hi],[lo,hi],"--",color="#555555"); axes[0].set_xlabel("观测 Loss"); axes[0].set_ylabel("OOF 预测 Loss")
    byq = sub.assign(residual=sub["predicted"]-sub["observed"]).groupby("Q_score")["residual"].agg(["mean","std"])
    axes[1].errorbar(byq.index, byq["mean"], yerr=byq["std"], marker="o", capsize=3)
    axes[1].axhline(0,color="#555555",ls="--"); axes[1].set_xlabel("质量 Q"); axes[1].set_ylabel("预测残差")
    save_fig(fig, FIGURES / "q2_05_quality_oof_diagnostics.pdf", also_png=True); plt.close(fig)
    contracts.append({"figure": "q2_05_quality_oof_diagnostics.pdf", "result_source": "q2_quality_oof_predictions.csv.gz",
                      "assertion": "质量模型误差需按 Q 水平检查而非只报总拟合优度"})

    # 6: iso-loss contours / marginal utility visual.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    # Restrict to intersection of B1/B7 marginal support. This is not causal evidence.
    ng = np.geomspace(.070542, 11.965825, 80); dg = np.geomspace(10, 299.893, 90)
    nn, dd = np.meshgrid(ng, dg)
    for ax, qv in zip(axes, [.5, .9]):
        zz = q_predict(q_selected["selected"], q_selected["pars"], base, nn.ravel(), dd.ravel(), np.full(nn.size,qv)).reshape(nn.shape)
        cs=ax.contour(nn,dd,zz,levels=8); ax.clabel(cs,fontsize=7); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N（十亿参数）"); ax.set_ylabel("D（十亿 Token）"); ax.text(.04,.94,f"Q={qv}",transform=ax.transAxes,va="top")
    save_fig(fig, FIGURES / "q2_06_isoloss_quality.pdf", also_png=True); plt.close(fig)
    contracts.append({"figure": "q2_06_isoloss_quality.pdf", "result_source": "selected generalized law",
                      "assertion": "质量变化改变等 Loss 曲线，但仅在半合成 Q 证据范围内"})

    # 7: finite quality substitution.
    fig, axes = plt.subplots(1, 3, figsize=(11.0,3.8), sharey=True)
    for ax, (dv, ds) in zip(axes, marginal.groupby("D_tokens_B")):
        for qv, sub in ds.groupby("Q_score"):
            ax.plot(sub["N_params_B"], sub["equivalent_N_multiplier"], marker="o", label=f"Q={qv:g}")
        ax.set_xscale("log"); ax.set_xlabel("N（十亿参数）")
        ax.text(.03,.96,f"D={dv:g}B",transform=ax.transAxes,va="top")
    axes[0].set_ylabel("Q+0.1 的等效参数倍率"); axes[-1].legend(fontsize=8)
    save_fig(fig, FIGURES / "q2_07_quality_substitution.pdf", also_png=True); plt.close(fig)
    contracts.append({"figure": "q2_07_quality_substitution.pdf", "result_source": "q2_marginal_and_substitution.csv",
                      "assertion": "Q+0.1 的规模替代不是常数，随 N、D、Q 改变"})

    # 8: A-source p bridge across scales.
    fig, axes = plt.subplots(1, 2, figsize=(9.3,4.0))
    for scale, sub in p_detail.groupby("scale"):
        axes[0].scatter(sub["R_A"], -sub["observed_standardized_mean_loss"], s=14, alpha=.6, label=scale)
    axes[0].set_xlabel("A 组相对配比指数 $R_A(p)$"); axes[0].set_ylabel("观测相对性能（越高越好）"); axes[0].legend(fontsize=7)
    axes[1].bar(p_summary["scale"], p_summary["spearman_p_index_vs_better_loss"])
    axes[1].axhline(0,color="#555555",ls="--"); axes[1].set_ylabel("Spearman"); axes[1].tick_params(axis="x",rotation=20)
    save_fig(fig, FIGURES / "q2_08_p_bridge_scale_check.pdf", also_png=True); plt.close(fig)
    contracts.append({"figure": "q2_08_p_bridge_scale_check.pdf", "result_source": "q2_p_bridge_summary.csv;q2_p_bridge_predictions.csv",
                      "assertion": "A 组仅提供相对配比排序；B 组幅度 eta_p 未识别"})
    return pd.DataFrame(contracts).assign(evidence_level="metadata")


def main() -> None:
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    log("问题二正式实验开始；禁止读取受污染数据说明 PDF")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        b = {k: read_zip_csv(zf, B_PREFIX + v) for k, v in B_FILES.items()}
        a = {k: read_zip_csv(zf, v) for k, v in A_MEMBERS.items()}
        b3_members = sorted(x for x in zf.namelist() if x.startswith(B_PREFIX + "training_trajectories/") and x.endswith(".csv"))
        b3 = pd.concat([read_zip_csv(zf, x).assign(source_file=Path(x).name) for x in b3_members], ignore_index=True)
        bmeta = {
            "B9": read_zip_csv(zf, B_PREFIX + "supplementary_large_models.csv"),
            "B11": read_zip_csv(zf, B_PREFIX + "open_model_family_metadata.csv"),
            "B12": read_zip_csv(zf, B_PREFIX + "pythia_checkpoint_index.csv"),
        }

    # ---------- 数据审计 ----------
    inv = []
    nature = {"B1":"real_observational", "B2":"semi_synthetic", "B4":"real_observational",
              "B5":"real_observational", "B6":"semi_synthetic", "B7":"semi_synthetic",
              "B8":"mixed_semi_synthetic_extrapolated", "B10":"estimated_extrapolated"}
    for key, df in b.items():
        inv.append({"dataset": key, "rows": len(df), "columns": len(df.columns),
                    "N_unique": df["N_params_B"].nunique(), "D_unique": df["D_tokens_B"].nunique(),
                    "Q_unique": df["Q_score"].nunique() if "Q_score" in df else 0,
                    "loss_min": df["val_loss"].min(), "loss_max": df["val_loss"].max(),
                    "missing_cells": int(df.isna().sum().sum()), "evidence_level": nature[key]})
    inv.append({"dataset":"B3", "rows":len(b3), "columns":len(b3.columns),
                "N_unique":b3["N_params_B"].nunique(), "D_unique":b3["D_tokens_B"].nunique(),
                "Q_unique":0, "loss_min":b3["val_loss"].min(), "loss_max":b3["val_loss"].max(),
                "missing_cells":int(b3.isna().sum().sum()), "evidence_level":"interpolated_from_B1"})
    for key, df in bmeta.items():
        inv.append({"dataset":key, "rows":len(df), "columns":len(df.columns),
                    "N_unique":df["N_params_B"].nunique() if "N_params_B" in df else np.nan,
                    "D_unique":df["D_tokens_B"].nunique() if "D_tokens_B" in df else np.nan,
                    "Q_unique":0, "loss_min":np.nan, "loss_max":np.nan,
                    "missing_cells":int(df.isna().sum().sum()), "evidence_level":"metadata"})
    inventory = pd.DataFrame(inv)
    inventory.to_csv(RESULTS / "q2_data_inventory.csv", index=False, encoding="utf-8-sig")

    keycols = ["N_params_B", "D_tokens_B", "Q_score"]
    ov67 = b["B6"].merge(b["B7"], on=keycols, suffixes=("_B6","_B7"))
    ov78 = b["B7"].merge(b["B8"], on=keycols, suffixes=("_B7","_B8"))
    overlap = pd.DataFrame([
        {"pair":"B6_B7", "overlap_keys":len(ov67), "left_rows":len(b["B6"]), "right_rows":len(b["B7"]),
         "identical_loss":int(np.isclose(ov67["val_loss_B6"],ov67["val_loss_B7"]).sum()),
         "conflicting_loss":int((~np.isclose(ov67["val_loss_B6"],ov67["val_loss_B7"])).sum()),
         "decision":"B6 is subset; never pool B6+B7"},
        {"pair":"B7_B8", "overlap_keys":len(ov78), "left_rows":len(b["B7"]), "right_rows":len(b["B8"]),
         "identical_loss":int(np.isclose(ov78["val_loss_B7"],ov78["val_loss_B8"]).sum()),
         "conflicting_loss":int((~np.isclose(ov78["val_loss_B7"],ov78["val_loss_B8"])).sum()),
         "decision":"versioned sensitivity only; never pool as independent"},
    ])
    overlap["evidence_level"] = "metadata"
    overlap.to_csv(RESULTS / "q2_overlap_audit.csv", index=False, encoding="utf-8-sig")
    log(f"数据审计完成：B6/B7 重叠 {len(ov67)} 条；B7/B8 重叠 {len(ov78)} 条")

    # ---------- 经典标度律 ----------
    classic_score, classic_pred, classic = classical_validation(b["B1"])
    classic_score.to_csv(RESULTS / "q2_classic_validation.csv", index=False, encoding="utf-8-sig")
    classic_pred.to_csv(RESULTS / "q2_classic_predictions.csv.gz", index=False, compression="gzip")
    base = np.asarray(classic["theta"], float)
    pd.DataFrame([dict(zip(["E","A","alpha","B","beta"], base)) | {
        "model": classic["selected"], **classic["diagnostics"], "evidence_level":"real_observational"}
    ]).to_csv(RESULTS / "q2_classic_parameters.csv", index=False, encoding="utf-8-sig")
    log(f"经典律选中 {classic['selected']}，参数={base.tolist()}")

    # ---------- 跨源形状校准（不等于绝对 Loss 合并） ----------
    src_rows, src_details = [], []
    for name in ["B2", "B4", "B5", "B10"]:
        rr, dd = source_calibration(name, b[name], base)
        src_rows.extend(rr); dd.insert(0,"dataset",name); src_details.append(dd)
    source_summary = pd.DataFrame(src_rows)
    source_summary.to_csv(RESULTS / "q2_source_calibration_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(src_details, ignore_index=True).to_csv(RESULTS / "q2_source_calibration_predictions.csv.gz", index=False, compression="gzip")

    # ---------- Q 候选模型：只用 B7 主比较，B6 作嵌套训练，B8 作压力测试 ----------
    q_score, q_pred, q_selected, q_primary = q_model_comparison(b["B6"], b["B7"], b["B8"], base)
    q_score.to_csv(RESULTS / "q2_quality_model_comparison.csv", index=False, encoding="utf-8-sig")
    q_pred.to_csv(RESULTS / "q2_quality_oof_predictions.csv.gz", index=False, compression="gzip")
    q_boot = bootstrap_q(b["B7"], base, q_selected["selected"], reps=300)
    q_boot.to_csv(RESULTS / "q2_quality_bootstrap.csv", index=False, encoding="utf-8-sig")
    log(f"质量函数选中 {q_selected['selected']}，参数={q_selected['pars'].tolist()}")

    # ---------- p 的无量纲相对桥接；不估计 B 源 eta_p ----------
    p_summary, p_detail, p_meta = p_bridge(a)
    p_summary.to_csv(RESULTS / "q2_p_bridge_summary.csv", index=False, encoding="utf-8-sig")
    p_detail.to_csv(RESULTS / "q2_p_bridge_predictions.csv", index=False, encoding="utf-8-sig")

    marginal = marginal_table(base, q_selected["selected"], q_selected["pars"], q_boot)
    marginal.to_csv(RESULTS / "q2_marginal_and_substitution.csv", index=False, encoding="utf-8-sig")

    # Explicit generalized-law interface. eta_p remains a named scenario parameter, default 0.
    interface = {
        "classic_model": {"formula": "E + A*N^(-alpha) + B*D^(-beta)",
                          "objective": classic["selected"],
                          "parameters": dict(zip(["E","A","alpha","B","beta"], map(float,base)))},
        "quality_model": {"kind": q_selected["selected"], "parameters": q_selected["pars"].tolist(),
                          "scale": "B7_Q_score", "Q1_mapping": "unidentified; do not pass Q1 score",
                          "evidence": "B7 semi-synthetic grouped-(N,D) CV; B8 is pressure test only"},
        "mixture_bridge": p_meta | {"eta_p_default": 0.0,
                                    "reason": "No joint B-source p variation or A/B absolute-Loss anchor; eta_p not identifiable",
                                    "A_source_descriptive_eta_range": [float(p_summary["eta_A_within_source"].min()),
                                                                       float(p_summary["eta_A_within_source"].max())],
                                    "conditional_sensitivity_grid": [0.0, 0.25, 0.5],
                                    "warning": "Nonzero eta_p values are scenarios, not B-source estimates"},
        "generalized_law": "L=E+R_Q(N,D,Q)*exp(-eta_p*R_A(p)); eta_p=0 for identified main prediction",
        "units": {"N":"billion parameters", "D":"billion tokens", "C":"FLOPs, C=6*(N*1e9)*(D*1e9)"},
        "support": {"B1": {c: [float(b["B1"][c].min()), float(b["B1"][c].max())]
                             for c in ["N_params_B", "D_tokens_B"]},
                    "B7": {c: [float(b["B7"][c].min()), float(b["B7"][c].max())]
                             for c in ["N_params_B", "D_tokens_B", "Q_score"]}},
        "revision_protocol": "reports/16_q2_revision_contract.md; retrospective, not a new untouched test",
        "evidence_limits": ["A/B absolute Loss not pooled", "B6/B7 overlap removed",
                            "Q effect is semi-synthetic", "p-to-B amplitude is conditional, not identified"],
    }
    with (RESULTS / "q2_interface_to_q3.json").open("w", encoding="utf-8") as f:
        json.dump(interface, f, ensure_ascii=False, indent=2)

    fig_contract = make_figures(b["B1"], classic_pred, classic_score, source_summary,
                                q_score, q_pred, q_selected, base, marginal, p_summary, p_detail)
    fig_contract.to_csv(RESULTS / "q2_figure_contract.csv", index=False, encoding="utf-8-sig")

    summary = {
        "data": {r["dataset"]: int(r["rows"]) for r in inv},
        "overlap": overlap.to_dict(orient="records"),
        "classic_selected": classic["selected"],
        "classic_parameters": dict(zip(["E","A","alpha","B","beta"], map(float,base))),
        "classic_validation": classic_score.to_dict(orient="records"),
        "quality_selected": q_selected["selected"],
        "quality_parameters": q_selected["pars"].tolist(),
        "quality_grouped_cv": q_primary.to_dict(orient="records"),
        "p_bridge": p_summary.to_dict(orient="records"),
        "unidentified": ["eta_p linking A-relative mixture effect to B absolute Loss",
                           "absolute A/B Loss mapping", "real-world causal quality intervention"],
    }
    with (RESULTS / "q2_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log("问题二主实验完成")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
