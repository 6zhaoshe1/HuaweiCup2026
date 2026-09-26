#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI assistance: OpenAI Codex, OpenAI, 2026-09-25; team review required.
"""F题问题三：证据分层的资源配置优化、稳健性与图表。

正式入口。读取问题二冻结接口、联合Bootstrap、A4/B7/B9/C7原始附件；
不读取老师/DeepSeek/队友的结果数值。N、D在Loss中均用十亿单位。
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "第二十三届中国研究生数学建模竞赛 - 中文题目" / "中文题目" /
       "F题" / "real_attachments")
RES, FIG, LOGDIR = ROOT / "results", ROOT / "figures", ROOT / "code" / "outputs"
for d in (RES, FIG, LOGDIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "_模板" / "scripts"))
from mpl_cn import CYCLE, panel_label, plt, save_fig  # noqa: E402

SEED = 42
ETA = 2e-4
OFFICIAL_BUDGETS = np.array([1e19, 1e22, 1e24], float)
Q0_MAIN = 0.5
Q0_SENS = (0.1, 0.3, 0.7)
COST_FAMILIES = ("exponential", "power", "logarithmic")
EVIDENCE_SUPPORTED = "semi_synthetic_supported_joint_box"
EVIDENCE_EXTRAP = "theoretical_extrapolation_b9_guardrail"


@dataclass(frozen=True)
class Bounds:
    n_min: float
    n_max: float
    d_min: float
    d_max: float
    q_min: float = 0.1
    q_max: float = 1.0


def load_inputs():
    interface = json.loads((RES / "q2_interface_to_q3.json").read_text(encoding="utf-8"))
    order = ("E", "A", "alpha", "B", "beta")
    p = {k: float(interface["classic_model"]["parameters"][k]) for k in order}
    p["kappa"] = float(interface["quality_model"]["parameters"][0])
    boot = pd.read_csv(RES / "q2_joint_bootstrap.csv")
    boot = boot.loc[boot["success"].astype(bool), ["bootstrap_id", *order, "kappa"]].copy()
    c7_path = RAW / "C_efficiency_evolution" / "model_architecture_metadata.csv"
    c7 = pd.read_csv(c7_path)
    contexts = sorted(c7["max_position_embeddings"].dropna().astype(int).unique().tolist())
    b1_path = RAW / "B_scaling_laws" / "pythia_training_log_existing.csv"
    b1 = pd.read_csv(b1_path)
    b7_path = RAW / "B_scaling_laws" / "supplementary_NQ_experiment_expanded.csv"
    b7 = pd.read_csv(b7_path)
    b9_path = RAW / "B_scaling_laws" / "supplementary_large_models.csv"
    b9 = pd.read_csv(b9_path)
    a4_path = RAW / "A_data_value" / "regmix_tables" / "train_mixture_1m.csv"
    a4 = pd.read_csv(a4_path)
    return interface, p, boot, c7, contexts, b1, b7, b9, a4, {
        "c7": c7_path, "b1": b1_path, "b7": b7_path, "b9": b9_path, "a4": a4_path,
        "interface": RES / "q2_interface_to_q3.json", "bootstrap": RES / "q2_joint_bootstrap.csv"
    }


def g_value(q, family: str):
    q = np.asarray(q, float)
    if family == "exponential":
        return 1e7 * np.exp(6.0 * q)
    if family == "power":
        return 5e9 * q**4
    if family == "logarithmic":
        return 2e9 * np.log1p(10.0 * q)
    raise ValueError(f"unknown cost family: {family}")


def g_prime(q, family: str):
    q = np.asarray(q, float)
    if family == "exponential":
        return 6e7 * np.exp(6.0 * q)
    if family == "power":
        return 2e10 * q**3
    if family == "logarithmic":
        return 2e10 / (1.0 + 10.0 * q)
    raise ValueError(family)


def quality_increment(q, q0: float, family: str):
    return np.maximum(g_value(q, family) - g_value(q0, family), 0.0)


def predict_loss(n_b, d_b, q, params: dict, p_multiplier=1.0):
    r = params["A"] * np.asarray(n_b)**(-params["alpha"]) + params["B"] * np.asarray(d_b)**(-params["beta"])
    return params["E"] + r * (1.0 + params["kappa"] * (1.0 - np.asarray(q))) * p_multiplier


def costs(n_b: float, d_b: float, q: float, q0: float, lctx: int, family: str):
    n, d = n_b * 1e9, d_b * 1e9
    c_train = 6.0 * n * d
    c_quality = d * float(quality_increment(q, q0, family))
    c_attention = ETA * n * d * lctx
    return c_train, c_quality, c_attention


def affordable_d_b(budget: float, n_b: float, q: float, q0: float, lctx: int, family: str):
    c = budget / 1e18
    h = 6.0 + ETA * lctx
    s = float(quality_increment(q, q0, family)) / 1e9
    return c / (h * n_b + s)


def best_n_at_q(budget: float, q: float, q0: float, lctx: int, family: str,
                bounds: Bounds, params: dict):
    """Exact D elimination, bounded 1-D minimization in log N, including kinks."""
    c = budget / 1e18
    h = 6.0 + ETA * lctx
    s = float(quality_increment(q, q0, family)) / 1e9
    feasible_n_hi = (c / bounds.d_min - s) / h
    n_hi = min(bounds.n_max, feasible_n_hi)
    if not np.isfinite(n_hi) or n_hi < bounds.n_min * (1 - 1e-12):
        return None
    n_hi = max(n_hi, bounds.n_min)

    def evaluate_log_n(logn: float):
        n = math.exp(logn)
        d_aff = c / (h * n + s)
        d = min(bounds.d_max, d_aff)
        if d < bounds.d_min * (1 - 2e-10):
            return float("inf")
        return float(predict_loss(n, d, q, params))

    logs = [math.log(bounds.n_min), math.log(n_hi)]
    n_kink = (c / bounds.d_max - s) / h
    if bounds.n_min <= n_kink <= n_hi:
        logs.append(math.log(n_kink))
    if n_hi > bounds.n_min * (1 + 1e-13):
        opt = minimize_scalar(evaluate_log_n, bounds=(math.log(bounds.n_min), math.log(n_hi)),
                              method="bounded", options={"xatol": 2e-11, "maxiter": 300})
        if opt.success:
            logs.append(float(opt.x))
    vals = [(evaluate_log_n(x), x) for x in logs]
    val, logn = min(vals, key=lambda z: z[0])
    if not np.isfinite(val):
        return None
    n = math.exp(logn)
    d_aff = c / (h * n + s)
    d = min(bounds.d_max, d_aff)
    return {"N_B": n, "D_B": d, "D_affordable_B": d_aff, "Q": q, "loss": val}


def solve_scenario(budget: float, lctx: int, family: str, q0: float, bounds: Bounds,
                   params: dict, q_grid: int = 161):
    q_lo = max(q0, bounds.q_min)
    if q_lo > bounds.q_max:
        return None
    qs = np.linspace(q_lo, bounds.q_max, q_grid)
    rows = [best_n_at_q(budget, float(q), q0, lctx, family, bounds, params) for q in qs]
    valid = [(i, r) for i, r in enumerate(rows) if r is not None]
    if not valid:
        return None
    i0, best = min(valid, key=lambda z: z[1]["loss"])

    def objective_q(q):
        r = best_n_at_q(budget, float(q), q0, lctx, family, bounds, params)
        return 1e9 if r is None else r["loss"]

    candidates = [best]
    lo_i, hi_i = max(0, i0 - 1), min(len(qs) - 1, i0 + 1)
    if qs[hi_i] > qs[lo_i]:
        opt = minimize_scalar(objective_q, bounds=(float(qs[lo_i]), float(qs[hi_i])), method="bounded",
                              options={"xatol": 2e-10, "maxiter": 200})
        if opt.success:
            r = best_n_at_q(budget, float(opt.x), q0, lctx, family, bounds, params)
            if r is not None:
                candidates.append(r)
    for q in (q_lo, bounds.q_max):
        r = best_n_at_q(budget, q, q0, lctx, family, bounds, params)
        if r is not None:
            candidates.append(r)
    ans = min(candidates, key=lambda r: r["loss"])
    return enrich(ans, budget, lctx, family, q0, bounds)


def enrich(ans: dict, budget: float, lctx: int, family: str, q0: float, bounds: Bounds):
    ct, cq, ca = costs(ans["N_B"], ans["D_B"], ans["Q"], q0, lctx, family)
    total = ct + cq + ca
    tol_rel = 3e-7
    n, d, q = ans["N_B"], ans["D_B"], ans["Q"]
    def at(v, b): return abs(v - b) <= max(1e-9, tol_rel * max(abs(b), 1.0))
    budget_active = abs(total - budget) <= max(1.0, 3e-9 * budget)
    active = []
    if at(n, bounds.n_min): active.append("N_lower")
    if at(n, bounds.n_max): active.append("N_upper")
    if at(d, bounds.d_min): active.append("D_lower")
    if at(d, bounds.d_max): active.append("D_upper")
    if at(q, max(q0, bounds.q_min)): active.append("Q_baseline")
    elif at(q, bounds.q_max): active.append("Q_upper")
    else: active.append("Q_interior")
    active.append("budget_active" if budget_active else "budget_slack")
    r = dict(ans)
    r.update({
        "budget_FLOPs": budget, "context_tokens": lctx, "cost_family": family, "Q0": q0,
        "C_train": ct, "C_quality": cq, "C_attention": ca, "C_total": total,
        "budget_utilization": total / budget, "budget_residual_FLOPs": budget - total,
        "train_share": ct / total, "quality_share": cq / total, "attention_share": ca / total,
        "attention_to_train": ca / ct, "active_set": ";".join(active),
        "N_at_lower": at(n, bounds.n_min), "N_at_upper": at(n, bounds.n_max),
        "D_at_lower": at(d, bounds.d_min), "D_at_upper": at(d, bounds.d_max),
        "Q_at_baseline": at(q, max(q0, bounds.q_min)), "Q_at_upper": at(q, bounds.q_max),
        "budget_active": budget_active,
    })
    return r


def fixed_q_baseline(budget, lctx, family, q0, bounds, params):
    r = best_n_at_q(budget, q0, q0, lctx, family, bounds, params)
    return None if r is None else enrich(r, budget, lctx, family, q0, bounds)


def de_validate(row: dict, bounds: Bounds, params: dict):
    budget, lctx, family, q0 = row["budget_FLOPs"], int(row["context_tokens"]), row["cost_family"], row["Q0"]
    qlo = max(q0, bounds.q_min)
    def obj(x):
        n, q = math.exp(float(x[0])), float(x[1])
        d = min(bounds.d_max, affordable_d_b(budget, n, q, q0, lctx, family))
        if d < bounds.d_min:
            return 1e4 + 1e3 * (bounds.d_min - d)
        return float(predict_loss(n, d, q, params))
    out = differential_evolution(obj, [(math.log(bounds.n_min), math.log(bounds.n_max)), (qlo, bounds.q_max)],
                                 seed=SEED, popsize=12, maxiter=160, tol=1e-10, polish=True, workers=1)
    n, q = math.exp(out.x[0]), out.x[1]
    d = min(bounds.d_max, affordable_d_b(budget, n, q, q0, lctx, family))
    return {"de_loss": float(out.fun), "de_N_B": n, "de_D_B": d, "de_Q": q,
            "loss_gap_de_minus_main": float(out.fun - row["loss"]), "de_success": bool(out.success)}


def _scaled_nearest(point: np.ndarray, cloud: np.ndarray):
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    distances = np.linalg.norm((cloud - point) / scale, axis=1)
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])


def support_geometry(official: pd.DataFrame, b1_raw: pd.DataFrame, b7_raw: pd.DataFrame,
                     bounds: Bounds):
    """Classify solutions by actual B1/B7 designs, not marginal ranges alone."""
    b1_cols = ["N_params_B", "D_tokens_B"]
    b7_cols = ["N_params_B", "D_tokens_B", "Q_score"]
    b1_points = b1_raw[b1_cols].drop_duplicates().to_numpy(float)
    b7_points = b7_raw[b7_cols].drop_duplicates().to_numpy(float)
    b1_cloud = np.column_stack([np.log(b1_points[:, 0]), np.log(b1_points[:, 1])])
    b7_cloud = np.column_stack([np.log(b7_points[:, 0]), np.log(b7_points[:, 1]), b7_points[:, 2]])
    b1_cartesian = len(b1_points) == b1_raw.N_params_B.nunique() * b1_raw.D_tokens_B.nunique()
    b7_cartesian = (len(b7_points) == b7_raw.N_params_B.nunique() * b7_raw.D_tokens_B.nunique()
                    * b7_raw.Q_score.nunique())
    rows = []
    for idx, row in official.iterrows():
        base = {"official_row": int(idx), "budget_FLOPs": row.budget_FLOPs,
                "context_tokens": int(row.context_tokens), "cost_family": row.cost_family,
                "evidence_level": row.evidence_level, "feasible": bool(row.feasible),
                "B1_design_is_cartesian": b1_cartesian, "B7_design_is_cartesian": b7_cartesian}
        if not bool(row.feasible):
            rows.append({**base, "overall_evidence_tier": "I_infeasible_under_frozen_constraints"})
            continue
        n, d, q = float(row.N_B), float(row.D_B), float(row.Q)
        ib1, dist1 = _scaled_nearest(np.array([np.log(n), np.log(d)]), b1_cloud)
        ib7, dist7 = _scaled_nearest(np.array([np.log(n), np.log(d), q]), b7_cloud)
        exact1 = bool(np.any(np.all(np.isclose(b1_points, [n, d], rtol=0, atol=1e-10), axis=1)))
        exact7 = bool(np.any(np.all(np.isclose(b7_points, [n, d, q], rtol=0, atol=1e-10), axis=1)))
        def within(value, values):
            lo, hi = float(np.min(values)), float(np.max(values))
            tol = max(1e-12, 1e-10 * max(abs(lo), abs(hi), 1.0))
            return lo - tol <= value <= hi + tol
        inside1 = within(n, b1_points[:, 0]) and within(d, b1_points[:, 1])
        inside7 = within(n, b7_points[:, 0]) and within(d, b7_points[:, 1]) and within(q, b7_points[:, 2])
        active = str(row.active_set)
        touches_support = any(flag in active for flag in ("N_lower", "N_upper", "D_lower", "D_upper", "Q_upper"))
        if row.evidence_level == EVIDENCE_EXTRAP or not (inside1 and inside7):
            tier = "D_theoretical_extrapolation"
        elif touches_support:
            tier = "C_supported_boundary_scenario"
        elif exact1 and exact7:
            tier = "A_observed_joint_anchor"
        else:
            tier = "B_supported_factorial_interpolation"
        rows.append({**base, "N_B": n, "D_B": d, "Q": q,
                     "B1_geometry": "observed_checkpoint" if exact1 else ("factorial_hull_interpolation" if inside1 else "outside_hull"),
                     "B7_geometry": "observed_grid_point" if exact7 else ("factorial_hull_interpolation" if inside7 else "outside_hull"),
                     "nearest_B1_scaled_log_distance": dist1,
                     "nearest_B7_scaled_logQ_distance": dist7,
                     "nearest_B1_N_B": b1_points[ib1, 0], "nearest_B1_D_B": b1_points[ib1, 1],
                     "nearest_B7_N_B": b7_points[ib7, 0], "nearest_B7_D_B": b7_points[ib7, 1],
                     "nearest_B7_Q": b7_points[ib7, 2], "touches_support_boundary": touches_support,
                     "overall_evidence_tier": tier})
    return pd.DataFrame(rows)


def quality_boundary_diagnostics(central: pd.DataFrame, bounds: Bounds, params: dict):
    """Profile derivative d min_{N,D}L / dQ and its benefit/cost decomposition."""
    rows = []
    for _, row in central.iterrows():
        budget, lctx, family, q0 = float(row.budget_FLOPs), int(row.context_tokens), row.cost_family, float(row.Q0)
        q, qlo, qhi, eps = float(row.Q), max(q0, bounds.q_min), bounds.q_max, 1e-4

        def profile(qq):
            ans = best_n_at_q(budget, float(qq), q0, lctx, family, bounds, params)
            return np.nan if ans is None else float(ans["loss"])

        if bool(row.Q_at_upper):
            derivative = (profile(q) - profile(max(qlo, q - eps))) / min(eps, q - qlo)
            derivative_side = "left"
        elif bool(row.Q_at_baseline):
            derivative = (profile(min(qhi, q + eps)) - profile(q)) / min(eps, qhi - q)
            derivative_side = "right"
        else:
            derivative = (profile(q + eps) - profile(q - eps)) / (2 * eps)
            derivative_side = "central"
        reducible = params["A"] * row.N_B**(-params["alpha"]) + params["B"] * row.D_B**(-params["beta"])
        direct_benefit = params["kappa"] * reducible
        displacement = derivative + direct_benefit
        if abs(displacement) < 1e-9:
            displacement = 0.0
        if bool(row.Q_at_upper) and not bool(row.budget_active):
            cause = "support_caps_with_budget_slack"
        elif bool(row.Q_at_upper) and derivative < 0:
            cause = "Q_upper_binds_net_benefit_remains"
        elif bool(row.Q_at_baseline) and derivative >= 0:
            cause = "marginal_cost_exceeds_benefit_at_Q0"
        elif not bool(row.Q_at_upper) and not bool(row.Q_at_baseline):
            cause = "interior_marginal_balance"
        else:
            cause = "boundary_requires_case_review"
        rows.append({"budget_FLOPs": budget, "context_tokens": lctx, "cost_family": family, "Q0": q0,
                     "N_B": row.N_B, "D_B": row.D_B, "Q": q, "active_set": row.active_set,
                     "g_prime_FLOPs_per_token_per_Q": float(g_prime(q, family)),
                     "direct_quality_benefit_per_Q": direct_benefit,
                     "resource_displacement_penalty_per_Q": displacement,
                     "profile_dLoss_dQ": derivative, "derivative_side": derivative_side,
                     "benefit_to_penalty_ratio": direct_benefit / displacement if displacement > 1e-14 else np.inf,
                     "diagnostic_cause": cause, "Q_effect_evidence": "B7_semi_synthetic",
                     "Q0_status": "external_scenario_not_estimated"})
    return pd.DataFrame(rows)


def mixture_scores(a4: pd.DataFrame, interface: dict):
    m = interface["mixture_bridge"]
    names = m["domain_names"]
    cols = [f"train_the_pile_{x}" for x in names]
    p_raw = a4[cols].to_numpy(float)
    sums = p_raw.sum(axis=1)
    # A4 shares are published to three decimals: row sums range 0.996--1.003.
    # Preserve and report the discrepancy, then close each composition exactly.
    if (sums <= 0).any() or np.max(np.abs(sums - 1.0)) > 0.005:
        raise AssertionError("A4 mixture deviation exceeds documented rounding tolerance")
    p = p_raw / sums[:, None]
    yh = np.asarray(m["intercepts"])[None, :] + np.log(p + float(m["delta"])) @ np.asarray(m["coefficients"]).T
    raw = ((yh - np.asarray(m["output_mean"])) / np.asarray(m["output_sd"])).mean(axis=1)
    r_a = -(raw - float(m["index_center"])) / float(m["index_scale"])
    out = a4[["index", *cols]].copy()
    out["original_share_sum"] = sums
    out["closure_factor"] = 1.0 / sums
    out.loc[:, cols] = p
    out["R_A"] = r_a
    out["rank_desc"] = pd.Series(r_a).rank(ascending=False, method="min").astype(int)
    out["evidence_level"] = "A_source_supported_recipe_ranking"
    return out.sort_values("rank_desc").reset_index(drop=True)


def detect_transitions(paths: pd.DataFrame, persistence=3):
    rows = []
    keys = ["cost_family", "context_tokens", "evidence_level"]
    for key, g in paths.groupby(keys, sort=False):
        g = g.sort_values("budget_FLOPs").reset_index(drop=True)
        states = g["active_set"].tolist()
        for i in range(1, len(g)):
            if states[i] == states[i-1] or i + persistence > len(g):
                continue
            if all(states[j] == states[i] for j in range(i, i+persistence)):
                rows.append({
                    **dict(zip(keys, key)), "budget_before_FLOPs": g.loc[i-1, "budget_FLOPs"],
                    "budget_after_FLOPs": g.loc[i, "budget_FLOPs"], "old_active_set": states[i-1],
                    "new_active_set": states[i], "persistence_points": persistence,
                    "boundary_driven": any(x in states[i] for x in ("N_upper", "D_upper", "Q_upper")),
                    "C_total_after": g.loc[i, "C_total"], "budget_utilization_after": g.loc[i, "budget_utilization"]
                })
    return pd.DataFrame(rows)


def bootstrap_optimize(boot: pd.DataFrame, scenarios: pd.DataFrame, bounds: Bounds):
    rows = []
    for _, s in scenarios.iterrows():
        for _, b in boot.iterrows():
            params = {k: float(b[k]) for k in ("E", "A", "alpha", "B", "beta", "kappa")}
            r = solve_scenario(float(s.budget_FLOPs), int(s.context_tokens), s.cost_family, float(s.Q0),
                               bounds, params, q_grid=61)
            if r is None:
                rows.append({"bootstrap_id": int(b.bootstrap_id), "success": False,
                             "budget_FLOPs": s.budget_FLOPs, "cost_family": s.cost_family})
                continue
            rows.append({"bootstrap_id": int(b.bootstrap_id), "success": True,
                         "budget_FLOPs": s.budget_FLOPs, "context_tokens": s.context_tokens,
                         "cost_family": s.cost_family, "Q0": s.Q0, "N_B": r["N_B"], "D_B": r["D_B"],
                         "Q": r["Q"], "loss": r["loss"], "active_set": r["active_set"]})
    return pd.DataFrame(rows)


def robust_config(boot: pd.DataFrame, scenario: pd.Series, bounds: Bounds):
    pars = boot[["E", "A", "alpha", "B", "beta", "kappa"]].to_numpy(float)
    budget, lctx, family, q0 = float(scenario.budget_FLOPs), int(scenario.context_tokens), scenario.cost_family, float(scenario.Q0)
    qlo = max(q0, bounds.q_min)

    def losses_at(x):
        n, q = math.exp(float(x[0])), float(x[1])
        d = min(bounds.d_max, affordable_d_b(budget, n, q, q0, lctx, family))
        if d < bounds.d_min:
            return None, n, d, q
        vals = pars[:, 0] + (pars[:, 1] * n**(-pars[:, 2]) + pars[:, 3] * d**(-pars[:, 4])) * (1 + pars[:, 5] * (1-q))
        return vals, n, d, q

    def obj(x):
        vals, _, d, _ = losses_at(x)
        if vals is None:
            return 1e4 + 1e3 * max(bounds.d_min-d, 0)
        return float(np.quantile(vals, .9))

    de = differential_evolution(obj, [(math.log(bounds.n_min), math.log(bounds.n_max)), (qlo, bounds.q_max)],
                                seed=SEED, popsize=10, maxiter=100, tol=2e-9, polish=True, workers=1)
    vals, n, d, q = losses_at(de.x)
    return {"N_B": n, "D_B": d, "Q": q, "p90_loss": float(np.quantile(vals, .9)),
            "median_loss": float(np.median(vals)), "mean_loss": float(np.mean(vals)), "success": bool(de.success)}


def build_figures(official: pd.DataFrame, paths: pd.DataFrame, boot_sum: pd.DataFrame,
                  robust: pd.DataFrame, mix: pd.DataFrame, mix_scen: pd.DataFrame, cost_curve: pd.DataFrame,
                  qdiag: pd.DataFrame, support: pd.DataFrame):
    contracts = []
    def save(fig, stem, claim, source, transform, assertion):
        pdf = FIG / f"{stem}.pdf"
        save_fig(fig, str(pdf), also_png=True)
        plt.close(fig)
        contracts.append({"figure": stem, "claim": claim, "source": source, "transform": transform,
                          "assertion": assertion, "pdf": str(pdf.relative_to(ROOT)),
                          "png": str(pdf.with_suffix('.png').relative_to(ROOT))})

    # 1 budget paths at median context, supported main
    sub = paths[(paths.context_tokens == 4096) & (paths.evidence_level == EVIDENCE_SUPPORTED)]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    for fam in COST_FAMILIES:
        g = sub[sub.cost_family == fam]
        axes[0].plot(g.budget_FLOPs, g.N_B, label=fam)
        axes[1].plot(g.budget_FLOPs, g.D_B, label=fam)
        axes[2].plot(g.budget_FLOPs, g.Q, label=fam)
    for ax, lab, letter in zip(axes, ["N（十亿参数）", "D（十亿Token）", "Q"], ["(a)", "(b)", "(c)"]):
        ax.set_xscale("log"); ax.set_ylabel(lab); panel_label(ax, letter, dx=-.10)
    axes[0].set_yscale("log"); axes[1].set_yscale("log"); axes[2].set_xlabel("算力预算（FLOPs）")
    axes[0].legend(ncol=3, loc="best")
    save(fig, "q3_01_budget_paths", "预算变化下N/D/Q数值最优轨迹", "results/q3_budget_paths.csv",
         "筛选4096上下文和支持域，按成本族连线", "所有点来自可行优化行")

    # 2 cost shares, official central
    sub = official[(official.evidence_level == EVIDENCE_SUPPORTED) & (official.context_tokens == 4096)].copy()
    sub["label"] = sub.cost_family.str.slice(0, 3) + "\n" + sub.budget_FLOPs.map(lambda x: f"1e{int(np.log10(x))}")
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(sub)); bottom = np.zeros(len(sub))
    for col, lab, color in [("train_share", "基础训练", CYCLE[0]), ("quality_share", "质量提升", CYCLE[1]),
                            ("attention_share", "注意力", CYCLE[2])]:
        ax.bar(x, sub[col], bottom=bottom, label=lab, color=color); bottom += sub[col].to_numpy()
    ax.set_xticks(x, sub.label); ax.set_ylabel("成本占比"); ax.set_ylim(0, 1.02); ax.legend(ncol=3)
    save(fig, "q3_02_cost_shares", "同预算口径下三项成本分解", "results/q3_official_solutions.csv",
         "筛选支持域、4096上下文、Q0=0.5", "每根柱三项之和为1")

    # 3 context sensitivity
    sub = official[(official.evidence_level == EVIDENCE_SUPPORTED) & (official.cost_family == "exponential")]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for budget, g in sub.groupby("budget_FLOPs"):
        ax.plot(g.context_tokens, g.loss, marker="o", label=f"1e{int(np.log10(budget))}")
    ax.axvline(30000, color="#666666", ls="--", lw=1.1, label="成本临界30000")
    ax.set_xscale("log", base=2); ax.set_xlabel("上下文长度"); ax.set_ylabel("预测Loss"); ax.legend()
    save(fig, "q3_03_context_sensitivity", "上下文只进入成本时的配置敏感性", "results/q3_official_solutions.csv",
         "指数成本、支持域、按预算连线", "不解释为上下文能力收益")

    # 4 supported vs extrapolation
    sub = official[(official.context_tokens == 4096) & (official.cost_family == "exponential")]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for ev, g in sub.groupby("evidence_level"):
        lab = "支持域" if ev == EVIDENCE_SUPPORTED else "理论外推"
        axes[0].plot(g.budget_FLOPs, g.loss, marker="o", label=lab)
        axes[1].plot(g.budget_FLOPs, g.max_extrap_factor, marker="o", label=lab)
    for ax in axes: ax.set_xscale("log"); ax.set_xlabel("算力预算（FLOPs）")
    axes[0].set_ylabel("预测Loss"); axes[1].set_ylabel("超出支持上界最大倍数"); axes[1].set_yscale("log")
    axes[0].legend(); panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)")
    save(fig, "q3_04_support_vs_extrapolation", "高预算结果的支持域边界效应", "results/q3_official_solutions.csv",
         "4096上下文、指数成本，按证据层比较", "外推倍数由N/D相对B1-B7上界重算")

    # 5 bootstrap intervals
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.5))
    for ax, v in zip(axes, ["N_B", "D_B", "Q"]):
        for fam in COST_FAMILIES:
            g = boot_sum[boot_sum.cost_family == fam].sort_values("budget_FLOPs")
            ax.errorbar(g.budget_FLOPs, g[f"{v}_median"],
                        yerr=[g[f"{v}_median"]-g[f"{v}_q025"], g[f"{v}_q975"]-g[f"{v}_median"]],
                        marker="o", capsize=2, label=fam)
        ax.set_xscale("log"); ax.set_xlabel("预算"); ax.set_ylabel(v)
    axes[0].set_yscale("log"); axes[1].set_yscale("log"); axes[0].legend(fontsize=8)
    save(fig, "q3_05_bootstrap_config", "问题二联合参数不确定性下的最优配置区间", "results/q3_bootstrap_summary.csv",
         "按预算和成本族汇总300次重优化的2.5/50/97.5分位", "区间不含结构和成本不确定性")

    # 6 robust regret.  Most scenarios are numerically indistinguishable from
    # zero, so a linear bar chart hides the actual order-of-magnitude result.
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    r = robust.copy(); xx = np.arange(len(r)); floor = 1e-12
    y_nom = np.maximum(r.nominal_p90_regret.to_numpy(float), floor)
    y_rob = np.maximum(r.robust_p90_regret.to_numpy(float), floor)
    ax.scatter(xx-.12, y_nom, s=34, label="名义解", color=CYCLE[0], zorder=3)
    ax.scatter(xx+.12, y_rob, s=34, marker="D", label="90%分位稳健解", color=CYCLE[1], zorder=3)
    for i, (yn, yr) in enumerate(zip(y_nom, y_rob)):
        ax.plot([i-.12, i+.12], [yn, yr], color="#BBBBBB", lw=.8, zorder=1)
    ax.set_xticks(xx, r.cost_family.str.slice(0,3)+"\n"+r.budget_FLOPs.map(lambda x:f"1e{int(np.log10(x))}"))
    ax.set_yscale("log"); ax.set_ylim(floor/1.8, max(y_nom.max(), y_rob.max())*2.2)
    ax.set_ylabel("Bootstrap后悔值90%分位（对数轴）"); ax.legend()
    ax.text(.01, .02, "小于 $10^{-12}$ 的数值误差置于图示下限", transform=ax.transAxes, fontsize=7, color="#666666")
    save(fig, "q3_06_robust_regret", "名义与风险分位配置的尾部后悔", "results/q3_robust_configs.csv",
         "同一300组参数情景下计算逐样本后悔的90%分位", "风险偏好为情景而非估计参数")

    # 7 quality cost curves
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for fam in COST_FAMILIES:
        g = cost_curve[cost_curve.cost_family == fam]
        axes[0].plot(g.Q, g.delta_g, label=fam); axes[1].plot(g.Q, g.g_prime, label=fam)
    axes[0].set_xlabel("Q"); axes[0].set_ylabel("增量成本（FLOPs/Token）"); axes[0].set_yscale("symlog", linthresh=1e6)
    axes[1].set_xlabel("Q"); axes[1].set_ylabel("边际成本 g'(Q)"); axes[1].set_yscale("log"); axes[0].legend()
    panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)")
    save(fig, "q3_07_quality_cost_curves", "三种题给质量成本的水平与边际差异", "results/q3_quality_cost_curves.csv",
         "Q0=0.5下增量成本及解析导数", "仅表示题给成本假设")

    # 8 mixture ranking and conditional multiplier
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].hist(mix.R_A, bins=30, color=CYCLE[0], alpha=.85); axes[0].set_xlabel("A源相对配比指数 R_A"); axes[0].set_ylabel("历史配方数")
    ms = mix_scen[(mix_scen.budget_FLOPs == 1e22) & (mix_scen.eta_p > 0)]
    for eta, g in ms.groupby("eta_p"):
        axes[1].plot(g.recipe_role, g.conditional_loss, marker="o", label=f"eta={eta:g}")
    axes[1].set_ylabel("条件情景Loss"); axes[1].set_xlabel("A4历史配方角色"); axes[1].legend()
    panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)")
    save(fig, "q3_08_mixture_scenarios", "A4配方只支持排序，非零幅度仅为条件情景", "results/q3_mixture_ranking.csv; results/q3_mixture_scenarios.csv",
         "直方图+1e22指数成本下三类历史配方条件Loss", "eta=0时配方不改变主Loss")

    # 9 why Q hits a boundary, and how far solutions are from observed grids.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    qd = qdiag.sort_values(["budget_FLOPs", "cost_family"]).copy()
    labels = qd.cost_family.str.slice(0, 3) + "\n" + qd.budget_FLOPs.map(lambda x: f"1e{int(np.log10(x))}")
    xx = np.arange(len(qd)); w = .36
    axes[0].bar(xx-w/2, qd.direct_quality_benefit_per_Q, width=w, label="固定N,D的质量收益", color=CYCLE[0])
    axes[0].bar(xx+w/2, qd.resource_displacement_penalty_per_Q, width=w, label="资源挤占代价", color=CYCLE[1])
    axes[0].set_xticks(xx, labels); axes[0].set_ylabel("每单位Q的Loss边际量"); axes[0].legend(fontsize=8)
    sg = support[(support.context_tokens == 4096) & (support.evidence_level == EVIDENCE_SUPPORTED) & support.feasible].copy()
    sg = sg.sort_values(["budget_FLOPs", "cost_family"]); xx2 = np.arange(len(sg))
    axes[1].scatter(xx2-.10, sg.nearest_B1_scaled_log_distance, label="到B1最近观测点", color=CYCLE[0])
    axes[1].scatter(xx2+.10, sg.nearest_B7_scaled_logQ_distance, marker="D", label="到B7最近网格点", color=CYCLE[2])
    axes[1].set_xticks(xx2, sg.cost_family.str.slice(0,3)+"\n"+sg.budget_FLOPs.map(lambda x:f"1e{int(np.log10(x))}"))
    axes[1].set_ylabel("归一化最近邻距离"); axes[1].legend(fontsize=8)
    panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)")
    save(fig, "q3_09_quality_support_diagnostics", "Q边界成因与联合支持距离诊断",
         "results/q3_quality_boundary_diagnostics.csv; results/q3_solution_evidence.csv",
         "4096上下文官方预算；边际收益/挤占分解与B1/B7实际设计最近距离",
         "距离和边际诊断不构成现实因果或严格全局最优证明")

    pd.DataFrame(contracts).to_csv(RES / "q3_figure_contract.csv", index=False, encoding="utf-8-sig")


def main():
    t0 = time.time()
    interface, params, boot, c7, contexts, b1_raw, b7, b9, a4, input_paths = load_inputs()
    if contexts != [2048, 4096, 8192, 32768, 131072]:
        raise AssertionError(f"unexpected C7 contexts: {contexts}")
    if not {"N_params_B", "D_tokens_B", "Q_score"}.issubset(b7.columns):
        raise AssertionError("B7 schema changed")
    b1sup = interface["support"]["B1"]; b7sup = interface["support"]["B7"]
    supported = Bounds(max(b1sup["N_params_B"][0], b7sup["N_params_B"][0]),
                       min(b1sup["N_params_B"][1], b7sup["N_params_B"][1]),
                       max(b1sup["D_tokens_B"][0], b7sup["D_tokens_B"][0]),
                       min(b1sup["D_tokens_B"][1], b7sup["D_tokens_B"][1]),
                       b7sup["Q_score"][0], b7sup["Q_score"][1])
    extrap = Bounds(supported.n_min, float(b9.N_params_B.max()), supported.d_min,
                    float(b9.D_tokens_B.max()), supported.q_min, supported.q_max)
    if supported.d_min != 10.0:
        raise AssertionError("joint B1/B7 D lower support must be 10B")

    # Input and unit audit
    audit = [
        {"item":"Q2 interface","rows":1,"role":"frozen model","fact":interface["generalized_law"],"path":str(input_paths["interface"])},
        {"item":"Q2 bootstrap","rows":len(boot),"role":"conditional parameter uncertainty","fact":"joint successful refits","path":str(input_paths["bootstrap"])},
        {"item":"B1","rows":len(b1_raw),"role":"real observational N-D trajectories","fact":f"N={b1_raw.N_params_B.nunique()},D={b1_raw.D_tokens_B.nunique()},Cartesian={len(b1_raw.drop_duplicates(['N_params_B','D_tokens_B'])) == b1_raw.N_params_B.nunique()*b1_raw.D_tokens_B.nunique()}","path":str(input_paths["b1"])},
        {"item":"B7","rows":len(b7),"role":"semi-synthetic Q support","fact":f"N={b7.N_params_B.nunique()},D={b7.D_tokens_B.nunique()},Q={b7.Q_score.nunique()}","path":str(input_paths["b7"])},
        {"item":"B9","rows":len(b9),"role":"extrapolation guardrail metadata","fact":f"Nmax={b9.N_params_B.max()},Dmax={b9.D_tokens_B.max()}","path":str(input_paths["b9"])},
        {"item":"C7","rows":len(c7),"role":"external context scenarios","fact":str(contexts),"path":str(input_paths["c7"])},
        {"item":"A4","rows":len(a4),"role":"supported recipe candidates","fact":"17 shares sum to one","path":str(input_paths["a4"])},
    ]
    pd.DataFrame(audit).to_csv(RES / "q3_input_audit.csv", index=False, encoding="utf-8-sig")
    unit_rows = []
    for l in contexts:
        unit_rows.append({"context_tokens":l,"eta":ETA,"attention_to_train":ETA*l/6,
                          "critical_context":6/ETA,"relation_to_critical":"below" if l<6/ETA else "above"})
    pd.DataFrame(unit_rows).to_csv(RES / "q3_cost_unit_audit.csv", index=False, encoding="utf-8-sig")

    # Official solutions: supported and theoretical extrapolation, central Q0.
    official_rows = []
    for ev, bounds in ((EVIDENCE_SUPPORTED, supported), (EVIDENCE_EXTRAP, extrap)):
        for budget in OFFICIAL_BUDGETS:
            for lctx in contexts:
                for fam in COST_FAMILIES:
                    r = solve_scenario(float(budget), lctx, fam, Q0_MAIN, bounds, params)
                    if r is None:
                        official_rows.append({"budget_FLOPs":budget,"context_tokens":lctx,"cost_family":fam,
                                              "Q0":Q0_MAIN,"evidence_level":ev,"feasible":False})
                        continue
                    r.update({"evidence_level":ev,"feasible":True,
                              "N_extrap_factor":max(r["N_B"]/supported.n_max,1.0),
                              "D_extrap_factor":max(r["D_B"]/supported.d_max,1.0)})
                    r["max_extrap_factor"] = max(r["N_extrap_factor"], r["D_extrap_factor"])
                    official_rows.append(r)
    official = pd.DataFrame(official_rows)
    support = support_geometry(official, b1_raw, b7, supported)
    add_cols = ["B1_design_is_cartesian", "B7_design_is_cartesian", "B1_geometry", "B7_geometry",
                "nearest_B1_scaled_log_distance", "nearest_B7_scaled_logQ_distance",
                "touches_support_boundary", "overall_evidence_tier"]
    official = official.join(support.set_index("official_row")[add_cols])
    official.to_csv(RES / "q3_official_solutions.csv", index=False, encoding="utf-8-sig")
    support.to_csv(RES / "q3_solution_evidence.csv", index=False, encoding="utf-8-sig")

    # Feasible baselines and Q0 sensitivity at the C7 sample median context.
    median_ctx = int(c7.max_position_embeddings.median())
    baseline_rows, q0_rows = [], []
    for budget in OFFICIAL_BUDGETS:
        for fam in COST_FAMILIES:
            base = fixed_q_baseline(float(budget), median_ctx, fam, Q0_MAIN, supported, params)
            joint = solve_scenario(float(budget), median_ctx, fam, Q0_MAIN, supported, params)
            for typ, r in (("fixed_Q0", base), ("joint_NDQ", joint)):
                if r:
                    r = dict(r); r.update({"strategy":typ,"evidence_level":EVIDENCE_SUPPORTED})
                    baseline_rows.append(r)
            for q0 in Q0_SENS:
                r = solve_scenario(float(budget), median_ctx, fam, q0, supported, params)
                if r:
                    r.update({"evidence_level":EVIDENCE_SUPPORTED,"Q0_role":"external_sensitivity"})
                    q0_rows.append(r)
    baselines = pd.DataFrame(baseline_rows)
    baselines.to_csv(RES / "q3_feasible_baselines.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(q0_rows).to_csv(RES / "q3_q0_sensitivity.csv", index=False, encoding="utf-8-sig")

    # Continuous budget paths and pre-registered active-set transitions.
    grid = np.unique(np.r_[np.geomspace(1e19, 1e24, 121), OFFICIAL_BUDGETS])
    path_rows = []
    for lctx in (median_ctx, 32768):
        for fam in COST_FAMILIES:
            for budget in grid:
                r = solve_scenario(float(budget), lctx, fam, Q0_MAIN, supported, params, q_grid=101)
                if r:
                    r.update({"evidence_level":EVIDENCE_SUPPORTED})
                    path_rows.append(r)
    paths = pd.DataFrame(path_rows)
    paths.to_csv(RES / "q3_budget_paths.csv", index=False, encoding="utf-8-sig")
    transitions = detect_transitions(paths, persistence=3)
    transitions.to_csv(RES / "q3_transitions.csv", index=False, encoding="utf-8-sig")

    # Independent numerical solver route for 9 central official supported scenarios.
    central = official[(official.evidence_level == EVIDENCE_SUPPORTED) & (official.context_tokens == median_ctx)].copy()
    qdiag = quality_boundary_diagnostics(central, supported, params)
    qdiag.to_csv(RES / "q3_quality_boundary_diagnostics.csv", index=False, encoding="utf-8-sig")
    val_rows = []
    for _, row in central.iterrows():
        v = de_validate(row.to_dict(), supported, params)
        val_rows.append({"budget_FLOPs":row.budget_FLOPs,"context_tokens":median_ctx,"cost_family":row.cost_family,
                         "main_loss":row.loss,"main_N_B":row.N_B,"main_D_B":row.D_B,"main_Q":row.Q,**v})
    validation = pd.DataFrame(val_rows)
    validation.to_csv(RES / "q3_solver_validation.csv", index=False, encoding="utf-8-sig")

    # Mixture supported ranking and conditional eta scenarios.
    mix = mixture_scores(a4, interface)
    mix.to_csv(RES / "q3_mixture_ranking.csv", index=False, encoding="utf-8-sig")
    roles = {"best_supported": mix.iloc[0], "median_supported": mix.iloc[len(mix)//2], "worst_supported": mix.iloc[-1]}
    mix_rows = []
    for _, row in central[central.cost_family == "exponential"].iterrows():
        for role, rec in roles.items():
            for eta_p in (0.0, 0.25, 0.5):
                mult = math.exp(-eta_p * float(rec.R_A))
                cond = params["E"] + (float(row.loss)-params["E"]) * mult
                mix_rows.append({"budget_FLOPs":row.budget_FLOPs,"context_tokens":median_ctx,"cost_family":"exponential",
                                 "recipe_role":role,"recipe_index":int(rec["index"]),"R_A":rec.R_A,"eta_p":eta_p,
                                 "multiplier":mult,"N_B":row.N_B,"D_B":row.D_B,"Q":row.Q,
                                 "main_loss_eta0":row.loss,"conditional_loss":cond,
                                 "argmin_NDQ_unchanged":True,"evidence_level":"conditional_model_scenario"})
    mix_scen = pd.DataFrame(mix_rows)
    mix_scen.to_csv(RES / "q3_mixture_scenarios.csv", index=False, encoding="utf-8-sig")

    # Cost curves.
    qcurve = np.linspace(.1, 1, 181)
    curve_rows = []
    for fam in COST_FAMILIES:
        for q in qcurve:
            curve_rows.append({"cost_family":fam,"Q0":Q0_MAIN,"Q":q,
                               "g":float(g_value(q,fam)),"delta_g":float(quality_increment(q,Q0_MAIN,fam)),
                               "g_prime":float(g_prime(q,fam))})
    cost_curve = pd.DataFrame(curve_rows)
    cost_curve.to_csv(RES / "q3_quality_cost_curves.csv", index=False, encoding="utf-8-sig")

    # Conditional parameter uncertainty and robust candidates at official budgets, median C7 context.
    boot_scen = central[["budget_FLOPs","context_tokens","cost_family","Q0"]].copy()
    boot_opt = bootstrap_optimize(boot, boot_scen, supported)
    boot_opt.to_csv(RES / "q3_bootstrap_optima.csv.gz", index=False, compression="gzip")
    summary_rows = []
    for key, g in boot_opt[boot_opt.success].groupby(["budget_FLOPs","context_tokens","cost_family","Q0"]):
        row = dict(zip(["budget_FLOPs","context_tokens","cost_family","Q0"],key)); row["n_success"] = len(g)
        for v in ("N_B","D_B","Q","loss"):
            qs = g[v].quantile([.025,.5,.975])
            row.update({f"{v}_q025":qs.loc[.025],f"{v}_median":qs.loc[.5],f"{v}_q975":qs.loc[.975]})
        row["N_upper_probability"] = g.active_set.str.contains("N_upper").mean()
        row["D_upper_probability"] = g.active_set.str.contains("D_upper").mean()
        row["Q_upper_probability"] = g.active_set.str.contains("Q_upper").mean()
        summary_rows.append(row)
    boot_sum = pd.DataFrame(summary_rows)
    boot_sum.to_csv(RES / "q3_bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    robust_rows, robust_eval_rows = [], []
    opt_lookup = boot_opt[boot_opt.success].set_index(["budget_FLOPs","cost_family","bootstrap_id"])["loss"]
    for _, s in boot_scen.iterrows():
        nominal = central[(central.budget_FLOPs == s.budget_FLOPs) & (central.cost_family == s.cost_family)].iloc[0]
        rob = robust_config(boot, s, supported)
        configs = {
            "nominal": (nominal.N_B, nominal.D_B, nominal.Q),
            "p90_robust": (rob["N_B"], rob["D_B"], rob["Q"]),
        }
        stats = {}
        for cname, (n,d,q) in configs.items():
            losses = boot.E + (boot.A*n**(-boot.alpha)+boot.B*d**(-boot.beta))*(1+boot.kappa*(1-q))
            regrets = []
            for bid, lv in zip(boot.bootstrap_id.astype(int), losses):
                regrets.append(float(lv - opt_lookup.loc[(s.budget_FLOPs,s.cost_family,bid)]))
                robust_eval_rows.append({"budget_FLOPs":s.budget_FLOPs,"cost_family":s.cost_family,
                                         "bootstrap_id":bid,"config":cname,"loss":lv,"regret":regrets[-1]})
            stats[cname] = {"median_loss":float(np.median(losses)),"p90_loss":float(np.quantile(losses,.9)),
                            "mean_loss":float(np.mean(losses)),"p90_regret":float(np.quantile(regrets,.9)),
                            "max_regret":float(np.max(regrets))}
        robust_rows.append({"budget_FLOPs":s.budget_FLOPs,"context_tokens":s.context_tokens,"cost_family":s.cost_family,
                            "Q0":s.Q0,"nominal_N_B":nominal.N_B,"nominal_D_B":nominal.D_B,"nominal_Q":nominal.Q,
                            "robust_N_B":rob["N_B"],"robust_D_B":rob["D_B"],"robust_Q":rob["Q"],
                            **{f"nominal_{k}":v for k,v in stats["nominal"].items()},
                            **{f"robust_{k}":v for k,v in stats["p90_robust"].items()},
                            "risk_level":.9,"evidence_level":"conditional_parameter_uncertainty"})
    robust = pd.DataFrame(robust_rows)
    robust.to_csv(RES / "q3_robust_configs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(robust_eval_rows).to_csv(RES / "q3_robust_evaluation.csv.gz", index=False, compression="gzip")

    build_figures(official, paths, boot_sum, robust, mix, mix_scen, cost_curve, qdiag, support)

    # Hard internal assertions before summary.
    feasible = official[official.feasible].copy()
    assert (feasible.C_total <= feasible.budget_FLOPs + np.maximum(1.0,1e-9*feasible.budget_FLOPs)).all()
    assert np.allclose(feasible.C_train+feasible.C_quality+feasible.C_attention, feasible.C_total, rtol=2e-14, atol=1.0)
    supported_feasible = official[(official.evidence_level==EVIDENCE_SUPPORTED) & official.feasible]
    assert (supported_feasible["D_B"] >= supported.d_min-1e-8).all()
    # Preserve infeasibility instead of relaxing the joint support.  The minimum
    # cost is attained at (N_min,D_min,Q0), because Q<Q0 is dominated.
    infeasible = official[(official.evidence_level==EVIDENCE_SUPPORTED) & ~official.feasible].copy()
    infeasible_rows = []
    for _, rr in infeasible.iterrows():
        ct, cq, ca = costs(supported.n_min, supported.d_min, Q0_MAIN, Q0_MAIN,
                           int(rr.context_tokens), rr.cost_family)
        minimum = ct+cq+ca
        infeasible_rows.append({"budget_FLOPs":rr.budget_FLOPs,"context_tokens":rr.context_tokens,
                                "cost_family":rr.cost_family,"minimum_feasible_cost_FLOPs":minimum,
                                "budget_gap_FLOPs":minimum-rr.budget_FLOPs,
                                "witness_N_B":supported.n_min,"witness_D_B":supported.d_min,
                                "witness_Q":Q0_MAIN,"evidence_level":EVIDENCE_SUPPORTED})
    pd.DataFrame(infeasible_rows).to_csv(RES/"q3_infeasible_scenarios.csv",index=False,encoding="utf-8-sig")
    max_solver_gap = float(validation.loss_gap_de_minus_main.abs().max())

    summary = {
        "status":"q3_modeling_complete_pending_independent_verification",
        "q2_parameters":params,"supported_bounds":supported.__dict__,"extrapolation_guardrail":extrap.__dict__,
        "contexts":contexts,"median_context":median_ctx,"critical_context":6/ETA,"official_budgets":OFFICIAL_BUDGETS.tolist(),
        "Q0_main":Q0_MAIN,"Q0_sensitivity":list(Q0_SENS),"bootstrap_success":int(len(boot)),
        "supported_infeasible_scenarios":int(len(infeasible_rows)),
        "max_solver_loss_gap_abs":max_solver_gap,"mixture_eta_main":0.0,
        "limitations":["Q effect is B7 semi-synthetic","Q1-to-B7 scale not calibrated","eta_p not identified",
                       "bootstrap excludes structural/source/cost uncertainty","context benefit absent",
                       "factorial-hull interpolation is not a directly observed joint configuration",
                       "numerical solver agreement is not a formal global-optimality proof"],
        "runtime_seconds":time.time()-t0
    }
    (RES/"q3_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
