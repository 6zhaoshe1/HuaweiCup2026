#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI assistance: OpenAI Codex, OpenAI, 2026-09-25; team review required.
"""Independent Q3 numerical audit. Does not import q3_modeling.py."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ETA = 2e-4
EV = "semi_synthetic_supported_joint_box"
EX = "theoretical_extrapolation_b9_guardrail"


def gv(q, family):
    if family == "exponential": return 1e7*np.exp(6*np.asarray(q))
    if family == "power": return 5e9*np.asarray(q)**4
    if family == "logarithmic": return 2e9*np.log1p(10*np.asarray(q))
    raise ValueError(family)


def main():
    meta = json.loads((RES/"q2_interface_to_q3.json").read_text(encoding="utf-8"))
    p = meta["classic_model"]["parameters"]
    E,A,alpha,B,beta = [float(p[x]) for x in ("E","A","alpha","B","beta")]
    kappa = float(meta["quality_model"]["parameters"][0])
    o = pd.read_csv(RES/"q3_official_solutions.csv")
    checks = []
    max_cost = max_loss = max_budget_excess = 0.0
    for i,r in o[o.feasible].iterrows():
        n,d,q,q0 = [float(r[x]) for x in ("N_B","D_B","Q","Q0")]
        na,da = n*1e9,d*1e9
        ct = 6*na*da
        cq = da*max(float(gv(q,r.cost_family)-gv(q0,r.cost_family)),0.0)
        ca = ETA*na*da*float(r.context_tokens)
        total = ct+cq+ca
        loss = E+(A*n**(-alpha)+B*d**(-beta))*(1+kappa*(1-q))
        max_cost=max(max_cost,abs(total-float(r.C_total)))
        max_loss=max(max_loss,abs(loss-float(r.loss)))
        max_budget_excess=max(max_budget_excess,total-float(r.budget_FLOPs))
        checks.append(total <= float(r.budget_FLOPs)+max(1,1e-9*float(r.budget_FLOPs)))
        if r.evidence_level==EV:
            checks.append(0.070542-1e-9<=n<=11.965825+1e-9 and 10-1e-9<=d<=299.893+1e-9 and .1<=q<=1)

    # Brute independent rectangular grid: main continuous solution must be no worse.
    central=o[(o.evidence_level==EV)&(o.context_tokens==4096)&(o.feasible)].copy()
    grid_gaps=[]
    ns=np.geomspace(.070542,11.965825,401)
    for _,r in central.iterrows():
        qs=np.linspace(max(float(r.Q0),.1),1,401)
        nn,qq=np.meshgrid(ns,qs,indexing="ij")
        c=float(r.budget_FLOPs)/1e18; h=6+ETA*4096
        ss=np.maximum(gv(qq,r.cost_family)-gv(float(r.Q0),r.cost_family),0)/1e9
        dd=np.minimum(299.893,c/(h*nn+ss))
        yy=E+(A*nn**(-alpha)+B*dd**(-beta))*(1+kappa*(1-qq))
        yy=np.where(dd>=10,yy,np.inf)
        grid=float(np.nanmin(yy)); gap=grid-float(r.loss); grid_gaps.append(gap)
        checks.append(gap>=-2e-7)

    # Infeasible certificates.
    inf=pd.read_csv(RES/"q3_infeasible_scenarios.csv")
    for _,r in inf.iterrows():
        n=.070542*1e9; d=10*1e9; q0=.5
        minimum=6*n*d+ETA*n*d*float(r.context_tokens)
        checks.append(abs(minimum-float(r.minimum_feasible_cost_FLOPs))<=1e-12*minimum)
        checks.append(minimum>float(r.budget_FLOPs))

    # Bootstrap summary recomputation.
    bo=pd.read_csv(RES/"q3_bootstrap_optima.csv.gz")
    bs=pd.read_csv(RES/"q3_bootstrap_summary.csv")
    max_boot_quantile=0.0
    for _,r in bs.iterrows():
        g=bo[(bo.budget_FLOPs==r.budget_FLOPs)&(bo.cost_family==r.cost_family)&bo.success]
        for v in ("N_B","D_B","Q","loss"):
            z=g[v].quantile([.025,.5,.975]).to_numpy()
            w=np.array([r[f"{v}_q025"],r[f"{v}_median"],r[f"{v}_q975"]],float)
            max_boot_quantile=max(max_boot_quantile,float(np.max(np.abs(z-w))))
        checks.append(len(g)==300)

    # Robust results and conditional mixture semantics.
    rb=pd.read_csv(RES/"q3_robust_configs.csv")
    checks.append(bool((rb.robust_p90_loss<=rb.nominal_p90_loss+2e-8).all()))
    re=pd.read_csv(RES/"q3_robust_evaluation.csv.gz")
    checks.append(float(re.regret.min())>=-2e-7)
    ms=pd.read_csv(RES/"q3_mixture_scenarios.csv")
    checks.append(np.allclose(ms.loc[ms.eta_p==0,"conditional_loss"],ms.loc[ms.eta_p==0,"main_loss_eta0"],rtol=0,atol=1e-12))
    checks.append(ms.argmin_NDQ_unchanged.astype(bool).all())
    mix=pd.read_csv(RES/"q3_mixture_ranking.csv")
    share_cols=[c for c in mix if c.startswith("train_the_pile_")]
    max_mix_closure=float(np.max(np.abs(mix[share_cols].sum(axis=1)-1)))
    checks.append(max_mix_closure<2e-12)
    checks.append(float(np.max(np.abs(mix.original_share_sum-1)))<=.0040001)

    sv=pd.read_csv(RES/"q3_solver_validation.csv")
    checks.append(float(sv.loss_gap_de_minus_main.abs().max())<2e-7)

    # Evidence geometry: source designs are Cartesian, but an interpolated
    # point is not promoted to a directly observed joint configuration.
    sg=pd.read_csv(RES/"q3_solution_evidence.csv")
    checks.append(bool(sg.B1_design_is_cartesian.astype(bool).all()))
    checks.append(bool(sg.B7_design_is_cartesian.astype(bool).all()))
    for _,r in sg.iterrows():
        tier=str(r.overall_evidence_tier)
        if not bool(r.feasible):
            checks.append(tier=="I_infeasible_under_frozen_constraints")
        elif r.evidence_level==EX:
            checks.append(tier=="D_theoretical_extrapolation")
        else:
            checks.append(tier in {"A_observed_joint_anchor","B_supported_factorial_interpolation",
                                   "C_supported_boundary_scenario"})
            checks.append(np.isfinite(float(r.nearest_B1_scaled_log_distance)))
            checks.append(np.isfinite(float(r.nearest_B7_scaled_logQ_distance)))

    # KKT/envelope sign audit for the Q profile at the nine central scenarios.
    qd=pd.read_csv(RES/"q3_quality_boundary_diagnostics.csv")
    checks.append(len(qd)==9)
    checks.append(bool((qd.direct_quality_benefit_per_Q>0).all()))
    checks.append(bool((qd.g_prime_FLOPs_per_token_per_Q>0).all()))
    for _,r in qd.iterrows():
        if r.diagnostic_cause=="interior_marginal_balance":
            checks.append(abs(float(r.profile_dLoss_dQ))<2e-5)
        elif r.diagnostic_cause=="marginal_cost_exceeds_benefit_at_Q0":
            checks.append(float(r.profile_dLoss_dQ)>=-2e-5)
        elif r.diagnostic_cause in {"Q_upper_binds_net_benefit_remains","support_caps_with_budget_slack"}:
            checks.append(float(r.profile_dLoss_dQ)<=2e-5)
        else:
            checks.append(False)
    out={
        "status":"PASS" if all(checks) else "FAIL","checks_total":len(checks),"checks_passed":int(sum(bool(x) for x in checks)),
        "max_cost_recompute_abs_FLOPs":max_cost,"max_loss_recompute_abs":max_loss,
        "max_budget_excess_FLOPs":max_budget_excess,"min_grid_minus_main_loss":float(min(grid_gaps)),
        "max_bootstrap_quantile_abs_diff":max_boot_quantile,"max_mixture_closure_abs":max_mix_closure,
        "notes":["author-independent formulas; no import from q3_modeling","grid is a lower-resolution global check, not proof of reality",
                 "scientific limitations in q3_summary remain"]
    }
    (RES/"q3_independent_verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    if out["status"] == "PASS":
        summary_path = RES / "q3_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["status"] = "q3_complete_pending_team_review"
        summary["independent_verification_status"] = "PASS"
        summary["independent_verification_checks"] = f"{out['checks_passed']}/{out['checks_total']}"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))
    if out["status"]!="PASS": raise SystemExit(2)


if __name__=="__main__": main()
