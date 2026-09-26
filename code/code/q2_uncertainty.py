#!/usr/bin/env python3
# AI assistance: OpenAI Codex, OpenAI, 2026-09-24; team review required.
"""Same-estimator joint conditional bootstrap; N-D constrained benchmark, NOT full Q3."""
import json
import zipfile
import numpy as np
import pandas as pd
from q2_modeling import (ROOT, ZIP_PATH, RESULTS as RES, FIGURES as FIG, B_PREFIX,
                        B_FILES, fit_classic, fit_q, plt, save_fig, log)
from q2_predict import equivalent_n

SCOPE = 'base+Q refit; fixed model; independent resampling assumption; excludes generator/source/residual/p uncertainty'


def main():
    rng=np.random.default_rng(42)
    with zipfile.ZipFile(ZIP_PATH) as z:
        b1=pd.read_csv(z.open(B_PREFIX+B_FILES['B1']))
        b7=pd.read_csv(z.open(B_PREFIX+B_FILES['B7']))
    meta=json.loads((RES/'q2_interface_to_q3.json').read_text(encoding='utf-8'))
    saved=pd.read_csv(RES/'q2_classic_parameters.csv').iloc[0]
    names=['E','A','alpha','B','beta']; p0=saved[names].to_numpy(float)
    groups=[g for _,g in b1.groupby('N_params_B')]
    qgroups=[g for _,g in b7.groupby(['N_params_B','D_tokens_B'])]
    rows=[]
    for rep in range(300):
        take=rng.integers(0,len(groups),len(groups)); qt=rng.integers(0,len(qgroups),len(qgroups))
        sample=pd.concat([groups[j] for j in take],ignore_index=True)
        qs=pd.concat([qgroups[j] for j in qt],ignore_index=True)
        row={'bootstrap_id':rep,'unique_trajectories':len(set(take)),'unique_ND_groups':len(set(qt)),
             'objective':saved.model,'evidence_level':'semi_synthetic','uncertainty_scope':SCOPE}
        try:
            # Same formal estimator, including its 12 starting points and bounds.
            pars,diag=fit_classic(sample.N_params_B,sample.D_tokens_B,sample.val_loss,saved.model,seed=1041,starts=12)
            qp,qd=fit_q(meta['quality_model']['kind'],qs,pars)
            row.update(dict(zip(names,pars)))
            row.update(kappa=float(qp[0]),success=bool(diag['success'] and qd['success']),
                       active_bounds=diag['active_bounds'],optimality=diag['optimality'],
                       jacobian_condition=diag['jacobian_condition'],q_jacobian_condition=qd['jacobian_condition'],error='')
        except Exception as exc:
            row.update(success=False,error=repr(exc))
        rows.append(row)
        if (rep+1)%25==0:
            log(f'联合 Bootstrap {rep+1}/300；成功={sum(x["success"] for x in rows)}')
            pd.DataFrame(rows).to_csv(RES/'q2_joint_bootstrap.csv',index=False,encoding='utf-8-sig')
    boot=pd.DataFrame(rows); good=boot[boot.success]
    boot.to_csv(RES/'q2_joint_bootstrap.csv',index=False,encoding='utf-8-sig')
    boot.to_csv(RES/'q2_classic_trajectory_bootstrap.csv',index=False,encoding='utf-8-sig')
    if len(good)<270:
        raise RuntimeError('More than 10% bootstrap failures: inspect before publishing intervals')
    intervals=[]
    for col in names+['kappa']:
        point=float(saved[col]) if col in names else meta['quality_model']['parameters'][0]
        intervals.append({'parameter':col,'point_estimate':point,'bootstrap_median':good[col].median(),
                          'ci2p5':good[col].quantile(.025),'ci97p5':good[col].quantile(.975),
                          'successful_replicates':len(good),'evidence_level':'semi_synthetic' if col=='kappa' else 'real_observational',
                          'uncertainty_scope':SCOPE})
    pd.DataFrame(intervals).to_csv(RES/'q2_classic_parameter_intervals.csv',index=False,encoding='utf-8-sig')
    good[names+['kappa']].corr().rename_axis('parameter').reset_index().assign(evidence_level='semi_synthetic').to_csv(
        RES/'q2_joint_parameter_correlations.csv',index=False,encoding='utf-8-sig')
    marginal=pd.read_csv(RES/'q2_marginal_and_substitution.csv'); limits=meta['support']
    for ix,r in marginal.iterrows():
        n,d,q,qn=r.N_params_B,r.D_tokens_B,r.Q_score,r.Q_plus_0p1
        draws=good.E+(good.A*n**(-good.alpha)+good.B*d**(-good.beta))*(1+good.kappa*(1-q))
        eqs=[equivalent_n(row[names].to_numpy(float),row.kappa,n,d,q,qn) for _,row in good.iterrows()]
        finite=np.array([x['N_equivalent']/n for x in eqs if x['status']=='finite'])
        marginal.loc[ix,'predicted_loss_joint_ci2p5'],marginal.loc[ix,'predicted_loss_joint_ci97p5']=np.quantile(draws,[.025,.975])
        if len(finite):
            marginal.loc[ix,'equivalent_N_multiplier_joint_ci2p5'],marginal.loc[ix,'equivalent_N_multiplier_joint_ci97p5']=np.quantile(finite,[.025,.975])
        marginal.loc[ix,'bootstrap_nonfinite_solution_fraction']=1-len(finite)/len(eqs)
        for source,lim in limits.items():
            inside=lim['N_params_B'][0]<=n<=lim['N_params_B'][1] and lim['D_tokens_B'][0]<=d<=lim['D_tokens_B'][1]
            eqinside=lim['N_params_B'][0]<=r.equivalent_N_params_B<=lim['N_params_B'][1] and lim['D_tokens_B'][0]<=d<=lim['D_tokens_B'][1]
            if 'Q_score' in lim:
                inside=inside and lim['Q_score'][0]<=q<=qn<=lim['Q_score'][1]
                eqinside=eqinside and lim['Q_score'][0]<=q<=qn<=lim['Q_score'][1]
            marginal.loc[ix,f'input_in_{source}_box']=bool(inside)
            marginal.loc[ix,f'equivalent_in_{source}_box']=bool(eqinside)
        marginal.loc[ix,'solution_support']='in_B1_B7_boxes' if marginal.loc[ix,'equivalent_in_B1_box'] and marginal.loc[ix,'equivalent_in_B7_box'] else 'extrapolated_solution'
    marginal['uncertainty_scope']=SCOPE
    marginal['engineering_feasibility']='not assessed: no engineering N bound supplied'
    marginal.to_csv(RES/'q2_marginal_and_substitution.csv',index=False,encoding='utf-8-sig')
    canonical=.006*b1.N_params_B*b1.D_tokens_B
    ca=b1[['run_id','N_params_B','D_tokens_B','C_FLOPs_1e21']].copy()
    ca['computed_C_1e21']=canonical; ca['absolute_difference']=abs(canonical-b1.C_FLOPs_1e21)
    ca['relative_difference']=ca.absolute_difference/canonical;ca['evidence_level']='real_observational'
    ca.to_csv(RES/'q2_compute_unit_audit.csv',index=False,encoding='utf-8-sig')
    nlo,nhi=limits['B1']['N_params_B'];dlo,dhi=limits['B1']['D_tokens_B'];e,a,alpha,b,beta=p0
    opt=[]
    for c21 in np.geomspace(canonical.min(),canonical.max(),30):
        nd=c21/.006;ns=((alpha*a)/(beta*b)*nd**beta)**(1/(alpha+beta));ds=nd/ns
        lo,hi=max(nlo,nd/dhi),min(nhi,nd/dlo)
        if lo>hi and not np.isclose(lo,hi,rtol=1e-12):
            raise RuntimeError('Canonical budget infeasible in B1 box')
        nc=np.clip(ns,lo,max(lo,hi));dc=nd/nc
        opt.append({'C_FLOPs_1e21':c21,'C_FLOPs':c21*1e21,'N_opt_B':ns,'D_opt_B':ds,
                    'D_over_N':ds/ns,'predicted_loss':e+a*ns**(-alpha)+b*ds**(-beta),
                    'N_compute_exponent':beta/(alpha+beta),'D_compute_exponent':alpha/(alpha+beta),
                    'N_in_B1_box':nlo<=ns<=nhi,'D_in_B1_box':dlo<=ds<=dhi,
                    'unconstrained_in_B1_box':nlo<=ns<=nhi and dlo<=ds<=dhi,
                    'N_box_opt_B':nc,'D_box_opt_B':dc,'box_opt_loss':e+a*nc**(-alpha)+b*dc**(-beta),
                    'box_compute_relative_residual':abs(.006*nc*dc-c21)/c21,
                    'evidence_level':'model_scenario','scope':'classic N-D benchmark only; not full Q3 costs'})
    opt=pd.DataFrame(opt);opt.to_csv(RES/'q2_compute_optimal.csv',index=False,encoding='utf-8-sig')
    fig,axes=plt.subplots(1,2,figsize=(9.2,4))
    for c in ['alpha','beta']: axes[0].hist(good[c],bins=24,alpha=.65,label=c)
    axes[0].set_xlabel('指数（正式 Huber 重拟合）');axes[0].set_ylabel('频数');axes[0].legend()
    axes[1].scatter(good.alpha,good.beta,s=12,alpha=.4);axes[1].set_xlabel(r'$\alpha$');axes[1].set_ylabel(r'$\beta$')
    save_fig(fig,FIG/'q2_09_parameter_uncertainty.pdf',also_png=True);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.5,4))
    for ax,v in zip(axes,['N','D']):
        ax.plot(opt.C_FLOPs_1e21,opt[f'{v}_opt_B'],'--',label='无约束解析基准')
        ax.plot(opt.C_FLOPs_1e21,opt[f'{v}_box_opt_B'],label='B1 矩形约束基准')
        lim=limits['B1']['N_params_B' if v=='N' else 'D_tokens_B']
        ax.axhspan(*lim,alpha=.1,color='#2E5A87');ax.set_xscale('log');ax.set_yscale('log')
        ax.set_xlabel(r'$C/10^{21}$ FLOPs');ax.set_ylabel(v+'（十亿）');ax.legend(fontsize=8)
    save_fig(fig,FIG/'q2_10_compute_optimal.pdf',also_png=True);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharey=True)
    for ax,(dv,ds) in zip(axes,marginal.groupby('D_tokens_B')):
        for qv,sub in ds.groupby('Q_score'):
            y=sub.equivalent_N_multiplier
            ax.errorbar(sub.N_params_B,y,yerr=[np.maximum(0,y-sub.equivalent_N_multiplier_joint_ci2p5),
                         np.maximum(0,sub.equivalent_N_multiplier_joint_ci97p5-y)],marker='o',capsize=2,label=f'Q={qv:g}')
        ext=ds[ds.solution_support=='extrapolated_solution']
        ax.scatter(ext.N_params_B,ext.equivalent_N_multiplier,marker='x',s=50,color='black',label='外推解')
        ax.set_xscale('log');ax.set_xlabel('N（十亿）');ax.text(.03,.95,f'D={dv:g}B',transform=ax.transAxes,va='top')
    axes[0].set_ylabel('Q+0.1 的等效 N 倍率');axes[-1].legend(fontsize=7)
    save_fig(fig,FIG/'q2_07_quality_substitution.pdf',also_png=True);plt.close(fig)
    pd.DataFrame([
        {'figure':'q2_09_parameter_uncertainty.pdf','result_source':'q2_joint_bootstrap.csv','assertion':SCOPE},
        {'figure':'q2_10_compute_optimal.pdf','result_source':'q2_compute_optimal.csv','assertion':'Budget support does not imply N/D support'},
        {'figure':'q2_07_quality_substitution.pdf','result_source':'q2_marginal_and_substitution.csv','assertion':'Conditional parameter intervals; extrapolated solutions marked'}
    ]).assign(evidence_level='metadata').to_csv(RES/'q2_uncertainty_figure_contract.csv',index=False,encoding='utf-8-sig')
    summary={'bootstrap_success':len(good),'bootstrap_total':len(boot),'active_bounds':int(good.active_bounds.sum()),
             'N_compute_exponent':float(beta/(alpha+beta)),'D_compute_exponent':float(alpha/(alpha+beta)),
             'unsupported_unconstrained_optima':int((~opt.unconstrained_in_B1_box).sum()),
             'extrapolated_equivalent_scenarios':int((marginal.solution_support=='extrapolated_solution').sum()),
             'scope':SCOPE,'note':'Fixed Q,p factors do not alter N:D optimum; not full Q3 optimization'}
    (RES/'q2_uncertainty_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    log(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__': main()
