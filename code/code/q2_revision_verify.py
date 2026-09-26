#!/usr/bin/env python3
# AI assistance: OpenAI Codex, OpenAI, 2026-09-24; team review required.
"""Independent numerical recomputation + interface tests (not independent human review).
No import of modeling/validation/uncertainty scripts or their evaluation functions.
"""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares,minimize_scalar

ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results'


def main():
    checks=[]
    def check(name,ok,**details):checks.append({'check':name,'pass':bool(ok),**details})
    with zipfile.ZipFile(ROOT/'第二十三届中国研究生数学建模竞赛 - 中文题目/中文题目/F题.zip') as z:
        prefix='real_attachments/B_scaling_laws/'
        b1=pd.read_csv(z.open(prefix+'pythia_training_log_existing.csv'))
        b6=pd.read_csv(z.open(prefix+'supplementary_NQ_experiment.csv'))
        b7=pd.read_csv(z.open(prefix+'supplementary_NQ_experiment_expanded.csv'))
        a4=pd.read_csv(z.open('real_attachments/A_data_value/regmix_tables/train_mixture_1m.csv'))
    meta=json.loads((RES/'q2_interface_to_q3.json').read_text(encoding='utf-8'))
    p=np.array([meta['classic_model']['parameters'][k] for k in ['E','A','alpha','B','beta']])
    k=meta['quality_model']['parameters'][0]
    def model(t,n,d):return t[0]+t[1]*n**(-t[2])+t[3]*d**(-t[4])
    def huber_cost(r):
        u=np.abs(r);return .5*np.sum(np.where(u<=.01,r*r,.02*u-.0001))
    n,d,y=[b1[c].to_numpy(float) for c in ['N_params_B','D_tokens_B','val_loss']]
    # Independent transformed-residual implementation of the exact log-Huber objective.
    def residual(t):
        r=np.log(model(t,n,d))-np.log(y);u=np.abs(r)
        return np.sign(r)*np.sqrt(np.where(u<=.01,r*r,.02*u-.0001))
    lower=[.05,.0001,.02,.0001,.02];upper=[y.min()*.999,5000,1.5,5000,1.5]
    fit=least_squares(residual,np.clip(p*1.05,np.array(lower)*1.01,np.array(upper)*.99),
                      bounds=(lower,upper),loss='linear',max_nfev=30000,xtol=1e-13,ftol=1e-13,gtol=1e-13)
    cost=huber_cost(np.log(model(p,n,d))-np.log(y));alt=.5*np.sum(residual(fit.x)**2)
    maxdiff=float(np.max(np.abs(model(p,n,d)-model(fit.x,n,d))))
    check('independent_log_Huber_objective',fit.success and cost-alt<1e-9 and maxdiff<1e-6,
          original_cost=float(cost),independent_cost=float(alt),max_prediction_difference=maxdiff)
    r=p[1]*b7.N_params_B.to_numpy()**(-p[2])+p[3]*b7.D_tokens_B.to_numpy()**(-p[4])
    q=b7.Q_score.to_numpy();yy=b7.val_loss.to_numpy()
    def qcost(kk):return huber_cost(np.log(p[0]+r*(1+kk*(1-q)))-np.log(yy))
    qfit=minimize_scalar(qcost,bounds=(0,20),method='bounded',options={'xatol':1e-12})
    lr=np.log(p[0]+r*(1+k*(1-q)))-np.log(yy)
    grad=float(np.dot(np.clip(lr,-.01,.01),r*(1-q)/(p[0]+r*(1+k*(1-q)))))
    check('independent_quality_Huber_optimum',qfit.success and qcost(k)-qfit.fun<1e-10 and abs(grad)<1e-7,
          original_k=float(k),independent_k=float(qfit.x),gradient=grad,objective_gap=float(qcost(k)-qfit.fun))
    x=r*(1-q);ols=float(x@(yy-p[0]-r)/(x@x))
    check('OLS_is_comparator_not_correctness_test',True,ols_k=ols,log_huber_k=float(k),note='different objectives; no arbitrary 0.03 acceptance rule')
    ov=b6.merge(b7,on=['N_params_B','D_tokens_B','Q_score'],suffixes=('_6','_7'))
    check('B6_subset_B7',len(ov)==len(b6) and np.allclose(ov.val_loss_6,ov.val_loss_7,atol=1e-12),overlap=len(ov))

    pred=pd.read_csv(RES/'q2_quality_nested_predictions.csv');saved=pd.read_csv(RES/'q2_quality_nested_metrics.csv').iloc[0]
    err=pred.predicted-pred.observed
    rmse=float(np.sqrt(np.mean(err**2)));r2=float(1-np.sum(err**2)/np.sum((pred.observed-pred.observed.mean())**2))
    raw=b7.iloc[pred.row_id.to_numpy(int)]
    check('nested_450_predictions_metric_and_truth',len(pred)==len(b7) and pred.record_id.is_unique and
          np.allclose(raw.val_loss,pred.observed,atol=1e-12) and abs(rmse-saved.rmse)<1e-12 and abs(r2-saved.r2)<1e-12,
          rmse=rmse,r2=r2)
    sm=pd.read_csv(RES/'q2_split_manifest.csv',dtype={'fold':str,'group_id':str})
    failures=[]
    for (protocol,fold),s in sm.groupby(['protocol','fold']):
        tr=s[s.role=='train'];te=s[s.role=='test']
        if set(tr.record_id)&set(te.record_id):failures.append([protocol,fold,'record overlap'])
        if ('ND' in protocol or 'leave' in protocol) and set(tr.group_id)&set(te.group_id):
            failures.append([protocol,fold,'group overlap'])
        if 'late' in protocol:
            for group,tst in te.groupby('group_id'):
                train=tr[tr.group_id==group]
                if len(train)==0 or train.time.max()>=tst.time.min():failures.append([protocol,fold,'time leakage'])
    for outer in range(5):
        outertest=set(sm[(sm.protocol=='B7_outer_ND')&(sm.fold==str(outer))&(sm.role=='test')].record_id)
        innerids=set(sm[sm.protocol==f'B7_inner_ND_outer_{outer}'].record_id)
        if outertest&innerids:failures.append(['outer_inner',outer,'leakage'])
    check('split_manifest_separation',not failures,failures=failures,assignment_rows=len(sm))

    # API is the object under test, never used to recompute the above statistics.
    from q2_predict import Q2Predictor,equivalent_n
    api=Q2Predictor();bad=[]
    for kwargs in [dict(N=-1,D=100,Q=.7,q_scale='B7_Q_score'),dict(N=1,D=100,Q=.7,q_scale='Q1'),
                   dict(N=1,D=100,Q=.7,q_scale='B7_Q_score',units='raw'),
                   dict(N=100,D=100,Q=.7,q_scale='B7_Q_score'),
                   dict(N=1,D=100,Q=.7,q_scale='B7_Q_score',eta_p=.5),
                   dict(N=1,D=100,Q=.7,q_scale='B7_Q_score',mode='p_scenario',p=np.ones(17),eta_p=.2),
                   dict(N=1,D=100,Q=.7,q_scale='B7_Q_score',mode='p_scenario',p=np.ones(17)/17)]:
        try:api.predict(**kwargs);bad.append(str(kwargs))
        except ValueError:pass
    check('interface_rejects_invalid_inputs',not bad,unexpected_acceptances=bad)
    calc=lambda n,d,q:p[0]+(p[1]*n**(-p[2])+p[3]*d**(-p[4]))*(1+k*(1-q))
    diffs=[]
    for qv in [.1,.7,1.]:
        got=api.predict(1.,100.,qv,q_scale='B7_Q_score');h=1e-5
        dn=(calc(1+h,100,qv)-calc(1-h,100,qv))/(2*h)
        dd=(calc(1,100+h,qv)-calc(1,100-h,qv))/(2*h)
        dq=(calc(1,100,qv)-calc(1,100,qv-h))/h if qv==1 else (calc(1,100,qv+h)-calc(1,100,qv-h))/(2*h)
        diffs.extend([abs(got['loss']-calc(1,100,qv)),abs(got['dL_dN']-dn),abs(got['dL_dD']-dd),abs(got['dL_dQ']-dq)])
    check('analytic_derivatives_including_Q_boundary',max(diffs)<1e-7,max_difference=max(diffs))
    mm=meta['mixture_bridge'];pv=a4[['train_the_pile_'+x for x in mm['domain_names']]].iloc[0].to_numpy(float);pv/=pv.sum()
    expected_ra=float(pd.read_csv(RES/'q2_p_bridge_predictions.csv').query("scale=='1M_train'").iloc[0].R_A)
    got=api.predict(1.,100.,.7,q_scale='B7_Q_score',mode='p_scenario',p=pv,eta_p=.25)
    expected=p[0]+(calc(1,100,.7)-p[0])*np.exp(-.25*expected_ra)
    check('frozen_A4_bridge_and_explicit_scenario',abs(got['R_A']-expected_ra)<1e-10 and abs(got['loss']-expected)<1e-10)
    cases=[equivalent_n([1,1,1,1,1],1,1,1,0,1)['status'],
           equivalent_n([1,1,1,1,1],1,10,1,0,1)['status']]
    check('analytic_limit_and_unattainable_cases',cases==['infinite_limit','unattainable'],statuses=cases)
    mt=pd.read_csv(RES/'q2_marginal_and_substitution.csv')
    target=calc(mt.N_params_B.to_numpy(),mt.D_tokens_B.to_numpy(),mt.Q_plus_0p1.to_numpy())
    eq=calc(mt.equivalent_N_params_B.to_numpy(),mt.D_tokens_B.to_numpy(),mt.Q_score.to_numpy())
    check('equivalent_N_loss_identity',np.isfinite(eq).all() and np.max(abs(target-eq))<1e-10,max_error=float(np.max(abs(target-eq))))
    opt=pd.read_csv(RES/'q2_compute_optimal.csv');lim=meta['support']['B1'];errors=[]
    for _,row in opt.iterrows():
        nd=row.C_FLOPs_1e21/.006;lo=max(lim['N_params_B'][0],nd/lim['D_tokens_B'][1]);hi=min(lim['N_params_B'][1],nd/lim['D_tokens_B'][0])
        grid=np.geomspace(lo,max(lo,hi),1001);loss=model(p,grid,nd/grid)
        errors.append(row.box_opt_loss-float(loss.min()))
    feasible=(opt.N_box_opt_B>=lim['N_params_B'][0]*(1-1e-10))&(opt.N_box_opt_B<=lim['N_params_B'][1]*(1+1e-10))&(
        opt.D_box_opt_B>=lim['D_tokens_B'][0]*(1-1e-10))&(opt.D_box_opt_B<=lim['D_tokens_B'][1]*(1+1e-10))
    check('box_optimum_feasibility_and_dense_grid',feasible.all() and opt.box_compute_relative_residual.max()<1e-12 and max(errors)<1e-9,
          maximum_loss_above_grid=max(errors))
    boot=pd.read_csv(RES/'q2_joint_bootstrap.csv')
    check('bootstrap_formal_objective_and_success',len(boot)==300 and boot.success.sum()>=270 and (boot.objective==meta['classic_model']['objective']).all(),
          success=int(boot.success.sum()),total=len(boot))
    result={'all_pass':all(x['pass'] for x in checks),'checks':checks,
            'scope':'independent numerical implementation / author self-audit; not external scientific review',
            'unresolved':['Q1 to B7 mapping','eta_p identification','cross-source Loss semantics','unknown semi-synthetic generator dependence']}
    (RES/'q2_independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    artifacts=[]
    for directory in [ROOT/'code',ROOT/'reports',RES,ROOT/'figures']:
        for path in sorted(directory.iterdir()):
            if not path.is_file() or not ('q2' in path.name) or path.name=='q2_artifact_manifest.csv':continue
            raw=path.read_bytes();artifacts.append({'path':path.relative_to(ROOT).as_posix(),'bytes':len(raw),
                                                  'sha256':hashlib.sha256(raw).hexdigest(),'evidence_level':'metadata'})
    pd.DataFrame(artifacts).to_csv(RES/'q2_artifact_manifest.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['all_pass']:raise SystemExit(1)


if __name__=='__main__':main()
