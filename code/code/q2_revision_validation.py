#!/usr/bin/env python3
# AI assistance: OpenAI Codex, OpenAI, 2026-09-24; team review required.
"""Retrospective revised validation: nested ND selection, source/family holdouts, evidence contracts."""
import json
import zipfile
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.model_selection import GroupKFold,KFold
from q2_modeling import (ROOT,RESULTS as RES,FIGURES as FIG,ZIP_PATH,B_PREFIX,B_FILES,
                        Q_BOUNDS,fit_q,q_predict,classic_predict,metrics,plt,save_fig,log)


def save(df,name):
    if 'evidence_level' not in df: df=df.assign(evidence_level='metadata')
    df.to_csv(RES/name,index=False,encoding='utf-8-sig')


def main():
    with zipfile.ZipFile(ZIP_PATH) as z:
        b={k:pd.read_csv(z.open(B_PREFIX+v)) for k,v in B_FILES.items()}
        provenance=json.loads(z.read('real_attachments/source_manifest.json'))
    meta=json.loads((RES/'q2_interface_to_q3.json').read_text(encoding='utf-8'))
    base=np.array(list(meta['classic_model']['parameters'].values()))
    records=[]
    def manifest(df,source,group,tr,te,protocol,fold):
        for role,ixs in [('train',tr),('test',te)]:
            for i in ixs:
                records.append({'record_id':f'{source}:{i}','group_id':str(group[i]),
                                'original_row_index':int(df.iloc[i].name),
                                'time':float(df.iloc[i].D_tokens_B),'time_definition':'cumulative D in billion tokens, not calendar time',
                                'source':source,'role':role,'fold':str(fold),'protocol':protocol,
                                'evidence_level':'semi_synthetic' if source in ['B2','B6','B7','B8'] else 'real_observational'})

    b1=b['B1']; g=b1.N_params_B.astype(str).to_numpy();d=b1.D_tokens_B.to_numpy()
    for i,gg in enumerate(np.unique(g)):
        manifest(b1,'B1',g,np.flatnonzero(g!=gg),np.flatnonzero(g==gg),'B1_leave_one_size_out',i)
    early=np.zeros(len(g),bool)
    for gg in np.unique(g):
        ix=np.flatnonzero(g==gg);early[ix[d[ix]<=np.quantile(d[ix],.75)]]=True
    manifest(b1,'B1',g,np.flatnonzero(early),np.flatnonzero(~early),'B1_late_token',0)
    for i,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=42).split(b1)):
        manifest(b1,'B1',g,tr,te,'B1_random_row_diagnostic_NOT_group_independent',i)

    df=b['B7'];g=df[['N_params_B','D_tokens_B']].astype(str).agg('|'.join,axis=1).to_numpy()
    n,d,q,y=[df[c].to_numpy(float) for c in ['N_params_B','D_tokens_B','Q_score','val_loss']]
    nested=[];selections=[];inner_rows=[]
    for outer,(tr,te) in enumerate(GroupKFold(5).split(df,groups=g)):
        manifest(df,'B7',g,tr,te,'B7_outer_ND',outer)
        inner=list(GroupKFold(4).split(df.iloc[tr],groups=g[tr]))
        for fold,(itr,ite) in enumerate(inner):
            manifest(df,'B7',g,tr[itr],tr[ite],f'B7_inner_ND_outer_{outer}',fold)
        candidates=[]
        for kind in Q_BOUNDS:
            oof=np.full(len(tr),np.nan)
            for fold,(itr,ite) in enumerate(inner):
                pars,diag=fit_q(kind,df.iloc[tr[itr]],base)
                if not diag['success']:raise RuntimeError(f'Q fit failed {outer}/{fold}/{kind}')
                ii=tr[ite];oof[ite]=q_predict(kind,pars,base,n[ii],d[ii],q[ii])
            score=metrics(y[tr],oof)
            candidates.append((score['rmse'],score['mae'],kind))
            inner_rows.append({'outer_fold':outer,'model':kind,**score,'evidence_level':'semi_synthetic'})
        chosen=sorted(candidates)[0][2]
        pars,diag=fit_q(chosen,df.iloc[tr],base)
        if not diag['success']:raise RuntimeError('Outer fit failed')
        pp=q_predict(chosen,pars,base,n[te],d[te],q[te])
        selections.append({'outer_fold':outer,'selected':chosen,'parameters_json':json.dumps(pars.tolist()),
                           **metrics(y[te],pp),'evidence_level':'semi_synthetic'})
        for i,p in zip(te,pp):
            nested.append({'record_id':f'B7:{i}','row_id':i,'group_id':g[i],'fold':outer,'selected':chosen,
                           'N_params_B':n[i],'D_tokens_B':d[i],'Q_score':q[i],'observed':y[i],'predicted':p,
                           'evidence_level':'semi_synthetic'})
        log(f'嵌套验证 outer={outer} 选择 {chosen}')
    npred=pd.DataFrame(nested)
    save(npred,'q2_quality_nested_predictions.csv')
    save(pd.DataFrame(selections),'q2_quality_nested_folds.csv')
    save(pd.DataFrame(inner_rows),'q2_quality_nested_selection.csv')
    nested_summary=metrics(npred.observed,npred.predicted)
    save(pd.DataFrame([{'protocol':'5 outer ND / 4 inner ND','model':'entire_six_candidate_selection',**nested_summary,
                       'evidence_level':'semi_synthetic','limitation':'retrospective; fixed B1 fit, unknown shared generator dependence'}]),'q2_quality_nested_metrics.csv')
    # New-Q rows are interpolation along Q, with N,D shared: not new experiments.
    key=['N_params_B','D_tokens_B','Q_score']
    keys6=set(map(tuple,b['B6'][key].round(10).to_numpy()))
    is_old=np.array([tuple(v) in keys6 for v in df[key].round(10).to_numpy()])
    manifest(df,'B7',g,np.flatnonzero(is_old),np.flatnonzero(~is_old),'B6_subset_to_B7_new_Q_interpolation',0)

    src_metrics=[];src_pred=[]
    # B4 contains Pythia, already used in the fixed B1 module. Remove it from
    # BOTH target calibration and evaluation as a separate sensitivity analysis.
    b['B4_without_Pythia']=b['B4'][~b['B4'].family.astype(str).str.contains('pythia',case=False)].copy()
    for source,group_col in [('B4','family'),('B5','source'),('B5','family'),('B4_without_Pythia','family')]:
        sd=b[source];sg=sd[group_col].astype(str).to_numpy()
        nn,dd,yy=[sd[c].to_numpy(float) for c in ['N_params_B','D_tokens_B','val_loss']]
        r=classic_predict(base,nn,dd)-base[0];x=np.column_stack([np.ones(len(sd)),r])
        pred=np.full(len(sd),np.nan);meanpred=np.full(len(sd),np.nan)
        for fold,gg in enumerate(np.unique(sg)):
            tr=np.flatnonzero(sg!=gg);te=np.flatnonzero(sg==gg)
            manifest(sd,source,sg,tr,te,f'{source}_leave_{group_col}_out',fold)
            pars=lsq_linear(x[tr],yy[tr],bounds=([0,0],[np.inf,np.inf])).x
            pred[te]=x[te]@pars;meanpred[te]=yy[tr].mean()
            for i in te:
                src_pred.append({'source':source,'protocol':f'leave_{group_col}_out','record_id':f'{source}:{i}',
                                 'group_id':gg,'fold':fold,'observed':yy[i],'predicted':pred[i],
                                 'train_mean_prediction':meanpred[i],'calibration_E':pars[0],'calibration_scale':pars[1],
                                 'evidence_level':'real_observational','scope':'source labels retained; loss comparability unverified'})
        for label,pp in [('heldout_affine',pred),('train_mean_baseline',meanpred),('uncalibrated_B1',r+base[0])]:
            src_metrics.append({'source':source,'grouping':group_col,'model':label,**metrics(yy,pp),
                                'evidence_level':'real_observational','scope':'heldout target group; internal source convention not independently verified'})
    b2=b['B2'];g=b2.run_id.astype(str).to_numpy();d=b2.D_tokens_B.to_numpy();early=np.zeros(len(b2),bool)
    for gg in np.unique(g):
        ix=np.flatnonzero(g==gg);early[ix[d[ix]<=np.quantile(d[ix],.7)]]=True
    manifest(b2,'B2',g,np.flatnonzero(early),np.flatnonzero(~early),'B2_early70_late30',0)
    save(pd.DataFrame(src_pred),'q2_source_heldout_predictions.csv')
    save(pd.DataFrame(src_metrics),'q2_source_heldout_metrics.csv')
    save(pd.DataFrame(records),'q2_split_manifest.csv')

    # Full by-scale and by-N/D/Q diagnostics, not averaged away.
    residual=[]
    cp=pd.read_csv(RES/'q2_classic_predictions.csv.gz')
    cp=cp[cp.model==meta['classic_model']['objective']]
    for (split,nn),sub in cp.groupby(['split','N_params_B']):
        residual.append({'source':'B1','protocol':split,'variable':'N_params_B','value':nn,
                         **metrics(sub.observed,sub.predicted),'evidence_level':'real_observational'})
    for col in ['N_params_B','D_tokens_B','Q_score']:
        for value,sub in npred.groupby(col):
            residual.append({'source':'B7','protocol':'nested_selection','variable':col,'value':value,
                             **metrics(sub.observed,sub.predicted),'evidence_level':'semi_synthetic'})
    save(pd.DataFrame(residual),'q2_residual_strata.csv')
    evidence=pd.read_csv(RES/'q2_data_inventory.csv')
    source_map={x['file']:x.get('source','not supplied') for x in provenance}
    for ix,row in evidence.iterrows():
        source=row.dataset
        if source in B_FILES:
            evidence.loc[ix,'manifest_source']=source_map.get('B_scaling_laws/'+B_FILES[source],'not supplied')
            for col in ['N_params_B','D_tokens_B','Q_score']:
                if col in b[source]:
                    evidence.loc[ix,col+'_min']=b[source][col].min();evidence.loc[ix,col+'_max']=b[source][col].max()
        evidence.loc[ix,'validation_set_identity']='not established across sources'
        evidence.loc[ix,'tokenizer_compatibility']='not established across sources'
        evidence.loc[ix,'loss_aggregation_compatibility']='val_loss label alone insufficient'
        evidence.loc[ix,'provenance_basis']='visible screenshots + source_manifest + complete actual tables; generator code not available'
    save(evidence,'q2_evidence_matrix.csv')
    unit=pd.DataFrame([{'source':'B1','records':len(b1),'run_id_unique':b1.run_id.nunique(),
                       'N_unique':b1.N_params_B.nunique(),'split_unit':'N trajectory; run_id is record-like'},
                      {'source':'B2','records':len(b2),'run_id_unique':b2.run_id.nunique(),
                       'N_unique':b2.N_params_B.nunique(),'split_unit':'run_id trajectory'}])
    save(unit,'q2_independent_unit_audit.csv')

    # Evidence plot: in-sample vs heldout with SAME axes, non-causal labels.
    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    sm=pd.DataFrame(src_metrics)
    old=pd.read_csv(RES/'q2_source_calibration_summary.csv')
    for ax,(src,grp) in zip(axes,[('B4','family'),('B5','source'),('B5','family')]):
        sub=sm[(sm.source==src)&(sm.grouping==grp)]
        vals=[float(old[(old.dataset==src)&(old.test=='target_affine_shape_fit')].rmse.iloc[0]),
              float(sub[sub.model=='heldout_affine'].rmse.iloc[0]),float(sub[sub.model=='uncalibrated_B1'].rmse.iloc[0]),
              float(sub[sub.model=='train_mean_baseline'].rmse.iloc[0])]
        ax.bar(['样本内','分组留出','未校准','训练均值'],vals,color=['#888888','#2E5A87','#E69F00','#D1495B'])
        ax.set_ylabel('RMSE');ax.text(.03,.96,f'{src} / {grp}',transform=ax.transAxes,va='top')
    save_fig(fig,FIG/'q2_11_source_heldout.pdf',also_png=True);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    for ax,col in zip(axes,['N_params_B','D_tokens_B','Q_score']):
        for value,sub in npred.groupby(col):
            err=sub.predicted-sub.observed
            ax.errorbar(value,err.mean(),yerr=err.std(),fmt='o',color='#2E5A87',capsize=2)
        ax.axhline(0,ls='--',color='#555555')
        ax.set_xlabel({'N_params_B':'N（十亿参数）','D_tokens_B':'D（十亿 Token）','Q_score':'B7 质量 Q'}[col])
        if col!='Q_score':ax.set_xscale('log')
    axes[0].set_ylabel('嵌套 OOF 残差均值 ± SD')
    fig.subplots_adjust(wspace=.28)
    save_fig(fig,FIG/'q2_12_nested_residuals.pdf',also_png=True);plt.close(fig)
    save(pd.DataFrame([
        {'figure':'q2_11_source_heldout.pdf','result_source':'q2_source_heldout_metrics.csv;q2_source_calibration_summary.csv',
         'assertion':'Same-source in-sample affine fit does not imply family/source holdout performance'},
        {'figure':'q2_12_nested_residuals.pdf','result_source':'q2_quality_nested_predictions.csv',
         'assertion':'Entire Q selection procedure evaluated in outer folds; semi-synthetic and retrospective'}
    ]),'q2_revision_figure_contract.csv')
    summary={'nested_metrics':nested_summary,'outer_selections':pd.DataFrame(selections).selected.value_counts().to_dict(),
             'source_heldout_metrics':src_metrics,'protocol':'retrospective revision, not pristine independent data'}
    (RES/'q2_revision_validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    log(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
