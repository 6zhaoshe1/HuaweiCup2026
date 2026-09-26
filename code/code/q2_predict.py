#!/usr/bin/env python3
# AI assistance: OpenAI Codex, OpenAI, 2026-09-24; team review required.
"""Validated Q2 interface. No Q1-to-B7 calibration is asserted.

Example: predict_loss(1.,100.,.7,q_scale='B7_Q_score')
N,D must be in billions; parameter intervals are NOT future-observation intervals.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def equivalent_n(base, k, n, d, q, q_next):
    """Solve L(N_eq,D,Q)=L(N,D,Q_next), including limiting cases."""
    e, a, alpha, b, beta = map(float, base)
    vals = [*base, k, n, d, q, q_next]
    if not np.isfinite(vals).all() or n <= 0 or d <= 0 or not 0 <= q <= q_next <= 1:
        raise ValueError('finite positive N,D and 0<=Q<=Q_next<=1 required')
    m = 1 + k*(1-q); m_next = 1 + k*(1-q_next)
    if a <= 0 or alpha <= 0 or m <= 0 or m_next <= 0:
        raise ValueError('invalid scaling parameters')
    rhs = ((a*n**(-alpha)+b*d**(-beta))*m_next/m - b*d**(-beta))/a
    # Exact equality gives the asymptotic limit; no arbitrary huge root bound.
    if rhs < 0:
        return {'N_equivalent': None, 'status': 'unattainable', 'log_N_equivalent': None}
    if rhs == 0:
        return {'N_equivalent': None, 'status': 'infinite_limit', 'log_N_equivalent': None}
    logn = -math.log(rhs)/alpha
    if logn > math.log(np.finfo(float).max):
        return {'N_equivalent': None, 'status': 'finite_but_numeric_overflow', 'log_N_equivalent': logn}
    return {'N_equivalent': math.exp(logn), 'status': 'finite', 'log_N_equivalent': logn}


class Q2Predictor:
    def __init__(self, interface_path=None, bootstrap_path=None):
        path = Path(interface_path or ROOT/'results/q2_interface_to_q3.json')
        self.meta = json.loads(path.read_text(encoding='utf-8'))
        if self.meta['quality_model']['kind'] != 'reducible_linear':
            raise ValueError('Interface implements reducible_linear only; revalidate before changing model')
        self.base = np.array([self.meta['classic_model']['parameters'][k] for k in ['E','A','alpha','B','beta']])
        self.k = self.meta['quality_model']['parameters'][0]
        bp = Path(bootstrap_path or ROOT/'results/q2_joint_bootstrap.csv')
        self.boot = pd.read_csv(bp) if bp.exists() else None

    def predict(self, N, D, Q, *, q_scale, mode='main', p=None, eta_p=None,
                units='billions', allow_extrapolation=False, interval=True):
        if units != 'billions':
            raise ValueError('Only units=billions accepted. Convert raw counts before calling.')
        if q_scale != 'B7_Q_score':
            raise ValueError('Q must use B7_Q_score; Q1-to-B7 mapping is unidentified')
        if not np.isfinite([N,D,Q]).all() or N <= 0 or D <= 0 or not 0 <= Q <= 1:
            raise ValueError('N,D must be finite positive scalars; Q must lie in [0,1]')
        if mode not in ['main','p_scenario']:
            raise ValueError('mode must be main or p_scenario')
        if mode == 'main' and (p is not None or eta_p not in [None,0]):
            raise ValueError('Main model does not identify p: use explicit p_scenario mode')
        warnings = ['Q effect is semi-synthetic, not a verified causal intervention',
                    'Potential shared B1/B7 generator dependence is unresolved; intervals are conditional']
        support = {}
        for source, limits in self.meta['support'].items():
            values = {'N_params_B':N,'D_tokens_B':D,'Q_score':Q}
            support[source] = all(lo <= values[c] <= hi for c,(lo,hi) in limits.items())
        if not all(support.values()):
            if not allow_extrapolation:
                raise ValueError(f'Outside marginal support: {support}; explicit allow_extrapolation required')
            warnings.append('Extrapolation: outside B1 and/or B7 marginal support')
        warnings.append('Marginal-box membership alone is not proof of joint support or transfer validity')
        r_a, multiplier = 0., 1.
        if mode == 'p_scenario':
            if eta_p is None or not np.isfinite(eta_p) or eta_p < 0:
                raise ValueError('Scenario requires an explicit finite nonnegative eta_p assumption')
            p = np.asarray(p,float)
            m = self.meta['mixture_bridge']
            if p.shape != (len(m['domain_names']),) or not np.isfinite(p).all() or (p<0).any():
                raise ValueError('p must be a finite nonnegative vector in the documented 17-domain order')
            if abs(p.sum()-1) > 1e-8:
                raise ValueError('p must sum to one; no silent normalization')
            yh = np.asarray(m['intercepts']) + np.asarray(m['coefficients']) @ np.log(p+m['delta'])
            raw = np.mean((yh-np.asarray(m['output_mean']))/np.asarray(m['output_sd']))
            r_a = -(raw-m['index_center'])/m['index_scale']
            multiplier = np.exp(-eta_p*r_a)
            if not np.isfinite(multiplier) or multiplier == 0:
                raise ValueError('Scenario multiplier over/underflow: reject extreme eta_p')
            warnings.extend(['eta_p is supplied, not estimated; p is not optimizable in the main model',
                             'A4 mixture support not certified; p-response and Q1-to-B7 mapping remain uncalibrated'])
        e,a,alpha,b,beta = self.base
        rn,rd = a*N**(-alpha), b*D**(-beta)
        mq = 1+self.k*(1-Q)
        loss = e+(rn+rd)*mq*multiplier
        out = {'loss':float(loss),'mode':mode,'units':'N,D in billions; loss B-source convention',
               'R_A':float(r_a),'eta_p':0. if mode=='main' else float(eta_p),
               'dL_dN':float(-alpha*rn/N*mq*multiplier),
               'dL_dD':float(-beta*rd/D*mq*multiplier),
               'dL_dQ':float(-self.k*(rn+rd)*multiplier),
               'marginal_support':support,'warnings':warnings,
               'evidence_level':'semi_synthetic' if mode=='main' else 'conditional_model_scenario',
               'parameter_interval_95':None,
               'uncertainty_scope':'base+Q refit; fixed model/eta/p; excludes source, generator, Q mapping and residual uncertainty'}
        if interval and self.boot is not None:
            bs = self.boot[self.boot.success]
            vals = bs.E+(bs.A*N**(-bs.alpha)+bs.B*D**(-bs.beta))*(1+bs.kappa*(1-Q))*multiplier
            out['parameter_interval_95'] = np.quantile(vals,[.025,.975]).tolist()
        if not np.isfinite(loss):
            raise ValueError('Non-finite prediction')
        return out


def predict_loss(N,D,Q,**kwargs):
    return Q2Predictor().predict(N,D,Q,**kwargs)


if __name__ == '__main__':
    print(json.dumps(predict_loss(1.,100.,.7,q_scale='B7_Q_score'),ensure_ascii=False,indent=2))
