# -*- coding: utf-8 -*-
"""在扩池面板上重新进化因子（不是固定旧因子），目标=稳健真alpha Sharpe。
alpha基准固定原5资产等权，保证与原17.07%可比。
"""
import sys, json
sys.path.insert(0,'/workspace/style-rotation-nav/factor-evolution/model')
import numpy as np, pandas as pd
import fe_factors as FF, fe_pool as FP

BASE='/workspace/style-rotation-nav/factor-evolution'
base=pd.read_csv(f'{BASE}/data/etf_ohlc.csv',parse_dates=['day']).set_index('day')
cand=pd.read_csv(f'{BASE}/data/etf_candidates.csv',parse_dates=['day']).set_index('day')
ORIG5=['CSI300ETF','NDX100ETF','GoldETF','ChiNextETF','STAR50ETF']
BOND='BondETF'
class Obj: pass

def build_panel(new_names):
    order=ORIG5+new_names+[BOND]
    cols={}
    for a in ORIG5+[BOND]:
        for suf in ('open','high','low','close','vol'): cols[f'{a}_{suf}']=base[f'{a}_{suf}'].reindex(base.index)
    for a in new_names:
        for suf in ('open','high','low','close','vol'): cols[f'{a}_{suf}']=cand[f'{a}_{suf}'].reindex(base.index)
    panel=Obj()
    for attr in ('open','high','low','close','vol'):
        setattr(panel,attr,np.column_stack([cols[f'{a}_{attr}'] for a in order]))
    panel.ret=np.zeros_like(panel.close); panel.ret[1:]=panel.close[1:]/panel.close[:-1]-1; panel.ret[0]=np.nan
    panel.T=len(base); panel.cols=order
    ri=list(range(len(ORIG5)+len(new_names))); mi=list(range(len(ORIG5))); bi=len(ORIG5)+len(new_names)
    dc=FP.detrend_market_pool(panel.close,mi)
    dp=Obj()
    for attr in ('open','high','low','vol'): setattr(dp,attr,getattr(panel,attr))
    dp.close=dc; dp.ret=np.zeros_like(dc); dp.ret[1:]=dc[1:]/dc[:-1]-1; dp.ret[0]=np.nan
    dp.T=panel.T; dp.cols=order
    Fc=FF.build_factors(dp)
    return panel,dc,Fc,ri,mi,bi

def di(s): return int(np.argmax(base.index>=pd.Timestamp(s)))
segs={'train':(di('2014-01-01'),di('2020-01-01')-1),'valid':(di('2020-01-01'),di('2023-01-01')-1),'test':(di('2023-01-01'),len(base)-1)}
basep=dict(rebal=10,vol_win=20,vol_target=0.40,vol_cap=0.25,top_n=1,dd_limit=0.45,cost_bps=10,warmup=260)

POOLS={
 '5+能源化工':['EnergyChemETF'],
 '5+豆粕':['SoybeanETF'],
 '5+能源化工+豆粕':['EnergyChemETF','SoybeanETF'],
 '5+标普500':['SP500ETF'],
 '5+能源化工+豆粕+标普':['EnergyChemETF','SoybeanETF','SP500ETF'],
}
POP,GEN=28,16
allres={}

for poolname,new in POOLS.items():
    panel,dc,Fc,ri,mi,bi=build_panel(new)
    names=[n for n in Fc if np.nanmean(np.isfinite(Fc[n][:,ri]))>0.5]
    def ev(fw):
        r=FP.run_alpha_pool(panel,dc,Fc,{**basep,'factors':fw},ri,mi,bi)
        return r,FP.seg_metrics(r['alpha_equity'],260,segs)
    def fitness(w):
        fw={names[i]:float(w[i]) for i in range(len(names)) if abs(w[i])>1e-9}
        if not fw: return -9
        r,sm=ev(fw); a=r['alpha']
        worst=min(sm['train']['sharpe'],sm['valid']['sharpe'],sm['test']['sharpe'])
        gap=abs(sm['train']['sharpe']-sm['test']['sharpe'])
        return a['sharpe']+0.4*worst-0.25*gap-0.008*np.count_nonzero(np.abs(w)>1e-9)
    rng=np.random.default_rng(20261121)
    pop=[rng.choice([-1,0,1],size=len(names)).astype(float)*rng.uniform(0.2,1,len(names)) for _ in range(POP)]
    for sn in ['beta20','mom20','tsmom20','cntp20']:
        if sn in names:
            v=np.zeros(len(names)); v[names.index(sn)]=1.0; pop.append(v)
    best=None;bf=-9
    for g in range(GEN):
        scored=sorted([(fitness(ind),ind) for ind in pop],key=lambda x:-x[0])
        if scored[0][0]>bf: bf,best=scored[0]
        elite=[ind for _,ind in scored[:6]]; newpop=list(elite)
        while len(newpop)<POP:
            a,b=elite[rng.integers(0,len(elite))],elite[rng.integers(0,len(elite))]
            c=0.5*a+0.5*b+rng.normal(0,0.25,len(names)); c[np.abs(c)<0.15]=0
            newpop.append(c)
        pop=newpop
    fw={names[i]:float(best[i]) for i in range(len(names)) if abs(best[i])>1e-9}
    r,sm=ev(fw); a=r['alpha']
    print('%-22s alpha全 %6.2f%%(Sh%.2f) test %6.2f%% 账面 %6.2f%% 因子%d'%(
        poolname,a['cagr']*100,a['sharpe'],sm['test']['cagr']*100,r['book']['cagr']*100,len(fw)),flush=True)
    allres[poolname]=dict(factors=fw,alpha=a,seg=sm,book=r['book'],fitness=float(bf),n_assets=len(new))

json.dump(allres,open(f'{BASE}/out/pool_evolution.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
print('\n已存 out/pool_evolution.json')
