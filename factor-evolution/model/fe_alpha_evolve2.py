# -*- coding: utf-8 -*-
"""去漂移 alpha 进化 v2：市场中性基准（r_alpha = r_asset - r_mkt）。
账面与 alpha 一致，差额=市场beta。目标=稳健真alpha Sharpe。"""
import sys, json
sys.path.insert(0, '/workspace/style-rotation-nav/factor-evolution/model')
import numpy as np
import fe_factors as FF
import fe_alpha as FA

OUT = '/workspace/style-rotation-nav/factor-evolution/out'
p, dates = FF.load('/workspace/style-rotation-nav/factor-evolution/data')
T = p.T
dc = FA.detrend_market(p.close)
q = FF.Panel.__new__(FF.Panel)
q.P, q.cols, q.T = p.P, p.cols, p.T; q.close = dc
q.open=p.open; q.high=p.high; q.low=p.low; q.vol=p.vol
q.ret=np.zeros_like(dc); q.ret[1:]=dc[1:]/dc[:-1]-1; q.ret[0]=np.nan
Fc = FF.build_factors(q); G = FA.build_git_factors(p)

def di(s): return int(np.argmax(dates.values >= np.datetime64(s)))
segs = {'train':(di('2014-01-01'),di('2020-01-01')-1),
        'valid':(di('2020-01-01'),di('2023-01-01')-1),
        'test':(di('2023-01-01'),T-1)}
base = dict(rebal=10, vol_win=20, vol_target=0.40, vol_cap=0.25, top_n=1,
            dd_limit=0.45, cost_bps=10, warmup=260)

names = [n for n in Fc if np.nanmean(np.isfinite(Fc[n])) > 0.6]
def ev(fw):
    r = FA.run_alpha(p, dc, Fc, G, {}, {**base, 'factors':fw}, {})
    return r, FA.seg_metrics(r['alpha_equity'], base['warmup'], segs)

# 阶段1 单因子排名
rank = []
for n in names:
    r, sm = ev({n:1.0})
    rank.append((n, r['alpha']['cagr'], r['alpha']['sharpe'], sm['train']['sharpe'],
                 sm['valid']['sharpe'], sm['test']['sharpe'], r['alpha']['maxdd'], r['book']['cagr']))
rank.sort(key=lambda x:-x[2])

# 阶段2 GA（目标：稳健真alpha Sharpe）
topn = [x[0] for x in rank[:18]]
rng = np.random.default_rng(20261121)
POP, GEN = 36, 22

def fitness(w):
    fw = {topn[i]:float(w[i]) for i in range(len(topn)) if abs(w[i])>1e-9}
    if not fw: return -9
    r, sm = ev(fw)
    a = r['alpha']
    worst = min(sm['train']['sharpe'], sm['valid']['sharpe'], sm['test']['sharpe'])
    gap = abs(sm['train']['sharpe']-sm['test']['sharpe'])
    fit = a['sharpe'] + 0.4*worst - 0.25*gap - 0.008*np.count_nonzero(np.abs(w)>1e-9)
    return fit

pop = [rng.choice([-1,0,1],size=len(topn)).astype(float)*rng.uniform(0.2,1.0,len(topn)) for _ in range(POP)]
for sn in ['beta20','tsmom20','mom20','cntp20']:
    if sn in topn:
        v=np.zeros(len(topn)); v[topn.index(sn)]=1.0; pop.append(v)
best=None; best_fit=-9; trace=[]
for g in range(GEN):
    scored=sorted([(fitness(ind),ind) for ind in pop],key=lambda x:-x[0])
    if scored[0][0]>best_fit: best_fit,best=scored[0]
    trace.append(best_fit)
    elite=[ind for _,ind in scored[:7]]
    newpop=list(elite)
    while len(newpop)<POP:
        a,b=elite[rng.integers(0,len(elite))],elite[rng.integers(0,len(elite))]
        c=0.5*a+0.5*b+rng.normal(0,0.25,len(topn)); c[np.abs(c)<0.15]=0
        newpop.append(c)
    pop=newpop

fw_best={topn[i]:float(best[i]) for i in range(len(topn)) if abs(best[i])>1e-9}
r,sm=ev(fw_best); a=r['alpha']
# 对照：beta20 单因子
rb,sb=ev({'beta20':1.0})

print('=== 去漂移alpha进化冠军（市场中性）===')
print('权重:',{k:round(v,3) for k,v in fw_best.items()})
for tag,rr,sres in [('GA冠军',r,sm),('beta20单因子',rb,sb)]:
    aa=rr['alpha']; print('\n[%s] alpha全 %.2f%% Sh%.3f MDD%.2f%% Calmar%.3f | 账面 %.2f%%'%(
        tag,aa['cagr']*100,aa['sharpe'],aa['maxdd']*100,aa['calmar'],rr['book']['cagr']*100))
    for k in ['train','valid','test']:
        v=sres[k]; print('   %-6s alpha %6.2f%% Sh%.3f MDD%6.2f%%'%(k,v['cagr']*100,v['sharpe'],v['maxdd']*100))
r2,_=ev(fw_best); parity=abs(r['alpha_equity'][-1]-r2['alpha_equity'][-1])<1e-12

result={
 'version':'去漂移alpha进化 v2.0 (市场中性)',
 'note':'r_alpha=r_asset-r_mkt; 剔除共同beta保留相对alpha; 目标=稳健真alpha Sharpe; git因子仅观察',
 'detrend':'market_neutral_r_asset_minus_r_mkt',
 'market_beta_note':'账面CAGR = 真alpha + 市场beta(等权市场年化约12%), 差额即不可外推的beta漂移',
 'gene_pool':topn,
 'single_factor_rank':[dict(name=x[0],alpha_cagr=x[1],sharpe=x[2],tr_sh=x[3],va_sh=x[4],te_sh=x[5],mdd=x[6],book_cagr=x[7]) for x in rank],
 'champion_factors':fw_best,
 'fitness':float(best_fit),'trace':[float(x) for x in trace],
 'metrics':{'full':a,**sm},'book_cagr':float(r['book']['cagr']),
 'beta20_metrics':{'full':rb['alpha'],**sb,'book_cagr':float(rb['book']['cagr'])},
 'parity':bool(parity),
}
json.dump(result,open(f'{OUT}/alpha_evolution.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
print('\nparity:',parity,'→ 已存 out/alpha_evolution.json')
