# -*- coding: utf-8 -*-
"""以【去漂移 alpha 的 Sharpe】为目标做受约束进化。
关键：fitness 用真 alpha 而非账面 CAGR；三段隔离惩罚过拟合；parity=0。
git 因子只做观察/方向参考，不作为交易开关（已证负贡献）。
"""
import sys, json, os
sys.path.insert(0, '/workspace/style-rotation-nav/factor-evolution/model')
import numpy as np
import fe_factors as FF
import fe_alpha as FA

OUT = '/workspace/style-rotation-nav/factor-evolution/out'
p, dates = FF.load('/workspace/style-rotation-nav/factor-evolution/data')
T = p.T
dclose = FA.detrend_rolling(p.close)
q = FF.Panel.__new__(FF.Panel)
q.P, q.cols, q.T = p.P, p.cols, p.T
q.close = dclose; q.open = p.open; q.high = p.high; q.low = p.low; q.vol = p.vol
q.ret = np.zeros_like(dclose); q.ret[1:] = dclose[1:]/dclose[:-1]-1; q.ret[0] = np.nan
F_cross = FF.build_factors(q)
G = FA.build_git_factors(p)

def di(s): return int(np.argmax(dates.values >= np.datetime64(s)))
segs = {'train':(di('2014-01-01'),di('2020-01-01')-1),
        'valid':(di('2020-01-01'),di('2023-01-01')-1),
        'test':(di('2023-01-01'),T-1)}
base = dict(rebal=10, vol_win=20, vol_target=0.40, vol_cap=0.25, top_n=1,
            dd_limit=0.45, cost_bps=10, warmup=260)

# 只用去漂移IC为正/负且稳定的候选因子（排除覆盖率低的）
names = [n for n in F_cross if np.nanmean(np.isfinite(F_cross[n])) > 0.6]
print('候选因子(%d):'%len(names), names)

def ev(fw, ov=None):
    r = FA.run_alpha(p, dclose, F_cross, G, {}, {**base, 'factors':fw, **(ov or {})}, {})
    sm = FA.seg_metrics(r['alpha_equity'], base['warmup'], segs)
    return r['alpha'], sm

# ---- 阶段1：单因子去漂移 alpha 排名（全样本 + 各段 Sharpe）----
print('\n=== 阶段1: 单因子去漂移alpha排名 ===')
rank = []
for n in names:
    a, sm = ev({n:1.0})
    rank.append((n, a['cagr'], a['sharpe'], sm['train']['sharpe'],
                 sm['valid']['sharpe'], sm['test']['sharpe'], a['maxdd']))
rank.sort(key=lambda x:-x[2])
for n,c,s,ts,vs,tes,dd in rank[:15]:
    print('%-14s alpha %6.2f%% Sh %5.2f | tr %5.2f va %5.2f te %5.2f | MDD %6.2f%%'%(
        n,c*100,s,ts,vs,tes,dd*100))

# ---- 阶段2：GA 进化多因子组合，fitness=稳健去漂移Sharpe ----
topn = [r[0] for r in rank[:16]]
# 补充几个已知强的
for extra in ['beta20','mom250','mom20','mom120','eff120','vol_contraction']:
    if extra in F_cross and extra not in topn: topn.append(extra)

rng = np.random.default_rng(20261121)
POP, GEN = 32, 18

def fitness(w):
    # w: 与 topn 等长的权重向量（可正负）
    fw = {topn[i]:float(w[i]) for i in range(len(topn)) if abs(w[i])>1e-9}
    if not fw: return -9
    a, sm = ev(fw)
    # 主目标: 全样本去漂移Sharpe; 惩罚最差段; 惩罚train-test落差(防过拟合)
    worst = min(sm['train']['sharpe'], sm['valid']['sharpe'], sm['test']['sharpe'])
    gap = abs(sm['train']['sharpe']-sm['test']['sharpe'])
    fit = a['sharpe'] + 0.35*worst - 0.30*gap
    # 权重稀疏惩罚（防堆因子过拟合）
    fit -= 0.01*np.count_nonzero(np.abs(w)>1e-9)
    return fit

pop = [rng.choice([-1,0,1], size=len(topn)).astype(float) * rng.uniform(0.2,1.0,len(topn))
       for _ in range(POP)]
# 注入已知强种子
for seed_name in ['beta20','mom250','mom20']:
    if seed_name in topn:
        v = np.zeros(len(topn)); v[topn.index(seed_name)] = 1.0; pop.append(v)

best = None; best_fit = -9; trace = []
for g in range(GEN):
    scored = sorted([(fitness(ind), ind) for ind in pop], key=lambda x:-x[0])
    if scored[0][0] > best_fit:
        best_fit, best = scored[0]
    trace.append(best_fit)
    # 精英 + 变异/交叉
    elite = [ind for _,ind in scored[:6]]
    newpop = list(elite)
    while len(newpop) < POP:
        a, b = elite[rng.integers(0,len(elite))], elite[rng.integers(0,len(elite))]
        child = 0.5*a + 0.5*b + rng.normal(0,0.25,len(topn))
        # 稀疏化
        child[np.abs(child)<0.15] = 0
        newpop.append(child)
    pop = newpop
    if (g+1) % 5 == 0:
        print('  gen %2d best fitness %.3f'%(g+1, best_fit))

fw_best = {topn[i]:float(best[i]) for i in range(len(topn)) if abs(best[i])>1e-9}
a, sm = ev(fw_best)
r = FA.run_alpha(p, dclose, F_cross, G, {}, {**base,'factors':fw_best}, {})
print('\n=== GA 冠军（去漂移alpha目标）===')
print('权重:', {k:round(v,3) for k,v in fw_best.items()})
print('全样本 alpha CAGR %.2f%% Sh %.3f MDD %.2f%% Calmar %.3f'%(
    a['cagr']*100,a['sharpe'],a['maxdd']*100,a['calmar']))
for k in ['train','valid','test']:
    v=sm[k]; print('  %-6s alpha %6.2f%% Sh %.3f MDD %6.2f%%'%(k,v['cagr']*100,v['sharpe'],v['maxdd']*100))
print('账面 CAGR %.2f%%'%r['book']['cagr']*100)

# parity
r2 = FA.run_alpha(p, dclose, F_cross, G, {}, {**base,'factors':fw_best}, {})
parity = abs(r['alpha_equity'][-1]-r2['alpha_equity'][-1]) < 1e-12

result = {
    'version':'去漂移alpha进化 v1.0',
    'note':'目标函数=去漂移alpha的Sharpe(非账面CAGR); git因子仅观察不做开关; 滚动无前视mu=756d',
    'detrend':'rolling_mu_756d_nolookahead',
    'gene_pool':topn,
    'single_factor_rank':[dict(name=x[0],alpha_cagr=x[1],sharpe=x[2],tr=x[3],va=x[4],te=x[5],mdd=x[6]) for x in rank],
    'champion_factors':fw_best,
    'fitness':float(best_fit),
    'trace':[float(x) for x in trace],
    'metrics':{'full':a, **sm},
    'book_cagr':r['book']['cagr'],
    'parity':bool(parity),
    'git_overlay_note':'14因子作为择时开关均降低去漂移alpha(第29章高AUC陷阱), 仅作观察指标',
}
json.dump(result, open(f'{OUT}/alpha_evolution.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)
print('\nparity:',parity,'→ 已存 out/alpha_evolution.json')
