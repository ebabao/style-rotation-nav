"""
factor-evolution · 遗传算法进化学习 (冲击 35% 收益目标)
=========================================================
两段式:
  Stage A: 进化因子权重组合 (从49因子库中选)
  Stage B: 进化执行参数 (rebal/vol_target/vol_cap/top_n + 择时)
防过拟合:
  - 三段隔离 train(2013-2019) / valid(2020-2022) / test(2023-2026)
  - fitness 惩罚 train-valid 落差 (稳健性约束)
  - 最终报告 test 样本外, 不参与进化
  - champion/challenger 版本化, 只提名不自动切换
"""
import numpy as np
import json, os, sys
import fe_factors as FF
import fe_engine as FE

SEED = 20261121
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')

FACTOR_POOL = [
    'mom5','mom10','mom20','mom60','mom120','mom250','rev1','rev3','rev5',
    'tsmom20','tsmom60','tsmom120','kmid','klen','ksft','kup','klow',
    'ma5','ma10','ma20','ma60','rsqr20','rsqr60','beta20','beta60',
    'std20','std60','vol_contraction','down_vol20','corr5','corr20','vol_surge','vstd20',
    'cntp20','cntp60','sumd20','sumd60','fip252','skew20','skew60','kurt20','kurt60',
    'eff20','eff60','amihud20','rank20','rank60','ddhigh20','ddhigh60',
]


def segment_metrics(eq_curve, warmup, seg_start, seg_end):
    """从完整 equity 曲线切一段算指标 (seg 为绝对 index)。"""
    seg = eq_curve[seg_start:seg_end+1]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 60:
        return None
    # 段内重新归一
    seg = seg / seg[0]
    yrs = len(seg)/252.0
    cagr = seg[-1]**(1/yrs)-1 if yrs>0 else 0
    peak = np.maximum.accumulate(seg)
    mdd = float(np.min(seg/peak-1))
    dret = np.diff(seg)/seg[:-1]
    sharpe = float(np.mean(dret)/(np.std(dret)+1e-9)*np.sqrt(252))
    return {'cagr':float(cagr),'maxdd':mdd,'sharpe':sharpe,'calmar':float(cagr/abs(mdd)) if mdd<0 else 0}


def evaluate(panel, F, params, segs):
    r = FE.run(panel, F, params)
    eq = r['equity']
    warm = params['warmup']
    out = {'full': None, 'train': None, 'valid': None, 'test': None}
    out['full'] = segment_metrics(eq, warm, warm, panel.T-1)
    for k,(a,b) in segs.items():
        out[k] = segment_metrics(eq, warm, a, b)
    out['_raw'] = r
    return out


def random_factor_weights(rng, n_active):
    idx = rng.choice(len(FACTOR_POOL), size=n_active, replace=False)
    w = {}
    for i in idx:
        w[FACTOR_POOL[i]] = float(rng.uniform(-1.5, 1.5))
    # 保证至少有一个正权重(否则全空)
    if not any(v>0 for v in w.values()):
        w[FACTOR_POOL[idx[0]]] = abs(w[FACTOR_POOL[idx[0]]]) + 0.5
    return w


def mutate_weights(w, rng, rate=0.2):
    w = dict(w)
    keys = list(FACTOR_POOL)
    for k in list(w.keys()):
        if rng.random() < rate:
            w[k] = float(np.clip(w[k] + rng.normal(0, 0.4), -2, 2))
    if rng.random() < rate and len(w) < 12:
        k = rng.choice(keys)
        w[k] = float(rng.uniform(-1, 1))
    if rng.random() < rate*0.5 and len(w) > 2:
        w.pop(rng.choice(list(w.keys())))
    return w


def main():
    rng = np.random.default_rng(SEED)
    panel, dates = FF.load(DATA)
    F = FF.build_factors(panel)
    T = panel.T
    # 三段: train 2013-2019 / valid 2020-2022 / test 2023-2026
    def di(s):
        return int(np.argmax(dates.values >= np.datetime64(s)))
    segs = {
        'train': (di('2014-01-01'), di('2020-01-01')-1),
        'valid': (di('2020-01-01'), di('2023-01-01')-1),
        'test':  (di('2023-01-01'), T-1),
    }
    print('分段:', {k:(str(dates[a].date()),str(dates[b].date())) for k,(a,b) in segs.items()})

    base = dict(rebal=10, vol_win=20, vol_target=0.40, vol_cap=0.25,
                top_n=1, dd_limit=0.45, cost_bps=10, warmup=260,
                timing_mode='none')

    # ---------- Stage A: 因子权重进化 ----------
    POP, GEN = 40, 25
    pop = [random_factor_weights(rng, rng.integers(2, 8)) for _ in range(POP)]
    best = None; best_fit = -1e9
    history = []
    for g in range(GEN):
        scored = []
        for w in pop:
            p = {**base, 'factors': w}
            m = evaluate(panel, F, p, {'train':segs['train'],'valid':segs['valid']})
            tr, va = m['train'], m['valid']
            if tr is None or va is None:
                scored.append((-9, w, m)); continue
            # fitness: 训练收益 + 验证收益 - 落差惩罚 - 回撤惩罚
            gap = abs(tr['cagr'] - va['cagr'])
            fit = 0.5*tr['cagr'] + 0.5*va['cagr'] - 1.0*gap - 0.15*abs(tr['maxdd']) - 0.15*abs(va['maxdd'])
            scored.append((fit, w, m))
        scored.sort(key=lambda x: -x[0])
        if scored[0][0] > best_fit:
            best_fit = scored[0][0]; best = (scored[0][1], scored[0][2])
        # 精英 + 变异
        elite = [s[1] for s in scored[:8]]
        newpop = elite + [mutate_weights(rng.choice(elite), rng) for _ in range(POP-len(elite))]
        pop = newpop
        history.append({'gen':g,'best_fit':float(scored[0][0]),
                        'train_cagr':scored[0][2]['train']['cagr'] if scored[0][2]['train'] else 0,
                        'valid_cagr':scored[0][2]['valid']['cagr'] if scored[0][2]['valid'] else 0})
        if (g+1)%5==0:
            print('Gen %2d best_fit %.4f train %.2f%% valid %.2f%%'%(
                g+1,scored[0][0],history[-1]['train_cagr']*100,history[-1]['valid_cagr']*100))
    fw_best = best[0]
    print('\n=== Stage A 冠军因子权重 ===')
    for k,v in sorted(fw_best.items(), key=lambda kv:-abs(kv[1])):
        print('  %-14s %+.3f  [%s]'%(k,v,FF.SOURCE.get(k,'经典')))

    # ---------- Stage B: 执行参数微调 (小网格, 固定冠军因子) ----------
    print('\n=== Stage B: 执行参数网格 ===')
    grid = []
    for rebal in (5, 10, 15, 20):
        for vt in (0.30, 0.40, 0.50):
            for vc in (0.20, 0.25, 0.30):
                p = {**base, 'factors': fw_best, 'rebal':rebal, 'vol_target':vt, 'vol_cap':vc}
                m = evaluate(panel, F, p, {'train':segs['train'],'valid':segs['valid'],'test':segs['test']})
                tr,va = m['train'],m['valid']
                if not tr or not va: continue
                fit = 0.5*tr['cagr']+0.5*va['cagr']-1.0*abs(tr['cagr']-va['cagr'])-0.15*abs(tr['maxdd'])-0.15*abs(va['maxdd'])
                grid.append((fit,rebal,vt,vc,m))
    grid.sort(key=lambda x:-x[0])
    print('Top5 执行参数 (按train+valid稳健fitness):')
    for fit,rb,vt,vc,m in grid[:5]:
        print('  fit %.4f rebal=%d vt=%.2f vc=%.2f | train %.2f%% valid %.2f%% test %.2f%%'%(
            fit,rb,vt,vc,m['train']['cagr']*100,m['valid']['cagr']*100,
            m['test']['cagr']*100 if m['test'] else 0))

    # ---------- 最终冠军: 用 train+valid 选, test 只报告 ----------
    fit,rb,vt,vc,m = grid[0]
    champion = {**base,'factors':fw_best,'rebal':rb,'vol_target':vt,'vol_cap':vc}
    final_eval = evaluate(panel,F,champion,segs)
    result = {
        'version':'factor-evolution v1.0 (独立因子进化)',
        'seed':SEED,
        'segments':{k:{'start':str(dates[a].date()),'end':str(dates[b].date())} for k,(a,b) in segs.items()},
        'factor_pool_size':len(FACTOR_POOL),
        'n_factors_selected':len(fw_best),
        'champion_params':{k:v for k,v in champion.items() if k!='factors'},
        'champion_factors':fw_best,
        'metrics':{k:v for k,v in final_eval.items() if k!='_raw'},
        'ga_history':history,
        'exec_grid_top':[{'fit':float(f),'rebal':rb2,'vol_target':vt2,'vol_cap':vc2,
                          'train':mm['train']['cagr'],'valid':mm['valid']['cagr'],
                          'test':mm['test']['cagr'] if mm['test'] else None}
                         for f,rb2,vt2,vc2,mm in grid[:10]],
        'paradigm':'独立因子库进化 / 三段隔离 / train-valid稳健fitness / test不外泄',
    }
    os.makedirs(OUT,exist_ok=True)
    json.dump(result,open(os.path.join(OUT,'evolution_result.json'),'w',encoding='utf-8'),
              ensure_ascii=False,indent=2)
    # parity 复核
    m2 = evaluate(panel,F,champion,segs)
    parity = abs(m2['full']['cagr']-final_eval['full']['cagr'])<1e-12
    result['parity']=parity
    json.dump(result,open(os.path.join(OUT,'evolution_result.json'),'w',encoding='utf-8'),
              ensure_ascii=False,indent=2)
    print('\n=== 最终冠军 (test 样本外) ===')
    for k in ['full','train','valid','test']:
        v=final_eval[k]
        if v: print('  %-6s CAGR %6.2f%%  MDD %7.2f%%  Sharpe %.3f  Calmar %.3f'%(
            k,v['cagr']*100,v['maxdd']*100,v['sharpe'],v['calmar']))
    print('parity:',parity)
    print('结果已存:',os.path.join(OUT,'evolution_result.json'))


if __name__=='__main__':
    main()
