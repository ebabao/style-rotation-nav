"""
路径穷尽实验 —— 回答"真的没办法达到35%收益/19%回撤吗"
=====================================================
前面所有实验的引擎都把仓位硬封顶1.0(零杠杆), 然后用这些结果证明"零杠杆做不到",
属循环论证。本脚本解除封顶, 系统扫描从未验证过的四条路径:
  1) 固定杠杆 max_exp ∈ {1.0, 1.1, 1.2, 1.3}
  2) 动态杠杆 + 趋势择时 (牛市加杠杆/熊市空仓)
  3) 多资产配置 top_n ∈ {1,2}
  4) 目标波动 vol_target 扫描
严格无前视: 趋势分数t日可见, t-1决策t生效, 10bp成本。
"""
import sys, json
import numpy as np
sys.path.insert(0, 'model')
import fe_factors as FF

p, dates = FF.load('data')
F = FF.build_factors(p)
C = p.close
R = p.ret
T, A = C.shape
warm = 260

def di(s):
    return int(np.argmax(dates.values >= np.datetime64(s)))

segs = {
    'train': (di('2014-01-01'), di('2020-01-01') - 1),
    'valid': (di('2020-01-01'), di('2023-01-01') - 1),
    'test': (di('2023-01-01'), T - 1),
}

def roll_std(x, w):
    out = np.full(len(x), np.nan)
    for t in range(len(x)):
        seg = x[max(0, t-w+1):t+1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= max(3, w//2):
            out[t] = np.std(seg, ddof=1)
    return out

def zcross(mat):
    out = np.full_like(mat, np.nan)
    for t in range(mat.shape[0]):
        row = mat[t]; ok = np.isfinite(row)
        if ok.sum() >= 3:
            mu, sd = row[ok].mean(), row[ok].std()
            out[t, ok] = (row[ok] - mu) / (sd + 1e-9)
    return out

# 预计算
vol = np.full((T, A), np.nan)
for j in range(A):
    vol[:, j] = roll_std(R[:, j], 20) * np.sqrt(252)
Fz = {k: zcross(v) for k, v in F.items()}

# 市场代理与趋势分数 (t日可见, 无前视)
idx = np.nanmean(C[:, :5], axis=1)
def trend_frac(t, k=120):
    votes = 0.0; tot = 0.0
    if t >= k and np.isfinite(idx[t]) and np.isfinite(idx[t-k]):
        tot += 1; votes += 1.0 if idx[t] > np.nanmean(idx[t-k+1:t+1]) else 0.0
    cnt = 0; okc = 0
    for j in range(5):
        if np.isfinite(C[t, j]) and np.isfinite(C[t-k, j]):
            okc += 1
            cnt += 1 if C[t, j] > np.nanmean(C[t-k+1:t+1, j]) else 0
    if okc > 0:
        tot += 1; votes += cnt / okc
    if t >= 60 and np.isfinite(idx[t]) and np.isfinite(idx[t-60]):
        tot += 1; votes += 1.0 if idx[t] > idx[t-60] else 0.0
    return votes / tot if tot > 0 else 0.5

TREND = np.array([trend_frac(t) if t >= warm else 0.5 for t in range(T)])

def run(fw, max_exp=1.0, dyn=False, top_n=1, vol_target=0.40, vol_cap=0.25,
        low_exp=0.15, cost_bps=10.0, rebal=10, dd_limit=0.45):
    """独立引擎: 支持杠杆与动态杠杆。fw={因子:权重}"""
    eq = 1.0; peak = 1.0
    eqc = np.full(T, np.nan)
    cw = np.zeros(A); ce = 0.0
    cooldown = 0; turn = 0.0
    for t in range(warm, T):
        dr = 0.0
        for j in range(A):
            if np.isfinite(R[t, j]) and np.isfinite(cw[j]) and cw[j] != 0:
                dr += cw[j] * R[t, j]
        eq *= (1 + dr); eqc[t] = eq
        peak = max(peak, eq); dd = eq/peak - 1
        if (t - warm) % rebal == 0 and t < T-1:
            score = np.zeros(A)
            for nm, w in fw.items():
                if nm in Fz:
                    score += w * np.nan_to_num(Fz[nm][t], nan=0.0)
            elig = np.array([np.isfinite(vol[t,j]) and vol[t,j] <= vol_cap and np.isfinite(C[t,j])
                             for j in range(5)])
            if elig.sum() == 0:
                nw = np.zeros(A); nw[5] = 1.0; ne = 0.0
            else:
                cand = np.where(elig)[0]
                k = min(top_n, len(cand))
                order = cand[np.argsort(-score[cand])][:k]
                nw = np.zeros(A)
                iv = 1.0/np.maximum(vol[t,order], 1e-6); iv /= iv.sum()
                for ix, j in enumerate(order): nw[j] = iv[ix]
                vsel = np.nan_to_num(vol[t,order], nan=0.0)
                port_vol = float(np.sqrt(np.sum((iv*vsel)**2) + 2e-6))
                vscaled = vol_target/(port_vol+1e-9)  # 波动倒数仓位
                if dyn:
                    # 动态杠杆: 趋势分数线性映射到 [low_exp, max_exp]
                    f = TREND[t]
                    ne = low_exp + (max_exp - low_exp)*np.clip((f-0.35)/(0.65-0.35), 0, 1)
                    # 波动约束: 不超过波动倒数算出的仓位*max_exp
                    ne = min(ne, vscaled*max_exp)
                else:
                    ne = np.clip(vscaled, 0.0, max_exp)
                if dd <= -dd_limit:
                    cooldown = 10
                if cooldown > 0:
                    ne *= 0.5; cooldown -= 1
                ne = float(np.clip(ne, 0.0, max_exp))
                nw[5] = 1.0 - ne  # <0 即杠杆借款
            delta = np.abs(nw-cw).sum() + abs(ne-ce)
            turn += delta
            eq *= (1 - cost_bps/1e4*delta)
            cw = nw; ce = ne
    e = eqc[warm:]; e = e[np.isfinite(e)]
    yrs = len(e)/252.0
    cagr = (e[-1]/e[0])**(1/yrs)-1 if yrs > 0 else 0
    pk = np.maximum.accumulate(e); mdd = float(np.min(e/pk-1))
    dr = np.diff(e)/e[:-1]
    shp = float(np.mean(dr)/(np.std(dr)+1e-9)*np.sqrt(252))
    return {'cagr':cagr, 'maxdd':mdd, 'sharpe':shp,
            'calmar':cagr/abs(mdd) if mdd<0 else 0, 'equity':eqc, 'turn':turn}

def seg_cagr(eqc, a, b):
    s = eqc[a:b+1]; s = s[np.isfinite(s)]
    if len(s) < 60: return 0.0
    s = s/s[0]; return (s[-1])**(252.0/len(s))-1

bases = {
    'mom250(吃beta)': {'mom250':1.0},
    'beta20(真alpha)': {'beta20':1.0},
}

results = []
print("="*100)
print("路径穷尽: 目标 收益≥35% 且 回撤≤19%")
print("="*100)

for bn, fw in bases.items():
    print(f"\n【底盘 {bn}】")
    print(f"{'方案':<28} {'全样本CAGR':>10} {'MDD':>9} {'夏普':>7} {'Calmar':>8} {'样本外':>9} {'判定':>10}")
    # 路径1: 固定杠杆
    for me in [1.0, 1.1, 1.2, 1.3]:
        r = run(fw, max_exp=me)
        hit = r['cagr']>=0.35 and r['maxdd']>=-0.19
        te = seg_cagr(r['equity'], *segs['test'])
        print(f"{'固定杠杆'+str(me):<28} {r['cagr']*100:>9.2f}% {r['maxdd']*100:>8.2f}% {r['sharpe']:>7.2f} {r['calmar']:>8.2f} {te*100:>8.2f}% {'✅达标' if hit else '':>10}")
        results.append({'base':bn,'path':'fixed_lev','max_exp':me,**{k:r[k] for k in ['cagr','maxdd','sharpe','calmar']},'test':te,'hit':hit})
    # 路径2: 动态杠杆+趋势
    for me in [1.2, 1.3, 1.5]:
        for lo in [0.05, 0.15]:
            r = run(fw, max_exp=me, dyn=True, low_exp=lo)
            hit = r['cagr']>=0.35 and r['maxdd']>=-0.19
            te = seg_cagr(r['equity'], *segs['test'])
            tag = f"动态杠杆{me}/底{lo}"
            print(f"{tag:<28} {r['cagr']*100:>9.2f}% {r['maxdd']*100:>8.2f}% {r['sharpe']:>7.2f} {r['calmar']:>8.2f} {te*100:>8.2f}% {'✅达标' if hit else '':>10}")
            results.append({'base':bn,'path':'dyn_lev','max_exp':me,'low_exp':lo,**{k:r[k] for k in ['cagr','maxdd','sharpe','calmar']},'test':te,'hit':hit})
    # 路径3: 多资产
    for tn in [2, 3]:
        r = run(fw, max_exp=1.0, top_n=tn)
        print(f"{'多资产top'+str(tn)+'零杠杆':<28} {r['cagr']*100:>9.2f}% {r['maxdd']*100:>8.2f}% {r['sharpe']:>7.2f} {r['calmar']:>8.2f} {'':>9} {'':>10}")
        results.append({'base':bn,'path':'topn','top_n':tn,**{k:r[k] for k in ['cagr','maxdd','sharpe','calmar']},'hit':False})
    # 路径4: 高目标波动(零杠杆封顶)
    for vt in [0.6, 0.8]:
        r = run(fw, max_exp=1.0, vol_target=vt)
        print(f"{'目标波动'+str(vt)+'零杠杆':<28} {r['cagr']*100:>9.2f}% {r['maxdd']*100:>8.2f}% {r['sharpe']:>7.2f} {r['calmar']:>8.2f} {'':>9} {'':>10}")
        results.append({'base':bn,'path':'voltarget','vol_target':vt,**{k:r[k] for k in ['cagr','maxdd','sharpe','calmar']},'hit':False})

json.dump(results, open('out/path_exhaust.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)
hits = [r for r in results if r['hit']]
print("\n" + "="*100)
print(f"达标(≥35% 且 回撤≤19%)方案数: {len(hits)}")
for r in hits:
    print(f"  ✅ {r['base']} {r['path']} max_exp={r.get('max_exp')} low={r.get('low_exp')} : {r['cagr']*100:.2f}% / {r['maxdd']*100:.2f}%")
print("结果已存 out/path_exhaust.json")
