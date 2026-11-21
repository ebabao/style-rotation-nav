"""
factor-evolution · 机器学习训练对照
=====================================
用 Alpha101/Alpha158/第29章共49因子作特征, 预测未来10日横截面收益排名。
严格 purged walk-forward: 训练标签窗口与预测点完全错开, 杜绝前视。
三种模型: LogisticRegression / RandomForest / XGBoost。
ML 打分注入独立引擎, 与 GA 规则进化对照。
"""
import numpy as np, os, json, sys
import fe_factors as FF
import fe_engine as FE

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')


def make_labels(panel, h=10, gap=1):
    """标签: t 日 → t+gap ~ t+gap+h-1 收益 (与 t 日因子错开)。返回 (T,A)。"""
    c = panel.close; T, A = c.shape
    lab = np.full((T, A), np.nan)
    for t in range(T - gap - h + 1):
        lab[t] = c[t + gap + h - 1] / c[t + gap - 1] - 1.0
    # 横截面 rank pct (去漂移)
    for t in range(T):
        ok = np.isfinite(lab[t, :5])
        if ok.sum() >= 3:
            ranks = np.argsort(np.argsort(lab[t, :5][ok])) / max(1, ok.sum()-1)
            lab[t, np.where(ok)[0]] = ranks - 0.5
        else:
            lab[t, :5] = np.nan
    return lab


def ml_scores(panel, F, model_kind, h=10, gap=1, step=63, train_win=1200, min_train=400):
    """purged walk-forward 生成 (T,A) 打分矩阵。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None
    T, A = panel.close.shape
    names = list(F.keys())
    X = np.column_stack([F[n].ravel() for n in names]).reshape(T, A, len(names))
    lab = make_labels(panel, h, gap)
    scores = np.zeros((T, A))
    # 特征标准化(横截面 z-score, 逐因子)
    Xz = np.full_like(X, np.nan)
    for k in range(len(names)):
        Xz[:, :, k] = FE.zscore_cross(X[:, :, k])

    def new_model():
        if model_kind == 'logit':
            return LogisticRegression(max_iter=300, C=0.1)
        if model_kind == 'rf':
            return RandomForestClassifier(n_estimators=120, max_depth=4,
                                          min_samples_leaf=20, random_state=42, n_jobs=-1)
        if model_kind == 'xgb' and XGBClassifier is not None:
            return XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                                 n_jobs=-1, eval_metric='logloss')
        return LogisticRegression(max_iter=300, C=0.1)

    # 二分类: 未来收益排名 top40% = 1
    y_all = (lab > 0.1).astype(int)
    for tau in range(min_train, T, step):
        lo = max(0, tau - train_win)
        hi = tau - h - gap  # purged: 标签窗口截止到 tau 之前
        if hi - lo < 200:
            continue
        Xtr = Xz[lo:hi+1, :5, :].reshape(-1, len(names))
        ytr = y_all[lo:hi+1, :5].ravel()
        ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
        if ok.sum() < 100 or len(np.unique(ytr[ok])) < 2:
            continue
        m = new_model()
        m.fit(Xtr[ok], ytr[ok])
        # 预测 tau..tau+step 的打分 (用 tau 日可见的特征)
        for t in range(tau, min(tau+step, T)):
            Xte = Xz[t, :5, :]
            okte = np.isfinite(Xte).all(axis=1)
            if okte.sum() >= 2:
                prob = m.predict_proba(Xte[okte])[:, 1]
                scores[t, np.where(okte)[0]] = prob - 0.5
    return scores


def main():
    panel, dates = FF.load(DATA)
    F = FF.build_factors(panel)
    # 用 GA 冠军的执行参数(若有), 否则默认
    champ = {}
    evo_path = os.path.join(OUT, 'evolution_result.json')
    base = dict(rebal=10, vol_win=20, vol_target=0.40, vol_cap=0.25,
                top_n=1, dd_limit=0.45, cost_bps=10, warmup=260, factors={})
    if os.path.exists(evo_path):
        d = json.load(open(evo_path, encoding='utf-8'))
        base.update({k:v for k,v in d['champion_params'].items()})
        champ['factors'] = d['champion_factors']
    fw = champ.get('factors') or {'mom250':1.0,'ddhigh60':0.8}

    def seg(a,b):
        s=panel.close[warm:]; # 简化: 全样本 + 切段
    results = {}
    # 规则基线(用GA冠军因子)
    rb = FE.run(panel, F, {**base, 'factors':fw})
    results['rule_baseline'] = {'cagr':rb['cagr'],'maxdd':rb['maxdd'],'sharpe':rb['sharpe']}
    print('规则基线(GA冠军因子): CAGR %.2f%% MDD %.2f%% Sharpe %.3f'%(
        rb['cagr']*100, rb['maxdd']*100, rb['sharpe']))

    for kind in ['logit','rf','xgb']:
        print('训练 %s ...'%kind, flush=True)
        sc = ml_scores(panel, F, kind)
        # 纯ML打分(inject覆盖): factors置空, 用scores
        r = FE.run(panel, F, {**base, 'factors':{}}, scores=sc)
        # 混合: 规则因子 + ML 打分
        r2 = FE.run(panel, F, {**base, 'factors':fw}, scores=sc)
        results[kind] = {
            'ml_only': {'cagr':r['cagr'],'maxdd':r['maxdd'],'sharpe':r['sharpe']},
            'hybrid': {'cagr':r2['cagr'],'maxdd':r2['maxdd'],'sharpe':r2['sharpe']},
        }
        print('  %s ML-only CAGR %.2f%%/MDD %.2f%%/Sharpe %.3f | 混合 CAGR %.2f%%/MDD %.2f%%/Sharpe %.3f'%(
            kind, r['cagr']*100,r['maxdd']*100,r['sharpe'],
            r2['cagr']*100,r2['maxdd']*100,r2['sharpe']))
    json.dump(results, open(os.path.join(OUT,'ml_result.json'),'w',encoding='utf-8'),
              ensure_ascii=False,indent=2)
    print('结果已存:',os.path.join(OUT,'ml_result.json'))


if __name__=='__main__':
    main()
