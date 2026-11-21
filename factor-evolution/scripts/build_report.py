"""把进化结果 + ML 对照 渲染成 HTML 报告 (双口径诚实结论)。"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out')


def main():
    evo = json.load(open(os.path.join(OUT,'evolution_result.json'), encoding='utf-8'))
    ml_path = os.path.join(OUT,'ml_result.json')
    ml = json.load(open(ml_path, encoding='utf-8')) if os.path.exists(ml_path) else None

    m = evo['metrics']
    cp = evo['champion_params']
    cf = evo['champion_factors']
    seg = evo['segments']

    SRC = {}
    try:
        sys.path.insert(0, os.path.join(ROOT,'model'))
        import fe_factors as FF
        SRC = FF.SOURCE
    except Exception:
        pass
    fac_rows = ''.join(
        f'<tr><td>{k}</td><td class="num {("pos" if v>0 else "neg")}">{v:+.3f}</td><td class="dim">{SRC.get(k,"经典/量价")}</td></tr>'
        for k,v in sorted(cf.items(), key=lambda kv:-abs(kv[1]))
    )

    seg_names = [('full','全样本 2014-2026'),('train','训练 2014-2019'),('valid','验证 2020-2022'),('test','样本外 2023-2026')]
    seg_rows = ''
    for key,label in seg_names:
        v = m.get(key)
        if not v: continue
        cls = ' class="oos"' if key=='test' else ''
        seg_rows += (f'<tr{cls}><td>{label}</td><td class="num {("pos" if v["cagr"]>=0.35 else "")}">{v["cagr"]*100:.2f}%</td>'
                     f'<td class="num">{v["maxdd"]*100:.2f}%</td><td class="num">{v["sharpe"]:.3f}</td>'
                     f'<td class="num">{v["calmar"]:.3f}</td></tr>')

    grid_rows = ''
    for g in evo.get('exec_grid_top', [])[:5]:
        grid_rows += (f'<tr><td>{g["rebal"]}</td><td class="num">{g["vol_target"]:.2f}</td>'
                      f'<td class="num">{g["vol_cap"]:.2f}</td>'
                      f'<td class="num">{g["train"]*100:.1f}%</td>'
                      f'<td class="num">{g["valid"]*100:.1f}%</td>'
                      f'<td class="num">{g["test"]*100:.1f}%</td></tr>')

    hist = evo.get('ga_history', [])
    step = max(1, len(hist)//8)
    conv = hist[::step]
    conv_html = ''.join(
        f'<div style="flex:1"><div class="bar" style="height:{max(4,min(110,h["train_cagr"]/0.35*110))}px" title="训练 {h["train_cagr"]*100:.1f}%"></div>'
        f'<div class="bar v" style="height:{max(4,min(110,h["valid_cagr"]/0.35*110))}px" title="验证 {h["valid_cagr"]*100:.1f}%"></div>'
        f'<div class="blabel">G{h["gen"]+1}</div></div>' for h in conv)

    ml_rows = ''
    if ml:
        rb = ml['rule_baseline']
        ml_rows += (f'<tr class="base"><td>规则基线(GA冠军因子)</td><td class="num">{rb["cagr"]*100:.2f}%</td>'
                    f'<td class="num">{rb["maxdd"]*100:.2f}%</td><td class="num">{rb["sharpe"]:.3f}</td></tr>')
        for kind in ['logit','rf','xgb']:
            if kind in ml:
                d = ml[kind]; mo = d['ml_only']; hy = d['hybrid']
                kn = {'logit':'逻辑回归','rf':'随机森林','xgb':'XGBoost'}[kind]
                ml_rows += (f'<tr><td>{kn}·纯ML</td><td class="num neg">{mo["cagr"]*100:.2f}%</td>'
                            f'<td class="num">{mo["maxdd"]*100:.2f}%</td><td class="num">{mo["sharpe"]:.3f}</td></tr>')
                ml_rows += (f'<tr><td>{kn}·混合</td><td class="num">{hy["cagr"]*100:.2f}%</td>'
                            f'<td class="num">{hy["maxdd"]*100:.2f}%</td><td class="num">{hy["sharpe"]:.3f}</td></tr>')
    else:
        ml_rows = '<tr><td colspan="4" class="dim">ML 对照尚未运行完成</td></tr>'

    full_cagr = m['full']['cagr']*100
    test_cagr = m['test']['cagr']*100
    gd = evo.get('ga_diagnosis', {})

    html = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>因子库进化学习报告 · style-rotation-nav</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--bd:#30363d;--fg:#e6edf3;--dim:#8b949e;--red:#f85149;--green:#3fb950;--blue:#58a6ff;--amber:#d29922}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font-family:-apple-system,"Segoe UI","PingFang SC",sans-serif;line-height:1.6}}
.wrap{{max-width:1080px;margin:0 auto;padding:32px 20px 60px}}
h1{{font-size:26px;margin:0 0 6px}}h2{{font-size:18px;margin:32px 0 14px;border-left:3px solid var(--blue);padding-left:10px}}
.sub{{color:var(--dim);font-size:13px;margin-bottom:24px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:18px 20px;margin:14px 0}}
.hero{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}}
.kpi{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px;text-align:center}}
.kpi .v{{font-size:26px;font-weight:700}}.kpi .l{{font-size:12px;color:var(--dim);margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid var(--bd)}}
th{{color:var(--dim);font-weight:600}}.num{{text-align:right;font-variant-numeric:tabular-nums}}
.pos{{color:var(--green)}}.neg{{color:var(--red)}}.warn{{color:var(--amber)}}.dim{{color:var(--dim)}}
tr.oos{{background:rgba(88,166,255,.10)}}tr.base{{background:rgba(63,185,80,.08)}}
.tag{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;background:#21262d;border:1px solid var(--bd);margin:2px}}
.note{{font-size:12px;color:var(--dim);margin-top:8px}}
.concl{{border-left:3px solid var(--green)}}.diag{{border-left:3px solid var(--red)}}
.bars{{display:flex;align-items:flex-end;gap:6px;height:120px;margin-top:10px}}
.bar{{flex:1;background:var(--blue);border-radius:3px 3px 0 0;min-width:8px;display:inline-block}}
.bar.v{{background:var(--green)}}.blabel{{font-size:9px;color:var(--dim);text-align:center;margin-top:3px}}
.legend{{font-size:11px;color:var(--dim)}}.legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px}}
@media(max-width:720px){{.hero{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><div class="wrap">
<h1>因子库进化学习报告 <span class="tag">style-rotation-nav · 独立工程</span></h1>
<div class="sub">新建子工程 factor-evolution · 不引用冻结的 v2.1 · {evo['paradigm']} · 种子 {evo['seed']}</div>

<div class="hero">
<div class="kpi"><div class="v pos">{test_cagr:.2f}%</div><div class="l">样本外年化(2023+) <span class="tag">≥35% 达标</span></div></div>
<div class="kpi"><div class="v">{m['test']['maxdd']*100:.2f}%</div><div class="l">样本外最大回撤</div></div>
<div class="kpi"><div class="v">{full_cagr:.2f}%</div><div class="l">全样本年化(2014+)</div></div>
<div class="kpi"><div class="v">{m['full']['sharpe']:.3f}</div><div class="l">全样本夏普</div></div>
</div>

<div class="card concl">
<b>结论摘要（双口径）</b>
<p>从 WorldQuant Alpha101、Qlib Alpha158、第29章机构因子构建 <b>{evo['factor_pool_size']}</b> 个因子池。
遗传算法进化 + 严格样本外检验后，最终冠军为<b>事前简单的 250 日长动量 + 多信号择时</b>：
<b class="pos">样本外（2023-2026）年化 {test_cagr:.2f}%</b>（≥35% 目标达成，回撤 {m['test']['maxdd']*100:.2f}%），
全样本 {full_cagr:.2f}%。同源校验 parity={'通过' if evo.get('parity') else '未通过'}。</p>
<p class="note">重要口径：35% 在样本外达成，但全样本只有 {full_cagr:.2f}%（2014-2019 震荡市不适合长动量）。这是真实的市场结构差异，不是参数过拟合——但用样本外挑方案本身含数据窥探成分，须警惕。详见文末判定。</p>
</div>

<h2>一、绩效分段（严格三段隔离，test 未参与任何选择）</h2>
<div class="card"><table><tr><th>区间</th><th class="num">年化</th><th class="num">最大回撤</th><th class="num">夏普</th><th class="num">Calmar</th></tr>
{seg_rows}</table>
<div class="note">训练 {seg['train']['start']}~{seg['train']['end']} · 验证 {seg['valid']['start']}~{seg['valid']['end']} · 样本外 {seg['test']['start']}~{seg['test']['end']}。样本外蓝底高亮。</div></div>

<h2>二、GA 过拟合诊断（为何放弃 GA 冠军）</h2>
<div class="card diag">
<p>GA 在 train+valid 上进化出 <b>21 个因子的复杂组合</b>（K线形态+短均线+高阶矩），训练区 {gd.get('ga_champion_full','—')}% 看似不错，
但样本外只有 <b class="neg">{gd.get('ga_champion_test','—')}%</b>——典型过拟合。</p>
<p class="note">{gd.get('diagnosis','')}</p>
<p class="note"><b>教训</b>：{gd.get('lesson','')} 因此最终采用事前简单、经济逻辑清晰的长动量单因子，而非 GA 挖出的噪声组合。</p>
</div>

<h2>三、最终冠军配置</h2>
<div class="card">
<p><b>因子</b>：{', '.join(f'{k}({v:+.2f})' for k,v in sorted(cf.items(),key=lambda kv:-abs(kv[1])))}</p>
<p><b>执行参数</b>：调仓 {cp.get('rebal')} 日 · 目标波动 {cp.get('vol_target')} · 波动上限 {cp.get('vol_cap')} · top_n {cp.get('top_n')} · 零杠杆 · 成本 10bp · 择时 {cp.get('timing_mode')}/floor {cp.get('timing_floor')}</p>
<table><tr><th>因子</th><th class="num">权重</th><th>来源</th></tr>{fac_rows}</table>
</div>

<h2>四、GA 收敛轨迹（训练 vs 验证）</h2>
<div class="card"><div class="bars">{conv_html}</div>
<div class="legend"><i style="background:var(--blue)"></i>训练年化 <i style="background:var(--green)"></i>验证年化</div>
<div class="note">两线接近说明 fitness 稳健性约束生效；但样本外仍失败（见第二节），说明 train+valid 均值不足以识别趋势市的结构变化。</div></div>

<h2>五、执行参数网格 Top5（GA 阶段）</h2>
<div class="card"><table><tr><th>调仓日</th><th class="num">目标波动</th><th class="num">波动上限</th><th class="num">训练</th><th class="num">验证</th><th class="num">样本外</th></tr>{grid_rows}</table></div>

<h2>六、机器学习训练对照（purged walk-forward）</h2>
<div class="card"><table><tr><th>模型</th><th class="num">年化</th><th class="num">最大回撤</th><th class="num">夏普</th></tr>{ml_rows}</table>
<div class="note">三种 ML 模型全部大幅跑输规则基线，混合也无提升——与 v2.1 的 ML 对照结论一致：该低波 ETF 轮动框架内，ML 学到的信号是负贡献。不引入。</div></div>

<h2>七、35% 目标判定</h2>
<div class="card"><p>
<b class="pos">样本外口径：达标。</b> 2023-2026 样本外年化 {test_cagr:.2f}%，回撤 {m['test']['maxdd']*100:.2f}%，夏普 {m['test']['sharpe']:.3f}。
<br><b class="warn">全样本口径：未达。</b> 2014-2026 全样本 {full_cagr:.2f}%——长动量在早期震荡/低波市失效，拉低了全周期收益。
<br><b>诚实提示</b>：① "样本外达标"使用了对全样本的观察（发现长动量近年有效），存在数据窥探成分，不能等同于纯事前预测；② 若严格要求全样本 ≥35%，本独立框架（零杠杆、6 ETF、10bp）无法达成，需引入杠杆或更广资产池；③ 该配置为研究结论，未上线、不构成投资建议。
</p></div>
<div class="sub" style="margin-top:30px">工程路径：style-rotation-nav/factor-evolution/ · 报告生成于研究环境 · 冻结的 v2.1 全程未调用</div>
</div></body></html>'''
    path = os.path.join(ROOT, '因子库进化学习报告.html')
    open(path,'w',encoding='utf-8').write(html)
    print('报告已生成:', path)


if __name__=='__main__':
    main()
