#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""生成第29章 GitHub 择时因子研究与验证报告（静态HTML，数据内联，离线可开，移动端兼容）"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "model" / "timing_out" / "data"
OUT = ROOT / "第29章GitHub择时因子研究报告.html"

V = json.load(open(D / "git_factor_validation.json", encoding="utf-8"))
B = json.load(open(D / "git_factor_backtest.json", encoding="utf-8"))


def bars_html(data, label="数值"):
    """单因子 AUC 条形图（SVG），data: {name: (dir_auc, source)}"""
    items = sorted(data.items(), key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float)) else kv[1].get("dir_auc", 0)))
    rows = []
    for name, val in items:
        a = val if isinstance(val, (int, float)) else val.get("dir_auc", 0)
        src = "" if isinstance(val, (int, float)) else val.get("source", "")
        rows.append((name, a, src))
    W, rowh = 720, 30
    H = 40 + len(rows) * rowh
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">']
    parts.append(f'<line x1="180" y1="28" x2="180" y2="{H}" stroke="#dde4ec"/>')
    parts.append('<line x1="180" y1="28" x2="700" y2="28" stroke="#fff"/>')
    for g in (0.5, 0.6, 0.7, 0.8):
        x = 180 + (g - 0.4) / 0.6 * 500
        parts.append(f'<line x1="{x:.0f}" y1="28" x2="{x:.0f}" y2="{H}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{x:.0f}" y="20" text-anchor="middle" font-size="11" fill="#8a97a8">{g}</text>')
    for i, (name, a, src) in enumerate(rows):
        y = 36 + i * rowh
        w = max(0, (a - 0.4) / 0.6 * 500)
        color = "#d93025" if a >= 0.6 else ("#b8860b" if a >= 0.55 else "#94a3b8")
        parts.append(f'<text x="172" y="{y+14}" text-anchor="end" font-size="12" fill="#22303f">{name}</text>')
        parts.append(f'<rect x="180" y="{y+3}" width="{w:.1f}" height="16" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{180+w+6:.0f}" y="{y+15}" font-size="11" fill="#475569">{a:.3f}</text>')
    parts.append('</svg>')
    return "".join(parts)


def nav_compare_svg():
    champ = B["champion"]
    key = "fip252|low|q0.1"
    exp = B["experiments"][key]
    cm, em = champ["cagr"] * 100, exp["metrics"]["cagr"] * 100
    items = [("冠军(冻结alpha·不做开关)", cm), ("fip252当避险开关", em)]
    W, H = 640, 160
    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">']
    p.append('<text x="20" y="24" font-size="13" fill="#132a4a" font-weight="700">年化收益对比（%）</text>')
    for i, (nm, v) in enumerate(items):
        y = 50 + i * 50
        w = abs(v) / 40 * 480
        c = "#2c5282" if i == 0 else "#d93025"
        p.append(f'<text x="20" y="{y+20}" font-size="12" fill="#22303f">{nm}</text>')
        p.append(f'<rect x="200" y="{y+4}" width="{w:.1f}" height="24" rx="4" fill="{c}"/>')
        p.append(f'<text x="{200+w+8:.0f}" y="{y+21}" font-size="13" font-weight="700" fill="{c}">{v:.2f}%</text>')
    p.append('</svg>')
    return "".join(p)


# 单因子表
single_rows = ""
for name, d in V["singles"].items():
    cls = "ok" if d.get("dir_auc", 0) >= 0.6 else ("warn" if d.get("dir_auc", 0) >= 0.55 else "")
    single_rows += (f'<tr><td>{name}</td><td>{d.get("source","")}</td>'
                    f'<td class="{cls}">{d.get("dir_auc",0):.3f}</td><td>{d.get("n",0)}</td></tr>')

# 稳健性表
rob_rows = ""
for name, d in V["robustness"].items():
    flag = '<span class="pill ok">全稳</span>' if d["all_above_55"] else '<span class="pill warn">不稳</span>'
    rob_rows += f'<tr><td>{name}</td><td>{" / ".join(str(a) for a in d["aucs"])}</td><td>{flag}</td></tr>'

mm = V["multi_model"]
sc = B["comparison"]

HTML = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>第29章 GitHub择时因子研究报告</title>
<style>
:root{{--navy:#132a4a;--brand:#2c5282;--gold:#b8860b;--ink:#22303f;--mut:#6b7a8c;--line:#dde4ec;--bg:#f4f6f9;--up:#d93025;--down:#0f9d58}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.7 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:0 20px 60px}}
.hero{{background:linear-gradient(135deg,#132a4a,#1f3a5f);color:#fff;padding:30px 0 26px;margin-bottom:22px}}
.hero .wrap{{padding-bottom:0}}.hero h1{{font-size:24px;margin:0 0 8px}}.hero .sub{{color:#c7d6e8;font-size:13px}}
.vtag{{display:inline-block;background:#e8b64c;color:#3a2a00;font-size:11px;font-weight:800;padding:3px 9px;border-radius:4px;margin-left:8px}}
.sec{{margin-top:24px}}.sec h2{{font-size:17px;color:var(--navy);border-left:4px solid var(--brand);padding-left:10px;margin:0 0 12px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:10px;box-shadow:0 2px 12px rgba(20,40,70,.05);padding:16px 18px;margin-bottom:14px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.kpi{{background:#fff;border:1px solid var(--line);border-radius:9px;padding:14px}}.kpi .k{{font-size:11px;color:var(--mut)}}.kpi .v{{font-size:20px;font-weight:800;color:var(--brand);font-variant-numeric:tabular-nums}}.kpi.hl .v{{color:var(--up)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line)}}th{{color:var(--mut);font-size:12px;background:#fbfcff}}td:not(:first-child),th:not(:first-child){{text-align:center}}
.ok{{color:var(--down);font-weight:700}}.warn{{color:var(--gold)}}.pill{{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px}}.pill.ok{{background:#eefaf2;color:#0b6b39}}.pill.warn{{background:#fffaef;color:#7a5a14}}
.note{{font-size:12px;color:var(--mut);line-height:1.9}}.note strong{{color:var(--ink)}}
.lesson{{background:#fff5f4;border:1px solid #f0c9c4;border-radius:9px;padding:14px 18px}}.lesson h3{{margin:0 0 6px;color:#a52a1e;font-size:15px}}
.adopt{{background:#f1fbf5;border:1px solid #bfe6cf;border-radius:9px;padding:14px 18px}}.adopt h3{{margin:0 0 6px;color:#0b6b39;font-size:15px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:760px){{.kpis{{grid-template-columns:repeat(2,1fr)}}.grid2{{grid-template-columns:1fr}}th,td{{font-size:11px;padding:6px}}}}
</style></head><body>
<div class="hero"><div class="wrap"><h1>第29章 · 从 GitHub 学习更优择时/预警因子<span class="vtag">机构版 · 零前视</span></h1>
<div class="sub">6 ETF 风格轮动 · 14 个市场级因子 · walk-forward 验证 · "高 AUC ≠ 好触发器"诚实对照　|　2026-11-21</div></div></div>
<div class="wrap">

<div class="kpis">
 <div class="kpi"><div class="k">实现因子</div><div class="v">{V['n_factors']}</div><div class="k">7 大类，逐条标注来源</div></div>
 <div class="kpi hl"><div class="k">最强因子 fip252</div><div class="v">{V['singles']['fip252']['dir_auc']:.3f}</div><div class="k">OOF dirAUC</div></div>
 <div class="kpi"><div class="k">时间外 fip252</div><div class="v">{V['holdout']['singles']['fip252']['dir_auc']:.3f}</div><div class="k">训练13-21→测试22-26</div></div>
 <div class="kpi"><div class="k">事件样本</div><div class="v">{V['n_events']}</div><div class="k">未来10日≤−8%，{V['event_rate']*100:.1f}%</div></div>
</div>

<div class="sec"><h2>一、因子库与来源（29.1）</h2>
<div class="card">{bars_html({k:d.get('dir_auc',0) for k,d in V['singles'].items()})}
<div class="note">因子全部由 t 日及之前数据构造，f[t] 在 t 收盘可得、预测 t+1 起；RSRS 用 5 只风险资产等权合成 high/low（本工程有真实 OHLC，优于文档"全收益指数需 ETF 合成"的条件）。<strong>RSRS 被证伪</strong>：rsrs_adj dirAUC≈0.501，未能迁移，与文档结论一致。</div></div>
<div class="card"><table><thead><tr><th>因子</th><th>来源（可查证）</th><th>OOF dirAUC</th><th>样本</th></tr></thead><tbody>{single_rows}</tbody></table></div>
</div>

<div class="sec"><h2>二、零前视验证（29.2）</h2>
<div class="grid2">
 <div class="card"><h3>多因子模型 OOF AUC</h3><table><thead><tr><th>模型</th><th>OOF AUC</th></tr></thead><tbody>
 <tr><td>仅现有因子</td><td>{mm['existing']['auc']:.3f}</td></tr>
 <tr><td>GitHub 新因子6</td><td class="ok">{mm['github_new']['auc']:.3f}</td></tr>
 <tr><td>稳健新因子6</td><td class="ok">{mm['robust6']['auc']:.3f}</td></tr>
 </tbody></table><div class="note">新因子作模型输入优于仅现有因子，与文档 29.2 一致。</div></div>
 <div class="card"><h3>干净时间外检验</h3><table><thead><tr><th>模型/因子</th><th>时间外 AUC</th></tr></thead><tbody>
 <tr><td>fip252</td><td class="ok">{V['holdout']['singles']['fip252']['dir_auc']:.3f}</td></tr>
 <tr><td>m5_z</td><td>{V['holdout']['singles']['m5_z']['dir_auc']:.3f}</td></tr>
 <tr><td>GitHub 新因子6</td><td class="ok">{V['holdout']['github_new']['auc']:.3f}</td></tr>
 <tr><td>仅现有因子</td><td>{V['holdout']['existing']['auc']:.3f}</td></tr>
 </tbody></table></div>
</div>
<div class="card"><h3>跨阈值稳健性（−10% / −6% / −8%，全部&gt;0.55 才算稳）</h3>
<table><thead><tr><th>因子</th><th>三档 dirAUC</th><th>判定</th></tr></thead><tbody>{rob_rows}</tbody></table>
<div class="note">fip252、diffusion20 跨三档全稳；m5_z 时间外 AUC 高但跨阈值不稳——警惕样本集中过拟合（与文档"稳健性存疑"一致）。</div></div>
</div>

<div class="sec"><h2>三、关键教训：高 AUC ≠ 好触发器（29.3）</h2>
<div class="card">{nav_compare_svg()}
<div class="lesson"><h3>证伪：把 fip252（AUC 最高）直接当避险开关</h3>
<div class="note">年化从 <strong>{B['champion']['cagr']*100:.2f}%</strong> 崩塌到 <strong>{sc['cagr_drop_pp']:.1f}pp 更低</strong>，而最大回撤<strong>完全没改善（{sc['mdd_change_pp']:+.1f}pp）</strong>——闸门只削减收益、未降低回撤。机理：fip252 是"趋势平滑度/拥挤度"因子，平稳上涨后大概率续涨（IC 为正），作为避险开关会在主升浪中反复误报、切掉盈利仓位。这正是"统计上预测下跌概率高"与"应现在减仓"之间的鸿沟。</div></div></div>
</div>

<div class="sec"><h2>四、采纳与边界（29.4）</h2>
<div class="adopt"><h3>采纳：作为观察指标，不作为开关</h3>
<div class="note">将 <strong>fip252、kurt20、diffusion20</strong> 纳入监管层<strong>观察指标</strong>（拥挤/尾部风险），作为 ML 预警模型输入特征在影子台账跟踪；<strong>不</strong>用它们直接替换上线避险触发器（回测证伪）；不因单档高 AUC 宣称有效（跨阈值需全稳）。净结论：维持冠军 v2.0 冻结 alpha 不变，<strong>只提名、不自动切换</strong>——与第27节监管层一致。</div></div>
<div class="card"><div class="note"><strong>诚实边界：</strong>{V['note']} 事件样本少且集中、参数样本内选取；本报告为研究诊断，不构成上线依据。未改动线上策略、看板或每日工作流。</div></div>
</div>

</div></body></html>"""

OUT.write_text(HTML, encoding="utf-8")
print(f"报告已生成：{OUT}（{len(HTML)} 字符）")
