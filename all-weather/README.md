# 全天候策略 v1.4 · 每日净值看板

本目录按 `ebabao/style-rotation-nav` 的既有架构部署：**静态页面 + `data/` 公开接口 + GitHub Actions 每日提交 + GitHub Pages 自动发布**。

- 看板：`https://ebabao.github.io/style-rotation-nav/all-weather/`
- 仓库：`https://github.com/ebabao/style-rotation-nav/tree/main/all-weather`
- 数据接口：`data/meta.json`、`nav.json`、`signal.json`、`weights.json`、`rebalances.csv`、`monthly.csv`、`sync_audit.json`

## 每日同步链

1. `scripts/sync_daily.py` 复用 `model/aw14/data14.py` 抓取 7 只真实 ETF 前复权行情；重试至多 2 次、硬超时 150 秒。
2. 复用 `model/aw14/engine14.py` 与 `model/active_params.json` 回放策略，生成公开接口。
3. 校验净值有限性、日期单调、序列长度与 `as_of` 一致性，并与训练快照对账。
4. 有数据变化才提交 `chore(nav): 更新全天候每日净值看板`；既有 Pages（`main / root`）自动发布子站。
5. 页面通过 `fetch('./data/*.json', {cache:'no-store'})` 取数，不内嵌数据、不引用外部 CDN，因此无需每日重建 HTML。

## 失败回退与停更兜底

- 抓数失败：恢复仓库内上一版 parquet 快照，标记 `source=cache_fallback`，继续生成接口，**不编造行情**。
- 接口校验失败：放弃当次写入，线上沿用上一版。
- 幂等：数据与参数指纹未变时只追加 `skipped` 巡检记录。
- 参数回退：删除 `model/active_params.json` 后重跑，即回到引擎默认基准参数。
- 工作流：仓库根目录 `.github/workflows/all_weather_sync.yml` 每日 UTC 11:20（北京 19:20）运行；`all_weather_watchdog.yml` 每 3 小时巡检，停更超过 26 小时自动重启。

每次运行均在 `data/sync_audit.json` 留痕：`run_id / scraped / new / md / status / source / retry / fallback / reconcile_pp`。

## 本地复现

```bash
cd all-weather
pip install -r requirements.txt
python3.11 scripts/sync_daily.py --no-refresh   # 用仓库快照回放
python3.11 scripts/sync_daily.py                # 抓数并完整同步
python3.11 -m http.server 8000                  # http://localhost:8000
```

必须通过 HTTP 访问页面；`file://` 无法读取接口。

## 研究边界

评估期自 2019-12-06 起，7 只 ETF 全部为真实行情、无指数代理。回测不代表未来收益；QDII 场内溢价与汇率影响未剥离。本工程仅用于策略研究与演示，不构成投资建议。
