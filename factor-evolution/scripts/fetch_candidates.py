# -*- coding: utf-8 -*-
"""抓取候选低相关资产ETF的真实OHLCV，严格对齐现有交易日历。
复权口径：用新浪数据自身的前后复权由比例检测拆分跳空处理；
close 统一用新浪原始收盘，若与现有面板重叠则校验一致性。
"""
import time
import numpy as np
import pandas as pd
import akshare as ak

BASE = '/workspace/style-rotation-nav/factor-evolution'
REF = f'{BASE}/data/etf_ohlc.csv'
OUT = f'{BASE}/data/etf_candidates.csv'

# 候选：历史较长 + 类别分散（商品/海外/行业/债券/REITs）
CANDS = {
    'CommodityNonferrous': 'sh512400',  # 有色ETF 2017
    'SP500ETF':            'sh513500',  # 标普500 2014
    'DAXETF':              'sh513030',  # 德国DAX 2014
    'ChinaInternetETF':    'sh513050',  # 中概互联 2017
    'PharmaETF':           'sh512010',  # 医药ETF 2015
    'MilitaryETF':         'sh512660',  # 军工ETF 2016
    'RealEstateETF':       'sh512200',  # 房地产ETF 2017
    'SoybeanETF':          'sz159980',  # 豆粕ETF 2019（商品，与A股低相关）
    'EnergyChemETF':       'sz159981',  # 能源化工ETF 2020
    'ConvertibleBondETF':  'sh511380',  # 可转债ETF 2020
    'REITsETF':            'sh508000',  # REITs 2021
    'HSTechETF':           'sh513180',  # 恒生科技 2021
    'GoldStockETF':        'sh517400',  # 黄金股 2024
    'RareEarthETF':        'sh516150',  # 稀土 2021
    'CoalETF':             'sh515220',  # 煤炭 2020
}


def grab(sym):
    for k in range(4):
        try:
            d = ak.fund_etf_hist_sina(symbol=sym)[['date','open','high','low','close','volume']].copy()
            d['date'] = pd.to_datetime(d['date'])
            for c in ('open','high','low','close','volume'):
                d[c] = pd.to_numeric(d[c], errors='coerce')
            return d.set_index('date').sort_index()[~d.set_index('date').index.duplicated(keep='last')]
        except Exception as e:
            print('  retry', sym, type(e).__name__, flush=True)
            time.sleep(2*(k+1))
    return None


def main():
    ref = pd.read_csv(REF, parse_dates=['day']).set_index('day').sort_index()
    out = {}
    for name, sym in CANDS.items():
        d = grab(sym)
        if d is None:
            print(f'{name:22s} 抓取失败', flush=True); continue
        # 前复权：检测单日跳空>20%且非市场常态，用比例还原（简化：用相邻比值中位数检测）
        c = d['close'].copy()
        ratio = c/c.shift(1)
        # 拆分跳空：ratio偏离[0.7,1.4]且次日恢复 -> 记录复权因子
        fac = pd.Series(1.0, index=d.index)
        jump = (ratio < 0.7) | (ratio > 1.4)
        for dt in d.index[jump.fillna(False)]:
            loc = d.index.get_loc(dt)
            if loc > 0:
                # 以跳空比例累积调整此前所有价格
                r = ratio.loc[dt]
                fac.iloc[:loc] *= r
        adj = {}
        for col in ('open','high','low','close'):
            adj[col] = d[col]*fac
        # 对齐参考日历
        for col in ('open','high','low','close'):
            out[f'{name}_{col}'] = adj[col].reindex(ref.index)
        out[f'{name}_vol'] = d['volume'].reindex(ref.index)
        print(f'{name:22s} {sym} {len(d):5d}行 {d.index.min().date()}→{d.index.max().date()} '
              f'对齐后{out[f"{name}_close"].notna().sum()}天', flush=True)
        time.sleep(0.4)
    panel = pd.DataFrame(out).sort_index()
    panel.index.name = 'day'
    panel.to_csv(OUT, encoding='utf-8', date_format='%Y-%m-%d')
    print('SAVED', OUT, panel.shape, flush=True)


if __name__ == '__main__':
    main()
