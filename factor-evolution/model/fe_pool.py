# -*- coding: utf-8 -*-
"""扩池引擎：支持任意风险资产池，alpha 基准固定为原5资产等权市场。
设计：
  - panel.close 列 = [原5风险资产..., 新风险资产..., BondETF(最后一列)]
  - RISK_IDX: 所有参与选券的风险资产列索引
  - MKT_IDX: 用于计算市场基准的资产（固定原5，保证 alpha 口径跨方案可比）
  - BOND_IDX: 最后一列（债券/现金补位）
  - r_alpha[t,j] = r[t,j] - r_mkt[t]，其中 r_mkt = MKT_IDX 等权收益
  - 评估真 alpha（相对原5资产市场）；账面用原始收益。
"""
import numpy as np
import pandas as pd


def detrend_market_pool(close, mkt_idx):
    with np.errstate(all='ignore'):
        ret = close[1:] / close[:-1] - 1.0
        rmkt = np.nanmean(ret[:, mkt_idx], axis=1)
        ra = ret - rmkt[:, None]
    out = np.full_like(close, np.nan)
    out[0] = 1.0
    for t in range(1, close.shape[0]):
        for j in range(close.shape[1]):
            prev = out[t-1, j] if np.isfinite(out[t-1, j]) else 1.0
            out[t, j] = prev * (1.0 + (ra[t-1, j] if np.isfinite(ra[t-1, j]) else 0.0))
    return out


def zscore_cross(mat, idx):
    idx = np.asarray(idx)
    out = np.full_like(mat, np.nan)
    for t in range(mat.shape[0]):
        row = mat[t, idx]; ok = np.isfinite(row)
        if ok.sum() >= 3:
            out[t, idx[ok]] = (row[ok] - row[ok].mean()) / (row[ok].std() + 1e-9)
    return out


def run_alpha_pool(panel, dclose, Fc, params, risk_idx, mkt_idx, bond_idx):
    p = params
    fw = p['factors']
    dret = np.zeros_like(dclose); dret[1:] = dclose[1:]/dclose[:-1]-1; dret[0] = np.nan
    oret = panel.ret; T, A = dclose.shape
    rebal=int(p.get('rebal',10)); vol_win=int(p.get('vol_win',20))
    vol_target=p.get('vol_target',0.40); vol_cap=p.get('vol_cap',0.25)
    top_n=int(p.get('top_n',1)); dd_limit=p.get('dd_limit',0.45)
    cost=p.get('cost_bps',10)/1e4; warmup=int(p.get('warmup',260))
    floor=p.get('timing_floor',0.0)

    vol=np.full((T,A),np.nan)
    for j in risk_idx:
        vol[:,j]=pd.Series(dret[:,j]).rolling(vol_win,min_periods=max(3,vol_win//2)).std().to_numpy()*np.sqrt(252)
    Fz={name:zscore_cross(Fc[name],risk_idx) for name,w in fw.items() if name in Fc and abs(w)>1e-12}

    eq_a=1.0; eq_b=1.0; peak_a=1.0
    ea=np.full(T,np.nan); eb=np.full(T,np.nan)
    cur_w=np.zeros(A); cur_exp=0.0; trades=0; turn=0.0; cooldown=0
    for t in range(warmup,T):
        da=0.0; db=0.0
        for j in risk_idx+[bond_idx]:
            if np.isfinite(cur_w[j]) and cur_w[j]!=0:
                if np.isfinite(dret[t,j]): da+=cur_w[j]*dret[t,j]
                if np.isfinite(oret[t,j]): db+=cur_w[j]*oret[t,j]
        eq_a*=(1+da); eq_b*=(1+db)
        ea[t]=eq_a; eb[t]=eq_b
        peak_a=max(peak_a,eq_a); dd=eq_a/peak_a-1

        if (t-warmup)%rebal==0 and t<T-1:
            score=np.zeros(A)
            for name,w in fw.items():
                if name in Fz: score+=w*np.nan_to_num(Fz[name][t],nan=0.0)
            elig=np.array([np.isfinite(vol[t,j]) and vol[t,j]<=vol_cap and np.isfinite(dclose[t,j]) for j in risk_idx])
            if elig.sum()==0:
                nw=np.zeros(A); nw[bond_idx]=1.0; ne=0.0
            else:
                cand=np.array(risk_idx)[elig]; k=min(top_n,len(cand))
                order=cand[np.argsort(-score[cand])][:k]
                nw=np.zeros(A)
                iv=1/np.maximum(np.nan_to_num(vol[t,order],nan=1.0),1e-6); iv/=iv.sum()
                for idx,j in enumerate(order): nw[j]=iv[idx]
                pv=float(np.sqrt(np.sum((iv*np.nan_to_num(vol[t,order],nan=0.0))**2)+2e-6))
                exp=np.clip(vol_target/(pv+1e-9),0,1)
                exp=np.clip(exp,floor,1.0)
                if dd<=-dd_limit: cooldown=int(p.get('dd_cooldown',10))
                if cooldown>0: exp*=0.5; cooldown-=1
                nw[bond_idx]=1-exp; ne=exp
            delta=np.abs(nw-cur_w).sum()+abs(ne-cur_exp)
            eq_a*=(1-cost*delta); eq_b*=(1-cost*delta)
            turn+=delta; trades+=int(np.sum(np.abs(nw[ risk_idx ]-cur_w[ risk_idx ])>1e-6))
            cur_w,cur_exp=nw,ne

    def metrics(curve):
        s=curve[warmup:]; s=s[np.isfinite(s)]
        if len(s)<60: return dict(cagr=0,maxdd=0,sharpe=0,calmar=0)
        s=s/s[0]; yrs=len(s)/252
        cagr=s[-1]**(1/yrs)-1
        pk=np.maximum.accumulate(s); mdd=float(np.min(s/pk-1))
        dr=np.diff(s)/s[:-1]; sh=float(np.mean(dr)/(np.std(dr)+1e-9)*np.sqrt(252))
        return dict(cagr=float(cagr),maxdd=mdd,sharpe=sh,calmar=float(cagr/abs(mdd)) if mdd<0 else 0)
    return dict(alpha_equity=ea,book_equity=eb,alpha=metrics(ea),book=metrics(eb),
                turn=float(turn),trades=trades)


def seg_metrics(curve,warm,segs):
    out={}
    for k,(a,b) in segs.items():
        s=curve[a:b+1]; s=s[np.isfinite(s)]
        if len(s)<60: out[k]=dict(cagr=0,maxdd=0,sharpe=0); continue
        s=s/s[0]; yrs=len(s)/252
        cagr=s[-1]**(1/yrs)-1; pk=np.maximum.accumulate(s); mdd=float(np.min(s/pk-1))
        dr=np.diff(s)/s[:-1]; sh=float(np.mean(dr)/(np.std(dr)+1e-9)*np.sqrt(252))
        out[k]=dict(cagr=float(cagr),maxdd=float(mdd),sharpe=float(sh))
    return out
