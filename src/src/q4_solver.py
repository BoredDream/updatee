"""
2026 高教社杯 C 题 —— 问题 4：波动电价下的购电策略
==================================================
在问题 2/3 的基础上，电价 p_{d,t} 由确定的分时电价变成附件 4 的随机过程。

与问题 3 的三点本质差别
  1) 0:00 时当天电价未知，必须先建电价预测模型；6/12/18 点可用已实现电价滚动修正；
  2) 违约价/紧急价都定义为"交易时刻电价"的倍数，所以相对价格结构不变，
     费用化简式 C_t = 0.5 p x + 0.5 p q + p (q-x)^+ + 5 p z 原样成立；
  3) 本模型有意采用混合价格目标：不带场景下标的共享合同决策使用当前阶段的
     中心预测价 p_center；未来合同 q^k、超计划项 u^k 和紧急购电 z^k 等
     价格相关 recourse 项与价格场景配对，使用 p^k。
     价格残差场景不为满足 p_center=mean_k(p^k) 而重新中心化，因此该目标不是
     将全部价格项置于同一场景均值下的严格统一 SAA 期望费用。

口径（与问题 1/2/3 一致，见 README「固定口径」）：
  * 区间起点和物理参数一致，但报表时间框不同：计划/调整保留模板行，
    实际执行、SOC、紧急购电和费用跨行映射为自然日00:00--24:00；
  * 充/放电量均定义在交流母线侧，S_j = S_{j-1} + eta*c_j - g_j/eta，
    两者单时段上限同为 CMAX = 5000*DT kWh（母线侧）；
  * 结算按题面口径 C = p*min(x,q) + 1.5p*(q-x)^+ + 0.5p*(x-q)^+ + 5p*z，
    其中 p 为【实际】电价。

运行（用 scripts/export_q4.py，见 docs/q4_model.md）。
"""
from pathlib import Path

import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.optimize import linprog

import q3_multistage as Q3BASE
from q3_multistage import (UP, T, DT, ETA_CHARGE, ETA_DISCHARGE, SMIN, SMAX, CMAX, LAM, TSTAGE, ND, P,
                           DATES, DSTR, L, G, load_hat, pv_hat, dispatch,
                           clock_min, dispatch_locked_interval, midnight_center,
                           midnight_actual, natural_emergency_segments)
from efficiency import EfficiencyParameters

# Compatibility alias for existing validators when both one-way efficiencies match.
ETA = ETA_CHARGE


def set_efficiency(parameters: EfficiencyParameters) -> None:
    """Set the efficiency convention before a backtest."""
    global ETA_CHARGE, ETA_DISCHARGE, ETA
    Q3BASE.set_efficiency(parameters)
    ETA_CHARGE = parameters.eta_charge
    ETA_DISCHARGE = parameters.eta_discharge
    ETA = ETA_CHARGE

# ----------------------------------------------------------------------
# 0. 电价数据
# ----------------------------------------------------------------------
PMAT = pd.read_excel(UP+'附件4.xlsx', header=0).iloc[:, 1:].to_numpy(float)   # 365x144 元/kWh
WIN = 20            # 电价形态滚动窗口(天)
PHI_PRIOR = 0.0     # 无历史时不外推瞬时偏离；不使用评价期离线调参
ALPHA = 5.0         # LP 中紧急电价倍数
PRICE_OBJECTIVE_POLICY = {
    "shared_price_basis": "center_forecast",
    "recourse_price_basis": "scenario_price",
    "objective_type": "hybrid_center_and_scenario_recourse",
    "scenario_price_centering": "not_recentered",
}
_PHAT_CACHE = {}
_PHI_CACHE = {}
# 附件2缺少2025-01-01 00:00--00:10。Q4显式保留附件1仅作这一段冷启动，
# 不把它混入附件4结算价或交付期统计。
Q4_COLD_MIDNIGHT_LG = midnight_actual(0)


def price_objective_policy():
    """返回 Q4-3 价格目标口径；返回副本以免导出脚本改动模块常量。"""
    return dict(PRICE_OBJECTIVE_POLICY)


def midnight_center4(d):
    return midnight_center(d) if d > 0 else Q4_COLD_MIDNIGHT_LG


def midnight_actual4(d):
    return midnight_actual(d) if d > 0 else Q4_COLD_MIDNIGHT_LG

# ----------------------------------------------------------------------
# 1. 电价预测子模型
# ----------------------------------------------------------------------
def price_ar_phi(d, m):
    """仅由决策时已完整实现的历史价格行估计AR(1)衰减系数。"""
    key = (d, m)
    if key in _PHI_CACHE:
        return _PHI_CACHE[key]
    end = d - 1 if m == 0 else d  # exclusive；阶段0排除含未实现午夜末列的d-1行
    hist = PMAT[max(0, end-WIN):end]
    if len(hist) < 2:
        phi = PHI_PRIOR
    else:
        normalized = hist / np.maximum(hist.mean(1, keepdims=True), 1e-9)
        residual = np.log(np.maximum(normalized, 1e-9))
        x, y = residual[:, :-1].ravel(), residual[:, 1:].ravel()
        denom = float(x @ x)
        phi = PHI_PRIOR if denom <= 1e-12 else float(np.clip((x @ y) / denom, 0.0, 0.99))
    _PHI_CACHE[key] = phi
    return phi


def clear_price_caches():
    """供因果扰动测试在修改输入后清除派生量缓存。"""
    _PHAT_CACHE.clear()
    _PHI_CACHE.clear()


def price_hat(d, m):
    """两因子电价预测： p̂ = 形态_t × 水平_d × 日内AR(1)修正。

      * 形态/水平：仅用决策时已经完整实现的近 WIN 个模板行；
        0:00排除d-1行，因为其末列是当天尚未实现的00:00--00:10；
      * 日内更新(m>=1)：用 [0,t_m) 已实现电价算水平比与瞬时偏离，
        偏离衰减系数也仅由上述完整历史行滚动估计。
    """
    key = (d, m)
    if key in _PHAT_CACHE:
        return _PHAT_CACHE[key]
    end = d - 1 if m == 0 else d
    lo = max(end-WIN, 0)
    if end <= 0:
        # Q4没有附件4历史。显式以附件1的已知分时价格作为冷启动先验，
        # 不能把1月1日尚未实现的附件4价格当成预测。
        base = P.copy()
    else:
        win = PMAT[lo:end]
        shape = win.mean(0) / win.mean()
        level = 0.5*PMAT[end-1].mean() + 0.5*win.mean()
        base = shape * level
    tm = TSTAGE[m]
    if tm > 0:
        r = PMAT[d, :tm] / np.maximum(base[:tm], 1e-6)
        lvl = float(np.clip(r[-18:].mean(), 0.6, 1.6))       # 近 3 小时水平比
        dev = float(np.clip(r[-1]/lvl, 0.5, 2.0))            # 瞬时偏离
        k = np.arange(1, T-tm+1)
        base = base.copy()
        phi = price_ar_phi(d, m)
        base[tm:] = base[tm:] * lvl * (1 + (dev-1)*phi**k)
    out = np.maximum(base, 1e-3)
    _PHAT_CACHE[key] = out
    return out

def water_value(phat):
    """末端储能水价：用当日预测的最低 12 个时段均价折算的重置成本 p_谷/η。"""
    return float(np.sort(phat)[:12].mean() / ETA_DISCHARGE)

# ----------------------------------------------------------------------
# 2. 联合场景生成（价格 / 负载 / 光伏必须同日配对抽样以保留相关性）
# ----------------------------------------------------------------------
def scenarios4(d, m, K=30):
    Lh, Gh, Ph = load_hat(d, m), pv_hat(d, m), price_hat(d, m)
    sl, sg, spz = [], [], []
    first_lag = 2 if m == 0 else 1
    for j in range(first_lag, first_lag+K):
        dd = d - j
        if dd < 10:
            continue
        sl.append(np.maximum(Lh + (L[dd] - load_hat(dd, m)), 0))
        sg.append(np.maximum(Gh + (G[dd] - pv_hat(dd, m)), 0))
        # 价格用乘性残差，避免出现负价
        ratio = PMAT[dd] / np.maximum(price_hat(dd, m), 1e-6)
        spz.append(np.maximum(Ph * np.clip(ratio, 0.3, 3.0), 1e-3))
    if not sl:
        sl, sg, spz = [Lh], [Gh], [Ph]
    return sl, sg, spz


def midnight_price_center(d):
    """午夜首段价格预测使用上一已完成的23:50--24:00区间。"""
    if d > 0:
        return float(PMAT[d-1, T-2])
    # Q4没有更早历史；该常数是显式冷启动先验，只影响1月1日预热起点。
    return float(P[-1])


def midnight_price_actual(d):
    return float(PMAT[d-1, T-1]) if d > 0 else midnight_price_center(0)


def scenarios4_145(d, K=30):
    """Q4阶段0联合145段场景，最晚使用d-2完整历史轮廓。"""
    Lh, Gh, Ph = load_hat(d, 0), pv_hat(d, 0), price_hat(d, 0)
    ml, mg = midnight_center4(d)
    centers = (np.r_[ml, Lh], np.r_[mg, Gh], np.r_[midnight_price_center(d), Ph])
    sl, sg, spz = [], [], []
    for j in range(2, K+2):
        dd = d-j
        if dd < 10:
            continue
        aml, amg = midnight_actual4(dd); hml, hmg = midnight_center4(dd)
        hist_l, hist_g = np.r_[aml, L[dd]], np.r_[amg, G[dd]]
        hat_l, hat_g = np.r_[hml, load_hat(dd, 0)], np.r_[hmg, pv_hat(dd, 0)]
        hist_p = np.r_[midnight_price_actual(dd), PMAT[dd]]
        hat_p = np.r_[midnight_price_center(dd), price_hat(dd, 0)]
        sl.append(np.maximum(centers[0] + hist_l-hat_l, 0.0))
        sg.append(np.maximum(centers[1] + hist_g-hat_g, 0.0))
        ratio = hist_p / np.maximum(hat_p, 1e-6)
        spz.append(np.maximum(centers[2] * np.clip(ratio, 0.3, 3.0), 1e-3))
    return (sl, sg, spz) if sl else ([centers[0]], [centers[1]], [centers[2]])

# ----------------------------------------------------------------------
# 3. 阶段 LP（价格随机版）
# ----------------------------------------------------------------------
def stage_lp4(m, x_ref, S_cur, scenL, scenG, scenP, p_center,
              adjust=True, alpha=ALPHA, lam=0.478, commit_end=None):
    """第 m 阶段两阶段随机 LP。

    共享合同项使用中心预测价 p_center；未来合同、超计划和紧急购电等
    价格相关场景追索项使用 scenP[k]。二者不要求具有相同的样本均值。

    adjust=False 时（问题 4-2）不设 recourse 购电变量，即 q≡x，
    费用退化为 p·x + 5p·z，与问题 2 完全一致。

    变量布局
        [0,nD)                本阶段锁定的合约量 D_t
        [nD,nD+nU0)           m>=1 时 committed 段的 U_t=(a_t-x_ref_t)^+
        每场景 k：Q(nR) U(nR) c(nT) g(nT) z(nT) w(nT) S(nT)
    """
    tm = TSTAGE[m]
    tn = (TSTAGE[m+1] if commit_end is None else int(commit_end)) if adjust else T
    if not (tm < tn <= T):
        raise ValueError(f"无效承诺区间: stage={m}, [{tm},{tn})")
    K = len(scenL)
    nT = T - tm
    stage0 = (m == 0)
    dec = list(range(0, T)) if stage0 else list(range(tm, tn))
    R = list(range(tn, T))
    nD, nR = len(dec), len(R)
    dpos = {t: i for i, t in enumerate(dec)}
    nU0 = 0 if stage0 else nD
    base = nD + nU0
    per = 2*nR + 5*nT
    nv = base + K*per
    o = lambda k: base + k*per

    c = np.zeros(nv)
    # 方案A：共享合同决策固定使用中心预测价，不以场景均价替代。
    if stage0:
        # t<tn 的计划即最终合约(不可再调) -> 系数 p；t>=tn 的计划只承担一半 -> 0.5p
        for t in dec:
            c[dpos[t]] = p_center[t] if t < tn else 0.5*p_center[t]
    else:
        for i, t in enumerate(dec):
            c[i] = 0.5*p_center[t]
            c[nD+i] = p_center[t]              # committed 段的超计划罚项

    rows, cols, vals, beq = [], [], [], []
    iru, icu, ivu, bub = [], [], [], []
    nr = nq = 0
    for k in range(K):
        pk = scenP[k]
        ok = o(k)
        oQ, oU = ok, ok+nR
        oc = ok+2*nR; og = oc+nT; oz = og+nT; ow = oz+nT; oS = ow+nT
        for j, t in enumerate(R):
            c[oQ+j] += 0.5*pk[t]/K
            c[oU+j] += pk[t]/K
        for j in range(nT):
            c[oz+j] += alpha*pk[tm+j]/K
        c[oS+nT-1] -= lam/K
        for j in range(nT):                    # 功率平衡（四项均交流母线侧）
            t = tm + j
            col = dpos[t] if t < tn else oQ + (t-tn)
            rows.append(nr); cols.append(col); vals.append(1.0)
            for cc, vv in ((oz+j, 1.0), (og+j, 1.0), (oc+j, -1.0), (ow+j, -1.0)):
                rows.append(nr); cols.append(cc); vals.append(vv)
            beq.append(scenL[k][t] - scenG[k][t]); nr += 1
        for j in range(nT):                    # SOC 转移 S_j = S_{j-1} + eta*c_j - g_j/eta
            for cc, vv in ((oS+j, 1.0), (oc+j, -ETA_CHARGE), (og+j, 1.0/ETA_DISCHARGE)):
                rows.append(nr); cols.append(cc); vals.append(vv)
            if j > 0:
                rows.append(nr); cols.append(oS+j-1); vals.append(-1.0); beq.append(0.0)
            else:
                beq.append(S_cur)
            nr += 1
        for j, t in enumerate(R):              # U^k >= Q^k - x_t
            iru.append(nq); icu.append(oQ+j); ivu.append(1.0)
            iru.append(nq); icu.append(oU+j); ivu.append(-1.0)
            if stage0:
                iru.append(nq); icu.append(dpos[t]); ivu.append(-1.0); bub.append(0.0)
            else:
                bub.append(float(x_ref[t]))
            nq += 1
    if not stage0:
        for i, t in enumerate(dec):
            iru.append(nq); icu.append(i); ivu.append(1.0)
            iru.append(nq); icu.append(nD+i); ivu.append(-1.0)
            bub.append(float(x_ref[t])); nq += 1

    Aeq = sp.csr_matrix((vals, (rows, cols)), shape=(nr, nv))
    Aub = sp.csr_matrix((ivu, (iru, icu)), shape=(nq, nv)) if nq else None
    lb = np.zeros(nv); ub = np.full(nv, np.inf)
    for k in range(K):
        ok = o(k); oc = ok+2*nR; og = oc+nT; oS = og+3*nT
        ub[oc:oc+nT] = CMAX; ub[og:og+nT] = CMAX
        lb[oS:oS+nT] = SMIN; ub[oS:oS+nT] = SMAX
    r = linprog(c, A_ub=Aub, b_ub=np.array(bub) if nq else None,
                A_eq=Aeq, b_eq=np.array(beq),
                bounds=np.column_stack([lb, ub]), method='highs')
    if r.x is None:
        raise RuntimeError('LP infeasible: stage %d' % m)
    return r.x[:nD]


def stage0_lp4_145(committed_q, S_cur, scenL, scenG, scenP, p_center, lam,
                   commit_end=TSTAGE[1]):
    """波动电价下0:00的145段LP；共享项使用 p_center，h=0为前日已锁定午夜段。"""
    K, H, tn = len(scenL), T+1, int(commit_end)
    if not (0 < tn <= T):
        raise ValueError(f"无效0:00承诺区间终点: {tn}")
    R, nD = list(range(tn, T)), T
    nR = len(R); per = 2*nR + 5*H; nv = nD + K*per
    def o(k): return nD + k*per
    obj = np.zeros(nv); obj[:tn] = p_center[:tn]; obj[tn:nD] = 0.5*p_center[tn:]
    rows=[]; cols=[]; vals=[]; beq=[]; iru=[]; icu=[]; ivu=[]; bub=[]; nr=nq=0
    for k in range(K):
        ok=o(k); oQ=ok; oU=ok+nR; oc=ok+2*nR; og=oc+H; oz=og+H; ow=oz+H; oS=ow+H
        for j,t in enumerate(R):
            obj[oQ+j] += 0.5*scenP[k][t+1]/K
            obj[oU+j] += scenP[k][t+1]/K
        obj[oz:oz+H] += ALPHA*scenP[k]/K; obj[oS+H-1] -= lam/K
        for h in range(H):
            if h == 0:
                rhs = scenL[k][h]-scenG[k][h]-committed_q
            else:
                t=h-1
                cc = t if t < tn else oQ+t-tn
                rows.append(nr); cols.append(cc); vals.append(1.0)
                rhs = scenL[k][h]-scenG[k][h]
            for cc,vv in ((oz+h,1.0),(og+h,1.0),(oc+h,-1.0),(ow+h,-1.0)):
                rows.append(nr); cols.append(cc); vals.append(vv)
            beq.append(float(rhs)); nr+=1
        for h in range(H):
            for cc,vv in ((oS+h,1.0),(oc+h,-ETA_CHARGE),(og+h,1.0/ETA_DISCHARGE)):
                rows.append(nr); cols.append(cc); vals.append(vv)
            if h:
                rows.append(nr); cols.append(oS+h-1); vals.append(-1.0); beq.append(0.0)
            else: beq.append(float(S_cur))
            nr+=1
        for j,t in enumerate(R):
            iru += [nq,nq,nq]; icu += [oQ+j,oU+j,t]; ivu += [1.0,-1.0,-1.0]
            bub.append(0.0); nq+=1
    Aeq=sp.csr_matrix((vals,(rows,cols)),shape=(nr,nv)); Aub=sp.csr_matrix((ivu,(iru,icu)),shape=(nq,nv))
    lb=np.zeros(nv); ub=np.full(nv,np.inf)
    for k in range(K):
        ok=o(k); oc=ok+2*nR; og=oc+H; oS=ok+2*nR+4*H
        ub[oc:oc+H]=CMAX; ub[og:og+H]=CMAX; lb[oS:oS+H]=SMIN; ub[oS:oS+H]=SMAX
    result=linprog(obj,A_ub=Aub,b_ub=np.asarray(bub),A_eq=Aeq,b_eq=np.asarray(beq),
                   bounds=np.column_stack([lb,ub]),method='highs')
    if result.x is None: raise RuntimeError('Q4 LP infeasible at stage 0 (145 intervals)')
    return result.x[:T]

# ----------------------------------------------------------------------
# 4. 单日主流程与结算
# ----------------------------------------------------------------------
def solve_day4(d, S0, committed_q, K=30, stages=(0, 1, 2, 3)):
    enabled = tuple(sorted(set(int(m) for m in stages)))
    if not enabled or enabled[0] != 0 or any(m not in (0, 1, 2, 3) for m in enabled):
        raise ValueError("stages必须包含0，且只能取0/1/2/3")
    first_adjust = TSTAGE[enabled[1]] if len(enabled) > 1 else T
    out = {k: np.zeros(T) for k in ('z', 'c', 'g', 'w', 'S')}
    sl, sg, spz = scenarios4_145(d, K)
    p_center = price_hat(d, 0); lam = LAM  # 与Q3统一，避免横向比较混入终端价值变化
    x = stage0_lp4_145(committed_q, S0, sl, sg, spz, p_center, lam,
                       commit_end=first_adjust)
    q = x.copy(); S = S0
    ml, mg = midnight_actual4(d)
    S, midnight = dispatch_locked_interval(committed_q, S, ml, mg)
    S_after_midnight = S
    for m in range(4):
        tm, tn = TSTAGE[m], TSTAGE[m+1]
        if m >= 1 and m in enabled:
            later = [j for j in enabled if j > m]
            commit_end = TSTAGE[later[0]] if later else T
            sl, sg, spz = scenarios4(d, m, K)
            p_center = price_hat(d, m); lam = LAM
            q[tm:commit_end] = stage_lp4(m, x, S, sl, sg, spz, p_center, adjust=True,
                                         lam=lam, commit_end=commit_end)
        S = dispatch(tm, min(tn, T-1), q, S, L[d], G[d], out)
    natural = {k: np.r_[midnight[k], out[k][:T-1]] for k in ('c','g','z','w')}
    natural['S'] = np.r_[S0, S_after_midnight, out['S'][:T-1]]
    return x, q, out, natural, S

def day_cost4(d, x, q, out):
    """按【实际】电价结算： 0.5p x + 0.5p q + p(q-x)^+ + 5p z"""
    p = PMAT[d]
    plan = (0.5*p*x).sum()
    adj = (0.5*p*q).sum() + (p*np.maximum(q-x, 0)).sum()
    emg = (5*p*out['z']).sum()
    return plan+adj+emg, plan+adj, emg

def settle_parts4(d, x, q, z):
    """按题面口径把单日费用拆成填写工作簿用的三项（三者之和恒等于 day_cost4 的总费用）：
        计划购电费用   = Σ p*min(x,q)
        调整相关费用   = Σ [1.5p*(q-x)^+ + 0.5p*(x-q)^+]
        紧急购电费用   = Σ 5p*z
    其中 p 为当日【实际】电价 PMAT[d]。"""
    p = PMAT[d]
    up, dn = np.maximum(q-x, 0.0), np.maximum(x-q, 0.0)
    return (p*np.minimum(x, q), 1.5*p*up + 0.5*p*dn, 5.0*p*z)


def natural_price4(d):
    return np.r_[midnight_price_actual(d), PMAT[d, :T-1]]


def natural_settle_parts4(d, x, q, z):
    p=natural_price4(d); up=np.maximum(q-x,0.0); dn=np.maximum(x-q,0.0)
    return p*np.minimum(x,q), 1.5*p*up+0.5*p*dn, 5.0*p*z

def storage_blocks(charge, discharge):
    """自然日六个4小时段。"""
    return [{"time_range": "%s-%s" % (clock_min(24*k*10), clock_min((24*k+24)*10)),
             "charge_kwh": float(charge[24*k:24*k+24].sum()),
             "discharge_kwh": float(discharge[24*k:24*k+24].sum())} for k in range(6)]

def backtest4(d0=0, d1=ND, K=30, stages=(0, 1, 2, 3), S0=6000.0, verbose=False):
    S, rec = S0, {}
    # 合同量独立冷启动为0；缺失的首段负荷/光伏及价格先验显式来自附件1。
    committed_x = committed_q = 0.0
    for d in range(d0, d1):
        Ss = S
        x, q, out, natural, S = solve_day4(d, S, committed_q, K, stages)
        rec[d] = dict(x=x,q=q,S0=Ss,S24=S,natural_x=np.r_[committed_x,x[:T-1]],
                      natural_q=np.r_[committed_q,q[:T-1]],natural=natural,**out)
        committed_x, committed_q = float(x[-1]), float(q[-1])
        if verbose and d % 20 == 0:
            print(d, DSTR[d], 'S=%.0f' % S, flush=True)
    return rec, S
