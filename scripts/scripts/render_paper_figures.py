"""渲染数学建模竞赛论文图件（图5-1、图6-1 ~ 图6-9）。

统一视觉规范
------------
* 白底；科研蓝为主色，橙色为唯一重点强调色，对照与次要结果用灰 / 灰蓝；
* 低饱和度配色；去掉顶部与右侧边框；仅保留浅灰弱网格；
* 同族系列用「颜色 + 线型 + marker」多重区分，保证黑白打印可辨识；
* 不使用双纵轴；柱状图纵轴自 0 起；输出 300—600 dpi（本脚本取 500 dpi）。

设计约束
--------
1. 每张图回答论文中的一个明确问题，不作装饰。
2. 全部数值来自当前仓库的真实附件、审计产物、summary/payload，或由当前模型代码重算；
   本脚本不手工填写任何数据点，核对失败即中止。
3. 严格区分：自然日 144 时段、模板行 144 项、阶段 0 的 145 段规划窗口。
4. 时间标签一律为区间起点；自然日 00:00 首段取自前一原始行末项，禁止同行循环右移。
5. 问题二至四的费用比较期为 2025-02-01 至 2025-12-31，共 334 天 / 48096 时段。
6. 问题三、四为滚动两阶段 / SAA 近似，图题图注不称其为严格多阶段随机最优模型。
7. 审计结论只支持物理可行性、费用算术与数据一致性，不构成全局最优性证明。
8. 未使用 论文修订/media 下的旧 PNG，未引用旧电脑路径。

输出：论文修订/论文修订/figures 下的矢量 PDF 与 500 dpi PNG，
      同目录 figure_captions.md 与 figure_manifest.json。
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from scipy.stats import pearsonr, spearmanr

SCRIPT_VERSION = "3.1.0"

ROOT = Path(r"C:\Users\admin\Desktop\update")
SRC = ROOT / "src" / "src"
DATA = ROOT / "data" / "data"
ANNEX5 = DATA / "附件5"
Q2_AUDIT = ROOT / "q2_paper_audit" / "q2-paper-audit"
Q3_AUDIT = ROOT / "q3_paper_audit" / "q3-paper-audit"
Q4 = ROOT / "q4" / "q4"
PAPER = ROOT / "论文修订" / "论文修订" / "简洁_问题一二三四修订.md"
FIGDIR = ROOT / "论文修订" / "论文修订" / "figures"

sys.path.insert(0, str(SRC))

MM = 1.0 / 25.4
FULL_WIDTH_MM = 160.0
PNG_DPI = 500

# ------------------------------------------------------------------ 配色系统
INK = "#1A1A1A"          # 文字与坐标轴
BLUE = "#2F5C8A"         # 科研蓝：主色（主方案 / 计划量 / 关键状态 / 模型输出）
BLUE_DK = "#1E3E60"      # 深蓝：蓝色系第二层次
GRAYBLUE = "#A9BCCF"     # 灰蓝：次要结果 / 背景区间 / 反向动作
GRAYBLUE_LT = "#DFE7EF"  # 浅灰蓝：浅填充
GRAY = "#8C8C8C"         # 灰：对照基准
GRAY_LT = "#DCDCDC"      # 浅灰：网格 / 弱包络
ORANGE = "#D9822B"       # 橙：唯一重点强调色
ORANGE_LT = "#F2DCC2"    # 浅橙：强调区间底纹

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans SC", "SimHei", "Source Han Sans CN",
                        "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 8.5,
    "axes.titlesize": 10.0,
    "axes.labelsize": 8.8,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.8,
    "axes.linewidth": 0.8,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "hatch.linewidth": 0.5,
})

CHECKS: list[tuple[str, str, float, float, str]] = []


def check(label: str, got: float, want: float, tol: float, unit: str = "") -> None:
    ok = abs(got - want) <= tol
    CHECKS.append((label, f"{got:,.6f}{unit}", f"{want:,.6f}{unit}",
                   abs(got - want), "OK" if ok else "FAIL"))
    if not ok:
        raise AssertionError(f"核对失败：{label} 实得 {got!r}，期望 {want!r}（容差 {tol}）")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def style(ax, grid_axis: str | None = "y", grid: bool = True) -> None:
    """统一轴样式：去顶右边框、浅灰弱网格、外向刻度。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.8)
    if grid and grid_axis:
        ax.grid(axis=grid_axis, color=GRAY_LT, linewidth=0.5, alpha=0.85)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def time_ticks(ax, step: int = 24) -> None:
    """横轴为区间起点口径的 00:00—24:00。"""
    ticks = list(range(0, 145, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 6:02d}:{(t % 6) * 10:02d}" for t in ticks])
    ax.set_xlim(0, 144)


def note(ax, x, y, text, ha="left", va="top", size=7.4, color=GRAY, box=False):
    """图内说明文字：置于轴内、不参与布局计算、可读性由白底保证。

    box=True 时给文字加一层不透明白底，用于说明文字与网格线、参考线交叉的场合。
    """
    bbox = (dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none",
                 alpha=0.92) if box else None)
    t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                fontsize=size, color=color, linespacing=1.5, bbox=bbox)
    t.set_in_layout(False)
    return t


def save(fig, stem: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{stem}.pdf")
    fig.savefig(FIGDIR / f"{stem}.png", dpi=PNG_DPI)
    plt.close(fig)
    print(f"  已输出 {stem}.png / {stem}.pdf")


# ==================================================================== 数据装载
def load_q1() -> dict:
    """用当前 q1_solver.py 读附件1重算完整 144 段调度轨迹。"""
    import q1_solver as q1

    cfg = q1.Config()
    inputs = q1.load_inputs(DATA, cfg)
    main = q1.solve_day(inputs, cfg, cfg.initial_soc_kwh, cyclic=True)
    baseline_cost = float(inputs["price"] @ np.maximum(inputs["net"], 0.0))
    return {
        "price": inputs["price"], "load": inputs["load"], "pv": inputs["pv"],
        "net": inputs["net"], "grid": main["grid"], "charge": main["charge"],
        "discharge": main["discharge"], "soc": main["soc"],
        "baseline_grid": np.maximum(inputs["net"], 0.0),
        "baseline_cost": baseline_cost,
        "main_cost": float(main["purchase_cost_yuan"]),
    }


def load_q2() -> dict:
    """审计 CSV 提供实际/预测/紧急购电；情景带由当前 q2_solver.py 重建。"""
    import q2_solver as q2

    with (Q2_AUDIT / "interval_detail.csv").open(encoding="utf-8-sig") as fh:
        interval = list(csv.DictReader(fh))
    with (Q2_AUDIT / "daily_metrics.csv").open(encoding="utf-8-sig") as fh:
        daily = list(csv.DictReader(fh))

    cfg = q2.Config()
    inputs = q2.load_inputs(DATA, cfg)
    net, cold, dates = inputs["net_actual"], inputs["cold_start_net"], inputs["dates"]

    days, env_hits, band_hits = [], 0, 0
    for target in (date(2025, 3, 20), date(2025, 6, 21),
                   date(2025, 9, 23), date(2025, 12, 21)):
        day = dates.index(target)
        weight = q2.choose_weight(net, cold, day, cfg)
        scenarios, center, window = q2.scenario_set(net, cold, day, weight, cfg)
        band, actual = scenarios[:, :144], net[day]
        emin, emax = band.min(axis=0), band.max(axis=0)
        lo, hi = np.quantile(band, 0.10, axis=0), np.quantile(band, 0.90, axis=0)
        env, bnd = (actual >= emin) & (actual <= emax), (actual >= lo) & (actual <= hi)
        env_hits += int(env.sum())
        band_hits += int(bnd.sum())
        rows = [r for r in interval if r["date"] == str(target)]
        days.append({
            "date": target, "center": center[:144], "actual": actual,
            "env_min": emin, "env_max": emax, "band_lo": lo, "band_hi": hi,
            "emergency": np.array([float(r["emergency_kwh"]) for r in rows]),
            "audit_forecast": np.array([float(r["forecast_net_kwh"]) for r in rows]),
            "audit_actual": np.array([float(r["actual_net_kwh"]) for r in rows]),
            "env_hits": int(env.sum()), "band_hits": int(bnd.sum()),
            "weight": weight, "window": window,
            "scenario_count": int(scenarios.shape[0]),
        })

    return {
        "days": days, "env_hits": env_hits, "band_hits": band_hits,
        "total": 144 * 4, "daily": daily,
        "emergency": np.array([float(r["emergency_kwh"]) for r in daily]),
        "mae": np.array([float(r["forecast_mae_kwh"]) for r in daily]),
        "daily_dates": [r["date"] for r in daily],
        # 图6-9 需要实际净负荷本身；与 q4 backtest 取的是同一份 net_actual
        # （backtest 只在第 0 天改写首段为冷启动值，图6-9 选的日期不是第 0 天）。
        "net_actual": net, "dates": dates,
    }


def load_q3() -> dict:
    with (Q3_AUDIT / "q3_stage_comparison.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def load_q4() -> dict:
    def rd(name):
        with (Q4 / name).open(encoding="utf-8") as fh:
            return json.load(fh)

    return {"s2": rd("summary_q4-2_K30.json"), "s3": rd("summary_q4-3_K30.json"),
            "p2": rd("payload_q4-2_K30.json"), "p3": rd("payload_q4-3_K30.json")}


# ============================================================== 图5-1 方法示意
def fig_5_1() -> None:
    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 140 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.10])

    # ---------------------------------------------- a 原始模板行 → 自然日 144 段
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1); ax.set_ylim(-0.24, 1.0); ax.axis("off")
    ax.set_title("a　原始模板行的 144 项与自然日 144 段的跨行映射", loc="left", pad=6)

    x0, x1, n_show, gap_cells = 0.180, 0.968, 3, 1.9
    w = (x1 - x0) / (2 * n_show + gap_cells)
    h = 0.115
    y_prev, y_nat, y_cur = 0.855, 0.560, 0.265

    def draw_row(y, left, right, prefix, suffix):
        for side, group in enumerate((left, right)):
            for i, (lab, fc, ec, lw) in enumerate(group):
                cx = x0 + i * w if side == 0 else x1 - (n_show - i) * w
                ax.add_patch(Rectangle((cx, y), w, h, facecolor=fc, edgecolor=ec,
                                       linewidth=lw, zorder=3))
                ax.text(cx + w / 2, y + h / 2, lab, ha="center", va="center",
                        fontsize=7.0, color=INK, zorder=4)
        ax.text((x0 + x1) / 2 + w * 0.15, y + h / 2, "⋯", ha="center", va="center",
                fontsize=10, color=GRAY)
        ax.text(x0 - 0.014, y + h / 2, prefix, ha="right", va="center",
                fontsize=8.2, color=INK)
        if suffix:
            ax.text(x1 + 0.010, y + h / 2, suffix, ha="left", va="center",
                    fontsize=7.0, color=GRAY)

    W = ("#FFFFFF", GRAY_LT, 0.7)                       # 普通单元
    H = ("#E8EFF6", BLUE, 1.2)                          # 高亮单元（跨行首末项）

    draw_row(y_prev, [("0:10", *W), ("0:20", *W), ("0:30", *W)],
             [("23:40", *W), ("23:50", *W), ("0:00+1", *H)],
             "日期 ν−1 原始行", "末项属次日")
    draw_row(y_nat, [("00:00", *H), ("00:10", *W), ("00:20", *W)],
             [("23:30", *W), ("23:40", *W), ("23:50", *W)],
             "自然日 ν", "144 段")
    draw_row(y_cur, [("0:10", *W), ("0:20", *W), ("0:30", *W)],
             [("23:40", *W), ("23:50", *W), ("0:00+1", *W)],
             "日期 ν 原始行", "首项属当日")

    # 跨行映射：前一行末项 → 自然日首段
    ax.add_patch(FancyArrowPatch(
        (x1 - w / 2, y_prev), (x0 + w / 2, y_nat + h),
        connectionstyle="arc3,rad=0.10", arrowstyle="-|>", mutation_scale=11,
        linewidth=1.6, color=ORANGE, zorder=6))
    ax.text((x0 + x1) / 2, (y_prev + y_nat + h) / 2 - 0.014,
            "自然日首段取自前一行末项：$X^{nat}_{ν,0}=X^{raw}_{ν-1,143}$",
            ha="center", va="center", fontsize=8.0, color=ORANGE)

    # 本行前 143 项 → 自然日第 2—144 段
    ax.add_patch(FancyArrowPatch(
        (x0 + w / 2, y_cur + h), (x0 + 1.5 * w, y_nat),
        connectionstyle="arc3,rad=-0.25", arrowstyle="-|>", mutation_scale=11,
        linewidth=1.6, color=BLUE, zorder=6))
    ax.text(x0 + 2.05 * w, (y_cur + h + y_nat) / 2 - 0.014,
            "本行第 1—143 项按区间起点对位 → 自然日第 2—144 段",
            ha="left", va="center", fontsize=8.0, color=BLUE)
    ax.add_patch(FancyArrowPatch(
        (x1 - w / 2, y_cur), (x1 + 0.022, y_cur - 0.075),
        connectionstyle="arc3,rad=0.35", arrowstyle="-|>", mutation_scale=10,
        linewidth=1.2, color=GRAY, zorder=6))
    ax.text(x1 + 0.022, y_cur - 0.140, "本行末项归入次一自然日", ha="right",
            va="center", fontsize=7.6, color=GRAY)
    ax.text(0.0, -0.115,
            "时间标签一律为区间起点；同一行内禁止循环右移。"
            "自然日 144 段 = 前一行末项 1 段 + 本行前 143 项。",
            ha="left", va="center", fontsize=7.8, color=GRAY)

    # ------------------------------------------------ b 四阶段决策与信息截止
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 30.5); ax.set_ylim(-2.05, 5.30); ax.axis("off")
    ax.set_title("b　0:00 / 6:00 / 12:00 / 18:00 决策节点、已实现信息与合同锁定范围",
                 loc="left", pad=6)

    nodes, names = [0.0, 6.0, 12.0, 18.0], ["0:00", "6:00", "12:00", "18:00"]
    nexts, next_labels = [6.0, 12.0, 18.0, 24.0], ["6:00", "12:00", "18:00", "24:00"]
    row_h, gap = 0.62, 0.30

    for i, (node, nm, nxt, nlab) in enumerate(zip(nodes, names, nexts, next_labels)):
        y = (3 - i) * (row_h + gap)
        ax.add_patch(Rectangle((0, y), node, row_h, facecolor=GRAYBLUE_LT,
                               edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((node, y), 24 - node, row_h, facecolor="#FFFFFF",
                               edgecolor=GRAY_LT, linewidth=0.6, hatch="///",
                               zorder=2))
        ax.add_patch(Rectangle((node, y + 0.11), nxt - node, row_h - 0.22,
                               facecolor=BLUE, edgecolor="none", zorder=4))
        ax.text((node + nxt) / 2, y + row_h / 2, f"合同锁定 {nm}—{nlab}",
                ha="center", va="center", color="white", fontsize=7.4, zorder=5)
        ax.plot([node, node], [y - 0.07, y + row_h + 0.07], color=ORANGE,
                linewidth=1.6, zorder=6)
        ax.text(node, y + row_h + 0.15, nm, ha="center", va="bottom",
                fontsize=8.2, color=ORANGE)

    # 18:00 另锁定次日 00:00—00:10
    ax.add_patch(Rectangle((24.0, 0.11), 1.05, row_h - 0.22, facecolor=ORANGE,
                           edgecolor="none", zorder=4))
    ax.text(25.15, row_h / 2, "次日\n00:00—00:10", ha="left", va="center",
            fontsize=7.4, color=ORANGE)

    # 信息类型图例
    key = [(GRAYBLUE_LT, "none", ""), ("#FFFFFF", GRAY_LT, "///"), (BLUE, "none", "")]
    labels = ["已实现信息", "未来未知区间", "合同锁定范围"]
    kx = 0.0
    for (fc, ec, ht), lab in zip(key, labels):
        ax.add_patch(Rectangle((kx, 4.60), 0.75, 0.36, facecolor=fc, edgecolor=ec,
                               linewidth=0.7, hatch=ht, zorder=4))
        ax.text(kx + 0.95, 4.78, lab, ha="left", va="center", fontsize=7.6,
                color=GRAY)
        kx += 4.1

    ax.annotate("", xy=(24, -0.34), xytext=(0, -0.34),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=0.9))
    for t in [0, 6, 12, 18, 24]:
        ax.plot([t, t], [-0.42, -0.26], color=INK, linewidth=0.9)
        ax.text(t, -0.55, f"{t:02d}:00" if t < 24 else "24:00", ha="center",
                va="top", fontsize=7.8, color=INK)
    ax.text(0, -1.16, "自然日 144 段（00:00—24:00，时间标签为区间起点）",
            ha="left", va="top", fontsize=8.0, color=GRAY)
    ax.text(0, 4.12,
            "0:00 的规划窗口为 145 段 = 已承诺的午夜首段 1 段 + 当日 144 段；"
            "6:00 / 12:00 / 18:00 仅实施至下一启用阶段前的合同。",
            ha="left", va="center", fontsize=7.8, color=GRAY)
    ax.text(0, -1.74,
            "历史样本截止位置随节点后移：0:00 时最新完整模板行为 ν−2，"
            "6:00 及以后可用至 ν−1，并利用当日已完成时段。",
            ha="left", va="top", fontsize=7.8, color=GRAY)

    save(fig, "fig_5_1_time_mapping_and_information_boundary")


_Q2_STATS: dict | None = None


def _q2_stats(q2: dict) -> dict:
    """两张问题二图件共用的取值与核对。缓存一次：两张图各调一次，
    不缓存会把同一组 check 记两遍，核对汇总里出现重复行。"""
    global _Q2_STATS
    if _Q2_STATS is None:
        _Q2_STATS = _build_q2_stats(q2)
    return _Q2_STATS


def _build_q2_stats(q2: dict) -> dict:
    check("问题二情景包络覆盖段数", q2["env_hits"], 394.0, 1e-9)
    check("问题二分位带覆盖段数", q2["band_hits"], 314.0, 1e-9)
    check("问题二指定日期时段总数", q2["total"], 576.0, 1e-9)

    em, daily_dates = q2["emergency"], q2["daily_dates"]
    check("问题二评价期天数", len(daily_dates), 334.0, 1e-9)
    check("问题二有紧急购电的天数", float(np.count_nonzero(em > 1e-9)), 170.0, 1e-9)
    imax = int(np.argmax(em))
    check("问题二最大日紧急购电量", float(em[imax]), 15465.516113, 0.01, " kWh")
    check("问题二最大日紧急购电日期",
          0.0 if daily_dates[imax] == "2025-06-01" else 1.0, 0.0, 1e-9)
    check("问题二全年紧急购电量", float(em.sum()), 267693.199361, 0.01, " kWh")
    idx0923 = daily_dates.index("2025-09-23")
    check("问题二09-23紧急购电量", float(em[idx0923]), 4471.624170, 0.01, " kWh")
    r_p = float(pearsonr(q2["mae"], em).statistic)
    r_s = float(spearmanr(q2["mae"], em).statistic)
    check("问题二MAE与紧急购电Pearson相关", r_p, 0.315540, 1e-4)
    check("问题二MAE与紧急购电Spearman相关", r_s, 0.074394, 1e-4)

    for day in q2["days"]:
        check(f"{day['date']}中心预测与审计一致",
              float(np.max(np.abs(day["center"] - day["audit_forecast"]))), 0.0, 1e-8, " kWh")
        check(f"{day['date']}实际净负荷与审计一致",
              float(np.max(np.abs(day["actual"] - day["audit_actual"]))), 0.0, 1e-8, " kWh")

    return {"days": q2["days"], "em": em, "dates": daily_dates, "imax": imax,
            "idx0923": idx0923, "r_p": r_p, "r_s": r_s,
            "n_emg": int(np.count_nonzero(em > 1e-9))}


# ======================================== 图6-3 评价期逐日紧急购电量（独立成图）
def fig_6_3_q2_daily_emergency(q2: dict) -> None:
    st = _q2_stats(q2)
    em, dates = st["em"], st["dates"]
    imax, idx0923 = st["imax"], st["idx0923"]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_MM * MM, 96 * MM),
                           layout="constrained")

    xd = np.arange(len(em))
    hi = (xd == imax) | (xd == idx0923)
    # 334 根柱里只有两日需要被看见：橙色只给这两天，其余一律灰蓝。
    # 单独成图后柱宽从 0.9 缩到 0.82，柱间留出缝隙，密集区不至于糊成一片。
    ax.bar(xd, np.where(hi, em, np.nan), width=0.82, color=ORANGE, linewidth=0)
    ax.bar(xd, np.where(hi, np.nan, em), width=0.82, color=GRAYBLUE, linewidth=0)

    # 全年日均线：给 334 根柱一个量级参照，否则读者只能比高低、读不出水平。
    mean_all = float(em.mean())
    ax.axhline(mean_all, color=GRAY, linestyle=(0, (5, 2)), linewidth=0.9,
               zorder=4)
    ax.text(len(em) - 1, mean_all + 420, f"全年日均 {mean_all:,.0f} kWh ",
            ha="right", va="bottom", fontsize=7.2, color=GRAY, zorder=6)

    # 两条标注分列最高柱两侧：最大值向右、09-23 向左，x 区间不重叠，
    # 即便两者高度接近也不会撞在一起。
    ax.annotate(f"最大 {dates[imax]}：{em[imax]:,.2f} kWh",
                xy=(imax, em[imax]), xytext=(imax + 22, em[imax] * 0.99),
                fontsize=7.8, color=ORANGE, va="top",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=0.9))
    ax.annotate(f"{dates[idx0923]}：{em[idx0923]:,.2f} kWh",
                xy=(idx0923, em[idx0923]), xytext=(idx0923 - 16, em[idx0923] + 5200),
                fontsize=7.8, color=ORANGE, ha="right",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=0.9))

    ax.set_ylabel("每日紧急购电量 / kWh")
    ax.set_xlabel("评价期日期（2025-02-01 至 2025-12-31，共 334 天）")
    ticks = list(range(0, 334, 30))
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[t][5:] for t in ticks])
    ax.set_xlim(-1, 334)
    ax.set_ylim(0, 20000)
    style(ax)

    note(ax, 0.005, 0.985,
         f"评价期 334 天中有 {st['n_emg']} 天发生紧急购电，"
         f"全年紧急购电 {em.sum() / 1e4:,.4f} 万kWh。\n"
         f"中心预测 MAE 与日紧急购电量：Pearson r = {st['r_p']:.3f}、"
         f"Spearman ρ = {st['r_s']:.3f}，仅为有限统计关联。",
         ha="left", va="top", size=7.8).set_bbox(
        dict(boxstyle="square,pad=0.28", facecolor="white", edgecolor="none",
             alpha=0.88))

    save(fig, "fig_6_3_q2_daily_emergency_purchase")


# ================================= 图6-4 指定日期净负荷预测与经验情景范围（独立成图）
def fig_6_4_q2_scenario_range(q2: dict) -> None:
    st = _q2_stats(q2)

    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 142 * MM),
                     layout="constrained")
    gs = fig.add_gridspec(2, 2)
    # 单独成图后不再与逐日柱状图争高度，四格放宽，行距与列距交给布局引擎，
    # 只留一点缝，让相邻两格的水平网格线不至于连成一条而误读成同一坐标系。
    fig.get_layout_engine().set(hspace=0.10, wspace=0.07)

    for k, day in enumerate(st["days"]):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        x = np.arange(144)
        ax.fill_between(x, day["env_min"], day["env_max"], color=GRAY_LT,
                        linewidth=0, zorder=1)
        ax.fill_between(x, day["band_lo"], day["band_hi"], color=GRAYBLUE,
                        alpha=0.75, linewidth=0, zorder=2)
        ax.plot(x, day["center"], color=BLUE, linewidth=1.2, linestyle="--",
                dashes=(4, 2), zorder=4)
        ax.plot(x, day["actual"], color="#4D4D4D", linewidth=1.1, zorder=5)
        seg = day["emergency"] > 1e-9
        if seg.any():
            start = None
            for i, flag in enumerate(np.r_[seg, False]):
                if flag and start is None:
                    start = i
                elif not flag and start is not None:
                    ax.axvspan(start, max(i, start + 1), color=ORANGE_LT,
                               alpha=0.85, linewidth=0, zorder=3)
                    start = None
            ax.plot(x[seg], day["actual"][seg], linestyle="none", marker="o",
                    markersize=2.8, color=ORANGE, zorder=6)
        ax.set_title(f"{day['date']}　情景数 {day['scenario_count']}",
                     loc="left", pad=4, fontsize=9.2)
        ax.set_ylim(0, 1600)
        time_ticks(ax, 36)
        style(ax)
        if k % 2 == 0:
            ax.set_ylabel("净负荷电量 / kWh\n（每 10 分钟）")
        if k // 2 == 1:
            ax.set_xlabel("时间（区间起点）")
        note(ax, 0.99, 0.96,
             f"包络 {day['env_hits']}/144\n分位带 {day['band_hits']}/144",
             ha="right", va="top", size=7.4)

    handles = [Patch(facecolor=GRAY_LT, label="最小—最大包络"),
               Patch(facecolor=GRAYBLUE, label="10%—90% 分位带"),
               Line2D([], [], color=BLUE, linewidth=1.2, linestyle="--",
                      dashes=(4, 2), label="中心预测"),
               Line2D([], [], color="#4D4D4D", linewidth=1.1, label="实际净负荷"),
               Line2D([], [], color=ORANGE, marker="o", markersize=3.2,
                      linestyle="none", label="紧急购电时段")]
    fig.legend(handles=handles, loc="outside upper center", ncol=5,
               frameon=False, handlelength=1.6, columnspacing=1.1, fontsize=7.4)

    save(fig, "fig_6_4_q2_forecast_scenario_range")


# ========================================================== 图6-5 问题三阶段价值
_Q3_STATS: dict | None = None


def _q3_stage_stats(q3: dict) -> dict:
    """两张阶段价值图共用的取值与核对。缓存一次：两张图各调一次，
    不缓存会把同一组 check 记两遍，核对汇总里出现重复行。"""
    global _Q3_STATS
    if _Q3_STATS is None:
        _Q3_STATS = _build_q3_stage_stats(q3)
    return _Q3_STATS


def _build_q3_stage_stats(q3: dict) -> dict:
    rows = q3["rows"]
    totals = {r["policy"]: r["total_cost_yuan"] for r in rows}
    check("问题三仅0:00总费用", totals["0-only"], 13986857.304819, 0.01, " 元")
    check("问题三全阶段总费用", totals["0+6+12+18"], 13193305.636976, 0.01, " 元")
    saving = totals["0-only"] - totals["0+6+12+18"]
    check("问题三全阶段节省", saving, 793551.667844, 0.01, " 元")
    check("问题三节省率", saving / totals["0-only"], 0.0567355, 1e-6)

    shap = q3["shapley_savings_vs_0_only_for_full_policy_yuan"]
    check("Shapley 6:00", shap["6:00"], 232769.778448, 0.01, " 元")
    check("Shapley 12:00", shap["12:00"], 219725.819189, 0.01, " 元")
    check("Shapley 18:00", shap["18:00"], 341056.070207, 0.01, " 元")
    check("Shapley加总等于全阶段节省", sum(shap.values()), saving, 0.01, " 元")
    marg = q3["conditional_marginal_savings"]
    allmarg = [m["saving_yuan"] for v in marg.values() for m in v]
    check("条件边际节省项数", float(len(allmarg)), 12.0, 1e-9)
    check("条件边际节省最小值", float(min(allmarg)), 104071.346872, 0.01, " 元")
    check("条件边际节省全部为正", 0.0 if min(allmarg) > 0 else 1.0, 0.0, 1e-9)
    return {"rows": rows, "totals": totals, "saving": saving,
            "shap": shap, "marg": marg}


# ============================ 图6-5 八种阶段组合的费用构成（独立成图）
def fig_6_5(q3: dict) -> None:
    st = _q3_stage_stats(q3)
    ordered = sorted(st["rows"], key=lambda r: r["total_cost_yuan"])
    labels = ["仅 0:00" if r["policy"] == "0-only" else r["policy"] for r in ordered]
    plan = np.array([r["plan_cost_yuan"] for r in ordered]) / 1e4
    adj = np.array([r["adjust_cost_yuan"] for r in ordered]) / 1e4
    emg = np.array([r["emergency_cost_yuan"] for r in ordered]) / 1e4
    x = np.arange(len(ordered))
    base = st["totals"]["0-only"] / 1e4
    saving = st["saving"]

    # 交给 constrained 布局排边距：两面板的纵轴标签都是两行竖排
    # （「总费用」/「万元」），手工给的 left 一旦小于它们加刻度标签的宽度，
    # 最左侧那行字就会被画到画布外裁掉。行距改用布局引擎的 hspace 控制。
    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 150 * MM),
                     layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.72])
    fig.get_layout_engine().set(hspace=0.09)
    # 紧急费用改用深蓝阶梯而非橙色：左上角的橙色文字已经在讲「节省」，
    # 同一坐标系里再放橙色柱，橙色会同时指向成本与收益两个相反含义。
    # 三段蓝的相邻边界灰阶为 84|185 与 185|56，黑白打印下逐界可分。

    # ---- a 总费用构成（自 0 起，保留量级）
    ax = fig.add_subplot(gs[0])
    ax.bar(x, plan, width=0.66, color=BLUE, linewidth=0, label="计划费用")
    ax.bar(x, adj, width=0.66, bottom=plan, color=GRAYBLUE, linewidth=0.6,
           edgecolor=GRAY, label="调整费用")
    ax.bar(x, emg, width=0.66, bottom=plan + adj, color=BLUE_DK, linewidth=0,
           label="紧急费用")
    ax.axhline(base, color=GRAY, linestyle="--", dashes=(5, 2), linewidth=1.0,
               zorder=6)
    for xi, r in zip(x, ordered):
        ax.text(xi, r["total_cost_yuan"] / 1e4 + 16,
                f"{r['total_cost_yuan'] / 1e4:.2f}",
                ha="center", va="bottom", fontsize=7.0, color=INK)
    ax.set_xticks(x)
    # 不用 sharex：共用格式化器时，下面板 set_xticklabels 会把标签一起送上上面板
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("总费用 / 万元")
    ax.set_xlim(-0.68, len(ordered) - 1 + 0.68)
    ax.set_ylim(0, 1660)
    ax.set_title("a　八种组合的总费用构成（自 0 起，量级可比）", loc="left", pad=4)
    style(ax)

    # 基线含义交给图例里的虚线样式，不再在轴内横排文字——
    # 基线恰与最右柱等高，文字放哪一端都会撞上那根柱的数值标签。
    handles, names = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=GRAY, linestyle="--", dashes=(5, 2),
                          linewidth=1.0))
    names.append("仅 0:00 基线 1398.69 万元")
    fig.legend(handles, names, loc="outside upper center", ncol=4, frameon=False,
               fontsize=7.6, handlelength=1.8, columnspacing=1.8)

    t = ax.text(0.008, 0.975,
                f"全阶段相对仅 0:00 节省 {saving / 1e4:.2f} 万元"
                f"（{100 * saving / st['totals']['0-only']:.2f}%）",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.4,
                color=ORANGE)
    t.set_in_layout(False)

    # ---- b 去掉计划费用后的放大：一切变化都在调整与紧急两段里
    # 计划费用近 1210—1290 万元且几乎不随组合变动，占满每根柱的九成，
    # 把总费用那 5.67% 的差别压成看不出的台阶；单独放大这两段才能读出
    # 「紧急费用塌缩、调整费用顶上来」这一此消彼长。
    ax = fig.add_subplot(gs[1])
    nonplan = adj + emg
    ax.bar(x, adj, width=0.66, color=GRAYBLUE, linewidth=0.6, edgecolor=GRAY,
           label="调整费用")
    ax.bar(x, emg, width=0.66, bottom=adj, color=BLUE_DK, linewidth=0,
           label="紧急费用")
    for xi, v in zip(x, nonplan):
        ax.text(xi, v + 0.035 * nonplan.max(), f"{v:.2f}", ha="center",
                va="bottom", fontsize=7.0, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("调整 + 紧急\n/ 万元")
    ax.set_xlabel("预报发布与调整时刻组合（按总费用升序）")
    ax.set_xlim(-0.68, len(ordered) - 1 + 0.68)
    ax.set_ylim(0, float(nonplan.max()) * 1.28)
    ax.set_title("b　非计划部分放大：调整费用与紧急费用的此消彼长", loc="left", pad=4)
    style(ax)
    note(ax, 0.994, 0.965,
         f"计划费用 {plan.min():.2f}—{plan.max():.2f} 万元已从本面板略去",
         ha="right", va="top", size=7.0)

    save(fig, "fig_6_5_q3_stage_cost_composition")


# ============================ 图6-6 三阶段 Shapley 收益分摊（独立成图）
def fig_6_6_shapley(q3: dict) -> None:
    st = _q3_stage_stats(q3)
    order = ["6:00", "12:00", "18:00"]
    vals = np.array([st["shap"][k] for k in order]) / 1e4
    share = vals / (st["saving"] / 1e4) * 100
    y = np.arange(len(order))[::-1]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_MM * MM, 88 * MM),
                           layout="constrained")
    # 横条用橙色：这三根柱本身就是「节省分摊」，与全篇「橙 = 收益」一致。
    ax.barh(y, vals, height=0.40, color=ORANGE, linewidth=0)
    for yi, v, s, k in zip(y, vals, share, order):
        # 两条标签一律落在柱外右侧：原图的「占 xx%」压在橙色柱身上，
        # 灰字配橙底几乎读不出来。
        ax.text(v + 0.9, yi + 0.115, f"{v:,.2f} 万元", va="center", ha="left",
                fontsize=8.4, color=INK)
        ax.text(v + 0.9, yi - 0.115, f"占 {s:.2f}%", va="center", ha="left",
                fontsize=7.4, color=GRAY)
        ctx = [m["saving_yuan"] / 1e4 for m in st["marg"][k]]
        ax.plot(ctx, [yi - 0.32] * len(ctx), linestyle="none", marker="|",
                markersize=7, markeredgewidth=1.0, color=GRAY, zorder=5)
        ax.plot([np.mean(ctx)], [yi - 0.32], linestyle="none", marker="o",
                markersize=3.2, color=GRAY, zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("相对仅 0:00 的节省分摊 / 万元")
    ax.set_ylabel("日内预报更新与调整时刻")
    ax.set_xlim(0, 45)
    ax.set_ylim(-1.02, 2.34)
    style(ax, "x")
    # 图例移出轴外单排：轴内右上角正是各阶段条件边际节省的落点（最右一项达
    # 37.8 万元），图例手柄与数据刻线同为灰色竖线，挤在同一角里无法分辨。
    fig.legend([Line2D([], [], color=GRAY, marker="|", linestyle="none",
                       markersize=7, markeredgewidth=1.0),
                Line2D([], [], color=GRAY, marker="o", linestyle="none",
                       markersize=3.2)],
               ["该时刻在 4 种上下文下的条件边际节省", "四者均值"],
               loc="outside lower center", ncol=2, frameon=False, fontsize=7.4,
               handlelength=1.0, columnspacing=2.4)
    note(ax, 0.0, 0.012,
         "分摊对象为预报更新、合同重优化与储能状态演化的综合节省，\n"
         "不是预报信息的纯因果价值；三阶段单独取值不可直接相加。",
         ha="left", va="bottom", size=7.4)

    save(fig, "fig_6_6_q3_shapley_allocation")


# ====================================================== 图6-7 问题四两策略比较
def fig_6_7(q4: dict) -> None:
    t2, t3 = q4["s2"]["totals"], q4["s3"]["totals"]
    check("4-2总费用", t2["total_cost_yuan"], 15201795.404210, 0.01, " 元")
    check("4-3总费用", t3["total_cost_yuan"], 13876490.551360, 0.01, " 元")
    check("4-2紧急购电量", t2["emergency_kwh"], 272134.502503, 0.01, " kWh")
    check("4-3紧急购电量", t3["emergency_kwh"], 41456.549387, 0.01, " kWh")
    diff = t2["total_cost_yuan"] - t3["total_cost_yuan"]
    check("两策略总费用差", diff, 1325304.852849, 0.01, " 元")
    check("两策略总费用降幅", diff / t2["total_cost_yuan"], 0.08718081, 1e-7)
    red = (t2["emergency_kwh"] - t3["emergency_kwh"]) / t2["emergency_kwh"]
    check("紧急购电量降幅", red, 0.84766155, 1e-7)
    check("4-2调整费用为零", t2["adjust_cost_yuan"], 0.0, 1e-9, " 元")

    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 116 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])

    names = ["4-2\n每日一次计划", "4-3\n四阶段滚动"]
    x = np.arange(2)
    plan = np.array([t2["plan_cost_yuan"], t3["plan_cost_yuan"]]) / 1e4
    adj = np.array([t2["adjust_cost_yuan"], t3["adjust_cost_yuan"]]) / 1e4
    emg = np.array([t2["emergency_cost_yuan"], t3["emergency_cost_yuan"]]) / 1e4
    total = plan + adj + emg

    # ---- a 费用构成
    ax = fig.add_subplot(gs[0])
    ax.bar(x, plan, width=0.46, color=BLUE, linewidth=0, label="计划费用")
    ax.bar(x, adj, width=0.46, bottom=plan, color=GRAYBLUE, linewidth=0.6,
           edgecolor=GRAY, label="调整费用")
    ax.bar(x, emg, width=0.46, bottom=plan + adj, color=BLUE_DK, linewidth=0,
           label="紧急费用")
    for xi, tot in zip(x, total):
        ax.text(xi, tot + 16, f"总费用 {tot:,.2f} 万元", ha="center", va="bottom",
                fontsize=8.2, color=INK)
    for xi, p, a, e in zip(x, plan, adj, emg):
        ax.text(xi, p / 2, f"{p:,.2f}", ha="center", va="center", color="white",
                fontsize=7.8)
        if a > 60:
            ax.text(xi, p + a / 2, f"{a:,.2f}", ha="center", va="center",
                    color=INK, fontsize=7.8)
        if e > 60:
            ax.text(xi, p + a + e / 2, f"{e:,.2f}", ha="center", va="center",
                    color="white", fontsize=7.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("费用 / 万元")
    ax.set_title("a　两种策略的费用构成（334 天评价期）", loc="left", pad=4)
    ax.set_xlim(-0.72, 1.72); ax.set_ylim(0, 1800)
    style(ax)
    ax.legend(loc="upper center", ncol=3, frameon=True, facecolor="white",
              edgecolor=GRAY_LT, framealpha=0.95, handlelength=1.4,
              columnspacing=1.3, fontsize=7.6)

    # ---- b 紧急购电量
    ax = fig.add_subplot(gs[1])
    ek = np.array([t2["emergency_kwh"], t3["emergency_kwh"]]) / 1e4
    # 4-2 是基准方案，按全篇约定用灰蓝；4-3 是改进结果用科研蓝。
    ax.bar(x, ek, width=0.46, color=[GRAYBLUE, BLUE], linewidth=0)
    for xi, v in zip(x, ek):
        ax.text(xi, v + 0.75, f"{v:,.4f} 万kWh", ha="center", va="bottom",
                fontsize=8.4, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("紧急购电量 / 万kWh")
    ax.set_title("b　两种策略的紧急购电量", loc="left", pad=4)
    ax.set_xlim(-0.72, 1.72); ax.set_ylim(0, 46)
    style(ax)
    ax.annotate("", xy=(1, ek[1]), xytext=(0, ek[0]),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, linewidth=0.9,
                                connectionstyle="arc3,rad=-0.25"))
    t = ax.text(0.62, 15.5, f"减少 {red * 100:.4f}%", ha="center", va="center",
                fontsize=9.0, color=ORANGE,
                bbox=dict(boxstyle="square,pad=0.25", facecolor="white",
                          edgecolor="none", alpha=0.9))
    t.set_in_layout(False)
    note(ax, 0.0, 0.985,
         "相对 4-2 的分项变化：\n"
         f"计划费用 −{abs((t2['plan_cost_yuan'] - t3['plan_cost_yuan']) / 1e4):.4f} 万元\n"
         f"紧急费用 −{abs((t2['emergency_cost_yuan'] - t3['emergency_cost_yuan']) / 1e4):.4f} 万元\n"
         f"调整费用 +{t3['adjust_cost_yuan'] / 1e4:.4f} 万元\n"
         f"总费用 −{diff / 1e4:.4f} 万元（{100 * diff / t2['total_cost_yuan']:.4f}%）",
         ha="left", va="top", size=7.6)

    save(fig, "fig_6_7_q4_aggregate_comparison")


# ================================================== 图6-8 逐日与累计节省
def fig_6_8(q4: dict) -> None:
    d2, d3 = q4["s2"]["daily"], q4["s3"]["daily"]
    dates = [r["date"] for r in d2]
    assert dates == [r["date"] for r in d3]
    c2 = np.array([r["total_cost_yuan"] for r in d2])
    c3 = np.array([r["total_cost_yuan"] for r in d3])

    p2 = [r["total_cost_yuan"] for r in q4["p2"]["days"]]
    p3 = [r["total_cost_yuan"] for r in q4["p3"]["days"]]
    check("payload 4-2 逐日费用与 summary 最大差",
          float(np.max(np.abs(np.asarray(p2) - c2))), 0.0, 1e-6, " 元")
    check("payload 4-3 逐日费用与 summary 最大差",
          float(np.max(np.abs(np.asarray(p3) - c3))), 0.0, 1e-6, " 元")

    delta, cum = c2 - c3, np.cumsum(c2 - c3)
    check("评价期天数", float(len(delta)), 334.0, 1e-9)
    check("逐日节省合计", float(delta.sum()), 1325304.852849, 0.01, " 元")
    check("累计节省终点", float(cum[-1]), 1325304.852849, 0.01, " 元")
    check("4-3费用较低天数", float(np.count_nonzero(delta > 1e-9)), 237.0, 1e-9)
    check("4-3费用较高天数", float(np.count_nonzero(delta < -1e-9)), 97.0, 1e-9)
    ip, ineg = int(np.argmax(delta)), int(np.argmin(delta))
    check("最大单日节省", float(delta[ip]), 68895.878600, 0.01, " 元")
    check("最大单日节省日期", 0.0 if dates[ip] == "2025-06-01" else 1.0, 0.0, 1e-9)
    check("最大单日负节省", float(delta[ineg]), -5212.581200, 0.01, " 元")
    check("最大单日负节省日期", 0.0 if dates[ineg] == "2025-05-04" else 1.0, 0.0, 1e-9)

    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 150 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.10)
    x = np.arange(len(dates))
    ticks = list(range(0, 334, 30))

    # ---- a 每日费用节省
    ax = fig.add_subplot(gs[0])
    # 橙 = 节省、灰蓝 = 增支：全篇橙色的含义是「收益」，
    # 原先橙 = 费用较高的那 97 天，与图6-2 的橙 = 节省正好相反。
    ax.bar(x, delta / 1e4, width=0.9,
           color=np.where(delta >= 0, ORANGE, GRAYBLUE), linewidth=0)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[t][5:] for t in ticks])
    ax.set_xlim(-1, 334); ax.set_ylim(-2.3, 9.0)
    ax.set_ylabel("每日费用节省 $\\Delta C_d$ / 万元")
    ax.set_title("a　逐日费用节省（$\\Delta C_d = C_{4\\text{-}2,d} - C_{4\\text{-}3,d}$）",
                 loc="left", pad=4)
    style(ax)
    fig.legend(handles=[Patch(facecolor=ORANGE,
                              label=f"4-3 费用较低（{np.count_nonzero(delta >= 0)} 天）"),
                        Patch(facecolor=GRAYBLUE,
                              label=f"4-3 费用较高（{np.count_nonzero(delta < 0)} 天）")],
               loc="outside upper center", ncol=2, frameon=False, fontsize=7.8,
               handlelength=1.6, columnspacing=2.0)
    for label in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"):
        i = dates.index(label)
        ax.axvline(i, color=GRAY_LT, linestyle="-", linewidth=0.8)
        ax.text(i, 6.95, label[5:], rotation=90, ha="right", va="bottom",
                fontsize=7.0, color=GRAY)
    ax.annotate(f"最大节省 {dates[ip][5:]}：{delta[ip]:,.2f} 元",
                xy=(ip, delta[ip] / 1e4), xytext=(ip + 16, 5.30),
                fontsize=7.8, color=ORANGE, va="center",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=0.9))
    ax.annotate(f"最大负节省 {dates[ineg][5:]}：{delta[ineg]:,.2f} 元",
                xy=(ineg, delta[ineg] / 1e4), xytext=(ineg + 16, -1.55),
                fontsize=7.8, color=GRAY, va="center",
                arrowprops=dict(arrowstyle="-|>", color=GRAY, linewidth=0.9))

    # ---- b 累计节省
    ax = fig.add_subplot(gs[1], sharex=ax)
    ax.fill_between(x, 0, cum / 1e4, color=ORANGE_LT, linewidth=0)
    ax.plot(x, cum / 1e4, color=ORANGE, linewidth=1.5)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[t][5:] for t in ticks])
    ax.set_xlabel("评价期日期（2025-02-01 至 2025-12-31，共 334 天）")
    ax.set_ylabel("累计节省 / 万元")
    ax.set_title("b　334 天累计节省", loc="left", pad=4)
    ax.set_ylim(-12, 205)
    style(ax)
    for label in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"):
        i = dates.index(label)
        ax.axvline(i, color=GRAY_LT, linestyle="-", linewidth=0.8)
        ax.text(i, 143, label[5:], rotation=90, ha="right", va="bottom",
                fontsize=7.0, color=GRAY)
    ax.annotate(f"终点 {cum[-1]:,.2f} 元（{cum[-1] / 1e4:,.4f} 万元）",
                xy=(len(dates) - 1, cum[-1] / 1e4), xytext=(333, 180),
                fontsize=8.2, color=ORANGE, ha="right",
                bbox=dict(boxstyle="square,pad=0.28", facecolor="white",
                          edgecolor="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=0.9))

    save(fig, "fig_6_8_q4_daily_and_cumulative_savings")


# ================================ 图6-9 4-2 参考调度与实际补救（代表日 2025-09-23）
# 与前面几张问题四图件不同，本图不看费用汇总，而是把一个自然日的执行过程摊开：
# 优化器给的参考动作 → 实际出现供需偏差 → 储能动作被削减/追加 → 剩余缺口紧急购电。
# 配色沿用全篇语义：蓝=储能动作与状态，橙=紧急购电（与图6-3、图6-4 的橙色语义一致），
# 灰=基准与剩余；充放电一律靠方向区分，深浅只用来区分参考/实际，
# 这样同一个图里橙色不会既指放电又指紧急购电。
REF_TRAJ = Q4 / "q4-2_reference_trajectory_K30.npz"
REF_DAY = "2025-09-23"
_Q4_REF_STATS: dict | None = None


def _q4_ref_stats(q2: dict, q4: dict) -> dict:
    """图6-9 的全部取值与核对，只取代表日 2025-09-23。"""
    global _Q4_REF_STATS
    if _Q4_REF_STATS is not None:
        return _Q4_REF_STATS

    traj = np.load(REF_TRAJ, allow_pickle=False)
    day = int(np.where(traj["dates"] == REF_DAY)[0][0])
    # q4 backtest 只在第 0 天把首段改写为冷启动值；代表日不在第 0 天，
    # 因此 q2.load_inputs 取到的 net_actual 与 q4 backtest 用的是同一份。
    assert day != 0, "代表日不能是第 0 天，否则实际净负荷与 q4 backtest 不同源"
    net = q2["net_actual"][day]

    nat, x = traj["natural_x"][day], traj["x"][day]
    c, g = traj["c"][day], traj["g"][day]
    z, w = traj["z"][day], traj["w"][day]
    soc = traj["S"][day]
    ref_c = traj["reference_charge_145"][day][:144]
    ref_g = traj["reference_discharge_145"][day][:144]
    gap = net - nat

    payload = {str(d["date"]): d for d in q4["p2"]["days"]}[REF_DAY]
    # 与交付产物对账：图上的计划购电量、紧急购电量、起止储电量必须与交付 payload 一致。
    check("图6-9 情景数", float(traj["scenario_count"][day]), 14.0, 1e-9)
    check("图6-9 计划购电量与交付 payload 一致",
          float(x.sum()), payload["plan_total_kwh"], 1e-6, " kWh")
    check("图6-9 紧急购电量与交付 payload 一致",
          float(z.sum()), payload["emergency_total_kwh"], 1e-6, " kWh")
    check("图6-9 起始储电量与交付 payload 一致",
          float(soc[0]), payload["soc_start_kwh"], 1e-6, " kWh")
    check("图6-9 日末储电量与交付 payload 一致",
          float(soc[-1]), payload["soc_end_kwh"], 1e-6, " kWh")
    check("图6-9 参考充电合计", float(ref_c.sum()), 20663.215081, 0.01, " kWh")
    check("图6-9 参考放电合计", float(ref_g.sum()), 15373.025114, 0.01, " kWh")
    check("图6-9 实际充电合计", float(c.sum()), 18536.499875, 0.01, " kWh")
    check("图6-9 实际放电合计", float(g.sum()), 15014.782713, 0.01, " kWh")
    check("图6-9 剩余电量合计", float(w.sum()), 22.461233, 0.01, " kWh")
    check("图6-9 缺口合计", float(gap[gap > 0].sum()), 19623.191587, 0.01, " kWh")
    check("图6-9 富余合计", float(gap[gap < 0].sum()), -18558.961108, 0.01, " kWh")
    check("图6-9 紧急购电时段数", float((z > 1e-9).sum()), 25.0, 1e-9)
    # 中图三层必须与浅灰缺口柱严丝合缝：缺口 = 放电 − 充电 + 紧急购电 − 剩余。
    check("图6-9 缺口分解恒等式残差",
          float(np.abs(gap - (g - c + z - w)).max()), 0.0, 1e-6, " kWh")
    # 同一式子的另一读法：实际净负荷可由轨迹反算，用来说明浅灰柱取自实测而非估计。
    check("图6-9 实际净负荷自洽残差",
          float(np.abs(net - (nat + g - c + z - w)).max()), 0.0, 1e-6, " kWh")
    check("图6-9 SOC 触及下界", float(soc.min()), 1200.0, 1e-9, " kWh")
    check("图6-9 SOC 未越上界", float(max(soc.max() - 10800.0, 0.0)), 0.0, 1e-9, " kWh")

    _Q4_REF_STATS = {
        "day": day, "net": net, "nat": nat, "x": x, "gap": gap,
        "ref_c": ref_c, "ref_g": ref_g, "c": c, "g": g, "z": z, "w": w,
        "soc": soc, "emergency_seg": z > 1e-9,
        "cut_charge": float(np.clip(ref_c - c, 0, None).sum()),
        "add_discharge": float(np.clip(g - ref_g, 0, None).sum()),
    }
    return _Q4_REF_STATS


def fig_6_9_q4_2_reference_replay(q2: dict, q4: dict) -> None:
    st = _q4_ref_stats(q2, q4)
    seg = np.arange(144)
    soc_x = np.arange(145)

    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 184 * MM), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.80])
    fig.get_layout_engine().set(hspace=0.16)

    # ------------------------------------------------- a 参考动作与实际动作
    ax = fig.add_subplot(gs[0])
    # 参考柱占满整段、实际柱窄一半叠在正中：深浅之外再加宽度差，
    # 黑白打印时不会把「参考」误读成「实际」的浅色版本。
    ax.bar(seg, st["ref_c"], width=1.0, align="edge", color=GRAYBLUE, linewidth=0)
    ax.bar(seg, -st["ref_g"], width=1.0, align="edge", color=GRAYBLUE, linewidth=0)
    ax.bar(seg + 0.21, st["c"], width=0.58, align="edge", color=BLUE, linewidth=0)
    ax.bar(seg + 0.21, -st["g"], width=0.58, align="edge", color=BLUE, linewidth=0)
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.set_ylim(-980, 980)
    ax.set_title("a　参考动作与实际动作", loc="left", pad=4, fontsize=9.4)
    ax.set_ylabel("充放电量 / kWh\n（每 10 分钟）")
    time_ticks(ax, 24)
    style(ax)
    note(ax, 0.005, 0.97,
         f"向上充电、向下放电　|　参考充/放 {st['ref_c'].sum():,.0f} / {st['ref_g'].sum():,.0f} kWh，"
         f"实际充/放 {st['c'].sum():,.0f} / {st['g'].sum():,.0f} kWh",
         va="top", color=INK)
    ax.legend(handles=[Patch(facecolor=GRAYBLUE, label="参考动作（SAA 情景平均）"),
                       Patch(facecolor=BLUE, label="实际执行动作")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False,
              handlelength=1.3, columnspacing=1.2)

    # ------------------------------------------------- b 供需偏差与补救构成
    ax = fig.add_subplot(gs[1])
    # 中灰全宽柱＝待补的缺口本身；窄柱＝把它补掉的三种机制。
    # 三者之和恒等于全宽柱高度，所以窄柱必然把灰柱填满，不残留。
    # 缺口柱用中灰而非浅灰：浅灰在 144 根窄柱的缝隙里几乎看不见，读者就只
    # 看到补救构成、看不到被补的那个缺口。橙柱另加白色斜纹边，
    # 与灰度相近的灰柱在黑白打印时仍分得开。
    ax.bar(seg, st["gap"], width=1.0, align="edge", color=GRAY, linewidth=0,
           label="供需偏差（实际净负荷 − 计划购电量）")
    ax.bar(seg + 0.25, st["g"], width=0.50, align="edge", color=BLUE, linewidth=0)
    ax.bar(seg + 0.25, st["z"], width=0.50, align="edge", bottom=st["g"], color=ORANGE,
           hatch="////", edgecolor="white", linewidth=0, label="紧急购电")
    ax.bar(seg + 0.25, -st["c"], width=0.50, align="edge", color=BLUE, linewidth=0)
    ax.bar(seg + 0.25, -st["w"], width=0.50, align="edge", bottom=-st["c"], color=GRAY_LT,
           linewidth=0, label="剩余电量")
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.set_ylim(-1000, 1000)
    ax.set_title("b　供需偏差与补救构成", loc="left", pad=4, fontsize=9.4)
    ax.set_ylabel("电量 / kWh\n（每 10 分钟）")
    time_ticks(ax, 24)
    style(ax)
    note(ax, 0.005, 0.97,
         f"缺口 {st['gap'][st['gap'] > 0].sum():,.0f} kWh、富余 {abs(st['gap'][st['gap'] < 0].sum()):,.0f} kWh"
         f"（{int((np.abs(st['gap']) > 1e-6).sum())}/144 段非零）　|　"
         f"逐段缺口 = 放电 − 充电 + 紧急购电 − 剩余，残差 0",
         va="top", color=INK)
    ax.legend(handles=[Patch(facecolor=GRAY, label="供需偏差（缺口 / 富余）"),
                       Patch(facecolor=BLUE, label="储能放电（上）/ 充电（下）"),
                       Patch(facecolor=ORANGE, hatch="////", edgecolor="white",
                             label="紧急购电"),
                       Patch(facecolor=GRAY_LT, label="剩余电量")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False,
              handlelength=1.3, columnspacing=1.2)

    # ------------------------------------------------- c 储电量轨迹与运行边界
    ax = fig.add_subplot(gs[2])
    for bound, tag, va in ((10800.0, "上限 10800 kWh", "bottom"),
                           (1200.0, "下限 1200 kWh", "top")):
        ax.axhline(bound, color=GRAY, linewidth=0.9, linestyle="--", dashes=(4, 2))
        ax.text(143.5, bound + (170 if va == "bottom" else -170), tag, ha="right",
                va=va, fontsize=7.2, color=GRAY)
    ax.plot(soc_x, st["soc"], color=BLUE, linewidth=1.3, zorder=4)
    hot = st["emergency_seg"]
    ax.plot(soc_x[:144][hot], st["soc"][:144][hot], linestyle="none", marker="o",
            markersize=2.7, color=ORANGE, zorder=5)
    i_min = int(st["soc"].argmin())
    ax.annotate(f"最低 {st['soc'].min():,.0f} kWh（{i_min // 6:02d}:{(i_min % 6) * 10:02d}）",
                xy=(i_min, st["soc"].min()), xytext=(i_min + 6, 3600),
                fontsize=7.4, color=INK, ha="left", va="center",
                bbox=dict(boxstyle="square,pad=0.28", facecolor="white",
                          edgecolor="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, linewidth=0.8))
    ax.set_ylim(0, 11500)
    ax.set_title("c　储电量轨迹与运行边界", loc="left", pad=4, fontsize=9.4)
    ax.set_ylabel("储电量 / kWh")
    ax.set_xlabel("时间（区间起点）")
    time_ticks(ax, 24)
    style(ax)
    note(ax, 0.005, 0.97,
         f"起点 {st['soc'][0]:,.0f} kWh → 日末 {st['soc'][-1]:,.0f} kWh　|　"
         f"全程落在运行边界内，其中 {int((np.abs(st['soc'] - 1200.0) < 1e-6).sum())} 段压在下限上",
         va="top", color=INK, box=True)
    ax.legend(handles=[Line2D([], [], color=BLUE, linewidth=1.3, label="储电量"),
                       Line2D([], [], color=GRAY, linewidth=0.9, linestyle="--",
                              dashes=(4, 2), label="运行边界"),
                       Line2D([], [], color=ORANGE, marker="o", markersize=3.0,
                              linestyle="none", label="紧急购电时段")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False,
              handlelength=1.3, columnspacing=1.2)

    save(fig, "fig_6_9_q4_2_reference_replay")


# ============================================================= 附件5 对账抽查
def crosscheck_annex5(q1: dict, q4: dict) -> None:
    from openpyxl import load_workbook

    ws = load_workbook(ANNEX5 / "result1.xlsx", read_only=True, data_only=True)["计划购电量"]
    total = sum(float(r[1]) for r in ws.iter_rows(min_row=2, max_row=145, values_only=True)
                if r[1] is not None)
    check("附件5 result1 计划购电量合计", total, q1["grid"].sum(), 1e-6, " kWh")

    ws = load_workbook(ANNEX5 / "result4-3.xlsx", read_only=True, data_only=True)["紧急购电量"]
    tot = 0.0
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r and r[-1] is not None:
            try:
                tot += float(r[-1])
            except (TypeError, ValueError):
                pass
    check("附件5 result4-3 紧急购电量合计", tot, q4["s3"]["totals"]["emergency_kwh"],
          0.05, " kWh")


# ================================================================== 图注与清单
CAPTIONS = {
    "fig_5_1_time_mapping_and_information_boundary": (
        "图5-1　原始模板行到自然日 144 段的跨行映射与四阶段决策信息边界",
        [
            "图中给出论文全部时间口径的统一定义。原始模板行共 144 项，行标为区间起点，"
            "自 0:10 起至次日 0:00+1 止；自然日统计窗口为 00:00—24:00 的 144 段，"
            "其首段取自前一原始行的末项 $X^{raw}_{\\nu-1,143}$（橙色箭头），其余 143 段取自当日原始行"
            "第 1—143 项（蓝色箭头），因此标准化的时间轴不通过在同一行内循环右移获得。",
            "面板 b 给出 0:00、6:00、12:00、18:00 四个决策节点的信息边界：节点左侧为已实现信息，"
            "右侧为尚未观测的区间，蓝色实心区间为该节点起至下一启用阶段前被锁定的合同范围；"
            "18:00 另以橙色标出次日 00:00—00:10 的午夜晚段。",
            "0:00 的规划窗口为 145 段，由已承诺的午夜首段与当日 144 段共同构成，"
            "与自然日 144 段、模板行 144 项三者口径不同，不可混用。",
            "历史样本的可用截止位置随节点后移：0:00 时最新完整模板行为 $\\nu-2$，"
            "6:00 及以后可用至 $\\nu-1$，并可利用当日已完成时段。",
        ],
        "方法示意图。图中不含实测量值；仅示意 144 项模板行、144 段自然日与 145 段规划窗口的口径差别，"
        "以及四个决策节点的信息可观测范围。",
        "无（方法示意图）。口径依据：论文 5.2—5.5 节；映射规则实现见 "
        "src/src/q1_solver.py:load_inputs 与 src/src/q2_solver.py:load_inputs。",
        "本图为口径示意，不构成任何数值结论；具体取值须以各问的求解输出为准。",
        [PAPER, SRC / "q1_solver.py", SRC / "q2_solver.py"],
    ),
    "fig_5_2_pv_forecast_interpolation": (
        "图5-2　光伏预报的时间尺度转换：整点预报到 10 分钟交付序列的插值过程",
        [
            "以 2025-06-21 06:00 发布为例：附件3 给出当日 18 个整点预报节点（空心蓝圆），"
            "发布时刻前最后一个已实现时段的光伏出力 735.83 kW 作为锚点（橙色菱形），"
            "防止插值曲线在发布时刻附近悬空。",
            "面板 a 中科研蓝实线为本文采用的 PCHIP 保形插值：它在整点节点之间生成 10 分钟序列，"
            "节点处不出现折角，且不产生过冲、天然保持非负。灰虚线为线性插值，"
            "两者的最大偏差为 225.44 kW，出现在 13:10，偏差集中在日出后与日落前的陡升陡降段。",
            "面板 b 放大 18:00—20:00 日落段：三次样条（深蓝点划线）在出力陡降处过冲，"
            "最低达 −50.07 kW（19:20），当日共 15 个交付时段被插成负功率（浅橙底纹），物理不可行；"
            "PCHIP 与线性插值在该段的最小值均为 0。",
        ],
        "案例日 2025-06-21，发布时刻 06:00，预报窗口 06:00—24:00，共 109 个交付时段；"
        "整点节点间隔 1 h，交付时段间隔 10 min。功率单位为 kW。",
        "附件3 的光伏整点预报与附件2 的实测光伏功率；插值实现见 src/src/q2_solver.py 的"
        "预报降尺度与情景构造函数。",
        "本图只说明时间尺度转换的算法选择，不评价预报精度；替代插值的缺陷是算法性质，"
        "与所选日期无关。降尺度方式会改变问题二至四的净负荷序列，"
        "因此该口径须与各问求解器保持一致。",
        [DATA / "附件3.xlsx", DATA / "附件2.xlsx", SRC / "q2_solver.py"],
    ),
    "fig_5_3_annual_data_heatmap": (
        "图5-3　全年数据特征的时段热力图：小区负载、光伏出力与电网电价",
        [
            "三个子图共用日期轴（2025 年 365 天）与日内时刻轴（00:00—24:00，10 分钟一格），"
            "用分档色深同时读出日周期与季节结构。负载量程 1995.7—7978.9 kW，"
            "日内峰值在 09:10、谷值在 22:10；光伏量程 0—10216.2 kW，日内峰值在 12:10，"
            "全年 44.79% 的时段零出力，月均出力以 8 月最强、12 月最弱。",
            "电价量程 0.0076—1.7936 元/kWh，日内峰值在 20:40、谷值在 05:40，"
            "月均电价以 12 月最高、5 月最低，全年呈 U 形走势。三个子图的季节结构并不一致："
            "光伏夏强冬弱、负载夏冬双高、电价年末最高，这正是储能跨时段套利的可操作空间。",
            "左上角 1 格为 2025-01-01 的 00:00—00:10，无原始行可取，以虚线方框标为空缺点；"
            "横贯三图的白色虚线为评价期起点 2025-02-01，其上的 1 月为预热期，不参与费用统计。",
        ],
        "全年 365 个自然日、每日 144 个 10 分钟时段。负载与光伏单位为 kW，电价为元/kWh。",
        "附件2 的小区负载与光伏发电实际功率、附件4 的电网电价；"
        "原始行（区间起点标签，自 00:10 至次日 00:00）到自然日窗口的映射见 "
        "src/src/q2_solver.py 与 src/src/q4_q2_solver.py 的 load_inputs。",
        "本图只描述输入数据的分布特征，不构成任何调度或费用结论。"
        "热力图的分档边界取整以便读数，边界附近的取值不宜作精确比较；"
        "2025-01-01 首段的空缺由求解器冷启动补足，不影响评价期统计。",
        [DATA / "附件2.xlsx", DATA / "附件4.xlsx"],
    ),
    "fig_6_1_q1_price_and_storage_dispatch": (
        "图6-1　分时电价与储能调度结果",
        [
            "三个子图共用 00:00—24:00 横轴，每段 10 分钟，时间标签为区间起点。"
            "面板 a 为分时电价阶梯（附件1 电价列），最高 1.395 元/kWh（20:40）、"
            "最低 0.371 元/kWh（05:40）；面板 b 的储能动作为充电向上（科研蓝）、"
            "放电向下并加斜纹（橙色）；面板 c 为逐段起点储电量，灰虚线为 1200 与 10800 kWh 容量边界，"
            "点划线为首末 6000 kWh 水平，方块为起止锚点。",
            "该日储电量在 1200.0000 kWh 触下限、在 10800.0000 kWh 触上限，两处边界均为紧约束；"
            "充放电峰值 833.33 kWh/10min，恰好等于 5000 kW 功率上限对应的单段电量上限。",
            "储能方案全天购电费 35126.85 元，无储能对照 48052.05 元，节省 12925.20 元（26.90%），"
            "弃光 0.00 kWh。储能不减少总负荷，它的作用是按价差把购电从高价段搬到低价段。",
        ],
        "代表日：附件1给定的一天，144 个 10 分钟时段。电量为 kWh（每 10 分钟），"
        "电价为元/kWh，费用为元，储电量为 kWh。",
        "附件1.xlsx；由 src/src/q1_solver.py 读附件1 重算的完整 144 段轨迹，"
        "并对容量边界、首末储电量、递推关系、能量平衡与功率上限逐条对账。",
        "反映了题设代表日与当前效率参数（$\\eta_c=\\eta_d=0.9$，往返 81%）下的调度结果，"
        "不表示计入储能投资、老化与维护后的全生命周期收益；也不构成购电路径唯一性的结论——"
        "独立重解曾给出费用相同但逐时段购电向量不同的另一组解。本图只反映该代表日，"
        "不能直接外推至全年。",
        [DATA / "附件1.xlsx", SRC / "q1_solver.py"],
    ),
    "fig_6_2_q1_period_cost_difference": (
        "图6-2　分时段费用变化",
        [
            "纵轴为「储能方案购电费 − 无储能方案购电费」，按 4 小时分为六段（每段 24 个 10 分钟区间）；"
            "负值表示该时段储能比无储能少支出（节省，橙色），正值表示多支出（灰蓝描边浅填充）。"
            "无储能对照按逐段净负荷 $\\max(\\text{净负荷},0)$ 全额购电、余量弃光计，不承担储能损耗。",
            "六段依次为 00:00—04:00 +1917.95、04:00—08:00 −4755.30、08:00—12:00 −1735.78、"
            "12:00—16:00 +1056.52、16:00—20:00 −6780.34、20:00—24:00 −2628.24 元，"
            "合计 −12925.20 元，与储能方案全天购电费 35126.85 元、无储能 48052.05 元之差完全一致。",
            "节省集中在 04:00—08:00 与 16:00—20:00 两段，合计 11535.64 元，占全天净节省的 89.25%；"
            "其余四段合计仍多支出 2974.47 元——储能的收益来自跨时段价差，"
            "而不是在每个时段都比无储能便宜。",
        ],
        "代表日：附件1给定的一天，144 个 10 分钟时段，每 4 小时合为一段。费用为元（每 4 小时）。",
        "附件1.xlsx；由 src/src/q1_solver.py 读附件1 重算，六段之和与全天购电费差额逐位对账一致。",
        "本图只反映该代表日的分段费用转移，不表示全年各时段的费用分布；"
        "分段边界取整点 4 小时，改用其他分段方式时各段数值会变但合计不变。"
        "收益未计储能投资、老化与维护成本。",
        [DATA / "附件1.xlsx", SRC / "q1_solver.py"],
    ),
    "fig_6_3_q2_daily_emergency_purchase": (
        "图6-3　评价期逐日紧急购电量",
        [
            "评价期 334 天中有 170 天出现紧急购电，全年紧急购电 26.7693 万kWh，"
            "折合日均 801.5 kWh；最高日 2025-06-01 达 15465.52 kWh，约为日均的 19.3 倍，"
            "灰色虚线给出全年日均水平。",
            "题目指定日 2025-09-23 为 4471.62 kWh，在全年由高到低排第 18 位；"
            "而 2025-03-20、2025-06-21 与 2025-12-21 分别仅为 9.80、0.00 与 12.90 kWh，"
            "说明紧急购电总体占比不高，仍可能伴随个别日期的集中缺口。",
            "中心预测平均绝对误差与日紧急购电量的 Pearson 相关为 0.316、Spearman 相关为 0.074，"
            "只能说明二者存在有限的统计关联。",
        ],
        "评价期 2025-02-01 至 2025-12-31，共 334 天；纵轴为每日紧急购电量（kWh）。",
        "q2_paper_audit 的 daily_metrics.csv、interval_detail.csv；"
        "逐日紧急购电量由 src/src/q2_solver.py 当前实现重算，"
        "与 daily_metrics.csv 的当日合计逐日一致，"
        "全年合计 267693.199361 kWh、最大日与 2025-09-23 数值均通过核对。",
        "紧急购电量为按当日 144 段自然日口径聚合的补购电量，不含计划购电与合同调整；"
        "逐日柱高受当日净负荷水平、预测误差方向与储电量状态共同影响，"
        "MAE 与紧急购电量的相关系数仅为描述性统计，不构成因果识别。",
        [Q2_AUDIT / "daily_metrics.csv", Q2_AUDIT / "interval_detail.csv",
         SRC / "q2_solver.py", DATA / "附件2.xlsx"],
    ),
    "fig_6_4_q2_forecast_scenario_range": (
        "图6-4　指定日期净负荷预测与经验情景范围",
        [
            "四个指定日期（按面板顺序为 2025-03-20、2025-06-21、2025-09-23、2025-12-21）"
            "的中心预测净负荷（蓝色虚线）、实际净负荷（深灰实线）、"
            "经验最小—最大包络（浅灰）与经验 10%—90% 分位带（灰蓝）。"
            "四日共 576 个自然日时段中，实际值落入经验包络 394 段（68.40%），"
            "落入 10%—90% 分位带 314 段（54.51%）；"
            "分日包络覆盖依次为 125/144、55/144、90/144、124/144，"
            "分日 10%—90% 分位带覆盖依次为 101/144、37/144、63/144、113/144。",
            "橙色圆点与浅橙底色标出该时段实际发生了紧急购电：分位带越窄或实际净负荷越靠近包络上缘，"
            "可用的调节余量越少，越容易触发紧急购电。",
        ],
        "四个指定日期各 144 段自然日时段，时间标签为区间起点；"
        "纵轴为净负荷电量（kWh，每 10 分钟）。",
        "q2_paper_audit 的 interval_detail.csv、coverage.json；"
        "情景带由 src/src/q2_solver.py 的当前情景构造函数重建（未使用正态区间替代），"
        "重建后覆盖计数与 coverage.json 完全一致（576 / 394 / 314），"
        "且四日的预测与实际序列与 interval_detail.csv 逐段一致。",
        "包络与分位带为历史残差情景的经验范围，不是统计置信区间，"
        "也不代表全年覆盖率或全天无紧急购电概率。",
        [Q2_AUDIT / "interval_detail.csv", Q2_AUDIT / "coverage.json",
         SRC / "q2_solver.py", DATA / "附件2.xlsx"],
    ),
    "fig_6_5_q3_stage_cost_composition": (
        "图6-5　问题三八种阶段组合的费用构成",
        [
            "面板 a 按总费用升序给出八种预报发布与调整时刻组合的费用构成（自 0 起，量级可比）："
            "计划费用（科研蓝）+ 调整费用（灰蓝）+ 紧急费用（深蓝），灰色虚线为仅 0:00 基线 1398.69 万元。"
            "仅 0:00 决策时总费用 1398.69 万元；启用 6:00、12:00、18:00 三次更新后降至 1319.33 万元，"
            "节省 79.36 万元、降幅 5.67%，八种组合的总费用全部不高于基线——"
            "更新时刻最少的 0+6 组合也已降至 1362.94 万元，哪怕只增加一次日内更新也能获益。",
            "面板 b 略去计划费用、只放大调整与紧急两段。计划费用在八种组合间仅由 1292.55 万元变到 "
            "1210.16 万元，却占每根柱的九成以上，在面板 a 中把 5.67% 的差别压成读不出的台阶；"
            "放大后可见费用结构是「紧急费用塌缩、调整费用顶上来」：紧急费用由 106.14 万元降至 21.52 万元"
            "（−84.62 万元），新增调整费用 87.65 万元。",
            "两段相抵后非计划部分反而净增 3.03 万元（106.14 → 109.17 万元），"
            "即总费用的下降几乎全部来自计划费用减少的 82.38 万元。"
            "另需注意 0+18 组合的非计划部分最低（94.28 万元），却因计划费用偏高（1252.76 万元）"
            "总费用只排第五，单看任一分项都不足以判定组合优劣。",
        ],
        "评价期 2025-02-01 至 2025-12-31，共 334 天、48096 个自然日时段；情景数 K=30；费用为万元。",
        "q3_paper_audit 的 q3_stage_comparison.csv 与 q3_stage_comparison.json"
        "（对应交付工作簿 result3.xlsx）。",
        "结论限于当前数据、参数（含终端价值 $\\lambda=0.478$ 元/kWh、$K=30$）与实时补救规则，"
        "且不另计预报获取成本。费用构成只反映各组合的总费用拆分，不表示各时刻更新的边际价值，"
        "更新时刻越多并不必然越省。模型为滚动两阶段 / SAA 近似，不构成严格多阶段随机最优模型。",
        [Q3_AUDIT / "q3_stage_comparison.csv", Q3_AUDIT / "q3_stage_comparison.json",
         ANNEX5 / "result3.xlsx"],
    ),
    "fig_6_6_q3_shapley_allocation": (
        "图6-6　问题三三阶段 Shapley 收益分摊",
        [
            "相对仅 0:00 的节省在三个日内更新时刻上的 Shapley 分摊：6:00 为 23.28 万元（29.33%）、"
            "12:00 为 21.97 万元（27.69%）、18:00 为 34.11 万元（42.98%），18:00 贡献最大；"
            "三者加总 79.36 万元，与全阶段相对仅 0:00 的总节省一致。",
            "每根柱下方的灰色竖线为该时刻在 4 种上下文下的条件边际节省（共 12 项），圆点为四者均值；"
            "12 项条件边际节省全部为正，最小值为 10.41 万元。",
            "若仅能保留一次日内更新，0:00 与 18:00 的组合在三个单次更新方案中费用最低（1347.05 万元）。",
        ],
        "评价期 2025-02-01 至 2025-12-31，共 334 天、48096 个自然日时段；情景数 K=30；费用为万元。",
        "q3_paper_audit 的 q3_stage_comparison.csv 与 q3_stage_comparison.json"
        "（对应交付工作簿 result3.xlsx）。",
        "Shapley 值分摊的是预报更新、合同重优化与储能状态演化的综合节省，"
        "不是预报信息本身的纯因果价值；单独引入各时刻的收益不可直接相加——"
        "三根柱之和等于总节省，但任一根都不等于“只增加该时刻”所能得到的节省。"
        "结论限于当前数据、参数与实时补救规则。模型为滚动两阶段 / SAA 近似，"
        "不构成严格多阶段随机最优模型。",
        [Q3_AUDIT / "q3_stage_comparison.csv", Q3_AUDIT / "q3_stage_comparison.json",
         ANNEX5 / "result3.xlsx"],
    ),
    "fig_6_7_q4_aggregate_comparison": (
        "图6-7　问题四两种策略的费用构成与紧急购电量比较",
        [
            "4-2 在 334 天评价期的总费用为 1520.18 万元，其中计划费用 1349.17 万元、紧急费用 171.01 万元，"
            "无调整费用；4-3 的总费用为 1387.65 万元，其中计划费用 1273.42 万元、调整费用 91.61 万元、"
            "紧急费用 22.62 万元。紧急费用在 4-2 中占据显著份额，在 4-3 中被压缩到很小比例。",
            "4-3 相对 4-2 总费用减少 132.53 万元，降幅 8.72%。该净差由计划费用减少 75.74 万元、"
            "紧急费用减少 148.40 万元与新增 91.61 万元调整费用共同形成。",
            "紧急购电量由 27.2135 万kWh 降至 4.1457 万kWh，降幅 84.77%，紧急费用占比由 11.25% 降至 1.63%。",
        ],
        "评价期 2025-02-01 至 2025-12-31，共 334 个自然日、48096 个区间；情景数 K=30；"
        "费用为万元，紧急购电量为万kWh。",
        "q4 的 summary_q4-2_K30.json 与 summary_q4-3_K30.json"
        "（对应交付工作簿 result4-2.xlsx、result4-3.xlsx）。",
        "4-2 与 4-3 同时在净负荷预测、情景构造、调整机会、实时储能规则与终端价值等方面存在差异，"
        "费用差为两套完整策略的综合比较结果，不能全部归因于日内价格更新或新增调整机会。"
        "模型为滚动两阶段 / SAA 近似，不构成严格多阶段随机最优模型。",
        [Q4 / "summary_q4-2_K30.json", Q4 / "summary_q4-3_K30.json",
         ANNEX5 / "result4-2.xlsx", ANNEX5 / "result4-3.xlsx"],
    ),
    "fig_6_8_q4_daily_and_cumulative_savings": (
        "图6-8　问题四逐日费用节省与 334 天累计节省",
        [
            "逐日节省定义为 $\\Delta C_d = C_{4\\text{-}2,d} - C_{4\\text{-}3,d}$。334 天中 4-3 费用较低的为 237 天（科研蓝），"
            "费用较高的为 97 天（橙色），全年费用较低并不意味着每日均占优。",
            "最大单日节省出现在 2025-06-01，为 68895.88 元；最大单日负节省出现在 2025-05-04，为 −5212.58 元。"
            "题目指定的四个日期中，2025-03-20 的 4-3 费用比 4-2 高 832.81 元，2025-06-21、2025-09-23、"
            "2025-12-21 则分别低 4981.64、21617.95 与 859.90 元。",
            "累计节省曲线全程保持正向且逐步抬升，终点为 1325304.85 元（132.5305 万元），"
            "与两策略全年总费用之差一致。",
        ],
        "评价期 2025-02-01 至 2025-12-31，共 334 天；费用为元（图中以万元显示）。",
        "q4 的 payload_q4-2_K30.json、payload_q4-3_K30.json 与 summary_q4-2_K30.json、"
        "summary_q4-3_K30.json（逐日费用经两两互校，最大差为 0）。",
        "日度差异反映两套完整策略在不同日期下的综合执行结果，单日胜负不构成机制的因果识别；"
        "累计结果只对应当前年度样本与当前参数，不外推至其他年份或其他情景构造方式。",
        [Q4 / "payload_q4-2_K30.json", Q4 / "payload_q4-3_K30.json",
         Q4 / "summary_q4-2_K30.json", Q4 / "summary_q4-3_K30.json"],
    ),
    "fig_6_9_q4_2_reference_replay": (
        "图6-9　4-2 策略的参考调度与实际补救过程",
        [
            "以 2025-09-23 为代表日（当日情景数 14，为评价期内的上限档）。上图比较 SAA 情景平均参考"
            "充放电量与实际执行量：参考充电 20663.22 kWh、参考放电 15373.03 kWh，实际充电 18536.50 kWh、"
            "实际放电 15014.78 kWh，即实际相对参考少充 2126.72 kWh、少放 358.24 kWh。"
            "按段统计，削减充电合计 3793.96 kWh、追加放电合计 2869.29 kWh，"
            "与“缺电时先削充电、不足再追加放电、仍有缺口才紧急购电”的补救次序一致。",
            "中图以灰色柱给出实际供需偏差（实际净负荷 − 计划购电量），当日缺额合计 19623.19 kWh、"
            "富余合计 18558.96 kWh。缺口由储能放电（蓝，向上）与紧急购电（橙，斜纹，向上）逐级补足，"
            "富余由储能充电（蓝，向下）与剩余电量（浅灰，向下）吸收；窄柱之和逐段严格等于灰柱高度"
            "（残差为 0），即缺口 = 放电 − 充电 + 紧急购电 − 剩余，不残留未被覆盖的偏差。",
            "当日紧急购电 4608.41 kWh，集中在 25 个时段，最大单段 827.18 kWh；"
            "这些时段在下图以橙点标出，可见紧急购电发生在储电量已被用到接近下限的时段。",
            "下图给出储电量变化：自 1217.94 kWh 出发，全程落在 1200—10800 kWh 运行边界之内，"
            "其中 25 段恰好压在下限 1200.00 kWh 上，日末回到 1217.69 kWh，与起点几乎一致。"
            "补救过程把缺口补满，同时没有越出储能的物理边界。",
        ],
        "代表日 2025-09-23，属 2025-02-01 至 2025-12-31 评价期；自然日 144 段，时间标签为区间起点；"
        "电量为 kWh（每 10 分钟）；储电量为 kWh。参考轨迹含 τ=144 的日末边界，"
        "与实际 144 段动作比较时取前 144 项。",
        "q4/q4/q4-2_reference_trajectory_K30.npz（重跑保存的情景平均参考轨迹）、"
        "q4/q4/payload_q4-2_K30.json（交付工作簿 payload）与 "
        "q4/q4/q4-2_reference_replay_verification.json（独立重放检验报告）；"
        "实际净负荷取自 data/data 附件，经 src/src/q2_solver.py:load_inputs 读取。",
        "计划购电量按自然日口径执行：00:00 段承接前一日的午夜承诺，其余段沿用当日计划，"
        "当日计划购电量合计 61109.54 kWh（与交付 payload 一致）。"
        "本图只还原一个代表日的执行链路，用来说明参考动作如何被逐时段修正为实际动作，"
        "不构成全年费用或节省结论。模型为滚动两阶段 / SAA 近似，不属严格多阶段随机最优模型；"
        "该日执行过程可复现也不等于最优解唯一或全年全局最优。",
        [REF_TRAJ, Q4 / "payload_q4-2_K30.json",
         Q4 / "q4-2_reference_replay_verification.json"],
    ),
}


# 由同级脚本单独生成、只在此登记图注与清单的图件：stem -> 生成脚本。
# 这几张图的数据源、口径与自检都在各自脚本里，本脚本不重绘它们，
# 只在组装清单前确认产物已在盘上；图注文字与各脚本 caption() 保持一致，
# 改动其中一处时须同步另一处。
EXTERNAL_STEMS = {
    "fig_5_2_pv_forecast_interpolation":
        "scripts/scripts/render_pv_interpolation_figure.py",
    "fig_5_3_annual_data_heatmap":
        "scripts/scripts/render_data_characteristics_figure.py",
    "fig_6_1_q1_price_and_storage_dispatch":
        "scripts/scripts/render_dispatch_overview_figure.py",
    "fig_6_2_q1_period_cost_difference":
        "scripts/scripts/render_period_cost_figure.py",
}


def write_captions() -> None:
    lines = [
        "# 论文图件图注与分析",
        "",
        f"生成脚本：`scripts/scripts/render_paper_figures.py`（版本 {SCRIPT_VERSION}）　"
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "本文件同时收录由同级脚本生成的图件，各自的数据源与自检见各脚本："
        + "；".join(f"`{s}`（`{n}`）" for n, s in EXTERNAL_STEMS.items()) + "。",
        "",
        "数据口径统一说明：时间标签一律为区间起点；自然日 00:00 首段取自前一原始行末项。"
        "问题二至四的费用比较期为 2025-02-01 至 2025-12-31，共 334 天、48096 个时段。"
        "问题三与问题四采用滚动两阶段 / SAA 近似，不属严格多阶段随机最优模型。",
        "",
        "视觉规范：白底；科研蓝为主色，橙色为唯一重点强调色，对照与次要结果用灰 / 灰蓝；"
        "去顶部与右侧边框；浅灰弱网格；同族系列以颜色 + 线型 + marker 多重区分。",
        "",
        "---",
        "",
    ]
    for stem, (title, analysis, period, sources, limits, _files) in CAPTIONS.items():
        lines += [f"## {title}", "", f"**文件**：`{stem}.pdf`、`{stem}.png`", "",
                  "**分析**", ""]
        lines += [f"{i}. {text}" for i, text in enumerate(analysis, 1)]
        lines += ["", f"**数据期间与单位**：{period}", "", f"**数据源**：{sources}", "",
                  f"**结论适用边界**：{limits}", "", "---", ""]
    (FIGDIR / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")
    print("  已输出 figure_captions.md")


def write_manifest(figures: list[dict]) -> None:
    manifest = {
        "generator": {
            "script": "scripts/scripts/render_paper_figures.py",
            "version": SCRIPT_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "png_dpi": PNG_DPI,
            "design_width_mm": FULL_WIDTH_MM,
        },
        "conventions": {
            "time_labels": "区间起点",
            "natural_day_intervals": 144,
            "template_row_items": 144,
            "stage0_planning_window_segments": 145,
            "cost_period": "2025-02-01 至 2025-12-31，334 天 / 48096 时段",
            "q34_model_class": "滚动两阶段 / SAA 近似（非严格多阶段随机最优）",
            "visual": {
                "background": "white",
                "main_color": "科研蓝",
                "accent_color": "橙色（唯一重点强调色）",
                "secondary": "灰 / 灰蓝",
                "spines": "仅保留左、下边框",
                "grid": "浅灰弱网格，仅纵轴（分类轴为横轴时用横网格）",
                "dual_axis": "不使用",
            },
            "palette": {
                "科研蓝（主色）": BLUE, "深蓝（第二层次）": BLUE_DK,
                "灰蓝（次要/背景）": GRAYBLUE, "浅灰蓝（浅填充）": GRAYBLUE_LT,
                "灰（对照基准）": GRAY, "浅灰（网格/弱包络）": GRAY_LT,
                "橙（唯一强调）": ORANGE, "浅橙（强调底纹）": ORANGE_LT,
                "文字与坐标轴": INK,
            },
        },
        "figures": figures,
    }
    (FIGDIR / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  已输出 figure_manifest.json")


def main() -> None:
    print("读取数据源并重算……")
    q1, q2, q3, q4 = load_q1(), load_q2(), load_q3(), load_q4()
    print("附件5 对账抽查……")
    crosscheck_annex5(q1, q4)

    print("绘制图件……")
    fig_5_1()
    fig_6_3_q2_daily_emergency(q2)
    fig_6_4_q2_scenario_range(q2)
    fig_6_5(q3)
    fig_6_6_shapley(q3)
    fig_6_7(q4)
    fig_6_8(q4)
    fig_6_9_q4_2_reference_replay(q2, q4)

    # 外挂图件由同级脚本各自生成（各脚本自带数据源与对账），此处只核对产物存在。
    # 它们不进上面这段绘制流程，只登记到图注与清单里。
    for stem, script in EXTERNAL_STEMS.items():
        for ext in (".pdf", ".png"):
            path = FIGDIR / f"{stem}{ext}"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} 不存在；请先运行 {script} 生成该图件")

    stems = [
        "fig_5_1_time_mapping_and_information_boundary",
        "fig_5_2_pv_forecast_interpolation",
        "fig_5_3_annual_data_heatmap",
        "fig_6_1_q1_price_and_storage_dispatch",
        "fig_6_2_q1_period_cost_difference",
        "fig_6_3_q2_daily_emergency_purchase",
        "fig_6_4_q2_forecast_scenario_range",
        "fig_6_5_q3_stage_cost_composition",
        "fig_6_6_q3_shapley_allocation",
        "fig_6_7_q4_aggregate_comparison",
        "fig_6_8_q4_daily_and_cumulative_savings",
        "fig_6_9_q4_2_reference_replay",
    ]
    from PIL import Image

    figures = []
    for stem in stems:
        png = FIGDIR / f"{stem}.png"
        with Image.open(png) as im:
            w_px, h_px = im.size
        figures.append({
            "id": stem,
            "title_cn": CAPTIONS[stem][0],
            "generator_script": EXTERNAL_STEMS.get(
                stem, "scripts/scripts/render_paper_figures.py"),
            "pdf": str((FIGDIR / f"{stem}.pdf").relative_to(ROOT)),
            "png": str(png.relative_to(ROOT)),
            "width_mm": round(w_px / PNG_DPI * 25.4, 2),
            "height_mm": round(h_px / PNG_DPI * 25.4, 2),
            "png_px": [w_px, h_px],
            "dpi": PNG_DPI,
            "data_sources": [
                {"path": str(Path(p).relative_to(ROOT)), "sha256": sha256(Path(p))}
                for p in CAPTIONS[stem][5]
            ],
        })

    write_captions()
    write_manifest(figures)

    print("\n================ 数值核对汇总 ================")
    width = max(len(c[0]) for c in CHECKS)
    for label, got, want, _, status in CHECKS:
        print(f"[{status}] {label:<{width}}  实得 {got:>20}  期望 {want:>20}")
    bad = [c for c in CHECKS if c[4] != "OK"]
    print(f"\n共 {len(CHECKS)} 项核对，通过 {len(CHECKS) - len(bad)} 项，失败 {len(bad)} 项。")
    print(f"输出目录：{FIGDIR}")


if __name__ == "__main__":
    main()
