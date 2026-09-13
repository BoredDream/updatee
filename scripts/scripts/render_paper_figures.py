"""渲染数学建模竞赛论文图件（图5-1、图6-1 ~ 图6-5）。

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

SCRIPT_VERSION = "2.0.0"

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


def note(ax, x, y, text, ha="left", va="top", size=7.4, color=GRAY):
    """图内说明文字：置于轴内、不参与布局计算、可读性由白底保证。"""
    t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                fontsize=size, color=color, linespacing=1.5)
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


# =========================================================== 图6-1 问题一调度机制
def fig_6_1(q1: dict) -> None:
    price, load, pv = q1["price"], q1["load"], q1["pv"]
    grid_main, grid_base = q1["grid"], q1["baseline_grid"]
    charge, discharge, soc = q1["charge"], q1["discharge"], q1["soc"]
    base_cost, main_cost = q1["baseline_cost"], q1["main_cost"]
    saving = base_cost - main_cost

    check("问题一无储能购电费用", base_cost, 48052.046591, 0.01, " 元")
    check("问题一储能方案购电费用", main_cost, 35126.848589, 0.01, " 元")
    check("问题一节省率", saving / base_cost, 0.268983, 1e-6)
    check("问题一0:00储电量", soc[0], 6000.0, 1e-6, " kWh")
    check("问题一24:00储电量", soc[-1], 6000.0, 1e-6, " kWh")

    x = np.arange(144)
    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 248 * MM), layout="constrained")
    gs = fig.add_gridspec(5, 1, height_ratios=[1.15, 0.62, 0.80, 0.80, 0.72])
    axes = []

    # ---- a 负荷 / 光伏 / 购电
    ax = fig.add_subplot(gs[0]); axes.append(ax)
    ax.plot(x, load, color="#4D4D4D", linewidth=1.0, label="负荷电量")
    ax.plot(x, pv, color=GRAYBLUE, linewidth=1.3, label="光伏电量")
    ax.plot(x, grid_base, color=GRAY, linewidth=1.0, linestyle="--",
            dashes=(4, 2), label="无储能方案购电量")
    ax.plot(x, grid_main, color=BLUE, linewidth=1.5, marker="o", markersize=2.0,
            markevery=12, markerfacecolor="white", markeredgewidth=0.8,
            label="储能方案购电量")
    ax.set_ylabel("每 10 分钟电量 / kWh")
    ax.set_title("a　负荷、光伏与外购电量", loc="left", pad=4)
    ax.set_ylim(0, 1680)
    time_ticks(ax); style(ax)
    ax.legend(loc="upper center", ncol=2, frameon=True, facecolor="white",
              edgecolor=GRAY_LT, framealpha=0.95, handlelength=2.2, borderpad=0.5)
    box = dict(boxstyle="square,pad=0.22", facecolor="white", edgecolor="none",
               alpha=0.85)
    note(ax, 0.995, 0.155,
         f"无储能购电费用 {base_cost:,.2f} 元　储能方案购电费用 {main_cost:,.2f} 元",
         ha="right", va="bottom", size=7.4).set_bbox(box)
    t = ax.text(0.995, 0.045, f"节省 {saving:,.2f} 元（{100 * saving / base_cost:.2f}%）",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0,
                color=ORANGE, bbox=box)
    t.set_in_layout(False)

    # ---- b 分时电价
    ax = fig.add_subplot(gs[1]); axes.append(ax)
    ax.step(np.r_[x, 144], np.r_[price, price[-1]], where="post", color=BLUE,
            linewidth=1.2)
    ax.set_ylabel("电价 /\n(元/kWh)")
    ax.set_title("b　分时电价", loc="left", pad=4)
    ax.set_ylim(0, 1.62)
    ax.set_yticks([0.4, 0.8, 1.2])
    time_ticks(ax); style(ax)

    # ---- c 储能充放电动作
    ax = fig.add_subplot(gs[2]); axes.append(ax)
    ax.bar(x, charge, width=1.0, color=BLUE, linewidth=0, label="充电量（零线上方）")
    ax.bar(x, -discharge, width=1.0, color=GRAYBLUE, linewidth=0,
           label="放电量（零线下方）")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_ylabel("充（+）/ 放（−）\n电量 / kWh")
    ax.set_ylim(-1000, 1050)
    ax.set_title("c　储能充放电动作", loc="left", pad=4)
    time_ticks(ax); style(ax)
    ax.legend(loc="lower left", ncol=2, frameon=True, facecolor="white",
              edgecolor=GRAY_LT, framealpha=0.95, handlelength=1.6)
    note(ax, 0.995, 0.06,
         f"累计充电 {charge.sum():,.2f} kWh，累计放电 {discharge.sum():,.2f} kWh",
         ha="right", va="bottom", size=7.4)

    # ---- d 储电量状态 SOC
    ax = fig.add_subplot(gs[3]); axes.append(ax)
    ax.plot(x, soc[:-1], color=BLUE, linewidth=1.5, marker="o", markersize=2.2,
            markevery=24, markerfacecolor="white", markeredgewidth=0.8,
            label="储电量 SOC")
    for value, lab, va in ((10800, "上限 10800 kWh", "bottom"),
                           (1200, "下限 1200 kWh", "top")):
        ax.axhline(value, color=ORANGE, linewidth=1.0, linestyle=(0, (1, 2)))
        ax.text(143, value + (150 if va == "bottom" else -150), lab, ha="right",
                va=va, fontsize=7.2, color=ORANGE)
    ax.axhline(6000, color=GRAY, linewidth=0.9, linestyle=(0, (5, 2, 1, 2)))
    ax.text(2, 6180, "首末储电量 6000 kWh", ha="left", va="bottom",
            fontsize=7.2, color=GRAY)
    ax.set_ylabel("储电量 SOC /\nkWh")
    ax.set_ylim(0, 12200)
    ax.set_yticks([1200, 6000, 10800])
    ax.set_title("d　储电量状态与上下限", loc="left", pad=4)
    time_ticks(ax); style(ax)

    # ---- e 4 小时分段费用增减
    ax = fig.add_subplot(gs[4]); axes.append(ax)
    deltas, centers = [], []
    for i in range(6):
        sl = slice(i * 24, (i + 1) * 24)
        deltas.append(float(price[sl] @ grid_main[sl] - price[sl] @ grid_base[sl]))
        centers.append(i * 24 + 12)
    deltas = np.asarray(deltas)
    colors = [ORANGE if d > 0 else BLUE for d in deltas]
    ax.bar(centers, deltas, width=19, color=colors, linewidth=0)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_ylabel("分段费用增减 /\n元（储能 − 无储能）")
    ax.set_xlabel("自然日时间（区间起点）")
    ax.set_ylim(-9200, 4300)
    ax.set_title("e　储能相对无储能的 4 小时分段费用增减", loc="left", pad=4)
    time_ticks(ax); style(ax)
    for cx, d in zip(centers, deltas):
        ax.text(cx, d + (280 if d > 0 else -280), f"{d:+,.0f}", ha="center",
                va="bottom" if d > 0 else "top", fontsize=7.4,
                color=ORANGE if d > 0 else BLUE)
    pos = deltas[1] + deltas[4]
    t = ax.text(0.995, 0.96,
                f"04:00—08:00 与 16:00—20:00 合计节省 {-pos:,.2f} 元，"
                f"占全天净节省 {100 * -pos / saving:.2f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.8,
                color=GRAY)
    t.set_in_layout(False)
    for label, series in (("充电量（零线上方）", None),):
        pass
    ax.legend([Patch(facecolor=ORANGE), Patch(facecolor=BLUE)],
              ["该段支出增加", "该段节省"], loc="lower left", frameon=True,
              facecolor="white", edgecolor=GRAY_LT, framealpha=0.95, ncol=2)

    for a in axes[:-1]:
        a.set_xticklabels([])

    check("问题一04:00—08:00分段节省", deltas[1], -4755.30, 0.01, " 元")
    check("问题一16:00—20:00分段节省", deltas[4], -6780.34, 0.01, " 元")
    check("问题一00:00—04:00分段增加", deltas[0], 1917.95, 0.01, " 元")
    check("问题一12:00—16:00分段增加", deltas[3], 1056.52, 0.01, " 元")
    check("问题一重点分段占净节省比", -pos / saving, 0.8925, 5e-5)
    check("问题一全天分段增减合计", deltas.sum(), main_cost - base_cost, 0.01, " 元")
    check("问题一无储能全天购电量", float(grid_base.sum()), 61789.935400, 0.01, " kWh")

    save(fig, "fig_6_1_q1_dispatch_mechanism")


# ========================================================= 图6-2 不确定性缺口风险
def fig_6_2(q2: dict) -> None:
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

    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 208 * MM), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.88])

    for k, day in enumerate(q2["days"]):
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
        time_ticks(ax, 36); style(ax)
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

    # ---- e 334 天逐日紧急购电
    ax = fig.add_subplot(gs[2, :])
    xd = np.arange(len(em))
    hi = (xd == imax) | (xd == idx0923)
    ax.bar(xd, np.where(hi, em, np.nan), width=0.9, color=ORANGE, linewidth=0)
    ax.bar(xd, np.where(hi, np.nan, em), width=0.9, color=GRAYBLUE, linewidth=0)
    ax.set_ylabel("每日紧急购电量 / kWh")
    ax.set_xlabel("评价期日期（2025-02-01 至 2025-12-31，共 334 天）")
    ax.set_title("e　334 天逐日紧急购电量分布", loc="left", pad=4)
    ticks = list(range(0, 334, 30))
    ax.set_xticks(ticks)
    ax.set_xticklabels([daily_dates[t][5:] for t in ticks])
    ax.set_xlim(-1, 334); ax.set_ylim(0, 21000)
    style(ax)
    ax.annotate(f"最大 {daily_dates[imax]}：{em[imax]:,.2f} kWh",
                xy=(imax, em[imax]), xytext=(imax + 24, em[imax] * 0.99),
                fontsize=7.8, color=ORANGE, va="top",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=0.9))
    ax.annotate(f"2025-09-23：{em[idx0923]:,.2f} kWh",
                xy=(idx0923, em[idx0923]), xytext=(idx0923 - 14, em[idx0923] + 5200),
                fontsize=7.8, color=ORANGE, ha="right",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=0.9))
    note(ax, 0.005, 0.985,
         f"评价期 334 天中有 {np.count_nonzero(em > 1e-9)} 天发生紧急购电，"
         f"全年紧急购电 {em.sum() / 1e4:,.4f} 万kWh。\n"
         f"中心预测 MAE 与日紧急购电量：Pearson r = {r_p:.3f}、"
         f"Spearman ρ = {r_s:.3f}，仅为有限统计关联。",
         ha="left", va="top", size=7.8).set_bbox(
        dict(boxstyle="square,pad=0.28", facecolor="white", edgecolor="none",
             alpha=0.88))

    for day in q2["days"]:
        check(f"{day['date']}中心预测与审计一致",
              float(np.max(np.abs(day["center"] - day["audit_forecast"]))), 0.0, 1e-8, " kWh")
        check(f"{day['date']}实际净负荷与审计一致",
              float(np.max(np.abs(day["actual"] - day["audit_actual"]))), 0.0, 1e-8, " kWh")

    save(fig, "fig_6_2_q2_uncertainty_and_shortage_risk")


# ========================================================== 图6-3 问题三阶段价值
def fig_6_3(q3: dict) -> None:
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

    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 132 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.52, 1.0])

    # ---- a 八种组合费用构成
    ax = fig.add_subplot(gs[0])
    ordered = sorted(rows, key=lambda r: r["total_cost_yuan"])
    labels = ["仅0:00" if r["policy"] == "0-only" else r["policy"] for r in ordered]
    plan = np.array([r["plan_cost_yuan"] for r in ordered]) / 1e4
    adj = np.array([r["adjust_cost_yuan"] for r in ordered]) / 1e4
    emg = np.array([r["emergency_cost_yuan"] for r in ordered]) / 1e4
    x = np.arange(len(ordered))
    ax.bar(x, plan, width=0.62, color=BLUE, linewidth=0, label="计划费用")
    ax.bar(x, adj, width=0.62, bottom=plan, color=GRAYBLUE, linewidth=0.6,
           edgecolor=GRAY, label="调整费用")
    ax.bar(x, emg, width=0.62, bottom=plan + adj, color=ORANGE, linewidth=0,
           label="紧急费用")
    base = totals["0-only"] / 1e4
    ax.axhline(base, color=GRAY, linestyle="--", dashes=(5, 2), linewidth=1.0)
    ax.text(7.72, base + 22, "仅 0:00 基线 1398.69 万元", ha="left", va="bottom",
            fontsize=7.0, color=GRAY)
    for xi, r in zip(x, ordered):
        ax.text(xi, r["total_cost_yuan"] / 1e4 + 25, f"{r['total_cost_yuan'] / 1e4:.2f}",
                ha="center", va="bottom", fontsize=6.0, color=INK, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.6)
    ax.set_ylabel("费用 / 万元")
    ax.set_xlabel("预报发布与调整时刻组合（按总费用升序）")
    ax.set_title("a　八种阶段组合的费用构成", loc="left", pad=4)
    ax.set_xlim(-0.7, 9.9); ax.set_ylim(0, 2150)
    style(ax)
    ax.legend(loc="upper right", ncol=1, frameon=True, facecolor="white",
              edgecolor=GRAY_LT, framealpha=0.95, handlelength=1.4, fontsize=7.2)
    note(ax, 0.006, 0.985, "全阶段相对仅 0:00", ha="left", va="top", size=7.6)
    t = ax.text(0.006, 0.930,
                f"节省 {saving / 1e4:.2f} 万元（{100 * saving / totals['0-only']:.2f}%）",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.2, color=ORANGE)
    t.set_in_layout(False)

    # ---- b Shapley 分摊
    ax = fig.add_subplot(gs[1])
    order = ["6:00", "12:00", "18:00"]
    vals = np.array([shap[k] for k in order]) / 1e4
    share = vals / (saving / 1e4) * 100
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals, height=0.46, color=BLUE, linewidth=0)
    for yi, v, s, k in zip(y, vals, share, order):
        ax.text(v + 0.7, yi + 0.13, f"{v:,.2f} 万元", va="center", ha="left",
                fontsize=8.0, color=INK)
        ax.text(v + 0.7, yi - 0.13, f"占 {s:.2f}%", va="center", ha="left",
                fontsize=7.2, color=GRAY)
        ctx = [m["saving_yuan"] / 1e4 for m in marg[k]]
        ax.plot(ctx, [yi - 0.30] * len(ctx), linestyle="none", marker="|",
                markersize=7, markeredgewidth=1.0, color=GRAY, zorder=5)
        ax.plot([np.mean(ctx)], [yi - 0.30], linestyle="none", marker="o",
                markersize=3.2, color=GRAY, zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("相对仅 0:00 的节省分摊 / 万元")
    ax.set_ylabel("日内预报更新与调整时刻")
    ax.set_title("b　Shapley 节省分摊", loc="left", pad=4)
    ax.set_xlim(0, 54); ax.set_ylim(-1.35, 2.62)
    style(ax, "x")
    ax.legend([Line2D([], [], color=GRAY, marker="|", linestyle="none", markersize=7)],
              ["该时刻的条件边际节省"], loc="upper right", frameon=False, fontsize=7.2)
    note(ax, 0.0, 0.02,
         "分摊对象为预报更新、合同重优化与\n储能状态演化的综合节省，不是预报\n信息的纯因果价值。",
         ha="left", va="bottom", size=7.4)

    save(fig, "fig_6_3_q3_stage_value")


# ====================================================== 图6-4 问题四两策略比较
def fig_6_4(q4: dict) -> None:
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
    ax.bar(x, emg, width=0.46, bottom=plan + adj, color=ORANGE, linewidth=0,
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
    ax.bar(x, ek, width=0.46, color=[ORANGE, BLUE], linewidth=0)
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

    save(fig, "fig_6_4_q4_aggregate_comparison")


# ================================================== 图6-5 逐日与累计节省
def fig_6_5(q4: dict) -> None:
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
    ax.bar(x, delta / 1e4, width=0.9,
           color=np.where(delta >= 0, BLUE, ORANGE), linewidth=0)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[t][5:] for t in ticks])
    ax.set_xlim(-1, 334); ax.set_ylim(-2.3, 9.0)
    ax.set_ylabel("每日费用节省 $\\Delta C_d$ / 万元")
    ax.set_title("a　逐日费用节省（$\\Delta C_d = C_{4\\text{-}2,d} - C_{4\\text{-}3,d}$）",
                 loc="left", pad=4)
    style(ax)
    fig.legend(handles=[Patch(facecolor=BLUE,
                              label=f"4-3 费用较低（{np.count_nonzero(delta >= 0)} 天）"),
                        Patch(facecolor=ORANGE,
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
                fontsize=7.8, color=BLUE, va="center",
                arrowprops=dict(arrowstyle="-|>", color=BLUE, linewidth=0.9))
    ax.annotate(f"最大负节省 {dates[ineg][5:]}：{delta[ineg]:,.2f} 元",
                xy=(ineg, delta[ineg] / 1e4), xytext=(ineg + 16, -1.55),
                fontsize=7.8, color=ORANGE, va="center",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=0.9))

    # ---- b 累计节省
    ax = fig.add_subplot(gs[1], sharex=ax)
    ax.fill_between(x, 0, cum / 1e4, color=GRAYBLUE, alpha=0.45, linewidth=0)
    ax.plot(x, cum / 1e4, color=BLUE, linewidth=1.5)
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
                fontsize=8.2, color=BLUE, ha="right",
                bbox=dict(boxstyle="square,pad=0.28", facecolor="white",
                          edgecolor="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, linewidth=0.9))

    save(fig, "fig_6_5_q4_daily_and_cumulative_savings")


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
    "fig_6_1_q1_dispatch_mechanism": (
        "图6-1　问题一代表日储能调度与外购费用转移机制",
        [
            "代表日负荷电量在 551.57—993.16 kWh/10min 之间波动，光伏电量集中于 04:40—19:20，"
            "净负荷在 −352.92—826.08 kWh/10min 之间变化。面板 a 中储能方案购电量（科研蓝）"
            "在夜间与午间高价段低于无储能基线（灰色虚线），在 04:00—08:00 与 16:00—20:00 显著高于基线。",
            "面板 b 给出分时电价；面板 c 显示充电集中在 00:00—04:00 与 12:00—16:00，"
            "放电集中在 04:00—08:00 与 16:00—20:00；"
            "面板 d 的储电量全程处于 1200—10800 kWh（橙色虚线）之间，00:00 与 24:00 均为 6000 kWh。",
            "对应到面板 e，00:00—04:00 与 12:00—16:00 的分段费用分别增加 1917.95 元与 1056.52 元（橙色），"
            "04:00—08:00 与 16:00—20:00 分别节省 4755.30 元与 6780.34 元（蓝色），"
            "两段合计贡献全天净节省的 89.25%。全天购电费用由无储能的 48052.05 元降至 35126.85 元，节省率 26.90%。",
        ],
        "代表日：附件1给定的一天，144 个 10 分钟时段。电量为 kWh（每 10 分钟），电价为元/kWh，"
        "费用为元，储电量为 kWh。",
        "附件1.xlsx；由 src/src/q1_solver.py 当前重算的完整 144 段轨迹（附件5 result1.xlsx 仅含题目指定输出，"
        "不足以重建完整轨迹，已按其计划购电量合计 59482.70 kWh 对账一致）。",
        "反映了题设代表日与当前效率参数（$\\eta_c=\\eta_d=0.9$，往返 81%）下的购电费用转移，"
        "不表示计入储能投资、老化与维护后的全生命周期收益；也不构成购电路径唯一性的结论——"
        "独立重解曾给出费用相同但逐时段购电向量不同的另一组解。",
        [DATA / "附件1.xlsx", SRC / "q1_solver.py", SRC / "efficiency.py",
         ANNEX5 / "result1.xlsx"],
    ),
    "fig_6_2_q2_uncertainty_and_shortage_risk": (
        "图6-2　问题二指定日期的净负荷情景覆盖与全年紧急购电风险分布",
        [
            "面板 a—d 给出四个指定日期的中心预测净负荷（蓝色虚线）、实际净负荷（深灰实线）、"
            "经验最小—最大包络（浅灰）与经验 10%—90% 分位带（灰蓝）。四日共 576 个自然日时段中，"
            "实际值落入经验包络 394 段（68.40%），落入 10%—90% 分位带 314 段（54.51%）；"
            "分日覆盖为 125/144、55/144、90/144、124/144 与 101/144、37/144、63/144、113/144。",
            "面板 e 给出 334 天逐日紧急购电量，橙色柱为最需注意的两日。评价期内 170 天出现紧急购电，"
            "最大日 2025-06-01 为 15465.52 kWh；题目指定日 2025-09-23 为 4471.62 kWh，"
            "而 2025-03-20、2025-06-21 与 2025-12-21 分别仅为 9.80、0.00 与 12.90 kWh，"
            "说明总体占比不高仍可能伴随个别日期的集中缺口。",
            "中心预测平均绝对误差与日紧急购电量的 Pearson 相关为 0.316、Spearman 相关为 0.074，"
            "只能说明二者存在有限的统计关联。",
        ],
        "评价期 2025-02-01 至 2025-12-31，共 334 天；面板 a—d 为四个指定日期的 144 段。"
        "净负荷与紧急购电量为 kWh（每 10 分钟）与 kWh（每日）。",
        "q2_paper_audit 的 interval_detail.csv、daily_metrics.csv、coverage.json；"
        "情景带由 src/src/q2_solver.py 的当前情景构造函数重建（未使用正态区间替代），"
        "重建后覆盖计数与 coverage.json 完全一致（576 / 394 / 314），"
        "且四日的预测与实际序列与 interval_detail.csv 逐段一致。",
        "包络与分位带为历史残差情景的经验范围，不是统计置信区间，也不代表全年覆盖率或全天无紧急购电概率；"
        "MAE 与紧急购电量的相关系数仅为描述性统计，不构成因果识别——紧急购电还受计划量、储电量状态"
        "与逐时段误差方向影响。",
        [Q2_AUDIT / "interval_detail.csv", Q2_AUDIT / "daily_metrics.csv",
         Q2_AUDIT / "coverage.json", SRC / "q2_solver.py", DATA / "附件2.xlsx"],
    ),
    "fig_6_3_q3_stage_value": (
        "图6-3　问题三八种阶段组合的费用构成与各更新时刻的 Shapley 节省分摊",
        [
            "面板 a 按总费用升序给出八种组合的费用构成。仅 0:00 决策时总费用 1398.69 万元；"
            "启用 6:00、12:00、18:00 三次更新后降至 1319.33 万元，节省 79.36 万元、降幅 5.67%。"
            "该差额由计划费用减少 82.38 万元、紧急购电费减少 84.62 万元与新增调整费用 87.65 万元共同形成；"
            "八种组合的紧急费用（橙色）随更新时刻增多而大幅压缩，是总费用下降的主要来源。",
            "面板 b 给出三个时刻的 Shapley 节省分摊：6:00 为 23.28 万元（29.33%）、12:00 为 21.97 万元（27.69%）、"
            "18:00 为 34.11 万元（42.98%），18:00 贡献最大。灰色横线为该时刻在 4 种上下文下的条件边际节省，"
            "12 项条件边际节省全部为正，最小值为 10.41 万元。",
            "若仅能保留一次日内更新，0:00 与 18:00 的组合在三个单次更新方案中费用最低（1347.05 万元）。",
        ],
        "评价期 2025-02-01 至 2025-12-31，共 334 天、48096 个自然日时段；情景数 K=30；费用为万元。",
        "q3_paper_audit 的 q3_stage_comparison.csv 与 q3_stage_comparison.json"
        "（对应交付工作簿 result3.xlsx）。",
        "结论限于当前数据、参数（含终端价值 $\\lambda=0.478$ 元/kWh、$K=30$）与实时补救规则，"
        "且不另计预报获取成本。Shapley 值分摊的是预报更新、合同重优化与储能状态演化的综合节省，"
        "不是预报信息本身的纯因果价值；单独引入各时刻的收益不可直接相加。"
        "模型为滚动两阶段 / SAA 近似，不构成严格多阶段随机最优模型。",
        [Q3_AUDIT / "q3_stage_comparison.csv", Q3_AUDIT / "q3_stage_comparison.json",
         ANNEX5 / "result3.xlsx"],
    ),
    "fig_6_4_q4_aggregate_comparison": (
        "图6-4　问题四两种策略的费用构成与紧急购电量比较",
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
    "fig_6_5_q4_daily_and_cumulative_savings": (
        "图6-5　问题四逐日费用节省与 334 天累计节省",
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
}


def write_captions() -> None:
    lines = [
        "# 论文图件图注与分析",
        "",
        f"生成脚本：`scripts/scripts/render_paper_figures.py`（版本 {SCRIPT_VERSION}）　"
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
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
    fig_6_1(q1)
    fig_6_2(q2)
    fig_6_3(q3)
    fig_6_4(q4)
    fig_6_5(q4)

    stems = [
        "fig_5_1_time_mapping_and_information_boundary",
        "fig_6_1_q1_dispatch_mechanism",
        "fig_6_2_q2_uncertainty_and_shortage_risk",
        "fig_6_3_q3_stage_value",
        "fig_6_4_q4_aggregate_comparison",
        "fig_6_5_q4_daily_and_cumulative_savings",
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
