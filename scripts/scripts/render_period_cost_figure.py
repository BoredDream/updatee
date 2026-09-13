# -*- coding: utf-8 -*-
"""图6-2　分时段费用变化：储能相对无储能的 4 小时分段购电费之差。

六个 4 小时时段各一根柱，纵轴为「储能方案购电费 − 无储能方案购电费」：

    负值  储能比无储能少花钱，即该时段节省；
    正值  储能比无储能多花钱，即该时段支出增加。

符号口径与滚动的相机决策一致——无储能方案取「净负荷为正时全额购电、
有余量时弃光」的逐段对照 `max(net_t, 0)`，不承担任何储能损耗。

视觉沿用 render_paper_figures.py：白底、橙色为唯一强调色（此处压在被强调的
「节省」一侧）、灰蓝描边浅填充表示次要的「支出增加」一侧——只用深浅不够，
再加描边，保证黑白打印时两类柱仍能一眼分开。输出 500 dpi 与矢量 PDF。

数据不取自任何落盘副本，而是用本仓库 src/src/q1_solver.py 读附件1 重算，
并对账到全天购电费差额；对不上即中止。

输出：论文修订/论文修订/figures/fig_6_2_q1_period_cost_difference.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_VERSION = "1.0.0"

ROOT = Path(r"C:\Users\admin\Desktop\update")
SRC = ROOT / "src" / "src"
DATA = ROOT / "data" / "data"
FIGDIR = ROOT / "论文修订" / "论文修订" / "figures"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import q1_solver as q1                        # noqa: E402
import render_paper_figures as rpf            # noqa: E402  复用配色与轴样式

MM = 1.0 / 25.4
FULL_WIDTH_MM = 160.0
FIG_HEIGHT_MM = 88.0
PNG_DPI = 500

INK, BLUE, GRAY, ORANGE = rpf.INK, rpf.BLUE, rpf.GRAY, rpf.ORANGE
GRAYBLUE, GRAYBLUE_LT = rpf.GRAYBLUE, rpf.GRAYBLUE_LT

STEM = "fig_6_2_q1_period_cost_difference"
PERIODS = 6                 # 六个 4 小时时段
SLOTS_PER_PERIOD = 24       # 4 h ÷ 10 min


def period_label(k: int) -> str:
    return f"{k * 4:02d}:00—{(k + 1) * 4:02d}:00"


def load() -> dict:
    """重算代表日调度，按 4 小时聚合购电费之差，并与全天差额对账。"""
    cfg = q1.Config()
    inputs = q1.load_inputs(DATA, cfg)
    main = q1.solve_day(inputs, cfg, cfg.initial_soc_kwh, cyclic=True)

    price, net, grid = inputs["price"], inputs["net"], main["grid"]
    if len(price) != PERIODS * SLOTS_PER_PERIOD:
        raise ValueError(f"时段数应为 {PERIODS * SLOTS_PER_PERIOD}，实为 {len(price)}")

    baseline = np.maximum(net, 0.0)             # 无储能：余量弃光、缺额购电
    diff_slot = price * (grid - baseline)       # 储能购电费 − 无储能购电费

    per_period = np.array([diff_slot[k * SLOTS_PER_PERIOD:
                                     (k + 1) * SLOTS_PER_PERIOD].sum()
                           for k in range(PERIODS)])

    base_cost = float(price @ baseline)
    main_cost = float(main["purchase_cost_yuan"])
    net_diff = main_cost - base_cost            # 应为负（储能更省）

    # 逐段之和必须等于全天费用之差，否则聚合口径有错
    if abs(per_period.sum() - net_diff) > 1e-6:
        raise ValueError(f"分段之和 {per_period.sum():.6f} 与全天差额 "
                         f"{net_diff:.6f} 不一致")
    if net_diff >= 0:
        raise ValueError("全天差额非负，储能未节省，与论文结论不符")

    saving = -net_diff
    lead = per_period[1] + per_period[4]        # 04—08 与 16—20 两段

    return {
        "per_period": per_period,
        "slot": diff_slot,
        "increase": float(per_period[per_period > 0].sum()),
        "decrease": float(-per_period[per_period < 0].sum()),
        "net_diff": net_diff,
        "main_cost": main_cost,
        "base_cost": base_cost,
        "saving": saving,
        "saving_pct": 100.0 * saving / base_cost,
        "lead_saving": float(-lead),
        "lead_share": 100.0 * (-lead) / saving,
    }


def fig_6_2(d: dict) -> None:
    diff = d["per_period"]
    x = np.arange(PERIODS)
    save, over = diff < 0, diff > 0

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_MM * MM, FIG_HEIGHT_MM * MM),
                           layout="constrained")

    # 「支出增加」用浅填充 + 描边：与橙色不仅深浅不同，黑白下还有边线可辨
    ax.bar(x[over], diff[over], width=0.62, color=GRAYBLUE_LT,
           edgecolor=GRAYBLUE, linewidth=0.5, label="支出增加（储能 > 无储能）")
    ax.bar(x[save], diff[save], width=0.62, color=ORANGE, linewidth=0,
           label="节省（储能 < 无储能）")
    ax.axhline(0, color=INK, linewidth=0.8, zorder=5)

    # 纵轴只按数值标签占位：标签距柱端 gap，再留出文字自身的高度。
    # 不留这两项，最高的 +1918 与最低的 -6780 会被轴端裁掉。
    span = diff.max() - diff.min()
    gap, text_h = 0.055 * span, 0.085 * span
    ax.set_ylim(diff.min() - gap - text_h, diff.max() + gap + text_h)
    ax.set_xlim(-0.62, PERIODS - 1 + 0.62)

    for xi, value in zip(x, diff):
        ax.text(xi, value + (gap if value > 0 else -gap),
                f"{value:+,.0f}",
                ha="center", va="bottom" if value > 0 else "top",
                fontsize=7.8, color=INK, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels([period_label(k) for k in range(PERIODS)])
    rpf.style(ax)
    ax.set_ylabel("储能费用 − 无储能费用\n（元 / 4h）")
    ax.set_xlabel("自然日时段（每段 4 小时，含 24 个 10 分钟区间）")
    ax.set_title("各 4 小时时段购电费之差（负值节省、正值支出增加）",
                 loc="left", pad=6)

    # 图例移出轴外单排居中：轴内左上会压住 +1918 的数值标签，
    # 轴内右上则被 +1057 的柱子与合计文字占满，四个角无一可用。
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               frameon=False, handlelength=1.6, columnspacing=2.6)

    # 合计放在标题行右端，与左对齐的标题同行
    ax.text(1.0, 1.02, f"全天合计 {d['net_diff']:+,.2f} 元",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
            color=ORANGE)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{STEM}.pdf")
    fig.savefig(FIGDIR / f"{STEM}.png", dpi=PNG_DPI)
    plt.close(fig)


def caption(d: dict) -> str:
    parts = "、".join(f"{period_label(k)} {v:+,.2f}"
                      for k, v in enumerate(d["per_period"]))
    return (
        f"图6-2　分时段费用变化。纵轴为「储能方案购电费 − 无储能方案购电费」，"
        f"按 4 小时分为六段（每段 24 个 10 分钟区间）；负值表示该时段储能比无储能"
        f"少支出（节省，橙色），正值表示多支出（灰蓝描边浅填充）。六段依次为 "
        f"{parts} 元，合计 {d['net_diff']:+,.2f} 元，与储能方案全天购电费 "
        f"{d['main_cost']:,.2f} 元、无储能 {d['base_cost']:,.2f} 元之差完全一致。"
        f"节省集中在 04:00—08:00 与 16:00—20:00 两段，合计 {d['lead_saving']:,.2f} 元，"
        f"占全天净节省的 {d['lead_share']:.2f}%；其余各段合计仍多支出 "
        f"{d['increase']:,.2f} 元。无储能对照按逐段净负荷 max(净负荷, 0) 全额购电、"
        f"余量弃光计，不承担储能损耗。数据由本仓库 src/src/q1_solver.py 读附件1 重算，"
        "非引用落盘副本；本图只反映该代表日，不能直接外推至全年。"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"render_period_cost_figure v{SCRIPT_VERSION}")
    d = load()
    for k, v in enumerate(d["per_period"]):
        print(f"  {period_label(k)}  Δ = {v:+12.4f} 元")
    print(f"  六段合计 = {d['per_period'].sum():+.6f} 元")
    print(f"  全天差额 = {d['net_diff']:+.6f} 元（储能 {d['main_cost']:.6f} / "
          f"无储能 {d['base_cost']:.6f}）")
    print(f"  增加支出合计 {d['increase']:,.4f} 元；节省合计 {d['decrease']:,.4f} 元")
    print(f"  04—08 与 16—20 两段节省 {d['lead_saving']:,.4f} 元，"
          f"占净节省 {d['lead_share']:.4f}%")
    fig_6_2(d)
    print(f"输出：{FIGDIR / (STEM + '.pdf')}")
    print(f"      {FIGDIR / (STEM + '.png')}（{PNG_DPI} dpi）")
    print("\n---- 图注 ----")
    print(caption(d))


if __name__ == "__main__":
    main()
