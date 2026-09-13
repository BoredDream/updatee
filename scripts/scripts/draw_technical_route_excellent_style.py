from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


# Four question colors, following the visual language of the reference paper.
Q1 = "#E58B42"
Q1_LIGHT = "#FFF5E9"
Q2 = "#5B9BD5"
Q2_LIGHT = "#EFF6FC"
Q3 = "#70AD47"
Q3_LIGHT = "#F1F8ED"
Q4 = "#D96B6B"
Q4_LIGHT = "#FDF0F0"
NEUTRAL = "#59636E"
NEUTRAL_LIGHT = "#F7F8FA"
TEXT = "#222222"
PENDING = "#999999"


def panel(ax, x, y, w, h, color, title):
    """Square-cornered question panel, matching compact contest-paper figures."""
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (x, y + h - 4.2),
            w,
            4.2,
            facecolor="white",
            edgecolor="none",
            zorder=0.2,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 2.1,
        title,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=TEXT,
        zorder=4,
    )
    ax.plot([x + 1.5, x + w - 1.5], [y + h - 4.2, y + h - 4.2], color=color, lw=0.9, zorder=1)


def group(ax, x, y, w, h, color, fill):
    """Pastel dashed submodule used inside each question panel."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.15,rounding_size=0.35",
            facecolor=fill,
            edgecolor=color,
            linewidth=1.0,
            linestyle=(0, (2.0, 1.6)),
            zorder=0.5,
        )
    )


def section_tab(ax, x, y, w, h, color, text):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=color, linewidth=0.8, zorder=2))
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        rotation=90,
        fontsize=8.2,
        fontweight="bold",
        color="white",
        zorder=4,
    )


def node(
    ax,
    x,
    y,
    w,
    h,
    text,
    color,
    *,
    fill="white",
    fontsize=8.6,
    linewidth=1.0,
    linestyle="-",
    bold=False,
    radius=0.28,
):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0.18,rounding_size={radius}",
        facecolor=fill,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=TEXT,
        linespacing=1.2,
        zorder=4,
    )


def diamond(ax, x, y, w, h, text, color, fill="white", fontsize=8.4):
    points = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(
        Polygon(points, closed=True, facecolor=fill, edgecolor=color, linewidth=1.2, zorder=2)
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
        linespacing=1.15,
        zorder=4,
    )


def arrow(
    ax,
    start,
    end,
    *,
    color=NEUTRAL,
    linewidth=1.05,
    linestyle="-",
    connectionstyle="arc3,rad=0",
    label=None,
    label_xy=None,
    fontsize=8.1,
    mutation_scale=11,
):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            zorder=1.2,
        )
    )
    if label and label_xy:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=0.55),
            zorder=4,
        )


def poly_arrow(
    ax,
    points,
    *,
    color=NEUTRAL,
    linewidth=1.05,
    linestyle="-",
    label=None,
    label_xy=None,
    fontsize=8.1,
    mutation_scale=11,
):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=1.2,
        )
    )
    if label and label_xy:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=0.55),
            zorder=4,
        )


def main():
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 113)
    ax.axis("off")
    plt.subplots_adjust(left=0.035, right=0.96, top=0.985, bottom=0.04)

    # ------------------------------------------------------------------
    # Shared preparation base
    # ------------------------------------------------------------------
    ax.add_patch(
        Rectangle((6, 99), 88, 12, facecolor="white", edgecolor=NEUTRAL, linewidth=1.4, zorder=0)
    )
    ax.text(
        50,
        109.1,
        "研究目标：考虑供需与电价不确定性的微网购电及储能滚动调度",
        ha="center",
        va="center",
        fontsize=11.2,
        fontweight="bold",
        color=TEXT,
        zorder=4,
    )
    ax.plot([8, 92], [107.2, 107.2], color="#B6BDC5", lw=0.8, zorder=1)

    node(ax, 13, 103.5, 12, 4.8, "附件1—4\n原始数据", NEUTRAL, fill=NEUTRAL_LIGHT, fontsize=8.4)
    node(ax, 30, 103.5, 14, 4.8, "时间轴与单位统一\n跨日映射 / kW→kWh", NEUTRAL, fill=NEUTRAL_LIGHT, fontsize=8.2)
    node(ax, 47, 103.5, 14, 4.8, "分阶段预报处理\n因果截断 / 经验情景", NEUTRAL, fill=NEUTRAL_LIGHT, fontsize=8.2)
    node(
        ax,
        65,
        103.5,
        16,
        4.8,
        "公共物理模型\n供需平衡 / SOC / 功率效率",
        NEUTRAL,
        fill="#EEF1F4",
        fontsize=8.3,
        linewidth=1.8,
        bold=True,
    )
    node(ax, 86, 103.5, 12, 4.8, "附件5\n结果模板", NEUTRAL, fill=NEUTRAL_LIGHT, fontsize=8.4)
    arrow(ax, (19, 103.5), (23, 103.5))
    arrow(ax, (37, 103.5), (40, 103.5))
    arrow(ax, (54, 103.5), (57, 103.5))

    # ------------------------------------------------------------------
    # Four question panels: 2 x 2, like the reference framework figure
    # ------------------------------------------------------------------
    panel(ax, 6, 59, 42, 38, Q1, "问题一：确定性单日购电与储能调度")
    panel(ax, 52, 59, 42, 38, Q2, "问题二：供需不确定下的日前购电计划")
    panel(ax, 6, 18, 42, 38, Q3, "问题三：日内预报更新与合同滚动调整")
    panel(ax, 52, 18, 42, 38, Q4, "问题四：波动电价下的两类策略扩展")

    # Shared physical-model bus; five explicit outgoing branches.
    poly_arrow(ax, [(65, 101.1), (50, 101.1), (50, 20)], color=NEUTRAL, linewidth=1.3)
    ax.text(
        50.7,
        89.0,
        "公共约束复用",
        rotation=90,
        ha="left",
        va="center",
        fontsize=7.9,
        color=NEUTRAL,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.4),
        zorder=4,
    )

    # ------------------------------------------------------------------
    # Q1 panel
    # ------------------------------------------------------------------
    section_tab(ax, 7.2, 82.0, 2.2, 9.0, Q1, "数据处理")
    section_tab(ax, 7.2, 68.0, 2.2, 12.0, Q1, "模型建立")
    section_tab(ax, 7.2, 60.2, 2.2, 5.8, Q1, "求解检验")
    group(ax, 10.2, 82.0, 36.2, 9.0, Q1, Q1_LIGHT)
    group(ax, 10.2, 68.0, 36.2, 12.0, Q1, Q1_LIGHT)
    group(ax, 10.2, 60.2, 36.2, 5.8, Q1, Q1_LIGHT)

    node(ax, 15.0, 86.5, 8.0, 4.3, "电价\n序列", Q1, fontsize=8.0)
    node(ax, 27.0, 86.5, 10.0, 4.3, "负荷与光伏\n10分钟序列", Q1, fontsize=8.0)
    node(ax, 40.0, 86.5, 9.0, 4.3, "单日确定性\n输入", Q1, fontsize=8.0)
    arrow(ax, (19, 86.5), (22, 86.5), color=Q1)
    arrow(ax, (32, 86.5), (35.5, 86.5), color=Q1)

    node(ax, 15.0, 76.5, 8.5, 4.0, "购电成本\n最小", Q1, fontsize=8.0)
    node(ax, 28.0, 76.5, 16.0, 5.2, "确定性单日 LP\n购电—充放电联合优化", Q1, fill="white", linewidth=1.5, bold=True)
    node(ax, 41.0, 76.5, 8.5, 4.0, "日末SOC\n闭合", Q1, fontsize=8.0)
    arrow(ax, (19.3, 76.5), (20, 76.5), color=Q1)
    arrow(ax, (36, 76.5), (36.8, 76.5), color=Q1)
    arrow(ax, (40, 84.3), (28, 79.1), color=Q1, connectionstyle="arc3,rad=0.08")

    node(ax, 15.5, 63.1, 9.5, 3.7, "HiGHS\n线性求解", Q1, fontsize=7.9)
    node(ax, 28.3, 63.1, 12.0, 3.7, "购电与储能\n调度结果", Q1, fontsize=7.9)
    node(ax, 41.0, 63.1, 9.0, 3.7, "独立重解\n效率敏感性", Q1, fontsize=7.8)
    arrow(ax, (28, 73.9), (15.5, 65.0), color=Q1)
    arrow(ax, (20.25, 63.1), (22.3, 63.1), color=Q1)
    arrow(ax, (34.3, 63.1), (36.5, 63.1), color=Q1)

    # ------------------------------------------------------------------
    # Q2 panel
    # ------------------------------------------------------------------
    section_tab(ax, 53.2, 82.0, 2.2, 9.0, Q2, "误差建模")
    section_tab(ax, 53.2, 68.0, 2.2, 12.0, Q2, "随机规划")
    section_tab(ax, 53.2, 60.2, 2.2, 5.8, Q2, "执行检验")
    group(ax, 56.2, 82.0, 36.2, 9.0, Q2, Q2_LIGHT)
    group(ax, 56.2, 68.0, 36.2, 12.0, Q2, Q2_LIGHT)
    group(ax, 56.2, 60.2, 36.2, 5.8, Q2, Q2_LIGHT)

    node(ax, 61.5, 86.5, 9.0, 4.3, "日前净负荷\n点预测", Q2, fontsize=8.0)
    node(ax, 74.0, 86.5, 10.5, 4.3, "历史预测误差\n经验分布", Q2, fontsize=8.0)
    node(ax, 87.0, 86.5, 8.5, 4.3, "供需经验\n情景集", Q2, fontsize=8.0)
    arrow(ax, (66, 86.5), (68.75, 86.5), color=Q2)
    arrow(ax, (79.25, 86.5), (82.75, 86.5), color=Q2)

    node(ax, 62.0, 76.5, 9.0, 4.0, "紧急补购\n惩罚", Q2, fontsize=8.0)
    node(ax, 74.0, 76.5, 14.0, 5.2, "经验情景随机规划\nSAA 日前购电计划", Q2, fill="white", linewidth=1.5, bold=True)
    node(ax, 87.0, 76.5, 9.0, 4.0, "终端价值\n近似", Q2, fontsize=8.0)
    arrow(ax, (66.5, 76.5), (67.0, 76.5), color=Q2)
    arrow(ax, (81.0, 76.5), (82.5, 76.5), color=Q2)
    arrow(ax, (87, 84.3), (76, 79.1), color=Q2, connectionstyle="arc3,rad=-0.08")

    node(ax, 61.5, 63.1, 9.0, 3.7, "每日0:00\n制定计划", Q2, fontsize=7.9)
    node(ax, 74.0, 63.1, 10.5, 3.7, "实际供需\n补救执行", Q2, fontsize=7.9)
    node(ax, 87.0, 63.1, 8.5, 3.7, "误差覆盖\n全年一致性", Q2, fontsize=7.8)
    arrow(ax, (74, 73.9), (61.5, 65.0), color=Q2)
    arrow(ax, (66, 63.1), (68.75, 63.1), color=Q2)
    arrow(ax, (79.25, 63.1), (82.75, 63.1), color=Q2)

    # ------------------------------------------------------------------
    # Q3 panel
    # ------------------------------------------------------------------
    section_tab(ax, 7.2, 42.0, 2.2, 9.5, Q3, "信息更新")
    section_tab(ax, 7.2, 28.0, 2.2, 12.0, Q3, "滚动求解")
    section_tab(ax, 7.2, 19.2, 2.2, 6.8, Q3, "评价归因")
    group(ax, 10.2, 42.0, 36.2, 9.5, Q3, Q3_LIGHT)
    group(ax, 10.2, 28.0, 36.2, 12.0, Q3, Q3_LIGHT)
    group(ax, 10.2, 19.2, 36.2, 6.8, Q3, Q3_LIGHT)

    node(ax, 14.0, 46.8, 6.2, 3.6, "0:00\n预报", Q3, fontsize=7.8)
    node(ax, 23.0, 46.8, 6.2, 3.6, "6:00\n更新", Q3, fontsize=7.8)
    node(ax, 32.0, 46.8, 6.2, 3.6, "12:00\n更新", Q3, fontsize=7.8)
    node(ax, 41.0, 46.8, 6.2, 3.6, "18:00\n更新", Q3, fontsize=7.8)
    arrow(ax, (17.1, 46.8), (19.9, 46.8), color=Q3)
    arrow(ax, (26.1, 46.8), (28.9, 46.8), color=Q3)
    arrow(ax, (35.1, 46.8), (37.9, 46.8), color=Q3)

    node(ax, 17.0, 36.0, 11.0, 4.4, "因果信息截断\n仅使用当前可得信息", Q3, fontsize=7.9)
    node(ax, 31.0, 36.0, 15.0, 5.2, "滚动两阶段近似\n更新合同与储能计划", Q3, fill="white", linewidth=1.5, bold=True)
    node(ax, 42.5, 36.0, 6.0, 4.4, "锁定\n交付区间", Q3, fontsize=7.8)
    arrow(ax, (22.5, 36), (23.5, 36), color=Q3)
    arrow(ax, (38.5, 36), (39.5, 36), color=Q3)
    poly_arrow(ax, [(41, 45.0), (41, 41.0), (31, 38.6)], color=Q3)

    node(ax, 16.0, 23.0, 10.0, 3.8, "八种阶段\n组合对照", Q3, fontsize=7.8)
    node(ax, 28.5, 23.0, 10.0, 3.8, "Shapley\n信息价值分摊", Q3, fontsize=7.8)
    node(ax, 41.0, 23.0, 8.5, 3.8, "费用复算\n轨迹重放", Q3, fontsize=7.7)
    arrow(ax, (31, 33.4), (16, 24.9), color=Q3)
    arrow(ax, (21, 23), (23.5, 23), color=Q3)
    arrow(ax, (33.5, 23), (36.75, 23), color=Q3)
    arrow(
        ax,
        (42.5, 33.8),
        (17.0, 38.2),
        color=Q3,
        linewidth=1.0,
        connectionstyle="arc3,rad=0.30",
        label="实际供需补救 → 更新SOC",
        label_xy=(29.0, 31.0),
        fontsize=7.5,
    )

    # ------------------------------------------------------------------
    # Q4 panel: strict dual branch and convergence
    # ------------------------------------------------------------------
    group(ax, 55.5, 43.0, 17.0, 8.0, Q4, Q4_LIGHT)
    group(ax, 55.5, 29.0, 17.0, 9.0, Q4, Q4_LIGHT)
    group(ax, 74.5, 26.0, 17.0, 25.0, Q4, Q4_LIGHT)
    node(
        ax,
        64.0,
        47.0,
        14.5,
        5.3,
        "4-2 每日一次计划\n因果价格预测 + 实际价结算",
        Q4,
        fill="white",
        linewidth=1.35,
        fontsize=8.0,
        bold=True,
    )
    node(
        ax,
        64.0,
        33.5,
        14.5,
        6.2,
        "4-3 四阶段滚动\n价格日内更新\n三变量配对情景",
        Q4,
        fill="white",
        linewidth=1.35,
        fontsize=8.0,
        bold=True,
    )
    diamond(ax, 83.0, 40.0, 13.0, 8.5, "完整策略比较\n4-2 vs 4-3", Q4, fill="white", fontsize=8.2)
    node(ax, 83.0, 31.0, 13.0, 4.4, "统一实际电价\n结算与费用复算", Q4, fontsize=8.0)
    node(ax, 66.0, 21.8, 20.0, 4.4, "已执行：结算复算 / 物理检查\n4-3轨迹重放", Q4, fontsize=7.8)
    node(
        ax,
        85.5,
        21.8,
        13.5,
        4.4,
        "未完成：多年回测\n参数扫描 / 跨算法",
        PENDING,
        fill="#F4F4F4",
        fontsize=7.6,
        linewidth=1.0,
        linestyle="--",
    )
    arrow(ax, (71.25, 47.0), (78.0, 43.0), color=Q4, linewidth=1.3)
    arrow(ax, (71.25, 33.5), (78.0, 37.0), color=Q4, linewidth=1.3)
    arrow(ax, (83.0, 35.75), (83.0, 33.2), color=Q4, linewidth=1.2, label="统一结算", label_xy=(88.0, 34.4))
    arrow(ax, (83.0, 28.8), (66.0, 24.0), color=Q4, connectionstyle="arc3,rad=0.10")
    arrow(ax, (83.0, 28.8), (85.5, 24.0), color=PENDING, linestyle="--", linewidth=0.9)

    # ------------------------------------------------------------------
    # Cross-question inheritance and expansion arrows
    # ------------------------------------------------------------------
    arrow(
        ax,
        (48, 92.3),
        (52, 92.3),
        color=Q2,
        linewidth=1.8,
        label="不确定性扩展",
        label_xy=(50, 94.1),
        fontsize=7.8,
    )
    arrow(
        ax,
        (52, 60.5),
        (48, 54.5),
        color=Q3,
        linewidth=1.8,
        label="信息更新扩展",
        label_xy=(50.0, 57.7),
        fontsize=7.8,
    )
    arrow(
        ax,
        (64, 59),
        (64, 49.65),
        color=Q4,
        linewidth=1.8,
        label="Q2 + 波动电价",
        label_xy=(70, 56.9),
        fontsize=7.8,
    )
    arrow(
        ax,
        (48, 33.5),
        (56.75, 33.5),
        color=Q4,
        linewidth=1.8,
        label="Q3 + 波动电价",
        label_xy=(52.4, 35.1),
        fontsize=7.8,
    )

    # Explicit common-model branches to Q1, Q2, Q3, 4-2 and 4-3.
    arrow(ax, (50, 76.5), (36, 76.5), color=NEUTRAL, linewidth=1.1)
    arrow(ax, (50, 76.5), (67.0, 76.5), color=NEUTRAL, linewidth=1.1)
    arrow(ax, (50, 36.0), (38.5, 36.0), color=NEUTRAL, linewidth=1.1)
    arrow(ax, (50, 47.0), (56.75, 47.0), color=NEUTRAL, linewidth=1.1)
    poly_arrow(
        ax,
        [(50, 30.0), (53.0, 30.0), (53.0, 28.0), (64.0, 28.0), (64.0, 30.4)],
        color=NEUTRAL,
        linewidth=1.1,
    )

    # ------------------------------------------------------------------
    # Unified implementation, delivery, validation and conclusions
    # ------------------------------------------------------------------
    node(
        ax,
        50,
        13.0,
        76,
        5.0,
        "统一实现与交付：Python + SciPy linprog + HiGHS   →   决策 / 状态 / 统计   →   result1—result4-3.xlsx",
        NEUTRAL,
        fill=NEUTRAL_LIGHT,
        fontsize=8.8,
        linewidth=1.2,
        radius=0.25,
    )
    node(
        ax,
        50,
        6.5,
        76,
        5.0,
        "模型评价与结论推广：削峰填谷降费 / 情景规划可执行 / 日内更新具有费用优势 / 四阶段策略总体占优",
        NEUTRAL,
        fill="white",
        fontsize=8.8,
        linewidth=1.4,
        bold=True,
        radius=0.25,
    )
    poly_arrow(
        ax,
        [(86, 101.1), (96.5, 101.1), (96.5, 13.0), (88, 13.0)],
        color=NEUTRAL,
        linewidth=0.9,
        label="附件5：交付模板",
        label_xy=(96.5, 57.0),
        fontsize=7.6,
    )
    arrow(ax, (50, 18), (50, 15.5), color=NEUTRAL, linewidth=1.2)
    arrow(ax, (50, 10.5), (50, 9.0), color=NEUTRAL, linewidth=1.2)

    legend_handles = [
        Line2D([0], [0], color=NEUTRAL, lw=1.1, label="执行流程 / 公共约束"),
        Line2D([0], [0], color=Q2, lw=1.8, label="模型继承与扩展"),
        Line2D([0], [0], color=PENDING, lw=1.0, ls="--", label="未完成实验"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.94, 0.078),
        frameon=False,
        fontsize=7.8,
        ncol=3,
        handlelength=2.2,
        columnspacing=1.6,
    )

    fig.text(0.5, 0.018, "图1  技术路线图", ha="center", va="center", fontsize=13.5)

    output_dir = Path(__file__).resolve().parents[2] / "论文修订" / "论文修订" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "技术路线图_优秀论文风格"
    plt.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
