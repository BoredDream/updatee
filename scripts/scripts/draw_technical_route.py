from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as MplPath


# -----------------------------
# Global style and color palette
# -----------------------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COLOR_LAYER = "#F5F7FA"
COLOR_MAIN_FILL = "#EAF2FB"
COLOR_MAIN_EDGE = "#3B6EA5"
COLOR_PUBLIC_FILL = "#DCE9F7"
COLOR_PUBLIC_EDGE = "#1F4E79"
COLOR_CHECK_FILL = "#EAF7EC"
COLOR_CHECK_EDGE = "#3F7A47"
COLOR_PENDING_FILL = "#F2F2F2"
COLOR_PENDING_EDGE = "#999999"
COLOR_ARROW = "#5A5A5A"
COLOR_INHERIT = "#1F4E79"
COLOR_FEEDBACK = "#B04A4A"


def add_layer(ax, x, y, w, h, title):
    """Add a rounded layer background and a vertical layer label."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.3,rounding_size=1.5",
        facecolor=COLOR_LAYER,
        edgecolor="#CDD3DA",
        linewidth=1.2,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        3.4,
        y,
        title,
        ha="center",
        va="center",
        rotation=90,
        fontsize=12,
        fontweight="bold",
        color="#333333",
        zorder=3,
    )


def add_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    facecolor=COLOR_MAIN_FILL,
    edgecolor=COLOR_MAIN_EDGE,
    linewidth=1.5,
    linestyle="-",
    fontsize=9.5,
    rounding=0.8,
    fontweight="normal",
):
    """Add an explicitly positioned rounded rectangle node."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0.3,rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
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
        fontweight=fontweight,
        color="#222222",
        linespacing=1.28,
        zorder=3,
    )


def add_parallelogram(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    facecolor=COLOR_MAIN_FILL,
    edgecolor=COLOR_MAIN_EDGE,
    linewidth=1.3,
    fontsize=9.0,
):
    """Add an explicitly positioned parallelogram data node."""
    skew = min(1.5, w * 0.12)
    points = [
        (x - w / 2 + skew, y - h / 2),
        (x + w / 2, y - h / 2),
        (x + w / 2 - skew, y + h / 2),
        (x - w / 2, y + h / 2),
    ]
    patch = Polygon(
        points,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
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
        color="#222222",
        linespacing=1.22,
        zorder=3,
    )


def add_diamond(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    facecolor=COLOR_MAIN_FILL,
    edgecolor=COLOR_MAIN_EDGE,
    linewidth=1.7,
    fontsize=9.0,
):
    """Add an explicitly positioned comparison diamond."""
    points = [
        (x, y + h / 2),
        (x + w / 2, y),
        (x, y - h / 2),
        (x - w / 2, y),
    ]
    patch = Polygon(
        points,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
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
        color="#222222",
        linespacing=1.18,
        zorder=3,
    )


def add_arrow(
    ax,
    start,
    end,
    *,
    color=COLOR_ARROW,
    linewidth=1.35,
    linestyle="-",
    connectionstyle="arc3,rad=0",
    label=None,
    label_xy=None,
    fontsize=9.0,
    mutation_scale=14,
):
    """Add a FancyArrowPatch between two explicit points."""
    arrow = FancyArrowPatch(
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
        zorder=1,
    )
    ax.add_patch(arrow)
    if label and label_xy:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.8),
            zorder=3,
        )


def add_poly_arrow(
    ax,
    points,
    *,
    color=COLOR_ARROW,
    linewidth=1.25,
    linestyle="-",
    label=None,
    label_xy=None,
    fontsize=9.0,
    mutation_scale=13,
):
    """Add a manually routed polyline arrow through explicit vertices."""
    vertices = list(points)
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 1)
    path = MplPath(vertices, codes)
    arrow = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    ax.add_patch(arrow)
    if label and label_xy:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.8),
            zorder=3,
        )


def main():
    fig, ax = plt.subplots(figsize=(14, 20))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 150)
    ax.axis("off")
    plt.subplots_adjust(left=0.035, right=0.88, top=0.99, bottom=0.045)

    # -----------------------------
    # Layer backgrounds (zorder = 0)
    # -----------------------------
    add_layer(ax, 50, 139.75, 98, 18.5, "问题分析层")
    add_layer(ax, 50, 116.50, 98, 26.0, "数据预处理层")
    add_layer(ax, 50, 77.75, 98, 48.5, "模型建立层")
    add_layer(ax, 50, 36.25, 98, 33.5, "模型求解层")
    add_layer(ax, 50, 10.00, 98, 17.0, "检验与结论层")

    # -----------------------------
    # Layer 5: problem analysis
    # -----------------------------
    add_box(
        ax,
        50,
        145,
        32,
        5,
        "研究目标与四问分解",
        linewidth=1.8,
        rounding=1.5,
        fontsize=11,
        fontweight="bold",
    )
    add_box(ax, 18, 138, 18, 6, "问题一\n确定性单日购电与储能调度", fontsize=9.2)
    add_box(ax, 40, 138, 18, 6, "问题二\n供需不确定下的日前购电计划", fontsize=9.2)
    add_box(ax, 62, 138, 18, 6, "问题三\n日内预报更新与合同滚动调整", fontsize=9.2)
    add_box(ax, 84, 138, 18, 6, "问题四\n波动电价下的两类策略扩展", fontsize=9.2)
    add_box(ax, 78, 132, 10.5, 4.2, "4-2\n每日一次计划策略", fontsize=9.0)
    add_box(ax, 90, 132, 10.5, 4.2, "4-3\n四阶段滚动调整策略", fontsize=9.0)

    add_arrow(ax, (50, 142.5), (18, 141.0), linewidth=1.15)
    add_arrow(ax, (50, 142.5), (40, 141.0), linewidth=1.15)
    add_arrow(ax, (50, 142.5), (62, 141.0), linewidth=1.15)
    add_arrow(ax, (50, 142.5), (84, 141.0), linewidth=1.15)
    add_arrow(ax, (82.0, 135.0), (78, 134.1), linewidth=1.15)
    add_arrow(ax, (86.0, 135.0), (90, 134.1), linewidth=1.15)

    # -----------------------------
    # Layer 4: attachments and data preprocessing
    # -----------------------------
    add_parallelogram(ax, 18, 127, 13, 4.5, "附件1\n→ 问题一")
    add_parallelogram(ax, 40, 127, 13, 4.5, "附件1、2\n→ 问题二")
    add_parallelogram(ax, 62, 127, 13, 4.5, "附件1、2、3\n→ 问题三")
    add_parallelogram(ax, 82, 127, 13, 4.5, "附件1—4\n→ 问题四")
    add_parallelogram(ax, 94, 127, 7, 4.5, "附件5\n输出模板", fontsize=9.0)

    # Thin upward attachment-to-question mapping arrows.
    add_arrow(ax, (18, 129.25), (18, 135.0), linewidth=0.9, mutation_scale=11)
    add_arrow(ax, (40, 129.25), (40, 135.0), linewidth=0.9, mutation_scale=11)
    add_arrow(ax, (62, 129.25), (62, 135.0), linewidth=0.9, mutation_scale=11)
    add_arrow(ax, (82, 129.25), (84, 135.0), linewidth=0.9, mutation_scale=11)

    add_box(ax, 12, 118, 12, 7, "数据读取与\n完整性检查", fontsize=9.0)
    add_box(
        ax,
        28,
        118,
        12,
        7,
        "时间轴统一\n区间起点与跨日映射",
        linewidth=2.5,
        fontsize=9.0,
        fontweight="bold",
    )
    add_box(ax, 44, 118, 12, 7, "单位与变量转换\nkW→kWh / PCHIP", fontsize=9.0)
    add_box(ax, 60, 118, 12, 7, "分阶段预报处理\n点预测", fontsize=9.0)
    add_box(
        ax,
        76,
        118,
        16,
        7,
        "因果信息截断\n与经验情景构造",
        facecolor=COLOR_PUBLIC_FILL,
        edgecolor=COLOR_PUBLIC_EDGE,
        linewidth=2.5,
        fontsize=9.0,
        fontweight="bold",
    )
    add_box(ax, 91, 118, 11, 7, "描述统计\n与数据检查", fontsize=9.0)
    add_box(ax, 68, 108, 11, 4.5, "价格预测", fontsize=9.0)
    add_box(ax, 86, 108, 14, 4.5, "统一模型输入", fontsize=9.0)

    add_arrow(ax, (18, 118), (22, 118))
    add_arrow(ax, (34, 118), (38, 118))
    add_arrow(ax, (50, 118), (54, 118))
    add_arrow(ax, (66, 118), (68, 118))
    add_arrow(ax, (84, 118), (85.5, 118))

    # Description checks feed back to the four data products.
    add_arrow(
        ax,
        (89.0, 114.5),
        (60, 121.5),
        color=COLOR_FEEDBACK,
        linestyle="--",
        linewidth=1.25,
        connectionstyle="arc3,rad=-0.25",
        label="点预测复核",
        label_xy=(72, 112.8),
        fontsize=9.0,
    )
    add_arrow(
        ax,
        (90.5, 114.5),
        (76, 121.5),
        color=COLOR_FEEDBACK,
        linestyle="--",
        linewidth=1.25,
        connectionstyle="arc3,rad=-0.18",
        label="情景构造复核",
        label_xy=(83, 113.6),
        fontsize=9.0,
    )
    add_arrow(
        ax,
        (91.5, 114.5),
        (68, 110.25),
        color=COLOR_FEEDBACK,
        linestyle="--",
        linewidth=1.25,
        connectionstyle="arc3,rad=0.18",
        label="价格预测复核",
        label_xy=(78.5, 108.6),
        fontsize=9.0,
    )
    add_arrow(
        ax,
        (93, 114.5),
        (86, 110.25),
        color=COLOR_FEEDBACK,
        linestyle="--",
        linewidth=1.25,
        connectionstyle="arc3,rad=0.12",
        label="模型输入复核",
        label_xy=(91.2, 111.3),
        fontsize=9.0,
    )

    # Unified data input descends to the common physical model.
    add_poly_arrow(
        ax,
        [(86, 105.75), (96, 105.75), (96, 100.5), (13, 100.5), (13, 90)],
        color=COLOR_INHERIT,
        linewidth=1.5,
        label="清洗后的模型输入",
        label_xy=(56, 100.5),
        fontsize=9.0,
    )

    # Attachment 5 constrains every delivered workbook through the output node.
    add_poly_arrow(
        ax,
        [(94, 124.75), (98, 124.75), (98, 42), (86, 42)],
        color=COLOR_ARROW,
        linewidth=0.9,
        label="附件5：交付格式约束",
        label_xy=(96.0, 82.0),
        fontsize=9.0,
        mutation_scale=11,
    )

    # -----------------------------
    # Layer 3: common physical model and Q1-Q4 model topology
    # -----------------------------
    add_box(
        ax,
        13,
        80,
        14,
        20,
        "供需平衡 / SOC状态递推\n储能容量与功率约束\n充放电效率 / 跨日状态衔接",
        facecolor=COLOR_PUBLIC_FILL,
        edgecolor=COLOR_PUBLIC_EDGE,
        linewidth=2.5,
        fontsize=9.0,
        rounding=0.7,
        fontweight="bold",
    )
    add_box(ax, 32, 80, 14, 8, "问题一\n确定性单日LP", fontsize=9.5)
    add_box(ax, 52, 80, 15, 8, "问题二\n经验情景随机规划", fontsize=9.3)
    add_box(ax, 72, 80, 15, 8, "问题三\n滚动两阶段近似", fontsize=9.3)
    add_box(ax, 58, 64, 18, 10, "4-2 每日一次计划\n因果价格预测\n+ 实际价格结算", fontsize=9.0)
    add_box(ax, 82, 64, 19, 10, "4-3 四阶段滚动调整\n价格日内更新\n+ 三变量配对情景", fontsize=9.0)
    add_diamond(ax, 70, 50, 18, 8, "策略比较\n4-2 与 4-3 完整策略比较", fontsize=9.0)

    # Required Q1 -> Q2 -> Q3 inheritance/extension main line.
    add_arrow(
        ax,
        (39, 80),
        (44.5, 80),
        color=COLOR_INHERIT,
        linewidth=2.2,
        label="加入供需不确定性\n与紧急补购",
        label_xy=(41.8, 84.2),
        fontsize=9.0,
    )
    add_arrow(
        ax,
        (59.5, 80),
        (64.5, 80),
        color=COLOR_INHERIT,
        linewidth=2.2,
        label="加入分阶段预报\n与合同调整",
        label_xy=(62.0, 84.2),
        fontsize=9.0,
    )

    # Required Q4 double branch: Q2 -> 4-2 and Q3 -> 4-3 only.
    add_arrow(
        ax,
        (52, 76),
        (58, 69),
        color=COLOR_INHERIT,
        linewidth=2.2,
        label="加入波动电价",
        label_xy=(56.6, 72.4),
        fontsize=9.0,
    )
    add_arrow(
        ax,
        (72, 76),
        (82, 69),
        color=COLOR_INHERIT,
        linewidth=2.2,
        label="加入波动电价",
        label_xy=(78.0, 72.4),
        fontsize=9.0,
    )
    add_arrow(ax, (58, 59), (65.3, 52.1), color=COLOR_INHERIT, linewidth=2.2)
    add_arrow(ax, (82, 59), (74.7, 52.1), color=COLOR_INHERIT, linewidth=2.2)
    ax.text(
        70,
        57.0,
        "统一实际电价结算",
        ha="center",
        va="center",
        fontsize=9.0,
        color=COLOR_INHERIT,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.8),
        zorder=3,
    )

    # The common physical model explicitly connects to all five model nodes.
    add_arrow(
        ax,
        (20, 80),
        (25, 80),
        color=COLOR_INHERIT,
        linewidth=2.2,
        label="约束复用",
        label_xy=(22.5, 77.6),
        fontsize=9.0,
    )
    add_poly_arrow(
        ax,
        [(18, 90), (18, 94), (52, 94), (52, 84)],
        color=COLOR_INHERIT,
        linewidth=2.2,
        label="模型结构继承",
        label_xy=(35, 94),
        fontsize=9.0,
    )
    add_poly_arrow(
        ax,
        [(16, 90), (16, 97.5), (72, 97.5), (72, 84)],
        color=COLOR_INHERIT,
        linewidth=2.2,
        label="模型结构继承",
        label_xy=(58, 97.5),
        fontsize=9.0,
    )
    add_poly_arrow(
        ax,
        [(20, 72), (43, 72), (43, 64), (49, 64)],
        color=COLOR_INHERIT,
        linewidth=2.2,
        label="约束复用",
        label_xy=(34, 72),
        fontsize=9.0,
    )
    add_poly_arrow(
        ax,
        [(13, 70), (13, 56.5), (82, 56.5), (82, 59)],
        color=COLOR_INHERIT,
        linewidth=2.2,
        label="约束复用",
        label_xy=(29, 56.5),
        fontsize=9.0,
    )

    # -----------------------------
    # Layer 2: solution chain and rolling loop
    # -----------------------------
    add_box(ax, 15, 42, 16, 7, "算法设计\n稀疏连续LP / SAA\n滚动优化", fontsize=9.0)
    add_box(ax, 35, 42, 16, 7, "参数设置\n公共参数\n+ 问题特定参数", fontsize=9.0)
    add_box(ax, 55, 42, 16, 7, "编程求解\nPython + SciPy linprog\n+ HiGHS", fontsize=9.0)
    add_box(
        ax,
        76,
        42,
        20,
        7,
        "结果输出\n决策 / 状态 / 统计 / 交付\nresult1—result4-3.xlsx",
        fontsize=9.0,
    )

    add_arrow(ax, (23, 42), (27, 42))
    add_arrow(ax, (43, 42), (47, 42))

    # Model-to-solver transition; the comparison diamond is a convergence node,
    # not a yes/no decision branch.
    add_poly_arrow(
        ax,
        [(61, 50), (15, 50), (15, 45.5)],
        color=COLOR_ARROW,
        linewidth=1.4,
        label="模型接口与策略配置",
        label_xy=(37, 50),
        fontsize=9.0,
    )

    # Explicit rolling-execution frame centered near (35, 28).
    rolling_frame = FancyBboxPatch(
        (6, 20.5),
        58,
        16,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor="none",
        edgecolor=COLOR_PUBLIC_EDGE,
        linewidth=2.2,
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(rolling_frame)
    ax.text(
        8,
        35.4,
        "滚动执行循环",
        ha="left",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=COLOR_PUBLIC_EDGE,
        zorder=3,
    )

    add_box(ax, 14.5, 32, 14, 5, "读取当前\n可用信息", fontsize=9.0, rounding=0.5)
    add_box(ax, 34.5, 32, 14, 5, "更新预测\n与经验情景", fontsize=9.0, rounding=0.5)
    add_box(ax, 54.5, 32, 14, 5, "求解当前\n阶段LP", fontsize=9.0, rounding=0.5)
    add_box(ax, 54.5, 24.5, 14, 5, "锁定当前\n交付区间", fontsize=9.0, rounding=0.5)
    add_box(ax, 34.5, 24.5, 14, 5, "按实际供需\n储能补救", fontsize=9.0, rounding=0.5)
    add_box(ax, 14.5, 24.5, 14, 5, "更新SOC\n与跨日合同", fontsize=9.0, rounding=0.5)

    add_arrow(ax, (21.5, 32), (27.5, 32), linewidth=1.2)
    add_arrow(ax, (41.5, 32), (47.5, 32), linewidth=1.2)
    add_arrow(ax, (54.5, 29.5), (54.5, 27.0), linewidth=1.2)
    add_arrow(ax, (47.5, 24.5), (41.5, 24.5), linewidth=1.2)
    add_arrow(ax, (27.5, 24.5), (21.5, 24.5), linewidth=1.2)
    add_arrow(
        ax,
        (10.0, 24.5),
        (10.0, 32.0),
        color=COLOR_INHERIT,
        linewidth=1.7,
        connectionstyle="arc3,rad=0.42",
    )

    # Enter the loop from implementation and export stage outputs.
    add_poly_arrow(
        ax,
        [(55, 38.5), (55, 37.5), (14.5, 37.5), (14.5, 34.5)],
        color=COLOR_ARROW,
        linewidth=1.25,
    )
    add_poly_arrow(
        ax,
        [(61.5, 32), (65, 32), (65, 38.5), (76, 38.5)],
        color=COLOR_ARROW,
        linewidth=1.25,
    )

    add_box(
        ax,
        81.5,
        28.5,
        27,
        13,
        "问题一：执行一次\n问题二 / 4-2：每日0:00一次\n问题三 / 4-3：每日最多四次",
        facecolor="#FFFFFF",
        edgecolor=COLOR_PUBLIC_EDGE,
        linewidth=1.3,
        fontsize=9.0,
        rounding=0.6,
    )

    # -----------------------------
    # Layer 1: completed checks, pending work, and conclusions
    # -----------------------------
    add_box(
        ax,
        11.5,
        13,
        11,
        7,
        "公共检验\n能量平衡 / SOC边界\n功率 / 跨日 / 对账",
        facecolor=COLOR_CHECK_FILL,
        edgecolor=COLOR_CHECK_EDGE,
        linewidth=1.7,
        fontsize=9.0,
    )
    add_box(
        ax,
        29,
        13,
        12,
        7,
        "问题一\n可行性核验 / 独立重解\n效率敏感性",
        facecolor=COLOR_CHECK_FILL,
        edgecolor=COLOR_CHECK_EDGE,
        linewidth=1.7,
        fontsize=9.0,
    )
    add_box(
        ax,
        46.5,
        13,
        14,
        7,
        "问题二\n误差与情景覆盖\n终端价值敏感性 / 全年一致性",
        facecolor=COLOR_CHECK_FILL,
        edgecolor=COLOR_CHECK_EDGE,
        linewidth=1.7,
        fontsize=9.0,
    )
    add_box(
        ax,
        64.5,
        13,
        14,
        7,
        "问题三\n阶段组合 / Shapley分摊\n费用复算 / 轨迹重放",
        facecolor=COLOR_CHECK_FILL,
        edgecolor=COLOR_CHECK_EDGE,
        linewidth=1.7,
        fontsize=9.0,
    )
    add_box(
        ax,
        78.5,
        13,
        10,
        7,
        "问题四\n结算复算 / 物理检查\n4-3轨迹重放",
        facecolor=COLOR_CHECK_FILL,
        edgecolor=COLOR_CHECK_EDGE,
        linewidth=1.7,
        fontsize=9.0,
    )
    add_box(
        ax,
        92,
        13,
        14,
        7,
        "未完成实验（可拓展）\n多年度 / 参数扫描 / 跨算法稳定\n完全已知电价 / 单因素信息价值",
        facecolor=COLOR_PENDING_FILL,
        edgecolor=COLOR_PENDING_EDGE,
        linewidth=1.7,
        linestyle="--",
        fontsize=9.0,
        rounding=0.5,
    )
    add_box(
        ax,
        50,
        4.5,
        42,
        4.5,
        "模型评价与结论推广\n削峰填谷降费 / 情景规划可执行 / 日内更新有费用优势 / 四阶段总体占优",
        facecolor=COLOR_CHECK_FILL,
        edgecolor=COLOR_CHECK_EDGE,
        linewidth=1.9,
        fontsize=9.0,
        rounding=1.5,
        fontweight="bold",
    )

    # Result-to-validation bus and five completed validation arrows.
    add_poly_arrow(
        ax,
        [(86, 42), (97.5, 42), (97.5, 18.7)],
        color=COLOR_CHECK_EDGE,
        linewidth=1.4,
        label="结果检验",
        label_xy=(96.1, 20.5),
        fontsize=9.0,
    )
    ax.plot([11.5, 92], [18.7, 18.7], color=COLOR_CHECK_EDGE, linewidth=1.2, zorder=1)
    add_arrow(ax, (11.5, 18.7), (11.5, 16.5), color=COLOR_CHECK_EDGE, linewidth=1.2)
    add_arrow(ax, (29, 18.7), (29, 16.5), color=COLOR_CHECK_EDGE, linewidth=1.2)
    add_arrow(ax, (46.5, 18.7), (46.5, 16.5), color=COLOR_CHECK_EDGE, linewidth=1.2)
    add_arrow(ax, (64.5, 18.7), (64.5, 16.5), color=COLOR_CHECK_EDGE, linewidth=1.2)
    add_arrow(ax, (78.5, 18.7), (78.5, 16.5), color=COLOR_CHECK_EDGE, linewidth=1.2)

    # Completed checks converge to the conclusion. Pending experiments stay dashed.
    add_arrow(ax, (11.5, 9.5), (32, 6.75), color=COLOR_CHECK_EDGE, linewidth=1.15)
    add_arrow(ax, (29, 9.5), (39, 6.75), color=COLOR_CHECK_EDGE, linewidth=1.15)
    add_arrow(ax, (46.5, 9.5), (47, 6.75), color=COLOR_CHECK_EDGE, linewidth=1.15)
    add_arrow(ax, (64.5, 9.5), (56, 6.75), color=COLOR_CHECK_EDGE, linewidth=1.15)
    add_arrow(ax, (78.5, 9.5), (66, 6.75), color=COLOR_CHECK_EDGE, linewidth=1.15)
    add_arrow(
        ax,
        (92, 9.5),
        (71, 6.0),
        color=COLOR_PENDING_EDGE,
        linewidth=1.0,
        linestyle="--",
        connectionstyle="arc3,rad=0.18",
        label="后续拓展",
        label_xy=(83.5, 7.6),
        fontsize=9.0,
        mutation_scale=11,
    )

    # Legend: execution, inheritance, feedback, and pending work.
    legend_handles = [
        Line2D([0], [0], color=COLOR_ARROW, linewidth=1.5, linestyle="-", label="实线：执行流程"),
        Line2D([0], [0], color=COLOR_INHERIT, linewidth=2.5, linestyle="-", label="粗实线：模型继承/扩展"),
        Line2D([0], [0], color=COLOR_FEEDBACK, linewidth=1.5, linestyle="--", label="虚线：反馈修正"),
        Line2D([0], [0], color=COLOR_PENDING_EDGE, linewidth=1.5, linestyle="--", label="灰虚线：未完成实验"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(1.005, 0.01),
        frameon=False,
        fontsize=9.0,
        borderaxespad=0,
    )

    fig.text(0.46, 0.016, "图1 技术路线图", ha="center", va="center", fontsize=14, fontweight="bold")

    output_dir = Path(__file__).resolve().parents[2] / "论文修订" / "论文修订" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "技术路线图.svg", bbox_inches="tight")
    plt.savefig(output_dir / "技术路线图.pdf", bbox_inches="tight")
    plt.savefig(output_dir / "技术路线图.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
