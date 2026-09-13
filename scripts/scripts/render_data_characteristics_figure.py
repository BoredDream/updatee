# -*- coding: utf-8 -*-
"""图 5-3　全年数据特征的时段热力图：小区负载、光伏出力与电网电价。

三个并列子图，横轴为日内时刻（00:00—24:00，10 分钟一格），纵轴为日期
（2025-01-01 起 365 个自然日），用色深同时读出两类结构：

    * 日周期——负载的双峰、光伏的日出日落、电价的午间低谷与早晚高峰；
    * 季节变化——光伏的夏强冬弱、负载的夏季与冬季双高、电价的 U 形年走势。

时间口径与图5-1 保持一致：附件原始行的 144 项为区间起点标签，自当日 00:10
至次日 00:00，跨越 [00:10, 次日 00:10)；自然日统计窗口为 00:00—24:00 的 144
段，其首段取自前一原始行的末项 X^raw_{ν-1,143}（与 q2_solver.load_inputs 和
q4_q2_solver.load_inputs 的映射实现相同）。据此 2025-01-01 的 00:00—00:10 段
无原始行可取，图中以虚线方框标为空缺点，求解器中该段由附件1 冷启动补足。

2025-01 为预热期，图中以虚线标出评价期起点 2025-02-01。

视觉规范沿用 render_paper_figures.py：白底、低饱和分档色阶（负载与光伏用科研蓝、
电价用橙色强调）、去掉顶部与右侧边框；三个子图共用日期轴，仅最左子图标注月份。
色阶为实色分档而非连续渐变，最低档不用近白，档位边界取整便于读数。

输出：论文修订/论文修订/figures/fig_5_3_annual_data_heatmap.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

SCRIPT_VERSION = "1.0.0"

ROOT = Path(r"C:\Users\admin\Desktop\update")
DATA = ROOT / "data" / "data"
FIGDIR = ROOT / "论文修订" / "论文修订" / "figures"

MM = 1.0 / 25.4
FULL_WIDTH_MM = 160.0
FIG_HEIGHT_MM = 112.0
PNG_DPI = 500

# ------------------------------------------------------------------ 配色系统
INK = "#1A1A1A"
BLUE = "#2F5C8A"
BLUE_DK = "#1E3E60"
GRAYBLUE = "#A9BCCF"
GRAY = "#8C8C8C"
GRAY_LT = "#DCDCDC"
ORANGE = "#D9822B"
ORANGE_DK = "#A85F14"

# 分档色阶：每档为实色，最低档不用近白，避免大面积发白而读成渐变云图；
# 同为单色系、非彩虹色，色深随量值单调递增，黑白打印退化为灰阶后顺序不变。
BLUE_STEPS = ["#CBD9E6", "#A9BCCF", "#7C9DBE", "#547FA8", "#2F5C8A", "#1E3E60"]
ORANGE_STEPS = ["#F2DCC2", "#EDC79A", "#E3AC69", "#D9822B", "#B96A17", "#96520F"]

T, DT = 144, 1 / 6.0
MONTH_START = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
MONTH_LABEL = [f"{m}月" for m in range(1, 13)]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans SC", "SimHei", "Source Han Sans CN",
                        "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.7,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def hm(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60:02d}:{m % 60:02d}"


def to_natural_day(raw: np.ndarray) -> np.ndarray:
    """附件原始行(365x144) -> 自然日窗口(365x144)。

    原始行第 j 项标签为区间起点 00:10+10j，覆盖 [00:10+10j, 00:20+10j)，
    即整行跨越 [当日 00:10, 次日 00:10)。自然日 00:00—24:00 的首段只能取
    前一原始行的末项；2025-01-01 无前行可取，留 NaN。
    """
    natural = np.full_like(raw, np.nan)
    natural[:, 1:] = raw[:, :143]
    natural[1:, 0] = raw[:-1, 143]
    return natural


def load_data() -> dict:
    load = pd.read_excel(DATA / "附件2.xlsx", sheet_name="小区负载",
                         header=0).iloc[:, 1:].to_numpy(float)
    pv = pd.read_excel(DATA / "附件2.xlsx", sheet_name="光伏发电实际功率",
                       header=0).iloc[:, 1:].to_numpy(float)
    price = pd.read_excel(DATA / "附件4.xlsx", header=0).iloc[:, 1:].to_numpy(float)
    return {
        "负载": to_natural_day(load),
        "光伏": to_natural_day(pv),
        "电价": to_natural_day(price),
    }


def verify(fields: dict) -> dict:
    """核对形状、缺失格与需要写进图注的结构性数字。"""
    n_day = next(iter(fields.values())).shape[0]
    n_slot = next(iter(fields.values())).shape[1]
    for name, V in fields.items():
        assert V.shape == (365, 144), f"{name} 形状应为 365x144，实得 {V.shape}"
        assert np.isnan(V).sum() == 1, f"{name} 应只有 2025-01-01 首段一个缺失格"
        assert np.isnan(V[0, 0]), f"{name} 的缺失格应为 2025-01-01 首段"

    load, pv, price = fields["负载"], fields["光伏"], fields["电价"]
    diurnal = {k: np.nanmean(v, axis=0) for k, v in fields.items()}
    days = np.arange(365)
    month_of = np.searchsorted(MONTH_START, days, side="right")

    out = {
        "天数": n_day, "日内时段数": n_slot, "缺失格数": int(np.isnan(load).sum()),
        "负载 全局/kW": (round(float(np.nanmin(load)), 1), round(float(np.nanmax(load)), 1)),
        "光伏 全局/kW": (round(float(np.nanmin(pv)), 1), round(float(np.nanmax(pv)), 1)),
        "电价 全局/元": (round(float(np.nanmin(price)), 4), round(float(np.nanmax(price)), 4)),
        "负载 日内峰值时刻": hm(10 * int(np.argmax(diurnal["负载"]))),
        "负载 日内谷值时刻": hm(10 * int(np.argmin(diurnal["负载"]))),
        "光伏 日内峰值时刻": hm(10 * int(np.argmax(diurnal["光伏"]))),
        "电价 日内峰值时刻": hm(10 * int(np.argmax(diurnal["电价"]))),
        "电价 日内谷值时刻": hm(10 * int(np.argmin(diurnal["电价"]))),
        "光伏 月均最强/最弱": (
            int(np.argmax([np.nanmean(pv[month_of == m]) for m in range(1, 13)])) + 1,
            int(np.argmin([np.nanmean(pv[month_of == m]) for m in range(1, 13)])) + 1),
        "电价 月均最高/最低": (
            int(np.argmax([np.nanmean(price[month_of == m]) for m in range(1, 13)])) + 1,
            int(np.argmin([np.nanmean(price[month_of == m]) for m in range(1, 13)])) + 1),
        "光伏 零出力占比": round(float((pv == 0).mean()), 4),
    }
    return out


# (数据键, 标题, 分档色, 档位边界, 色标是否带溢出箭头, 单位)
PANELS = [
    ("负载", "a　小区负载（实测）", BLUE_STEPS,
     [2000, 3000, 4000, 5000, 6000, 7000, 8000], "min", "kW"),
    ("光伏", "b　光伏发电功率（实测）", BLUE_STEPS,
     [0, 2000, 4000, 6000, 8000, 10000], "max", "kW"),
    ("电价", "c　外部电网电价（附件4）", ORANGE_STEPS,
     [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8], None, "元/kWh"),
]


def fig_5_3(fields: dict) -> None:
    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, FIG_HEIGHT_MM * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.030], hspace=0.30,
                          left=0.082, right=0.988, top=0.905, bottom=0.095,
                          wspace=0.14)

    axes = []
    for col, (key, title, steps, edges, extend, unit) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, col], sharey=axes[0] if axes else None)
        axes.append(ax)

        V = fields[key]
        cmap = ListedColormap(steps)
        cmap.set_under(steps[0])
        cmap.set_over(steps[-1])
        cmap.set_bad("white")
        norm = BoundaryNorm(edges, cmap.N)

        im = ax.imshow(V, aspect="auto", origin="upper", interpolation="nearest",
                       extent=(0, 1440, 365, 0), cmap=cmap, norm=norm)

        # 2025-01-01 00:00—00:10 无原始行可取，框出空缺格以免与低值混淆。
        # 方框四边必须内缩：贴边绘制会与左/上轴脊重合而看不出来。
        ax.add_patch(Rectangle((1.0, 0.10), 8.0, 0.80, fill=False, edgecolor=INK,
                               linewidth=0.8, linestyle=(0, (1.4, 1.1)), zorder=6))

        # 评价期起点：1 月为预热期
        ax.axhline(31, color="white", linewidth=0.9, linestyle=(0, (3.5, 2.5)),
                   zorder=5)

        # 首末刻度标签向内对齐，否则相邻子图的 24:00 与 00:00 会顶在一起
        ax.set_xlim(0, 1440)
        xt = np.arange(0, 1441, 360)
        ax.set_xticks(xt)
        ax.set_xticklabels([hm(v) for v in xt])
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
        ax.set_xlabel("日内时刻")

        ax.set_title(title, loc="left", pad=5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        cax = fig.add_subplot(gs[1, col])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=edges,
                          extend=extend, spacing="uniform")
        cb.ax.tick_params(labelsize=6.6, length=2.0, pad=1.5)
        cb.outline.set_linewidth(0.6)
        cb.outline.set_edgecolor(GRAY)
        cb.set_label(unit, fontsize=6.8, labelpad=1.5)

    # 日期轴只在最左子图标注；三图共用，便于横向比较同一日期
    axes[0].set_yticks(MONTH_START)
    axes[0].set_yticklabels(MONTH_LABEL)
    axes[0].set_ylabel("日期（2025 年）")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    # 两条注记同处 1 月内的一行，左注记带引线指向空缺格；同排靠左右两端对齐，
    # 中央留白，避免与 2 月 1 日的预热分界线（y=31）重叠。
    note_box = dict(boxstyle="square,pad=0.24", facecolor="white",
                    edgecolor=GRAY_LT, linewidth=0.5, alpha=0.94)
    axes[0].annotate("左上角 1 格无原始数据",
                     xy=(5, 0.5), xytext=(40, 15.5), textcoords="data",
                     ha="left", va="center", size=6.6, color=INK, zorder=7,
                     bbox=note_box,
                     arrowprops=dict(arrowstyle="-|>", color=GRAY, linewidth=0.6,
                                     mutation_scale=4.5, shrinkA=1.5, shrinkB=0.5))
    axes[0].text(1430, 15.5, "1 月为预热期", ha="right", va="center", size=6.6,
                 color=INK, zorder=7, bbox=note_box)

    fig.savefig(FIGDIR / "fig_5_3_annual_data_heatmap.pdf")
    fig.savefig(FIGDIR / "fig_5_3_annual_data_heatmap.png", dpi=PNG_DPI)
    plt.close(fig)


def main() -> None:
    fields = load_data()
    stats = verify(fields)
    print(f"render_data_characteristics_figure v{SCRIPT_VERSION}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    fig_5_3(fields)
    print(f"输出：{FIGDIR / 'fig_5_3_annual_data_heatmap.pdf'}")
    print(f"      {FIGDIR / 'fig_5_3_annual_data_heatmap.png'}（{PNG_DPI} dpi）")


if __name__ == "__main__":
    main()
