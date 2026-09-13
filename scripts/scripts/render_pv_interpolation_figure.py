# -*- coding: utf-8 -*-
"""图 5-2　光伏预报的时间尺度转换：整点预报到 10 分钟交付序列的插值过程。

以「一个日期 + 一个发布版本」（2025-06-21 06:00 发布）为例，展示
附件3 的未来 24 小时整点预报如何转成逐 10 分钟的交付时段功率：

    1. 原始输入：附件3 在发布时刻给出的整点预报（kW），本图为 18 个当日整点；
    2. 锚点：发布时刻前最后一个已实现时段的光伏出力（取自附件2 实测），
       防止插值曲线在发布时刻附近悬空；
    3. 保形插值：PCHIP（分段三次 Hermite）在整点节点间生成 10 分钟序列，
       不产生过冲，天然保持非负与单调段；
    4. 采样：插值曲线在发布时刻及其后的每个 10 分钟交付时刻取值。

对照两条常用替代做法，说明为何选 PCHIP：
    * 线性插值——节点处出现折角，日出/日落段斜率突变；
    * 三次样条——节点间过冲，在日落段产出负功率（物理不可行）。

视觉规范沿用 render_paper_figures.py：白底、科研蓝主色、橙色唯一强调、
对照用灰、同族系列以「颜色 + 线型 + marker」多重区分，保证黑白打印可辨。

输出：论文修订/论文修订/figures/fig_5_2_pv_forecast_interpolation.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline, PchipInterpolator

SCRIPT_VERSION = "1.0.0"

ROOT = Path(r"C:\Users\admin\Desktop\update")
DATA = ROOT / "data" / "data"
FIGDIR = ROOT / "论文修订" / "论文修订" / "figures"

MM = 1.0 / 25.4
FULL_WIDTH_MM = 160.0
PNG_DPI = 500

# ------------------------------------------------------------------ 配色系统
INK = "#1A1A1A"
BLUE = "#2F5C8A"          # 科研蓝：主色（本模型采用的 PCHIP 插值）
BLUE_DK = "#1E3E60"       # 深蓝：第二层次（三次样条对照）
GRAYBLUE = "#A9BCCF"
GRAYBLUE_LT = "#DFE7EF"   # 浅灰蓝：已实现时段底纹
GRAY = "#8C8C8C"          # 灰：对照基准（线性插值 / 实测参考）
GRAY_LT = "#DCDCDC"
ORANGE = "#D9822B"        # 橙：唯一强调色（发布时刻锚点）
ORANGE_LT = "#F2DCC2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans SC", "SimHei", "Source Han Sans CN",
                        "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 8.5,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.7,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 7.6,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": GRAY_LT,
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.35,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "grid.color": GRAY_LT,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ------------------------------------------------------------------ 案例设定
CASE_DATE = "2025-06-21"
CASE_ISSUE_H = 6                    # 发布时刻（小时）
T, DT = 144, 1 / 6.0
TSTAGE = [0, 35, 71, 107, 144]
H0 = [0, 6, 12, 18]
SLOT_MIN = np.array([(i + 1) * 10 for i in range(T)])   # 每个交付时段的采样时刻(分钟)


def hm(minutes: float) -> str:
    """分钟 -> HH:MM"""
    m = int(round(minutes))
    return f"{m // 60:02d}:{m % 60:02d}"


def load_case() -> dict:
    """读取附件2 实测光伏与附件3 整点预报，重建该发布版本的插值。"""
    pv_actual = pd.read_excel(DATA / "附件2.xlsx", sheet_name="光伏发电实际功率",
                              header=0).iloc[:, 1:].to_numpy(float)          # kW, 365x144
    dates = pd.to_datetime(pd.read_excel(DATA / "附件2.xlsx", sheet_name="小区负载",
                                         header=0).iloc[:, 0])
    dstr = [d.strftime("%Y-%m-%d") for d in dates]

    fc = pd.read_excel(DATA / "附件3.xlsx", header=0)
    fc.iloc[:, 0] = fc.iloc[:, 0].ffill()
    fdate = pd.to_datetime(fc.iloc[:, 0])
    fiss = fc.iloc[:, 1].astype(str)
    fval = fc.iloc[:, 2:].to_numpy(float)                                   # kW, 1..24 小时

    table = {(fdate[i].strftime("%Y-%m-%d"), int(fiss[i].split(":")[0])): fval[i]
             for i in fc.index}

    d = dstr.index(CASE_DATE)
    m = H0.index(CASE_ISSUE_H)
    t0 = CASE_ISSUE_H * 60
    horizon = table[(CASE_DATE, CASE_ISSUE_H)]                              # 24 项

    # 锚点：发布时刻前最后一个已实现时段的光伏出力（与 q3_multistage.pv_from_forecast 一致）
    anchor = pv_actual[d - 1, T - 2] if CASE_ISSUE_H == 0 else pv_actual[d, TSTAGE[m] - 1]

    # 节点：发布时刻(锚点) + 未来 1..24 小时整点预报
    ts = np.concatenate([[t0], t0 + 60 * np.arange(1, 25)])
    vs = np.concatenate([[anchor], horizon])

    grid = np.arange(t0, 1440 + 1, 10.0)                                    # 发布后逐 10 分钟
    pch = np.maximum(PchipInterpolator(ts, vs)(grid), 0.0)
    lin = np.interp(grid, ts, vs)
    spl = CubicSpline(ts, vs)(grid)

    keep = ts <= 1440                                                       # 当日 24:00 以内的节点
    return {
        "date": CASE_DATE, "issue_h": CASE_ISSUE_H, "t0": t0,
        "anchor": anchor, "ts": ts, "vs": vs, "grid": grid,
        "pchip": pch, "linear": lin, "spline": spl,
        "node_t": ts[keep], "node_v": vs[keep],
        "actual_full": pv_actual[d], "d": d,
    }


def verify(c: dict) -> dict:
    """核对图件引用量与源数据一致，并给出需要标注的对照数字。"""
    n_node = len(c["node_t"])
    n_slot = int((SLOT_MIN >= c["t0"]).sum())
    gap_lin = float(np.max(np.abs(c["pchip"] - c["linear"])))
    i_lin = int(np.argmax(np.abs(c["pchip"] - c["linear"])))
    neg = c["spline"] < 0
    spl_min = float(c["spline"].min())
    spl_min_t = float(c["grid"][int(np.argmin(c["spline"]))])
    out = {
        "节点数(当日整点预报+锚点)": n_node,
        "交付时段数(发布后)": n_slot,
        "PCHIP 最小值/kW": round(float(c["pchip"].min()), 4),
        "线性插值 最小值/kW": round(float(c["linear"].min()), 4),
        "三次样条 最小值/kW": round(spl_min, 4),
        "三次样条 最小值时刻": hm(spl_min_t),
        "三次样条 负值点数": int(neg.sum()),
        "PCHIP 与线性 最大偏差/kW": round(gap_lin, 4),
        "该偏差时刻": hm(float(c["grid"][i_lin])),
        "锚点/kW": round(float(c["anchor"]), 4),
    }
    assert n_node == 19, f"当日整点节点数应为 19（锚点 + 18 个整点），实得 {n_node}"
    assert n_slot == T - TSTAGE[H0.index(c["issue_h"])], "交付时段数与阶段起点不一致"
    return out


def style(ax, grid_axis="y"):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
    ax.grid(True, axis=grid_axis, color=GRAY_LT, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def day_ticks(ax):
    ax.set_xlim(0, 1440)
    ax.set_xticks(np.arange(0, 1440 + 1, 120))
    ax.set_xticklabels([hm(v) for v in np.arange(0, 1440 + 1, 120)])
    # 首末刻度正好落在轴的两端，标签居中会各越出半个字宽：24:00 一直顶到
    # 画布右缘（只剩 2 px），与其余图件的 ~22 px 留白不一致。改为向内对齐，
    # 与图5-3 的处理相同。
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[-1].set_ha("right")


def draw(ax, c: dict, xlim, ylim, *, show_prefix: bool, lw_main: float = 1.5):
    t0 = c["t0"]
    grid, pchip = c["grid"], c["pchip"]

    if show_prefix:
        ax.axvspan(0, t0, color=GRAYBLUE_LT, alpha=0.55, lw=0, zorder=0)

    # 实测光伏功率（参考）：全时段细灰线
    t_act = np.arange(10, 1441, 10.0)
    ax.plot(t_act, c["actual_full"], color=GRAY_LT, lw=0.9, zorder=2)

    # 对照 1：线性插值（折角）
    ax.plot(grid, c["linear"], color=GRAY, lw=1.0, ls=(0, (4.5, 2.2)), zorder=3)

    # 对照 2：三次样条（过冲，日落段为负）
    ax.plot(grid, c["spline"], color=BLUE_DK, lw=1.0, ls=(0, (1.2, 1.6, 4.5, 1.6)),
            zorder=4)

    # 主结果：PCHIP 保形插值的 10 分钟序列
    ax.plot(grid, pchip, color=BLUE, lw=lw_main, solid_capstyle="round", zorder=6)

    # 原始整点预报节点
    ax.plot(c["node_t"][1:], c["node_v"][1:], ls="none", marker="o", ms=3.0,
            mfc="white", mec=BLUE, mew=0.9, zorder=7)

    # 发布时刻锚点
    ax.plot([t0], [c["anchor"]], ls="none", marker="D", ms=5.0,
            mfc=ORANGE, mec="white", mew=0.7, zorder=8)

    ax.axvline(t0, color=ORANGE, lw=0.9, ls="-.", zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    style(ax)


def fig_5_2(c: dict, stats: dict) -> None:
    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 140 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.30, 1.0], hspace=0.42,
                          left=0.088, right=0.978, top=0.935, bottom=0.225)

    t0 = c["t0"]
    ymax = float(max(c["node_v"].max(), c["spline"].max())) * 1.10
    ymin = -0.09 * ymax

    # ---------------- 面板 a：全天 ----------------
    axa = fig.add_subplot(gs[0])
    draw(axa, c, (0, 1440), (ymin, ymax), show_prefix=True)

    axa.set_ylabel("光伏功率 / kW")
    axa.set_xlabel("交付时间（时刻）")
    axa.set_title("a　2025-06-21 全天：06:00 发布的整点预报与 10 分钟交付序列",
                  loc="left", pad=6)
    day_ticks(axa)

    # 注记全部使用轴内相对坐标：曲线在 y/ymax>0.92 一带（除正午峰值外）为空，
    # 顶部三条注记横向错开排布，避免任何两条落在同一水平带内。
    axa.text(0.125, 0.20, "已实现时段\n不参与插值",
             transform=axa.transAxes, ha="center", va="center",
             size=7.2, color=GRAY)
    axa.text(0.275, 0.985, "发布时刻 06:00\n预报窗口 06:00—24:00",
             transform=axa.transAxes, ha="left", va="top", size=7.2, color=ORANGE)

    axa.annotate(f"附件3 整点预报节点（当日 {len(c['node_t']) - 1} 点，间隔 1 h）",
                 xy=(c["node_t"][10], c["node_v"][10]),
                 xytext=(0.985, 0.955), textcoords=axa.transAxes,
                 ha="right", va="center", size=7.4, color=BLUE,
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9,
                                 shrinkA=2, shrinkB=4, mutation_scale=9))

    axa.annotate(f"锚点 {c['anchor']:.1f} kW（实测）",
                 xy=(t0, c["anchor"]), xytext=(0.240, 0.082),
                 textcoords=axa.transAxes,
                 ha="right", va="center", size=7.4, color=ORANGE,
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.9,
                                 shrinkA=1, shrinkB=4, mutation_scale=9))

    # 放在曲线退落之后的空档（16:50 以后曲线均低于 0.42·ymax）
    axa.text(0.985, 0.450,
             f"PCHIP 与线性插值\n最大偏差 {stats['PCHIP 与线性 最大偏差/kW']:.1f} kW"
             f"（{stats['该偏差时刻']}）",
             transform=axa.transAxes, ha="right", va="center",
             size=7.2, color=GRAY)

    # ---------------- 面板 b：日落段放大 ----------------
    axb = fig.add_subplot(gs[1])
    x0, x1 = 18 * 60, 20 * 60
    win = (c["grid"] >= x0) & (c["grid"] <= x1)
    yhi = float(c["pchip"][win].max())
    ylo = float(c["spline"][win].min())
    pad = 0.16 * (yhi - ylo)
    draw(axb, c, (x0, x1), (ylo - pad, yhi + pad), show_prefix=False, lw_main=1.6)

    # 三次样条负值区（物理不可行）以浅橙底纹强调
    neg = (c["grid"] >= x0) & (c["grid"] <= x1) & (c["spline"] < 0)
    axb.fill_between(c["grid"][neg], c["spline"][neg], 0.0,
                     color=ORANGE, alpha=0.32, lw=0, zorder=2)

    # 10 分钟交付取值（插值曲线上的采样点）
    axb.plot(c["grid"][win], c["pchip"][win], ls="none", marker="o", ms=2.1,
             mfc=BLUE, mec="white", mew=0.35, zorder=7)

    axb.set_ylabel("光伏功率 / kW")
    axb.set_xlabel("交付时间（时刻）")
    axb.set_title("b　日落段放大：10 分钟采样点与两种替代插值的对照",
                  loc="left", pad=6)
    axb.set_xticks(np.arange(x0, x1 + 1, 30))
    axb.set_xticklabels([hm(v) for v in np.arange(x0, x1 + 1, 30)])
    axb.get_xticklabels()[0].set_ha("left")       # 18:00 落在左端
    axb.get_xticklabels()[-1].set_ha("right")     # 20:00 落在右端，居中会越出画布
    axb.axhline(0, color=INK, lw=0.7, zorder=3)

    # 最低点位于 19:20，注记向左展开，避免越出右边界
    imin = int(np.argmin(np.where(win, c["spline"], np.inf)))
    axb.annotate(f"三次样条最低 {c['spline'][imin]:.1f} kW（{hm(c['grid'][imin])}）",
                 xy=(c["grid"][imin], c["spline"][imin]),
                 xytext=(c["grid"][imin] - 14, ylo - pad * 0.55),
                 ha="right", va="center", size=7.4, color=ORANGE,
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.9,
                                 shrinkA=1, shrinkB=2, mutation_scale=9))

    axb.text(0.985, 0.94,
             f"当日共 {stats['三次样条 负值点数']} 个交付时段被插成负功率",
             transform=axb.transAxes, ha="right", va="top", size=7.2, color=ORANGE)

    # ---------------- 共享图例 ----------------
    handles = [
        Line2D([], [], color=BLUE, lw=1.6, label="PCHIP 保形插值（本模型采用）"),
        Line2D([], [], color=BLUE, lw=0, marker="o", ms=3.2, mfc="white",
               mec=BLUE, mew=0.9, label="附件3 整点预报节点"),
        Line2D([], [], color=BLUE, lw=0, marker="o", ms=2.6, mfc=BLUE, mec="white",
               mew=0.35, label="10 分钟交付取值"),
        Line2D([], [], color=ORANGE, lw=0, marker="D", ms=4.6,
               label="发布时刻锚点（实测）"),
        Line2D([], [], color=GRAY, lw=1.0, ls=(0, (4.5, 2.2)), label="线性插值（折角）"),
        Line2D([], [], color=BLUE_DK, lw=1.0, ls=(0, (1.2, 1.6, 4.5, 1.6)),
               label="三次样条（过冲，日落段为负）"),
        Line2D([], [], color=GRAY_LT, lw=1.2, label="附件2 实测光伏（参考）"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
               columnspacing=1.4, handlelength=2.4, borderpad=0.6,
               labelspacing=0.5, bbox_to_anchor=(0.5, 0.005))

    fig.savefig(FIGDIR / "fig_5_2_pv_forecast_interpolation.pdf")
    fig.savefig(FIGDIR / "fig_5_2_pv_forecast_interpolation.png", dpi=PNG_DPI)
    plt.close(fig)


def main() -> None:
    c = load_case()
    stats = verify(c)
    print(f"render_pv_interpolation_figure v{SCRIPT_VERSION}")
    print(f"案例：{c['date']}　{c['issue_h']:02d}:00 发布")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    fig_5_2(c, stats)
    print(f"输出：{FIGDIR / 'fig_5_2_pv_forecast_interpolation.pdf'}")
    print(f"      {FIGDIR / 'fig_5_2_pv_forecast_interpolation.png'}（{PNG_DPI} dpi）")


if __name__ == "__main__":
    main()
