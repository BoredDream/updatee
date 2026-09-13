# -*- coding: utf-8 -*-
"""图6-1　分时电价与储能调度结果。

三个上下排列的子图，共用 00:00—24:00 横轴（自然日 144 段，时间标签为区间起点）：

    a  分时电价阶梯线（附件1 的外生输入，与负载、光伏同表）；
    b  储能充放电柱状图，充电向上、放电向下；
    c  储电量轨迹，标注 1200 / 10800 kWh 容量边界与首末 6000 kWh 起止锚点。

时间口径与图5-1、图6-5 一致：自然日 144 段，第 t 段覆盖 [t, t+1) 个 10 分钟，
t=0 对应 00:00—00:10，t=143 对应 23:50—24:00。电价阶梯与柱状图一律用
`align="edge"` / `where="post"`，使每段严格占满 [t, t+1]，不把区间起点画成区间中心。
储电量序列有 145 项：s[0] 为 00:00 起点，s[144] 为 24:00 终点（= s[0]，周期约束）。

数据不取自任何落盘副本，而是用本仓库 src/src/q1_solver.py 读附件1 重算，
再对物理量与费用做对账（越界、能量平衡、首末储电量、功率上限），任一条不满足即中止。

视觉规范沿用 render_paper_figures.py：白底、科研蓝主色、橙色为唯一强调色、
灰色用于容量边界与对照，去掉顶部与右侧边框，弱网格，输出 500 dpi 与矢量 PDF。

输出：论文修订/论文修订/figures/fig_6_1_q1_price_and_storage_dispatch.{pdf,png}
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
FIG_HEIGHT_MM = 134.0
PNG_DPI = 500

INK, BLUE, GRAY, ORANGE = rpf.INK, rpf.BLUE, rpf.GRAY, rpf.ORANGE
GRAYBLUE_LT, GRAY_LT = rpf.GRAYBLUE_LT, rpf.GRAY_LT

STEM = "fig_6_1_q1_price_and_storage_dispatch"

SOC_MIN, SOC_MAX, SOC_TERM = 1200.0, 10800.0, 6000.0
PLOT_X0, PLOT_X1 = 0, 144


def hm(t: int) -> str:
    """时段索引 -> 区间起点时刻。t=144 为 24:00。"""
    return f"{t // 6:02d}:{(t % 6) * 10:02d}"


def load() -> dict:
    """用当前 q1_solver 重算代表日 144 段调度，并逐条对账。"""
    cfg = q1.Config()
    inputs = q1.load_inputs(DATA, cfg)
    main = q1.solve_day(inputs, cfg, cfg.initial_soc_kwh, cyclic=True)

    price, load, pv, net = (inputs["price"], inputs["load"],
                            inputs["pv"], inputs["net"])
    grid, charge = main["grid"], main["charge"]
    discharge, soc = main["discharge"], main["soc"]
    curtail = main["curtail"]

    if len(price) != cfg.intervals_per_day or len(soc) != cfg.intervals_per_day + 1:
        raise ValueError(f"时段数不符：电价 {len(price)}、储电量 {len(soc)}")

    # 1) 容量边界：储电量不得越界
    if soc.min() < SOC_MIN - 1e-6 or soc.max() > SOC_MAX + 1e-6:
        raise ValueError(f"储电量越界：[{soc.min():.4f}, {soc.max():.4f}]")

    # 2) 首末储电量：周期约束要求 s[0] = s[144] = 6000
    for tag, value in (("起点", soc[0]), ("终点", soc[-1])):
        if abs(value - SOC_TERM) > 1e-6:
            raise ValueError(f"{tag}储电量应为 {SOC_TERM}，实为 {value:.6f}")

    # 3) 递推一致：s[t+1] = s[t] + eta*c[t] - d[t]/eta
    step = soc[1:] - soc[:-1]
    expect = cfg.eta_charge * charge - discharge / cfg.eta_discharge
    if np.max(np.abs(step - expect)) > 1e-6:
        raise ValueError(f"储电量递推不成立，最大残差 {np.max(np.abs(step - expect)):.3e}")

    # 4) 能量平衡：光伏 + 购电 + 放电 = 负载 + 充电 + 弃电
    residual = pv + grid + discharge - load - charge - curtail
    if np.max(np.abs(residual)) > 1e-6:
        raise ValueError(f"能量平衡不成立，最大残差 {np.max(np.abs(residual)):.3e}")

    # 5) 功率上限：单段充/放电量不得超过 5000 kW × (1/6) h
    cap = cfg.power_limit_kw * cfg.dt_hours
    if max(charge.max(), discharge.max()) > cap + 1e-6:
        raise ValueError(f"充放电量超出功率上限 {cap:.2f} kWh/10min")

    baseline = np.maximum(net, 0.0)
    base_cost = float(price @ baseline)
    main_cost = float(main["purchase_cost_yuan"])
    if base_cost <= main_cost:
        raise ValueError("无储能费用未高于储能方案，节省为负，结论与论文不符")

    return {
        "price": price, "charge": charge, "discharge": discharge, "soc": soc,
        "soc_min_hit": float(soc.min()), "soc_max_hit": float(soc.max()),
        "curtail": float(curtail.sum()),
        "power_cap": cap,
        "charge_peak": float(charge.max()), "discharge_peak": float(discharge.max()),
        "base_cost": base_cost, "main_cost": main_cost,
        "saving": base_cost - main_cost,
        "saving_pct": 100.0 * (1.0 - main_cost / base_cost),
    }


def fig_6_1(d: dict) -> None:
    price, charge, discharge, soc = (d["price"], d["charge"],
                                     d["discharge"], d["soc"])

    # 电价与柱状图按「区间起点」作图：末段补一个右端点，否则 23:50—24:00 画不出来
    seg = np.arange(144)
    edge = np.append(seg, 144)
    price_edge = np.append(price, price[-1])
    soc_x = np.arange(145)

    fig, axes = plt.subplots(3, 1, figsize=(FULL_WIDTH_MM * MM, FIG_HEIGHT_MM * MM),
                             sharex=True, layout="constrained")
    box = dict(facecolor="white", edgecolor="none", pad=1.2)

    # ------------------------------------------------------------ a 分时电价
    ax = axes[0]
    ax.fill_between(edge, 0, price_edge, step="post", color=GRAYBLUE_LT,
                    linewidth=0, zorder=1)
    ax.step(edge, price_edge, where="post", color=BLUE, linewidth=1.2, zorder=4)
    ax.set_ylim(0, float(price.max()) * 1.30)
    rpf.style(ax)
    ax.set_title("a　分时电价（阶梯）", loc="left", pad=5)
    ax.set_ylabel("电价\n（元/kWh）")

    i_hi, i_lo = int(np.argmax(price)), int(np.argmin(price))
    # 两条注记一律朝左（ha="right"）展开：谷值点右侧 06:00—08:00 是电价的爬升段，
    # 朝右展开的白底会切断曲线；谷值点左侧是平段，上方整片留白。
    ax.annotate(f"最高 {price[i_hi]:.3f}（{hm(i_hi)}）",
                xy=(i_hi + 0.5, price[i_hi]), xytext=(-8, 8),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=7.4, color=ORANGE, zorder=6, bbox=box,
                arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.7))
    ax.annotate(f"最低 {price[i_lo]:.3f}（{hm(i_lo)}）",
                xy=(i_lo + 0.5, price[i_lo]), xytext=(-8, 8),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=7.4, color=GRAY, zorder=6, bbox=box,
                arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.7))

    # -------------------------------------------------------- b 充放电柱状图
    ax = axes[1]
    ax.bar(seg, charge, width=1.0, align="edge", color=BLUE, linewidth=0,
           label="充电")
    # 放电向下并加白斜纹：方向之外再加纹理编码，黑白打印也能与充电分开
    ax.bar(seg, -discharge, width=1.0, align="edge", color=ORANGE, hatch="////",
           edgecolor="white", linewidth=0, label="放电")
    ax.axhline(0, color=INK, linewidth=0.8, zorder=5)
    lim = max(d["charge_peak"], d["discharge_peak"]) * 1.28
    ax.set_ylim(-lim, lim)
    rpf.style(ax)
    ax.set_title("b　储能充放电动作（充电为正、放电为负）", loc="left", pad=5)
    ax.set_ylabel("充（+）/ 放（−）\n电量（kWh/10min）")
    ax.legend(loc="lower left", frameon=False, ncol=2, handlelength=1.6,
              columnspacing=1.4)

    # ------------------------------------------------------ c 储电量与边界
    ax = axes[2]
    ax.axhline(10800, color=GRAY, linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)
    ax.axhline(1200, color=GRAY, linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)
    # 首末 6000 kWh 用点划线，与两条容量边界的虚线在黑白下也能区分
    ax.axhline(SOC_TERM, color=GRAY, linestyle=(0, (1.6, 1.6, 0.8, 1.6)),
               linewidth=0.8, zorder=2)
    # 三条标注一律 va="bottom" 置于线之上：置于线之下时白底框会盖住底部轴脊。
    # 下限放在左端——左端 00:00—04:00 的储电量在 6000 kWh 以上，1200 线上方是空的；
    # 右端 21:00—24:00 曲线正从 1200 爬回 6000，容不下这条标注。
    ax.text(144, 10800, "上限 10800 kWh ", ha="right", va="bottom", fontsize=7.4,
            color=GRAY, bbox=box, zorder=6)
    ax.text(0.5, 1200, "下限 1200 kWh", ha="left", va="bottom", fontsize=7.4,
            color=GRAY, bbox=box, zorder=6)
    ax.text(144, 6600, "首末 6000 kWh ", ha="right", va="bottom", fontsize=7.4,
            color=GRAY, bbox=box, zorder=6)

    ax.plot(soc_x, soc, color=BLUE, linewidth=1.4, zorder=4)
    ax.plot([PLOT_X0, PLOT_X1], [SOC_TERM, SOC_TERM], linestyle="none",
            marker="s", markersize=3.8, color=BLUE, markeredgecolor="white",
            markeredgewidth=0.7, zorder=6)

    ax.set_ylim(600, 12000)
    ax.set_yticks([3000, 6000, 9000])
    rpf.style(ax)
    ax.set_title("c　储电量轨迹与容量边界（方块为起止锚点）", loc="left", pad=5)
    ax.set_ylabel("储电量\n（kWh）")
    rpf.time_ticks(ax, step=24)
    ax.set_xlabel("自然日时刻（区间起点，每段 10 分钟）")

    fig.suptitle(
        f"储能方案全天购电费 {d['main_cost']:,.2f} 元，较无储能降低 "
        f"{d['saving_pct']:.2f}%（省 {d['saving']:,.2f} 元）；"
        f"弃光 {d['curtail']:.2f} kWh",
        x=0.008, ha="left", fontsize=9.5, color=INK)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{STEM}.pdf")
    fig.savefig(FIGDIR / f"{STEM}.png", dpi=PNG_DPI)
    plt.close(fig)


def caption(d: dict) -> str:
    return (
        f"图6-1　分时电价与储能调度结果。三个子图共用 00:00—24:00 横轴，"
        f"每段 10 分钟，时间标签为区间起点。(a) 外部电网分时电价（附件1 电价列），"
        f"最高 {d['price'].max():.3f} 元/kWh、最低 {d['price'].min():.3f} 元/kWh；"
        f"(b) 储能充放电量，充电为正、放电为负并加斜纹；"
        f"(c) 逐段起点储电量，灰虚线为 1200 / 10800 kWh 容量边界，"
        f"点划线为首末 6000 kWh 水平，方块为起止锚点。"
        f"该日储电量在 {d['soc_min_hit']:.0f} kW·h 触下限、在 {d['soc_max_hit']:.0f} kW·h "
        f"触上限，两处边界均为紧约束；充放电峰值 {d['charge_peak']:.2f} / "
        f"{d['discharge_peak']:.2f} kWh/10min，均未超过 5000 kW 对应的 "
        f"{d['power_cap']:.2f} kWh/10min。储能方案全天购电费 {d['main_cost']:,.2f} 元，"
        f"无储能对照 {d['base_cost']:,.2f} 元，节省 {d['saving']:,.2f} 元"
        f"（{d['saving_pct']:.2f}%），弃光 {d['curtail']:.2f} kWh。"
        "数据由本仓库 src/src/q1_solver.py 读附件1 重算，非引用落盘副本；"
        "本图只反映该代表日，不能直接外推至全年。"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"render_dispatch_overview_figure v{SCRIPT_VERSION}")
    d = load()
    for key in ("price", "charge", "discharge", "soc"):
        print(f"  {key:>9s} 长度 {len(d[key])}")
    print(f"  储电量触界：下限 {d['soc_min_hit']:.4f} / 上限 {d['soc_max_hit']:.4f}")
    print(f"  充/放峰值：{d['charge_peak']:.4f} / {d['discharge_peak']:.4f} "
          f"kWh/10min（上限 {d['power_cap']:.2f}）")
    print(f"  日购电费：储能 {d['main_cost']:.6f} 元；无储能 {d['base_cost']:.6f} 元；"
          f"省 {d['saving']:.6f} 元（{d['saving_pct']:.4f}%）")
    print(f"  弃光：{d['curtail']:.6f} kWh")
    fig_6_1(d)
    print(f"输出：{FIGDIR / (STEM + '.pdf')}")
    print(f"      {FIGDIR / (STEM + '.png')}（{PNG_DPI} dpi）")
    print("\n---- 图注 ----")
    print(caption(d))


if __name__ == "__main__":
    main()
