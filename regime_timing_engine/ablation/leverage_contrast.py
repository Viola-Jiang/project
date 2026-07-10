"""
ablation/leverage_contrast.py
================================
对照实验：仓位边界 clip(0,1)（不允许做空/杠杆的长仓模式）vs 允许做空/杠杆
的更宽区间。

engine/decision.py 的 calibrate_target_exposures 和 ablation/s2~s4 的
generate_positions 都支持可配置的 position_bounds=(lo, hi)，默认(0,1)是
长仓模式（数值上与早期硬编码clip(0,1)完全一致）。本脚本用 S4（终版核心）
分别跑一遍(0,1)和(-1,2)两种边界，对比夏普/回撤/换手率——两种模式都要能跑、
能对比，而不是只实现了不用的那一半。

运行方式：
  python ablation/leverage_contrast.py   (需先运行 pipeline/01, pipeline/02)
输出：
  outputs/ablation/results/leverage_contrast.csv
  outputs/ablation/figures/leverage_contrast.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import load_data, RESULTS_DIR, FIGURES_DIR  # noqa: E402
from ablation.s4_hsmm_duration import generate_positions  # noqa: E402
from engine.evaluation import compute_backtest_metrics, print_metrics  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BOUNDS_LONG_ONLY = (0.0, 1.0)
BOUNDS_LEVERAGE = (-1.0, 2.0)


def make_diagnostic_plot(long_df: pd.DataFrame, lev_df: pd.DataFrame, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    seg_id = (long_df["ref_regime"] != long_df["ref_regime"].shift(1)).cumsum()
    for _, seg in long_df.groupby(seg_id):
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1],
                       color=REGIME_COLORS.get(seg["ref_regime"].iloc[0], "lightgray"), alpha=0.10, lw=0)

    long_equity = np.exp(np.cumsum(long_df["w_held"].shift(1).fillna(0.0) * long_df["log_return"]))
    lev_equity = np.exp(np.cumsum(lev_df["w_held"].shift(1).fillna(0.0) * lev_df["log_return"]))
    axes[0].plot(long_df["date"], long_equity, color="#3f6fa8", lw=1.3, label=f"仓位边界{BOUNDS_LONG_ONLY}(长仓)")
    axes[0].plot(lev_df["date"], lev_equity, color="#d94f4f", lw=1.3, label=f"仓位边界{BOUNDS_LEVERAGE}(允许做空/杠杆)")
    axes[0].set_ylabel("净值(对数轴)")
    axes[0].set_yscale("log")
    axes[0].set_title("仓位边界对照：S4长仓 vs S4允许做空/杠杆")
    axes[0].legend(fontsize=9)

    axes[1].plot(long_df["date"], long_df["w_held"], color="#3f6fa8", lw=0.8, label="长仓仓位")
    axes[1].plot(lev_df["date"], lev_df["w_held"], color="#d94f4f", lw=0.8, alpha=0.8, label="做空/杠杆仓位")
    axes[1].axhline(0, color="black", lw=0.6, ls=":")
    axes[1].set_ylabel("仓位")
    axes[1].set_xlabel("日期")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    df = load_data()

    print(f"=== 仓位边界对照：S4 分别跑 {BOUNDS_LONG_ONLY}（长仓）与 {BOUNDS_LEVERAGE}（允许做空/杠杆） ===\n")

    long_df = generate_positions(df, position_bounds=BOUNDS_LONG_ONLY)
    long_metrics = compute_backtest_metrics(long_df["log_return"], long_df["w_held"],
                                             regime_labels=long_df["ref_regime"])
    print_metrics(f"S4-长仓{BOUNDS_LONG_ONLY}", long_metrics)

    lev_df = generate_positions(df, position_bounds=BOUNDS_LEVERAGE)
    lev_metrics = compute_backtest_metrics(lev_df["log_return"], lev_df["w_held"],
                                            regime_labels=lev_df["ref_regime"])
    print_metrics(f"S4-做空/杠杆{BOUNDS_LEVERAGE}", lev_metrics)

    print(f"\n=== 对比小结 ===")
    print(f"{'指标':<16}{'长仓(0,1)':>16}{'做空/杠杆(-1,2)':>18}")
    print(f"{'夏普':<16}{long_metrics['sharpe']:>16.2f}{lev_metrics['sharpe']:>18.2f}")
    print(f"{'最大回撤':<16}{long_metrics['max_dd']*100:>15.1f}%{lev_metrics['max_dd']*100:>17.1f}%")
    print(f"{'换手率':<16}{long_metrics['turnover_rate']*100:>15.1f}%{lev_metrics['turnover_rate']*100:>17.1f}%")

    contrast_df = pd.DataFrame([
        {"mode": "long_only", "bounds": str(BOUNDS_LONG_ONLY),
         **{k: v for k, v in long_metrics.items()
            if k not in ("equity", "strategy_return", "by_regime", "rebalance_gap_days")}},
        {"mode": "leverage", "bounds": str(BOUNDS_LEVERAGE),
         **{k: v for k, v in lev_metrics.items()
            if k not in ("equity", "strategy_return", "by_regime", "rebalance_gap_days")}},
    ])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    contrast_df.to_csv(RESULTS_DIR / "leverage_contrast.csv", index=False)
    print(f"\n结果已保存 -> {RESULTS_DIR / 'leverage_contrast.csv'}")

    make_diagnostic_plot(long_df, lev_df, FIGURES_DIR / "leverage_contrast.png")
