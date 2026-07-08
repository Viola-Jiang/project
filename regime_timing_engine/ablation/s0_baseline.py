"""
ablation/s0_baseline.py
=========================
对应方法论文档 §4.2「S0・基线（Baseline）」："设定恒定满仓或经典均线择时为
基准。意义在于设定'及格线'——后续任一增项须超越之，方证其价值。"

文档给了两个可选基线，本模块两个都实现，但六级消融对比表里只取
**恒定满仓**作为 S0 代表（mode="constant"）：恒定满仓是零参数、零模型假设
的最严格及格线；均线择时（mode="ma"）作为次要参照一并跑出来、打印结果，
但不计入六级主表，避免"S0本身是不是已经用了一种择时模型"这种歧义。

两种基线都不需要任何估参、不需要 walk-forward（没有可拟合的参数），
自然也没有前视问题。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import load_data, RESULTS_DIR, FIGURES_DIR  # noqa: E402
from engine.evaluation import compute_backtest_metrics, print_metrics  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def generate_positions(df: pd.DataFrame, mode: str = "constant", ma_window: int = 200) -> pd.DataFrame:
    """
    mode="constant": 恒定满仓，w_held 全程为 1.0
    mode="ma"：经典均线择时，价格高于 ma_window 日均线时满仓，否则空仓
               （只用截至当天的价格，天然因果，不存在前视）
    返回: DataFrame['date','true_regime','log_return','w_held']
    """
    out = df[["date", "regime", "log_return"]].rename(columns={"regime": "true_regime"}).copy()

    if mode == "constant":
        out["w_held"] = 1.0
    elif mode == "ma":
        ma = df["price"].rolling(ma_window, min_periods=ma_window).mean()
        out["w_held"] = np.where(df["price"] > ma, 1.0, 0.0)
        out["w_held"] = out["w_held"].where(~ma.isna(), 0.0)  # 均线未成型前空仓
    else:
        raise ValueError(f"未知mode: {mode}")

    return out


def make_diagnostic_plot(const_df, ma_df, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    seg_id = (const_df["true_regime"] != const_df["true_regime"].shift(1)).cumsum()
    for _, seg in const_df.groupby(seg_id):
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1],
                       color=REGIME_COLORS.get(seg["true_regime"].iloc[0], "lightgray"), alpha=0.10, lw=0)

    const_equity = np.exp(np.cumsum(const_df["w_held"].shift(1).fillna(0.0) * const_df["log_return"]))
    ma_equity = np.exp(np.cumsum(ma_df["w_held"].shift(1).fillna(0.0) * ma_df["log_return"]))
    axes[0].plot(const_df["date"], const_equity, color="gray", lw=1.2, label="S0恒定满仓")
    axes[0].plot(ma_df["date"], ma_equity, color="#3f6fa8", lw=1.2, label="S0均线择时(参照,不计入主表)")
    axes[0].set_ylabel("净值(对数轴)")
    axes[0].set_yscale("log")
    axes[0].set_title("S0基线：恒定满仓 vs 均线择时")
    axes[0].legend(fontsize=9)

    axes[1].plot(ma_df["date"], ma_df["w_held"], color="#3f6fa8", lw=1)
    axes[1].set_ylabel("均线择时仓位")
    axes[1].set_xlabel("日期")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    df = load_data()

    const_df = generate_positions(df, mode="constant")
    ma_df = generate_positions(df, mode="ma")

    print("=== S0 基线 ===")
    const_metrics = compute_backtest_metrics(const_df["log_return"], const_df["w_held"],
                                              regime_labels=const_df["true_regime"])
    print_metrics("S0-恒定满仓(计入主表)", const_metrics)

    ma_metrics = compute_backtest_metrics(ma_df["log_return"], ma_df["w_held"],
                                           regime_labels=ma_df["true_regime"])
    print_metrics("S0-均线择时(仅参照)", ma_metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    const_df.to_csv(RESULTS_DIR / "s0_baseline_constant.csv", index=False)
    ma_df.to_csv(RESULTS_DIR / "s0_baseline_ma.csv", index=False)
    make_diagnostic_plot(const_df, ma_df, FIGURES_DIR / "s0_baseline.png")
