"""
ablation/s1_offline_hmm.py
=============================
§4.2「S1・离线HMM」。

"以全样本估计的两/多状态 HMM、平滑状态序列驱动仓位。该版本隐含使用了未来
信息，不可上线，仅作'区制信息理论上界'的参照。其与 S2 的绩效落差，即量化了
前视偏差的幅度——这是本框架'严格因果'主张的实证支点。"

刻意不做 walk-forward（没有"当前时点"的概念，整段历史一次性喂给HMM估参+
Viterbi解码），因为这正是S1存在的意义：作为"完全不管前视代价"的参照上限。

除了"参数怎么估、状态怎么定"之外，其余环节（状态->目标暴露的映射方式、
仓位滞后一天执行、评估口径）都与其它级别保持一致，确保 S1 vs S2 的绩效
落差只反映"因果 vs 非因果"这一个变量。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import load_data, RESULTS_DIR, FIGURES_DIR, K_REGIMES  # noqa: E402
from engine.hmm_offline import fit_offline_hmm_positions  # noqa: E402
from engine.evaluation import compute_backtest_metrics, print_metrics  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def generate_positions(df: pd.DataFrame, k: int = K_REGIMES, seed: int = 0):
    """
    对整段历史一次性拟合离线HMM并给出平滑仓位序列。
    返回: (DataFrame['date','ref_regime','log_return','w_held'], state_stats)
    """
    w_target, state_stats = fit_offline_hmm_positions(df, k=k, seed=seed)
    out = df[["date", "ref_regime", "log_return"]].copy()
    out["w_held"] = w_target.values
    return out, state_stats


def make_diagnostic_plot(result_df: pd.DataFrame, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    seg_id = (result_df["ref_regime"] != result_df["ref_regime"].shift(1)).cumsum()
    for _, seg in result_df.groupby(seg_id):
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1],
                       color=REGIME_COLORS.get(seg["ref_regime"].iloc[0], "lightgray"), alpha=0.10, lw=0)

    w_applied = result_df["w_held"].shift(1).fillna(0.0)
    equity = np.exp(np.cumsum(w_applied * result_df["log_return"]))
    axes[0].plot(result_df["date"], equity, color="#8a4fd9", lw=1.3, label="S1离线HMM(不可执行,前视上界)")
    axes[0].set_ylabel("净值(对数轴)")
    axes[0].set_yscale("log")
    axes[0].set_title("S1：离线平滑HMM给出的仓位（用了未来信息，仅作理论上界）")
    axes[0].legend(fontsize=9)

    axes[1].plot(result_df["date"], result_df["w_held"], color="#8a4fd9", lw=1)
    axes[1].set_ylabel("HMM平滑仓位")
    axes[1].set_xlabel("日期")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    df = load_data()
    result_df, state_stats = generate_positions(df)

    print("=== S1 离线HMM===")
    print("各隐藏状态的收益统计与标定暴露：")
    for name, s in state_stats.items():
        print(f"  {name}: mu={s['mu']:.5f}, sigma={s['sigma']:.5f}, w*={s['target_exposure']:.2f}")

    metrics = compute_backtest_metrics(result_df["log_return"], result_df["w_held"],
                                        regime_labels=result_df["ref_regime"])
    print_metrics("S1-离线HMM", metrics)
    print("分区制绩效:")
    for name, s in metrics["by_regime"].items():
        print(f"  {name}: 年化收益={s['ann_return']*100:.2f}%  夏普={s['sharpe']:.2f}  n={s['n_obs']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(RESULTS_DIR / "s1_offline_hmm.csv", index=False)
    make_diagnostic_plot(result_df, FIGURES_DIR / "s1_offline_hmm.png")
