"""
ablation/lookahead_contrast.py
==================================
对应方法论文档 §6.4「前视偏差对照实验」：

"将同一 HMM 分别以平滑（含未来信息）与滤波（严格因果）两种模式运行，
二者绩效之差即前视偏差幅度；BOCPD 天然为滤波形态。该对照正面坐实
'主流离线HMM择时绩效部分源于前视、而本框架因果稳健'的核心主张。"

与 run_ablation_summary.py 里"S1 vs S2"对比的区别（两者互补，不是重复）：
  S1 vs S2 ：模型类别（HMM vs BOCPD）+ 因果性（平滑 vs 滤波）**同时**变化，
             回答"换成本框架的因果BOCPD后，整体代价多大"。
  本脚本   ：**同一个** HMM、**同一组**参数、**同一套**状态->暴露标定，
             唯一变量是"状态推断时看没看未来"，回答"单纯'平滑'这个动作
             本身贡献了多少虚假绩效"——这是文档§6.4点名要做的更干净的
             对照实验，控制变量比 S1 vs S2 更严格。

运行方式：
  python ablation/lookahead_contrast.py   (需先运行 pipeline/01, pipeline/02)
输出：
  outputs/ablation/results/lookahead_contrast.csv
  outputs/ablation/figures/lookahead_contrast.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import load_data, RESULTS_DIR, FIGURES_DIR, K_REGIMES  # noqa: E402
from engine.hmm_offline import fit_hmm, decode_smoothed, decode_filtered, calibrate_state_exposures  # noqa: E402
from engine.evaluation import compute_backtest_metrics, print_metrics  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def run_contrast(df: pd.DataFrame, k: int = K_REGIMES, seed: int = 0):
    """
    拟合一个HMM，分别用Viterbi(平滑)和前向算法(滤波)解码，套用同一套
    状态->暴露标定，返回两个可直接喂给 compute_backtest_metrics 的 DataFrame。
    """
    model, feat = fit_hmm(df, k=k, seed=seed)
    smoothed_states = decode_smoothed(model, feat)
    filtered_states = decode_filtered(model, feat)

    # 状态->暴露的标定统一用"平滑"状态的收益统计——两侧共用同一套映射，
    # 确保绩效落差只反映"解码时看没看未来"，不掺杂"标定依据不同"的干扰。
    state_stats = calibrate_state_exposures(smoothed_states, df["log_return"].values, k)

    def to_positions(state_seq):
        out = df[["date", "regime", "log_return"]].rename(columns={"regime": "true_regime"}).copy()
        out["w_held"] = [state_stats[f"state_{s}"]["target_exposure"] for s in state_seq]
        return out

    smoothed_df = to_positions(smoothed_states)
    filtered_df = to_positions(filtered_states)
    return smoothed_df, filtered_df, state_stats, smoothed_states, filtered_states


def make_diagnostic_plot(smoothed_df, filtered_df, smoothed_states, filtered_states, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1, 1]})

    seg_id = (smoothed_df["true_regime"] != smoothed_df["true_regime"].shift(1)).cumsum()
    for _, seg in smoothed_df.groupby(seg_id):
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1],
                       color=REGIME_COLORS.get(seg["true_regime"].iloc[0], "lightgray"), alpha=0.10, lw=0)

    smoothed_equity = np.exp(np.cumsum(smoothed_df["w_held"].shift(1).fillna(0.0) * smoothed_df["log_return"]))
    filtered_equity = np.exp(np.cumsum(filtered_df["w_held"].shift(1).fillna(0.0) * filtered_df["log_return"]))
    axes[0].plot(smoothed_df["date"], smoothed_equity, color="#8a4fd9", lw=1.3, label="平滑(Viterbi,含未来信息)")
    axes[0].plot(filtered_df["date"], filtered_equity, color="#d94f4f", lw=1.3, label="滤波(前向算法,严格因果)")
    axes[0].set_ylabel("净值(对数轴)")
    axes[0].set_yscale("log")
    axes[0].set_title("§6.4 前视偏差对照实验：同一HMM，平滑 vs 滤波")
    axes[0].legend(fontsize=9)

    axes[1].plot(smoothed_df["date"], smoothed_states, color="#8a4fd9", lw=0.8, label="平滑状态序列")
    axes[1].set_ylabel("状态编号")
    axes[1].legend(fontsize=8)

    axes[2].plot(filtered_df["date"], filtered_states, color="#d94f4f", lw=0.8, label="滤波状态序列")
    axes[2].set_ylabel("状态编号")
    axes[2].set_xlabel("日期")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    df = load_data()
    smoothed_df, filtered_df, state_stats, smoothed_states, filtered_states = run_contrast(df)

    print("=== §6.4 前视偏差对照实验：同一HMM，平滑(Viterbi) vs 滤波(前向算法) ===\n")
    print("状态->暴露标定（两侧共用，来自平滑状态的收益统计）：")
    for name, s in state_stats.items():
        print(f"  {name}: mu={s['mu']:.5f}, sigma={s['sigma']:.5f}, w*={s['target_exposure']:.2f}")

    agree_pct = (smoothed_states == filtered_states).mean() * 100
    print(f"\n两种解码逐日状态判定一致率: {agree_pct:.1f}%（不一致的部分即'未来信息帮了忙'的天数）")

    print()
    smoothed_metrics = compute_backtest_metrics(smoothed_df["log_return"], smoothed_df["w_held"],
                                                 regime_labels=smoothed_df["true_regime"])
    print_metrics("平滑(Viterbi,含未来信息,不可执行)", smoothed_metrics)

    filtered_metrics = compute_backtest_metrics(filtered_df["log_return"], filtered_df["w_held"],
                                                 regime_labels=filtered_df["true_regime"])
    print_metrics("滤波(前向算法,严格因果,可执行)", filtered_metrics)

    sharpe_gap = smoothed_metrics["sharpe"] - filtered_metrics["sharpe"]
    print(f"\n前视偏差幅度(同一模型，纯粹因'平滑'一个动作产生) = {sharpe_gap:.2f} 个夏普点")
    print("这个数字比 run_ablation_summary.py 里的 S1 vs S2 落差更'干净'："
          "S1 vs S2 还混杂了'模型从HMM换成BOCPD'这个额外变量，本实验把模型和"
          "标定都锁死，只让'解码时看没看未来'这一件事变化。")

    contrast_df = pd.DataFrame([
        {"mode": "smoothed_viterbi", "executable": False,
         **{k: v for k, v in smoothed_metrics.items()
            if k not in ("equity", "strategy_return", "by_regime", "rebalance_gap_days")}},
        {"mode": "filtered_forward", "executable": True,
         **{k: v for k, v in filtered_metrics.items()
            if k not in ("equity", "strategy_return", "by_regime", "rebalance_gap_days")}},
    ])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    contrast_df.to_csv(RESULTS_DIR / "lookahead_contrast.csv", index=False)
    print(f"\n结果已保存 -> {RESULTS_DIR / 'lookahead_contrast.csv'}")

    make_diagnostic_plot(smoothed_df, filtered_df, smoothed_states, filtered_states,
                          FIGURES_DIR / "lookahead_contrast.png")
