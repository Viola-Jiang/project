"""
pipeline/01_data_simulation.py
================================
对应方法论文档 §2「数据与特征构建」的数据来源说明，以及 §6.5「方法论流程
验证」中"在合成的牛/震荡/熊三区制数据上运行终版引擎"的数据生成部分。

目的：
  在接入真实中证800数据之前，先构造一份"已知真值"的合成数据，
  用于验证后续 BOCPD / HSMM 引擎的工程链路是否正确。

设计要点：
  1. 三区制：牛市(bull) / 震荡(sideways) / 危机(bear)，每个区制有各自的
     收益均值 mu、波动 sigma。
  2. 关键：区制久期不服从几何分布，而是服从负二项分布(Negative-Binomial)，
     使得 hazard 函数随"段龄"变化（老区制更容易结束/或更稳定），这正是
     HSMM 相对 HMM 的核心差异化优势所要刻画的对象（文档 §3.4）。
     若久期用几何分布模拟，则 HSMM 相对 HMM 将不会有任何边际收益。
  3. 收益使用 Student-t 分布抽样（自由度较低），保留厚尾特征（文档 §3.3）。
  4. 区制间转移：设定非对称的转移矩阵（危机后更倾向进入震荡而非直接回牛市）。

运行方式：
  python pipeline/01_data_simulation.py
输出：
  data/synthetic_prices.csv
  outputs/figures/01_data_simulation.png

注：核心生成逻辑（区制参数、久期分布、转移矩阵、simulate()）已抽到
engine/synthetic_data.py，因为 ablation/s5_multi_asset_robustness.py 需要
用不同随机种子复用同一套生成逻辑（作为"多指数并行"的沙盒代理）。本脚本
只负责：调用一次（种子42）、存主线数据、画诊断图。
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
sys.path.insert(0, str(REPO_ROOT))

from engine.synthetic_data import simulate, DEFAULT_SEED  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SEED = DEFAULT_SEED


def make_diagnostic_plot(df: pd.DataFrame, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1, 1]})
    seg_id = (df["regime_age_true"] == 1).cumsum()
    for _, seg in df.groupby(seg_id):
        for ax in axes[:2]:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1],
                       color=REGIME_COLORS[seg["regime"].iloc[0]], alpha=0.15, lw=0)

    axes[0].plot(df["date"], df["price"], color="black", lw=1)
    axes[0].set_ylabel("指数点位")
    axes[0].set_title("合成区制数据：价格路径与真实区制标注（红=牛市 黄=震荡 蓝=危机）")

    axes[1].plot(df["date"], df["log_return"], color="gray", lw=0.5)
    axes[1].set_ylabel("日对数收益 r_t")

    roll_vol = df["log_return"].rolling(20).std()
    axes[2].plot(df["date"], roll_vol, color="#8a4fd9", lw=1)
    axes[2].set_ylabel("20日滚动波动 σ_t")
    axes[2].set_xlabel("日期")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    df = simulate()

    DATA_DIR.mkdir(exist_ok=True, parents=True)
    out_path = DATA_DIR / "synthetic_prices.csv"
    df.to_csv(out_path, index=False)
    print(f"生成 {len(df)} 个交易日的合成数据 -> {out_path}")

    print("\n区制分布（交易日占比）：")
    print(df["regime"].value_counts(normalize=True).round(3))

    print("\n各区制真实段数（变点次数）：")
    n_segments = (df["regime_age_true"] == 1).sum()
    print(f"  总变点数（含起点）: {n_segments}")
    print(df[df["regime_age_true"] == 1]["regime"].value_counts())

    segment_id = (df["regime_age_true"] == 1).cumsum()
    seg_durations = df.groupby(segment_id).agg(
        regime=("regime", "first"), duration=("regime_age_true", "max"))
    print("\n各区制真实平均久期（按段计算，交易日）：")
    print(seg_durations.groupby("regime")["duration"].agg(["mean", "std", "count"]).round(1))

    print("\n各区制收益统计：")
    print(df.groupby("regime")["log_return"].agg(["mean", "std", "count"]).round(5))

    make_diagnostic_plot(df, FIGURES_DIR / "01_data_simulation.png")
