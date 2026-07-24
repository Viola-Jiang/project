"""
feature_validation/visualize_clustering.py
==============================================
把之前诊断脚本里的标准化，从"一次性用全样本均值/标准差"改成"滚动、因果"的版本
（跟L=500天的透支轴回看窗口保持一致），然后画图看看这个二维聚类到底长什么样：
  1. 散点图：标准化后的[t_stat, extremity]二维点云，按簇上色
  2. 时间序列图：t_stat、extremity随时间变化，背景按簇上色，直观看聚类标签
     到底在时间轴上翻转得有多频繁
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.features import rolling_trend_vol_features  # noqa: E402
from engine.plotting import setup_cjk_font  # noqa: E402
from feature_validation.validate_features import (  # noqa: E402
    compute_extremity_percentile, STRENGTH_WINDOW, EXTREMITY_M, EXTREMITY_L,
)

STANDARDIZE_WINDOW = EXTREMITY_L  # 沿用透支轴的500天回看窗口，标准化也用同样长度
K = 3
CLUSTER_COLORS = ["#3f6fa8", "#e8b84b", "#d94f4f", "#5a9367", "#8a5ca8"]


def rolling_standardize(x: np.ndarray, window: int) -> np.ndarray:
    """因果滚动z-score：第t天只用[t-window+1, t]的均值/标准差，不看未来。
    样本不足的窗口期用能取到的最早有效值向前填充。"""
    s = pd.Series(x)
    min_periods = max(20, window // 5)
    roll_mean = s.rolling(window, min_periods=min_periods).mean().bfill()
    roll_std = s.rolling(window, min_periods=min_periods).std()
    roll_std = roll_std.replace(0, np.nan).bfill().fillna(1.0)
    return ((s - roll_mean) / roll_std).to_numpy()


def main():
    from ablation.common import load_data  # noqa: E402
    df = load_data().reset_index(drop=True)
    log_return = df["log_return"].fillna(0.0).to_numpy()

    trend, vol = rolling_trend_vol_features(log_return, window=STRENGTH_WINDOW)
    t_stat = trend / (vol * np.sqrt(STRENGTH_WINDOW))
    extremity = compute_extremity_percentile(log_return, m=EXTREMITY_M, lookback=EXTREMITY_L)

    t_stat_std = rolling_standardize(t_stat, STANDARDIZE_WINDOW)
    extremity_std = rolling_standardize(extremity, STANDARDIZE_WINDOW)

    X_std = np.column_stack([t_stat_std, extremity_std])
    labels = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(X_std)

    print(f"各簇大小: {pd.Series(labels).value_counts().sort_index().to_dict()}")

    setup_cjk_font()

    # --- 图1：散点图 ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for k in range(K):
        mask = labels == k
        ax.scatter(X_std[mask, 0], X_std[mask, 1], s=6, alpha=0.5,
                   color=CLUSTER_COLORS[k], label=f"簇{k} (n={mask.sum()})")
    ax.set_xlabel("t_stat (滚动标准化后)")
    ax.set_ylabel("extremity (滚动标准化后)")
    ax.set_title("t_stat × extremity 二维点云，按KMeans簇上色")
    ax.legend()
    fig.tight_layout()
    out1 = REPO_ROOT / "outputs" / "feature_validation" / "cluster_scatter.png"
    out1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out1, dpi=130)
    plt.close(fig)
    print(f"散点图已保存 -> {out1}")

    # --- 图2：时间序列图，背景按簇上色 ---
    plot_df = pd.DataFrame({"date": df["date"], "t_stat": t_stat_std,
                             "extremity": extremity_std, "cluster": labels})
    seg_id = (plot_df["cluster"] != plot_df["cluster"].shift(1)).cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    for _, seg in plot_df.groupby(seg_id):
        color = CLUSTER_COLORS[seg["cluster"].iloc[0]]
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1], color=color, alpha=0.25, lw=0)
    axes[0].plot(plot_df["date"], plot_df["t_stat"], color="black", lw=0.6)
    axes[0].set_ylabel("t_stat(滚动标准化)")
    axes[0].set_title("t_stat / extremity 随时间变化，背景色=当天所属簇")
    axes[1].plot(plot_df["date"], plot_df["extremity"], color="black", lw=0.6)
    axes[1].set_ylabel("extremity(滚动标准化)")
    axes[1].set_xlabel("日期")
    fig.tight_layout()
    out2 = REPO_ROOT / "outputs" / "feature_validation" / "cluster_timeseries.png"
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    print(f"时间序列图已保存 -> {out2}")

    # --- 图3：放大最近3年，看清楚翻转频率 ---
    recent = plot_df[plot_df["date"] >= plot_df["date"].max() - pd.Timedelta(days=365 * 3)]
    seg_id_recent = (recent["cluster"] != recent["cluster"].shift(1)).cumsum()
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    for _, seg in recent.groupby(seg_id_recent):
        color = CLUSTER_COLORS[seg["cluster"].iloc[0]]
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1], color=color, alpha=0.25, lw=0)
    axes[0].plot(recent["date"], recent["t_stat"], color="black", lw=0.8)
    axes[0].set_ylabel("t_stat")
    axes[0].set_title("最近3年放大版")
    axes[1].plot(recent["date"], recent["extremity"], color="black", lw=0.8)
    axes[1].set_ylabel("extremity")
    fig.tight_layout()
    out3 = REPO_ROOT / "outputs" / "feature_validation" / "cluster_timeseries_recent3y.png"
    fig.savefig(out3, dpi=130)
    plt.close(fig)
    print(f"最近3年放大图已保存 -> {out3}")


if __name__ == "__main__":
    main()
