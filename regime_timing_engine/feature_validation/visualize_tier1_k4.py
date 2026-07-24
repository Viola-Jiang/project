"""
feature_validation/visualize_tier1_k4.py
============================================
把test_tier1_unified_clustering.py里K=4（三维[t_stat, extremity, vol_level]）
这次成功的聚类结果画出来：
  1. 三组两两散点图（每两维一张），按簇上色
  2. 价格 + 三个特征随时间变化，背景按簇上色——直接看这四个簇落在历史上
     哪些具体时期，有没有说得通的经济含义
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.plotting import setup_cjk_font  # noqa: E402
from feature_validation.test_tier1_unified_clustering import build_feature_matrix  # noqa: E402

K = 4
CLUSTER_COLORS = ["#3f6fa8", "#e8b84b", "#d94f4f", "#5a9367"]
FEAT_NAMES = ["t_stat(强度)", "extremity(透支)", "vol_level(波动率水平)"]


def main():
    from ablation.common import load_data  # noqa: E402
    df = load_data().reset_index(drop=True)

    X = build_feature_matrix(df)
    labels = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(X)
    print("各簇大小:", pd.Series(labels).value_counts().sort_index().to_dict())
    print("各簇特征均值(标准化后):")
    print(pd.DataFrame(X, columns=["t_stat", "extremity", "vol_level"])
          .assign(cluster=labels).groupby("cluster").mean().round(3).to_string())

    setup_cjk_font()
    out_dir = REPO_ROOT / "outputs" / "feature_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 图1：三组两两散点 ---
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (i, j) in zip(axes, pairs):
        for k in range(K):
            mask = labels == k
            ax.scatter(X[mask, i], X[mask, j], s=5, alpha=0.4,
                       color=CLUSTER_COLORS[k], label=f"簇{k}")
        ax.set_xlabel(FEAT_NAMES[i])
        ax.set_ylabel(FEAT_NAMES[j])
    axes[0].legend(fontsize=8)
    fig.suptitle("K=4聚类，三组两两散点图（标准化后）")
    fig.tight_layout()
    out1 = out_dir / "tier1_k4_scatter.png"
    fig.savefig(out1, dpi=130)
    plt.close(fig)
    print(f"散点图已保存 -> {out1}")

    # --- 图2：价格 + 三特征，随时间变化，背景按簇上色 ---
    plot_df = pd.DataFrame({
        "date": df["date"], "price": df["price"],
        "t_stat": X[:, 0], "extremity": X[:, 1], "vol_level": X[:, 2],
        "cluster": labels,
    })
    seg_id = (plot_df["cluster"] != plot_df["cluster"].shift(1)).cumsum()

    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)
    for _, seg in plot_df.groupby(seg_id):
        color = CLUSTER_COLORS[seg["cluster"].iloc[0]]
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1], color=color, alpha=0.25, lw=0)

    axes[0].plot(plot_df["date"], plot_df["price"], color="black", lw=0.8)
    axes[0].set_ylabel("价格")
    axes[0].set_yscale("log")
    axes[0].set_title("价格 / t_stat / extremity / vol_level 随时间变化，背景色=K=4聚类簇")
    axes[1].plot(plot_df["date"], plot_df["t_stat"], color="black", lw=0.6)
    axes[1].set_ylabel("t_stat")
    axes[2].plot(plot_df["date"], plot_df["extremity"], color="black", lw=0.6)
    axes[2].set_ylabel("extremity")
    axes[3].plot(plot_df["date"], plot_df["vol_level"], color="black", lw=0.6)
    axes[3].set_ylabel("vol_level")
    axes[3].set_xlabel("日期")
    fig.tight_layout()
    out2 = out_dir / "tier1_k4_timeseries.png"
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    print(f"时间序列图已保存 -> {out2}")

    # --- 图3：最近4年放大 ---
    recent = plot_df[plot_df["date"] >= plot_df["date"].max() - pd.Timedelta(days=365 * 4)]
    seg_id_recent = (recent["cluster"] != recent["cluster"].shift(1)).cumsum()
    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)
    for _, seg in recent.groupby(seg_id_recent):
        color = CLUSTER_COLORS[seg["cluster"].iloc[0]]
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1], color=color, alpha=0.25, lw=0)
    axes[0].plot(recent["date"], recent["price"], color="black", lw=1.0)
    axes[0].set_ylabel("价格")
    axes[0].set_title("最近4年放大版")
    axes[1].plot(recent["date"], recent["t_stat"], color="black", lw=0.8)
    axes[1].set_ylabel("t_stat")
    axes[2].plot(recent["date"], recent["extremity"], color="black", lw=0.8)
    axes[2].set_ylabel("extremity")
    axes[3].plot(recent["date"], recent["vol_level"], color="black", lw=0.8)
    axes[3].set_ylabel("vol_level")
    fig.tight_layout()
    out3 = out_dir / "tier1_k4_timeseries_recent4y.png"
    fig.savefig(out3, dpi=130)
    plt.close(fig)
    print(f"最近4年放大图已保存 -> {out3}")


if __name__ == "__main__":
    main()
