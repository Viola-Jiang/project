"""
feature_validation/plot_multiaxis_scatter.py
================================================
给2.0阶段性报告配图：按生产代码engine.calibration.estimate_regime_params_multiaxis
实际使用的简化标量标准化（对整个历史窗口算一次std再除，不是滚动标准化），
在全部历史上一次性跑K=4聚类，画三个两两投影的散点图，跟tier1_k4_scatter.png
（滚动标准化版本）对照，展示标注1里说的那处差异。
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.features import (  # noqa: E402
    rolling_trend_vol_features, compute_t_stat, compute_extremity_percentile, compute_rolling_percentile,
)
from engine.plotting import setup_cjk_font  # noqa: E402

K = 4
CLUSTER_COLORS = ["#3f6fa8", "#e8b84b", "#d94f4f", "#5a9367"]


def main():
    from ablation.common import load_data, EXTREMITY_M, EXTREMITY_LOOKBACK, VOL_LOOKBACK, TREND_VOL_WINDOW  # noqa: E402
    df = load_data().reset_index(drop=True)
    log_returns = df["log_return"].values

    trend, vol = rolling_trend_vol_features(log_returns, window=TREND_VOL_WINDOW)
    t_stat = compute_t_stat(trend, vol, TREND_VOL_WINDOW)
    extremity = compute_extremity_percentile(log_returns, m=EXTREMITY_M, lookback=EXTREMITY_LOOKBACK)
    vol_level = compute_rolling_percentile(df["realized_vol"].values, VOL_LOOKBACK)

    # 生产代码（engine.calibration.estimate_regime_params_multiaxis）实际用的是
    # feat / feature_scale（不减均值，对整个历史窗口算一次std再除，不是逐日滚动
    # 重算）。这里额外减一下均值，只是为了让画出来的图跟tier1_k4_scatter.png/
    # s1_hmm_states_scatter.png（两者都做了真正的z-score）坐标轴范围可比——减均值
    # 不改变KMeans实际分出来的类别（欧氏距离对整体平移不敏感），所以只在这个画图
    # 脚本里处理，不改生产代码，不影响S1~S4已有的回测结果。
    feat = np.column_stack([t_stat, extremity, vol_level])
    feature_scale = feat.std(axis=0)
    feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
    feat_std = (feat - feat.mean(axis=0)) / feature_scale

    labels = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(feat_std)
    print("各簇大小:", {k: int((labels == k).sum()) for k in range(K)})

    setup_cjk_font()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [("t_stat", feat_std[:, 0], "extremity", feat_std[:, 1]),
             ("t_stat", feat_std[:, 0], "vol_level", feat_std[:, 2]),
             ("extremity", feat_std[:, 1], "vol_level", feat_std[:, 2])]
    for ax, (xname, x, yname, y) in zip(axes, pairs):
        for k in range(K):
            mask = labels == k
            ax.scatter(x[mask], y[mask], s=5, alpha=0.4, color=CLUSTER_COLORS[k], label=f"簇{k}")
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
    axes[0].legend(fontsize=8, markerscale=2)
    fig.suptitle("K=4聚类，标量标准化版+画图用减均值（对照tier1_k4_scatter.png的滚动标准化版）")
    fig.tight_layout()

    out = REPO_ROOT / "outputs" / "feature_validation" / "multiaxis_k4_scatter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"散点图已保存 -> {out}")


if __name__ == "__main__":
    main()
