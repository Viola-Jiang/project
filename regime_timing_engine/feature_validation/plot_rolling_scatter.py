"""
feature_validation/plot_rolling_scatter.py
==============================================
跟plot_multiaxis_scatter.py做受控对照：两份脚本用完全相同的特征计算
（engine.features里的生产函数）、完全相同的K=4、完全相同的随机种子，
唯一的差别是标准化方式——这份用rolling_standardize（滚动，逐日因果重算），
plot_multiaxis_scatter.py用的是标量（对整个历史窗口算一次std再除，不减均值）。

这样两张图之间的任何差异，都能干净地归因于"标准化方式不同"这一个变量，
不会像跟tier1_k4_scatter.png比那样，还混着"脚本实现细节是否完全一致"这层
不确定性。
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.features import (  # noqa: E402
    rolling_trend_vol_features, compute_t_stat, compute_extremity_percentile,
    compute_rolling_percentile, rolling_standardize,
)
from engine.plotting import setup_cjk_font  # noqa: E402

K = 4
STANDARDIZE_WINDOW = 500
CLUSTER_COLORS = ["#3f6fa8", "#e8b84b", "#d94f4f", "#5a9367"]


def main():
    from ablation.common import load_data, EXTREMITY_M, EXTREMITY_LOOKBACK, VOL_LOOKBACK, TREND_VOL_WINDOW  # noqa: E402
    df = load_data().reset_index(drop=True)
    log_returns = df["log_return"].values

    trend, vol = rolling_trend_vol_features(log_returns, window=TREND_VOL_WINDOW)
    t_stat = compute_t_stat(trend, vol, TREND_VOL_WINDOW)
    extremity = compute_extremity_percentile(log_returns, m=EXTREMITY_M, lookback=EXTREMITY_LOOKBACK)
    vol_level = compute_rolling_percentile(df["realized_vol"].values, VOL_LOOKBACK)

    # 唯一跟plot_multiaxis_scatter.py不同的地方：这里用rolling_standardize
    # （每天用自己前STANDARDIZE_WINDOW天的均值/标准差，逐日因果重算），
    # 不是"对整个历史窗口算一次标量std再除"。
    feat_std = np.column_stack([
        rolling_standardize(t_stat, STANDARDIZE_WINDOW),
        rolling_standardize(extremity, STANDARDIZE_WINDOW),
        rolling_standardize(vol_level, STANDARDIZE_WINDOW),
    ])

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
    fig.suptitle("K=4聚类，滚动标准化版（跟plot_multiaxis_scatter.py受控对照，唯一变量是标准化方式）")
    fig.tight_layout()

    out = REPO_ROOT / "outputs" / "feature_validation" / "rolling_k4_scatter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"散点图已保存 -> {out}")


if __name__ == "__main__":
    main()
