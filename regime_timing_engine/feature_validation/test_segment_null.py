"""
feature_validation/test_segment_null.py
===========================================
验证"KMeans聚出来的段落"里，到底有没有超出纯机械假象之外的真实结构。

第一版只比较了段长中位数，结论是"中位数落在打乱顺序的零假设范围内"——但
后来画图发现，真实数据里其实混杂着两种完全不同的行为：大量在簇边界附近来回
抖动的短段（噪声），和少数确实很长、很稳定的段（比如连续4个月都在同一个簇）。
中位数被数量占多数的短段主导，会把这些真正有意义的长段掩盖掉。

这一版改成比较"尾部"指标——真实数据里"特别长的段"，是不是比打乱顺序的
零假设更多、更极端：
  - 段长的90/95分位数
  - 最长单段的长度
  - 落在"超过30天"的长段里的天数占全部天数的比例

同时把标准化换成跟visualize_clustering.py一致的滚动（因果）版本，不再用
一次性全样本的StandardScaler。
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
    compute_extremity_percentile, extract_segments_from_cluster_labels,
    STRENGTH_WINDOW, EXTREMITY_M, EXTREMITY_L,
)
from feature_validation.visualize_clustering import rolling_standardize, STANDARDIZE_WINDOW  # noqa: E402

N_SHUFFLES = 100
K = 3
RNG_SEED = 42
LONG_SEGMENT_THRESHOLD = 30  # 天


def build_features_and_cluster(log_return: np.ndarray, k: int = K, rng: np.random.Generator = None):
    """给定一条log_return序列，算t_stat/extremity，滚动标准化后跑KMeans，返回段长数组。"""
    trend, vol = rolling_trend_vol_features(log_return, window=STRENGTH_WINDOW)
    t_stat = trend / (vol * np.sqrt(STRENGTH_WINDOW))
    extremity = compute_extremity_percentile(log_return, m=EXTREMITY_M, lookback=EXTREMITY_L)

    t_stat_std = rolling_standardize(t_stat, STANDARDIZE_WINDOW)
    extremity_std = rolling_standardize(extremity, STANDARDIZE_WINDOW)

    X_std = np.column_stack([t_stat_std, extremity_std])
    seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X_std)

    df = pd.DataFrame({"cluster": labels, "log_return": log_return})
    seg_df = extract_segments_from_cluster_labels(df, "cluster")
    return seg_df["duration"].to_numpy()


def tail_stats(durations: np.ndarray, threshold: int = LONG_SEGMENT_THRESHOLD) -> dict:
    total_days = durations.sum()
    long_days = durations[durations > threshold].sum()
    return {
        "p90": np.percentile(durations, 90),
        "p95": np.percentile(durations, 95),
        "max": durations.max(),
        "frac_long": long_days / total_days if total_days > 0 else 0.0,
    }


def main():
    from ablation.common import load_data  # noqa: E402
    df = load_data().reset_index(drop=True)
    log_return = df["log_return"].fillna(0.0).to_numpy()

    print(f"数据: {len(df)}行")
    print("\n[真实数据] 聚类切段:")
    real_durations = build_features_and_cluster(log_return)
    real_stats = tail_stats(real_durations)
    print(f"  真实段长: n={len(real_durations)}, 中位数={np.median(real_durations):.1f}, "
          f"p90={real_stats['p90']:.1f}, p95={real_stats['p95']:.1f}, "
          f"最长={real_stats['max']:.0f}天, "
          f"超过{LONG_SEGMENT_THRESHOLD}天的段占全部天数的{real_stats['frac_long']*100:.1f}%")

    print(f"\n[零假设] 把log_return完全打乱顺序{N_SHUFFLES}次，重算特征、重新聚类、重新切段:")
    rng = np.random.default_rng(RNG_SEED)
    null_stats_list = []
    for i in range(N_SHUFFLES):
        shuffled = rng.permutation(log_return)
        durations = build_features_and_cluster(shuffled, rng=rng)
        null_stats_list.append(tail_stats(durations))
        if (i + 1) % 20 == 0:
            print(f"  已完成 {i + 1}/{N_SHUFFLES}")

    null_df = pd.DataFrame(null_stats_list)

    print("\n--- 尾部指标对比：真实值 vs 零假设(打乱顺序)分布 ---")
    for metric, label in [("p90", "段长90分位数"), ("p95", "段长95分位数"),
                          ("max", "最长单段"), ("frac_long", f"超过{LONG_SEGMENT_THRESHOLD}天的段占比")]:
        null_vals = null_df[metric].to_numpy()
        real_val = real_stats[metric]
        p_le = (null_vals >= real_val).mean()  # 真实值比零假设更极端(更大)的概率
        print(f"  {label}: 真实值={real_val:.3f}  "
              f"零假设[5%,50%,95%分位]=[{np.percentile(null_vals,5):.3f}, "
              f"{np.percentile(null_vals,50):.3f}, {np.percentile(null_vals,95):.3f}]  "
              f"P(零假设>=真实值)={p_le:.3f}"
              f"{'  <-- 真实值显著更极端' if p_le < 0.05 else ''}")

    # 画图：p95段长 和 最长单段 的零假设分布直方图，标出真实值
    setup_cjk_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, label in [(axes[0], "p95", "段长95分位数(天)"), (axes[1], "max", "最长单段(天)")]:
        ax.hist(null_df[metric], bins=20, color="#8a9bb0", alpha=0.8, label="零假设(打乱顺序)")
        ax.axvline(real_stats[metric], color="#d94f4f", lw=2, label="真实数据")
        ax.set_xlabel(label)
        ax.set_ylabel("次数")
        ax.legend(fontsize=8)
    fig.suptitle("真实数据 vs 打乱顺序零假设：段长尾部指标对比")
    fig.tight_layout()
    out = REPO_ROOT / "outputs" / "feature_validation" / "segment_tail_null_test.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n尾部指标对比图已保存 -> {out}")


if __name__ == "__main__":
    main()
