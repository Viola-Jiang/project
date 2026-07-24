"""
feature_validation/test_tier1_unified_clustering.py
=======================================================
用户的决策树第1步：既然vol_level对久期有强信号，再试一次"同一套聚类同时
处理未来收益和未来久期"，这次把vol_level也加进聚类特征里（不再只是
[t_stat, extremity]二维，扩到三维：[t_stat, extremity, vol_level]）。

跑法：
  1. K=3/K=4，滚动标准化，KMeans（跟之前一致的方法）
  2. 联合充分性检验：收益端Kruskal-Wallis+两两比较，久期端日频卡方+段级Kruskal-Wallis
  3. 零假设检验：这次聚出来的段长（含尾部指标），是不是依然跟"打乱顺序"的
     噪声没区别——因为vol_level本身的"持续性"是被Ljung-Box证实过的真实结构
     （不是单纯滚动窗口的机械假象），如果它进了聚类，段长有可能第一次
     摆脱"纯噪声"的嫌疑
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.features import rolling_trend_vol_features  # noqa: E402
from feature_validation.validate_features import (  # noqa: E402
    compute_extremity_percentile, check_joint_sufficiency_on_clusters,
    extract_segments_from_cluster_labels, STRENGTH_WINDOW, EXTREMITY_M, EXTREMITY_L,
)
from feature_validation.visualize_clustering import rolling_standardize, STANDARDIZE_WINDOW  # noqa: E402
from feature_validation.test_additional_candidates import compute_rolling_percentile, VOL_LOOKBACK  # noqa: E402

K_VALUES = (3, 4)
N_SHUFFLES = 60
RNG_SEED = 42
LONG_SEGMENT_THRESHOLD = 30


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    log_return = df["log_return"].fillna(0.0).to_numpy()
    trend, vol = rolling_trend_vol_features(log_return, window=STRENGTH_WINDOW)
    t_stat = trend / (vol * np.sqrt(STRENGTH_WINDOW))
    extremity = compute_extremity_percentile(log_return, m=EXTREMITY_M, lookback=EXTREMITY_L)
    vol_level = compute_rolling_percentile(df["realized_vol"].to_numpy(), VOL_LOOKBACK)

    t_stat_std = rolling_standardize(t_stat, STANDARDIZE_WINDOW)
    extremity_std = rolling_standardize(extremity, STANDARDIZE_WINDOW)
    vol_level_std = rolling_standardize(vol_level, STANDARDIZE_WINDOW)
    return np.column_stack([t_stat_std, extremity_std, vol_level_std])


def tail_stats(durations: np.ndarray, threshold: int = LONG_SEGMENT_THRESHOLD) -> dict:
    total_days = durations.sum()
    long_days = durations[durations > threshold].sum()
    return {"median": np.median(durations), "p95": np.percentile(durations, 95),
            "max": durations.max(), "frac_long": long_days / total_days if total_days > 0 else 0.0}


def run_null_test(log_return: np.ndarray, k: int, real_stats: dict, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(N_SHUFFLES):
        shuffled = rng.permutation(log_return)
        trend, vol = rolling_trend_vol_features(shuffled, window=STRENGTH_WINDOW)
        t_stat = trend / (vol * np.sqrt(STRENGTH_WINDOW))
        extremity = compute_extremity_percentile(shuffled, m=EXTREMITY_M, lookback=EXTREMITY_L)
        vol_series = pd.Series(shuffled).rolling(20, min_periods=5).std().bfill().to_numpy()
        vol_level = compute_rolling_percentile(vol_series, VOL_LOOKBACK)

        X = np.column_stack([rolling_standardize(t_stat, STANDARDIZE_WINDOW),
                              rolling_standardize(extremity, STANDARDIZE_WINDOW),
                              rolling_standardize(vol_level, STANDARDIZE_WINDOW)])
        seed = int(rng.integers(0, 2**31 - 1))
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        tmp = pd.DataFrame({"cluster": labels, "log_return": shuffled})
        seg_df = extract_segments_from_cluster_labels(tmp, "cluster")
        rows.append(tail_stats(seg_df["duration"].to_numpy()))
    return pd.DataFrame(rows)


def main():
    from ablation.common import load_data  # noqa: E402
    df = load_data().reset_index(drop=True)
    log_return = df["log_return"].fillna(0.0).to_numpy()
    print(f"数据: {len(df)}行")

    X = build_feature_matrix(df)
    print("聚类特征: [t_stat, extremity, vol_level]（三维，滚动标准化）")

    rng = np.random.default_rng(RNG_SEED)

    for k in K_VALUES:
        print(f"\n{'=' * 70}\nK={k}\n{'=' * 70}")
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        clustered = df.copy()
        clustered["cluster"] = labels
        print("各簇大小:", pd.Series(labels).value_counts().sort_index().to_dict())

        check_joint_sufficiency_on_clusters(clustered, cluster_col="cluster",
                                             title=f"K={k}, 三维[t_stat,extremity,vol_level]原始聚类")

        seg_df = extract_segments_from_cluster_labels(clustered, "cluster")
        real_stats = tail_stats(seg_df["duration"].to_numpy())
        print(f"\n段长尾部指标: 中位数={real_stats['median']:.1f}, p95={real_stats['p95']:.1f}, "
              f"最长={real_stats['max']:.0f}, 超{LONG_SEGMENT_THRESHOLD}天占比={real_stats['frac_long']*100:.1f}%")

        print(f"\n零假设检验（打乱顺序{N_SHUFFLES}次，重新算三维特征、重新聚类）:")
        null_df = run_null_test(log_return, k, real_stats, rng)
        for metric, label in [("median", "中位数"), ("p95", "95分位"),
                              ("max", "最长单段"), ("frac_long", f"超{LONG_SEGMENT_THRESHOLD}天占比")]:
            null_vals = null_df[metric].to_numpy()
            real_val = real_stats[metric]
            p = (null_vals >= real_val).mean()
            print(f"  {label}: 真实={real_val:.3f}  零假设[5%,50%,95%]="
                  f"[{np.percentile(null_vals,5):.3f}, {np.percentile(null_vals,50):.3f}, "
                  f"{np.percentile(null_vals,95):.3f}]  P(零假设>=真实)={p:.3f}"
                  f"{'  <-- 真实值显著更极端' if p < 0.05 else ''}")


if __name__ == "__main__":
    main()
