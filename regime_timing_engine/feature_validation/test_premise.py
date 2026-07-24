"""
feature_validation/test_premise.py
=====================================
在继续挖特征之前，先独立验证前提本身：这只指数是否存在"未来收益分布""未来
持续时间"这两种可被划分的结构——不涉及validate_features.py里那4个候选特征，
避免任何循环论证。

两条检验：
  1. 收益结构：Ljung-Box检验squared/absolute log_return的自相关——测波动率
     聚集性（有没有"平静期/剧烈期"这种基本区分）。
  2. 久期结构：用ref_regime（完全独立于候选特征的历史分段）切出真实段，
     测"活得越久，今天结束的条件概率会不会变"——如果跟年龄(段龄)无关，
     说明这个过程接近无记忆(几何分布)，久期没有可开发的结构。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.diagnostic import acorr_ljungbox

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def check_volatility_clustering(log_return: pd.Series) -> None:
    print("\n[前提检验1] 收益结构是否存在：波动率聚集性（Ljung-Box）")
    r = log_return.dropna()
    for label, series in [("squared return (r^2)", r ** 2), ("absolute return (|r|)", r.abs())]:
        result = acorr_ljungbox(series, lags=[10, 20, 40], return_df=True)
        print(f"  {label}:")
        print(result.rename(columns={"lb_stat": "LB统计量", "lb_pvalue": "p值"}).to_string())
    print("  (若p值远小于0.05，说明波动率不是恒定的，存在'平静/剧烈'这种基本"
          "结构——这是收益端区制存在的必要条件，不是充分条件)")


def check_duration_structure(df: pd.DataFrame) -> None:
    print("\n[前提检验2] 久期结构是否存在：结束风险是否随'已存活时长(段龄)'变化")
    print("  (用ref_regime——完全独立于候选特征的历史分段——避免循环论证)")

    labels = df["ref_regime"]
    valid = labels.notna()
    seg_id = (labels != labels.shift(1)).cumsum()
    last_idx = df.index[valid][-1]

    records = []
    for sid, g in df.loc[valid].groupby(seg_id[valid]):
        start, end = g.index[0], g.index[-1]
        is_censored = end == last_idx
        for i, day_idx in enumerate(range(start, end + 1)):
            is_last_day = day_idx == end
            event = 0 if is_censored else int(is_last_day)
            records.append({"age": i + 1, "event": event})
    hazard_df = pd.DataFrame(records)
    print(f"  共{len(hazard_df)}个'天'级观测，来自ref_regime切出的段（已排除末尾右截断段的贡献方式："
          f"其所有天event强制为0，段本身仍计入分母）")

    age_bins = pd.qcut(hazard_df["age"], 4, duplicates="drop")
    contingency = pd.crosstab(age_bins, hazard_df["event"])
    print("\n  按段龄(age)分4组，各组日均结束风险:")
    print((contingency[1] / contingency.sum(axis=1)).round(5).to_string())

    chi2_stat, p_value, dof, _ = chi2_contingency(contingency)
    print(f"\n  卡方检验（结束风险是否随段龄变化）: chi2={chi2_stat:.2f}, df={dof}, p={p_value:.4g}"
          f"{'  <-- 显著，久期存在可开发的年龄相依结构' if p_value < 0.05 else '  <-- 未见显著差异，接近无记忆过程'}")

    print("\n  另外看一眼原始的段长分布（不分组，整体是否像几何分布）:")
    seg_lengths = df.loc[valid].groupby(seg_id[valid]).apply(lambda g: g.index[-1] - g.index[0] + 1)
    seg_lengths = seg_lengths[:-1]  # 剔除末尾右截断段
    mean_d, var_d = seg_lengths.mean(), seg_lengths.var()
    geo_var_if_memoryless = mean_d * (mean_d - 1)  # 几何分布 Var ≈ mean*(mean-1)，粗略参照
    print(f"  段长: n={len(seg_lengths)}, 均值={mean_d:.1f}, 方差={var_d:.1f}, "
          f"（若为无记忆几何分布，方差理论上≈{geo_var_if_memoryless:.1f}）")
    print(f"  实际方差/理论几何方差 = {var_d / geo_var_if_memoryless:.2f} "
          f"(远大于1说明比无记忆过程更'过度分散'，暗示存在异质性结构)")


def main():
    from ablation.common import load_data  # noqa: E402
    df = load_data().reset_index(drop=True)
    print(f"数据: {len(df)}行, {df['date'].min().date()} ~ {df['date'].max().date()}")

    check_volatility_clustering(df["log_return"])
    check_duration_structure(df)


if __name__ == "__main__":
    main()
