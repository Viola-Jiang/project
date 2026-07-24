"""
ablation/diag_s1_interpretability.py
========================================
诊断脚本：S1（K=3, trend_vol闸门后HMM）现在的三个状态，是否真的对应着
干净的经济含义？还是只是"收益排序凑出来的bull/sideways/bear"？

背景（1.2S1层级优化.md）：S1现在用K=3的Gaussian HMM在[trend_W(闸门后),
vol_W]上聚类，按状态历史平均收益排名命名bull/sideways/bear。这只验证了
"三个状态的收益是否单调"，没验证"这三个状态在方向x波动率两个维度上是否
对应清晰的市场状态"——如果某个状态其实是"有方向的温和趋势"和"无方向的
宽幅震荡"两种日子的混合体，收益排序仍可能凑巧单调，但这种状态不能被稳定
解释为"牛市/熊市/震荡"。

方法（照搬中邮《市场脉搏》图表3/图表4/图表5的方法论，但不采用其模型，
只借用其"用规则切格子做可解释性对照"的思路）：
  1. 用S1同一套[trend_W, vol_W]特征（含显著性闸门），按"方向显著性 x
     波动率相对滚动均值高低"切出6个纯规则格子（趋势上涨/温和上涨/窄幅
     震荡/宽幅震荡/温和下跌/趋势下跌），不依赖任何聚类模型。
  2. 独立算这6个格子各自的收益/波动/胜率统计（对应图表5），先检查规则
     本身在我们的数据上是否成立（不然HMM怎么分都无意义）。
  3. 复现S1生产用的HMM拟合+Viterbi解码，与第1步的6个格子做交叉列联表
     （对应图表4）。若每个HMM状态都集中对应1~2个格子，说明当前的三态
     划分可解释；若分布分散，说明"bull/sideways/bear"这个命名当前只是
     数值上排得开，不是真正对应清晰的市场状态。

本脚本只读、不改动S1生产代码、不改K，纯诊断对照。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import (  # noqa: E402
    load_data, K_REGIMES, TREND_VOL_WINDOW, TREND_GATE_THRESHOLD, VOL_SPLIT_WINDOW,
)
from engine.features import rolling_trend_vol_features, gate_trend_by_significance, classify_high_vol  # noqa: E402
from engine.hmm_offline import fit_hmm, decode_smoothed  # noqa: E402
from engine.regime import name_clusters_by_return_rank  # noqa: E402


def build_rule_based_categories(df: pd.DataFrame) -> pd.Series:
    """
    纯规则切格子，不依赖任何聚类：方向（闸门后trend符号，闸门=0记为无方向）
    x 波动率（相对90天滚动均值高/低，engine.features.classify_high_vol），
    共6类。作为S1 HMM划分是否可解释的独立参照系。
    """
    trend, vol = rolling_trend_vol_features(df["log_return"].values, window=TREND_VOL_WINDOW)
    gated = gate_trend_by_significance(trend, vol, window=TREND_VOL_WINDOW, t_threshold=TREND_GATE_THRESHOLD)

    high_vol = classify_high_vol(vol, window=VOL_SPLIT_WINDOW)
    direction = np.where(gated > 0, "up", np.where(gated < 0, "down", "flat"))

    cat = np.select(
        [
            (direction == "up") & high_vol,
            (direction == "up") & ~high_vol,
            (direction == "down") & high_vol,
            (direction == "down") & ~high_vol,
            (direction == "flat") & high_vol,
            (direction == "flat") & ~high_vol,
        ],
        ["趋势上涨", "温和上涨", "趋势下跌", "温和下跌", "宽幅震荡", "窄幅震荡"],
        default="未分类",
    )
    return pd.Series(cat, index=df.index, name="rule_category")


def rule_category_stats(df: pd.DataFrame, category: pd.Series) -> pd.DataFrame:
    """图表5对照：6个规则格子各自的收益/波动/胜率，不涉及任何模型。"""
    rows = []
    r = df["log_return"]
    for name, sub in r.groupby(category.values):
        rows.append({
            "category": name,
            "n_days": len(sub),
            "day_share": len(sub) / len(df),
            "ann_ret": sub.mean() * 252,
            "ann_vol": sub.std() * np.sqrt(252),
            "win_rate": (sub > 0).mean(),
        })
    return pd.DataFrame(rows).sort_values("ann_ret", ascending=False).reset_index(drop=True)


def s1_hmm_state_seq(df: pd.DataFrame, k: int = K_REGIMES, seed: int = 0) -> pd.Series:
    """复现S1生产用的HMM拟合+Viterbi平滑解码，按收益排名命名（与S1口径一致）。"""
    model, feat = fit_hmm(df, k=k, seed=seed, feature_mode="trend_vol",
                           trend_vol_window=TREND_VOL_WINDOW, gate_threshold=TREND_GATE_THRESHOLD)
    state_seq = decode_smoothed(model, feat)
    name_of = name_clusters_by_return_rank(state_seq, df["log_return"].values, k)
    return pd.Series([name_of[s] for s in state_seq], index=df.index, name="s1_state")


def crosstab_pct(state: pd.Series, category: pd.Series) -> pd.DataFrame:
    """图表4对照：行=HMM状态，列=规则格子，值=行内百分比。"""
    ct = pd.crosstab(state, category)
    order = ["趋势上涨", "温和上涨", "窄幅震荡", "宽幅震荡", "温和下跌", "趋势下跌"]
    ct = ct.reindex(columns=[c for c in order if c in ct.columns])
    return (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)


if __name__ == "__main__":
    df = load_data()

    category = build_rule_based_categories(df)
    print("=== 1. 规则格子自身的经济特征（图表5对照，不涉及任何模型）===")
    print(rule_category_stats(df, category).to_string(index=False))

    state = s1_hmm_state_seq(df)
    print("\n=== 2. S1(K=3, trend_vol, 闸门1.5) 状态 vs 规则格子 交叉表（图表4对照，行百分比）===")
    ct = crosstab_pct(state, category)
    print(ct.to_string())

    print("\n=== 3. 每个HMM状态在规则格子上的最大集中度（越接近100%越可解释，越分散越像\"排序凑出来的\"）===")
    print(ct.max(axis=1).to_string())
