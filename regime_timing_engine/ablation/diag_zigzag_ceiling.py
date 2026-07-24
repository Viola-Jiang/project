"""
ablation/diag_zigzag_ceiling.py
===================================
诊断：把zigzag规则式标签（engine/zigzag_labeling.py，允许全样本前视，代表
"完美分类"参照）套上与S1同一套凯利标定机制，实际跑一遍回测，得到一个当前
代码库下可复现的"理论上限"数字，与S1(HMM)的0.74直接对比。

背景：1.2S1层级优化.md提到过一个"完美分类，1.91"的夏普数字，但仓库里没有
生成它的脚本（搜过git历史也没有），无法确认当时是用什么代码/参数算出来的，
不应该继续引用一个对不上现在代码的数字。这里用当前代码重新算一遍，口径是：
  1. zigzag_label_regimes 对整段历史价格做规则式bull/sideways/bear划分
     （Zig-Zag转折点+Binseg趋势衰竭修正，全样本，等于"事后已知走势"的参照）；
  2. 对每个标签算mu/sigma，用engine.decision.calibrate_target_exposures做
     与S1完全相同的分数凯利标定（不做S1那版波动强度细分，先看最朴素的版本，
     跟"以前的1.91"最可能的口径对齐）；
  3. engine.evaluation.compute_backtest_metrics跑回测，lag=1（次日执行），
     与S0~S4完全同一套评估口径。

不追求复现1.91这个具体数字（做不到，代码版本对不上），追求的是"用同一套
凯利标定+回测逻辑，规则式全样本标签 vs 统计模型全样本标签，差距具体有多大、
差在哪"。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import load_data  # noqa: E402
from engine.zigzag_labeling import zigzag_label_regimes  # noqa: E402
from engine.decision import calibrate_target_exposures  # noqa: E402
from engine.evaluation import compute_backtest_metrics, print_metrics  # noqa: E402


def calibrate_by_label(log_returns: np.ndarray, labels: pd.Series) -> dict:
    """按标签算mu/sigma、用与S1同一个calibrate_target_exposures做分数凯利标定。"""
    stats = {}
    for name, idx in pd.Series(labels).groupby(labels).groups.items():
        r = log_returns[idx]
        stats[name] = {"mu": float(r.mean()), "sigma": float(r.std())}
    exposures = calibrate_target_exposures(stats)
    for name in stats:
        stats[name]["target_exposure"] = exposures[name]
    return stats


if __name__ == "__main__":
    df = load_data()
    labeled = zigzag_label_regimes(df)

    label_stats = calibrate_by_label(df["log_return"].values, labeled["ref_regime"])
    print("=== zigzag规则式标签(全样本前视) 的凯利标定 ===")
    for name, s in label_stats.items():
        print(f"  {name}: mu={s['mu']:.5f}, sigma={s['sigma']:.5f}, w*={s['target_exposure']:.2f}")

    w_target = labeled["ref_regime"].map(lambda name: label_stats[name]["target_exposure"])
    metrics = compute_backtest_metrics(df["log_return"], w_target, regime_labels=labeled["ref_regime"])
    print_metrics("zigzag上限(规则式,全样本)", metrics)
    print("分区制绩效:")
    for name, s in metrics["by_regime"].items():
        print(f"  {name}: 年化收益={s['ann_return']*100:.2f}%  夏普={s['sharpe']:.2f}  n={s['n_obs']}")

    print("\n=== zigzag标签 vs S1(HMM)状态 的段数/持仓结构对比 ===")
    seg_id = (labeled["ref_regime"] != labeled["ref_regime"].shift(1)).cumsum()
    n_segments = labeled.groupby(seg_id)["ref_regime"].first().shape[0]
    print(f"zigzag切出的段数: {n_segments}（越少说明标签越'干脆'，换手需求越低）")
    print(f"zigzag下bull/sideways/bear天数占比: "
          f"{labeled['ref_regime'].value_counts(normalize=True).round(3).to_dict()}")
