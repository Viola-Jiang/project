"""
ablation/diag_trend_window_grid.py
======================================
诊断/探索：路径2的第一步——验证"固定21天窗口"是否是S1覆盖率过低（16% vs
zigzag的68.7%，见1.2S1层级优化.md）的真正瓶颈，具体做法是看拉长窗口能不能
让更多"温和但持续"的趋势通过显著性闸门。

直觉（可推导）：t_stat = trend_W / (vol_W * sqrt(W))。若某段时期日收益
近似独立同分布、真实日均漂移为mu、日波动为sigma，则 trend_W≈mu*W，
vol_W≈sigma（近似不随W变化），t_stat≈mu*sqrt(W)/sigma——随窗口变长按
sqrt(W)增长。也就是说，一段"温和但持续"的真实趋势（mu不大但方向稳定），
在21天窗口里可能怎么也凑不够显著性，换成63/126天窗口，同样的mu会因为
sqrt(W)变大而被放大成显著。这正是zigzag（不预设固定窗口，只要总涨跌幅
够大，不管用了多少天）能覆盖68.7%天数、而21天窗口只能覆盖16%的可能原因。

本脚本分两阶段：
  阶段1（便宜，不跑HMM）：对一组窗口x阈值组合，只算"闸门后trend非零"的
    天数占比——验证上面这个sqrt(W)直觉在真实数据上是否成立，不花HMM拟合
    的时间。
  阶段2（贵，跑HMM+回测）：对阶段1里"覆盖率明显提升"的窗口，挑一小组
    (窗口,阈值)组合实际跑一遍S1的HMM拟合+回测，看覆盖率提升是否真的转化
    成夏普提升（更宽的窗口也可能引入更多噪声/延迟，覆盖率涨不代表夏普涨，
    需要实测）。

不改变S1生产默认值（TREND_VOL_WINDOW=21），纯诊断/探索脚本。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import load_data, K_REGIMES, VOL_SPLIT_WINDOW  # noqa: E402
from engine.features import rolling_trend_vol_features, gate_trend_by_significance  # noqa: E402
from engine.hmm_offline import fit_offline_hmm_positions  # noqa: E402
from engine.evaluation import compute_backtest_metrics  # noqa: E402

WINDOW_GRID = [21, 42, 63, 94, 126, 189, 252]
THRESHOLD_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def significant_day_share(log_returns: np.ndarray, window: int, t_threshold: float) -> float:
    """闸门后trend非零的天数占比，不涉及任何模型拟合。"""
    trend, vol = rolling_trend_vol_features(log_returns, window=window)
    gated = gate_trend_by_significance(trend, vol, window=window, t_threshold=t_threshold)
    return float((gated != 0).mean())


if __name__ == "__main__":
    df = load_data()
    log_returns = df["log_return"].values

    print("=== 阶段1：不同(窗口,阈值)组合下，显著天数占比（不跑模型）===")
    print("对照：zigzag规则式标签的'有方向'天数占比 = 68.7%\n")
    rows = []
    for w in WINDOW_GRID:
        for t in THRESHOLD_GRID:
            rows.append({"window": w, "threshold": t,
                         "sig_day_share": significant_day_share(log_returns, w, t)})
    table1 = pd.DataFrame(rows).pivot(index="window", columns="threshold", values="sig_day_share")
    print((table1 * 100).round(1).to_string())

    print("\n=== 阶段2：挑一组(窗口,阈值)实测S1夏普（跑HMM+回测，vol_split默认开启）===")
    candidates = [(21, 1.5), (63, 1.5), (63, 2.0), (126, 1.5), (126, 2.0), (126, 2.5),
                  (252, 2.0), (252, 2.5), (252, 3.0)]
    rows2 = []
    for w, t in candidates:
        w_target, _ = fit_offline_hmm_positions(
            df, k=K_REGIMES, seed=0, feature_mode="trend_vol", trend_vol_window=w,
            gate_threshold=t, vol_split_window=VOL_SPLIT_WINDOW)
        m = compute_backtest_metrics(df["log_return"], w_target)
        rows2.append({"window": w, "threshold": t, "sig_day_share": significant_day_share(log_returns, w, t),
                      "ann_return": m["ann_return"], "ann_vol": m["ann_vol"], "sharpe": m["sharpe"],
                      "max_dd": m["max_dd"], "turnover": m["turnover_rate"]})
    table2 = pd.DataFrame(rows2)
    print(table2.round(4).to_string(index=False))
    best = table2.loc[table2["sharpe"].idxmax()]
    print(f"\n最优组合: window={best['window']}, threshold={best['threshold']}, 夏普={best['sharpe']:.3f}")
    print("对照：现生产配置 window=21, threshold=1.5 的夏普=0.737")
