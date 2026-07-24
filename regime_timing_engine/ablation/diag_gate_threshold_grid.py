"""
ablation/diag_gate_threshold_grid.py
========================================
诊断：在"按波动强度细分标定凯利仓位"（vol_split_window，见1.2S1层级优化.md
第三次修正）上线之后，重新网格搜索trend显著性闸门阈值（gate_threshold），
检查1.5是否还是最优点。

背景：1.5这个阈值是在旧的标定方式（每个状态整体算一个mu/sigma，见
calibrate_state_exposures）下网格搜出来的最优点。新标定方式下，方向稍弱
的日子不再是"要么0要么1"，而是能按波动强度分到更细的仓位，闸门本身的
"过滤噪声"这个角色可能不需要以前那么激进——所以有必要在新标定方式下
重新搜一遍，而不是继续沿用旧阈值。

不改变S1生产默认值（仍是gate_threshold=1.5），纯诊断脚本；如果搜出更好的
阈值，由人工决定是否更新ablation/common.py里的TREND_GATE_THRESHOLD。
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import load_data, K_REGIMES, TREND_VOL_WINDOW, VOL_SPLIT_WINDOW  # noqa: E402
from engine.hmm_offline import fit_offline_hmm_positions  # noqa: E402
from engine.evaluation import compute_backtest_metrics  # noqa: E402

GATE_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def run_one(df: pd.DataFrame, gate_threshold: float | None, vol_split_window: int | None,
            seed: int = 0) -> dict:
    w_target, _ = fit_offline_hmm_positions(
        df, k=K_REGIMES, seed=seed, feature_mode="trend_vol", trend_vol_window=TREND_VOL_WINDOW,
        gate_threshold=gate_threshold, vol_split_window=vol_split_window)
    m = compute_backtest_metrics(df["log_return"], w_target)
    return {
        "gate_threshold": gate_threshold,
        "ann_return": m["ann_return"], "ann_vol": m["ann_vol"], "sharpe": m["sharpe"],
        "max_dd": m["max_dd"], "turnover": m["turnover_rate"],
    }


if __name__ == "__main__":
    df = load_data()

    print("=== 新标定方式(vol_split_window=90)下，闸门阈值网格搜索 ===")
    rows_new = [run_one(df, g, VOL_SPLIT_WINDOW) for g in GATE_GRID]
    table_new = pd.DataFrame(rows_new)
    print(table_new.round(4).to_string(index=False))
    best_new = table_new.loc[table_new["sharpe"].idxmax()]
    print(f"\n新标定方式下最优阈值: {best_new['gate_threshold']}  夏普={best_new['sharpe']:.3f}")

    print("\n=== 对照：旧标定方式(不做vol_split)下，同一组阈值重新跑一遍，确认1.5仍是旧标定下的最优点 ===")
    rows_old = [run_one(df, g, None) for g in GATE_GRID]
    table_old = pd.DataFrame(rows_old)
    print(table_old.round(4).to_string(index=False))
    best_old = table_old.loc[table_old["sharpe"].idxmax()]
    print(f"\n旧标定方式下最优阈值: {best_old['gate_threshold']}  夏普={best_old['sharpe']:.3f}")
