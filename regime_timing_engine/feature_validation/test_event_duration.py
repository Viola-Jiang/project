"""
feature_validation/test_event_duration.py
=============================================
第一、二步：给"久期"找一个不需要先分段、能直接从未来价格路径读出来的客观目标，
再拿四个候选特征去测跟它的IC——方法跟validate_features.py验证收益IC完全一样，
只是把目标换成"距下一次X%回撤还有多少天"，避免久期检验必须先分段才能验证的
循环论证（见1.3文档讨论，以及这次对话本身）。

目标定义：
  time_to_drawdown(t, X) = 从t+1开始，未来价格路径相对"自t以来的运行峰值"，
  第一次回撤超过X%所需的交易日数。超过max_horizon仍未发生的，按max_horizon
  右截断（记录censored标记，供分开看是否censored样本主导结果）。

用两个阈值（5%、10%）做敏感性检验，避免只测一个阈值导致结论脆弱。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from feature_validation.validate_features import compute_all_features  # noqa: E402

MAX_HORIZON = 252  # 约1年，未来路径最多看这么远
DRAWDOWN_THRESHOLDS = (0.05, 0.10)


def compute_time_to_drawdown(price: np.ndarray, threshold: float, max_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    对每个t，从t+1起看未来价格路径，找"相对t以来运行峰值"的回撤第一次
    超过threshold的最小交易日数。max_horizon内未发生的，按max_horizon
    右截断，censored标记为True。
    """
    n = len(price)
    time_to_event = np.full(n, np.nan)
    censored = np.zeros(n, dtype=bool)
    for t in range(n - 1):
        horizon = min(max_horizon, n - 1 - t)
        if horizon < 5:
            continue
        path = price[t + 1: t + 1 + horizon]
        running_peak = np.maximum.accumulate(np.concatenate([[price[t]], path]))[1:]
        drawdown = (running_peak - path) / running_peak
        hit = np.where(drawdown >= threshold)[0]
        if len(hit) > 0:
            time_to_event[t] = hit[0] + 1
        else:
            time_to_event[t] = horizon
            censored[t] = True
    return time_to_event, censored


def check_event_duration_ic(df: pd.DataFrame) -> None:
    price = df["price"].to_numpy()
    feat_cols = [("feat_t_stat", "t_stat(强度)"), ("feat_er", "ER(路径)"),
                 ("feat_autocorr", "autocorr(路径)"), ("feat_extremity", "extremity(透支)")]

    for threshold in DRAWDOWN_THRESHOLDS:
        print(f"\n[久期替代目标] 距下一次{threshold*100:.0f}%回撤的天数 (max_horizon={MAX_HORIZON})")
        time_to_event, censored = compute_time_to_drawdown(price, threshold, MAX_HORIZON)
        valid = ~np.isnan(time_to_event)
        n_censored = int(censored[valid].sum())
        print(f"  有效样本{valid.sum()}个，其中右截断（{MAX_HORIZON}天内未发生{threshold*100:.0f}%回撤）"
              f"{n_censored}个({n_censored/valid.sum()*100:.1f}%)")

        for col, label in feat_cols:
            rho_all, p_all = spearmanr(df[col][valid], time_to_event[valid])
            uncensored = valid & ~censored
            rho_unc, p_unc = spearmanr(df[col][uncensored], time_to_event[uncensored])
            flag_all = "  <-- 有意义" if abs(rho_all) > 0.05 and p_all < 0.01 else ""
            print(f"    IC({label:16s}) 全样本={rho_all:+.4f}(p={p_all:.4g}){flag_all}   "
                  f"仅未截断样本={rho_unc:+.4f}(p={p_unc:.4g}, n={uncensored.sum()})")


def main():
    from ablation.common import load_data  # noqa: E402
    df = load_data().reset_index(drop=True)
    print(f"数据: {len(df)}行, {df['date'].min().date()} ~ {df['date'].max().date()}")

    df = compute_all_features(df)
    check_event_duration_ic(df)


if __name__ == "__main__":
    main()
