"""
feature_validation/test_additional_candidates.py
====================================================
只用现有数据（price, log_return, realized_vol, date），把能测的候选都测一遍：

  1. 波动率水平（realized_vol相对自身L=500天历史的百分位）——不是当t_stat的
     分母用，是独立的一个特征，理论依据来自已验证的波动率聚集性
  2. 涨跌不对称性：
     a. 滚动窗口内的偏度 skew(log_return, W=60)
     b. 下跌日/上涨日波动率之比 downside_vol / upside_vol (W=60)
  3. 日历效应：星期几、月份

每个候选分别测：
  - 对同期/未来收益的IC（复用validate_features.py的方法）
  - 对"距下一次X%回撤天数"这个久期替代目标的IC（复用test_event_duration.py的方法）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal, percentileofscore

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from feature_validation.validate_features import FORWARD_HORIZONS  # noqa: E402
from feature_validation.test_event_duration import compute_time_to_drawdown, MAX_HORIZON, DRAWDOWN_THRESHOLDS  # noqa: E402

VOL_LOOKBACK = 500  # 跟extremity轴的长期基准窗口保持一致
ASYMMETRY_WINDOW = 60  # 跟自相关轴窗口保持一致


# ------------------------------- 新增特征计算 -------------------------------

def compute_rolling_percentile(series: np.ndarray, lookback: int) -> np.ndarray:
    """给任意序列算"当前值在过去lookback天里的百分位排名"，因果、不看未来。"""
    s = pd.Series(series)
    out = np.full(len(s), np.nan)
    for i in range(len(s)):
        if np.isnan(s.iloc[i]):
            continue
        start = max(0, i - lookback + 1)
        window_vals = s.iloc[start:i + 1].dropna().to_numpy()
        if len(window_vals) < max(20, lookback // 10):
            continue
        out[i] = percentileofscore(window_vals, s.iloc[i], kind="rank")
    return pd.Series(out).bfill().fillna(50.0).to_numpy()


def compute_rolling_skew(log_return: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(log_return)
    min_periods = max(20, window // 3)
    return s.rolling(window, min_periods=min_periods).skew().bfill().fillna(0.0).to_numpy()


def compute_downside_upside_ratio(log_return: np.ndarray, window: int) -> np.ndarray:
    """滚动窗口内，下跌日收益的标准差 / 上涨日收益的标准差。>1表示跌起来比涨起来猛。"""
    s = pd.Series(log_return)
    min_periods = max(20, window // 3)

    def _ratio(x):
        down = x[x < 0]
        up = x[x > 0]
        if len(down) < 3 or len(up) < 3:
            return np.nan
        down_std, up_std = down.std(), up.std()
        if up_std < 1e-12:
            return np.nan
        return down_std / up_std

    out = s.rolling(window, min_periods=min_periods).apply(_ratio, raw=True)
    return out.bfill().fillna(1.0).to_numpy()


# ------------------------------- 检验 -------------------------------

def check_ic_against_forward_return(df: pd.DataFrame, col: str, label: str) -> None:
    log_returns = df["log_return"].fillna(0.0).to_numpy()
    n = len(df)
    print(f"\n  [{label}] 对未来收益的IC:")
    for horizon in FORWARD_HORIZONS:
        fwd = np.full(n, np.nan)
        cum = pd.Series(log_returns).rolling(horizon).sum().to_numpy()
        fwd[:n - horizon] = cum[horizon:]
        valid = ~np.isnan(fwd) & ~np.isnan(df[col])
        rho, p = spearmanr(df[col][valid], fwd[valid])
        flag = "  <-- 有意义" if abs(rho) > 0.05 and p < 0.01 else ""
        print(f"    未来{horizon}日: IC={rho:+.4f} (p={p:.4g}){flag}")


def check_ic_against_event_duration(df: pd.DataFrame, col: str, label: str, price: np.ndarray) -> None:
    print(f"\n  [{label}] 对'距下一次回撤天数'的IC:")
    for threshold in DRAWDOWN_THRESHOLDS:
        time_to_event, censored = compute_time_to_drawdown(price, threshold, MAX_HORIZON)
        valid = ~np.isnan(time_to_event) & ~np.isnan(df[col])
        rho, p = spearmanr(df[col][valid], time_to_event[valid])
        flag = "  <-- 有意义" if abs(rho) > 0.05 and p < 0.01 else ""
        print(f"    {threshold*100:.0f}%回撤阈值: IC={rho:+.4f} (p={p:.4g}, n={valid.sum()}){flag}")


def check_calendar_effects(df: pd.DataFrame) -> None:
    print("\n  [日历效应] 星期几 / 月份，对同期收益的Kruskal-Wallis检验:")
    dow = df["date"].dt.dayofweek
    groups_dow = [df["log_return"][dow == d].dropna().to_numpy() for d in range(5)]
    stat, p = kruskal(*groups_dow)
    print(f"    星期几: H={stat:.2f}, p={p:.4g}{'  <-- 显著' if p < 0.05 else ''}")
    print("    各星期几日均收益:", df.groupby(dow)["log_return"].mean().round(5).to_dict())

    month = df["date"].dt.month
    groups_month = [df["log_return"][month == m].dropna().to_numpy() for m in range(1, 13)]
    stat, p = kruskal(*groups_month)
    print(f"    月份: H={stat:.2f}, p={p:.4g}{'  <-- 显著' if p < 0.05 else ''}")


def main():
    from ablation.common import load_data  # noqa: E402
    df = load_data().reset_index(drop=True)
    log_return = df["log_return"].fillna(0.0).to_numpy()
    price = df["price"].to_numpy()

    print(f"数据: {len(df)}行, {df['date'].min().date()} ~ {df['date'].max().date()}")

    df["feat_vol_level"] = compute_rolling_percentile(df["realized_vol"].to_numpy(), VOL_LOOKBACK)
    df["feat_skew"] = compute_rolling_skew(log_return, ASYMMETRY_WINDOW)
    df["feat_down_up_ratio"] = compute_downside_upside_ratio(log_return, ASYMMETRY_WINDOW)

    print("\n" + "=" * 70)
    print("候选1: 波动率水平（realized_vol相对自身500天历史的百分位）")
    print("=" * 70)
    check_ic_against_forward_return(df, "feat_vol_level", "vol_level")
    check_ic_against_event_duration(df, "feat_vol_level", "vol_level", price)

    print("\n" + "=" * 70)
    print("候选2a: 滚动偏度 skew(log_return, W=60)")
    print("=" * 70)
    check_ic_against_forward_return(df, "feat_skew", "skew")
    check_ic_against_event_duration(df, "feat_skew", "skew", price)

    print("\n" + "=" * 70)
    print("候选2b: 下跌/上涨波动率之比 (W=60)")
    print("=" * 70)
    check_ic_against_forward_return(df, "feat_down_up_ratio", "down_up_ratio")
    check_ic_against_event_duration(df, "feat_down_up_ratio", "down_up_ratio", price)

    print("\n" + "=" * 70)
    print("候选3: 日历效应")
    print("=" * 70)
    check_calendar_effects(df)


if __name__ == "__main__":
    main()
