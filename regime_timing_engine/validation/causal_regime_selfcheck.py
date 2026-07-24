"""
validation/causal_regime_selfcheck.py
========================================
小实验：因果规则式在线区制识别，准确率如何？以及这个准确率有多少是被
"参照标签颗粒度"本身决定的，而不是在线信号质量本身。

做法：
  参照（近似真值）：engine/zigzag_labeling.zigzag_label_regimes 的离线标签，
                    对 min_days 做网格（42/63/126），代表不同颗粒度的"真值"。
  在线信号：        engine/causal_trend_signal.causal_trend_labels 的纯因果
                    滚动窗口标签，对 window 做网格（21/42/63/126/252）。
  交叉评估：window x min_days 网格，每格算逐日一致率、混淆矩阵、平均检测
  滞后（真实变点后，在线信号翻转到同方向需要等多少天）、在线信号自身的
  换手（标签翻转次数，颗粒度越细换手越高，代表"抓早"的代价）。

运行方式：
  python validation/causal_regime_selfcheck.py
输出：
  outputs/validation/results/causal_regime_grid.csv
  控制台打印透视表
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "outputs" / "validation" / "results"
sys.path.insert(0, str(REPO_ROOT))

from engine.zigzag_labeling import zigzag_label_regimes  # noqa: E402
from engine.causal_trend_signal import causal_trend_labels  # noqa: E402


def load_prices() -> pd.DataFrame:
    prices = (pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["date"])
              .rename(columns={"price": "close"}))
    return prices.sort_values("date").reset_index(drop=True)


def detection_lag(ref_regime: pd.Series, causal_regime: np.ndarray,
                   max_search_window: int = 30) -> tuple[float, float]:
    """
    真实变点（ref_regime_age==1，跳过第一段）后，在线信号第一次翻转为
    "同方向"需要等多少天。这里"同方向"指 causal_regime == 当天的新
    ref_regime（三态一致，不是二元涨跌）。
    """
    ref_regime_age = (ref_regime != ref_regime.shift(1)).cumsum()
    seg_start = ref_regime_age.drop_duplicates().index  # 近似，用变化点代替
    change_idx = [i for i in range(1, len(ref_regime)) if ref_regime.iloc[i] != ref_regime.iloc[i - 1]]

    lags = []
    for cp in change_idx:
        target = ref_regime.iloc[cp]
        detected = False
        for lag in range(max_search_window + 1):
            idx = cp + lag
            if idx >= len(causal_regime):
                break
            if causal_regime[idx] == target:
                lags.append(lag)
                detected = True
                break
        if not detected:
            lags.append(np.nan)
    lags = np.array(lags, dtype=float)
    detected_pct = np.mean(~np.isnan(lags)) * 100
    mean_lag = np.nanmean(lags) if detected_pct > 0 else np.nan
    return detected_pct, mean_lag


def run_grid(df: pd.DataFrame, min_days_grid=(42, 63, 126),
             window_grid=(21, 42, 63, 126, 252)) -> pd.DataFrame:
    rows = []
    for md in min_days_grid:
        ref_labeled = zigzag_label_regimes(df, min_days=md)
        ref_regime = ref_labeled["ref_regime"]
        n_ref_segments = int((ref_regime != ref_regime.shift(1)).sum())

        for w in window_grid:
            causal = causal_trend_labels(df["close"].to_numpy(dtype=float), window=w)
            valid = pd.notna(causal) & pd.notna(ref_regime)
            acc = float((causal[valid.values] == ref_regime[valid].values).mean())
            n_flips = int(np.sum(causal[valid.values][1:] != causal[valid.values][:-1]))
            det_pct, mean_lag = detection_lag(ref_regime, causal)

            rows.append({
                "ref_min_days": md, "n_ref_segments": n_ref_segments,
                "online_window": w, "accuracy": round(acc, 4),
                "n_online_flips": n_flips,
                "detect_pct_within_30d": round(det_pct, 1),
                "mean_detect_lag_days": round(mean_lag, 2) if not np.isnan(mean_lag) else np.nan,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_prices()
    grid = run_grid(df)

    print("=== 因果在线趋势信号 vs 不同颗粒度离线参照标签：准确率网格 ===\n")
    print(grid.to_string(index=False))

    print("\n=== 透视：准确率（行=在线窗口window，列=参照min_days）===")
    print(grid.pivot(index="online_window", columns="ref_min_days", values="accuracy").to_string())

    print("\n=== 透视：平均检测滞后天数（行=在线窗口window，列=参照min_days）===")
    print(grid.pivot(index="online_window", columns="ref_min_days", values="mean_detect_lag_days").to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(RESULTS_DIR / "causal_regime_grid.csv", index=False)
    print(f"\n结果已保存 -> {RESULTS_DIR / 'causal_regime_grid.csv'}")
