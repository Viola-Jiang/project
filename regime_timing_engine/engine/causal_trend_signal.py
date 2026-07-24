"""
engine/causal_trend_signal.py
=================================
在线区制识别的规则式因果实现（对照 engine/zigzag_labeling.py 的离线版）。

背景：zigzag_labeling.py 里的 Zig-Zag 转折点本质上需要事后确认（只有价格
从极值反向波动超过阈值之后，才能确认之前那个点是极值/转折点），因此不能
直接用作在线信号——它和现有 calibration.py 的因果 KMeans 一样，都需要一个
严格只用截至当前历史的版本才能真正上线。

规则：与申万宏源方法同一种思路（滚动窗口的年化收益 + 阈值），但改成纯因果
的滑动窗口，不做任何事后转折点确认：
  第 t 天，用过去 window 天的价格算年化收益 ann_t = (P_t/P_{t-window})^(252/window) - 1
  ann_t >= +min_ann_return  -> bull
  ann_t <= -min_ann_return  -> bear
  否则                      -> sideways
第一个 window 天没有足够历史，标为 NaN（因果warm-up，不引入信息）。

这不是 zigzag 方法的因果版重实现（Zig-Zag 的转折点确认机制本身就是非因果
的，没有"因果版Zig-Zag"这种东西），而是同一套"趋势/震荡"思想的一个独立
因果实现，用于评估：把这套经济含义规则搬到线上、不允许看未来，识别准确率
能做到多少。
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def causal_trend_labels(prices: np.ndarray, window: int = 63,
                         min_ann_return: float = 0.20) -> np.ndarray:
    """
    纯因果滚动窗口趋势标签。第 t 天只用 prices[t-window:t+1]（含当天），
    不使用任何未来价格。返回长度与 prices 相同的 object 数组，
    前 window 天为 None。
    """
    n = len(prices)
    labels = np.array([None] * n, dtype=object)
    if n <= window:
        return labels
    ann = (prices[window:] / prices[:-window]) ** (TRADING_DAYS / window) - 1.0
    labels[window:] = np.where(ann >= min_ann_return, "bull",
                       np.where(ann <= -min_ann_return, "bear", "sideways"))
    return labels


def causal_trend_label_df(df: pd.DataFrame, window: int = 63,
                           min_ann_return: float = 0.20) -> pd.DataFrame:
    """输入含 'close' 列的 df，输出新增 'causal_regime' 列的副本。"""
    out = df.copy()
    out["causal_regime"] = causal_trend_labels(
        df["close"].to_numpy(dtype=float), window=window, min_ann_return=min_ann_return)
    return out
