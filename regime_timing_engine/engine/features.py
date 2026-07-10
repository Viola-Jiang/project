"""
engine/features.py
====================
§2「数据与特征构建」的特征计算：
  1. 对数收益      r_t = ln(P_t / P_{t-1})
  2. 已实现波动    sigma_t = std(r_{t-w+1:t}), w = 20
  3. 波动归一化收益 z_t = r_t / sigma_t   <- BOCPD 的实际"发射观测"

被 ablation/02_feature_engineering.py 调用。
"""

import numpy as np
import pandas as pd

ROLLING_WINDOW = 20


def preprocess(price_df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    输入: price_df 至少包含 ['date', 'price'] 两列，按日期升序排列
    输出: 新增三列 ['log_return', 'realized_vol', 'z'] 的 DataFrame
    """
    df = price_df.copy().sort_values("date").reset_index(drop=True)
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df["realized_vol"] = df["log_return"].rolling(window=window, min_periods=window).std(ddof=1)
    df["z"] = df["log_return"] / df["realized_vol"]
    return df
