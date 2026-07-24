"""
engine/features.py
====================
§2「数据与特征构建」的特征计算：
  1. 对数收益      r_t = ln(P_t / P_{t-1})
  2. 已实现波动    sigma_t = std(r_{t-w+1:t}), w = 20
  3. 波动归一化收益 z_t = r_t / sigma_t   <- BOCPD 的实际"发射观测"

被 ablation/02_feature_engineering.py 调用。
"""

from scipy.stats import percentileofscore

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


def rolling_trend_vol_features(log_returns: np.ndarray, window: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """
    因果滚动窗口描述子 [trend_W, vol_W]，供 engine.calibration.
    estimate_regime_params_rule_based（离线拟合）与 engine.hmm_offline.fit_hmm
    （feature_mode="trend_vol"）共用同一套特征定义，避免各处各写一份导致
    定义不一致。放在 engine/features.py（不依赖包内其它模块）而非
    engine/calibration.py，是为了避免 hmm_offline -> calibration ->
    zigzag_labeling -> regime_labeling -> hmm_offline 的循环导入。

    trend_W = 过去 window 天的累计对数收益（方向 + 幅度）
    vol_W   = 过去 window 天对数收益的标准差

    与 [mu_hat, sigma_hat]（BOCPD 对 z 的后验统计量）的关键差异：这里直接用
    原始 log_return，不先做波动率归一化——诊断发现 sigma_hat 因为 z 被设计
    成方差恒为1，几乎不携带区制信息（几乎处处约等于1），且 mu_hat 的真实
    方向信号会被 run-length 后验加权平均磨平；直接对原始收益取滚动窗口统计量
    可以保留方向与幅度，参见 1.2区制识别.md 的诊断记录。

    第 i 天的值只用 log_returns[:i+1]，不看未来。前 window 天数据不足处，
    用能取到的最早一个有效值向前填充（不产生 NaN、不引入未来信息）。
    """
    s = pd.Series(log_returns)
    min_periods = max(5, window // 2)
    trend = s.rolling(window, min_periods=min_periods).sum().bfill().to_numpy()
    vol = s.rolling(window, min_periods=min_periods).std().bfill().to_numpy()
    vol = np.where(np.isnan(vol) | (vol < 1e-8), 1e-8, vol)
    return trend, vol


def gate_trend_by_significance(trend: np.ndarray, vol: np.ndarray, window: int,
                                t_threshold: float = 1.5) -> np.ndarray:
    """
    用"趋势相对噪声的显著性"（而非趋势的绝对幅度）做硬阈值闸门，把方向不明确
    的天强制置零，方向明确的天保留原始 trend 值。

    背景：诊断1.2区制识别.md记录的S1（HMM+[trend_W,vol_W]）在2015年股灾+
    反弹+熔断这172天里全程判定为同一个（高波动主导的）隐藏状态，导致最大
    回撤跟"什么都不做"一样差——原因是trend_W的绝对值在剧烈震荡期本就容易
    很大（不管涨跌），仅按绝对幅度做阈值挡不住这种"方向噪声大、幅度也大"
    的情形。中邮证券《市场脉搏》报告的思路给了启发：他们判断"是否算趋势"
    用的是收益率相对局部σ的统计显著性（±0.524σ），不是绝对涨跌幅本身。

    做法：t_stat = trend_W / (vol_W * sqrt(window))——若逐日收益是纯噪声，
    W天累计收益的标准差应约为 vol_W*sqrt(W)；t_stat 因此衡量"这段累计收益
    相对同期噪声水平，有多少个标准差"。|t_stat| 达到阈值才保留原始trend，
    否则置零（视为方向不明确，不管当时波动多大）。

    对多组阈值网格搜索（1.2区制识别.md记录）：t_threshold=1.5 时 S1 夏普从
    0.43升到0.712、最大回撤从-48.5%收窄到-15.2%；阈值继续升高会因为
    "判定为不明确"的天数占比过高（>90%），可用信号被过度过滤，绩效反而下降。

    trend/vol: 均为 rolling_trend_vol_features 的输出，长度一致。
    返回: 闸门后的 gated_trend，长度与输入一致。
    """
    t_stat = trend / (vol * np.sqrt(window))
    return np.where(np.abs(t_stat) >= t_threshold, trend, 0.0)


def classify_high_vol(vol: np.ndarray, window: int = 90) -> np.ndarray:
    """
    按"当前波动率是否高于其滚动均值"把波动率分高/低两档，供状态内部按
    波动强度再细分标定用（见 engine.hmm_offline.calibrate_state_exposures_by_vol）。

    背景（1.2区制识别.md诊断）：K=3的HMM状态在"方向"上是干净的（bull/bear
    从不混上涨/下跌天），但状态内部混着"温和趋势"和"剧烈趋势"两种波动强度
    不同的天，二者的凯利分数（mu/sigma^2）并不相同——按状态整体均值算凯利
    会抹掉这个差异。中邮证券《市场脉搏》报告用90天滚动窗口判定"波动率高于/
    低于均值"，这里沿用同一窗口长度。

    vol: 通常是 rolling_trend_vol_features 返回的 vol_W。
    返回: 与 vol 等长的布尔数组，True 表示当天波动率高于滚动均值。
    """
    vol_mean = (pd.Series(vol).rolling(window, min_periods=max(5, window // 2))
                .mean().bfill().to_numpy())
    return vol >= vol_mean


def compute_t_stat(trend: np.ndarray, vol: np.ndarray, window: int) -> np.ndarray:
    """
    t统计量 = trend_W / (vol_W * sqrt(window))——"强度轴"，衡量这段累计收益
    相对同期噪声水平有多少个标准差，见gate_trend_by_significance的推导。
    trend/vol均为rolling_trend_vol_features的输出。
    """
    return trend / (vol * np.sqrt(window))


def compute_rolling_percentile(series: np.ndarray, lookback: int) -> np.ndarray:
    """
    因果滚动百分位：第t天的值，在过去lookback天（含当天）里排第几百分位(0~100)。
    不看未来。前lookback天数据不足处，用能取到的最早有效值向前填充。

    是"透支轴"（compute_extremity_percentile）和"波动率水平"（vol_level，
    直接对realized_vol调用本函数）共用的通用底层实现。
    """
    s = pd.Series(series)
    out = np.full(len(s), np.nan)
    min_obs = max(20, lookback // 10)
    for i in range(len(s)):
        if np.isnan(s.iloc[i]):
            continue
        start = max(0, i - lookback + 1)
        window_vals = s.iloc[start:i + 1].dropna().to_numpy()
        if len(window_vals) < min_obs:
            continue
        out[i] = percentileofscore(window_vals, s.iloc[i], kind="rank")
    return pd.Series(out).bfill().fillna(50.0).to_numpy()


def compute_extremity_percentile(log_returns: np.ndarray, m: int = 60, lookback: int = 500) -> np.ndarray:
    """
    "透支轴"：过去m天的累计对数收益，在过去lookback天（约2年）内同口径滚动值
    的历史分布里的百分位排名。

    选"累计涨幅的分位数"而非"绝对价格水平的分位数"：绝对价格分位数会被长期
    结构性上涨的指数拖累（一直创新高的指数价格分位数会一直贴着100%，分不出
    "这段走势本身有多透支"），累计涨幅分位数是拿"最近这段涨了多少"去跟"历史上
    这类涨法通常有多大"比，更贴近"这段趋势是否走过头"这个真正想问的问题。
    见 1.3区制分类的本质与约束.md 第9、13点。
    """
    cum_ret = pd.Series(log_returns).rolling(m, min_periods=max(5, m // 2)).sum().bfill().to_numpy()
    return compute_rolling_percentile(cum_ret, lookback)


def rolling_standardize(x: np.ndarray, window: int) -> np.ndarray:
    """
    因果滚动z-score：第t天只用[t-window+1, t]的均值/标准差，不看未来。
    用于把量纲不同的多维特征（如t_stat、extremity、vol_level）标准化到
    可比尺度，再喂给聚类/距离度量。样本不足的窗口期用能取到的最早有效值
    向前填充。
    """
    s = pd.Series(x)
    min_periods = max(20, window // 5)
    roll_mean = s.rolling(window, min_periods=min_periods).mean().bfill()
    roll_std = s.rolling(window, min_periods=min_periods).std()
    roll_std = roll_std.replace(0, np.nan).bfill().fillna(1.0)
    return ((s - roll_mean) / roll_std).to_numpy()
