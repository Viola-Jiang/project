"""
engine/hmm_offline.py
=======================
§4.2「S1・离线HMM（前视上界参照）」、§3.4「HMM与HSMM」、§6.4「前视偏差对照实验」。

作为整个系统的上限参考：HMM 看了全部数据（含未来）做状态推断，
能给出一份"如果区制身份已知"的仓位曲线。这个上限用来衡量因果方案
（BOCPD，只能看过去）还剩多少改进空间。

用 [z, log(realized_vol)] 两维特征，EM 算法拟合 GaussianHMM。
由于 EM 对随机初始化敏感（坏种子可能让某个状态的转移概率退化到接近 0，
导致状态逐日翻转），用 n_restarts 个不同种子各跑一遍，取对数似然最高者。

使用两种解码方式：
共用同一个拟合好的 HMM（同一组均值/协方差/转移矩阵），唯一区别是
解码时能看多少数据：

  decode_smoothed — Viterbi 平滑
    看完整段数据后反推每个时点的最可能状态。用来做上限参照，知道后面的行情再判断当时是什么状态。

  decode_filtered — 前向滤波
    只用当前及之前的数据逐日递推 P(z_t | 历史)，推断过程不偷看未来。
    它复用的 HMM 参数（means_, covars_, transmat_）来自全样本 EM。
    参数估计本身不是因果的。因此它和 decode_smoothed 的绩效差异只隔离了
    "推断"前视，不是完整的前视偏差（参数前视仍然在）。

拿到状态序列 → 按状态算收益统计 → 凯利公式标定目标仓位 → 模拟交易绩效。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from hmmlearn.hmm import GaussianHMM

from .decision import calibrate_target_exposures
from .features import (
    rolling_trend_vol_features, gate_trend_by_significance, classify_high_vol,
    compute_t_stat, compute_extremity_percentile, compute_rolling_percentile,
)


def fit_hmm(df: pd.DataFrame, k: int = 3, seed: int = 0, n_restarts: int = 5,
            feature_mode: str = "z_vol", trend_vol_window: int = 21,
            gate_threshold: float | None = 1.5,
            extremity_m: int = 60, extremity_lookback: int = 500, vol_lookback: int = 500):
    """
    EM 估参拟合 GaussianHMM。
    df 允许是整段历史（含"未来"），这是 S1/§6.4 故意要展示的前视行为。

    feature_mode:
      "z_vol"（默认）用 [z, log(realized_vol)] 两维特征，是 S1/
                  regime_labeling/lookahead_contrast 原有的设定；
      "z_only"    只用一维 z，与主链路（emission/BOCPD）的发射观测口径对齐，
                  供 feature_dim_contrast.py 隔离"特征维度不同"这一个变量用；
      "trend_vol" 用 [trend_W, vol_W]（因果滚动窗口的原始收益趋势/波动，
                  见 engine.calibration.rolling_trend_vol_features）——
                  诊断发现 [z, log_vol] 几乎不含方向信息（1.2区制识别.md），
                  "z_vol"/"z_only" 都继承了这个缺陷；"trend_vol" 是给 HMM
                  这个机制本身喂更好的观测，用于测"HMM若拿到含方向的特征、
                  又允许看未来，上限有多高"，不是把HMM换成别的机制。
      "multiaxis" 用 [t_stat, extremity, vol_level] 三维特征（与 S2~S4 现在
                  的区制识别特征同源，见 engine.calibration.
                  estimate_regime_params_multiaxis 与1.3文档第9~13点）——
                  S2~S4已经从"trend_vol"迁移到这套三维特征，S1若继续用旧的
                  "trend_vol"，S1 vs S2的落差就会混入"特征本身不同"这个额外
                  变量，不再是干净的"前视 vs 因果"对照；换成同一套特征，才能让
                  S1重新回到"给HMM这个机制同样的原材料、允许它看未来，上限
                  能到多高"这个干净的问题上。
    trend_vol_window: feature_mode="trend_vol"/"multiaxis" 时的滚动窗口天数，默认21。
    gate_threshold: 仅"trend_vol"用，对 trend_W 做统计显著性闸门
                  （engine.features.gate_trend_by_significance）的t统计量阈值，
                  默认1.5——诊断发现不加闸门时，HMM会在高波动但方向不明确的
                  时期（如2015年股灾+反弹+熔断）整段判定为同一个隐藏状态
                  （被高波动主导，而非真实方向），导致最大回撤跟不做任何
                  区分一样差；闸门把"趋势相对当期噪声是否统计显著"和"波动
                  高低"分离开，避免两者被统计聚类混为一谈（1.2区制识别.md
                  记录了网格搜索：1.5为验证效果最好的阈值）。传 None 关闭闸门
                  （用于对照/调试）。"multiaxis"不需要这个闸门——t_stat本身
                  就是"趋势相对噪声"的显著性统计量，已经内含这层归一化。
    extremity_m/extremity_lookback/vol_lookback: 仅"multiaxis"用，透传给
                  compute_extremity_percentile/compute_rolling_percentile。

    GaussianHMM 的 EM 对随机初始化敏感：单一种子有一定概率收敛到明显更差的
    局部最优（某状态自转移概率退化到接近0，导致状态逐日翻转、没有诊断价值）。
    这对 S1 尤其致命：S1 的角色是"允许前视的信息上限参照"，若这里恰好收敛到差的局部最优，
    S1 可能反而跑不赢因果版本，S1 vs S2 的"前视偏差"量化就失去意义。
    用 n_restarts 个不同种子各拟合一次，按对数似然 model.score(feat) 取最优，
    避免因初始化不走运而产出一份质量很差的模型（同一机制也被 engine/regime_labeling.py
    的自动标注复用）。

    返回: (拟合好的模型, 特征矩阵)
    """
    if feature_mode == "z_vol":
        feat = np.column_stack([df["z"].values, np.log(df["realized_vol"].values)])
    elif feature_mode == "z_only":
        feat = df["z"].values.reshape(-1, 1)
    elif feature_mode == "trend_vol":
        trend, vol = rolling_trend_vol_features(df["log_return"].values, window=trend_vol_window)
        if gate_threshold is not None:
            trend = gate_trend_by_significance(trend, vol, window=trend_vol_window, t_threshold=gate_threshold)
        feat = np.column_stack([trend, vol])
        # trend(~O(0.1))与vol(~O(0.01))尺度相差约一个数量级，GaussianHMM用
        # covariance_type="full"对尺度差异敏感，EM迭代中容易让某状态的协方差
        # 估计退化为病态矩阵；标准化到单位尺度只是数值处理，不改变特征本身
        # 携带的信息（z_vol/z_only两支各维本身尺度接近，不受此问题影响，未改）。
        feat = (feat - feat.mean(axis=0)) / feat.std(axis=0)
    elif feature_mode == "multiaxis":
        log_returns = df["log_return"].values
        trend, vol = rolling_trend_vol_features(log_returns, window=trend_vol_window)
        t_stat = compute_t_stat(trend, vol, trend_vol_window)
        extremity = compute_extremity_percentile(log_returns, m=extremity_m, lookback=extremity_lookback)
        vol_level = compute_rolling_percentile(df["realized_vol"].values, vol_lookback)
        feat = np.column_stack([t_stat, extremity, vol_level])
        # 三维量纲差异更大（t_stat~O(1~3) vs extremity/vol_level~O(0~100)），
        # 标准化理由同"trend_vol"分支——对本次样本算一次标量mean/std，不是
        # engine.calibration.estimate_regime_params_multiaxis里rolling_standardize
        # 那种逐日滚动版，两者不是同一个东西，见1.3阶段性报告标注1。
        feat = (feat - feat.mean(axis=0)) / feat.std(axis=0)
    else:
        raise ValueError(f"未知 feature_mode: {feature_mode}，必须是 'z_vol'/'z_only'/'trend_vol'/'multiaxis'")
    best_model, best_score = None, -np.inf
    for i in range(n_restarts):
        model = GaussianHMM(n_components=k, covariance_type="full", n_iter=200, random_state=seed + i)
        model.fit(feat)
        score = model.score(feat)
        if score > best_score:
            best_model, best_score = model, score
    return best_model, feat


def decode_smoothed(model: GaussianHMM, feat: np.ndarray) -> np.ndarray:
    """
    Viterbi 解码：用全部观测给出最可能的状态路径，即"平滑"。
    对应文档"以全样本估计的两/多状态HMM、平滑状态序列驱动仓位"。
    """
    return model.predict(feat)


def decode_filtered(model: GaussianHMM, feat: np.ndarray) -> np.ndarray:
    """
    前向算法解码：只用当日及之前的观测，逐日递推 P(z_t | x_1:t)。

    递推公式（对数空间）：
      t=1:  alpha_1(i) = P(z_1=i) × b_i(x_1)               ← startprob_
      t>1:  alpha_t(i) = Σ_j [alpha_{t-1}(j) × A_ji] × b_i(x_t)  ← transmat_
    每步归一化后取 argmax 作为该步状态。

    推断过程是因果的（不看未来），但参数 (means_, covars_, transmat_,
    startprob_) 来自全样本 EM 拟合，包含了未来信息。因此 decode_filtered
    不是严格因果的完整方案，只用于隔离"推断前视"这一个变量。
    """
    T, k = len(feat), model.n_components
    log_b = np.column_stack([
        multivariate_normal.logpdf(feat, mean=model.means_[i], cov=model.covars_[i])
        for i in range(k)
    ])
    log_start = np.log(model.startprob_ + 1e-300)
    log_trans = np.log(model.transmat_ + 1e-300)

    log_alpha = np.zeros((T, k))
    log_alpha[0] = log_start + log_b[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + log_b[t]

    log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
    filtered_posterior = np.exp(log_alpha - log_norm)
    return filtered_posterior.argmax(axis=1)


def calibrate_state_exposures(state_seq: np.ndarray, log_returns: np.ndarray, k: int) -> dict:
    """按状态条件收益的分数凯利标定目标暴露（复用 engine.decision 的通用函数）。"""
    state_stats = {}
    for s in range(k):
        mask = state_seq == s
        if mask.sum() == 0:
            continue
        state_stats[f"state_{s}"] = {"mu": float(log_returns[mask].mean()),
                                      "sigma": float(log_returns[mask].std())}
    target_exposures = calibrate_target_exposures(state_stats)
    for name in state_stats:
        state_stats[name]["target_exposure"] = target_exposures[name]
    return state_stats


def calibrate_state_exposures_by_vol(state_seq: np.ndarray, log_returns: np.ndarray,
                                      high_vol: np.ndarray, k: int) -> dict:
    """
    在 calibrate_state_exposures 基础上，把每个状态再按波动强度
    （engine.features.classify_high_vol）切成高/低两档分别标定凯利仓位。

    背景（1.2S1层级优化.md诊断）：K=3的HMM状态在"方向"上完全干净（bull/
    bear 从不混上涨/下跌天），但状态内部混着"温和趋势"和"剧烈趋势"两种
    波动强度不同的天，二者的凯利分数（mu/sigma^2）并不相同——用整个状态的
    平均 mu/sigma 计算凯利，会把这个差异抹平。这里保持状态识别（HMM机制、
    K、bull/sideways/bear 命名）完全不变，只是标定凯利仓位时改按
    (状态, 波动档位) 这个更细的分组重新算 mu/sigma，让同一方向下"猛"和
    "温"两种日子的仓位可以不同。

    返回: {"state_{s}_{high|low}vol": {"mu":..., "sigma":..., "target_exposure":...}}
    某个 (状态,档位) 组合样本量为0时会被跳过（不会出现在返回结果里）。
    """
    state_stats = {}
    for s in range(k):
        for is_high, tag in ((True, "high"), (False, "low")):
            mask = (state_seq == s) & (high_vol == is_high)
            if mask.sum() == 0:
                continue
            state_stats[f"state_{s}_{tag}vol"] = {
                "mu": float(log_returns[mask].mean()),
                "sigma": float(log_returns[mask].std()),
            }
    target_exposures = calibrate_target_exposures(state_stats)
    for name in state_stats:
        state_stats[name]["target_exposure"] = target_exposures[name]
    return state_stats


def fit_offline_hmm_positions(df: pd.DataFrame, k: int = 3, seed: int = 0, n_restarts: int = 5,
                               feature_mode: str = "z_vol", trend_vol_window: int = 21,
                               gate_threshold: float | None = 1.5,
                               vol_split_window: int | None = 90,
                               extremity_m: int = 60, extremity_lookback: int = 500,
                               vol_lookback: int = 500):
    """
    S1 专用入口：拟合HMM + Viterbi(平滑)解码 + 按平滑状态标定暴露。
    feature_mode/trend_vol_window/gate_threshold/extremity_*/vol_lookback 透传给
    fit_hmm，见其文档。

    vol_split_window: 标定凯利仓位时，把每个状态内部再按波动强度切两档
      分别标定（见 calibrate_state_exposures_by_vol 文档），窗口长度传给
      engine.features.classify_high_vol。传 None 关闭，退回按状态整体
      标定的旧行为（calibrate_state_exposures）——feature_mode="multiaxis"时
      vol_level已经是HMM观测的一个独立维度，不需要这个事后补丁，调用方应传None。

    返回: (w_target: 逐日目标仓位Series, state_stats)
      state_stats 始终包含按状态整体统计的 "state_{s}" 键（供打印/对照）；
      vol_split_window 不为 None 时另外附加 "state_{s}_{high|low}vol" 键，
      w_target 按这套更细的标定生成。
    """
    model, feat = fit_hmm(df, k=k, seed=seed, n_restarts=n_restarts,
                           feature_mode=feature_mode, trend_vol_window=trend_vol_window,
                           gate_threshold=gate_threshold, extremity_m=extremity_m,
                           extremity_lookback=extremity_lookback, vol_lookback=vol_lookback)
    state_seq = decode_smoothed(model, feat)
    state_stats = calibrate_state_exposures(state_seq, df["log_return"].values, k)

    if vol_split_window is None:
        w_target = pd.Series([state_stats[f"state_{s}"]["target_exposure"] for s in state_seq],
                              index=df.index, name="w_target")
        return w_target, state_stats

    _, vol = rolling_trend_vol_features(df["log_return"].values, window=trend_vol_window)
    high_vol = classify_high_vol(vol, window=vol_split_window)
    bucket_stats = calibrate_state_exposures_by_vol(state_seq, df["log_return"].values, high_vol, k)
    bucket_names = [f"state_{s}_{'high' if hv else 'low'}vol" for s, hv in zip(state_seq, high_vol)]
    w_target = pd.Series([bucket_stats[name]["target_exposure"] for name in bucket_names],
                          index=df.index, name="w_target")
    state_stats.update(bucket_stats)
    return w_target, state_stats
