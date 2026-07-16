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


def fit_hmm(df: pd.DataFrame, k: int = 3, seed: int = 0, n_restarts: int = 5,
            feature_mode: str = "z_vol"):
    """
    EM 估参拟合 GaussianHMM。
    df 允许是整段历史（含"未来"），这是 S1/§6.4 故意要展示的前视行为。

    feature_mode: "z_vol"（默认）用 [z, log(realized_vol)] 两维特征，是 S1/
                  regime_labeling/lookahead_contrast 原有的设定；"z_only" 只用
                  一维 z，与主链路（emission/BOCPD）的发射观测口径对齐，供
                  feature_dim_contrast.py 隔离"特征维度不同"这一个变量用
                  （S1比主链路多看了log(realized_vol)这一维，S1 vs S2 的
                  落差里混杂了前视和特征维度两个变量，此前没有实验单独
                  隔离过后者）。

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
    else:
        raise ValueError(f"未知 feature_mode: {feature_mode}，必须是 'z_vol' 或 'z_only'")
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


def fit_offline_hmm_positions(df: pd.DataFrame, k: int = 3, seed: int = 0, n_restarts: int = 5):
    """
    S1 专用入口：拟合HMM + Viterbi(平滑)解码 + 按平滑状态标定暴露。
    返回: (w_target: 逐日目标仓位Series, state_stats)
    """
    model, feat = fit_hmm(df, k=k, seed=seed, n_restarts=n_restarts)
    state_seq = decode_smoothed(model, feat)
    state_stats = calibrate_state_exposures(state_seq, df["log_return"].values, k)
    w_target = pd.Series([state_stats[f"state_{s}"]["target_exposure"] for s in state_seq],
                          index=df.index, name="w_target")
    return w_target, state_stats
