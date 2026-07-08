"""
engine/hmm_offline.py
=======================
对应方法论文档 §4.2「S1・离线HMM（前视上界参照）」、§3.4「HMM与HSMM」、
以及 §6.4「前视偏差对照实验」。

本模块提供同一个 HMM 的两种解码方式，供两处不同用途：

  decode_smoothed（Viterbi，平滑）——用全部观测（含"未来"）给出最可能的
    状态路径。这是 S1 使用的解码方式，也是文档 §6.4 对照实验里"平滑"的一侧。

  decode_filtered（前向算法，滤波）——只用截至当前时刻的观测递归计算
    P(z_t | x_1:t)，不看任何未来观测，严格因果。这是文档 §6.4 对照实验里
    "滤波"的一侧，也是"BOCPD 天然为滤波形态"这句话里"滤波"的准确含义。

两种解码共用**同一个**已拟合模型（同一组参数）与**同一套**状态->目标暴露
标定（都来自 smoothed 状态的收益统计），确保二者唯一的差异就是"状态估计
时看没看未来"这一件事——这样绩效落差才能被干净地解释为"前视偏差幅度"
（§6.4），而不会和"模型/标定不同"这些别的变量混在一起。

S1（ablation/s1_offline_hmm.py）只用 decode_smoothed；
§6.4 对照实验（ablation/lookahead_contrast.py）把 decode_smoothed 和
decode_filtered 都跑一遍、同台对比。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from hmmlearn.hmm import GaussianHMM

from .decision import calibrate_target_exposures


def fit_hmm(df: pd.DataFrame, k: int = 3, seed: int = 0):
    """
    用 [z, log(realized_vol)] 两维特征、EM 估参拟合 GaussianHMM。
    df 允许是整段历史（含"未来"），这是 S1/§6.4 故意要展示的前视行为。
    返回: (拟合好的模型, 特征矩阵)
    """
    feat = np.column_stack([df["z"].values, np.log(df["realized_vol"].values)])
    model = GaussianHMM(n_components=k, covariance_type="full", n_iter=200, random_state=seed)
    model.fit(feat)
    return model, feat


def decode_smoothed(model: GaussianHMM, feat: np.ndarray) -> np.ndarray:
    """
    Viterbi 解码：用全部观测（前后都看）给出最可能的状态路径，即"平滑"。
    对应文档"以全样本估计的两/多状态HMM、平滑状态序列驱动仓位"。
    """
    return model.predict(feat)


def decode_filtered(model: GaussianHMM, feat: np.ndarray) -> np.ndarray:
    """
    前向算法（forward algorithm）解码：只用 x_1:t 递归计算 P(z_t|x_1:t)，
    不使用任何 t 之后的观测，严格因果，即"滤波"。

    递归（对数空间，避免数值下溢）：
      log alpha_1(i) = log pi_i + log b_i(x_1)
      log alpha_t(i) = logsumexp_j[log alpha_{t-1}(j) + log A_ji] + log b_i(x_t)
    每一步归一化 exp(log_alpha_t) 即得 P(z_t=i | x_1:t)。
    取每步后验的 argmax 作为该步的滤波状态估计。

    这里复用 model 已经用全样本 EM 拟合好的参数（means_/covars_/transmat_/
    startprob_）——§6.4 要对照的是"状态推断时用没用未来观测"，不是"参数
    估计用没用未来数据"（后者已经在 engine/calibration.py 的 walk-forward
    机制里单独处理），两件事分开控制变量。
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


def fit_offline_hmm_positions(df: pd.DataFrame, k: int = 3, seed: int = 0):
    """
    S1 专用入口：拟合HMM + Viterbi(平滑)解码 + 按平滑状态标定暴露。
    返回: (w_target: 逐日目标仓位Series, state_stats)
    """
    model, feat = fit_hmm(df, k=k, seed=seed)
    state_seq = decode_smoothed(model, feat)
    state_stats = calibrate_state_exposures(state_seq, df["log_return"].values, k)
    w_target = pd.Series([state_stats[f"state_{s}"]["target_exposure"] for s in state_seq],
                          index=df.index, name="w_target")
    return w_target, state_stats
