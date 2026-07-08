"""
engine/decision.py
==================
对应方法论文档 §3.7「贝叶斯决策与仓位映射」。

    w_t = [ 区制混合暴露 sum_k P(z=k)*w_k* ]
          × φ( E[剩余|age] )   久期折减
          × ψ( H_t )           不确定性收缩

三个组件逐一说明：

1. 区制混合暴露：用 Step5 的区制后验加权各区制的目标暴露 w_k*。
   w_k* 的标定方式（对应文档"可由区制条件夏普或区制条件（分数）凯利标定"）：
   本实现采用"分数凯利，按最优区制归一化"——
       f_k = mu_k / sigma_k^2   （单区制最优凯利仓位，未分数化前）
       w_k* = clip(f_k / max(f_k, eps), 0, 1)
   即把凯利仓位最高的区制映射为满仓(1.0)，其余按比例缩放，凯利为负（预期
   亏钱）的区制映射为0（空仓/防御）。这是一个可解释、单调、避免"满仓杠杆"
   数值爆炸的简化标定方式，实盘可替换为更严谨的分数凯利或风险平价方案。

2. 久期折减 φ：随"混合预期剩余久期"相对"混合平均久期"的比值单调递减。
   直觉：当前区制越老、预期剩余越短，越应该向中性/防御仓位收缩
   （对应文档"在有利区制中随预期剩余久期收缩而逐步降仓"）。

3. 不确定性收缩 ψ：随 run-length 后验熵增大而向中性仓收缩
   （对应"不确定即减小敞口偏离，思想与分数凯利一致"）。
   中性仓位取 0.5（多头满仓与空仓的中点），可按需调整。
"""

from dataclasses import dataclass
import numpy as np


def calibrate_target_exposures(regime_stats: dict) -> dict:
    """
    regime_stats: {regime_name: {"mu": 日均收益, "sigma": 日波动}}
    返回 {regime_name: w_k*}，凯利仓位最高的区制映射为1.0，其余按比例缩放，
    凯利为负的区制映射为0。
    """
    kelly = {k: v["mu"] / (v["sigma"] ** 2) for k, v in regime_stats.items()}
    max_kelly = max(kelly.values())
    eps = 1e-8
    return {k: float(np.clip(f / (max_kelly + eps), 0.0, 1.0)) for k, f in kelly.items()}


def duration_discount(expected_remaining: float, reference_duration: float,
                       floor: float = 0.3) -> float:
    """
    φ(E[剩余]) = floor + (1-floor) * clip(E[剩余]/reference, 0, 1)
    floor 防止久期折减把仓位压到0（避免过度保守），reference 通常取当前
    区制混合的平均久期，使折减是"相对该区制典型久期"的比例概念。
    """
    if reference_duration <= 0:
        return 1.0
    ratio = np.clip(expected_remaining / reference_duration, 0.0, 1.0)
    return float(floor + (1 - floor) * ratio)


def uncertainty_shrinkage(entropy: float, lam: float = 0.5, neutral: float = 0.5) -> callable:
    """
    返回一个"收缩系数"，但由于收缩是"向中性仓拉近"而非单纯乘法折减，
    这里直接返回收缩强度 psi in [0,1]：
        w_final = neutral + psi * (w_raw - neutral)
    psi = 1/(1+lam*entropy)：熵越大，psi越小，w_final越靠近neutral。
    """
    return 1.0 / (1.0 + lam * entropy)


def apply_uncertainty_shrinkage(w_raw: float, entropy: float, lam: float = 0.5,
                                 neutral: float = 0.5) -> float:
    psi = uncertainty_shrinkage(entropy, lam=lam, neutral=neutral)
    return neutral + psi * (w_raw - neutral)


@dataclass
class RebalanceEngine:
    """
    无交易带 + 事件驱动调仓（对应文档 §5.3）。
    触发条件（任一满足即调仓）：
        1. |w_t - w_held| > delta
        2. 变点概率（prob_recent_reset）越过阈值
        3. 区制 MAP 判定发生翻转
    """
    delta: float = 0.08
    changepoint_threshold: float = 0.5

    def __post_init__(self):
        self.w_held = None
        self.last_map_regime = None
        self.n_rebalances = 0
        self.n_days = 0

    def step(self, w_target: float, prob_recent_reset: float, map_regime: str) -> float:
        self.n_days += 1
        if self.w_held is None:
            # 首日建仓
            self.w_held = w_target
            self.last_map_regime = map_regime
            self.n_rebalances += 1
            return self.w_held

        trigger = (
            abs(w_target - self.w_held) > self.delta
            or prob_recent_reset > self.changepoint_threshold
            or map_regime != self.last_map_regime
        )
        if trigger:
            self.w_held = w_target
            self.n_rebalances += 1
        self.last_map_regime = map_regime
        return self.w_held

    @property
    def turnover_rate(self) -> float:
        return self.n_rebalances / self.n_days if self.n_days > 0 else 0.0
