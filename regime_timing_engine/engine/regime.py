"""
engine/regime.py
================
对应方法论文档 §3.6「区制识别与软分配」。

在线时的软分配描述子严格采用文档原文定义：run-length后验加权的发射描述子
[μ̂t, σ̂t]，两个维度都来自同一个一维NIW/NIG发射模型的后验加权估计
（engine.emission.NIWConjugateEmission.posterior_weighted_mean_scale），
不掺杂任何外部特征。这个描述子怎么算出来，由调用方（如ablation/s2~s4的
在线循环）负责，本模块只负责"给定描述子之后怎么做距离度量+软分配"。

核心组件：
  RegimePrototype  —— 单个区制的（特征均值, 久期分布, 目标暴露占位）
  RegimeSoftAssigner —— 给定当前特征描述子，用标准化欧氏距离+高斯核做 softmax 软分配
                        （距离度量、核函数形式、bandwidth均为文档未明确规定的实现选择，
                         详见与用户的讨论记录/待明确问题清单）
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .duration import DiscreteDurationModel


@dataclass
class RegimePrototype:
    name: str
    mean_feature: np.ndarray          # 该区制在特征空间中的中心，如 [mu_z, log_sigma]
    duration_model: DiscreteDurationModel  # 该区制的久期分布（来自 Step3 拟合）
    target_exposure: Optional[float] = None  # 目标暴露 w_k*，留给 Step 6 标定


class RegimeSoftAssigner:
    """
    在线软分配：给定当前特征描述子，输出对 K 个区制原型的后验概率
        P(z=k | 描述子) ∝ exp( -||标准化后描述子 - 原型中心||^2 / (2*bandwidth^2) )
    """

    def __init__(self, prototypes: list[RegimePrototype],
                 feature_scale: np.ndarray, bandwidth: float = 1.0):
        """
        feature_scale: 每个特征维度的标准化尺度（如整体样本标准差），
                       避免量纲不同的维度（如 mu_z ~ O(0.1) vs log_sigma ~ O(1)）
                       在距离计算中被某一维主导。
        bandwidth: softmax 的"温度"参数，越小分配越硬（接近one-hot），越大越平滑。
        """
        self.prototypes = prototypes
        self.feature_scale = np.asarray(feature_scale)
        self.bandwidth = bandwidth
        self._proto_means = np.array([p.mean_feature for p in prototypes])  # (K, D)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.prototypes]

    def assign(self, feature: np.ndarray) -> np.ndarray:
        """返回长度K的概率数组，顺序与 self.prototypes 一致。"""
        diffs = (self._proto_means - feature) / self.feature_scale  # (K, D)
        sq_dists = np.sum(diffs ** 2, axis=1)  # (K,)
        logits = -sq_dists / (2 * self.bandwidth ** 2)
        logits -= logits.max()  # 数值稳定
        weights = np.exp(logits)
        return weights / weights.sum()

    def mixture_hazard(self, regime_probs: np.ndarray, run_lengths: np.ndarray) -> np.ndarray:
        """
        给定区制后验 regime_probs（长度K）与当前存活的 run-length 数组（长度N，
        值为 0..N-1），返回混合 hazard 数组（长度N）：
            h_mix(r) = sum_k P(z=k) * H_k(r+1)
        用于喂给 BOCPD.step(hazards_override=...)。
        """
        h_mix = np.zeros(len(run_lengths))
        for k, proto in enumerate(self.prototypes):
            if regime_probs[k] < 1e-8:
                continue
            h_k = np.array([proto.duration_model.hazard(r + 1) for r in run_lengths])
            h_mix += regime_probs[k] * h_k
        return h_mix
