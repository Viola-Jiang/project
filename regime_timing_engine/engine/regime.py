"""
engine/regime.py
================
§3.6「区制识别与软分配」。

在线时的软分配描述子严格采用文档原文定义：run-length后验加权的发射描述子
[μ̂t, σ̂t]，两个维度都来自同一个一维NIG发射模型的后验加权估计
（engine.emission.NIGConjugateEmission.posterior_weighted_mean_scale），
不掺杂任何外部特征。这个描述子的计算，由调用方（如ablation/s2~s4的
在线循环）负责，本模块只负责"给定描述子之后怎么做距离度量+软分配"。

────────────────────────────────────────────────────
软分配的构造逻辑：GMM 后验 → RBF softmax
────────────────────────────────────────────────────
离线估参用 KMeans 对历史描述子聚类，得到 K 个原型中心 mu_k。在线时
需要把"当前描述子 f 到各原型的距离"映射为概率 P(z=k | f)。最自然的
做法是假设各区制在特征空间服从高斯分布 N(mu_k, Sigma)，即 GMM：
    P(z=k | f) ∝ P(z=k) · N(f | mu_k, Sigma)
在等先验（P(z=k)=1/K）和共享球形协方差（Sigma=sigma² I，bandwidth
即该 sigma）下，高斯密度的归一化常数与先验在分子分母中约掉，后验退化为
    P(z=k | f) ∝ exp( -d_k² / (2 sigma²) )
其中 d_k² 为 Mahalanobis 或 Wasserstein 距离。对于 Mahalanobis 度量，
这等价于"共享协方差 GMM 的精确 E 步后验"；Wasserstein 度量放宽了共享
协方差假设，允许不同区制有不同的散布尺度。

换言之：RegimeSoftAssigner 不是"定义了"一个 softmax，而是 GMM 在
上述简化条件下后验公式的直接推导结果。参考：Bishop, PRML, Ch.9。
────────────────────────────────────────────────────

核心组件：
  name_clusters_by_return_rank —— 按历史平均收益排名给聚类/状态命名，
    K=3 时统一命名为 bull/sideways/bear。engine/calibration.py（S2~S4的
    KMeans因果估参）实际生产使用；ablation/diag_s1_interpretability.py
    诊断脚本也复用它给S1的HMM状态命名，避免各处各写一份导致命名口径不一致。
  RegimePrototype  —— 单个区制的（特征均值、协方差、久期分布、目标暴露）
  RegimeSoftAssigner —— 给定当前特征描述子，支持两种可插拔的距离度量做softmax 软分配：
      "mahalanobis"：用样本量加权的pooled协方差的逆，是马氏距离的标准定义。
      "wasserstein"：把每个区制原型看成一个高斯分布N(mean,cov)，当前描述子
        看成一个"点"（退化的高斯），用两个高斯间W2距离的闭式解
        d² = ||diff||² + trace(cov_k)——不需要矩阵开方/迭代求解。
    提供 assign_all_metrics() 方便并排对比调试。
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .duration import DiscreteDurationModel

DEFAULT_NAMES_BY_RANK = {3: ["bull", "sideways", "bear"]}


def name_clusters_by_return_rank(cluster_ids: np.ndarray, log_returns: np.ndarray, k: int) -> dict:
    """
    按聚类/状态历史平均收益从高到低排序命名。k=3 时用 bull/sideways/bear
    这三个具体名字（与 engine.plotting.REGIME_COLORS 的键对齐），k!=3 时
    退化为通用的 cluster_rank{i}（rank0=收益最高）。

    返回: {原始簇/状态编号: 命名后的字符串}
    """
    mean_ret_by_cluster = {c: log_returns[cluster_ids == c].mean() for c in range(k)}
    order = sorted(mean_ret_by_cluster, key=lambda c: -mean_ret_by_cluster[c])
    names = DEFAULT_NAMES_BY_RANK.get(k, [f"cluster_rank{i}" for i in range(k)])
    return {old: names[rank] for rank, old in enumerate(order)}


@dataclass
class RegimePrototype:
    name: str
    mean_feature: np.ndarray          # 该区制在特征空间中的中心，如 [mu_z, log_sigma]
    duration_model: DiscreteDurationModel  # 该区制的久期分布（来自 Step3 拟合）
    cov: Optional[np.ndarray] = None  # 该区制成员在原始特征空间下的2x2协方差矩阵
    target_exposure: Optional[float] = None  # 目标暴露 w_k*，留给 Step 6 标定
    n_obs: Optional[int] = None       # 该区制成员数，用于pooled协方差的样本量加权


class RegimeSoftAssigner:
    """
    在线软分配：给定当前特征描述子 f，输出对 K 个区制原型的后验概率
        P(z=k | f) ∝ exp( -d_k^2 / (2*bandwidth^2) )
    dist 由 metric 参数选择。

    构造逻辑：离线 KMeans 给出了原型中心，在线需要把距离转化为概率。
    将各区制看作特征空间中的高斯分布 N(mu_k, Sigma)，则 GMM 后验为
        P(z=k | f) ∝ P(z=k)·N(f | mu_k, Sigma)
    在等先验 + 共享球形协方差 (Sigma = bandwidth^2 * I) 下化简即得到上述
    RBF-softmax 形式（详见模块 docstring 和 assign() 注释）。
    """

    def __init__(self, prototypes: list[RegimePrototype],
                 feature_scale: np.ndarray, bandwidth: float = 1.0,
                 metric: str = "mahalanobis"):
        """
        feature_scale: 每个特征维度的标准化尺度（如整体样本标准差），
                       避免量纲不同的维度（如 mu_z ~ O(0.1) vs log_sigma ~ O(1)）
                       在距离计算中被某一维主导。
        bandwidth: softmax 中 GMM 的共享球形协方差 sigma=bandwidth*I。越小分配
                  越硬（接近one-hot），越大越平滑。等于1.0时在标准化空间中约意味
                  "偏离中心一个标准差时，该区制的权重约为0.61"，是一个合理的默认值。
        metric: "mahalanobis" 或 "wasserstein"，见模块docstring。
        """
        if metric not in ("mahalanobis", "wasserstein"):
            raise ValueError(f"metric 必须是 'mahalanobis' 或 'wasserstein'，收到: {metric}")
        self.prototypes = prototypes
        self.feature_scale = np.asarray(feature_scale, dtype=float)
        self.bandwidth = bandwidth
        self.metric = metric
        self._proto_means = np.array([p.mean_feature for p in prototypes])  # (K, D)

        # 标准化空间下的协方差
        # 映射到标准化空间。scale_outer[i,j] = scale[i] * scale[j]，
        # 等价于 cov_std[i,j] = cov[i,j] / (scale[i] * scale[j])
        D = self._proto_means.shape[1]
        scale_outer = np.outer(self.feature_scale, self.feature_scale)
        covs_std = []
        weights = []
        for p in self.prototypes:
            cov = p.cov if p.cov is not None else np.zeros((D, D))
            covs_std.append(np.asarray(cov, dtype=float) / scale_outer)
            weights.append(p.n_obs if p.n_obs else 1)
        # _covs_std 为 (K, D, D)，两种度量都会用到：
        #   Mahalanobis → 用于加权平均得到 pooled_cov
        #   Wasserstein → 用于取 trace(cov_k) 作为各原型散布惩罚
        self._covs_std = np.array(covs_std)

        # pooled 协方差逆（仅 Mahalanobis 使用）
        # 各区制按样本量 n_obs 加权平均得到共享协方差 Σ_pooled，
        # 求逆后用于 Mahalanobis 距离 d² = diff^T · Σ_pooled^{-1} · diff。
        weights = np.array(weights, dtype=float)
        pooled_cov = np.average(self._covs_std, axis=0, weights=weights)
        if np.linalg.matrix_rank(pooled_cov) < D:
            # 数值保护：协方差退化（如原型只有1个观测）时加微量对角，避免求逆崩溃
            pooled_cov = pooled_cov + np.eye(D) * 1e-6
        self._pooled_cov_inv = np.linalg.inv(pooled_cov)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.prototypes]

    def _sq_distances(self, feature: np.ndarray, metric: str) -> np.ndarray:
        """返回长度K的平方距离数组，顺序与 self.prototypes 一致。"""
        diffs = (self._proto_means - feature) / self.feature_scale  # (K, D)，标准化空间

        if metric == "mahalanobis":
            # 马氏距离标准定义：所有原型共用同一个（样本量加权pooled）协方差的度量张量
            return np.einsum("ki,ij,kj->k", diffs, self._pooled_cov_inv, diffs)

        # wasserstein：两个高斯间W2距离的闭式解退化到"一个点 vs 一个高斯"的特例，
        # 每个原型用自己的协方差（而不是pooled协方差），能反映"这个区制本身分布多分散"
        traces = np.trace(self._covs_std, axis1=1, axis2=2)  # (K,)
        return np.sum(diffs ** 2, axis=1) + traces

    def assign(self, feature: np.ndarray, metric: Optional[str] = None) -> np.ndarray:
        """
        返回长度K的概率数组。

        本质是 GMM 后验（E 步）的特例：
        假设各区制在特征空间服从 N(mu_k, Sigma)，且满足
        - 等先验 P(z=k) = 1/K
        - 共享球形协方差 Sigma = sigma^2 I（bandwidth 即该 sigma）
        则在贝叶斯公式中，高斯密度的归一化常数和先验在分子分母中约掉，
        后验退化为 RBF 核的 softmax 形式：
            P(z=k | f) ∝ exp( -d_k^2 / (2 sigma^2) )
        其中 d_k^2 为特征 f 到原型 k 的平方距离。代码中的 logits 即 -d_k^2 / (2 sigma^2)。
        """
        sq_dists = self._sq_distances(feature, metric or self.metric)
        logits = -sq_dists / (2 * self.bandwidth ** 2)
        logits -= logits.max()  # 数值稳定
        weights = np.exp(logits)
        return weights / weights.sum()

    def assign_all_metrics(self, feature: np.ndarray) -> dict:
        """同时返回两种度量各自的软分配概率，方便并排对比调试。"""
        return {m: self.assign(feature, metric=m) for m in ("mahalanobis", "wasserstein")}

    def mixture_hazard(self, regime_probs: np.ndarray, run_lengths: np.ndarray) -> np.ndarray:
        """
        给定区制后验 regime_probs（长度K）与当前存活的 run-length 数组（长度N，
        值为 0..N-1），返回混合 hazard 数组（长度N）：
            h_mix(r) = sum_k P(z=k) * H_k(r+1)
        用于输入给 BOCPD.step(hazards_override=...)。
        """
        h_mix = np.zeros(len(run_lengths))
        for k, proto in enumerate(self.prototypes):
            if regime_probs[k] < 1e-8:
                continue
            h_k = np.array([proto.duration_model.hazard(r + 1) for r in run_lengths])
            h_mix += regime_probs[k] * h_k
        return h_mix
