"""
engine/duration.py
==================
对应方法论文档 §3.4「HMM与HSMM」、§3.5「久期建模、hazard函数与预期剩余久期」。

核心思想：
  HMM 的状态自转移概率 a_ii 隐含久期服从几何分布 —— 无记忆、峰值在 d=1，
  意味着"任何区制都以'下一步即结束'为最可能"，与市场经验不符。
  HSMM 为每个区制显式指定久期分布 g_k(d)（此处用负二项，允许过离散、
  峰值不在1），并据此推导"随段龄变化"的 hazard 函数，这正是 HSMM
  相对 HMM 的结构性差异所在。

统一接口（DiscreteDurationModel）：
  给定任意离散久期分布的 pmf 数组，自动推导：
    survival_gt(k)      = P(D > k)
    hazard(r)           = g(r) / P(D >= r) = g(r) / survival_gt(r-1)
    expected_remaining(r) = E[D - r | D > r]
  两个具体实现：
    NegBinomDuration —— 过离散，允许 hazard 随年龄非平凡变化
    GeometricDuration —— 对应 HMM 隐含假设，hazard 恒为常数（无记忆）

离线估参辅助函数（对应文档 §5.1「离线估参」中的"久期分布"一步）：
  extract_segment_durations_from_labels —— 给定任意标签序列，按连续段切出
                                            久期样本（真实数据没有oracle标签，
                                            这里喂入的是 engine.regime_labeling
                                            产出的自动标注参照标签，或模型自己
                                            在线聚类得到的标签）
  fit_regime_duration_models            —— 对指定区制分别拟合 NegBinom（HSMM用）
                                            与 Geometric（HMM隐含假设，仅作对照）
                                            两套久期模型
  这两个函数被 pipeline 中多个阶段复用（久期/hazard验证、区制软分配、仓位映射
  与回测），因此放在 engine 包内而非某个具体的 pipeline 脚本里，避免脚本之间
  互相 import。
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import nbinom, geom


@dataclass
class DiscreteDurationModel:
    """
    通用离散久期分布容器。支持的久期取值为 d = d_min, d_min+1, ..., d_min+len(pmf)-1。
    pmf 数组之外的概率质量视为可忽略（需在构造时检查尾部质量是否够小）。
    """
    pmf: np.ndarray      # pmf[i] = P(D = d_min + i)
    d_min: int = 1
    name: str = "duration"

    def __post_init__(self):
        total_mass = self.pmf.sum()
        if total_mass < 0.999:
            raise ValueError(
                f"[{self.name}] pmf 截断范围内总质量仅 {total_mass:.4f} < 0.999，"
                f"请扩大 max_duration 截断范围"
            )
        # 归一化，消除截断带来的微小质量损失
        self.pmf = self.pmf / total_mass
        # survival_gt[k] = P(D > d_min + k - 1) ... 我们直接建索引到 d 本身，见下方方法
        self._d_values = np.arange(self.d_min, self.d_min + len(self.pmf))
        # tail_from[i] = P(D >= d_values[i]) = sum_{j>=i} pmf[j]
        self._tail_from = np.cumsum(self.pmf[::-1])[::-1]

    def pmf_at(self, d: int) -> float:
        idx = d - self.d_min
        if idx < 0 or idx >= len(self.pmf):
            return 0.0
        return float(self.pmf[idx])

    def survival_ge(self, d: int) -> float:
        """P(D >= d)，即"段至少存续到年龄 d"的概率。"""
        idx = d - self.d_min
        if idx < 0:
            return 1.0
        if idx >= len(self._tail_from):
            return 0.0
        return float(self._tail_from[idx])

    def hazard(self, r: int) -> float:
        """
        H(r) = g(r) / P(D >= r)：段已存续至 r-1、恰在年龄 r 结束的条件概率。
        对应文档公式 H(r) = g(r)/S(r-1)，S(r-1) 定义为 sum_{d>=r} g(d) = P(D>=r)。
        """
        denom = self.survival_ge(r)
        if denom <= 1e-300:
            return 1.0  # 分母趋于0说明该年龄几乎不可能存活到，此时视为必然结束
        return self.pmf_at(r) / denom

    def hazard_curve(self, r_max: int) -> np.ndarray:
        return np.array([self.hazard(r) for r in range(self.d_min, r_max + 1)])

    def expected_remaining(self, r: int) -> float:
        """
        E[剩余 | 年龄=r] = sum_{d>r} (d-r) g(d) / P(D>r)
        """
        idx_r = r - self.d_min
        if idx_r < 0 or idx_r >= len(self.pmf) - 1:
            return 0.0
        d_vals = self._d_values[idx_r + 1:]
        g_vals = self.pmf[idx_r + 1:]
        denom = self.survival_ge(r + 1)
        if denom <= 1e-300:
            return 0.0
        numerator = np.sum((d_vals - r) * g_vals)
        return float(numerator / denom)

    def expected_remaining_curve(self, r_max: int) -> np.ndarray:
        return np.array([self.expected_remaining(r) for r in range(self.d_min, r_max + 1)])

    def mean(self) -> float:
        return float(np.sum(self._d_values * self.pmf))

    def var(self) -> float:
        m = self.mean()
        return float(np.sum((self._d_values - m) ** 2 * self.pmf))


def fit_negbinom_duration(mean: float, var: float, d_min: int = 1,
                           max_duration: int = 3000, name: str = "negbinom") -> DiscreteDurationModel:
    """
    以矩估计法（method of moments）由 (均值, 方差) 反推负二项参数，构造久期分布。
    约定：D - d_min ~ NegBinom(n, p)，即最小久期为 d_min。
    要求 var > (mean - d_min)，即相对于最小久期的过离散条件。
    """
    shifted_mean = mean - d_min
    shifted_var = var
    if shifted_var <= shifted_mean:
        raise ValueError(
            f"[{name}] 方差({shifted_var:.2f})必须大于均值({shifted_mean:.2f})才能构造"
            f"过离散的负二项久期分布；若数据本身接近泊松/欠离散，请改用其他久期族。"
        )
    p = shifted_mean / shifted_var
    n = shifted_mean * p / (1 - p)

    d_range = np.arange(0, max_duration - d_min + 1)
    pmf = nbinom.pmf(d_range, n, p)
    return DiscreteDurationModel(pmf=pmf, d_min=d_min, name=name)


def fit_geometric_duration(mean: float, d_min: int = 1,
                            max_duration: int = 3000, name: str = "geometric") -> DiscreteDurationModel:
    """
    构造与给定均值匹配的几何久期分布 —— 对应 HMM 状态自转移概率隐含的久期假设。
    几何分布：P(D=d) = a^{d-1}(1-a)，d=1,2,...；均值 = 1/(1-a)。
    用于与负二项分布做"HSMM vs HMM隐含假设"的对照。
    """
    if mean <= d_min:
        raise ValueError(f"[{name}] 均值必须大于 d_min")
    # 几何分布定义域从1开始，若d_min>1需要额外平移；此处按文档惯例假设 d_min=1
    a = 1 - 1 / (mean - d_min + 1)
    d_range = np.arange(0, max_duration - d_min + 1)
    pmf = geom.pmf(d_range + 1, 1 - a)  # scipy geom: P(K=k)=(1-a)^{k-1}*a, 需要用 p=1-a 且 k从1开始
    return DiscreteDurationModel(pmf=pmf, d_min=d_min, name=name)


def extract_segment_durations_from_labels(labels) -> pd.DataFrame:
    """
    通用版本：给定任意一串类别标签，按值发生变化的位置切分连续段，返回每段
    的标签与久期。喂入的标签可以是 engine.regime_labeling 产出的自动标注
    参照标签（离线诊断，见 pipeline/04、06），也可以是「§5.1 因果 walk-forward
    估参」里模型自己对历史的聚类结果（见 engine/calibration.py）——两处场景
    共用这一份底层实现，避免逻辑不一致。

    labels: 任意可转换为 pandas.Series 的一维标签序列，按时间顺序排列。
    返回: DataFrame，每行一段，列为 ['regime', 'duration']（列名沿用 'regime'
          是为了让 fit_regime_duration_models 可以不加区分地直接使用）。
    """
    labels = pd.Series(labels).reset_index(drop=True)
    seg_id = (labels != labels.shift(1)).cumsum()
    return pd.DataFrame({"regime": labels, "seg_id": seg_id}).groupby("seg_id").agg(
        regime=("regime", "first"), duration=("regime", "size"))


def fit_regime_duration_models(seg_stats: pd.DataFrame, regime: str, max_duration: int = 2000):
    """
    对指定区制拟合 NegBinom（HSMM，主用）与 Geometric（HMM隐含假设，仅作对照）
    两套久期模型，返回二者及原始段长样本。

    对应文档 §5.1 离线估参流程："粗扫历史估平均段长，标定 Negative-Binomial
    久期参数"。若样本方差不足以支撑过离散拟合（var <= mean），退化为轻微
    过离散的保守缺省方差，避免流水线因单个区制样本不足而中断。
    """
    durations = seg_stats.loc[seg_stats["regime"] == regime, "duration"].values.astype(float)
    n_segments = len(durations)
    mean_d = durations.mean()
    var_d = durations.var(ddof=1) if n_segments > 1 else mean_d * 2

    try:
        nb_model = fit_negbinom_duration(mean_d, var_d, d_min=1, max_duration=max_duration,
                                          name=f"{regime}_negbinom")
    except ValueError:
        nb_model = fit_negbinom_duration(mean_d, mean_d * 1.5, d_min=1, max_duration=max_duration,
                                          name=f"{regime}_negbinom")

    geo_model = fit_geometric_duration(mean_d, d_min=1, max_duration=max_duration, name=f"{regime}_geometric")
    return nb_model, geo_model, durations
