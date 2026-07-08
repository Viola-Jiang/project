"""
engine/bocpd.py
================
对应方法论文档 §3.2「贝叶斯在线变点检测（BOCPD）」+ §3.5 的 hazard 衔接。

核心递归（联合概率形式，而非仅后验，避免归一化项过早损失数值精度）：

  段延续：
    P(r_t=r+1, x_1:t) = P(r_{t-1}=r, x_1:t-1) * pi_t^(r) * (1 - H(r+1))

  段重置（变点）：
    P(r_t=0, x_1:t) = sum_r  P(r_{t-1}=r, x_1:t-1) * pi_t^(r) * H(r+1)

  归一化：
    P(r_t | x_1:t) = P(r_t, x_1:t) / sum_{r_t} P(r_t, x_1:t)

本实现的关键改动（对应文档"HSMM久期升级"）：
  原始 Adams & MacKay (2007) 使用常数 hazard（等价假设几何久期）。
  本引擎的 hazard 由 Step 3 拟合的 NegBinom 久期分布通过
  H(r) = g(r)/P(D>=r) 导出，随 run-length 变化 —— 这就是 S4 阶段
  "HSMM久期升级"在 BOCPD 递归里的具体落地方式。

工程处理：
  - 全程在 log 空间运算（log-sum-exp），避免长序列下的数值下溢。
  - run-length 支持截断（max_run_length），防止数组随 t 无限增长。
  - 每个区制可以有各自的 hazard 函数（若做多区制联合，可扩展为按区制
    加权平均hazard；本Step先实现"单一物理过程、hazard由段龄决定"的
    基础版本，区制身份识别留给 Step 5 的软分配）。

重要数学性质（务必了解，否则容易误用 changepoint_prob 做检测判据）：
  当 hazard 为**常数** h（与 run-length 无关）时，可以证明
      P(r_t=0 | x_1:t) 恒等于 h 本身，与观测数据完全无关！
  证明：记 M = sum_r P(r_{t-1}=r)*pi_t^(r)（边际似然），则
      grow 项之和 = (1-h)*M，reset 项 = h*M，
      归一化后 P(r_t=0|x_1:t) = h*M / [(1-h)*M + h*M] = h，
      数据相关的 M 在分子分母中被约去。
  这意味着：用 P(r_t=0) 单独作为"变点置信度"在常数hazard下是完全无意义的
  统计量（它根本不随数据变化）！哪怕换成年龄相依 hazard，这个退化也只是被
  削弱而非消除（P(r_t=0)的"基线"仍会锚定在 H(1) 附近，波动幅度有限）。
  工程实践中应使用：
    (a) MAP run length 是否骤降至小值（本文件 map_run_length 字段），或
    (b) 期望段龄 E[r_t]（expected_run_length 字段）是否骤降，或
    (c) 累积概率 P(r_t <= k)，k 取一个较小的正整数如 2~3
        （prob_recent_reset 字段，不受上述退化影响，因为它综合了
        grow(r=0..k) 多个真正随数据变化的桶）。
  本引擎在 step() 返回结果中同时提供以上四种诊断量，供下游按需选用。
"""

from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from typing import Callable, Optional

from .emission import NIWConjugateEmission


class BOCPD:
    """
    贝叶斯在线变点检测主引擎。

    使用方式：
        hazard_fn = lambda r: duration_model.hazard(r)   # r 从 1 开始
        bocpd = BOCPD(hazard_fn=hazard_fn, mu0=0, kappa0=1, alpha0=1, beta0=1)
        for x_t in stream:
            result = bocpd.step(x_t)
            # result.run_length_posterior: 当前时刻各 run-length 假设的后验概率
            # result.changepoint_prob: P(r_t=0 | x_1:t)，即"此刻发生变点"的后验概率
            # result.map_run_length: 后验众数对应的 run-length（MAP估计的段龄）
    """

    def __init__(self,
                 hazard_fn: Callable[[int], float],
                 mu0: float = 0.0, kappa0: float = 1.0,
                 alpha0: float = 1.0, beta0: float = 1.0,
                 max_run_length: Optional[int] = None):
        """
        hazard_fn: 函数 r -> H(r)，r 为"若段延续，新的 run-length"（从1开始计数）。
                   即 H(r+1) 对应"当前 run-length 为 r 的假设，下一步以 H(r+1) 概率结束"。
        max_run_length: run-length 假设数组的截断上限，超出部分概率质量归并入
                        最大 run-length 桶（截断近似，避免无限增长；标准BOCPD工程实践）。
        """
        self.hazard_fn = hazard_fn
        self.emission = NIWConjugateEmission.from_nig(mu0, kappa0, alpha0, beta0, max_run_length)
        self.max_run_length = max_run_length

        # log P(r_0 = 0) = 1，即时刻0必然是"新段的第0天"
        self.log_run_length_posterior = np.array([0.0])  # log(1)=0，长度1的数组
        self.t = 0

        # 记录历史，便于事后分析/绘图
        self.history_changepoint_prob = []
        self.history_map_run_length = []
        self.history_run_length_entropy = []

    @property
    def n_hypotheses(self) -> int:
        return len(self.log_run_length_posterior)

    def step(self, x_t: float, hazards_override: Optional[np.ndarray] = None) -> "BOCPDStepResult":
        """
        处理一个新观测 x_t，执行一次完整的 BOCPD 递归更新。

        hazards_override: 若提供，长度必须等于当前假设数 n_hypotheses，用于替代
            内部固定的 hazard_fn —— 这是 Step 5 "区制混合hazard"的接口：
            外部先算出 P(z=k|x_1:t-1) 的区制后验，再用
                h_mix(r) = sum_k P(z=k) * H_k(r+1)
            动态构造一个"融合了区制身份信息"的 hazard 数组，每一步都可以不同，
            比构造时固定的单一 hazard_fn 更贴近 HSMM"每个区制各自hazard"的设定。

        返回本步的诊断结果（当前 run-length 后验、变点概率、MAP段龄、后验熵）。
        """
        # 1. 计算段内预测似然 pi_t^(r)（对当前所有存活假设，即"更新x_t之前"的假设）
        log_pi = self.emission.predictive_logpdf(x_t)  # log pi_t^(r)，长度 = 当前假设数

        # 2. 取出对应的 hazard：假设当前 run-length 为 r（数组下标 r，值为r，因为
        #    我们的假设数组下标本身就代表 run-length：0,1,2,...,n-1）
        current_run_lengths = np.arange(self.n_hypotheses)  # r = 0,1,...,n-1
        if hazards_override is not None:
            assert len(hazards_override) == self.n_hypotheses, (
                f"hazards_override 长度({len(hazards_override)})必须等于当前假设数"
                f"({self.n_hypotheses})"
            )
            hazards = np.asarray(hazards_override)
        else:
            hazards = np.array([self.hazard_fn(r + 1) for r in current_run_lengths])
        hazards = np.clip(hazards, 1e-12, 1.0 - 1e-12)  # 数值保护

        log_joint_prev = self.log_run_length_posterior  # log P(r_{t-1}, x_1:t-1)（未归一化的联合，若上一步已归一化则此处即后验）

        # 3. 段延续：log P(r_t=r+1, x_1:t) = log_joint_prev(r) + log_pi(r) + log(1-H(r+1))
        log_grow = log_joint_prev + log_pi + np.log1p(-hazards)

        # 4. 段重置：log P(r_t=0, x_1:t) = logsumexp_r [ log_joint_prev(r) + log_pi(r) + log(H(r+1)) ]
        log_reset_terms = log_joint_prev + log_pi + np.log(hazards)
        log_reset = logsumexp(log_reset_terms)

        # 5. 拼接新的 run-length 联合分布：[reset, grow...]，长度 = 原长度+1
        log_joint_new = np.concatenate([[log_reset], log_grow])

        # 6. 归一化得后验 P(r_t | x_1:t)
        log_norm = logsumexp(log_joint_new)
        log_posterior_new = log_joint_new - log_norm

        # 7. 截断（若设置了 max_run_length）：超出部分质量并入最后一个桶
        if self.max_run_length is not None and len(log_posterior_new) > self.max_run_length + 1:
            keep = log_posterior_new[: self.max_run_length]
            overflow = logsumexp(log_posterior_new[self.max_run_length:])
            log_posterior_new = np.concatenate([keep, [overflow]])

        self.log_run_length_posterior = log_posterior_new

        # 8. 更新发射模型的充分统计量（对应所有假设吸收 x_t + 新增 r=0 假设）
        self.emission.update(x_t)

        self.t += 1

        posterior = np.exp(self.log_run_length_posterior)
        changepoint_prob = float(posterior[0])
        map_run_length = int(np.argmax(posterior))
        # 后验熵（run-length 分布的不确定性，越大越不确定）
        entropy = float(-np.sum(posterior * np.log(posterior + 1e-300)))
        # 期望段龄：E[r_t]，比 MAP 更平滑，常用于观察"是否正在收缩"
        expected_run_length = float(np.sum(np.arange(len(posterior)) * posterior))
        # 累积概率 P(r_t <= 3)：比单纯 P(r_t=0) 更能反映"数据认为最近很可能刚重启"，
        # 且不受"常数hazard下P(r_t=0)恒等于hazard"这一退化性质影响（详见文档说明）
        k_cutoff = min(3, len(posterior) - 1)
        prob_recent_reset = float(np.sum(posterior[: k_cutoff + 1]))

        self.history_changepoint_prob.append(changepoint_prob)
        self.history_map_run_length.append(map_run_length)
        self.history_run_length_entropy.append(entropy)

        return BOCPDStepResult(
            t=self.t,
            run_length_posterior=posterior,
            changepoint_prob=changepoint_prob,
            map_run_length=map_run_length,
            posterior_entropy=entropy,
            expected_run_length=expected_run_length,
            prob_recent_reset=prob_recent_reset,
        )


class BOCPDStepResult:
    """单步 BOCPD 递归的诊断结果封装。"""

    __slots__ = ["t", "run_length_posterior", "changepoint_prob",
                 "map_run_length", "posterior_entropy",
                 "expected_run_length", "prob_recent_reset"]

    def __init__(self, t, run_length_posterior, changepoint_prob,
                 map_run_length, posterior_entropy,
                 expected_run_length, prob_recent_reset):
        self.t = t
        self.run_length_posterior = run_length_posterior
        self.changepoint_prob = changepoint_prob
        self.map_run_length = map_run_length
        self.posterior_entropy = posterior_entropy
        self.expected_run_length = expected_run_length
        self.prob_recent_reset = prob_recent_reset
