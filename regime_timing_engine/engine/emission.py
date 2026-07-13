"""
engine/emission.py
==================
对应方法论文档 §3.3「共轭发射与 Student-t 预测分布」。

一元 NIG（Normal-Inverse-Gamma）共轭发射模型：先验 (mu0, kappa0, alpha0, beta0)，
观测服从条件高斯，后验预测分布为 Student-t（推导见文档§3.3及本模块函数注释）。

之前这里维护过一个通用 D 维 NIW（Normal-Inverse-Wishart）实现，是为文档
§2/§8 提到的"联合建模 [z_t, sigma_t]"这个可选扩展预留的，但目前全仓库
只用到 D=1，也没有计划近期做多维联合建模——不为不存在的需求预留抽象，
现在删掉多维部分，只留一元实现。以后真要做多维联合建模，届时再按需扩展。
"""

from __future__ import annotations
import numpy as np
from scipy.stats import t as student_t


class NIGConjugateEmission:
    """
    维护"所有存活 run-length 假设"的一元 NIG 后验超参数组，向量化实现。

    数学定义：
      先验：  mu | sigma^2 ~ N(mu0, sigma^2/kappa0),  sigma^2 ~ InvGamma(alpha0, beta0)
      后验更新（吸收单个观测 x，向量化维护所有存活 run-length 假设）：
        kappa' = kappa + 1
        mu'    = (kappa*mu + x) / kappa'
        alpha' = alpha + 1/2
        beta'  = beta + kappa/(2*kappa') * (x-mu)^2
      后验预测分布（Student-t）：
        df = 2*alpha
        scale = sqrt(beta*(kappa+1)/(alpha*kappa))
        x_new ~ Student_t(df=df, loc=mu, scale=scale)
    """

    def __init__(self, mu0: float, kappa0: float, alpha0: float, beta0: float,
                 max_run_length: int | None = None):
        self.prior = (float(mu0), float(kappa0), float(alpha0), float(beta0))
        self.max_run_length = max_run_length

        self.mu = np.array([mu0], dtype=float)
        self.kappa = np.array([kappa0], dtype=float)
        self.alpha = np.array([alpha0], dtype=float)
        self.beta = np.array([beta0], dtype=float)

    @property
    def n_hypotheses(self) -> int:
        return len(self.kappa)

    def predictive_logpdf(self, x: float) -> np.ndarray:
        """
        后验预测分布为 Student-t。后验预测分布为【这个具体的数 x_t 像不像这段历史该出现的样子】
        返回: 长度 n_hypotheses 的对数似然数组，对应文档公式
              pi_t^(r) = p(x_t | r_{t-1}=r, x_{(t-r):(t-1)})
        """
        df = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        return student_t.logpdf(x, df=df, loc=self.mu, scale=scale)

    def update(self, x: float) -> None:
        """
        用新观测 x 更新所有假设（段延续：每个假设吸收 x_t），并在最前面插入
        一个全新的 r=0 假设（对应"若此刻是变点，重新从先验开始"）。
        """
        kappa_new = self.kappa + 1.0
        mu_new = (self.kappa * self.mu + x) / kappa_new
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + 0.5 * self.kappa / kappa_new * (x - self.mu) ** 2

        mu0, kappa0, alpha0, beta0 = self.prior
        self.mu = np.concatenate([[mu0], mu_new])
        self.kappa = np.concatenate([[kappa0], kappa_new])
        self.alpha = np.concatenate([[alpha0], alpha_new])
        self.beta = np.concatenate([[beta0], beta_new])

        if self.max_run_length is not None and self.n_hypotheses > self.max_run_length + 1:
            self.mu = self.mu[: self.max_run_length + 1]
            self.kappa = self.kappa[: self.max_run_length + 1]
            self.alpha = self.alpha[: self.max_run_length + 1]
            self.beta = self.beta[: self.max_run_length + 1]

    def posterior_mean_scale(self, r_index: int = -1) -> tuple[float, float]:
        """辅助函数：返回指定 run-length 假设当前的后验预测均值与尺度，便于诊断和可视化。"""
        mu = self.mu[r_index]
        scale = np.sqrt(self.beta[r_index] * (self.kappa[r_index] + 1.0)
                         / (self.alpha[r_index] * self.kappa[r_index]))
        return float(mu), float(scale)

    def posterior_weighted_mean_scale(self, posterior: np.ndarray) -> tuple[float, float]:
        """
        对应文档§3.6「run-length 后验加权的发射描述子 [μ̂t, σ̂t]」：
        给定当前 run-length 后验权重（长度需等于 n_hypotheses），返回后验
        加权的 [mu_hat, sigma_hat]。

        sigma_hat 取该维度后验预测尺度参数（Student-t 的 scale，不是标准差）
        的后验加权平均——与 mu_hat 用的是同一套后验权重，两者是"同一个模型
        的两个输出"，不掺杂任何外部特征。
        """
        mu_hat = float(np.sum(posterior * self.mu))
        scale_per_hyp = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        sigma_hat = float(np.sum(posterior * scale_per_hyp))
        return mu_hat, sigma_hat

    def update_prior(self, mu0: float, kappa0: float, alpha0: float, beta0: float) -> None:
        """
        动态更新 r=0 假设所使用的先验超参（对应文档 §5.4「参数季度滚动重估」）。
        只影响之后每一步新增的 r=0 假设，不改动已经存活的 run-length>0 假设
        的充分统计量，因此不会引入前视。
        """
        self.prior = (float(mu0), float(kappa0), float(alpha0), float(beta0))


def batch_nig_posterior(x: np.ndarray, mu0: float, kappa0: float,
                         alpha0: float, beta0: float) -> tuple[float, float, float, float]:
    """
    一元NIG的批量闭式后验（离线一次性计算），用于与在线增量递归做正确性
    校验（validation/emission_validation.py）。这是独立于 NIGConjugateEmission
    之外、按文档§3.3批量公式重新推导的一份实现——保留两份独立推导互相校验，
    是刻意的正确性检验设计，不是重复造轮子。
    """
    n = len(x)
    if n == 0:
        return mu0, kappa0, alpha0, beta0
    xbar = x.mean()
    S = np.sum((x - xbar) ** 2)

    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * xbar) / kappa_n
    alpha_n = alpha0 + n / 2.0
    beta_n = beta0 + S / 2.0 + (kappa0 * n * (xbar - mu0) ** 2) / (2.0 * kappa_n)
    return mu_n, kappa_n, alpha_n, beta_n


def fit_nig_prior_from_moments(z_hist: np.ndarray, kappa0: float = 1.0,
                                alpha0: float = 3.0) -> tuple[float, float, float, float]:
    """
    以历史 z 的样本矩（均值、方差）矩匹配标定一元NIG先验超参
    （对应文档 §5.1 离线估参「发射先验：以历史 z 矩匹配标定 NIG/NIW 超参」）。

    做法：
      mu0 直接取样本均值。
      kappa0、alpha0 取为固定的"弱先验"强度（默认 kappa0=1 表示先验强度约等于
      1个虚拟观测；alpha0=3 保证先验预测分布（自由度 2*alpha0=6 的 Student-t）
      方差有限，不会退化）。
      再反解 beta0，使得先验预测分布的方差恰好等于样本方差：
        Var[Student_t(df, scale)] = scale^2 * df/(df-2)   (df=2*alpha0>2)
        scale^2 = beta0*(kappa0+1)/(alpha0*kappa0)
      联立可得 beta0，代码中直接给出该反解。

    z_hist: 用于估计矩的历史样本（调用方需保证这段样本严格早于当前决策时点，
            否则会引入前视）。样本量过小时方差估计不稳定，建议至少 60 个观测。
    """
    z_hist = np.asarray(z_hist)
    z_hist = z_hist[~np.isnan(z_hist)]
    if len(z_hist) < 10:
        raise ValueError(f"历史样本量({len(z_hist)})过小，无法稳健估计NIG先验矩")

    mu0 = float(z_hist.mean())
    sample_var = float(z_hist.var(ddof=1))
    sample_var = max(sample_var, 1e-8)

    df = 2.0 * alpha0
    scale_sq = sample_var * (df - 2.0) / df
    beta0 = scale_sq * alpha0 * kappa0 / (kappa0 + 1.0)
    return mu0, kappa0, alpha0, float(beta0)
