"""
engine/emission.py
==================
对应方法论文档 §3.3「共轭发射与 Student-t 预测分布」。

本模块实现的是**通用 D 维 NIW（Normal-Inverse-Wishart）共轭发射模型**。
文档§3.3原文："多变量情形（如收益与波动联合）取 Normal-Inverse-Wishart
（NIW）共轭，后验预测为多元 Student-t。"——D=1 时 NIW 精确退化为文档主线
使用的一元 NIG（Normal-Inverse-Gamma），两者不是"两套东西"，是同一族分布
在不同维度下的特例，因此这里只维护一份通用实现，不再像早期版本那样另外
单独维护一个一元 NIG 类——避免两份数学上等价、只是维度写死的代码同时
存在（本身就是一种冗余）。这样以后如果要试文档 §2/§8 提到的"联合建模
[z_t, sigma_t]"这个可选扩展，直接把维度参数从1改成2即可，不需要另起一个类。

数学定义：
  先验：  μ | Σ ~ N(mu0, Σ/kappa0),  Σ ~ InverseWishart(Psi0, nu0)
  后验更新（吸收单个观测 x，向量化维护所有存活 run-length 假设）：
    kappa' = kappa + 1
    mu'    = (kappa*mu + x) / kappa'
    nu'    = nu + 1
    Psi'   = Psi + kappa/(kappa+1) * (x-mu)(x-mu)^T
  后验预测分布（多元 Student-t）：
    df = nu - D + 1
    scale_matrix = Psi * (kappa+1) / (kappa*df)
    x_new ~ MultivariateT(df=df, loc=mu, shape=scale_matrix)

与一元 NIG(mu0, kappa0, alpha0, beta0) 的对应关系（D=1 时）：
  nu0 = 2*alpha0，Psi0 = 2*beta0（标量）。
  代入可验证：df = nu_n = 2*alpha_n，
              scale_matrix = 2*beta_n*(kappa_n+1)/(kappa_n*2*alpha_n)
                           = beta_n*(kappa_n+1)/(kappa_n*alpha_n)，
  与文档§3.3给出的一元NIG后验预测尺度公式完全一致。
  `NIWConjugateEmission.from_nig()` 就是按这个换算关系提供的便捷构造入口，
  全仓库对一维发射模型的构造统一走这个入口，只在一处维护换算关系。
"""

from __future__ import annotations
import numpy as np
from scipy.stats import multivariate_t, t as student_t


class NIWConjugateEmission:
    """
    维护"所有存活 run-length 假设"的 NIW 后验超参数组，向量化实现。
    D=1 时是文档主线实际使用的形态；D>1 时可用于文档提到的"联合建模"扩展。
    """

    def __init__(self, mu0, kappa0: float, psi0, nu0: float, max_run_length: int | None = None):
        """
        mu0: 长度 D 的先验均值向量（标量会被当作 D=1 自动包成长度1数组）
        kappa0: 先验强度（标量）
        psi0: D x D 先验散布矩阵（标量会被当作 D=1 自动包成 1x1 矩阵）
        nu0: 先验自由度，要求 nu0 > D-1（D=1 时对应 nu0 = 2*alpha0 > 0）
        max_run_length: 同原实现，run-length 假设数组的截断上限
        """
        mu0 = np.atleast_1d(np.asarray(mu0, dtype=float))
        self.dim = len(mu0)
        psi0 = np.asarray(psi0, dtype=float).reshape(self.dim, self.dim)

        self.prior = (mu0, float(kappa0), psi0, float(nu0))
        self.max_run_length = max_run_length

        self.mu = mu0[None, :].copy()                  # (1, D)
        self.kappa = np.array([kappa0], dtype=float)     # (1,)
        self.psi = psi0[None, :, :].copy()               # (1, D, D)
        self.nu = np.array([nu0], dtype=float)            # (1,)

    @classmethod
    def from_nig(cls, mu0: float, kappa0: float, alpha0: float, beta0: float,
                 max_run_length: int | None = None) -> "NIWConjugateEmission":
        """
        D=1 便捷构造：直接用文档§3.3的一元NIG记号 (mu0, kappa0, alpha0, beta0)
        构造，内部换算成 NIW 参数化 nu0=2*alpha0, psi0=[[2*beta0]]（推导见
        模块docstring）。BOCPD 与全部调用方统一走这个入口构造一维发射模型。
        """
        return cls(mu0=[mu0], kappa0=kappa0, psi0=[[2.0 * beta0]], nu0=2.0 * alpha0,
                   max_run_length=max_run_length)

    @property
    def n_hypotheses(self) -> int:
        return len(self.kappa)

    def _scale_matrices(self) -> np.ndarray:
        """每个存活假设的后验预测尺度矩阵，形状 (n_hypotheses, D, D)。"""
        D = self.dim
        df = self.nu - D + 1.0  # (n_hyp,)
        factor = (self.kappa + 1.0) / (self.kappa * df)  # (n_hyp,)
        return self.psi * factor[:, None, None]

    def predictive_logpdf(self, x) -> np.ndarray:
        """
        x: 长度 D 的观测向量（D=1 时可直接传标量）
        返回: 长度 n_hypotheses 的对数似然数组，对应文档公式
              pi_t^(r) = p(x_t | r_{t-1}=r, x_{(t-r):(t-1)})

        D=1 时走向量化快速路径（用 scipy.stats.t 一次性处理全部假设，
        与原一元NIG实现同一数量级的性能）；D>1 时逐个假设调用
        scipy.stats.multivariate_t（BOCPD运行时假设数可达上千，逐个调用
        存在明显性能开销，但目前D>1只是为未来"联合建模"扩展预留的路径，
        暂不是性能敏感场景）。
        """
        if self.dim == 1:
            df = self.nu - 1.0 + 1.0  # D=1: df = nu
            scale = np.sqrt(self.psi[:, 0, 0] * (self.kappa + 1.0) / (self.kappa * df))
            return student_t.logpdf(np.atleast_1d(x)[0], df=df, loc=self.mu[:, 0], scale=scale)

        x = np.atleast_1d(np.asarray(x, dtype=float))
        D = self.dim
        df = self.nu - D + 1.0
        scale_matrices = self._scale_matrices()
        logpdfs = np.empty(self.n_hypotheses)
        for i in range(self.n_hypotheses):
            logpdfs[i] = multivariate_t.logpdf(x, loc=self.mu[i], shape=scale_matrices[i], df=df[i])
        return logpdfs

    def update(self, x) -> None:
        """
        用新观测 x 更新所有假设（段延续：每个假设吸收 x_t），并在最前面插入
        一个全新的 r=0 假设（对应"若此刻是变点，重新从先验开始"）。
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        kappa_new = self.kappa + 1.0
        mu_new = (self.kappa[:, None] * self.mu + x[None, :]) / kappa_new[:, None]
        nu_new = self.nu + 1.0
        diff = x[None, :] - self.mu                      # (n_hyp, D)
        outer = np.einsum("ni,nj->nij", diff, diff)        # (n_hyp, D, D)
        psi_new = self.psi + (self.kappa / kappa_new)[:, None, None] * outer

        mu0, kappa0, psi0, nu0 = self.prior
        self.mu = np.concatenate([mu0[None, :], mu_new], axis=0)
        self.kappa = np.concatenate([[kappa0], kappa_new])
        self.psi = np.concatenate([psi0[None, :, :], psi_new], axis=0)
        self.nu = np.concatenate([[nu0], nu_new])

        if self.max_run_length is not None and self.n_hypotheses > self.max_run_length + 1:
            self.mu = self.mu[: self.max_run_length + 1]
            self.kappa = self.kappa[: self.max_run_length + 1]
            self.psi = self.psi[: self.max_run_length + 1]
            self.nu = self.nu[: self.max_run_length + 1]

    def posterior_mean_scale(self, r_index: int = -1):
        """
        辅助函数：返回指定 run-length 假设（数组下标，默认最新的 r=0 假设）
        当前的后验预测均值与尺度，便于诊断和可视化。D=1 时返回 (float, float)，
        D>1 时返回 (长度D向量, 长度D向量)。
        """
        D = self.dim
        mu = self.mu[r_index]
        df = self.nu[r_index] - D + 1.0
        scale_matrix = self.psi[r_index] * (self.kappa[r_index] + 1.0) / (self.kappa[r_index] * df)
        scale = np.sqrt(np.diag(scale_matrix))
        if D == 1:
            return float(mu[0]), float(scale[0])
        return mu, scale

    def posterior_weighted_mean_scale(self, posterior: np.ndarray):
        """
        对应文档§3.6「run-length 后验加权的发射描述子 [μ̂t, σ̂t]」：
        给定当前 run-length 后验权重（长度需等于 n_hypotheses），返回后验
        加权的 [mu_hat, sigma_hat]。D=1 时返回两个 float，方便直接拼成
        二维特征向量喂给区制软分配；D>1 时返回两个长度D的向量。

        sigma_hat 每一维取该维度后验预测尺度参数（Student-t 的 scale，不是
        标准差）的后验加权平均——与 mu_hat 用的是同一套后验权重，两者是
        "同一个模型的两个输出"，不掺杂任何外部特征。
        """
        mu_hat = np.sum(posterior[:, None] * self.mu, axis=0)  # (D,)
        D = self.dim
        df = self.nu - D + 1.0                                   # (n_hyp,)
        factor = (self.kappa + 1.0) / (self.kappa * df)           # (n_hyp,)
        diag_psi = np.diagonal(self.psi, axis1=1, axis2=2)         # (n_hyp, D)
        scale_per_hyp = np.sqrt(diag_psi * factor[:, None])         # (n_hyp, D)
        sigma_hat = np.sum(posterior[:, None] * scale_per_hyp, axis=0)  # (D,)
        if D == 1:
            return float(mu_hat[0]), float(sigma_hat[0])
        return mu_hat, sigma_hat

    def update_prior(self, mu0: float, kappa0: float, alpha0: float, beta0: float) -> None:
        """
        动态更新 r=0 假设所使用的先验超参（对应文档 §5.4「参数季度滚动重估」）。
        入参沿用一元NIG记号(mu0,kappa0,alpha0,beta0)，内部换算成NIW参数化——
        全仓库只在 from_nig / update_prior 这两处维护"NIG记号<->NIW记号"的
        换算关系，避免两套记号的转换逻辑散落在多处。

        只影响之后每一步新增的 r=0 假设，不改动已经存活的 run-length>0 假设
        的充分统计量，因此不会引入前视。
        """
        self.prior = (np.array([mu0]), float(kappa0), np.array([[2.0 * beta0]]), 2.0 * alpha0)


def batch_nig_posterior(x: np.ndarray, mu0: float, kappa0: float,
                         alpha0: float, beta0: float) -> tuple[float, float, float, float]:
    """
    一元NIG的批量闭式后验（离线一次性计算），用于与在线增量递归做正确性
    校验（pipeline/03_emission_validation.py）。这是独立于 NIWConjugateEmission
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
    返回的仍是(mu0,kappa0,alpha0,beta0)这套NIG记号，直接喂给
    NIWConjugateEmission.from_nig() 或 update_prior() 使用。

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
