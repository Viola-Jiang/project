"""
engine/emission.py
==================
对应方法论文档 §3.3「共轭发射与 Student-t 预测分布」。

核心思想：
  BOCPD 需要对每一个存活的 run-length 假设 r（"段已存续 r 步"）分别维护
  一套充分统计量，并给出"段内预测模型"(UPM) 的预测概率
      pi_t^(r) = p(x_t | r_{t-1}=r, x_{(t-r):(t-1)})
  若发射分布取共轭指数族（此处：单变量高斯，未知均值方差 -> NIG共轭先验），
  该预测概率有闭式解——为 Student-t 分布，且可单步 O(1) 摊还更新，无需重算。

数据结构：
  在时刻 t，我们同时维护 t 个（或截断后 R_max 个）"平行宇宙"假设：
    - 假设 r=0：这一步是新变点，重新从先验开始
    - 假设 r=1,2,...,t-1：段延续，已经吸收了 r 个历史观测
  每个假设对应一组 NIG 超参 (mu, kappa, alpha, beta)。

关键公式（对应文档 §3.3）：
  批量闭式（离线，用于正确性校验）：
    kappa_n = kappa_0 + n
    mu_n    = (kappa_0*mu_0 + n*xbar) / kappa_n
    alpha_n = alpha_0 + n/2
    beta_n  = beta_0 + S/2 + kappa_0*n*(xbar-mu_0)^2 / (2*kappa_n)
    其中 xbar 为样本均值，S 为离差平方和。

  单观测增量递归（在线，工程实际使用，等价于上式但每步O(1)）：
    kappa' = kappa + 1
    mu'    = (kappa*mu + x) / kappa'
    alpha' = alpha + 0.5
    beta'  = beta + kappa*(x-mu)^2 / (2*kappa')

  后验预测分布（Student-t，用于计算 pi_t^(r)）：
    x_t ~ Student_t( df=2*alpha, loc=mu, scale=sqrt(beta*(kappa+1)/(alpha*kappa)) )
"""

import numpy as np
from scipy.stats import t as student_t


class NIGConjugateEmission:
    """
    维护"所有存活 run-length 假设"的 NIG 后验超参数组，向量化实现。

    使用方式（典型时间步循环，细节见 BOCPD 主循环 Step 4）：
        emission = NIGConjugateEmission(mu0, kappa0, alpha0, beta0)
        for x_t in stream:
            log_pi = emission.predictive_logpdf(x_t)   # 各 run-length 假设的预测对数似然
            # ... 用 log_pi 驱动 BOCPD 的 run-length 后验递归 ...
            emission.update(x_t)                        # 所有假设吸收 x_t，并新增 r=0 假设
    """

    def __init__(self, mu0: float = 0.0, kappa0: float = 1.0,
                 alpha0: float = 1.0, beta0: float = 1.0,
                 max_run_length: int | None = None):
        """
        mu0, kappa0, alpha0, beta0: NIG 先验超参
        max_run_length: 若设定，run-length 假设数组超过该长度时从尾部截断
                        （对应工程实现中"只保留近期若干个run-length"的截断近似，
                         避免 t 很大时数组无限增长；None 表示不截断）
        """
        self.prior = (mu0, kappa0, alpha0, beta0)
        self.max_run_length = max_run_length

        # 初始时刻（尚未观测到任何数据）：只有 r=0 这一个假设，即先验本身
        self.mu = np.array([mu0], dtype=float)
        self.kappa = np.array([kappa0], dtype=float)
        self.alpha = np.array([alpha0], dtype=float)
        self.beta = np.array([beta0], dtype=float)

    @property
    def n_hypotheses(self) -> int:
        return len(self.mu)

    def update_prior(self, mu0: float, kappa0: float, alpha0: float, beta0: float) -> None:
        """
        动态更新 r=0 假设所使用的先验超参（对应文档 §5.4「参数季度滚动重估」）。

        只影响之后每一步新增的 r=0 假设（见 update() 方法里 self.prior 的用法），
        不改动已经存活的 run-length>0 假设的充分统计量 —— 这正是"在线滤波每日进行、
        参数季度重估"两种节奏可以共存而不冲突的原因：重估只是换了一套"未来新变点
        重新出发时的起点"，不会回头篡改已经吸收过历史观测的假设，因此不引入前视。
        """
        self.prior = (mu0, kappa0, alpha0, beta0)

    def predictive_logpdf(self, x: float) -> np.ndarray:
        """
        计算当前所有存活 run-length 假设下，观测到 x 的对数预测概率（Student-t）。
        返回数组长度 = 当前假设数（= 当前最大可能 run-length + 1）。
        对应文档公式：pi_t^(r) = p(x_t | r_{t-1}=r, x_{(t-r):(t-1)})
        """
        df = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        return student_t.logpdf(x, df=df, loc=self.mu, scale=scale)

    def update(self, x: float) -> None:
        """
        用新观测 x 更新所有假设（段延续：每个假设吸收 x_t），
        并在最前面插入一个全新的 r=0 假设（对应"若此刻是变点，重新从先验开始"）。
        更新后数组长度 +1（除非触发截断）。
        """
        kappa_new = self.kappa + 1.0
        mu_new = (self.kappa * self.mu + x) / kappa_new
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + self.kappa * (x - self.mu) ** 2 / (2.0 * kappa_new)

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

    def posterior_mean_scale(self, r_index: int) -> tuple[float, float]:
        """
        辅助函数：返回指定 run-length 假设（数组下标）当前的后验预测均值与尺度，
        便于诊断和可视化（例如观察"段龄越长，预测越收敛到段内真实均值/方差"）。
        """
        mu = self.mu[r_index]
        kappa = self.kappa[r_index]
        alpha = self.alpha[r_index]
        beta = self.beta[r_index]
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        return float(mu), float(scale)


def batch_nig_posterior(x: np.ndarray, mu0: float, kappa0: float,
                         alpha0: float, beta0: float) -> tuple[float, float, float, float]:
    """
    批量闭式 NIG 后验（离线一次性计算），用于与在线增量递归做正确性校验。
    对应文档 §3.3 的批量公式。
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
    以历史 z 的样本矩（均值、方差）矩匹配标定 NIG 先验超参
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
    sample_var = max(sample_var, 1e-8)  # 数值保护，避免样本方差退化到0

    df = 2.0 * alpha0
    scale_sq = sample_var * (df - 2.0) / df
    beta0 = scale_sq * alpha0 * kappa0 / (kappa0 + 1.0)
    return mu0, kappa0, alpha0, float(beta0)
