"""
engine/calibration.py
======================
对应方法论文档 §5.1「离线估参」与 §5.4「更新节奏」。

离线区制参数估计。给定一段历史行情，输出可直接用于在线推理的
RegimeSoftAssigner（包含各区制的原型中心、协方差、久期模型、目标仓位）。

在线推理时，RegimeSoftAssigner 需要知道"每个区制长什么样"——描述子中心
在哪、散布多大、区制通常持续多久、该配多少仓。这些信息必须从历史数据中
估计，且只能用当前时点之前的数据（严格因果，无未来信息）。

输入:  hist_df（含 z 列和 log_return 列，截至当前时点的历史窗口）
输出:  (RegimeSoftAssigner, regime_stats)

 ① 临时 BOCPD → 把 z 序列翻译成 [μ̂, σ̂] 轨迹
    新建一个独立的 BOCPD 实例（常数 hazard=0.01），从头逐日喂入历史 z，
    记录每一步的 NIG 后验加权 [μ̂t, σ̂t]。

 ② KMeans 聚类 → 发现区制
    将 [μ̂, σ̂] 两维各自除以标准差做标准化，然后 KMeans（默认 K=3）。
    得到 K 个簇，每簇有一个特征中心和一个协方差。

 ③ 命名 + 久期拟合
    按簇内对数收益均值从高到低 → bull / sideways / bear。
    用聚类标签序列切出各区制的连续段，拟合久期分布：
      "geometric" → 常数 hazard（几何分布），对应 HMM 假设
      "negbinom"  → 年龄相依 hazard（负二项），对应 HSMM 升级
    这是消融实验的核心开关：两者其余步骤完全一致，绩效差异可干净归因于
    久期建模本身。

 ④ 凯利标定 → 目标仓位 w_k*
    对每个区制用簇内收益 μ 和 σ 算凯利仓位 f = μ/σ²，最大 f 映射到
    仓位区间上界 hi，其余按比例缩放。默认 (lo, hi) = (0, 1) 纯多头，
    也可传 (-1, 2) 等多空/杠杆区间。

 ⑤ 打包 → RegimeSoftAssigner
    将原型中心、协方差、久期模型、目标仓位、特征标准化尺度一并打包。
    在线时即可对新的 [μ̂, σ̂] 做 GMM-softmax 软分配。


第 ① 步不能借用在线主循环的 BOCPD：

在线 BOCPD 的 hazard 是由当前 assigner 构造的 HSMM 混合 hazard。
如果用它跑历史 [μ̂, σ̂] 然后聚类得到新 assigner，就形成了循环：
  assigner(t−1) → hazard → BOCPD后验 → [μ̂,σ̂] → KMeans → assigner(t)
新 assigner 会受到旧 assigner 的"污染"——旧版区制结构通过 hazard 渗透
进 [μ̂, σ̂]，聚类结果就不再是纯粹的"数据在说什么"。

因此第 ① 步独立新建 BOCPD，用常数 hazard=0.01——不给它任何区制先验，
让 [μ̂, σ̂] 轨迹只反映 z 序列自身的结构。KMeans 就能无偏地发现区制。

此外 KMeans 需要整条 [μ̂, σ̂] 轨迹（每个历史时点一个点），而在线 BOCPD
只保留当前后验状态，不存储历史轨迹。

────────────────────────────────────────────────────
参数平滑 blend_assigners

季度重估时直接切换新参数会导致仓位跳变。blend_assigners() 对新旧两版
按 new_weight:(1-new_weight) 混合（默认 7:3）：
  - mean_feature、cov、target_exposure → 直接数值加权
  - 久期分布 → 取各自的 (mean, var) 加权后重新拟合（PMF 不能直接加权）
  - feature_scale → 加权平均
按 name（bull/sideways/bear）匹配新旧原型，不受 KMeans 簇编号影响。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .bocpd import BOCPD
from .duration import (
    extract_segment_durations_from_labels, fit_geometric_duration, fit_negbinom_duration,
)
from .regime import RegimePrototype, RegimeSoftAssigner, name_clusters_by_return_rank
from .decision import calibrate_target_exposures


def compute_posterior_descriptor_trajectory(z_series: np.ndarray, pooled_hazard: float = 0.01):
    """
    跑一遍临时 BOCPD（不影响任何实际仓位决策），逐日记录后验加权的[μ̂t, σ̂t]。

    hazard 在这一步取一个粗糙的常数，不给它任何区制先验，让 [μ̂, σ̂] 轨迹只反映
    z 序列自身的结构。KMeans 就能无偏地发现区制。

    返回: (mu_hats, sigma_hats)，长度均为 len(z_series)。
    """
    bocpd = BOCPD(hazard_fn=lambda r: pooled_hazard, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    mu_hats = np.empty(len(z_series))
    sigma_hats = np.empty(len(z_series))
    for i, z_t in enumerate(z_series):
        posterior_prev = np.exp(bocpd.log_run_length_posterior)
        mu_hats[i], sigma_hats[i] = bocpd.emission.posterior_weighted_mean_scale(posterior_prev)
        bocpd.step(z_t)
    return mu_hats, sigma_hats


def estimate_regime_params_causal(hist_df: pd.DataFrame, k: int = 3, max_duration: int = 2000,
                                   kmeans_seed: int = 0, duration_family: str = "negbinom",
                                   position_bounds: tuple = (0.0, 1.0), metric: str = "mahalanobis"):
    """
    hist_df: 必须至少包含 ['z', 'log_return'] 两列，且严格只包含当前决策
             时点"之前"的观测（由调用方保证，本函数不做时间校验）。
    k: 聚类簇数（区制数），固定为3时聚类结果直接命名为bull/sideways/bear。
    duration_family: "geometric"（S2/S3，hazard恒为常数） 或 "negbinom"（S4/S5，
                      hazard随年龄变化，即HSMM久期升级）。
    position_bounds: 目标暴露的(lo, hi)区间，透传给 engine.decision.
                      calibrate_target_exposures，默认长仓(0,1)。
    metric: RegimeSoftAssigner 软分配的距离度量，"mahalanobis" 或 "wasserstein"。
    返回:
      assigner: RegimeSoftAssigner，可直接喂给在线 BOCPD 循环
      regime_stats: {cluster_name: {"mu":..., "sigma":..., "n_obs":...}}，用于日志/诊断
    """
    if duration_family not in ("geometric", "negbinom"):
        raise ValueError(f"duration_family 必须是 'geometric' 或 'negbinom'，收到: {duration_family}")

    if len(hist_df) < max(60, k * 20):
        raise ValueError(
            f"历史样本量({len(hist_df)})过小，无法稳健估计{k}个区制的聚类与久期模型，"
            f"调用方应在样本不足时跳过重估、沿用上一版参数或走保守缺省仓位。"
        )

    mu_hats, sigma_hats = compute_posterior_descriptor_trajectory(hist_df["z"].values)
    feat = np.column_stack([mu_hats, sigma_hats])
    feature_scale = np.array([feat[:, 0].std(), feat[:, 1].std()])
    feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)  # 数值保护
    feat_std = feat / feature_scale

    km = KMeans(n_clusters=k, n_init=10, random_state=kmeans_seed)
    cluster_ids = km.fit_predict(feat_std)

    log_returns = hist_df["log_return"].values
    # K=3时直接命名为bull/sideways/bear
    name_of = name_clusters_by_return_rank(cluster_ids, log_returns, k)
    named_labels = pd.Series([name_of[c] for c in cluster_ids])

    seg_stats = extract_segment_durations_from_labels(named_labels)

    prototypes, regime_stats = [], {}
    for old_c, name in name_of.items():
        mask = cluster_ids == old_c
        mean_feature = feat[mask].mean(axis=0)
        # 协方差在原始特征空间下计算（不是feat_std），RegimeSoftAssigner内部
        # 自己按feature_scale标准化——全仓库只在一处维护"标准化"这件事
        cov = np.cov(feat[mask], rowvar=False) if mask.sum() > 1 else np.eye(feat.shape[1]) * 1e-6

        if duration_family == "negbinom":
            duration_model, _, durations = fit_regime_duration_models(seg_stats, name, max_duration=max_duration)
        else:  # geometric
            durations = seg_stats.loc[seg_stats["regime"] == name, "duration"].values.astype(float)
            duration_model = fit_geometric_duration(durations.mean(), d_min=1, max_duration=max_duration,
                                                      name=f"{name}_geometric")

        prototypes.append(RegimePrototype(name=name, mean_feature=mean_feature, duration_model=duration_model,
                                           cov=cov, n_obs=int(mask.sum())))
        regime_stats[name] = {
            "mu": float(log_returns[mask].mean()),
            "sigma": float(log_returns[mask].std()),
            "n_obs": int(mask.sum()),
            "n_segments": len(durations),
        }

    target_exposures = calibrate_target_exposures(regime_stats, bounds=position_bounds)
    for p in prototypes:
        p.target_exposure = target_exposures[p.name]

    # bandwidth=1.0 对应 GMM 的共享球形协方差 sigma=1.0*I。由于 feature_scale
    # 已将各维度缩放至单位标准差，标准化空间中距离约等于"偏离中心几个标准差"，
    # 此设置是 GMM 后验 softmax 的自然默认值（详见 engine/regime.py 模块 docstring）。
    assigner = RegimeSoftAssigner(prototypes, feature_scale=feature_scale, bandwidth=1.0, metric=metric)
    return assigner, regime_stats


def _refit_duration_like(duration_family: str, mean: float, var: float,
                          max_duration: int, name: str):
    """按 duration_family 用融合后的(mean, var)重新拟合久期分布，供 blend_assigners 使用。"""
    if duration_family == "negbinom":
        try:
            return fit_negbinom_duration(mean, var, d_min=1, max_duration=max_duration, name=name)
        except ValueError:
            return fit_negbinom_duration(mean, mean * 1.5, d_min=1, max_duration=max_duration, name=name)
    return fit_geometric_duration(mean, d_min=1, max_duration=max_duration, name=name)


def blend_assigners(new_assigner: RegimeSoftAssigner, old_assigner: "RegimeSoftAssigner | None",
                     duration_family: str, new_weight: float = 0.7, max_duration: int = 2000
                     ) -> RegimeSoftAssigner:
    """
    季度walk-forward重估的新旧参数平滑（对应文档§5.4"更新节奏"）：直接切换到
    新一轮估参结果会造成仓位/区制判定的跳变，这里对新旧两版原型做
    new_weight:1-new_weight 的加权平均过渡。

    old_assigner is None（首次重估，还没有"旧版"可言）时直接返回 new_assigner，
    不做任何混合。按 name 匹配新旧原型——K=3固定后 name 稳定是bull/sideways/
    bear（见 engine.regime.name_clusters_by_return_rank），不会因为某次KMeans
    聚类簇编号打乱而对错对象。

    对 mean_feature/cov/target_exposure 直接做数值加权平均；久期分布没有
    "加权平均两个pmf"这种简单操作，改为取各自的 (mean, var) 做加权平均后，
    用 duration_family 对应的拟合函数（engine.duration.fit_negbinom_duration /
    fit_geometric_duration）重新拟合一个新的 DiscreteDurationModel——复用
    现有拟合函数，不新写久期数学。
    """
    if old_assigner is None:
        return new_assigner

    old_by_name = {p.name: p for p in old_assigner.prototypes}
    blended_protos = []
    for new_p in new_assigner.prototypes:
        old_p = old_by_name.get(new_p.name)
        if old_p is None:
            blended_protos.append(new_p)
            continue

        w = new_weight
        mean_feature = w * new_p.mean_feature + (1 - w) * old_p.mean_feature
        cov = w * new_p.cov + (1 - w) * old_p.cov
        target_exposure = w * new_p.target_exposure + (1 - w) * old_p.target_exposure
        n_obs = new_p.n_obs

        new_mean, new_var = new_p.duration_model.mean(), new_p.duration_model.var()
        old_mean, old_var = old_p.duration_model.mean(), old_p.duration_model.var()
        blend_mean = w * new_mean + (1 - w) * old_mean
        blend_var = w * new_var + (1 - w) * old_var
        duration_model = _refit_duration_like(duration_family, blend_mean, blend_var,
                                               max_duration, name=f"{new_p.name}_{duration_family}_blended")

        blended_protos.append(RegimePrototype(name=new_p.name, mean_feature=mean_feature, cov=cov,
                                               duration_model=duration_model,
                                               target_exposure=target_exposure, n_obs=n_obs))

    feature_scale = new_weight * new_assigner.feature_scale + (1 - new_weight) * old_assigner.feature_scale
    return RegimeSoftAssigner(blended_protos, feature_scale=feature_scale,
                               bandwidth=new_assigner.bandwidth, metric=new_assigner.metric)
