"""
engine/calibration.py
======================
对应方法论文档 §5.1「离线估参」与 §5.4「更新节奏」。

本模块提供**严格因果**的区制参数估计：调用方传入的 hist_df 必须是截至当前
决策时点为止、已经实际发生过的历史数据（不含当前及未来），函数内部也绝不
使用任何"真实区制标签"（那是仿真数据自带的上帝视角信息，只允许在离线诊断/
oracle上限参照场景中使用，见 04、06 脚本）。

与 06/07 脚本里另一套"监督原型"（用真实标签直接算区制中心）的关系：
  监督原型 = oracle 上限参照，用于回答"如果区制身份是已知的，天花板在哪"；
  本模块  = 真正可执行的因果版本，用于回答"只用过去的数据，模型自己能估到
            什么程度"，对应文档 §5.1 "对滚动[r̄,σ]聚类（如KMeans）定K个区制，
            按区制条件夏普标定目标暴露"。
  两者应该分开跑、分开报告，不能把 oracle 版本的绩效当成引擎的真实绩效。

核心函数 estimate_regime_params_causal：
  1. 先跑一遍"影子 BOCPD"过一遍历史窗口（用简单常数hazard，此刻还没有任何
     区制信息，这一步本身就是从零开始识别区制），逐日记录后验加权的
     [μ̂t, σ̂t]——这正是文档§3.6"run-length后验加权的发射描述子[μ̂t,σ̂t]"
     的字面定义，两个维度都来自同一个一维NIG发射模型的后验，不掺杂
     任何外部特征（如原始已实现波动率）。
  2. 对这条 [μ̂t, σ̂t] 轨迹做 KMeans 无监督聚类（不使用真实标签），得到 K 个
     "数据驱动的区制"。
  3. 按聚类簇的历史平均收益从高到低命名（复用 engine.regime.name_clusters_
     by_return_rank，K=3时直接得到bull/sideways/bear，不再走通用cluster_
     rank{i}命名——区制数已固定为3，没必要保留通用K的命名路径）。
  4. 用聚类标签序列（而非真实标签）切分连续段，对每个聚类簇拟合久期模型
     （复用 engine.duration 的通用函数）——duration_family 参数决定拟合哪一族：
       "geometric" ：几何久期（HMM隐含假设，hazard恒为常数），供 S2/S3 使用，
                     确保 S2/S3 阶段还没有引入"久期升级"这个变量。
       "negbinom"  ：负二项久期（HSMM，hazard随年龄变化），供 S4/S5 使用。
     这个开关是 S3→S4 消融能否成立的关键：两者除了 duration_family 不同，
     其余全部一致，绩效落差才能干净地归因于"HSMM久期升级"本身。
  5. 用聚类条件的历史收益均值/方差做分数凯利标定目标暴露（复用
     engine.decision.calibrate_target_exposures），position_bounds 参数
     透传（默认长仓(0,1)，也支持做空/杠杆的更宽区间，见该函数docstring）。
  全部计算只读取传入的 hist_df，不接触调用方之外的任何"未来"信息。

注：早期版本这里直接用"z的滚动均值 + log(原始已实现波动率)"做聚类特征，
第二维绕开了 BOCPD 自己的后验、直接引入外部原始波动率——这样做的区制
判别力更强（原始波动率未被z_t的归一化压制），但与文档§3.6"发射描述子"
的字面定义不符（且和线上逐日软分配时用的描述子不是同一个坐标系）。现已
改为严格对齐文档定义，代价是判别力可能下降（pipeline/03已验证z_t自身
的信号偏弱），这是字面对齐文档后预期会出现的效果，不是bug。

季度walk-forward重估的新旧参数平滑（对应文档§5.4"更新节奏"，缓解直接
切换到新估参数造成的仓位/区制判定跳变）：blend_assigners()。按name
（K=3固定后name稳定是bull/sideways/bear）匹配新旧原型，对
mean_feature/cov/target_exposure以及久期分布的(mean,var)做加权平均，
久期分布按融合后的(mean,var)重新拟合。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .bocpd import BOCPD
from .duration import (
    extract_segment_durations_from_labels, fit_regime_duration_models,
    fit_geometric_duration, fit_negbinom_duration,
)
from .regime import RegimePrototype, RegimeSoftAssigner, name_clusters_by_return_rank
from .decision import calibrate_target_exposures


def compute_posterior_descriptor_trajectory(z_series: np.ndarray, pooled_hazard: float = 0.01):
    """
    跑一遍"影子" BOCPD（不影响任何实际仓位决策），逐日记录后验加权的
    [μ̂t, σ̂t]（对应文档§3.6「发射描述子」的字面定义）。

    hazard 在这一步取一个粗糙的常数：聚类估参本身就是"从零开始识别区制"
    这一步，此刻还没有任何区制信息可用来构造更精细的hazard（HSMM久期升级
    是S4/S5才引入的东西，不在这一步）。

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
    metric: RegimeSoftAssigner 的距离度量，"mahalanobis" 或 "wasserstein"。
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
    # K=3时直接命名为bull/sideways/bear，不再走通用cluster_rank{i}命名
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
