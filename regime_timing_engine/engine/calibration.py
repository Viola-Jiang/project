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
  1. 对历史窗口的特征 [z的10日滚动均值, log(已实现波动)] 做 KMeans 无监督聚类
     （不使用真实标签），得到 K 个"数据驱动的区制"。
  2. 按聚类簇的历史平均收益从高到低命名（仅用于可读性，不影响计算）。
  3. 用聚类标签序列（而非真实标签）切分连续段，对每个聚类簇拟合久期模型
     （复用 engine.duration 的通用函数）——duration_family 参数决定拟合哪一族：
       "geometric" ：几何久期（HMM隐含假设，hazard恒为常数），供 S2/S3 使用，
                     确保 S2/S3 阶段还没有引入"久期升级"这个变量。
       "negbinom"  ：负二项久期（HSMM，hazard随年龄变化），供 S4/S5 使用。
     这个开关是 S3→S4 消融能否成立的关键：两者除了 duration_family 不同，
     其余全部一致，绩效落差才能干净地归因于"HSMM久期升级"本身。
  4. 用聚类条件的历史收益均值/方差做分数凯利标定目标暴露（复用
     engine.decision.calibrate_target_exposures）。
  全部计算只读取传入的 hist_df，不接触调用方之外的任何"未来"信息。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .duration import extract_segment_durations_from_labels, fit_regime_duration_models, fit_geometric_duration
from .regime import RegimePrototype, RegimeSoftAssigner
from .decision import calibrate_target_exposures


def estimate_regime_params_causal(hist_df: pd.DataFrame, k: int = 3, max_duration: int = 2000,
                                   kmeans_seed: int = 0, duration_family: str = "negbinom"):
    """
    hist_df: 必须至少包含 ['z', 'realized_vol', 'log_return'] 三列，且严格只
             包含当前决策时点"之前"的观测（由调用方保证，本函数不做时间校验）。
    k: 聚类簇数（区制数）。
    duration_family: "geometric"（S2/S3，hazard恒为常数） 或 "negbinom"（S4/S5，
                      hazard随年龄变化，即HSMM久期升级）。
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

    feat = np.column_stack([
        hist_df["z"].rolling(10, min_periods=1).mean().values,
        np.log(hist_df["realized_vol"].values),
    ])
    feature_scale = np.array([feat[:, 0].std(), feat[:, 1].std()])
    feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)  # 数值保护
    feat_std = feat / feature_scale

    km = KMeans(n_clusters=k, n_init=10, random_state=kmeans_seed)
    cluster_ids = km.fit_predict(feat_std)

    log_returns = hist_df["log_return"].values
    # 按历史平均收益从高到低给簇命名（rank0=收益最高），仅为可读性，不影响后续计算
    mean_ret_by_cluster = {c: log_returns[cluster_ids == c].mean() for c in range(k)}
    order = sorted(mean_ret_by_cluster, key=lambda c: -mean_ret_by_cluster[c])
    name_of = {old: f"cluster_rank{rank}" for rank, old in enumerate(order)}
    named_labels = pd.Series([name_of[c] for c in cluster_ids])

    seg_stats = extract_segment_durations_from_labels(named_labels)

    prototypes, regime_stats = [], {}
    for old_c, name in name_of.items():
        mask = cluster_ids == old_c
        mean_feature = feat[mask].mean(axis=0)

        if duration_family == "negbinom":
            duration_model, _, durations = fit_regime_duration_models(seg_stats, name, max_duration=max_duration)
        else:  # geometric
            durations = seg_stats.loc[seg_stats["regime"] == name, "duration"].values.astype(float)
            duration_model = fit_geometric_duration(durations.mean(), d_min=1, max_duration=max_duration,
                                                      name=f"{name}_geometric")

        prototypes.append(RegimePrototype(name=name, mean_feature=mean_feature, duration_model=duration_model))
        regime_stats[name] = {
            "mu": float(log_returns[mask].mean()),
            "sigma": float(log_returns[mask].std()),
            "n_obs": int(mask.sum()),
            "n_segments": len(durations),
        }

    target_exposures = calibrate_target_exposures(regime_stats)
    for p in prototypes:
        p.target_exposure = target_exposures[p.name]

    assigner = RegimeSoftAssigner(prototypes, feature_scale=feature_scale, bandwidth=1.0)
    return assigner, regime_stats
