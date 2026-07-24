"""
engine/calibration.py
======================
§5.1「离线估参」与 §5.4「更新节奏」。

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
from scipy.optimize import linear_sum_assignment

from .bocpd import BOCPD
from .duration import (
    extract_segment_durations_from_labels,
    fit_geometric_duration, fit_negbinom_duration,
)
from .regime import RegimePrototype, RegimeSoftAssigner, name_clusters_by_return_rank
from .decision import calibrate_target_exposures
from .zigzag_labeling import zigzag_label_regimes
from .features import (  # noqa: F401  (对外重导出，S2~S4沿用旧导入路径)
    rolling_trend_vol_features, compute_t_stat, compute_extremity_percentile,
    compute_rolling_percentile, rolling_standardize,
)


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
        cov = np.cov(feat[mask], rowvar=False) if mask.sum() > 1 else np.eye(feat.shape[1]) * 1e-6

        durations = seg_stats.loc[seg_stats["regime"] == name, "duration"].values.astype(float)
        mean_d = durations.mean()
        var_d = durations.var(ddof=1) if len(durations) > 1 else mean_d * 2
        duration_model = _refit_duration_like(duration_family, mean_d, var_d, max_duration,
                                               name=f"{name}_{duration_family}")

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
    # 已将各维度缩放至单位标准差，标准化空间中距离约等于"偏离中心几个标准差"。
    assigner = RegimeSoftAssigner(prototypes, feature_scale=feature_scale, bandwidth=1.0, metric=metric)
    return assigner, regime_stats


def estimate_regime_params_rule_based(hist_df: pd.DataFrame, max_duration: int = 2000,
                                       duration_family: str = "negbinom",
                                       position_bounds: tuple = (0.0, 1.0),
                                       metric: str = "mahalanobis", window: int = 21,
                                       zigzag_kwargs: dict | None = None):
    """
    §3.6"可由经济含义固定"这一支的具体实现，与 estimate_regime_params_causal
    （"或由段级发射统计无监督聚类得到"这一支）互为替代，接口一致、可直接互换。

    背景：诊断发现 estimate_regime_params_causal 的 KMeans 在 [mu_hat, sigma_hat]
    上聚类严重退化——sigma_hat 几乎恒为1（z 被设计成方差归一，其尺度统计量
    不携带区制信息），mu_hat 的真实方向信号又被 BOCPD 自身的 run-length 后验
    加权平均磨平，导致聚类后 85.6% 的样本堆进同一个簇，退化为随机划分。

    做法：不做无监督聚类去"发现"原型，改用 engine.zigzag_labeling.
    zigzag_label_regimes（申万宏源趋势/震荡划分的规则式实现，已在 1.1 验证过
    方向可解释）在 hist_df 上先标出 bull/sideways/bear 三段式历史，再以
    rolling_trend_vol_features（因果滚动窗口的收益趋势与波动）在这三段各自
    的均值/协方差作为区制原型中心——原型"发现"的方式从统计聚类换成规则标注，
    其余（RegimeSoftAssigner 的距离-softmax软分配机制、久期拟合、目标暴露
    标定、blend_assigners 新旧平滑）与 estimate_regime_params_causal 完全一致。

    hist_df: 必须包含 ['close', 'log_return'] 两列，且严格只含当前决策时点
             "之前"的观测（由调用方保证）。'close' 供 zigzag_label_regimes
             识别价格转折点用；hist_df 本身整体在决策时点之前，因此 Zig-Zag
             在这段历史内部使用"未来"确认转折点不构成对决策时点的前视。
    window: [trend_W, vol_W] 的因果滚动窗口天数，默认21（约一个月）。
    zigzag_kwargs: 透传给 zigzag_label_regimes 的参数覆盖（转折阈值/最小
                   年化收益/最短时长/斜率比等），默认 None 即用其自身缺省值
                   （转折阈值10%、最小年化收益20%、最短时长63天）。
    返回: (assigner, regime_stats)，接口与 estimate_regime_params_causal 一致。
    """
    if duration_family not in ("geometric", "negbinom"):
        raise ValueError(f"duration_family 必须是 'geometric' 或 'negbinom'，收到: {duration_family}")
    if len(hist_df) < 120:
        raise ValueError(
            f"历史样本量({len(hist_df)})过小，zigzag规则标注需要足够长的历史"
            f"才能切出有意义的趋势段，调用方应在样本不足时跳过重估。"
        )

    zigzag_kwargs = zigzag_kwargs or {}
    labeled = zigzag_label_regimes(hist_df, **zigzag_kwargs)
    named_labels = labeled["ref_regime"]

    trend, vol = rolling_trend_vol_features(hist_df["log_return"].values, window=window)
    feat = np.column_stack([trend, vol])
    feature_scale = np.array([feat[:, 0].std(), feat[:, 1].std()])
    feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)  # 数值保护

    seg_stats = extract_segment_durations_from_labels(named_labels)
    log_returns = hist_df["log_return"].values

    prototypes, regime_stats = [], {}
    for name in ["bull", "sideways", "bear"]:
        mask = (named_labels == name).to_numpy()
        if mask.sum() < 5:
            continue
        mean_feature = feat[mask].mean(axis=0)
        cov = np.cov(feat[mask], rowvar=False) if mask.sum() > 1 else np.eye(feat.shape[1]) * 1e-6

        durations = seg_stats.loc[seg_stats["regime"] == name, "duration"].values.astype(float)
        mean_d = durations.mean() if len(durations) else float(window) * 2.0
        var_d = durations.var(ddof=1) if len(durations) > 1 else mean_d * 2
        duration_model = _refit_duration_like(duration_family, mean_d, var_d, max_duration,
                                               name=f"{name}_{duration_family}")

        prototypes.append(RegimePrototype(name=name, mean_feature=mean_feature, duration_model=duration_model,
                                           cov=cov, n_obs=int(mask.sum())))
        regime_stats[name] = {
            "mu": float(log_returns[mask].mean()),
            "sigma": float(log_returns[mask].std()),
            "n_obs": int(mask.sum()),
            "n_segments": len(durations),
        }

    if len(prototypes) < 3:
        raise ValueError(
            "规则标注在当前历史窗口内未能切出全部三个区制（样本量不足5天），"
            "调用方应跳过本次重估、沿用上一版参数。"
        )

    target_exposures = calibrate_target_exposures(regime_stats, bounds=position_bounds)
    for p in prototypes:
        p.target_exposure = target_exposures[p.name]

    assigner = RegimeSoftAssigner(prototypes, feature_scale=feature_scale, bandwidth=1.0, metric=metric)
    return assigner, regime_stats


def estimate_regime_params_multiaxis(hist_df: pd.DataFrame, k: int = 4, max_duration: int = 2000,
                                      duration_family: str = "negbinom",
                                      position_bounds: tuple = (0.0, 1.0),
                                      metric: str = "mahalanobis",
                                      t_stat_window: int = 21, extremity_m: int = 60,
                                      extremity_lookback: int = 500, vol_lookback: int = 500,
                                      kmeans_seed: int = 0,
                                      prev_assigner: "RegimeSoftAssigner | None" = None,
                                      min_cluster_frac: float = 0.03):
    """
    见 1.3区制分类的本质与约束.md 第9~13点：estimate_regime_params_causal
    （KMeans聚[μ̂,σ̂]）退化严重，estimate_regime_params_rule_based（zigzag+
    [trend_W,vol_W]）绕开了退化问题但只服务收益、久期端测不出区分度。

    这是第三支实现：直接对三维特征[t_stat（强度）, extremity（透支）,
    vol_level（波动率水平）]跑KMeans（K=4），三个特征都独立于BOCPD计算
    （对应第12点"BOCPD与区制分类解耦"），也不依赖zigzag规则标注。
    K=4 + 这三维特征组合，是经过feature_validation/脚本实证验证过的结果
    （原始、未平滑的聚类即同时通过收益两两比较和久期日频卡方检验，且段长
    尾部经零假设置换检验证明不是纯粹的滚动窗口机械假象），K=3同一特征组合
    未通过久期端检验。

    标准化沿用另两支估参函数的简单做法（对本次历史窗口算一次标量std，而非
    rolling_standardize那种逐日滚动标准化）——RegimeSoftAssigner的feature_scale
    机制本身就是"用一个固定尺度去除原始特征再算距离"，在线调用方只需要传入
    原始[t_stat, extremity, vol_level]即可，不需要额外维护"冻结的滚动标准化参数"
    这一层簿记，接口上与estimate_regime_params_rule_based完全一致。

    【与验证阶段的差异，需明确记录】feature_validation/test_tier1_unified_
    clustering.py验证K=4可行时，用的是rolling_standardize（逐日滚动标准化）；
    这里为了跟另两支估参函数、RegimeSoftAssigner的既有接口保持一致，简化成了
    标量标准化——这不是同一个东西，简化后的版本没有被单独重新跑过那一整套
    验证（收益两两比较、久期日频卡方、段长零假设置换检验），当前只是"看起来
    应该也行"，不是"已经验证过"。见1.3阶段性报告标注1。

    hist_df: 必须包含 ['log_return', 'realized_vol'] 两列，且严格只含当前
             决策时点"之前"的观测（由调用方保证）。

    prev_assigner: 上一轮（上季度）的assigner，用于两件事，解决"KMeans每次
      重新拟合、簇编号是任意的、按收益排名匹配可能因为排名偶然对调而张冠
      李戴"这个问题：
      ① 热启动：把prev_assigner各原型的mean_feature（原始量纲）按本次
         feature_scale换算到标准化空间，作为KMeans的初始簇心（而非随机/
         k-means++初始化），让新一轮拟合更容易收敛到"看起来还是同一批簇"
         的结果，从源头减少编号被打乱的情况。
      ② 命名：拟合完成后，不再用name_clusters_by_return_rank（按收益排名，
         排名可能因噪声对调），改成把新簇心（原始量纲）与prev_assigner各
         原型的mean_feature做全局最优的一一匹配（匈牙利算法，
         scipy.optimize.linear_sum_assignment，按欧氏距离），新簇沿用匹配到
         的旧原型的name——直接问"这个新簇的位置像不像上次哪个簇"，而不是
         "收益排名像不像"。
      传None（首次估参，没有上一轮可参照）或prev_assigner的原型数量与k不
      一致时，退回原来的按收益排名命名（name_clusters_by_return_rank）。

    min_cluster_frac: 热启动拟合后，若任一簇的样本占比低于这个阈值（默认3%，
      视为退化/不稳定的拟合结果），放弃热启动结果，退回随机初始化重新拟合
      （同时也放弃距离匹配命名，改回按收益排名命名）——避免热启动在某些
      历史窗口里意外收敛到空簇或极小簇。

    返回: (assigner, regime_stats)，接口与另两支估参函数一致。
    """
    if duration_family not in ("geometric", "negbinom"):
        raise ValueError(f"duration_family 必须是 'geometric' 或 'negbinom'，收到: {duration_family}")
    if len(hist_df) < max(120, k * 30):
        raise ValueError(
            f"历史样本量({len(hist_df)})过小，无法稳健估计{k}个区制的聚类与久期模型，"
            f"调用方应在样本不足时跳过重估、沿用上一版参数。"
        )

    log_returns = hist_df["log_return"].to_numpy()
    trend, vol = rolling_trend_vol_features(log_returns, window=t_stat_window)
    t_stat = compute_t_stat(trend, vol, t_stat_window)
    extremity = compute_extremity_percentile(log_returns, m=extremity_m, lookback=extremity_lookback)
    vol_level = compute_rolling_percentile(hist_df["realized_vol"].to_numpy(), vol_lookback)

    feat = np.column_stack([t_stat, extremity, vol_level])
    feature_scale = feat.std(axis=0)
    feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
    feat_std = feat / feature_scale

    use_prev = prev_assigner is not None and len(prev_assigner.prototypes) == k
    cluster_ids = None
    if use_prev:
        old_centers_raw = np.array([p.mean_feature for p in prev_assigner.prototypes])
        old_centers_std = old_centers_raw / feature_scale
        km_warm = KMeans(n_clusters=k, init=old_centers_std, n_init=1)
        warm_ids = km_warm.fit_predict(feat_std)
        smallest_frac = pd.Series(warm_ids).value_counts(normalize=True).min()
        if smallest_frac >= min_cluster_frac:
            cluster_ids = warm_ids
        else:
            use_prev = False  # 热启动退化成空簇/极小簇，放弃，走下面的随机初始化兜底

    if cluster_ids is None:
        km = KMeans(n_clusters=k, n_init=10, random_state=kmeans_seed)
        cluster_ids = km.fit_predict(feat_std)

    if use_prev:
        new_centers_raw = np.array([feat[cluster_ids == c].mean(axis=0) for c in range(k)])
        old_centers_raw = np.array([p.mean_feature for p in prev_assigner.prototypes])
        cost = np.linalg.norm(new_centers_raw[:, None, :] - old_centers_raw[None, :, :], axis=2)
        new_idx, old_idx = linear_sum_assignment(cost)
        name_of = {int(new_c): prev_assigner.prototypes[int(old_c)].name
                   for new_c, old_c in zip(new_idx, old_idx)}
    else:
        name_of = name_clusters_by_return_rank(cluster_ids, log_returns, k)
    named_labels = pd.Series([name_of[c] for c in cluster_ids])

    seg_stats = extract_segment_durations_from_labels(named_labels)
    # 右截断处理（1.3文档第13点）：数据末尾仍未结束的最后一段剔除，不当完整段计入
    # 久期拟合样本——否则会系统性低估段长（这段几乎必然还没走完，只是被窗口边缘
    # 硬切断）。
    seg_stats = seg_stats.iloc[:-1] if len(seg_stats) > 1 else seg_stats

    prototypes, regime_stats = [], {}
    for old_c, name in name_of.items():
        mask = cluster_ids == old_c
        mean_feature = feat[mask].mean(axis=0)
        cov = np.cov(feat[mask], rowvar=False) if mask.sum() > 1 else np.eye(feat.shape[1]) * 1e-6

        durations = seg_stats.loc[seg_stats["regime"] == name, "duration"].values.astype(float)
        mean_d = durations.mean() if len(durations) else float(t_stat_window) * 2.0
        var_d = durations.var(ddof=1) if len(durations) > 1 else mean_d * 2
        duration_model = _refit_duration_like(duration_family, mean_d, var_d, max_duration,
                                               name=f"{name}_{duration_family}")

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
    """按 duration_family 用融合后的(mean, var)重新拟合久期分布。"""
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
    季度walk-forward重估的新旧参数平滑（§5.4"更新节奏"）：直接切换到
    新一轮估参结果会造成仓位/区制判定的跳变，这里对新旧两版原型做
    new_weight:1-new_weight 的加权平均过渡。

    old_assigner is None（首次重估）时直接返回 new_assigner，不做任何混合。
    按 name 匹配新旧原型。K=3固定后 name 稳定是bull/sideways/bear。

    对 mean_feature/cov/target_exposure 直接做数值加权平均；久期分布没有
    "加权平均两个pmf"这种简单操作，改为取各自的 (mean, var) 做加权平均后，
    用 duration_family 对应的拟合函数（engine.duration.fit_negbinom_duration /
    fit_geometric_duration）重新拟合一个新的 DiscreteDurationModel。
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
