"""
engine
======
全天候指数仓位择时框架的核心可复用模块。

────────────────────────────────────────────────────
模块一览
────────────────────────────────────────────────────
  emission       — NIG 共轭发射模型、Student-t 预测分布、先验标定
  duration       — 离散久期分布、hazard 函数、分段久期提取与拟合
  bocpd          — 贝叶斯在线变点检测主递归引擎
  regime         — 区制原型与 softmax 软分配
  decision       — 凯利仓位标定、久期折减、不确定性收缩、调仓引擎
  calibration    — 严格因果的 walk-forward 区制参数季度重估
  evaluation     — 回测绩效指标（夏普、最大回撤、Deflated Sharpe、BH-FDR 等）
  hmm_offline    — 离线 HMM（前视上界参照），含 Viterbi 平滑与 forward 滤波两种解码
  features       — 原始数据预处理：对数收益 → z-score → 已实现波动率
  zigzag_labeling — 规则式参照标签（Zig-Zag+Binseg），生产实际使用，产出 ref_regime
  regime_labeling — 仅剩 _merge_short_segments 通用短段合并工具（历史上的
                    HMM自动标注 auto_label_regimes 已因无调用方被移除，
                    ref_regime 现由 zigzag_labeling 产出）
  plotting       — 中文字体加载、区制背景着色等绘图辅助

────────────────────────────────────────────────────
结构约定
────────────────────────────────────────────────────
本包是框架的中枢。ablation/ 和 validation/ 两个目录下的所有脚本单向依赖本包。
"""

# ---------- emission ----------
from .emission import (
    NIGConjugateEmission,
    batch_nig_posterior,
    fit_nig_prior_from_moments,
)

# ---------- duration ----------
from .duration import (
    DiscreteDurationModel,
    extract_segment_durations_from_labels,
    fit_geometric_duration,
    fit_negbinom_duration,
    fit_regime_duration_models,
)

# ---------- bocpd ----------
from .bocpd import (
    BOCPD,
    BOCPDStepResult,
)

# ---------- regime ----------
from .regime import (
    RegimePrototype,
    RegimeSoftAssigner,
    name_clusters_by_return_rank,
)

# ---------- decision ----------
from .decision import (
    RebalanceEngine,
    apply_uncertainty_shrinkage,
    calibrate_target_exposures,
    duration_discount,
    uncertainty_shrinkage,
)

# ---------- calibration ----------
from .calibration import (
    blend_assigners,
    compute_posterior_descriptor_trajectory,
    estimate_regime_params_causal,
)

# ---------- evaluation ----------
from .evaluation import (
    benjamini_hochberg,
    compute_backtest_metrics,
    deflated_sharpe_ratio,
    detection_lag_stats,
    print_metrics,
    purged_embargo_windows,
)

# ---------- hmm_offline ----------
from .hmm_offline import (
    calibrate_state_exposures,
    calibrate_state_exposures_by_vol,
    decode_filtered,
    decode_smoothed,
    fit_hmm,
    fit_offline_hmm_positions,
)

# ---------- features ----------
from .features import preprocess

# ---------- zigzag_labeling ----------
from .zigzag_labeling import zigzag_label_regimes

# ---------- plotting ----------
from .plotting import (
    REGIME_COLORS,
    setup_cjk_font,
    shade_regimes,
)

__all__ = [
    # emission
    "NIGConjugateEmission",
    "batch_nig_posterior",
    "fit_nig_prior_from_moments",
    # duration
    "DiscreteDurationModel",
    "extract_segment_durations_from_labels",
    "fit_geometric_duration",
    "fit_negbinom_duration",
    "fit_regime_duration_models",
    # bocpd
    "BOCPD",
    "BOCPDStepResult",
    # regime
    "RegimePrototype",
    "RegimeSoftAssigner",
    "name_clusters_by_return_rank",
    # decision
    "RebalanceEngine",
    "apply_uncertainty_shrinkage",
    "calibrate_target_exposures",
    "duration_discount",
    "uncertainty_shrinkage",
    # calibration
    "blend_assigners",
    "compute_posterior_descriptor_trajectory",
    "estimate_regime_params_causal",
    # evaluation
    "benjamini_hochberg",
    "compute_backtest_metrics",
    "deflated_sharpe_ratio",
    "detection_lag_stats",
    "print_metrics",
    "purged_embargo_windows",
    # hmm_offline
    "calibrate_state_exposures",
    "calibrate_state_exposures_by_vol",
    "decode_filtered",
    "decode_smoothed",
    "fit_hmm",
    "fit_offline_hmm_positions",
    # features
    "preprocess",
    # zigzag_labeling
    "zigzag_label_regimes",
    # plotting
    "REGIME_COLORS",
    "setup_cjk_font",
    "shade_regimes",
]
