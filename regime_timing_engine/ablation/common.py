"""
ablation/common.py
====================
S0~S5 六级共用的：路径常量、数据加载、walk-forward 节奏参数。

节奏参数（§5.4「更新节奏」）在这里统一定义，S2~S5 都引用这份常量。
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "outputs" / "ablation" / "results"
FIGURES_DIR = REPO_ROOT / "outputs" / "ablation" / "figures"

# ------------------------- walk-forward 节奏（对应 §5.4）-------------------------
MIN_HISTORY = 252        # 首次可用估参所需的最短历史
REESTIMATE_EVERY = 63    # 参数重估间隔（约一个季度的交易日数）
PRIOR_LOOKBACK = 252     # 发射先验滚动估计所用的回看窗口
K_REGIMES = 3            # 区制数（S1~S4旧版[trend_W,vol_W]/zigzag流水线用）
TREND_VOL_WINDOW = 21    # engine.calibration.rolling_trend_vol_features 的滚动窗口（约一个月）

# ---------------- Approach A：[t_stat, extremity, vol_level]三维聚类（见1.3文档第9~13点）----------------
# 经feature_validation/脚本实证验证：K=4时该三维特征组合的原始(未平滑)聚类，
# 同时通过收益两两比较和久期日频卡方检验，且段长尾部经零假设置换检验证明不是
# 纯粹的滚动窗口机械假象；K=3同一特征组合未通过久期端检验。
K_REGIMES_MULTIAXIS = 4
EXTREMITY_M = 60         # 透支轴：当前累计收益的窗口
EXTREMITY_LOOKBACK = 500  # 透支轴：历史参照基准的回看长度（约2年）
VOL_LOOKBACK = 500       # 波动率水平：历史参照基准的回看长度
TREND_GATE_THRESHOLD = 1.5  # engine.features.gate_trend_by_significance 的t统计量阈值（S1用）
VOL_SPLIT_WINDOW = 90       # engine.features.classify_high_vol 的滚动窗口（S1按波动强度细分标定凯利仓位用）

# 季度重估的新旧参数平滑权重（新版本占比），缓解直接切换到新估参数造成的仓位/区制判定跳变。
BLEND_NEW_WEIGHT = 0.7

# S5（多时间窗稳健性 + 统计去伪）暂时先不做（见 ablation/s5_multi_seed_robustness.py
# 顶部说明）。这里统一开关：其余汇总脚本（run_ablation_summary.py、
# backtest_report.py）据此跳过S5相关计算，不强制要求先跑一遍S5。
# 之后要恢复S5，把这里改回 True 即可，不需要动汇总脚本。
RUN_S5 = False


def load_data() -> pd.DataFrame:
    """统一数据读取。附带 close 列（= price 别名），供
    engine.calibration.estimate_regime_params_rule_based 调用
    zigzag_label_regimes 时使用。"""
    df = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["date"])
    df = df.dropna(subset=["z", "realized_vol", "ref_regime"]).reset_index(drop=True)
    df["close"] = df["price"]
    return df


def blend_prior(new_prior: tuple, prev_prior: tuple | None, new_weight: float = BLEND_NEW_WEIGHT) -> tuple:
    """
    季度重估时对新旧发射先验做 7:3 混合，防止先验突变导致 BOCPD 行为跳变。
    每个参数独立加权：new_weight * new_val + (1-new_weight) * prev_val。
    prev_prior 为 None（首次估参，还没有旧版）时直接返回 new_prior。
    """
    if prev_prior is None:
        return new_prior
    w = new_weight
    return tuple(w * n + (1 - w) * p for n, p in zip(new_prior, prev_prior))
