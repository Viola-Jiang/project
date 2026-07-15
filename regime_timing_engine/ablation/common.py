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
K_REGIMES = 3            # 区制数

# 季度重估的新旧参数平滑权重（新版本占比），缓解直接切换到新估参数造成的仓位/区制判定跳变。
BLEND_NEW_WEIGHT = 0.7

# S5（多时间窗稳健性 + 统计去伪）暂时先不做（见 ablation/s5_multi_seed_robustness.py
# 顶部说明）。这里统一开关：其余汇总脚本（run_ablation_summary.py、
# backtest_report.py）据此跳过S5相关计算，不强制要求先跑一遍S5。
# 之后要恢复S5，把这里改回 True 即可，不需要动汇总脚本。
RUN_S5 = False


def load_data() -> pd.DataFrame:
    """统一数据读取"""
    df = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["date"])
    return df.dropna(subset=["z", "realized_vol", "ref_regime"]).reset_index(drop=True)


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
