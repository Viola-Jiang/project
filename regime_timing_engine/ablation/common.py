"""
ablation/common.py
====================
S0~S5 六级共用的：路径常量、数据加载、walk-forward 节奏参数。

节奏参数（对应文档 §5.4「更新节奏」）在这里统一定义一次，S2~S5 都必须
引用这份常量，不允许每一级各定义一套——否则"S4比S3好"这种结论可能只是
因为S4恰好用了更长的重估窗口，而不是HSMM久期升级本身的贡献。
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "outputs" / "ablation" / "results"
FIGURES_DIR = REPO_ROOT / "outputs" / "ablation" / "figures"

# ------------------------- walk-forward 节奏（对应文档 §5.4）-------------------------
MIN_HISTORY = 252        # 首次可用估参所需的最短历史（沙盒数据约12年，真实部署应更长，见文档§2）
REESTIMATE_EVERY = 63    # 参数重估间隔（约一个季度的交易日数）
PRIOR_LOOKBACK = 252     # 发射先验滚动估计所用的回看窗口
K_REGIMES = 3            # 区制/隐藏状态数，S1~S5 统一取值，确保"状态数"不是造成差异的变量

# 季度重估的新旧参数平滑权重（新版本占比），缓解直接切换到新估参数造成的
# 仓位/区制判定跳变。S2~S4 共用同一个值，避免"平滑力度不同"混进消融对比。
BLEND_NEW_WEIGHT = 0.7


def load_data() -> pd.DataFrame:
    """六级统一从这里读数据，保证用的是完全相同的一份真实数据、相同的预处理。"""
    df = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["date"])
    return df.dropna(subset=["z", "realized_vol", "ref_regime"]).reset_index(drop=True)


def blend_prior(new_prior: tuple, prev_prior: tuple | None, new_weight: float = BLEND_NEW_WEIGHT) -> tuple:
    """
    发射先验 (mu0, kappa0, alpha0, beta0) 的季度重估新旧平滑，与
    engine.calibration.blend_assigners 是同一套"7:3过渡"思路在发射先验上的
    对应实现（发射先验只是4个标量，不需要按name匹配原型那套逻辑，单独写
    一个小函数更直接）。prev_prior 为 None（首次重估）时直接返回 new_prior。
    """
    if prev_prior is None:
        return new_prior
    w = new_weight
    return tuple(w * n + (1 - w) * p for n, p in zip(new_prior, prev_prior))
