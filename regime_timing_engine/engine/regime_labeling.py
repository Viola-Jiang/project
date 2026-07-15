"""
engine/regime_labeling.py
============================
本模块提供**自动标注的参照标签**作为模型自己在全样本上给出的
事后参照，命名统一为 `ref_regime` / `ref_regime_age`。

产生方式：当前直接复用 `engine/hmm_offline.py` 已有的 `fit_hmm` +
`decode_smoothed`，即 ablation §4.2 S1 使用的那套"全样本 GaussianHMM +
Viterbi 平滑解码"机制。S1 本身的语义就是"允许前视的信息上限参照"，
拿来产生事后参照标签正合适：它用了全部历史（含"未来"）才能给出的最可能
状态路径，这正是我们需要的"事后来看，这段时间大概属于哪个区制"的参照，
而不是在线可执行的判断。

`ref_regime`/`ref_regime_age` 只能用于离线诊断与评估，
绝不能被喂回 `engine/calibration.py` 的因果 walk-forward 估参逻辑：那里
本来就是设计成不使用任何标签直接对特征聚类，这条因果边界现在依然成立。
"""

from __future__ import annotations
import pandas as pd

from .hmm_offline import fit_hmm, decode_smoothed
from .regime import name_clusters_by_return_rank


def auto_label_regimes(df: pd.DataFrame, k: int = 3, seed: int = 0, n_restarts: int = 5) -> pd.DataFrame:
    """
    输入: df 至少包含 ['z', 'realized_vol', 'log_return'] 三列，按日期升序排列。
    输出: df 的副本，新增两列：
      ref_regime      —— 自动标注的参照区制名（离线全样本HMM平滑解码得到）
      ref_regime_age  —— 该参照标签连续段内的段龄（每段第一天为1）

    k=3 时按收益排名映射为 bull/sideways/bear，方便直接复用
    engine.plotting.REGIME_COLORS 与全仓库既有的三区制文案/配色。

    n_restarts: 转发给 fit_hmm 的多随机种子重启次数
    （GaussianHMM 的 EM 对初始化敏感，单一种子有一定概率收敛到没有诊断价值
    的差解，多重启取对数似然最优可规避）。
    """
    model, feat = fit_hmm(df, k=k, seed=seed, n_restarts=n_restarts)
    state_seq = decode_smoothed(model, feat)

    name_of = name_clusters_by_return_rank(state_seq, df["log_return"].values, k)
    ref_regime = pd.Series([name_of[s] for s in state_seq], index=df.index)

    seg_id = (ref_regime != ref_regime.shift(1)).cumsum()
    ref_regime_age = ref_regime.groupby(seg_id).cumcount() + 1

    out = df.copy()
    out["ref_regime"] = ref_regime.values
    out["ref_regime_age"] = ref_regime_age.values
    return out
