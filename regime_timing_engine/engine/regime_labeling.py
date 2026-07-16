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
import numpy as np
import pandas as pd

from .hmm_offline import fit_hmm, decode_smoothed
from .regime import name_clusters_by_return_rank


def _merge_short_segments(state_seq: np.ndarray, min_duration: int) -> np.ndarray:
    """
    合并连续段长度小于min_duration的短段。

    背景：Viterbi在状态边界附近，只要某一两天的观测在似然上稍微更像另一个
    状态，就可能整段只切出1~3天的"区制"（真实数据里实测过，K=3时全样本
    111段里有约15%不到10天、其中好几段只有1~3天）——一个只持续1天的"市场
    区制"不构成有持续意义的状态切换，更像是Viterbi在边界上的噪声抖动，
    不是真实的区制更替。

    做法：每轮找出最短的一段，把它整体重标成左右相邻段中更长的那一段的
    状态（相邻段更长，说明该状态在附近的统计支持更强），重复直到不再有
    短于min_duration的段。这是事后处理，不改变原始的逐帧解码结果，只是
    对"哪些切换算数"做了一个最短持续时间的过滤。
    """
    state_seq = np.asarray(state_seq).copy()
    while True:
        change = np.empty(len(state_seq), dtype=bool)
        change[0] = True
        change[1:] = state_seq[1:] != state_seq[:-1]
        seg_start = np.flatnonzero(change)
        seg_end = np.append(seg_start[1:], len(state_seq))
        seg_len = seg_end - seg_start

        if len(seg_start) <= 1:
            break
        short_idx = np.flatnonzero(seg_len < min_duration)
        if len(short_idx) == 0:
            break

        # 每轮只处理当前最短的一段，处理完立刻重新切段——合并后相邻段可能
        # 状态相同、会连成一段更长的段，长度判断必须基于最新的切段结果。
        i = short_idx[np.argmin(seg_len[short_idx])]
        prev_len = seg_len[i - 1] if i > 0 else -1
        next_len = seg_len[i + 1] if i < len(seg_len) - 1 else -1
        target_state = state_seq[seg_start[i] - 1] if prev_len >= next_len else state_seq[seg_end[i]]
        state_seq[seg_start[i]:seg_end[i]] = target_state
    return state_seq


def auto_label_regimes(df: pd.DataFrame, k: int = 3, seed: int = 0, n_restarts: int = 5,
                        min_segment_days: int = 5) -> pd.DataFrame:
    """
    输入: df 至少包含 ['z', 'realized_vol', 'log_return'] 三列，按日期升序排列。
    输出: df 的副本，新增两列：
      ref_regime      —— 自动标注的参照区制名（离线全样本HMM平滑解码 + 短段合并得到）
      ref_regime_age  —— 该参照标签连续段内的段龄（每段第一天为1）

    k=3 时按收益排名映射为 bull/sideways/bear，方便直接复用
    engine.plotting.REGIME_COLORS 与全仓库既有的三区制文案/配色。

    n_restarts: 转发给 fit_hmm 的多随机种子重启次数
    （GaussianHMM 的 EM 对初始化敏感，单一种子有一定概率收敛到没有诊断价值
    的差解，多重启取对数似然最优可规避）。

    min_segment_days: Viterbi解码之后，短于这个天数的段会被合并进相邻段
    （见_merge_short_segments），压制状态边界附近的单日/几日抖动。这个
    合并只作用于ref_regime这条参照标签，不影响S1（engine.hmm_offline.
    fit_offline_hmm_positions）自己生成仓位时用的原始逐帧解码结果——
    S1的仓位序列和相关回测数字不受此改动影响。
    """
    model, feat = fit_hmm(df, k=k, seed=seed, n_restarts=n_restarts)
    state_seq = decode_smoothed(model, feat)
    if min_segment_days > 1:
        state_seq = _merge_short_segments(state_seq, min_segment_days)

    name_of = name_clusters_by_return_rank(state_seq, df["log_return"].values, k)
    ref_regime = pd.Series([name_of[s] for s in state_seq], index=df.index)

    seg_id = (ref_regime != ref_regime.shift(1)).cumsum()
    ref_regime_age = ref_regime.groupby(seg_id).cumcount() + 1

    out = df.copy()
    out["ref_regime"] = ref_regime.values
    out["ref_regime_age"] = ref_regime_age.values
    return out
