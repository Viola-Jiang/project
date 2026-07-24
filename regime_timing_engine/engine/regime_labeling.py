"""
engine/regime_labeling.py
============================
本模块曾经提供**自动标注的参照标签**（`ref_regime`/`ref_regime_age`）：
复用 `engine/hmm_offline.py` 的 `fit_hmm`+`decode_smoothed`（全样本
GaussianHMM + Viterbi 平滑解码）在全样本上给出事后参照。这个版本
（`auto_label_regimes`）已被移除——诊断发现它几乎只按波动率分档、不含
涨跌方向（bear 档年化 +4.4%，参见 `engine/zigzag_labeling.py` 顶部
说明），已被规则式的 `engine.zigzag_labeling.zigzag_label_regimes`
取代并在 `ablation/02_feature_engineering.py` 中投入生产使用，
`auto_label_regimes` 自此没有任何调用方，属于死代码，予以删除。

本模块现在只保留 `_merge_short_segments`：这是一个与标签来源无关的通用
后处理工具（合并连续段长度过短的碎段），继续被 `engine/zigzag_labeling.py`
复用，因此保留独立模块避免循环 import（zigzag_labeling 依赖它，若把它
挪进 zigzag_labeling.py 自身则无处安放）。

`ref_regime`/`ref_regime_age`（现由 zigzag_label_regimes 产出）只能用于
离线诊断与评估，绝不能被喂回 `engine/calibration.py` 的因果 walk-forward
估参逻辑：那里本来就是设计成不使用任何标签直接对特征聚类，这条因果边界
依然成立。
"""

from __future__ import annotations
import numpy as np


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
