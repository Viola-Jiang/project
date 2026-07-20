"""
engine/zigzag_labeling.py
============================
规则式参照标签：申万宏源《"趋势"、"震荡"环境的划分与择时策略》
"1.1 信号的人工标注"的两阶段算法（Zig-Zag 粗筛 + Binseg 断点修正）。

定位：`engine/regime_labeling.py`（离线全样本HMM）的替代实现。已定位到
HMM 版标签几乎只按波动率分档、不含涨跌方向（bear 档年化 +4.4%），本模块
直接按价格走势方向划分，产出金融意义上的 bull / bear / sideways 三态。

与 HMM 版同样属于**离线参照标签**：允许使用全样本信息（Zig-Zag 的转折点
本来就要事后才能确认），只能用于离线诊断与评估，不得喂回因果链路。

两阶段算法（参数默认值照搬原文，原文标定对象为上证指数2015+；
本项目应用于中证800全收益2009-2026，参数是否需调整由自检+网格搜索判断）：

阶段一｜Zig-Zag 粗筛：
  追踪价格极值，反向波动超过 pivot_threshold 即认定转折点，切出波段；
  波段需同时满足 |年化收益| >= min_ann_return 且时长 >= min_days
  才初步标为趋势（按方向分 bull/bear），否则标为震荡。

阶段二｜Binseg 断点修正：
  对每段趋势在对数价格上做两段线性拟合的最优断点搜索（二分残差最小化），
  若断点后半段斜率绝对值 < slope_ratio × 前半段，判定"趋势衰竭"，
  后半段重标为震荡。
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .regime_labeling import _merge_short_segments

TRADING_DAYS = 252

# 原文参数（上证指数 2015+ 标定）
DEFAULT_PIVOT_THRESHOLD = 0.10   # Zig-Zag 转折阈值
DEFAULT_MIN_ANN_RETURN = 0.20    # 趋势波段最小年化收益（绝对值）
DEFAULT_MIN_DAYS = 63            # 趋势波段最短时长（约一季度）
DEFAULT_SLOPE_RATIO = 0.5        # 断点后/前斜率绝对值之比低于此值视为衰竭
BINSEG_MIN_SEG = 20              # 断点两侧各自的最短天数，防止退化断点


# ------------------------------------------------------------------
# 阶段一：Zig-Zag
# ------------------------------------------------------------------
def zigzag_pivots(prices: np.ndarray, threshold: float = DEFAULT_PIVOT_THRESHOLD) -> list[int]:
    """
    经典 Zig-Zag：维护当前方向上的极值点，价格从极值反向波动超过
    threshold（相对幅度）即确认一个转折点（转折点取在极值处）。
    返回转折点下标列表，首尾（0 和 len-1）始终包含，作为波段边界。
    """
    n = len(prices)
    if n < 2:
        return [0] if n else []

    pivots = [0]
    # direction: +1 表示当前在寻找高点（上行段），-1 反之；初始方向未知
    direction = 0
    ext_idx = 0
    ext_price = prices[0]

    for i in range(1, n):
        p = prices[i]
        if direction >= 0 and p > ext_price:
            ext_idx, ext_price = i, p
            direction = 1 if direction == 0 else direction
        elif direction <= 0 and p < ext_price:
            ext_idx, ext_price = i, p
            direction = -1 if direction == 0 else direction

        if direction >= 0 and p < ext_price * (1 - threshold):
            # 从高点回撤超阈值：确认高点为转折点，转入下行
            if ext_idx != pivots[-1]:
                pivots.append(ext_idx)
            direction, ext_idx, ext_price = -1, i, p
        elif direction <= 0 and p > ext_price * (1 + threshold):
            if ext_idx != pivots[-1]:
                pivots.append(ext_idx)
            direction, ext_idx, ext_price = 1, i, p

    if pivots[-1] != n - 1:
        pivots.append(n - 1)
    return pivots


def _swing_is_trend(prices: np.ndarray, start: int, end: int,
                    min_ann_return: float, min_days: int) -> str | None:
    """波段[start, end]是否够格算趋势；返回 'bull'/'bear'/None。"""
    days = end - start
    if days < min_days:
        return None
    total = prices[end] / prices[start] - 1.0
    ann = (1.0 + total) ** (TRADING_DAYS / days) - 1.0
    if abs(ann) < min_ann_return:
        return None
    return "bull" if total > 0 else "bear"


# ------------------------------------------------------------------
# 阶段二：Binseg 断点修正
# ------------------------------------------------------------------
def _best_linear_breakpoint(log_prices: np.ndarray, min_seg: int = BINSEG_MIN_SEG):
    """
    在一段对数价格内搜索"两段线性拟合总残差最小"的断点。
    返回 (断点下标, 前段斜率, 后段斜率)；段太短无法搜索时返回 None。
    """
    n = len(log_prices)
    if n < 2 * min_seg + 1:
        return None
    x = np.arange(n, dtype=float)

    def fit_rss(x_seg, y_seg):
        slope, intercept = np.polyfit(x_seg, y_seg, 1)
        resid = y_seg - (slope * x_seg + intercept)
        return float(resid @ resid), float(slope)

    best = None
    for b in range(min_seg, n - min_seg):
        rss1, s1 = fit_rss(x[:b], log_prices[:b])
        rss2, s2 = fit_rss(x[b:], log_prices[b:])
        total = rss1 + rss2
        if best is None or total < best[0]:
            best = (total, b, s1, s2)
    _, b, s1, s2 = best
    return b, s1, s2


def _binseg_correct(labels: np.ndarray, log_prices: np.ndarray,
                    slope_ratio: float = DEFAULT_SLOPE_RATIO,
                    min_seg: int = BINSEG_MIN_SEG) -> np.ndarray:
    """对每段趋势做断点修正：后半段斜率明显放缓则重标为 sideways。"""
    labels = labels.copy()
    seg_id = np.cumsum(np.concatenate([[True], labels[1:] != labels[:-1]]))
    for sid in np.unique(seg_id):
        mask = seg_id == sid
        if labels[mask][0] == "sideways":
            continue
        idx = np.flatnonzero(mask)
        res = _best_linear_breakpoint(log_prices[idx], min_seg=min_seg)
        if res is None:
            continue
        b, s1, s2 = res
        if abs(s1) > 1e-12 and abs(s2) < slope_ratio * abs(s1):
            labels[idx[b:]] = "sideways"
    return labels


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
def zigzag_label_regimes(df: pd.DataFrame,
                          pivot_threshold: float = DEFAULT_PIVOT_THRESHOLD,
                          min_ann_return: float = DEFAULT_MIN_ANN_RETURN,
                          min_days: int = DEFAULT_MIN_DAYS,
                          slope_ratio: float = DEFAULT_SLOPE_RATIO,
                          min_segment_days: int = 5) -> pd.DataFrame:
    """
    输入: df 至少包含 ['close'] 列（收盘价，按日期升序）。
    输出: df 的副本，新增 ref_regime / ref_regime_age 两列，
          口径与 regime_labeling.auto_label_regimes 完全一致，可直接替换。

    min_segment_days: 修正后可能出现极短的标签碎段，复用
    _merge_short_segments 做与 HMM 版一致的短段合并（0/1 关闭）。
    """
    prices = df["close"].to_numpy(dtype=float)
    log_prices = np.log(prices)
    n = len(prices)

    # 阶段一
    labels = np.array(["sideways"] * n, dtype=object)
    pivots = zigzag_pivots(prices, threshold=pivot_threshold)
    for start, end in zip(pivots[:-1], pivots[1:]):
        trend = _swing_is_trend(prices, start, end, min_ann_return, min_days)
        if trend is not None:
            labels[start:end + 1] = trend

    # 阶段二
    labels = _binseg_correct(labels, log_prices, slope_ratio=slope_ratio)

    # 短段合并（复用 HMM 版同一实现，保证口径一致）
    if min_segment_days > 1:
        names = ["bull", "sideways", "bear"]
        code = {name: i for i, name in enumerate(names)}
        merged = _merge_short_segments(np.array([code[l] for l in labels]), min_segment_days)
        labels = np.array([names[c] for c in merged], dtype=object)

    ref_regime = pd.Series(labels, index=df.index)
    seg_id = (ref_regime != ref_regime.shift(1)).cumsum()
    ref_regime_age = ref_regime.groupby(seg_id).cumcount() + 1

    out = df.copy()
    out["ref_regime"] = ref_regime.values
    out["ref_regime_age"] = ref_regime_age.values
    return out


# ------------------------------------------------------------------
# 参数网格搜索（诊断工具：给出按自检标准打分的排名，不自动替换默认参数）
# ------------------------------------------------------------------
def regime_stats(df_labeled: pd.DataFrame) -> pd.DataFrame:
    """按标签汇总自检统计量：年化收益/日涨占比/年化波动/段数/平均段长/天数占比。"""
    g = df_labeled.groupby("ref_regime")
    seg_id = (df_labeled["ref_regime"] != df_labeled["ref_regime"].shift(1)).cumsum()
    segs = df_labeled.groupby(seg_id).agg(regime=("ref_regime", "first"),
                                           duration=("ref_regime", "size"))
    rows = []
    for name, sub in g:
        r = sub["log_return"].dropna()
        seg_sub = segs[segs["regime"] == name]
        rows.append({
            "regime": name,
            "ann_return": float(r.mean() * TRADING_DAYS),
            "up_day_pct": float((r > 0).mean()),
            "ann_vol": float(r.std() * np.sqrt(TRADING_DAYS)),
            "n_days": int(len(sub)),
            "day_share": float(len(sub) / len(df_labeled)),
            "n_segments": int(len(seg_sub)),
            "mean_seg_days": float(seg_sub["duration"].mean()) if len(seg_sub) else np.nan,
        })
    return pd.DataFrame(rows).set_index("regime")


def score_labeling(stats: pd.DataFrame) -> float:
    """
    按 1.1 文档"标准"给一组标签打分：
      硬约束（不满足直接 -inf）：三态齐全；bull>sideways>bear 的年化收益排序；
                                bear 年化收益 < 0 < bull 年化收益。
      软目标：bull 与 bear 的年化收益差越大越好；
              震荡天数占比离 50% 太远（<15% 或 >85%）扣分（防退化：
              全是趋势或全是震荡都不是有用的标签）。
    """
    need = {"bull", "sideways", "bear"}
    if not need.issubset(set(stats.index)):
        return float("-inf")
    bull, side, bear = (stats.loc[k, "ann_return"] for k in ("bull", "sideways", "bear"))
    if not (bull > side > bear) or bear >= 0 or bull <= 0:
        return float("-inf")
    score = bull - bear
    side_share = stats.loc["sideways", "day_share"]
    if side_share < 0.15 or side_share > 0.85:
        score -= 1.0
    return float(score)


def grid_search_params(df: pd.DataFrame,
                        pivot_grid=(0.05, 0.08, 0.10, 0.15),
                        ann_grid=(0.10, 0.15, 0.20, 0.30),
                        days_grid=(21, 42, 63, 126),
                        slope_grid=(0.3, 0.5, 0.7)) -> pd.DataFrame:
    """
    对参数网格逐组合跑标注+打分，返回按得分降序的结果表。
    仅作诊断参考：得分高只代表满足"方向分离"标准的程度高，
    最终用哪组参数仍由人工结合叠图判断。
    """
    rows = []
    for pt in pivot_grid:
        for ar in ann_grid:
            for md in days_grid:
                for sr in slope_grid:
                    labeled = zigzag_label_regimes(df, pivot_threshold=pt,
                                                    min_ann_return=ar, min_days=md,
                                                    slope_ratio=sr)
                    stats = regime_stats(labeled)
                    rows.append({
                        "pivot_threshold": pt, "min_ann_return": ar,
                        "min_days": md, "slope_ratio": sr,
                        "score": score_labeling(stats),
                        "bull_ann": stats["ann_return"].get("bull", np.nan),
                        "bear_ann": stats["ann_return"].get("bear", np.nan),
                        "sideways_share": stats["day_share"].get("sideways", np.nan),
                        "n_segments": int(stats["n_segments"].sum()),
                    })
    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))
