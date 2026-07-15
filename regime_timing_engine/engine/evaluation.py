"""
engine/evaluation.py
=====================
§6「回测框架与统计检验」。这是 S0~S5 六个级别**唯一**共用的评估入口。

对应关系：
  compute_backtest_metrics  -> §6.2 评估口径（年化收益/夏普/Calmar/最大回撤/
                                换手率/调仓频次分布/分区制绩效）
  detection_lag_stats       -> §6.2"检测滞后"（仅适用于基于BOCPD的级别 S2-S5）
  deflated_sharpe_ratio     -> §6.3 Deflated Sharpe Ratio
  benjamini_hochberg        -> §6.3 BH-FDR 多重检验校正
  purged_embargo_windows    -> §6.1 防泄漏（Purged K-Fold + Embargo），
                                用于 S5 的多窗口稳健性检验切分
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


# ============================================================================
# §6.2 评估口径
# ============================================================================
def compute_backtest_metrics(log_return, w_held, lag: int = 1, regime_labels=None) -> dict:
    """
    log_return: 逐日对数收益序列
    w_held: 逐日"决定仓位"序列（在决定当天收盘计算得到）
    lag: 仓位生效滞后天数，默认1，对应文档§5.3"次日开盘执行"：
         strategy_return[t] = w_held[t-lag] * log_return[t]
    regime_labels: 若提供（真实区制标签），额外给出分区制绩效（§6.2"分区制绩效"）

    返回字段：ann_return, ann_vol, sharpe, calmar, max_dd, n_rebalances,
             turnover_rate, rebalance_gap_days（调仓间隔天数分布，用于验证
             "非定频"）, equity, strategy_return, by_regime(可选)
    """
    w_held = pd.Series(w_held).reset_index(drop=True)
    log_return = pd.Series(log_return).reset_index(drop=True)
    w_applied = w_held.shift(lag).fillna(0.0) if lag > 0 else w_held
    strategy_return = (w_applied * log_return).values

    equity = np.exp(np.cumsum(strategy_return))
    drawdown = equity / np.maximum.accumulate(equity) - 1
    max_dd = float(drawdown.min())
    ann_return = float(strategy_return.mean() * 252)
    ann_vol = float(strategy_return.std(ddof=1) * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    changed = w_held.diff().abs() > 1e-9
    n_rebalances = int(changed.sum())
    turnover_rate = n_rebalances / len(w_held)
    change_positions = np.where(changed.values)[0]
    rebalance_gap_days = np.diff(change_positions) if len(change_positions) > 1 else np.array([])

    metrics = {
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe, "calmar": calmar,
        "max_dd": max_dd, "n_rebalances": n_rebalances, "turnover_rate": turnover_rate,
        "rebalance_gap_days": rebalance_gap_days, "equity": equity, "strategy_return": strategy_return,
    }

    if regime_labels is not None:
        regime_labels = pd.Series(regime_labels).reset_index(drop=True)
        by_regime = {}
        sr_series = pd.Series(strategy_return)
        for name, idx in regime_labels.groupby(regime_labels).groups.items():
            r = sr_series.loc[idx]
            r_ann_ret = float(r.mean() * 252)
            r_ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else np.nan
            by_regime[name] = {
                "ann_return": r_ann_ret, "ann_vol": r_ann_vol,
                "sharpe": (r_ann_ret / r_ann_vol) if (r_ann_vol and r_ann_vol > 0) else np.nan,
                "n_obs": int(len(r)),
            }
        metrics["by_regime"] = by_regime

    return metrics


def print_metrics(label: str, metrics: dict) -> None:
    """统一的单行指标打印格式，六个级别共用，避免各脚本打印格式不一致。"""
    print(f"[{label}] 年化收益={metrics['ann_return']*100:6.2f}%  "
          f"年化波动={metrics['ann_vol']*100:6.2f}%  夏普={metrics['sharpe']:5.2f}  "
          f"Calmar={metrics['calmar']:5.2f}  最大回撤={metrics['max_dd']*100:7.2f}%  "
          f"换手率={metrics['turnover_rate']*100:5.1f}%")


# ============================================================================
# §6.2 检测滞后
# ============================================================================
def detection_lag_stats(ref_regime_age, changepoint_signal, threshold: float = 0.5,
                         max_search_window: int = 30) -> dict:
    """
    ref_regime_age: 参照段龄序列（每段第一天为1）——真实数据下由
        engine.regime_labeling 给出的自动标注参照标签计算得到，不是真值。
    changepoint_signal: 逐日的"变点/重置置信度"信号（如BOCPD的prob_recent_reset）
    返回: 参照变点总数、检测到的比例、平均检测滞后天数
    """
    ref_regime_age = np.asarray(ref_regime_age)
    changepoint_signal = np.asarray(changepoint_signal)
    ref_cp_idx = [i for i in range(len(ref_regime_age)) if ref_regime_age[i] == 1 and i > 0]

    lags = []
    for cp_idx in ref_cp_idx:
        detected = False
        for lag in range(max_search_window + 1):
            idx = cp_idx + lag
            if idx >= len(changepoint_signal):
                break
            if changepoint_signal[idx] > threshold:
                lags.append(lag)
                detected = True
                break
        if not detected:
            lags.append(np.nan)
    lags = np.array(lags, dtype=float)

    return {
        "n_ref_changepoints": len(ref_cp_idx),
        "detected_pct": float(np.mean(~np.isnan(lags)) * 100) if lags.size else np.nan,
        "mean_lag": float(np.nanmean(lags)) if np.any(~np.isnan(lags)) else np.nan,
        "lags": lags,
    }


# ============================================================================
# §6.3 Deflated Sharpe Ratio
# ============================================================================
def deflated_sharpe_ratio(strategy_returns, n_trials: int, freq: int = 252) -> dict:
    """
    对"观测到的夏普比率有多大概率不是多次尝试选出来的偶然结果"给出量化答案。
    strategy_returns: 单期（如日频）策略收益序列
    n_trials: 本次比较中实际尝试过的独立配置数（例如 S0~S5 共6级，或 S5 内部
              多窗口/多种子搜索的次数）——试的次数越多，同样的观测夏普越可能
              只是运气好，DSR 会相应被压低。

    返回:
      sharpe_annualized: 观测到的年化夏普
      expected_max_sharpe_by_chance: 在"真实夏普为0、只是运气"的原假设下，
        试 n_trials 次能"期望"选出的最大夏普（年化），可以理解为"及格线"
      deflated_sharpe_ratio: 观测夏普超过这条及格线的概率(0~1)，
        越接近1，说明这个夏普是真本事而非撞大运
    """
    r = np.asarray(strategy_returns)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 30 or r.std(ddof=1) == 0:
        return {"sharpe_annualized": np.nan, "expected_max_sharpe_by_chance": np.nan,
                "deflated_sharpe_ratio": np.nan}

    sr = r.mean() / r.std(ddof=1)
    g3 = skew(r)
    g4 = kurtosis(r, fisher=False)  # 非excess峰度，正态分布=3

    sigma_sr = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12) / (T - 1))

    euler_gamma = 0.5772156649
    if n_trials > 1:
        sr0 = sigma_sr * ((1 - euler_gamma) * norm.ppf(1 - 1.0 / n_trials)
                           + euler_gamma * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    else:
        sr0 = 0.0

    dsr = float(norm.cdf((sr - sr0) / sigma_sr)) if sigma_sr > 0 else np.nan

    return {
        "sharpe_annualized": float(sr * np.sqrt(freq)),
        "expected_max_sharpe_by_chance": float(sr0 * np.sqrt(freq)),
        "deflated_sharpe_ratio": dsr,
    }


# ============================================================================
# §6.3 BH-FDR 多重检验校正
# ============================================================================
def benjamini_hochberg(p_values, alpha: float = 0.05) -> np.ndarray:
    """
    对一组同时进行的假设检验（如"S5每个窗口/种子的夏普是否显著>0"）做多重
    检验校正，控制错误发现率(FDR)不超过alpha。
    返回: 与输入等长的布尔数组，True表示该检验在校正后仍然显著。
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresh

    significant_ranked = np.zeros(n, dtype=bool)
    if passed.any():
        max_i = int(np.max(np.where(passed)))
        significant_ranked[: max_i + 1] = True

    significant = np.zeros(n, dtype=bool)
    significant[order] = significant_ranked
    return significant


# ============================================================================
# §6.1 防泄漏：Purged K-Fold + Embargo
# ============================================================================
def purged_embargo_windows(n_obs: int, n_windows: int, embargo_frac: float = 0.02) -> list[tuple[int, int]]:
    """
    把长度为 n_obs 的时间序列切成 n_windows 个不重叠的连续窗口，每个窗口边界
    两侧留出 embargo_frac * n_obs 天的隔离带（该隔离带内的数据既不用于估参，
    也不计入该窗口的评估）——避免滚动特征（如20日已实现波动、久期模型的段
    边界）跨越窗口边界，把相邻窗口的信息"泄漏"进来。

    返回: [(start, end), ...]，每个 tuple 是该窗口"纯净"区间的整数下标
          （左闭右开），长度可能小于 n_obs/n_windows（因为掐头去尾的隔离带）。
    """
    embargo = max(1, int(n_obs * embargo_frac))
    edges = np.linspace(0, n_obs, n_windows + 1).astype(int)
    windows = []
    for i in range(n_windows):
        start = edges[i] + (embargo if i > 0 else 0)
        end = edges[i + 1] - (embargo if i < n_windows - 1 else 0)
        if end > start:
            windows.append((int(start), int(end)))
    return windows
