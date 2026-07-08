"""
engine/synthetic_data.py
==========================
合成三区制（牛/震荡/危机）指数数据生成逻辑，对应文档 §2「数据与特征构建」
的数据来源说明与 §6.5「方法论流程验证」的合成数据设计要点：
  1. 三区制，各自独立的收益均值/波动。
  2. 区制久期服从负二项分布（非几何），使 hazard 随段龄变化——这是 HSMM
     相对 HMM 的核心差异化优势所要刻画的对象；若用几何分布模拟，S3→S4
     的消融就没有意义了。
  3. 收益用 Student-t 抽样，保留厚尾（对应§3.3金融意义）。
  4. 非对称区制转移矩阵（危机后倾向先进入震荡而非直接回牛市）。

本模块被两处调用：
  pipeline/01_data_simulation.py —— 生成"主线"合成数据（种子42），供
    pipeline/02-06 及 ablation/s0-s4 使用。
  ablation/s5_multi_asset_robustness.py —— 用不同种子生成多份独立的合成
    "指数"，作为文档§4.2 S5"多指数/行业并行"要求的沙盒代理（详见该脚本
    docstring里对这个近似处理的说明）。
"""

import numpy as np
import pandas as pd
from scipy.stats import nbinom, t as student_t

REGIME_PARAMS = {
    "bull": {"mu": 0.0006, "sigma": 0.010, "duration_mean": 220, "duration_var": 220 * 180},
    "sideways": {"mu": 0.0000, "sigma": 0.008, "duration_mean": 90, "duration_var": 90 * 70},
    "bear": {"mu": -0.0014, "sigma": 0.024, "duration_mean": 35, "duration_var": 35 * 25},
}

TRANSITION = {
    "bull": {"sideways": 0.55, "bear": 0.45},
    "sideways": {"bull": 0.50, "bear": 0.50},
    "bear": {"sideways": 0.70, "bull": 0.30},
}

STUDENT_T_DF = 5
N_DAYS_TARGET = 3000
DEFAULT_SEED = 42


def nbinom_params_from_mean_var(mean, var):
    """(均值,方差) -> scipy nbinom(n,p) 参数。要求 var>mean（过离散）。"""
    if var <= mean:
        raise ValueError("方差必须大于均值才能构造过离散的负二项久期分布")
    p = mean / var
    n = mean * p / (1 - p)
    return n, p


def sample_duration(regime_name, rng, min_duration=5):
    params = REGIME_PARAMS[regime_name]
    n, p = nbinom_params_from_mean_var(params["duration_mean"], params["duration_var"])
    d = nbinom.rvs(n, p, random_state=rng)
    return max(int(d), min_duration)


def sample_next_regime(current_regime, rng):
    trans = TRANSITION[current_regime]
    regimes, probs = zip(*trans.items())
    return rng.choice(regimes, p=probs)


def simulate(n_days: int = N_DAYS_TARGET, start_regime: str = "sideways",
             seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """
    生成一份合成的三区制指数数据。不同 seed 相当于文档§4.2 S5 里"另一个指数/
    行业"的沙盒代理——各自独立的一段随机路径，用来检验策略逻辑是否只在
    某一条特定路径上凑巧有效。
    """
    rng = np.random.default_rng(seed)

    regimes_seq, regime_age_seq, returns_seq = [], [], []
    current_regime = start_regime
    total_days = 0

    while total_days < n_days:
        duration = sample_duration(current_regime, rng)
        duration = min(duration, n_days - total_days)
        if duration <= 0:
            break

        params = REGIME_PARAMS[current_regime]
        t_std = np.sqrt(STUDENT_T_DF / (STUDENT_T_DF - 2))
        raw_t = student_t.rvs(STUDENT_T_DF, size=duration, random_state=rng)
        daily_returns = params["mu"] + params["sigma"] * (raw_t / t_std)

        regimes_seq.extend([current_regime] * duration)
        regime_age_seq.extend(range(1, duration + 1))
        returns_seq.extend(daily_returns.tolist())

        total_days += duration
        current_regime = sample_next_regime(current_regime, rng)

    regimes_seq = regimes_seq[:n_days]
    regime_age_seq = regime_age_seq[:n_days]
    returns_seq = np.array(returns_seq[:n_days])

    price = 100 * np.exp(np.cumsum(returns_seq))
    dates = pd.bdate_range(start="2013-01-01", periods=len(price))

    return pd.DataFrame({
        "date": dates, "regime": regimes_seq, "regime_age_true": regime_age_seq,
        "price": price, "log_return": returns_seq,
    })
