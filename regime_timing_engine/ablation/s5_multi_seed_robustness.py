"""
ablation/s5_multi_seed_robustness.py
========================================
对应方法论文档 §4.2「S5・多指数/行业并行 + 稳健性检验」。

"将引擎并行于多宽基/行业，顶层以风险预算分配；并在不同标的、不同时间窗、
不同随机种子下复算。理论动机：验证绩效源于机制而非特定样本的偶然。
验证：跨标的/跨窗口稳定为正且通过统计去伪（见六）。"

**关于"多指数"的一处必要近似说明**：本仓库目前只有一份基于中证800设想
构造的合成数据，没有真实的多宽基/行业数据可用。因此这里用"多个独立随机
种子生成的合成指数路径"作为§4.2 S5"多指数并行"要求的沙盒代理——每个种子
对应一条独立的模拟价格路径，近似"检验策略在不同标的上是否稳定"这件事的
统计精神（虽然生成机制相同，样本仍是相互独立的）。真实部署时应替换为
真实的多宽基/行业数据。

S5 = S4（HSMM久期升级）+ 以下两件事，除此之外不引入任何新的策略逻辑：
  1. 多指数(多种子) × 多时间窗（Purged K-Fold + Embargo, §6.1）并行复算，
     得到一组独立的夏普比率观测。
  2. 统计去伪（§6.3）：
     - 逐次试验的"策略收益是否显著>0"用单样本t检验给出p值，再用BH-FDR
       做多重检验校正。
     - 用全部试验汇总的收益序列计算 Deflated Sharpe Ratio，n_trials 取
       "本次消融从S0到S5实际尝试过的配置数 + S5内部的多窗口/多种子搜索数"，
       量化"这个夏普有多大概率不是撞大运撞出来的"。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import RESULTS_DIR, FIGURES_DIR, MIN_HISTORY  # noqa: E402
from ablation.s4_hsmm_duration import generate_positions as generate_positions_s4  # noqa: E402
from engine.synthetic_data import simulate  # noqa: E402
from engine.features import compute_features  # noqa: E402
from engine.evaluation import (  # noqa: E402
    compute_backtest_metrics, print_metrics, purged_embargo_windows,
    deflated_sharpe_ratio, benjamini_hochberg,
)
from engine.plotting import setup_cjk_font  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# "多指数"沙盒代理：5个独立随机种子 = 5条独立的合成指数路径（见模块docstring的说明）
SEEDS = [42, 7, 123, 2024, 31415]
N_WINDOWS_PER_SERIES = 3     # 每条路径切成3个不重叠、带隔离带的时间窗（§6.1 Purged K-Fold+Embargo）
MIN_TRIAL_LENGTH = MIN_HISTORY + 300  # 窗口太短则跳过（燃尽期都占不满，没有评估意义）


def build_series(seed: int) -> pd.DataFrame:
    """用给定种子生成一条独立的合成"指数"路径并计算特征。"""
    prices = simulate(seed=seed)
    feats = compute_features(prices)
    return feats.dropna(subset=["z", "realized_vol"]).reset_index(drop=True)


def run_all_trials():
    """
    对每个种子(指数) × 每个时间窗，独立跑一遍 S4 引擎（含各自的walk-forward
    重估、独立的燃尽期），返回逐试验的绩效列表。
    """
    trials = []
    for seed in SEEDS:
        df = build_series(seed)
        windows = purged_embargo_windows(len(df), N_WINDOWS_PER_SERIES, embargo_frac=0.02)
        for w_idx, (start, end) in enumerate(windows):
            sub = df.iloc[start:end].reset_index(drop=True)
            if len(sub) < MIN_TRIAL_LENGTH:
                print(f"  [跳过] seed={seed}, window={w_idx}: 长度{len(sub)}天 < 最短要求{MIN_TRIAL_LENGTH}天")
                continue

            result_df = generate_positions_s4(sub)
            metrics = compute_backtest_metrics(result_df["log_return"], result_df["w_held"],
                                                regime_labels=result_df["true_regime"])

            w_applied = result_df["w_held"].shift(1).fillna(0.0)
            strategy_return = (w_applied * result_df["log_return"]).values
            tstat, pval_two_sided = stats.ttest_1samp(strategy_return, 0.0)
            pval_one_sided = pval_two_sided / 2 if tstat > 0 else 1 - pval_two_sided / 2

            trials.append({
                "seed": seed, "window": w_idx, "n_obs": len(sub),
                "sharpe": metrics["sharpe"], "calmar": metrics["calmar"],
                "max_dd": metrics["max_dd"], "ann_return": metrics["ann_return"],
                "turnover_rate": metrics["turnover_rate"],
                "p_value": pval_one_sided, "strategy_return": strategy_return,
            })
            print(f"  [完成] seed={seed}, window={w_idx} ({len(sub)}天): "
                  f"夏普={metrics['sharpe']:.2f}  最大回撤={metrics['max_dd']*100:.1f}%  "
                  f"p值(单边,收益>0)={pval_one_sided:.3f}")

    return trials


def make_diagnostic_plot(trials: list, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sharpes = np.array([t["sharpe"] for t in trials])
    labels = [f"seed{t['seed']}\nw{t['window']}" for t in trials]
    colors = ["#3f6fa8" if s > 0 else "#d94f4f" for s in sharpes]
    axes[0].bar(range(len(trials)), sharpes, color=colors)
    axes[0].set_xticks(range(len(trials)))
    axes[0].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("夏普比率")
    axes[0].set_title(f"S5：{len(trials)}次独立试验(种子×时间窗)的夏普分布")

    pvals = np.array([t["p_value"] for t in trials])
    sig = benjamini_hochberg(pvals, alpha=0.05)
    bar_colors = ["#2e7d4f" if s else "lightgray" for s in sig]
    axes[1].bar(range(len(trials)), -np.log10(pvals), color=bar_colors)
    axes[1].axhline(-np.log10(0.05), color="red", ls="--", lw=0.8, label="p=0.05")
    axes[1].set_xticks(range(len(trials)))
    axes[1].set_xticklabels(labels, fontsize=7, rotation=45)
    axes[1].set_ylabel("-log10(p值)")
    axes[1].set_title("BH-FDR校正后仍显著(绿色)的试验")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    print("=== S5 多指数(多种子代理) × 多时间窗 稳健性检验 ===\n")
    trials = run_all_trials()

    sharpes = np.array([t["sharpe"] for t in trials])
    print(f"\n共完成 {len(trials)} 次独立试验")
    print(f"夏普比率: 均值={sharpes.mean():.2f}  标准差={sharpes.std():.2f}  "
          f"正比例={np.mean(sharpes > 0)*100:.1f}%")

    print("\n=== §6.3 统计去伪 ===")
    pvals = np.array([t["p_value"] for t in trials])
    sig = benjamini_hochberg(pvals, alpha=0.05)
    print(f"BH-FDR校正(alpha=0.05)后，{sig.sum()}/{len(trials)} 个试验仍显著(策略收益>0)")

    pooled_returns = np.concatenate([t["strategy_return"] for t in trials])
    # n_trials: S0~S5六级消融 + S5内部本次的多窗口/多种子搜索，均计入"尝试次数"
    n_total_trials = 6 + len(trials)
    dsr = deflated_sharpe_ratio(pooled_returns, n_trials=n_total_trials)
    print(f"\nDeflated Sharpe Ratio（n_trials={n_total_trials}，含六级消融与S5内部搜索次数）:")
    print(f"  观测年化夏普(全部试验汇总收益) = {dsr['sharpe_annualized']:.2f}")
    print(f"  多重尝试下靠运气就能达到的期望最大夏普 = {dsr['expected_max_sharpe_by_chance']:.2f}")
    print(f"  Deflated Sharpe Ratio = {dsr['deflated_sharpe_ratio']:.3f}"
          f"  (越接近1，说明观测夏普越可能是真本事而非撞大运)")

    trials_df = pd.DataFrame([{k: v for k, v in t.items() if k != "strategy_return"} for t in trials])
    trials_df["bh_fdr_significant"] = sig
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(RESULTS_DIR / "s5_multi_seed_robustness.csv", index=False)
    print(f"\n结果已保存 -> {RESULTS_DIR / 's5_multi_seed_robustness.csv'}")

    make_diagnostic_plot(trials, FIGURES_DIR / "s5_multi_seed_robustness.png")
