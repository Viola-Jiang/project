"""
ablation/s5_multi_seed_robustness.py
========================================
对应方法论文档 §4.2「S5・多指数/行业并行 + 稳健性检验」。

"将引擎并行于多宽基/行业，顶层以风险预算分配；并在不同标的、不同时间窗、
不同随机种子下复算。理论动机：验证绩效源于机制而非特定样本的偶然。
验证：跨标的/跨窗口稳定为正且通过统计去伪（见六）。"

**关于"多指数"的现状说明**：本仓库目前只有一份真实中证800数据，没有真实
的多宽基/行业数据可用，也不再用合成数据generate多个"指数"来近似（此前
版本靠 engine/synthetic_data.simulate(seed=...) 生成多条独立路径充当代理，
现已随合成数据一起下线）。因此本脚本**收窄为只覆盖"多时间窗稳健性 + 统计
去伪"这一半**——在唯一一份真实序列上按 Purged K-Fold + Embargo 切多个
不重叠时间窗、各自独立复算 S4；"多指数/行业并行"这部分文档要求（§4.2 S5
的另一半）在拿到真实多资产数据前无法落地，留作后续工作。

S5 = S4（HSMM久期升级）+ 以下两件事，除此之外不引入任何新的策略逻辑：
  1. 单一真实序列的多时间窗（Purged K-Fold + Embargo, §6.1）并行复算，
     得到一组独立的夏普比率观测。
  2. 统计去伪（§6.3）：
     - 逐次试验的"策略收益是否显著>0"用单样本t检验给出p值，再用BH-FDR
       做多重检验校正。
     - 用全部试验汇总的收益序列计算 Deflated Sharpe Ratio，n_trials 取
       "本次消融从S0到S5实际尝试过的配置数 + S5内部的多窗口搜索数"，
       量化"这个夏普有多大概率不是撞大运撞出来的"。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import load_data, RESULTS_DIR, FIGURES_DIR, MIN_HISTORY  # noqa: E402
from ablation.s4_hsmm_duration import generate_positions as generate_positions_s4  # noqa: E402
from engine.evaluation import (  # noqa: E402
    compute_backtest_metrics, print_metrics, purged_embargo_windows,
    deflated_sharpe_ratio, benjamini_hochberg,
)
from engine.plotting import setup_cjk_font  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

N_WINDOWS = 3                # 唯一真实序列切成几个不重叠、带隔离带的时间窗（§6.1 Purged K-Fold+Embargo）
                              # 此前(多序列版)每条序列各切3个窗；现在只有一条真实序列，窗口数不宜设太大，
                              # 否则单窗口长度可能不够覆盖 MIN_TRIAL_LENGTH（历史越长可适当调大）
MIN_TRIAL_LENGTH = MIN_HISTORY + 300  # 窗口太短则跳过（燃尽期都占不满，没有评估意义）


def run_all_trials():
    """
    对唯一一份真实数据切出的每个时间窗，独立跑一遍 S4 引擎（含各自的
    walk-forward重估、独立的燃尽期），返回逐试验的绩效列表。
    """
    df = load_data()
    windows = purged_embargo_windows(len(df), N_WINDOWS, embargo_frac=0.02)

    trials = []
    for w_idx, (start, end) in enumerate(windows):
        sub = df.iloc[start:end].reset_index(drop=True)
        if len(sub) < MIN_TRIAL_LENGTH:
            print(f"  [跳过] window={w_idx}: 长度{len(sub)}天 < 最短要求{MIN_TRIAL_LENGTH}天")
            continue

        result_df = generate_positions_s4(sub)
        metrics = compute_backtest_metrics(result_df["log_return"], result_df["w_held"],
                                            regime_labels=result_df["ref_regime"])

        w_applied = result_df["w_held"].shift(1).fillna(0.0)
        strategy_return = (w_applied * result_df["log_return"]).values
        tstat, pval_two_sided = stats.ttest_1samp(strategy_return, 0.0)
        pval_one_sided = pval_two_sided / 2 if tstat > 0 else 1 - pval_two_sided / 2

        trials.append({
            "window": w_idx, "n_obs": len(sub),
            "sharpe": metrics["sharpe"], "calmar": metrics["calmar"],
            "max_dd": metrics["max_dd"], "ann_return": metrics["ann_return"],
            "turnover_rate": metrics["turnover_rate"],
            "p_value": pval_one_sided, "strategy_return": strategy_return,
        })
        print(f"  [完成] window={w_idx} ({len(sub)}天): "
              f"夏普={metrics['sharpe']:.2f}  最大回撤={metrics['max_dd']*100:.1f}%  "
              f"p值(单边,收益>0)={pval_one_sided:.3f}")

    return trials


def make_diagnostic_plot(trials: list, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sharpes = np.array([t["sharpe"] for t in trials])
    labels = [f"window{t['window']}" for t in trials]
    colors = ["#3f6fa8" if s > 0 else "#d94f4f" for s in sharpes]
    axes[0].bar(range(len(trials)), sharpes, color=colors)
    axes[0].set_xticks(range(len(trials)))
    axes[0].set_xticklabels(labels, fontsize=8, rotation=45)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("夏普比率")
    axes[0].set_title(f"S5：{len(trials)}次独立试验(多时间窗)的夏普分布")

    pvals = np.array([t["p_value"] for t in trials])
    sig = benjamini_hochberg(pvals, alpha=0.05)
    bar_colors = ["#2e7d4f" if s else "lightgray" for s in sig]
    axes[1].bar(range(len(trials)), -np.log10(pvals), color=bar_colors)
    axes[1].axhline(-np.log10(0.05), color="red", ls="--", lw=0.8, label="p=0.05")
    axes[1].set_xticks(range(len(trials)))
    axes[1].set_xticklabels(labels, fontsize=8, rotation=45)
    axes[1].set_ylabel("-log10(p值)")
    axes[1].set_title("BH-FDR校正后仍显著(绿色)的试验")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    print("=== S5 多时间窗稳健性检验（单一真实序列；多指数并行部分待真实多资产数据）===\n")
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
    # n_trials: S0~S5六级消融 + S5内部本次的多窗口搜索，均计入"尝试次数"
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
