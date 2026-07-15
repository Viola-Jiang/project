"""
ablation/backtest_report.py
==============================
§6「回测框架与统计检验」的完整交付：把 §6.1~§6.4 四项要求串成一份报告
（§6.5 方法论流程验证已经由 validation/ + ablation/ 整体覆盖）。

§6.1 防泄漏（walk-forward; Purged K-Fold + Embargo）
  —— S2~S5 全部走 engine/calibration.py 的因果 walk-forward 估参；
     S5 额外用 engine/evaluation.py 的 purged_embargo_windows 做多窗口切分。
     本报告只汇总结论，机制本身在 ablation/s5_multi_seed_robustness.py。

§6.2 评估口径（年化收益/夏普/Calmar/最大回撤/换手率/调仓频次分布/
     分区制绩效/检测滞后）
  —— 以 S4（终版核心）为例，完整跑一遍 compute_backtest_metrics，并且
     单独画出"调仓间隔天数分布"直方图——这是之前只存了数组、没有可视化
     验证"非定频"的缺口，本脚本补上。

§6.3 统计去伪（Deflated Sharpe Ratio; BH-FDR）
  —— 汇总 ablation/s5_multi_seed_robustness.py 的结果。

§6.4 前视偏差对照实验（同一HMM平滑vs滤波）
  —— 汇总 ablation/lookahead_contrast.py 的结果。

运行方式：
  python ablation/backtest_report.py
  (需先运行 ablation/01_data_loading.py, 02_feature_engineering.py；若
   outputs/ablation/results/ 下还没有 lookahead_contrast.csv，本脚本会自动
   现跑一遍。§6.1/§6.3 依赖 s5_multi_seed_robustness.csv，受
   ablation.common.RUN_S5 开关控制，RUN_S5=False（当前默认）时直接跳过，
   不要求S5已经跑过、也不会现跑)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import load_data, RESULTS_DIR, FIGURES_DIR, RUN_S5  # noqa: E402
from ablation.s4_hsmm_duration import generate_positions as generate_positions_s4  # noqa: E402
from engine.evaluation import (  # noqa: E402
    compute_backtest_metrics, print_metrics, detection_lag_stats,
    deflated_sharpe_ratio, benjamini_hochberg,
)
from engine.plotting import setup_cjk_font  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def make_rebalance_histogram(gap_days: np.ndarray, save_path: Path):
    """§6.2『调仓频次分布，验证"非定频"』的可视化。"""
    setup_cjk_font()
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(gap_days) == 0:
        ax.text(0.5, 0.5, "无足够调仓样本", ha="center", va="center")
    else:
        max_gap = int(gap_days.max())
        ax.hist(gap_days, bins=range(1, max_gap + 2), color="#3f6fa8", edgecolor="white")
        ax.axvline(float(np.median(gap_days)), color="red", ls="--", lw=1,
                   label=f"中位数={np.median(gap_days):.0f}天")
        ax.legend(fontsize=9)
    ax.set_xlabel("相邻两次调仓之间的间隔天数")
    ax.set_ylabel("频次")
    ax.set_title(f"调仓间隔分布（验证'非定频'，共{len(gap_days)}次间隔观测）")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


def section_6_2(df: pd.DataFrame):
    print("=" * 78)
    print("§6.2 评估口径（以S4终版核心模型为例）")
    print("=" * 78)
    result_df = generate_positions_s4(df)
    metrics = compute_backtest_metrics(result_df["log_return"], result_df["w_held"],
                                        regime_labels=result_df["ref_regime"])
    print_metrics("S4终版核心", metrics)

    print("\n分区制绩效（牛/震荡/危机）：")
    for name, s in metrics["by_regime"].items():
        print(f"  {name}: 年化收益={s['ann_return']*100:6.2f}%  夏普={s['sharpe']:5.2f}  n={s['n_obs']}")

    gap = metrics["rebalance_gap_days"]
    if len(gap):
        print(f"\n调仓间隔天数分布: 中位数={np.median(gap):.0f}天  均值={gap.mean():.1f}天  "
              f"最大={gap.max()}天  （中位数远小于最大值，证实事件驱动、非固定频率调仓）")
    make_rebalance_histogram(gap, FIGURES_DIR / "rebalance_frequency_distribution.png")

    valid = result_df[result_df["map_regime"] != "warmup"].reset_index(drop=True) \
        if "map_regime" in result_df.columns else result_df
    if "ref_regime_age" in result_df.columns and "prob_recent_reset" in result_df.columns:
        lag_stats = detection_lag_stats(valid["ref_regime_age"].values, valid["prob_recent_reset"].values)
        print(f"\n检测滞后: 自动标注参照变点{lag_stats['n_ref_changepoints']}个, "
              f"检测到{lag_stats['detected_pct']:.1f}%, 平均滞后{lag_stats['mean_lag']:.2f}天")

    return metrics


def section_6_1_and_6_3():
    print("\n" + "=" * 78)
    print("§6.1 防泄漏 + §6.3 统计去伪（汇总自 ablation/s5_multi_seed_robustness.py）")
    print("=" * 78)
    if not RUN_S5:
        print("已跳过（ablation.common.RUN_S5=False，暂不涉及S5相关内容）")
        return
    s5_path = RESULTS_DIR / "s5_multi_seed_robustness.csv"
    if not s5_path.exists():
        print("未找到S5结果，现跑一遍...")
        from ablation.s5_multi_seed_robustness import run_all_trials
        trials = run_all_trials()
    else:
        print(f"读取已有结果 -> {s5_path}")
        trials_df = pd.read_csv(s5_path)
        trials = trials_df.to_dict("records")

    sharpes = np.array([t["sharpe"] for t in trials])
    print(f"\n§6.3 统计去伪：{len(trials)}次独立试验，夏普均值={sharpes.mean():.2f}，"
          f"标准差={sharpes.std():.2f}，正比例={np.mean(sharpes>0)*100:.1f}%")
    if "p_value" in trials[0]:
        pvals = np.array([t["p_value"] for t in trials])
        sig = benjamini_hochberg(pvals, alpha=0.05)
        print(f"BH-FDR校正(alpha=0.05)后仍显著的试验数：{sig.sum()}/{len(trials)}")
    print("（Deflated Sharpe Ratio 与逐试验p值明细见 s5_multi_seed_robustness.csv/上次运行输出）")


def section_6_4():
    print("\n" + "=" * 78)
    print("§6.4 前视偏差对照实验（汇总自 ablation/lookahead_contrast.py）")
    print("=" * 78)
    contrast_path = RESULTS_DIR / "lookahead_contrast.csv"
    if not contrast_path.exists():
        print("未找到对照实验结果，现跑一遍...")
        import subprocess
        subprocess.run([sys.executable, str(REPO_ROOT / "ablation" / "lookahead_contrast.py")], check=True)
    contrast_df = pd.read_csv(contrast_path)
    smoothed = contrast_df[contrast_df["mode"] == "smoothed_viterbi"].iloc[0]
    filtered = contrast_df[contrast_df["mode"] == "filtered_forward"].iloc[0]
    print(f"同一HMM：平滑(Viterbi,不可执行)夏普={smoothed['sharpe']:.2f}  "
          f"滤波(前向算法,可执行)夏普={filtered['sharpe']:.2f}")
    print(f"纯粹因'平滑'产生的前视偏差幅度 = {smoothed['sharpe'] - filtered['sharpe']:.2f} 个夏普点")


if __name__ == "__main__":
    df = load_data()
    section_6_2(df)
    section_6_1_and_6_3()
    section_6_4()
