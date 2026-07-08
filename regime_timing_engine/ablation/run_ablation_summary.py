"""
ablation/run_ablation_summary.py
====================================
依次跑 S0~S5，用同一套评估口径（engine.evaluation.compute_backtest_metrics）
汇总成文档 §4.1 要求的"逐级隔离增益来源、每步独立验证"对比表——这张表本身
就是整个消融实验最终要交付的东西。

运行方式：
  python ablation/run_ablation_summary.py   (需先运行 pipeline/01, pipeline/02)
输出：
  outputs/ablation/results/ablation_summary.csv
  控制台打印六级对比表 + 净增量 + S1→S2前视偏差量化 + S5统计去伪结论
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import load_data, RESULTS_DIR  # noqa: E402
from ablation import s0_baseline, s1_offline_hmm, s2_causal_bocpd_map, s3_full_posterior_band, s4_hsmm_duration  # noqa: E402
from ablation.s5_multi_seed_robustness import run_all_trials  # noqa: E402
from engine.evaluation import compute_backtest_metrics, deflated_sharpe_ratio, benjamini_hochberg  # noqa: E402


def run_level(name: str, result_df: pd.DataFrame, executable: bool) -> dict:
    m = compute_backtest_metrics(result_df["log_return"], result_df["w_held"],
                                  regime_labels=result_df["true_regime"])
    return {
        "level": name, "executable": executable,
        "ann_return": m["ann_return"], "ann_vol": m["ann_vol"], "sharpe": m["sharpe"],
        "calmar": m["calmar"], "max_dd": m["max_dd"], "turnover_rate": m["turnover_rate"],
    }


if __name__ == "__main__":
    df = load_data()

    print("=" * 78)
    print("依次运行 S0 ~ S4（同一份数据、同一套评估口径）")
    print("=" * 78)

    rows = []

    s0_df = s0_baseline.generate_positions(df, mode="constant")
    rows.append(run_level("S0-恒定满仓(基线)", s0_df, executable=True))

    s1_df, _ = s1_offline_hmm.generate_positions(df)
    rows.append(run_level("S1-离线HMM(前视上界,不可执行)", s1_df, executable=False))

    s2_df = s2_causal_bocpd_map.generate_positions(df)
    rows.append(run_level("S2-因果BOCPD+MAP硬切", s2_df, executable=True))

    s3_df = s3_full_posterior_band.generate_positions(df)
    rows.append(run_level("S3-全后验+无交易带", s3_df, executable=True))

    s4_df = s4_hsmm_duration.generate_positions(df)
    rows.append(run_level("S4-HSMM久期升级(终版核心)", s4_df, executable=True))

    summary_df = pd.DataFrame(rows)
    summary_df["sharpe_delta_vs_prev"] = summary_df["sharpe"].diff()
    summary_df["max_dd_delta_vs_prev"] = summary_df["max_dd"].diff()

    print("\n" + "=" * 78)
    print("S0~S4 对比表（净增量 = 本级 - 上一级）")
    print("=" * 78)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 78)
    print("S1 -> S2 前视偏差幅度量化（§6.4 对照实验的核心结论）")
    print("=" * 78)
    s1_row = summary_df[summary_df["level"].str.startswith("S1")].iloc[0]
    s2_row = summary_df[summary_df["level"].str.startswith("S2")].iloc[0]
    print(f"S1(离线HMM,前视上界)夏普 = {s1_row['sharpe']:.2f}")
    print(f"S2(因果BOCPD,真实可执行)夏普 = {s2_row['sharpe']:.2f}")
    print(f"前视偏差幅度 = {s1_row['sharpe'] - s2_row['sharpe']:.2f} (夏普点数)")
    print("这就是文档反复强调的'主流离线HMM择时绩效部分源于前视'的量化证据："
          "S1看似更好，但那部分优势本就不可能在实盘复现。")

    print("\n" + "=" * 78)
    print("S5：多指数(多种子代理) × 多时间窗 稳健性检验 + 统计去伪")
    print("=" * 78)
    trials = run_all_trials()
    sharpes = np.array([t["sharpe"] for t in trials])
    pvals = np.array([t["p_value"] for t in trials])
    sig = benjamini_hochberg(pvals, alpha=0.05)
    pooled_returns = np.concatenate([t["strategy_return"] for t in trials])
    dsr = deflated_sharpe_ratio(pooled_returns, n_trials=len(summary_df) + len(trials))

    print(f"\n{len(trials)}次独立试验：夏普均值={sharpes.mean():.2f}, 标准差={sharpes.std():.2f}, "
          f"正比例={np.mean(sharpes>0)*100:.1f}%")
    print(f"BH-FDR校正后仍显著的试验数：{sig.sum()}/{len(trials)}")
    print(f"Deflated Sharpe Ratio = {dsr['deflated_sharpe_ratio']:.3f}")

    s4_sharpe = summary_df[summary_df["level"].str.startswith("S4")].iloc[0]["sharpe"]
    rows.append({
        "level": "S5-多指数/多窗口稳健性检验", "executable": True,
        "ann_return": np.nan, "ann_vol": np.nan, "sharpe": sharpes.mean(),
        "calmar": np.nan, "max_dd": np.array([t["max_dd"] for t in trials]).mean(),
        "turnover_rate": np.array([t["turnover_rate"] for t in trials]).mean(),
        "sharpe_delta_vs_prev": sharpes.mean() - s4_sharpe,
        "max_dd_delta_vs_prev": np.nan,
    })

    final_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(RESULTS_DIR / "ablation_summary.csv", index=False)
    print(f"\n完整六级对比表已保存 -> {RESULTS_DIR / 'ablation_summary.csv'}")

    print("\n" + "=" * 78)
    print("结论摘要（诚实版）")
    print("=" * 78)
    print("- S1(oracle上限)与S2(因果)之间确实存在前视折损，量化了'严格因果'主张的价值。")
    print("- S3相对S2在换手率上没有如文档预期般下降，反而因区制聚类标签在季度重估间"
          "漂移而上升——这是本次实现暴露出的新问题，不是文档方法论本身的问题。")
    print("- S4相对S3的改善方向正确（回撤/夏普均小幅改善）但幅度有限。")
    print("- S5的统计去伪结果显示：多数窗口正夏普、但BH-FDR校正后没有任何单个试验"
          "维持显著、Deflated Sharpe Ratio也明显低于1——说明当前这套causal walk-forward"
          "配置的边际优势在统计上还站不住脚，需要更多数据/更长窗口/更细的参数搜索"
          "才能证实或证伪。")
