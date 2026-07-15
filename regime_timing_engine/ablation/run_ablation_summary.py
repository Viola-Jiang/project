"""
ablation/run_ablation_summary.py
====================================
依次跑 S0~S4（S5受 ablation.common.RUN_S5 开关控制，当前=False，暂不涉及），
用同一套评估口径（engine.evaluation.compute_backtest_metrics）汇总成文档
§4.1 要求的"逐级隔离增益来源、每步独立验证"对比表——这张表本身就是整个
消融实验最终要交付的东西。

运行方式：
  python ablation/run_ablation_summary.py   (需先运行 ablation/01_data_loading.py, 02_feature_engineering.py)
输出：
  outputs/ablation/results/ablation_summary.csv
  控制台打印对比表 + 净增量 + S1→S2前视偏差量化（+ RUN_S5=True 时的S5统计去伪结论）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import load_data, RESULTS_DIR, RUN_S5  # noqa: E402
from ablation import s0_baseline, s1_offline_hmm, s2_causal_bocpd_map, s3_full_posterior_band, s4_hsmm_duration  # noqa: E402
from engine.evaluation import compute_backtest_metrics, deflated_sharpe_ratio, benjamini_hochberg  # noqa: E402
# S5相关（ablation.s5_multi_seed_robustness）只在 RUN_S5=True 时按需导入，见下方
# S5小节——RUN_S5=False 时完全不触碰该模块，不要求它已经被跑过。


def run_level(name: str, result_df: pd.DataFrame, executable: bool) -> dict:
    m = compute_backtest_metrics(result_df["log_return"], result_df["w_held"],
                                  regime_labels=result_df["ref_regime"])
    return {
        "level": name, "executable": executable,
        "ann_return": m["ann_return"], "ann_vol": m["ann_vol"], "sharpe": m["sharpe"],
        "calmar": m["calmar"], "max_dd": m["max_dd"], "turnover_rate": m["turnover_rate"],
    }


if __name__ == "__main__":
    df = load_data()

    print("=" * 78)
    print("依次运行 S0 ~ S4")
    print("=" * 78)

    rows = []

    s0_df = s0_baseline.generate_positions(df, mode="constant")
    rows.append(run_level("S0-恒定满仓(基线)", s0_df, executable=True))

    s1_df, _ = s1_offline_hmm.generate_positions(df)
    rows.append(run_level("S1-离线HMM(前视上界)", s1_df, executable=False))

    s2_df = s2_causal_bocpd_map.generate_positions(df)
    rows.append(run_level("S2-因果BOCPD+MAP硬切", s2_df, executable=True))

    s3_df = s3_full_posterior_band.generate_positions(df)
    rows.append(run_level("S3-全后验+无交易带", s3_df, executable=True))

    s4_df = s4_hsmm_duration.generate_positions(df)
    rows.append(run_level("S4-HSMM久期升级", s4_df, executable=True))

    summary_df = pd.DataFrame(rows)
    summary_df["sharpe_delta_vs_prev"] = summary_df["sharpe"].diff()
    summary_df["max_dd_delta_vs_prev"] = summary_df["max_dd"].diff()

    # S0-均线择时不参与链式对比（它和S0-恒定满仓是同一级的两种可选基线，不是
    # 递进关系），净增量单独相对S0-恒定满仓计算，再插入到S0和S1之间展示
    s0_ma_df = s0_baseline.generate_positions(df, mode="ma")
    s0_ma_row = run_level("S0-均线择时(次要参照)", s0_ma_df, executable=True)
    s0_const_row = summary_df.iloc[0]
    s0_ma_row["sharpe_delta_vs_prev"] = s0_ma_row["sharpe"] - s0_const_row["sharpe"]
    s0_ma_row["max_dd_delta_vs_prev"] = s0_ma_row["max_dd"] - s0_const_row["max_dd"]
    rows.insert(1, s0_ma_row)
    summary_df = pd.concat(
        [summary_df.iloc[[0]], pd.DataFrame([s0_ma_row]), summary_df.iloc[1:]],
        ignore_index=True,
    )

    print("\n" + "=" * 78)
    print("S0~S4 对比表（净增量 = 本级 - 上一级；S0-均线择时的净增量例外，"
          "相对S0-恒定满仓计算，因为二者是同级的两种可选基线而非递进关系）")
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

    if RUN_S5:
        print("\n" + "=" * 78)
        print("S5：单一真实序列多时间窗稳健性检验 + 统计去伪")
        print("=" * 78)
        from ablation.s5_multi_seed_robustness import run_all_trials
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
            "level": "S5-多时间窗稳健性检验", "executable": True,
            "ann_return": np.nan, "ann_vol": np.nan, "sharpe": sharpes.mean(),
            "calmar": np.nan, "max_dd": np.array([t["max_dd"] for t in trials]).mean(),
            "turnover_rate": np.array([t["turnover_rate"] for t in trials]).mean(),
            "sharpe_delta_vs_prev": sharpes.mean() - s4_sharpe,
            "max_dd_delta_vs_prev": np.nan,
        })
    else:
        print("\n" + "=" * 78)
        print("S5：已跳过。")
        print("=" * 78)

    # 保存各方案明细 CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_saves = [
        ("s0_baseline_constant", s0_df),
        ("s0_baseline_ma", s0_ma_df),
        ("s1_offline_hmm", s1_df),
        ("s2_causal_bocpd_map", s2_df),
        ("s3_full_posterior_band", s3_df),
        ("s4_hsmm_duration", s4_df),
    ]
    for name, detail_df in detail_saves:
        path = RESULTS_DIR / f"{name}.csv"
        detail_df.to_csv(path, index=False)
        print(f"明细已保存 -> {path}")

    # 保存汇总对比表
    final_df = pd.DataFrame(rows)
    final_df.to_csv(RESULTS_DIR / "ablation_summary.csv", index=False)
    print(f"\n{'完整六级' if RUN_S5 else 'S0~S4五级'}对比表已保存 -> {RESULTS_DIR / 'ablation_summary.csv'}")
