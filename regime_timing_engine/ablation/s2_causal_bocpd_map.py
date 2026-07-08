"""
ablation/s2_causal_bocpd_map.py
==================================
对应方法论文档 §4.2「S2・因果BOCPD（首个可上线版）」。

"将 S1 替换为严格因果的 BOCPD，以 MAP run-length / 区制给出离散仓位。
理论动机：在线滤波规避前视。验证：在因果约束下，S2 应显著优于'同样退化为
因果模式的HMM'，并量化 S1→S2 的前视折损。"

S2 相对 S1 **只改一个变量**：把"整段历史一次性拟合、允许前视"换成"walk-forward
因果估参 + 在线BOCPD滤波"，其余（区制怎么定义、状态->仓位怎么映射：都是
"硬判定当前最可能的区制，直接取该区制的目标暴露"）尽量保持对应，确保 S1→S2
的绩效落差只反映"因果 vs 非因果"这一件事（§6.4"前视偏差对照实验"）。

S2 相对最终版（S4/S5）也只引入了"因果"这一个变量，其余尚未引入的变量：
  - 还没有"全后验连续仓位"（S3引入）：这里是 MAP 硬切，regime_probs 只用
    argmax，不做混合加权。
  - 还没有"无交易带"（S3引入）：MAP regime 变了就直接换仓，没有防抖机制。
  - 还没有"HSMM久期升级"（S4引入）：duration_family="geometric"，hazard
    对BOCPD而言是常数（对应"HMM隐含的几何久期假设"），只用于变点检测，
    不参与仓位映射（因为S2根本没有duration_discount这一步）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import (  # noqa: E402
    load_data, RESULTS_DIR, FIGURES_DIR, MIN_HISTORY, REESTIMATE_EVERY, PRIOR_LOOKBACK, K_REGIMES,
)
from engine.bocpd import BOCPD  # noqa: E402
from engine.emission import fit_nig_prior_from_moments  # noqa: E402
from engine.calibration import estimate_regime_params_causal  # noqa: E402
from engine.evaluation import compute_backtest_metrics, print_metrics, detection_lag_stats  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def generate_positions(df: pd.DataFrame, k_regimes: int = K_REGIMES) -> pd.DataFrame:
    """
    因果 walk-forward + BOCPD + MAP硬切离散仓位。
    返回: DataFrame['date','true_regime','true_regime_age','log_return',
                     'map_regime','prob_recent_reset','w_held']
    """
    bocpd = BOCPD(hazard_fn=lambda r: 0.01, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    assigner = None
    records = []

    for i, row in df.iterrows():
        if i >= MIN_HISTORY and (i - MIN_HISTORY) % REESTIMATE_EVERY == 0:
            hist = df.iloc[:i]
            try:
                mu0, kappa0, alpha0, beta0 = fit_nig_prior_from_moments(hist["z"].values[-PRIOR_LOOKBACK:])
                bocpd.emission.update_prior(mu0, kappa0, alpha0, beta0)
            except ValueError:
                pass
            try:
                assigner, _ = estimate_regime_params_causal(hist, k=k_regimes, duration_family="geometric")
            except ValueError:
                pass

        if assigner is None:
            bocpd.step(row["z"])
            records.append({"date": row["date"], "true_regime": row["regime"],
                             "true_regime_age": row["regime_age_true"], "log_return": row["log_return"],
                             "map_regime": "warmup", "prob_recent_reset": np.nan, "w_held": 0.0})
            continue

        z_t = row["z"]
        log_sigma_t = np.log(row["realized_vol"])
        regime_names = assigner.names

        posterior_prev = np.exp(bocpd.log_run_length_posterior)
        mu_hat_prev = float(np.sum(posterior_prev * bocpd.emission.mu))
        regime_probs = assigner.assign(np.array([mu_hat_prev, log_sigma_t]))
        map_regime = regime_names[int(np.argmax(regime_probs))]

        run_lengths = np.arange(bocpd.n_hypotheses)
        h_mix = assigner.mixture_hazard(regime_probs, run_lengths)
        result = bocpd.step(z_t, hazards_override=h_mix)

        # MAP硬切：直接取"当前最可能区制"的目标暴露，没有混合、没有久期折减、没有无交易带
        proto_by_name = {p.name: p for p in assigner.prototypes}
        w_held = proto_by_name[map_regime].target_exposure

        records.append({"date": row["date"], "true_regime": row["regime"],
                         "true_regime_age": row["regime_age_true"], "log_return": row["log_return"],
                         "map_regime": map_regime, "prob_recent_reset": result.prob_recent_reset,
                         "w_held": w_held})

    return pd.DataFrame(records)


def make_diagnostic_plot(result_df: pd.DataFrame, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    seg_id = (result_df["true_regime"] != result_df["true_regime"].shift(1)).cumsum()
    for _, seg in result_df.groupby(seg_id):
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1],
                       color=REGIME_COLORS.get(seg["true_regime"].iloc[0], "lightgray"), alpha=0.10, lw=0)

    w_applied = result_df["w_held"].shift(1).fillna(0.0)
    equity = np.exp(np.cumsum(w_applied * result_df["log_return"]))
    axes[0].plot(result_df["date"], equity, color="#d94f4f", lw=1.3, label="S2因果BOCPD+MAP硬切仓位")
    axes[0].set_ylabel("净值(对数轴)")
    axes[0].set_yscale("log")
    axes[0].set_title("S2：因果BOCPD + MAP硬切离散仓位")
    axes[0].legend(fontsize=9)

    axes[1].plot(result_df["date"], result_df["w_held"], color="#d94f4f", lw=1)
    axes[1].set_ylabel("仓位(硬切,无平滑)")
    axes[1].set_xlabel("日期")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    df = load_data()
    result_df = generate_positions(df)

    print("=== S2 因果BOCPD + MAP硬切离散仓位 ===")
    metrics = compute_backtest_metrics(result_df["log_return"], result_df["w_held"],
                                        regime_labels=result_df["true_regime"])
    print_metrics("S2-因果BOCPD(MAP硬切)", metrics)
    print("分区制绩效:")
    for name, s in metrics["by_regime"].items():
        print(f"  {name}: 年化收益={s['ann_return']*100:.2f}%  夏普={s['sharpe']:.2f}  n={s['n_obs']}")

    valid = result_df[result_df["map_regime"] != "warmup"].reset_index(drop=True)
    lag_stats = detection_lag_stats(valid["true_regime_age"].values, valid["prob_recent_reset"].values)
    print(f"检测滞后: 真实变点{lag_stats['n_true_changepoints']}个, "
          f"检测到{lag_stats['detected_pct']:.1f}%, 平均滞后{lag_stats['mean_lag']:.2f}天")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(RESULTS_DIR / "s2_causal_bocpd_map.csv", index=False)
    make_diagnostic_plot(result_df, FIGURES_DIR / "s2_causal_bocpd_map.png")
