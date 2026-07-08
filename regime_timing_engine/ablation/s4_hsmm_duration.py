"""
ablation/s4_hsmm_duration.py
===============================
对应方法论文档 §4.2「S4・HSMM久期升级（终版核心）」。

"将恒定 hazard 替换为 HSMM 久期导出的年龄相依 hazard，并将'预期剩余久期'
纳入映射（对应实现 BOCPD(duration=...)）。理论动机：修正几何久期假设、
刻画趋势末端脆弱性。验证：在区制末端/动量崩盘段，最大回撤与尾部损失
进一步压低。"

S4 相对 S3 只改一个变量：duration_family 从 "geometric" 换成 "negbinom"，
并在仓位映射里加入 duration_discount（φ，随预期剩余久期收缩）。其余
（全后验混合、不确定性收缩、无交易带、walk-forward节奏）与 S3 完全一致。
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
from engine.decision import duration_discount, apply_uncertainty_shrinkage, RebalanceEngine  # noqa: E402
from engine.evaluation import compute_backtest_metrics, print_metrics, detection_lag_stats  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def generate_positions(df: pd.DataFrame, k_regimes: int = K_REGIMES,
                        lam_uncertainty: float = 0.5, duration_floor: float = 0.3,
                        rebalance_delta: float = 0.08, cp_threshold: float = 0.5) -> pd.DataFrame:
    """
    与 S3 唯一的差异：duration_family="negbinom"（年龄相依hazard），
    并在混合暴露之后多乘一个 duration_discount(phi)。
    """
    bocpd = BOCPD(hazard_fn=lambda r: 0.01, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    rebalancer = RebalanceEngine(delta=rebalance_delta, changepoint_threshold=cp_threshold)
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
                assigner, _ = estimate_regime_params_causal(hist, k=k_regimes, duration_family="negbinom")
            except ValueError:
                pass

        if assigner is None:
            bocpd.step(row["z"])
            records.append({"date": row["date"], "true_regime": row["regime"],
                             "true_regime_age": row["regime_age_true"], "log_return": row["log_return"],
                             "map_regime": "warmup", "prob_recent_reset": np.nan,
                             "w_raw": 0.0, "w_held": rebalancer.step(0.0, 0.0, "warmup")})
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

        proto_by_name = {p.name: p for p in assigner.prototypes}
        mixture_exposure = sum(regime_probs[i2] * proto_by_name[name].target_exposure
                                for i2, name in enumerate(regime_names))

        age = result.map_run_length
        mixture_remaining = sum(
            regime_probs[i2] * proto_by_name[name].duration_model.expected_remaining(max(age, 1))
            for i2, name in enumerate(regime_names))
        mixture_mean_duration = sum(regime_probs[i2] * proto_by_name[name].duration_model.mean()
                                     for i2, name in enumerate(regime_names))
        phi = duration_discount(mixture_remaining, mixture_mean_duration, floor=duration_floor)

        w_before_shrink = mixture_exposure * phi
        w_raw = apply_uncertainty_shrinkage(w_before_shrink, result.posterior_entropy, lam=lam_uncertainty)
        w_raw = float(np.clip(w_raw, 0.0, 1.0))
        w_held = rebalancer.step(w_raw, result.prob_recent_reset, map_regime)

        records.append({"date": row["date"], "true_regime": row["regime"],
                         "true_regime_age": row["regime_age_true"], "log_return": row["log_return"],
                         "map_regime": map_regime, "prob_recent_reset": result.prob_recent_reset,
                         "w_raw": w_raw, "w_held": w_held})

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
    axes[0].plot(result_df["date"], equity, color="#2e7d4f", lw=1.3, label="S4 HSMM久期升级")
    axes[0].set_ylabel("净值(对数轴)")
    axes[0].set_yscale("log")
    axes[0].set_title("S4：在S3基础上加入HSMM年龄相依hazard与预期剩余久期折减")
    axes[0].legend(fontsize=9)

    axes[1].plot(result_df["date"], result_df["w_raw"], color="lightsteelblue", lw=0.6, label="w_raw")
    axes[1].plot(result_df["date"], result_df["w_held"], color="#d94f4f", lw=1.1, label="w_held(无交易带后)")
    axes[1].set_ylabel("仓位")
    axes[1].set_xlabel("日期")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    df = load_data()
    result_df = generate_positions(df)

    print("=== S4 HSMM久期升级（年龄相依hazard + 预期剩余久期折减）===")
    metrics = compute_backtest_metrics(result_df["log_return"], result_df["w_held"],
                                        regime_labels=result_df["true_regime"])
    print_metrics("S4-HSMM久期升级", metrics)
    print("分区制绩效:")
    for name, s in metrics["by_regime"].items():
        print(f"  {name}: 年化收益={s['ann_return']*100:.2f}%  夏普={s['sharpe']:.2f}  n={s['n_obs']}")
    gap = metrics["rebalance_gap_days"]
    if len(gap):
        print(f"调仓间隔天数分布: 中位数={np.median(gap):.0f}天, 最大={gap.max()}天 (验证'非定频'调仓)")

    valid = result_df[result_df["map_regime"] != "warmup"].reset_index(drop=True)
    lag_stats = detection_lag_stats(valid["true_regime_age"].values, valid["prob_recent_reset"].values)
    print(f"检测滞后: 真实变点{lag_stats['n_true_changepoints']}个, "
          f"检测到{lag_stats['detected_pct']:.1f}%, 平均滞后{lag_stats['mean_lag']:.2f}天")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(RESULTS_DIR / "s4_hsmm_duration.csv", index=False)
    make_diagnostic_plot(result_df, FIGURES_DIR / "s4_hsmm_duration.png")
