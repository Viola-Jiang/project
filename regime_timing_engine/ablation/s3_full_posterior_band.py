"""
ablation/s3_full_posterior_band.py
=====================================
对应方法论文档 §4.2「S3・全后验映射 + 无交易带」。

"不再以 MAP 硬切，而是将完整 run-length / 区制后验映射为连续仓位，叠加
无交易带 δ。理论动机：保留不确定性信息、平滑敞口、抑制噪声触发。
验证：夏普提升且换手显著下降。"

S3 相对 S2 只改两个变量（结构增益）：
  1. 区制混合暴露：用全部区制的后验概率加权目标暴露（而非只取MAP硬切），
     再叠加"后验熵越大、仓位越收缩向中性"的不确定性收缩 ψ（对应§3.7）。
  2. 无交易带：仓位变动、变点概率、区制翻转任一触发才调仓（RebalanceEngine）。

**尚未引入**HSMM久期升级（那是S4的贡献）：duration_family 仍然是
"geometric"（几何/常数hazard），也不调用 duration_discount（φ≡1）——
这样 S3→S4 的绩效落差才能干净地归因于"久期升级"这一个变量。
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
from engine.decision import apply_uncertainty_shrinkage, RebalanceEngine  # noqa: E402
from engine.evaluation import compute_backtest_metrics, print_metrics, detection_lag_stats  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def generate_positions(df: pd.DataFrame, k_regimes: int = K_REGIMES,
                        lam_uncertainty: float = 0.5,
                        rebalance_delta: float = 0.08, cp_threshold: float = 0.5) -> pd.DataFrame:
    """
    因果 walk-forward + BOCPD + 全后验混合暴露 + 不确定性收缩 + 无交易带。
    与 S2 唯一的估参差异：duration_family 仍是 geometric（常数hazard），
    没有久期折减这一步（那是S4）。
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
                assigner, _ = estimate_regime_params_causal(hist, k=k_regimes, duration_family="geometric")
            except ValueError:
                pass

        if assigner is None:
            bocpd.step(row["z"])
            records.append({"date": row["date"], "ref_regime": row["ref_regime"],
                             "ref_regime_age": row["ref_regime_age"], "log_return": row["log_return"],
                             "map_regime": "warmup", "prob_recent_reset": np.nan,
                             "w_raw": 0.0, "w_held": rebalancer.step(0.0, 0.0, "warmup")})
            continue

        z_t = row["z"]
        regime_names = assigner.names

        posterior_prev = np.exp(bocpd.log_run_length_posterior)
        mu_hat_prev, sigma_hat_prev = bocpd.emission.posterior_weighted_mean_scale(posterior_prev)
        regime_probs = assigner.assign(np.array([mu_hat_prev, sigma_hat_prev]))
        map_regime = regime_names[int(np.argmax(regime_probs))]

        run_lengths = np.arange(bocpd.n_hypotheses)
        h_mix = assigner.mixture_hazard(regime_probs, run_lengths)
        result = bocpd.step(z_t, hazards_override=h_mix)

        proto_by_name = {p.name: p for p in assigner.prototypes}
        mixture_exposure = sum(regime_probs[i2] * proto_by_name[name].target_exposure
                                for i2, name in enumerate(regime_names))
        # 没有久期折减：phi恒为1（S4才引入）
        w_raw = apply_uncertainty_shrinkage(mixture_exposure, result.posterior_entropy, lam=lam_uncertainty)
        w_raw = float(np.clip(w_raw, 0.0, 1.0))
        w_held = rebalancer.step(w_raw, result.prob_recent_reset, map_regime)

        records.append({"date": row["date"], "ref_regime": row["ref_regime"],
                         "ref_regime_age": row["ref_regime_age"], "log_return": row["log_return"],
                         "map_regime": map_regime, "prob_recent_reset": result.prob_recent_reset,
                         "w_raw": w_raw, "w_held": w_held})

    return pd.DataFrame(records)


def make_diagnostic_plot(result_df: pd.DataFrame, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    seg_id = (result_df["ref_regime"] != result_df["ref_regime"].shift(1)).cumsum()
    for _, seg in result_df.groupby(seg_id):
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1],
                       color=REGIME_COLORS.get(seg["ref_regime"].iloc[0], "lightgray"), alpha=0.10, lw=0)

    w_applied = result_df["w_held"].shift(1).fillna(0.0)
    equity = np.exp(np.cumsum(w_applied * result_df["log_return"]))
    axes[0].plot(result_df["date"], equity, color="#3f6fa8", lw=1.3, label="S3全后验+无交易带")
    axes[0].set_ylabel("净值(对数轴)")
    axes[0].set_yscale("log")
    axes[0].set_title("S3：全后验连续仓位 + 不确定性收缩 + 无交易带（尚无HSMM久期升级）")
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

    print("=== S3 全后验连续仓位 + 不确定性收缩 + 无交易带 ===")
    metrics = compute_backtest_metrics(result_df["log_return"], result_df["w_held"],
                                        regime_labels=result_df["ref_regime"])
    print_metrics("S3-全后验+无交易带", metrics)
    print("分区制绩效:")
    for name, s in metrics["by_regime"].items():
        print(f"  {name}: 年化收益={s['ann_return']*100:.2f}%  夏普={s['sharpe']:.2f}  n={s['n_obs']}")
    gap = metrics["rebalance_gap_days"]
    if len(gap):
        print(f"调仓间隔天数分布: 中位数={np.median(gap):.0f}天, 最大={gap.max()}天 (验证'非定频'调仓)")

    valid = result_df[result_df["map_regime"] != "warmup"].reset_index(drop=True)
    lag_stats = detection_lag_stats(valid["ref_regime_age"].values, valid["prob_recent_reset"].values)
    print(f"检测滞后: 自动标注参照变点{lag_stats['n_ref_changepoints']}个, "
          f"检测到{lag_stats['detected_pct']:.1f}%, 平均滞后{lag_stats['mean_lag']:.2f}天")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(RESULTS_DIR / "s3_full_posterior_band.csv", index=False)
    make_diagnostic_plot(result_df, FIGURES_DIR / "s3_full_posterior_band.png")
