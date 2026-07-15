"""
ablation/s2_causal_bocpd_map.py
==================================
§4.2「S2・因果BOCPD」。

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
    BLEND_NEW_WEIGHT, blend_prior,
)
from engine.bocpd import BOCPD  # noqa: E402
from engine.emission import fit_nig_prior_from_moments  # noqa: E402
from engine.calibration import estimate_regime_params_causal, blend_assigners  # noqa: E402
from engine.evaluation import compute_backtest_metrics, print_metrics, detection_lag_stats  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def generate_positions(df: pd.DataFrame, k_regimes: int = K_REGIMES,
                        position_bounds: tuple = (0.0, 1.0)) -> pd.DataFrame:
    """
    因果 walk-forward + BOCPD + MAP硬切离散仓位。
    返回: DataFrame['date','ref_regime','ref_regime_age','log_return',
                     'map_regime','prob_recent_reset','w_held']

    每次季度重估（发射先验、区制原型/久期/目标暴露）都与上一版做
    BLEND_NEW_WEIGHT:1-BLEND_NEW_WEIGHT 的加权平滑过渡（见
    ablation.common.blend_prior / engine.calibration.blend_assigners），
    缓解直接切换新参数造成的仓位/区制判定跳变。
    """
    bocpd = BOCPD(hazard_fn=lambda r: 0.01, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    assigner = None
    prior = None
    records = []

    for i, row in df.iterrows():
        if i >= MIN_HISTORY and (i - MIN_HISTORY) % REESTIMATE_EVERY == 0:
            hist = df.iloc[:i]
            try:
                new_prior = fit_nig_prior_from_moments(hist["z"].values[-PRIOR_LOOKBACK:])
                prior = blend_prior(new_prior, prior, new_weight=BLEND_NEW_WEIGHT)
                bocpd.emission.update_prior(*prior)
            except ValueError:
                pass
            try:
                new_assigner, _ = estimate_regime_params_causal(
                    hist, k=k_regimes, duration_family="geometric", position_bounds=position_bounds)
                assigner = blend_assigners(new_assigner, assigner, duration_family="geometric",
                                           new_weight=BLEND_NEW_WEIGHT)
            except ValueError:
                pass

        if assigner is None:
            bocpd.step(row["z"])
            records.append({"date": row["date"], "ref_regime": row["ref_regime"],
                             "ref_regime_age": row["ref_regime_age"], "log_return": row["log_return"],
                             "map_regime": "warmup", "prob_recent_reset": np.nan, "w_held": 0.0})
            continue

        z_t = row["z"]
        regime_names = assigner.names

        # 构造今天hazard只能用step前的信息
        posterior_prev = np.exp(bocpd.log_run_length_posterior)
        mu_hat_prev, sigma_hat_prev = bocpd.emission.posterior_weighted_mean_scale(posterior_prev)
        regime_probs_prev = assigner.assign(np.array([mu_hat_prev, sigma_hat_prev]))

        run_lengths = np.arange(bocpd.n_hypotheses)
        h_mix = assigner.mixture_hazard(regime_probs_prev, run_lengths)
        result = bocpd.step(z_t, hazards_override=h_mix)

        # step()跑完后用最新后验重新算一次区制概率，用于仓位决策/记录的区制判定
        posterior_now = np.exp(bocpd.log_run_length_posterior)
        mu_hat_now, sigma_hat_now = bocpd.emission.posterior_weighted_mean_scale(posterior_now)
        regime_probs = assigner.assign(np.array([mu_hat_now, sigma_hat_now]))
        map_regime = regime_names[int(np.argmax(regime_probs))]

        # MAP硬切：直接取"当前最可能区制"的目标暴露，没有混合、没有久期折减、没有无交易带
        proto_by_name = {p.name: p for p in assigner.prototypes}
        w_held = proto_by_name[map_regime].target_exposure

        records.append({"date": row["date"], "ref_regime": row["ref_regime"],
                         "ref_regime_age": row["ref_regime_age"], "log_return": row["log_return"],
                         "map_regime": map_regime, "prob_recent_reset": result.prob_recent_reset,
                         "w_held": w_held})

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
                                        regime_labels=result_df["ref_regime"])
    print_metrics("S2-因果BOCPD(MAP硬切)", metrics)
    print("分区制绩效:")
    for name, s in metrics["by_regime"].items():
        print(f"  {name}: 年化收益={s['ann_return']*100:.2f}%  夏普={s['sharpe']:.2f}  n={s['n_obs']}")

    valid = result_df[result_df["map_regime"] != "warmup"].reset_index(drop=True)
    lag_stats = detection_lag_stats(valid["ref_regime_age"].values, valid["prob_recent_reset"].values)
    print(f"检测滞后: 自动标注参照变点{lag_stats['n_ref_changepoints']}个, "
          f"检测到{lag_stats['detected_pct']:.1f}%, 平均滞后{lag_stats['mean_lag']:.2f}天")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(RESULTS_DIR / "s2_causal_bocpd_map.csv", index=False)
    make_diagnostic_plot(result_df, FIGURES_DIR / "s2_causal_bocpd_map.png")
