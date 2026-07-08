"""
pipeline/06_regime_assignment.py
==================================
对应方法论文档 §3.6「区制识别与软分配」。

核心目标：验证"区制身份识别"能否修复 05 阶段暴露的问题——
  用汇总(不分区制)的泛化hazard时，BOCPD在99.4%的时间里都跟踪失败。

流程：
  1. 用真实区制标签拟合"监督版"区制原型（oracle，作为上限参照）。
  2. 无监督KMeans聚类识别性检验（对应文档"或由段级发射统计无监督聚类得到"）。
  3. 在线跑一遍完整序列：每一步先用[posterior均值估计, 原始log波动率]做区制
     软分配，再用 P(z=k)加权 04 阶段拟合的分区制hazard，喂给 BOCPD。
  4. 对比：区制混合hazard vs 05阶段泛化hazard，MAP段龄跟踪能力是否有改善。

运行方式：
  python pipeline/06_regime_assignment.py   (需先运行 01, 02, 04, 05)
输出：
  outputs/results/06_regime_results.csv
  outputs/figures/06_regime_assignment.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "outputs" / "results"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
sys.path.insert(0, str(REPO_ROOT))

from engine.bocpd import BOCPD  # noqa: E402
from engine.duration import extract_true_segment_durations, fit_regime_duration_models  # noqa: E402
from engine.regime import RegimePrototype, RegimeSoftAssigner  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def load_data():
    df = pd.read_csv(DATA_DIR / "synthetic_features.csv", parse_dates=["date"])
    return df.dropna(subset=["z", "realized_vol"]).reset_index(drop=True)


def fit_supervised_prototypes(df: pd.DataFrame, seg_stats: pd.DataFrame, regimes: list):
    """用真实标签直接计算每个区制在特征空间[z, log_sigma]中的中心（oracle上限参照）。"""
    prototypes = []
    for regime in regimes:
        sub = df[df["regime"] == regime]
        mean_z = sub["z"].mean()
        mean_log_sigma = np.log(sub["realized_vol"]).mean()
        nb_model, _, _ = fit_regime_duration_models(seg_stats, regime)
        prototypes.append(RegimePrototype(
            name=regime, mean_feature=np.array([mean_z, mean_log_sigma]), duration_model=nb_model))
        print(f"  [监督原型] {regime}: mean_z={mean_z:.4f}, mean_log_sigma={mean_log_sigma:.4f}")
    return prototypes


def fit_unsupervised_prototypes_and_check_identifiability(df: pd.DataFrame, regimes: list):
    """无监督KMeans聚类识别性检验：不用真实标签，单纯用特征聚类看能否分离出接近真实区制的簇。"""
    from sklearn.cluster import KMeans
    from scipy.optimize import linear_sum_assignment

    feat = np.column_stack([
        df["z"].rolling(10, min_periods=1).mean().values, np.log(df["realized_vol"].values)])
    feat_std = (feat - feat.mean(axis=0)) / feat.std(axis=0)

    km = KMeans(n_clusters=len(regimes), n_init=10, random_state=0)
    cluster_labels = km.fit_predict(feat_std)

    true_labels = df["regime"].values
    confusion = pd.crosstab(cluster_labels, true_labels)
    row_ind, col_ind = linear_sum_assignment(-confusion.values)
    cluster_to_regime = {row: confusion.columns[col] for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([cluster_to_regime[c] for c in cluster_labels])
    accuracy = np.mean(mapped_pred == true_labels)

    print("\n=== 无监督KMeans区制识别性检验 ===")
    print("混淆矩阵（行=聚类簇，列=真实区制）：")
    print(confusion)
    print(f"\n最优簇->区制匹配后的整体准确率: {accuracy*100:.1f}%")
    return accuracy


def run_online_with_regime_mixture(df: pd.DataFrame, assigner: RegimeSoftAssigner):
    bocpd = BOCPD(hazard_fn=lambda r: 0.01, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    regime_names = assigner.names
    records = []

    for _, row in df.iterrows():
        z_t = row["z"]
        log_sigma_t = np.log(row["realized_vol"])

        posterior_prev = np.exp(bocpd.log_run_length_posterior)
        mu_hat_prev = float(np.sum(posterior_prev * bocpd.emission.mu))

        regime_probs = assigner.assign(np.array([mu_hat_prev, log_sigma_t]))
        map_regime = regime_names[int(np.argmax(regime_probs))]

        run_lengths = np.arange(bocpd.n_hypotheses)
        h_mix = assigner.mixture_hazard(regime_probs, run_lengths)
        result = bocpd.step(z_t, hazards_override=h_mix)

        records.append({
            "date": row["date"], "true_regime": row["regime"], "true_regime_age": row["regime_age_true"],
            "map_regime": map_regime,
            **{f"prob_{name}": regime_probs[i] for i, name in enumerate(regime_names)},
            "map_run_length": result.map_run_length, "prob_recent_reset": result.prob_recent_reset,
        })
    return pd.DataFrame(records)


def evaluate(result_df: pd.DataFrame):
    print("\n=== 区制识别准确率（监督原型，在线运行）===")
    acc = (result_df["map_regime"] == result_df["true_regime"]).mean()
    print(f"整体准确率: {acc*100:.1f}%")
    confusion = pd.crosstab(result_df["map_regime"], result_df["true_regime"])
    print("混淆矩阵（行=模型判定区制，列=真实区制）：")
    print(confusion)

    print("\n=== 核心对比：MAP段龄跟踪能力（vs 05阶段的0.6%）===")
    stuck_pct = (result_df["map_run_length"] <= 5).mean() * 100
    print(f"06（区制混合hazard）: MAP段龄<=5的天数占比 = {stuck_pct:.1f}%")
    print("05（泛化pooled hazard）: 同一指标 = 0.6% （19/2980天）")

    print("\n=== 检测滞后评估（信号=P(r_t<=3), 阈值=0.5）===")
    true_cp_idx = [i for i in result_df.index[result_df["true_regime_age"] == 1].tolist() if i > 0]
    recent_reset = result_df["prob_recent_reset"].values
    lags = []
    for cp_idx in true_cp_idx:
        detected = False
        for lag in range(31):
            idx = cp_idx + lag
            if idx >= len(recent_reset):
                break
            if recent_reset[idx] > 0.5:
                lags.append(lag)
                detected = True
                break
        if not detected:
            lags.append(np.nan)
    lags = np.array(lags, dtype=float)
    print(f"真实变点总数: {len(true_cp_idx)}")
    print(f"检测到比例: {np.mean(~np.isnan(lags))*100:.1f}%  (05阶段同指标: 17.2%)")
    print(f"平均检测滞后: {np.nanmean(lags):.2f}天  (05阶段同指标: 12.40天)")
    return acc, stuck_pct


def make_diagnostic_plot(result_df: pd.DataFrame, bocpd_prev_path: Path, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    seg_id = (result_df["true_regime_age"] == 1).cumsum()
    for _, seg in result_df.groupby(seg_id):
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1],
                       color=REGIME_COLORS[seg["true_regime"].iloc[0]], alpha=0.12, lw=0)

    for regime, c in REGIME_COLORS.items():
        axes[0].plot(result_df["date"], result_df[f"prob_{regime}"], color=c, lw=0.8, label=f"P({regime})")
    axes[0].set_ylabel("区制后验概率")
    axes[0].set_title("区制软分配概率轨迹")
    axes[0].legend(fontsize=8, ncol=3)

    regime_code = {"bear": 0, "sideways": 1, "bull": 2}
    axes[1].plot(result_df["date"], result_df["true_regime"].map(regime_code), color="black",
                 lw=1.2, label="真实区制")
    axes[1].plot(result_df["date"], result_df["map_regime"].map(regime_code), color="#8a4fd9",
                 lw=0.6, alpha=0.7, label="模型判定区制")
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["bear", "sideways", "bull"])
    axes[1].set_ylabel("区制")
    axes[1].set_title("真实区制 vs 模型判定区制")
    axes[1].legend(fontsize=8)

    if bocpd_prev_path.exists():
        prev = pd.read_csv(bocpd_prev_path, parse_dates=["date"])
        window = slice(0, 600)
        axes[2].plot(result_df["date"].iloc[window], result_df["true_regime_age"].iloc[window],
                     color="black", lw=1.2, label="真实段龄")
        axes[2].plot(result_df["date"].iloc[window], result_df["map_run_length"].iloc[window],
                     color="#3f6fa8", lw=1, label="06(区制混合hazard)")
        axes[2].plot(prev["date"].iloc[window], prev["map_run_length_adaptive"].iloc[window],
                     color="gray", lw=0.8, ls="--", alpha=0.7, label="05(泛化hazard)")
        axes[2].set_ylabel("段龄（天）")
        axes[2].set_xlabel("日期")
        axes[2].set_title("前600天：MAP段龄对比，06 vs 05")
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    df = load_data()
    seg_stats = extract_true_segment_durations(DATA_DIR / "synthetic_prices.csv")
    regimes = ["bull", "sideways", "bear"]

    print("=== 拟合监督区制原型（oracle上限参照）===")
    prototypes = fit_supervised_prototypes(df, seg_stats, regimes)

    fit_unsupervised_prototypes_and_check_identifiability(df, regimes)

    feature_scale = np.array([df["z"].std(), np.log(df["realized_vol"]).std()])
    assigner = RegimeSoftAssigner(prototypes, feature_scale=feature_scale, bandwidth=1.0)

    print("\n=== 在线运行：区制软分配 + 混合hazard 喂给 BOCPD ===")
    result_df = run_online_with_regime_mixture(df, assigner)
    evaluate(result_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(RESULTS_DIR / "06_regime_results.csv", index=False)
    print(f"\n结果已保存 -> {RESULTS_DIR / '06_regime_results.csv'}")

    make_diagnostic_plot(result_df, RESULTS_DIR / "05_bocpd_results.csv", FIGURES_DIR / "06_regime_assignment.png")
