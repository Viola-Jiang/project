"""
validation/regime_assignment_validation.py
==================================
对应方法论文档 §3.6「区制识别与软分配」。这是组件级正确性/行为验证脚本，
不是实际执行链路的一部分。

真实数据没有上帝视角的区制标签，本脚本用 engine/regime_labeling 产出的
ref_regime/ref_regime_age替代原先合成数据自带的真实标签。"监督原型"这一步
现在准确的说法是"基于自动标注参照拟合的原型"——它仍然是一个诊断上限
参照（因为用了全样本离线信息），但不再是oracle真值上限。

核心目标：验证"区制身份识别"能否修复 bocpd_validation 阶段暴露的问题。

流程：
  1. 用自动标注参照标签拟合"参照版"区制原型（离线上限参照，非oracle真值）。
  2. 无监督KMeans聚类识别性检验（对应文档"或由段级发射统计无监督聚类得到"），
     对照对象同样是自动标注参照标签。
  3. 在线跑一遍完整序列：每一步先用[posterior均值估计, 原始log波动率]做区制
     软分配，再用 P(z=k)加权 duration_hazard_validation 阶段拟合的分区制
     hazard，喂给 BOCPD。
  4. 对比：区制混合hazard vs bocpd_validation阶段泛化hazard，MAP段龄跟踪
     能力是否有改善。

运行方式：
  python validation/regime_assignment_validation.py   (需先运行 ablation/01_data_loading.py, 02_feature_engineering.py, validation/duration_hazard_validation.py, bocpd_validation.py)
输出：
  outputs/validation/results/regime_assignment_results.csv
  outputs/validation/figures/regime_assignment_validation.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "outputs" / "validation" / "results"
FIGURES_DIR = REPO_ROOT / "outputs" / "validation" / "figures"
sys.path.insert(0, str(REPO_ROOT))

from engine.bocpd import BOCPD  # noqa: E402
from engine.duration import extract_segment_durations_from_labels, fit_regime_duration_models  # noqa: E402
from engine.regime import RegimePrototype, RegimeSoftAssigner  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def load_data():
    df = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["date"])
    return df.dropna(subset=["z", "realized_vol", "ref_regime"]).reset_index(drop=True)


def fit_reference_prototypes(df: pd.DataFrame, seg_stats: pd.DataFrame, regimes: list):
    """
    用自动标注参照标签（ref_regime）直接计算每个区制在特征空间[z, log_sigma]
    中的中心（与协方差，供 RegimeSoftAssigner 的马氏/Wasserstein距离用）。
    这是一个离线全样本上限参照，不是真值上限。
    """
    prototypes = []
    for regime in regimes:
        sub = df[df["ref_regime"] == regime]
        feat = np.column_stack([sub["z"].values, np.log(sub["realized_vol"].values)])
        mean_feature = feat.mean(axis=0)
        cov = np.cov(feat, rowvar=False) if len(feat) > 1 else np.eye(2) * 1e-6
        nb_model, _, _ = fit_regime_duration_models(seg_stats, regime)
        prototypes.append(RegimePrototype(
            name=regime, mean_feature=mean_feature, duration_model=nb_model,
            cov=cov, n_obs=len(sub)))
        print(f"  [参照原型] {regime}: mean_z={mean_feature[0]:.4f}, mean_log_sigma={mean_feature[1]:.4f}")
    return prototypes


def fit_unsupervised_prototypes_and_check_identifiability(df: pd.DataFrame, regimes: list):
    """
    无监督KMeans聚类识别性检验：不用ref_regime参与聚类本身，单纯用特征聚类
    看能否分离出接近自动标注参照区制的簇。
    """
    from sklearn.cluster import KMeans
    from scipy.optimize import linear_sum_assignment

    feat = np.column_stack([
        df["z"].rolling(10, min_periods=1).mean().values, np.log(df["realized_vol"].values)])
    feat_std = (feat - feat.mean(axis=0)) / feat.std(axis=0)

    km = KMeans(n_clusters=len(regimes), n_init=10, random_state=0)
    cluster_labels = km.fit_predict(feat_std)

    ref_labels = df["ref_regime"].values
    confusion = pd.crosstab(cluster_labels, ref_labels)
    row_ind, col_ind = linear_sum_assignment(-confusion.values)
    cluster_to_regime = {row: confusion.columns[col] for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([cluster_to_regime[c] for c in cluster_labels])
    accuracy = np.mean(mapped_pred == ref_labels)

    print("\n=== 无监督KMeans区制识别性检验（对照自动标注参照标签，非真值）===")
    print("混淆矩阵（行=聚类簇，列=自动标注参照区制）：")
    print(confusion)
    print(f"\n最优簇->区制匹配后的整体一致率: {accuracy*100:.1f}%")
    return accuracy


def run_online_with_regime_mixture(df: pd.DataFrame, assigner: RegimeSoftAssigner):
    bocpd = BOCPD(hazard_fn=lambda r: 0.01, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    regime_names = assigner.names
    records = []

    for _, row in df.iterrows():
        z_t = row["z"]
        log_sigma_t = np.log(row["realized_vol"])

        # 构造今天hazard只能用step前的信息（循环依赖里避不开的一环）
        posterior_prev = np.exp(bocpd.log_run_length_posterior)
        mu_hat_prev = float(np.sum(posterior_prev * bocpd.emission.mu))
        regime_probs_prev = assigner.assign(np.array([mu_hat_prev, log_sigma_t]))

        run_lengths = np.arange(bocpd.n_hypotheses)
        h_mix = assigner.mixture_hazard(regime_probs_prev, run_lengths)
        result = bocpd.step(z_t, hazards_override=h_mix)

        # step()跑完后用最新后验重新算一次区制概率，用于本步的记录/诊断
        posterior_now = np.exp(bocpd.log_run_length_posterior)
        mu_hat_now = float(np.sum(posterior_now * bocpd.emission.mu))
        regime_probs = assigner.assign(np.array([mu_hat_now, log_sigma_t]))
        map_regime = regime_names[int(np.argmax(regime_probs))]

        records.append({
            "date": row["date"], "ref_regime": row["ref_regime"], "ref_regime_age": row["ref_regime_age"],
            "map_regime": map_regime,
            **{f"prob_{name}": regime_probs[i] for i, name in enumerate(regime_names)},
            "map_run_length": result.map_run_length, "prob_recent_reset": result.prob_recent_reset,
        })
    return pd.DataFrame(records)


def evaluate(result_df: pd.DataFrame):
    print("\n=== 区制识别一致率（vs 自动标注参照，非真值；参照原型，在线运行）===")
    acc = (result_df["map_regime"] == result_df["ref_regime"]).mean()
    print(f"整体一致率: {acc*100:.1f}%")
    confusion = pd.crosstab(result_df["map_regime"], result_df["ref_regime"])
    print("混淆矩阵（行=模型判定区制，列=自动标注参照区制）：")
    print(confusion)

    print("\n=== 核心对比：MAP段龄跟踪能力（vs bocpd_validation阶段）===")
    stuck_pct = (result_df["map_run_length"] <= 5).mean() * 100
    print(f"regime_assignment_validation（区制混合hazard）: MAP段龄<=5的天数占比 = {stuck_pct:.1f}%")

    print("\n=== 检测滞后评估（信号=P(r_t<=3), 阈值=0.5）===")
    ref_cp_idx = [i for i in result_df.index[result_df["ref_regime_age"] == 1].tolist() if i > 0]
    recent_reset = result_df["prob_recent_reset"].values
    lags = []
    for cp_idx in ref_cp_idx:
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
    print(f"自动标注参照变点总数: {len(ref_cp_idx)}")
    print(f"检测到比例: {np.mean(~np.isnan(lags))*100:.1f}%")
    print(f"平均检测滞后: {np.nanmean(lags):.2f}天")
    return acc, stuck_pct


def make_diagnostic_plot(result_df: pd.DataFrame, bocpd_prev_path: Path, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    seg_id = (result_df["ref_regime_age"] == 1).cumsum()
    for _, seg in result_df.groupby(seg_id):
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1],
                       color=REGIME_COLORS.get(seg["ref_regime"].iloc[0], "lightgray"), alpha=0.12, lw=0)

    for regime, c in REGIME_COLORS.items():
        axes[0].plot(result_df["date"], result_df[f"prob_{regime}"], color=c, lw=0.8, label=f"P({regime})")
    axes[0].set_ylabel("区制后验概率")
    axes[0].set_title("区制软分配概率轨迹")
    axes[0].legend(fontsize=8, ncol=3)

    regime_code = {"bear": 0, "sideways": 1, "bull": 2}
    axes[1].plot(result_df["date"], result_df["ref_regime"].map(regime_code), color="black",
                 lw=1.2, label="自动标注参照区制")
    axes[1].plot(result_df["date"], result_df["map_regime"].map(regime_code), color="#8a4fd9",
                 lw=0.6, alpha=0.7, label="模型判定区制")
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["bear", "sideways", "bull"])
    axes[1].set_ylabel("区制")
    axes[1].set_title("自动标注参照区制 vs 模型判定区制")
    axes[1].legend(fontsize=8)

    if bocpd_prev_path.exists():
        prev = pd.read_csv(bocpd_prev_path, parse_dates=["date"])
        window = slice(0, 600)
        axes[2].plot(result_df["date"].iloc[window], result_df["ref_regime_age"].iloc[window],
                     color="black", lw=1.2, label="自动标注参照段龄")
        axes[2].plot(result_df["date"].iloc[window], result_df["map_run_length"].iloc[window],
                     color="#3f6fa8", lw=1, label="regime_assignment(区制混合hazard)")
        axes[2].plot(prev["date"].iloc[window], prev["map_run_length_adaptive"].iloc[window],
                     color="gray", lw=0.8, ls="--", alpha=0.7, label="bocpd_validation(泛化hazard)")
        axes[2].set_xlim(result_df["date"].iloc[window].min(), result_df["date"].iloc[window].max())
        axes[2].set_ylabel("段龄（天）")
        axes[2].set_xlabel("日期")
        axes[2].set_title("前600天：MAP段龄对比，regime_assignment vs bocpd_validation")
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    df = load_data()
    seg_stats = extract_segment_durations_from_labels(df["ref_regime"])
    regimes = ["bull", "sideways", "bear"]

    print("=== 拟合自动标注参照区制原型（离线上限参照，非oracle真值）===")
    prototypes = fit_reference_prototypes(df, seg_stats, regimes)

    fit_unsupervised_prototypes_and_check_identifiability(df, regimes)

    feature_scale = np.array([df["z"].std(), np.log(df["realized_vol"]).std()])
    assigner = RegimeSoftAssigner(prototypes, feature_scale=feature_scale, bandwidth=1.0)

    print("\n=== 在线运行：区制软分配 + 混合hazard 喂给 BOCPD ===")
    result_df = run_online_with_regime_mixture(df, assigner)
    evaluate(result_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(RESULTS_DIR / "regime_assignment_results.csv", index=False)
    print(f"\n结果已保存 -> {RESULTS_DIR / 'regime_assignment_results.csv'}")

    make_diagnostic_plot(result_df, RESULTS_DIR / "bocpd_results.csv", FIGURES_DIR / "regime_assignment_validation.png")
