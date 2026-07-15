"""
validation/duration_hazard_validation.py
===========================================
对应方法论文档 §3.4「HMM与HSMM」、§3.5「久期建模、hazard函数与预期剩余久期」。
这是组件级正确性/行为验证脚本，不是实际执行链路的一部分。

真实数据没有上帝视角的区制标签，本脚本从 engine/regime_labeling 产出的
ref_regime里提取每个参照区制的历史段长样本。

流程：
  1. 从 ablation/02_feature_engineering.py 生成的特征数据中，按 ref_regime
     提取每个参照区制的历史段长样本。
  2. 对每个区制分别拟合 NegBinom 久期分布（HSMM）与 Geometric 久期分布（HMM隐含假设）。
  3. 核心对比：hazard 曲线与预期剩余久期曲线，NegBinom应随年龄变化，Geometric应为常数。
  4. 交叉验证：NegBinom拟合hazard vs 直接从段长样本估计的经验hazard。

注：分区制久期拟合的核心函数（extract_segment_durations_from_labels /
fit_regime_duration_models）定义在 engine/duration.py 中，被本脚本与
regime_assignment_validation.py 共同复用，避免脚本互相 import。

运行方式：
  python validation/duration_hazard_validation.py   (需先运行 ablation/01_data_loading.py, 02_feature_engineering.py)
输出：
  outputs/validation/results/hazard_curves.csv
  outputs/validation/results/segment_durations.csv
  outputs/validation/figures/duration_hazard_validation.png
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

from engine.duration import extract_segment_durations_from_labels, fit_regime_duration_models  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def empirical_hazard(durations: np.ndarray, r_max: int) -> np.ndarray:
    """H_hat(r) = (在年龄r结束的段数) / (存活到年龄r及以上的段数)"""
    hazards = []
    for r in range(1, r_max + 1):
        at_risk = np.sum(durations >= r)
        events = np.sum(durations == r)
        hazards.append(events / at_risk if at_risk > 0 else np.nan)
    return np.array(hazards)


def make_diagnostic_plot(curves_df: pd.DataFrame, seg_stats: pd.DataFrame, regimes: list, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for j, regime in enumerate(regimes):
        sub = curves_df[curves_df["regime"] == regime]
        durations = seg_stats.loc[seg_stats["regime"] == regime, "duration"].values.astype(float)
        r_check = min(30, int(durations.min()))
        emp_h = empirical_hazard(durations, r_check) if r_check >= 5 else None

        ax = axes[0, j]
        ax.plot(sub["age"], sub["hazard_negbinom"], color=REGIME_COLORS[regime], lw=2, label="NegBinom(HSMM)")
        ax.plot(sub["age"], sub["hazard_geometric"], color="gray", ls="--", lw=1.5, label="Geometric(HMM隐含)")
        if emp_h is not None:
            ax.scatter(range(1, r_check + 1), emp_h, color="black", s=15, zorder=5, label="经验hazard")
        ax.set_title(f"{regime}: hazard H(r)")
        ax.set_xlabel("段龄 r (天)")
        if j == 0:
            ax.set_ylabel("hazard")
        ax.legend(fontsize=8)

        ax = axes[1, j]
        ax.plot(sub["age"], sub["remaining_negbinom"], color=REGIME_COLORS[regime], lw=2, label="NegBinom(HSMM)")
        ax.plot(sub["age"], sub["remaining_geometric"], color="gray", ls="--", lw=1.5, label="Geometric(HMM隐含)")
        ax.set_title(f"{regime}: 预期剩余久期")
        ax.set_xlabel("段龄 r (天)")
        if j == 0:
            ax.set_ylabel("E[剩余|年龄=r]")
        ax.legend(fontsize=8)

    plt.suptitle("负二项(HSMM) vs 几何(HMM隐含假设) —— hazard与预期剩余久期对比", y=1.00)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    features = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["date"]).dropna(subset=["ref_regime"])
    seg_stats = extract_segment_durations_from_labels(features["ref_regime"])

    print("=== 各自动标注参照区制历史段长样本概览===")
    print(seg_stats.groupby("regime")["duration"].agg(["count", "mean", "std", "min", "max"]).round(1))
    print()

    regimes = ["bull", "sideways", "bear"]
    R_MAX = 150

    models = {}
    print("=== 拟合结果 ===")
    for regime in regimes:
        nb_model, geo_model, durations = fit_regime_duration_models(seg_stats, regime)
        var_d = durations.var(ddof=1) if len(durations) > 1 else durations.mean() * 2
        print(f"  区制='{regime}': 样本段数={len(durations)}, 均值={durations.mean():.1f}, "
              f"方差={var_d:.1f}, 过离散比(var/mean)={var_d/durations.mean():.2f}")
        models[regime] = {"negbinom": nb_model, "geometric": geo_model, "durations": durations}
    print()

    print("=== 交叉验证：NegBinom拟合hazard vs 直接经验hazard（前30个段龄的均方误差）===")
    for regime in regimes:
        nb_model = models[regime]["negbinom"]
        durations = models[regime]["durations"]
        r_check = min(30, int(durations.min()))
        if r_check < 5:
            print(f"  区制='{regime}': 最短段仅{durations.min():.0f}天，年龄范围过窄，跳过量化对比")
            continue
        emp_hazard = empirical_hazard(durations, r_check)
        fit_hazard = nb_model.hazard_curve(r_check)
        mse = np.nanmean((emp_hazard - fit_hazard) ** 2)
        print(f"  区制='{regime}': 对比年龄范围=1~{r_check}天, hazard MSE={mse:.5f}")

    curves = []
    for regime in regimes:
        nb_model = models[regime]["negbinom"]
        geo_model = models[regime]["geometric"]
        for r in range(1, R_MAX + 1):
            curves.append({
                "regime": regime, "age": r,
                "hazard_negbinom": nb_model.hazard(r), "hazard_geometric": geo_model.hazard(r),
                "remaining_negbinom": nb_model.expected_remaining(r),
                "remaining_geometric": geo_model.expected_remaining(r),
            })
    curves_df = pd.DataFrame(curves)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curves_df.to_csv(RESULTS_DIR / "hazard_curves.csv", index=False)
    seg_stats.to_csv(RESULTS_DIR / "segment_durations.csv", index=True)
    print(f"\n结果已保存 -> {RESULTS_DIR}")

    make_diagnostic_plot(curves_df, seg_stats, regimes, FIGURES_DIR / "duration_hazard_validation.png")
