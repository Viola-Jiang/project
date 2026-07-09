"""
pipeline/05_bocpd_validation.py
=================================
本阶段把 03（共轭发射）与 04（久期->hazard）缝合成完整的在线变点检测器，
对应文档 §3.2 + §5.2 的在线推理流程。

局限说明（重要）：
  本引擎目前尚不知道"当前处于哪个区制"（区制软分配是 06 的工作），
  因此 hazard 函数暂用"全部历史段落（不分区制）汇总拟合"的泛化久期分布，
  这是一个过渡性近似，06 会替换为"按区制混合"的 hazard。

真实数据没有上帝视角的区制标签，本脚本用 engine/regime_labeling 产出的
ref_regime/ref_regime_age（离线全样本HMM给出的**参照标签，不是真值**，
详见该模块docstring）作为评估检测滞后时的参照变点位置。

验证内容：
  A. 数值正确性：每一步 run-length 后验必须归一化（求和=1）。
  B. 检测行为：MAP run-length 轨迹是否大致跟踪自动标注参照段龄；平均检测滞后。
  C. 消融对比：年龄相依 hazard vs 原始 BOCPD 常数 hazard。

运行方式：
  python pipeline/05_bocpd_validation.py   (需先运行 01, 02, 04)
输出：
  outputs/results/05_bocpd_results.csv
  outputs/figures/05_bocpd_validation.png
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
from engine.duration import fit_negbinom_duration  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def load_data():
    df = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["date"])
    return df.dropna(subset=["z", "ref_regime"]).reset_index(drop=True)


def fit_pooled_hazard(df: pd.DataFrame, max_duration: int = 3000):
    """汇总全部自动标注参照区制的历史段长（不分区制），拟合一个泛化的 NegBinom 久期分布。"""
    seg_id = (df["ref_regime_age"] == 1).cumsum()
    durations = df.groupby(seg_id)["ref_regime_age"].max().values.astype(float)
    mean_d, var_d = durations.mean(), durations.var(ddof=1)
    print(f"汇总全部 {len(durations)} 个自动标注参照历史段：均值={mean_d:.1f}, 方差={var_d:.1f}, "
          f"过离散比={var_d/mean_d:.2f}")
    model = fit_negbinom_duration(mean_d, var_d, d_min=1, max_duration=max_duration, name="pooled")
    return model, durations


def run_bocpd(z_series: np.ndarray, hazard_fn, max_run_length=None):
    bocpd = BOCPD(hazard_fn=hazard_fn, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0,
                   max_run_length=max_run_length)
    results = [bocpd.step(z_t) for z_t in z_series]
    return bocpd, results


def check_normalization(results, tol=1e-6):
    """A. 数值正确性：每一步 run-length 后验概率求和应为1。"""
    max_dev = max(abs(r.run_length_posterior.sum() - 1.0) for r in results)
    print("=== A. 归一化校验 ===")
    print(f"全部 {len(results)} 步中，后验概率求和与1的最大偏差: {max_dev:.2e}")
    assert max_dev < tol, "归一化误差超出容忍范围，递归实现可能有bug"
    print("通过：每一步 run-length 后验均正确归一化。\n")


def evaluate_detection_lag(df: pd.DataFrame, results, prob_threshold: float = 0.5):
    """
    B. 检测滞后评估（重要修正）：
    最初版本直接用 P(r_t=0|x_1:t) 作为触发信号，但可证明该量在数学上恒等于
    hazard期望本身、与观测数据无关（详见 engine/bocpd.py 模块说明）。
    正确做法：改用累积概率 P(r_t<=k) 作为检测信号——直接复用每个
    BOCPDStepResult 自带的 prob_recent_reset 字段（k=3，定义见
    engine/bocpd.py），不再在这里另外用一个不同的 k 重复实现一遍同样的
    概念（此前这里曾用 low_r_threshold=5 独立算过一份，和 BOCPD 内置的
    k=3 不一致，属于同一件事两处硬编码，现已统一为一处）。
    """
    cum_low_r_probs = np.array([r.prob_recent_reset for r in results])
    map_run_lengths = np.array([r.map_run_length for r in results])

    ref_changepoint_idx = [i for i in df.index[df["ref_regime_age"] == 1].tolist() if i > 0]

    lags = []
    max_search_window = 30
    for cp_idx in ref_changepoint_idx:
        detected = False
        for lag in range(max_search_window + 1):
            idx = cp_idx + lag
            if idx >= len(cum_low_r_probs):
                break
            if cum_low_r_probs[idx] > prob_threshold:
                lags.append(lag)
                detected = True
                break
        if not detected:
            lags.append(np.nan)

    lags = np.array(lags, dtype=float)
    detected_pct = np.mean(~np.isnan(lags)) * 100
    mean_lag = np.nanmean(lags) if detected_pct > 0 else np.nan

    print(f"=== B. 检测滞后评估（信号=BOCPD内置prob_recent_reset, 阈值={prob_threshold}）===")
    print(f"自动标注参照变点总数: {len(ref_changepoint_idx)}")
    print(f"在窗口内被检测到的比例: {detected_pct:.1f}%")
    print(f"平均检测滞后: {mean_lag:.2f} 天\n")

    return lags, cum_low_r_probs, map_run_lengths


def compare_age_aware_vs_constant_hazard(df: pd.DataFrame, pooled_model, pooled_mean_duration):
    """C. 消融对比：年龄相依 hazard vs 常数 hazard（几何久期假设）。"""
    z_series = df["z"].values
    const_hazard_value = 1.0 / pooled_mean_duration

    print("=== C. 年龄相依 hazard vs 常数 hazard 消融对比 ===")
    print(f"常数hazard取值 = 1/{pooled_mean_duration:.1f} = {const_hazard_value:.5f}\n")

    print("[年龄相依 hazard]")
    _, results_adaptive = run_bocpd(z_series, hazard_fn=lambda r: pooled_model.hazard(r))
    check_normalization(results_adaptive)
    lags_a, low_r_a, map_rl_a = evaluate_detection_lag(df, results_adaptive)

    print("[常数 hazard]")
    _, results_const = run_bocpd(z_series, hazard_fn=lambda r: const_hazard_value)
    check_normalization(results_const)
    lags_c, low_r_c, map_rl_c = evaluate_detection_lag(df, results_const)

    print("=== 对比小结 ===")
    print(f"{'指标':<20}{'年龄相依hazard':>18}{'常数hazard':>18}")
    print(f"{'平均检测滞后(天)':<20}{np.nanmean(lags_a):>18.2f}{np.nanmean(lags_c):>18.2f}")
    print(f"{'检测到比例(%)':<20}{np.mean(~np.isnan(lags_a))*100:>18.1f}{np.mean(~np.isnan(lags_c))*100:>18.1f}")

    return {
        "adaptive": {"lags": lags_a, "low_r_probs": low_r_a, "map_rl": map_rl_a},
        "const": {"lags": lags_c, "low_r_probs": low_r_c, "map_rl": map_rl_c},
    }


def make_diagnostic_plot(out_df: pd.DataFrame, save_path: Path):
    setup_cjk_font()
    ref_cp_idx = out_df.index[out_df["ref_regime_age"] == 1].tolist()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    seg_id = (out_df["ref_regime_age"] == 1).cumsum()
    for _, seg in out_df.groupby(seg_id):
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1],
                       color=REGIME_COLORS.get(seg["ref_regime"].iloc[0], "lightgray"), alpha=0.12, lw=0)

    axes[0].plot(out_df["date"], out_df["ref_regime_age"], color="black", lw=1, label="自动标注参照段龄")
    axes[0].plot(out_df["date"], out_df["map_run_length_adaptive"], color="#3f6fa8", lw=0.8,
                 alpha=0.8, label="BOCPD估计段龄(MAP)")
    axes[0].set_ylabel("段龄（天）")
    axes[0].set_title("自动标注参照段龄 vs BOCPD估计段龄")
    axes[0].legend(fontsize=9)

    axes[1].plot(out_df["date"], out_df["low_r_prob_adaptive"], color="#8a4fd9", lw=0.8)
    for idx in ref_cp_idx:
        axes[1].axvline(out_df["date"].iloc[idx], color="red", alpha=0.3, lw=0.8, ls="--")
    axes[1].axhline(0.5, color="gray", ls=":", lw=1)
    axes[1].set_ylabel("P(r_t<=5)")
    axes[1].set_title("变点检测信号 P(r_t<=5)（红色虚线=自动标注参照变点）")

    window = slice(0, 600)
    axes[2].plot(out_df["date"].iloc[window], out_df["ref_regime_age"].iloc[window],
                 color="black", lw=1.2, label="自动标注参照段龄")
    axes[2].plot(out_df["date"].iloc[window], out_df["map_run_length_adaptive"].iloc[window],
                 color="#3f6fa8", lw=1, label="BOCPD估计(MAP)")
    axes[2].set_ylabel("段龄（天）")
    axes[2].set_xlabel("日期")
    axes[2].set_title("前600天细节放大")
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    df = load_data()
    pooled_model, durations = fit_pooled_hazard(df)
    comparison = compare_age_aware_vs_constant_hazard(df, pooled_model, durations.mean())

    out_df = pd.DataFrame({
        "date": df["date"].values, "ref_regime": df["ref_regime"].values,
        "ref_regime_age": df["ref_regime_age"].values,
        "low_r_prob_adaptive": comparison["adaptive"]["low_r_probs"],
        "map_run_length_adaptive": comparison["adaptive"]["map_rl"],
        "low_r_prob_const": comparison["const"]["low_r_probs"],
        "map_run_length_const": comparison["const"]["map_rl"],
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(RESULTS_DIR / "05_bocpd_results.csv", index=False)
    print(f"\n结果已保存 -> {RESULTS_DIR / '05_bocpd_results.csv'}")

    make_diagnostic_plot(out_df, FIGURES_DIR / "05_bocpd_validation.png")
