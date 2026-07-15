"""
validation/emission_validation.py
====================================
对应方法论文档 §3.3「共轭发射与 Student-t 预测分布」。
这是组件级正确性/行为验证脚本，不是实际执行链路的一部分。

真实数据没有上帝视角的区制标签，下文中 B/C 两类验证改用 engine/regime_labeling
产出的 ref_regime/ref_regime_age——离线全样本HMM给出的**参照标签。

三类验证：
  A. 正确性校验：在线增量递归 vs 批量闭式公式，误差应 < 1e-8。
  B. 行为验证：段内收敛（后验预测应逼近段内参照均值/标准差）+ 变点敏感性
     （老假设对新区制首个观测的预测似然应骤降）。
  C. 普遍性检验：遍历全部自动标注参照区制转移点，统计似然骤降幅度的整体
     分布——单个样例的结论未必有代表性，必须看全局。

运行方式：
  python validation/emission_validation.py   (需先运行 ablation/01_data_loading.py, 02_feature_engineering.py)
输出：
  outputs/validation/results/emission_convergence_trace.csv
  outputs/validation/results/emission_transitions_drop.csv
  outputs/validation/figures/emission_validation.png
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

from engine.emission import NIGConjugateEmission, batch_nig_posterior  # noqa: E402
from engine.plotting import setup_cjk_font  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# ------------------------------------------------------------------
# A. 正确性校验：增量递归 vs 批量闭式
# ------------------------------------------------------------------
def test_incremental_matches_batch():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.3, scale=1.5, size=200)
    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 1.0

    emission = NIGConjugateEmission(mu0, kappa0, alpha0, beta0)
    for xi in x:
        emission.update(xi)
    online_mu = emission.mu[-1]
    online_kappa = emission.kappa[-1]
    online_alpha = emission.alpha[-1]
    online_beta = emission.beta[-1]

    batch_mu, batch_kappa, batch_alpha, batch_beta = batch_nig_posterior(
        x, mu0, kappa0, alpha0, beta0)

    print("=== A. 正确性校验：在线增量递归 vs 批量闭式公式 ===")
    print(f"{'参数':<8}{'在线递归':>15}{'批量闭式':>15}{'绝对误差':>15}")
    for name, on, ba in [("mu", online_mu, batch_mu), ("kappa", online_kappa, batch_kappa),
                          ("alpha", online_alpha, batch_alpha), ("beta", online_beta, batch_beta)]:
        print(f"{name:<8}{on:>15.8f}{ba:>15.8f}{abs(on-ba):>15.2e}")

    max_err = max(abs(online_mu - batch_mu), abs(online_kappa - batch_kappa),
                  abs(online_alpha - batch_alpha), abs(online_beta - batch_beta))
    assert max_err < 1e-8, f"在线递归与批量闭式不一致！最大误差 {max_err}"
    print(f"通过：最大绝对误差 {max_err:.2e} < 1e-8\n")


# ------------------------------------------------------------------
# B. 行为验证
# ------------------------------------------------------------------
def load_features():
    df = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["date"])
    return df.dropna(subset=["z"]).reset_index(drop=True)


def pick_a_clean_segment(df: pd.DataFrame, min_length: int = 60):
    seg_id = (df["ref_regime_age"] == 1).cumsum()
    for _, seg in df.groupby(seg_id):
        if len(seg) >= min_length:
            seg_end_idx = seg.index[-1]
            if seg_end_idx + 1 < len(df):
                return seg, df.loc[seg_end_idx + 1]
    raise RuntimeError("未找到足够长的自动标注参照段，请检查数据")


def test_within_segment_convergence_and_changepoint_sensitivity():
    df = load_features()
    seg, next_obs = pick_a_clean_segment(df, min_length=80)
    regime_name = seg["ref_regime"].iloc[0]
    ref_mu, ref_sigma = seg["z"].mean(), seg["z"].std()

    print(f"=== B. 行为验证：选取一个自动标注参照 '{regime_name}' 区制段（长度={len(seg)}天）===")
    print(f"该段 z_t 的样本均值={ref_mu:.4f}, 样本标准差={ref_sigma:.4f}\n")

    emission = NIGConjugateEmission(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    convergence_trace = []
    for age, z_t in enumerate(seg["z"].values, start=1):
        log_pi = emission.predictive_logpdf(z_t)[-1]
        pred_mu, pred_scale = emission.posterior_mean_scale(-1)
        convergence_trace.append((age, pred_mu, pred_scale, z_t, log_pi))
        emission.update(z_t)

    trace_df = pd.DataFrame(convergence_trace, columns=["age", "pred_mu", "pred_scale", "z_t", "log_pi"])

    sample_idx = np.linspace(0, len(trace_df) - 1, 5).astype(int)
    print("段内不同阶段的后验预测收敛情况：")
    print(trace_df.loc[sample_idx, ["age", "pred_mu", "pred_scale"]]
          .assign(ref_mu=ref_mu, ref_sigma=ref_sigma).round(4).to_string(index=False))

    final_pred_mu, final_pred_scale = trace_df.iloc[-1][["pred_mu", "pred_scale"]]
    print(f"\n段末预测均值={final_pred_mu:.4f} (参照={ref_mu:.4f}), "
          f"段末预测尺度={final_pred_scale:.4f} (参照={ref_sigma:.4f})")

    warm_up = 10
    normal_log_pi = trace_df["log_pi"].iloc[warm_up:].mean()
    changepoint_log_pi = emission.predictive_logpdf(next_obs["z"])[-1]

    print(f"\n段内正常观测平均对数预测似然 (排除热身期): {normal_log_pi:.4f}")
    print(f"变点处（老假设预测新区制'{next_obs['ref_regime']}'首个观测）对数似然: {changepoint_log_pi:.4f}")
    print(f"似然骤降幅度: {normal_log_pi - changepoint_log_pi:.4f}\n")

    # 注意：这里只是抽样检查"选中的这一个"具体段/变点，不是硬性断言——真实数据下
    # 任意挑一个变点未必表现出骤降（z_t归一化后区制间均值差异本就不大，尤其相邻
    # 均值接近的区制转移），普遍性结论要看下面 C 的全量统计，不能靠单个样例断言。
    if changepoint_log_pi < normal_log_pi:
        print("（本例）变点处预测似然显著低于段内正常水平，符合预期。")
    else:
        print("（本例）变点处预测似然并未低于段内正常水平，说明这一个具体变点信号偏弱。"
              "普遍性结论见下面 C 的全量统计。")
    return trace_df


def test_all_transitions_likelihood_drop(min_length: int = 30):
    df = load_features()
    seg_id = (df["ref_regime_age"] == 1).cumsum()
    segments = [seg for _, seg in df.groupby(seg_id) if len(seg) >= min_length]

    records = []
    for seg in segments:
        seg_end_idx = seg.index[-1]
        if seg_end_idx + 1 >= len(df):
            continue
        next_obs = df.loc[seg_end_idx + 1]

        emission = NIGConjugateEmission(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        z_vals = seg["z"].values
        for z_t in z_vals:
            emission.update(z_t)

        warm_up = min(10, len(z_vals) // 3)
        emission2 = NIGConjugateEmission(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        normal_log_pis = []
        for i, z_t in enumerate(z_vals):
            if i >= warm_up:
                normal_log_pis.append(emission2.predictive_logpdf(z_t)[-1])
            emission2.update(z_t)
        normal_log_pi = float(np.mean(normal_log_pis))
        changepoint_log_pi = emission.predictive_logpdf(next_obs["z"])[-1]

        records.append({
            "from_regime": seg["ref_regime"].iloc[0], "to_regime": next_obs["ref_regime"],
            "seg_length": len(seg), "normal_log_pi": normal_log_pi,
            "changepoint_log_pi": changepoint_log_pi, "drop": normal_log_pi - changepoint_log_pi,
        })

    result_df = pd.DataFrame(records)
    result_df["transition"] = result_df["from_regime"] + "->" + result_df["to_regime"]

    print("=== C. 全部自动标注参照区制转移点的似然骤降统计（普遍性检验）===")
    print(f"共检验 {len(result_df)} 个自动标注参照变点\n")
    print(result_df.groupby("transition")["drop"].agg(["mean", "std", "count"]).round(4))

    pct_positive = (result_df["drop"] > 0).mean() * 100
    print(f"\n整体平均骤降幅度: {result_df['drop'].mean():.4f}")
    print(f"骤降为正的变点占比: {pct_positive:.1f}%")
    if pct_positive < 60:
        print("\n注意：相当比例的自动标注参照变点并未表现出显著的似然骤降 —— 因为 z_t 已做"
              "波动率归一化，区制间的均值差异本身就不大（尤其bull/sideways之间）。这正是"
              "文档§2'可选叠加市场宽度'/多维发射 [z_t, sigma_t] 的动机所在。")
    return result_df


def make_diagnostic_plot(trace_df, trans_df, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(trace_df["age"], trace_df["pred_mu"], color="#3f6fa8", label="后验预测均值 pred_mu")
    ax.axhline(trace_df["z_t"].mean(), color="#3f6fa8", ls="--", alpha=0.5, label="段内真实均值")
    ax2 = ax.twinx()
    ax2.plot(trace_df["age"], trace_df["pred_scale"], color="#d94f4f", label="后验预测尺度 pred_scale")
    ax2.axhline(np.std(trace_df["z_t"]), color="#d94f4f", ls="--", alpha=0.5, label="段内真实标准差")
    ax.set_xlabel("段龄 (age)")
    ax.set_ylabel("预测均值", color="#3f6fa8")
    ax2.set_ylabel("预测尺度", color="#d94f4f")
    ax.set_title("B. 段内收敛：后验预测随段龄增长逐渐逼近真值")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=8)

    ax = axes[1]
    summary = trans_df.groupby("transition")["drop"].agg(["mean", "std"]).reset_index().sort_values("mean")
    colors = ["#3f6fa8" if m > 0 else "#999999" for m in summary["mean"]]
    ax.barh(summary["transition"], summary["mean"], xerr=summary["std"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("平均似然骤降幅度 (nats)，越大越'意外'")
    ax.set_title("C. 各转移类型的变点信号强度")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    test_incremental_matches_batch()
    trace_df = test_within_segment_convergence_and_changepoint_sensitivity()
    transitions_df = test_all_transitions_likelihood_drop()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(RESULTS_DIR / "emission_convergence_trace.csv", index=False)
    transitions_df.to_csv(RESULTS_DIR / "emission_transitions_drop.csv", index=False)
    print(f"\n结果已保存 -> {RESULTS_DIR}")

    make_diagnostic_plot(trace_df, transitions_df, FIGURES_DIR / "emission_validation.png")
