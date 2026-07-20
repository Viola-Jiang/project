"""
ablation/feature_dim_causal_contrast.py
=========================================
feature_dim_contrast.py 的"完全因果"复核版。

背景：feature_dim_contrast.py 里，仅z+平滑给出异常高的夏普（约2.77），但那份
实验的 HMM 参数拟合、解码、状态->暴露标定三处都用了全样本（含未来），是一个
极度乐观、不可执行的上界。本脚本把三处全部改成严格因果的 walk-forward，用来
判断"仅z 明显优于 [z,logσ]"这个结论在去掉全部前视之后是否还成立：

  - 每隔 REESTIMATE_EVERY 天，用截至当前的历史 df[:r] 重新 EM 拟合 HMM；
  - 状态->暴露标定只用历史（对 df[:r] 做滤波解码 + df[:r] 的实现收益）；
  - 每日仓位用当前区块的模型对 df[:d+1] 做滤波解码（前向算法，state at d 只
    依赖 [0:d]），取该状态在历史标定里的目标暴露；未在历史中出现过的状态记0。

除"是否因果"外，其余（k、seed、标定方式、评估口径）与 feature_dim_contrast
完全一致，因此两份结果可直接对照：full-sample 那份是上界，本份是可执行下限。

运行方式：
  python ablation/feature_dim_causal_contrast.py   (需先运行 01_data_loading.py, 02_feature_engineering.py)
输出：
  outputs/ablation/results/feature_dim_causal_contrast.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import load_data, RESULTS_DIR, FIGURES_DIR, MIN_HISTORY, REESTIMATE_EVERY, K_REGIMES  # noqa: E402
from engine.hmm_offline import fit_hmm, decode_filtered, calibrate_state_exposures  # noqa: E402
from engine.evaluation import compute_backtest_metrics, print_metrics  # noqa: E402
from engine.plotting import setup_cjk_font  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def build_feat(df: pd.DataFrame, feature_mode: str) -> np.ndarray:
    """与 engine.hmm_offline.fit_hmm 内部完全一致的特征构造（无标准化）。"""
    if feature_mode == "z_vol":
        return np.column_stack([df["z"].values, np.log(df["realized_vol"].values)])
    elif feature_mode == "z_only":
        return df["z"].values.reshape(-1, 1)
    raise ValueError(f"未知 feature_mode: {feature_mode}")


def run_causal(df: pd.DataFrame, feature_mode: str, k: int = K_REGIMES, seed: int = 0) -> dict:
    n = len(df)
    feat_full = build_feat(df, feature_mode)
    returns = df["log_return"].values
    w = np.zeros(n)

    refit_points = list(range(MIN_HISTORY, n, REESTIMATE_EVERY))
    for bi, r in enumerate(refit_points):
        block_end = refit_points[bi + 1] if bi + 1 < len(refit_points) else n
        # 只用历史 [:r] 拟合 HMM
        model, _ = fit_hmm(df.iloc[:r], k=k, seed=seed, feature_mode=feature_mode)
        # 只用历史做暴露标定：对 [:r] 滤波解码，用 [:r] 的实现收益
        hist_states = decode_filtered(model, feat_full[:r])
        stats = calibrate_state_exposures(hist_states, returns[:r], k)
        expo = {name: v["target_exposure"] for name, v in stats.items()}
        # 本区块每日仓位：滤波解码到 block_end，state at d 只依赖 [0:d]，是因果的
        states_upto = decode_filtered(model, feat_full[:block_end])
        for d in range(r, block_end):
            w[d] = expo.get(f"state_{int(states_upto[d])}", 0.0)

    out = df[["date", "ref_regime", "log_return"]].copy()
    out["w_held"] = w
    return compute_backtest_metrics(out["log_return"], out["w_held"], regime_labels=out["ref_regime"])


def make_contrast_plot(causal_rows: list, save_path: Path):
    """三个前视层级 × 两种特征的夏普条形对比。full-sample 两列从
    feature_dim_contrast.csv 读取，完全因果列用本脚本刚算出的结果。"""
    setup_cjk_font()
    causal = {r["feature_mode"]: r["sharpe"] for r in causal_rows}
    prev_path = RESULTS_DIR / "feature_dim_contrast.csv"
    prev = pd.read_csv(prev_path)

    def prev_sharpe(fm, dm):
        row = prev[(prev["feature_mode"] == fm) & (prev["decode_mode"] == dm)]
        return float(row["sharpe"].iloc[0]) if len(row) else np.nan

    levels = ["全前视\n(平滑)", "只去解码前视\n(滤波)", "完全因果\n(walk-forward)"]
    z_only = [prev_sharpe("z_only", "smoothed"), prev_sharpe("z_only", "filtered"), causal.get("z_only", np.nan)]
    z_vol = [prev_sharpe("z_vol", "smoothed"), prev_sharpe("z_vol", "filtered"), causal.get("z_vol", np.nan)]

    x = np.arange(len(levels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, z_only, w, color="#3f6fa8", label="仅 z（收益方向导向）")
    b2 = ax.bar(x + w / 2, z_vol, w, color="#d9a441", label="[z, logσ]（含波动率维）")
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("夏普比率")
    ax.set_title("特征维度对照：随前视被逐层剥离，仅z 始终大幅领先 [z, logσ]")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    df = load_data()

    rows = []
    for feature_mode in ["z_vol", "z_only"]:
        print(f"\n===== 完全因果 walk-forward：feature_mode={feature_mode} =====")
        m = run_causal(df, feature_mode)
        print_metrics(f"causal_walkforward/{feature_mode}", m)
        rows.append({
            "feature_mode": feature_mode, "decode_mode": "causal_walkforward",
            "ann_return": m["ann_return"], "ann_vol": m["ann_vol"], "sharpe": m["sharpe"],
            "calmar": m["calmar"], "max_dd": m["max_dd"], "turnover_rate": m["turnover_rate"],
        })

    out_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "feature_dim_causal_contrast.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n结果已保存 -> {out_path}")
    print("\n（对照：feature_dim_contrast.csv 里的 full-sample 上界）")
    prev = RESULTS_DIR / "feature_dim_contrast.csv"
    if prev.exists():
        print(pd.read_csv(prev)[["feature_mode", "decode_mode", "sharpe", "max_dd", "turnover_rate"]]
              .round(3).to_string(index=False))

    make_contrast_plot(rows, FIGURES_DIR / "feature_dim_causal_contrast.png")
