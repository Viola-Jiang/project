"""
feature_validation/plot_s1_hmm_states.py
============================================
展示S1（离线HMM，feature_mode="multiaxis"，K=4）自己的Viterbi平滑状态序列——
不是跟ref_regime对照，是模型自己把每天分到了哪个隐藏状态，背景按状态上色，
叠加价格曲线看这个划分跟直觉上的走势对不对得上。
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.hmm_offline import fit_hmm, decode_smoothed, calibrate_state_exposures  # noqa: E402
from engine.plotting import setup_cjk_font  # noqa: E402

K = 4
STATE_COLORS = ["#3f6fa8", "#e8b84b", "#d94f4f", "#5a9367"]


def main():
    from ablation.common import load_data, TREND_VOL_WINDOW, EXTREMITY_M, EXTREMITY_LOOKBACK, VOL_LOOKBACK  # noqa: E402
    df = load_data().reset_index(drop=True)

    model, feat = fit_hmm(df, k=K, feature_mode="multiaxis", trend_vol_window=TREND_VOL_WINDOW,
                           extremity_m=EXTREMITY_M, extremity_lookback=EXTREMITY_LOOKBACK,
                           vol_lookback=VOL_LOOKBACK)
    state_seq = decode_smoothed(model, feat)
    state_stats = calibrate_state_exposures(state_seq, df["log_return"].values, K)

    print("各状态天数/收益统计/标定暴露：")
    for name, s in state_stats.items():
        n = int((state_seq == int(name.split("_")[1])).sum())
        print(f"  {name}: n={n}, mu={s['mu']:.5f}, sigma={s['sigma']:.5f}, w*={s['target_exposure']:.2f}")

    setup_cjk_font()
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

    seg_id = (state_seq != np.roll(state_seq, 1)).cumsum()
    import pandas as pd
    plot_df = pd.DataFrame({"date": df["date"], "state": state_seq, "price": df["price"], "seg": seg_id})
    for _, seg in plot_df.groupby("seg"):
        color = STATE_COLORS[seg["state"].iloc[0]]
        for ax in axes:
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1], color=color, alpha=0.25, lw=0)

    axes[0].plot(df["date"], df["price"], color="black", lw=0.7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("价格(对数轴)")
    axes[0].set_title("S1 HMM（multiaxis, K=4）Viterbi平滑状态序列，背景色=模型自己分的状态")

    for k in range(K):
        mask = state_seq == k
        axes[1].scatter(df["date"][mask], np.full(mask.sum(), k), s=2, color=STATE_COLORS[k])
    axes[1].set_yticks(range(K))
    axes[1].set_ylabel("隐藏状态编号")
    axes[1].set_xlabel("日期")

    fig.tight_layout()
    out = REPO_ROOT / "outputs" / "feature_validation" / "s1_hmm_states.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"图已保存 -> {out}")

    # 散点图版本：跟KMeans那几张对照，同样的[t_stat, extremity, vol_level]
    # 两两投影，这次按HMM自己的Viterbi状态上色（feat是fit_hmm内部标准化后
    # 的特征矩阵，跟状态序列一一对应）。
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [("t_stat", feat[:, 0], "extremity", feat[:, 1]),
             ("t_stat", feat[:, 0], "vol_level", feat[:, 2]),
             ("extremity", feat[:, 1], "vol_level", feat[:, 2])]
    for ax, (xname, x, yname, y) in zip(axes2, pairs):
        for k in range(K):
            mask = state_seq == k
            ax.scatter(x[mask], y[mask], s=5, alpha=0.4, color=STATE_COLORS[k], label=f"state_{k}")
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
    axes2[0].legend(fontsize=8, markerscale=2)
    fig2.suptitle("S1 HMM（multiaxis, K=4）Viterbi状态的两两投影散点图")
    fig2.tight_layout()
    out2 = REPO_ROOT / "outputs" / "feature_validation" / "s1_hmm_states_scatter.png"
    fig2.savefig(out2, dpi=130)
    plt.close(fig2)
    print(f"散点图已保存 -> {out2}")


if __name__ == "__main__":
    main()
