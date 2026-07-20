"""
validation/zigzag_label_selfcheck.py
=======================================
对 engine/zigzag_labeling.py（申万宏源 Zig-Zag + Binseg 规则式参照标签）
做 1.1 文档"自检"一节要求的检查：

  A. 分区制统计表：三态各自的年化收益/日涨占比/年化波动/段数/平均段长，
     底线：年化收益 bull > sideways > bear 且 bear < 0（对照现有 HMM 版
     标签 bear=+4.4% 的反例）。
  B. 叠图：标签色块叠在中证800全收益净值曲线上，肉眼核对 bull 段是否
     真在涨、bear 段是否真在跌；原文提到 2018 年应为持续下降趋势，可直接核对。
  C. 与现有 HMM 版 ref_regime 对比：一致率与分歧分布。
  D. 参数网格搜索（可选，--grid 开启）：原文参数是对上证指数2015+标定的，
     换到中证800全收益2009-2026 未必合适；按"方向分离"标准打分给出排名，
     供人工结合叠图选择，不自动替换默认参数。

运行方式：
  python validation/zigzag_label_selfcheck.py          # A+B+C
  python validation/zigzag_label_selfcheck.py --grid   # additionally D（较慢）
输出：
  outputs/validation/results/zigzag_label_stats.csv
  outputs/validation/results/zigzag_label_grid.csv     （--grid 时）
  outputs/validation/figures/zigzag_label_selfcheck.png
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

from engine.zigzag_labeling import zigzag_label_regimes, regime_stats, score_labeling, grid_search_params  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def load_data() -> pd.DataFrame:
    """价格 + 已有特征（含现有HMM版ref_regime，用于对比）。"""
    prices = (pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["date"])
              .rename(columns={"price": "close"}))
    feats = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["date"])
    df = prices.merge(feats[["date", "log_return", "ref_regime"]]
                      .rename(columns={"ref_regime": "hmm_ref_regime"}),
                      on="date", how="left")
    return df.sort_values("date").reset_index(drop=True)


def check_stats(labeled: pd.DataFrame) -> pd.DataFrame:
    stats = regime_stats(labeled)
    print("=== A. 分区制统计表（规则式标签）===")
    print(stats.round(4).to_string())
    ok_order = (stats["ann_return"].get("bull", np.nan) > stats["ann_return"].get("sideways", np.nan)
                > stats["ann_return"].get("bear", np.nan))
    ok_bear = stats["ann_return"].get("bear", np.nan) < 0
    print(f"\n底线检查：年化收益排序 bull>sideways>bear -> {'通过' if ok_order else '不通过'}"
          f"；bear年化<0 -> {'通过' if ok_bear else '不通过'}")
    print(f"综合得分 score_labeling = {score_labeling(stats):.4f}\n")
    return stats


def compare_with_hmm(labeled: pd.DataFrame):
    both = labeled.dropna(subset=["hmm_ref_regime"])
    agree = (both["ref_regime"] == both["hmm_ref_regime"]).mean()
    print("=== C. 与现有HMM版ref_regime对比 ===")
    print(f"逐日一致率: {agree*100:.1f}%")
    print("混淆矩阵（行=规则式，列=HMM版）：")
    print(pd.crosstab(both["ref_regime"], both["hmm_ref_regime"]).to_string(), "\n")


def make_plot(labeled: pd.DataFrame, save_path: Path):
    setup_cjk_font()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    nav = labeled["close"] / labeled["close"].iloc[0]

    for ax, col, title in [(axes[0], "ref_regime", "B. 规则式标签（Zig-Zag+Binseg）叠净值"),
                            (axes[1], "hmm_ref_regime", "对照：现有HMM版ref_regime叠净值")]:
        sub = labeled.dropna(subset=[col])
        seg_id = (sub[col] != sub[col].shift(1)).cumsum()
        for _, seg in sub.groupby(seg_id):
            ax.axvspan(seg["date"].iloc[0], seg["date"].iloc[-1],
                       color=REGIME_COLORS.get(seg[col].iloc[0], "lightgray"), alpha=0.25, lw=0)
        ax.plot(labeled["date"], nav, color="black", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel("净值(对数轴)")
        ax.set_title(title)

    handles = [plt.Rectangle((0, 0), 1, 1, color=REGIME_COLORS[k], alpha=0.4) for k in REGIME_COLORS]
    axes[0].legend(handles, list(REGIME_COLORS), fontsize=9, loc="upper left")
    axes[1].set_xlabel("日期")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"叠图已保存 -> {save_path}")


if __name__ == "__main__":
    df = load_data()
    labeled = zigzag_label_regimes(df)

    stats = check_stats(labeled)
    compare_with_hmm(labeled)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(RESULTS_DIR / "zigzag_label_stats.csv")
    make_plot(labeled, FIGURES_DIR / "zigzag_label_selfcheck.png")

    if "--grid" in sys.argv:
        print("\n=== D. 参数网格搜索（按方向分离标准打分，仅供参考）===")
        grid = grid_search_params(df)
        grid.to_csv(RESULTS_DIR / "zigzag_label_grid.csv", index=False)
        print(grid.head(10).round(4).to_string(index=False))
        print(f"\n完整结果已保存 -> {RESULTS_DIR / 'zigzag_label_grid.csv'}")
