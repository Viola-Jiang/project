"""
ablation/s1_offline_hmm.py
=============================
§4.2「S1・离线HMM」。

"以全样本估计的两/多状态 HMM、平滑状态序列驱动仓位。该版本隐含使用了未来
信息，不可上线，仅作'区制信息理论上界'的参照。其与 S2 的绩效落差，即量化了
前视偏差的幅度——这是本框架'严格因果'主张的实证支点。"

刻意不做 walk-forward（没有"当前时点"的概念，整段历史一次性看过），因为这
正是S1存在的意义：作为"完全不管前视代价"的参照上限。

【与旧版的区别，及为何没有直接改成zigzag】旧版用HMM在[z, log_sigma]（波动率
特征）上估状态；诊断发现这套特征几乎不含方向信息（1.2区制识别.md）。第一次
修正曾尝试直接用 ref_regime（zigzag规则对全样本的应用）替代HMM状态序列，
但这个改法被否掉了：S1存在的意义是测"HMM这个机制，若拿到含方向的特征、又
允许看未来，上限有多高"；直接套用zigzag（一个确定性规则+全样本前视，本身
就逼近完美分类）测出来的是"完美先知的上限"，跟HMM这个统计模型的能力毫无
关系，偏离了S1本该回答的问题。

正确做法：**只换观测特征，不换机制**——继续用HMM+Viterbi平滑解码，但观测从
[z, log_sigma] 换成 [trend_W, vol_W]（engine.hmm_offline.fit_hmm 的
feature_mode="trend_vol"，与S2~S4现在的区制识别特征同源）。这样S1回答的
仍是"HMM机制本身、给定好特征和全样本前视，能做到多好"，且与S2~S4的特征
基础一致，只是分类机制不同（HMM统计推断 vs BOCPD+规则式原型软分配）——
这一层机制差异目前有意保留，未强行拉齐，S1 vs S2 的落差需要如实说明包含
"因果性"以外，还可能混有"HMM vs BOCPD"这一机制差异。

旧版（HMM on [z, log_sigma]）逻辑仍在，`lookahead_contrast.py` /
`feature_dim_contrast.py` 继续用 feature_mode="z_vol"/"z_only" 做各自独立
的"平滑vs滤波""特征维度"对照实验，结论不受本次改动影响，未改动。

【第二次修正：给trend加统计显著性闸门】换成[trend_W, vol_W]后仍发现问题：
2015年6月~2016年2月（股灾+反弹+熔断，172天）HMM全程判定为同一个隐藏状态——
该状态实际由高波动主导（当期trend_W均值反而是负的），只是历史上高波动期
恰好正收益天数略多、均值刚好为正，凯利公式因此给了它满仓，与"什么都不做"
的S0最大回撤（-48.5%）完全一样。

参考中邮证券《市场脉搏》报告的思路：判断"是否算趋势"应看收益率相对局部
波动的统计显著性（±0.524σ），不是绝对涨跌幅——用绝对幅度做过阈值实验，
无效（危机期间trend_W绝对值本身就大，挡不住）；改用"趋势/噪声比"
（t_stat = trend_W / (vol_W×√W)，engine.features.gate_trend_by_significance）
做闸门后，网格搜索找到阈值1.5为最优：夏普0.43→0.712，最大回撤-48.5%→-15.2%，
2015危机期间状态开始正确切换。此闸门现为 fit_hmm 的默认行为
（gate_threshold默认1.5），本文件显式传参只为清晰起见。

【第三次修正：按波动强度细分标定凯利仓位】诊断脚本
`ablation/diag_s1_interpretability.py` 核查了当前3个状态是否真的可解释：
用中邮报告的规则（方向显著性 x 波动率相对90天滚动均值高低）独立切出6个
纯规则格子，与HMM的3个状态做交叉表。结果：HMM状态在**方向**上完全干净
（bull/bear从不混上涨/下跌天，可解释性成立），但状态内部混着"温和趋势"
和"剧烈趋势"两种波动强度不同的天——按整个状态的平均mu/sigma算凯利分数，
会抹掉二者的差异（"温和上涨"实际凯利分数≈48，"趋势上涨"仅≈23.5，因为
高波动的分母惩罚更大，理论上该更轻仓，之前却被按同一个w*=1.0处理）。

修正做法：不改HMM识别机制、不改K、不改bull/sideways/bear命名，只改标定
凯利仓位这一步——`engine.hmm_offline.calibrate_state_exposures_by_vol`把
每个状态再按`engine.features.classify_high_vol`（90天滚动均值）分高/低
两档波动，分别算mu/sigma/凯利分数标定仓位，当天用哪一档由当天自己的波动
强度决定。`fit_offline_hmm_positions`新增`vol_split_window`参数（默认90，
即生产默认开启；传None关闭退回旧行为）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import (  # noqa: E402
    load_data, RESULTS_DIR, FIGURES_DIR, K_REGIMES_MULTIAXIS, TREND_VOL_WINDOW,
    EXTREMITY_M, EXTREMITY_LOOKBACK, VOL_LOOKBACK,
)
from engine.hmm_offline import fit_offline_hmm_positions  # noqa: E402
from engine.evaluation import compute_backtest_metrics, print_metrics  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def generate_positions(df: pd.DataFrame, k: int = K_REGIMES_MULTIAXIS, seed: int = 0):
    """
    对整段历史一次性拟合离线HMM（观测用[t_stat, extremity, vol_level]三维特征，
    与S2~S4现在的区制识别特征同源，见1.3文档第9~13点）并给出平滑仓位序列。

    vol_split_window传None：vol_level已经是HMM观测的独立一维，不需要旧版
    "按状态再切高/低波动"这个事后补丁（那是为了弥补旧版[trend_W,vol_W]
    二维特征分辨率不够设计的，现在的三维特征已经内含这个信息）。
    返回: (DataFrame['date','ref_regime','log_return','w_held'], state_stats)
    """
    w_target, state_stats = fit_offline_hmm_positions(
        df, k=k, seed=seed, feature_mode="multiaxis", trend_vol_window=TREND_VOL_WINDOW,
        vol_split_window=None, extremity_m=EXTREMITY_M, extremity_lookback=EXTREMITY_LOOKBACK,
        vol_lookback=VOL_LOOKBACK)
    out = df[["date", "ref_regime", "log_return"]].copy()
    out["w_held"] = w_target.values
    return out, state_stats


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
    axes[0].plot(result_df["date"], equity, color="#8a4fd9", lw=1.3, label="S1离线HMM(不可执行,前视上界)")
    axes[0].set_ylabel("净值(对数轴)")
    axes[0].set_yscale("log")
    axes[0].set_title("S1：离线平滑HMM给出的仓位（用了未来信息，仅作理论上界）")
    axes[0].legend(fontsize=9)

    axes[1].plot(result_df["date"], result_df["w_held"], color="#8a4fd9", lw=1)
    axes[1].set_ylabel("HMM平滑仓位")
    axes[1].set_xlabel("日期")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    df = load_data()
    result_df, state_stats = generate_positions(df)

    print("=== S1 离线HMM（[t_stat, extremity, vol_level]三维特征）===")
    print("各隐藏状态的收益统计与标定暴露（w_held实际使用这一档；vol_level已是"
          "观测的独立一维，不再需要旧版按波动强度事后再切档）：")
    for name, s in state_stats.items():
        print(f"  {name}: mu={s['mu']:.5f}, sigma={s['sigma']:.5f}, w*={s['target_exposure']:.2f}")

    metrics = compute_backtest_metrics(result_df["log_return"], result_df["w_held"],
                                        regime_labels=result_df["ref_regime"])
    print_metrics("S1-离线HMM", metrics)
    print("分区制绩效:")
    for name, s in metrics["by_regime"].items():
        print(f"  {name}: 年化收益={s['ann_return']*100:.2f}%  夏普={s['sharpe']:.2f}  n={s['n_obs']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(RESULTS_DIR / "s1_offline_hmm.csv", index=False)
    make_diagnostic_plot(result_df, FIGURES_DIR / "s1_offline_hmm.png")
