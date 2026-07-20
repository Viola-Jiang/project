"""
ablation/02_feature_engineering.py
===================================
§2「数据与特征构建」。

真实执行链路（ablation/）的第二步，紧接 01_data_loading.py。

特征（对应实现 preprocess()）：
  1. 对数收益      r_t = ln(P_t / P_{t-1})
  2. 已实现波动    sigma_t = std(r_{t-w+1:t}), w = 20
  3. 波动归一化收益 z_t = r_t / sigma_t   <- BOCPD 的实际"发射观测"

真实数据没有上帝视角的区制标签，本脚本额外调用
engine/regime_labeling.auto_label_regimes 产出 ref_regime/ref_regime_age
两列——这是离线全样本HMM给出的**参照标签，不是真值**，只供 validation/
与 ablation/ 的诊断/评估使用（详见该模块 docstring 的边界说明）。

运行方式：
  python ablation/02_feature_engineering.py   (需先运行 01_data_loading.py)
输出：
  data/features.csv
  outputs/ablation/figures/02_feature_engineering.png

注：核心特征计算逻辑已抽到 engine/features.py，因为需要被 validation/ 与
ablation/ 的多个脚本复用，本脚本只负责：对主线数据调用一次、存csv、画诊断图。
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "outputs" / "ablation" / "figures"
sys.path.insert(0, str(REPO_ROOT))

from engine.features import preprocess  # noqa: E402
from engine.zigzag_labeling import zigzag_label_regimes  # noqa: E402
from engine.plotting import setup_cjk_font, REGIME_COLORS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def sanity_checks(df: pd.DataFrame) -> None:
    print("特征缺失值统计（前19行因滚动窗口不足产生 NaN，属预期行为）：")
    print(df[["log_return", "realized_vol", "z"]].isna().sum())

    valid = df.dropna(subset=["z"])
    print(f"\n有效样本数（去除warm-up期）: {len(valid)} / {len(df)}")
    print("\n全样本描述统计：")
    print(valid[["log_return", "z"]].describe().round(4))

    print("\n分区制标准差对比（验证归一化是否抑制了跨区制的波动差异；"
          "分组用的 ref_regime 是自动标注参照，非真值）：")
    cmp = valid.groupby("ref_regime").agg(r_std=("log_return", "std"), z_std=("z", "std")).round(4)
    print(cmp)
    r_cv = cmp["r_std"].std() / cmp["r_std"].mean()
    z_cv = cmp["z_std"].std() / cmp["z_std"].mean()
    print(f"\n区制间标准差的变异系数：r_t: {r_cv:.3f}  |  z_t: {z_cv:.3f}")
    print("（z_t 的变异系数更低，说明归一化确实压抑了跨区制的条件异方差）"
          if z_cv < r_cv else "（注意：本次结果中 z_t 并未比 r_t 更平稳，需检查窗口设定或数据）")


def make_diagnostic_plot(df: pd.DataFrame, save_path: Path):
    setup_cjk_font()
    valid = df.dropna(subset=["z"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for regime, c in REGIME_COLORS.items():
        sub = valid[valid["ref_regime"] == regime]["log_return"]
        axes[0].hist(sub, bins=40, alpha=0.5, density=True, color=c, label=regime)
    axes[0].set_title("归一化前：r_t 按自动标注参照分组的分布（尺度明显不同）")
    axes[0].set_xlabel("log_return r_t")
    axes[0].legend()

    for regime, c in REGIME_COLORS.items():
        sub = valid[valid["ref_regime"] == regime]["z"]
        axes[1].hist(sub, bins=40, alpha=0.5, density=True, color=c, label=regime)
    axes[1].set_title("归一化后：z_t 按自动标注参照分组的分布（尺度趋于一致）")
    axes[1].set_xlabel("z_t = r_t / sigma_t")
    axes[1].legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    raw = pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["date"])
    features = preprocess(raw)

    # zigzag_label_regimes 只需要价格，不受 z/realized_vol 的warm-up期限制；
    # 但仍只对warm-up期之后的有效样本做标注、按原始index对齐拼回去，
    # 保持 ref_regime/ref_regime_age 的NaN范围与 log_return/realized_vol/z
    # 一致，维持与此前HMM版本一致的行为边界，不引入额外信息。
    valid_mask = features["z"].notna() & features["realized_vol"].notna()
    input_df = features.loc[valid_mask].reset_index(drop=True).rename(columns={"price": "close"})
    labeled_valid = zigzag_label_regimes(input_df)
    features["ref_regime"] = pd.NA
    features["ref_regime_age"] = pd.NA
    features.loc[valid_mask, "ref_regime"] = labeled_valid["ref_regime"].values
    features.loc[valid_mask, "ref_regime_age"] = labeled_valid["ref_regime_age"].values

    out_path = DATA_DIR / "features.csv"
    features.to_csv(out_path, index=False)
    print(f"特征已保存 -> {out_path}\n")

    sanity_checks(features)
    make_diagnostic_plot(features, FIGURES_DIR / "02_feature_engineering.png")
