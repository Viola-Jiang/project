"""
ablation/feature_dim_contrast.py
====================================
补充对照实验：隔离"特征维度不同"这一个变量。

背景：S1（离线HMM，理论上界）和 engine/regime_labeling.py（产出`ref_regime`
参照标签）用的都是 [z, log(realized_vol)] 两维特征，但主链路（S2~S4实际跑的
engine/emission.py + engine/bocpd.py）只用一维 z——发射观测口径本身不对等。
run_ablation_summary.py 里的 S1 vs S2、以及 lookahead_contrast.py 里的
平滑vs滤波对照，都是在"两维特征"这个设定下测的，从未把"特征维度"本身当成
一个单独变量隔离出来看过。

本脚本用同一套 GaussianHMM 机制（engine.hmm_offline.fit_hmm 新增的
feature_mode 参数），交叉对比 {z_vol, z_only} x {smoothed, filtered} 四种
组合，其余（k、seed、状态->暴露标定方式、评估口径）全部锁死，只让"特征
维度"和"解码时看没看未来"这两个变量各自独立变化：
  z_vol  + smoothed  ：S1 实际配置（信息量最大、含前视）
  z_only + smoothed  ：只去掉多出来的那一维特征，前视仍然保留
  z_vol  + filtered  ：只去掉前视，特征维度仍然是两维
  z_only + filtered  ：两个变量都拿掉，最接近主链路S2实际能拿到的信息

z_vol+smoothed 与 z_only+smoothed 之间的夏普落差，就是"纯粹因为多看了一维
log(realized_vol)"贡献的绩效，与前视无关；filtered 那一行同理。

运行方式：
  python ablation/feature_dim_contrast.py   (需先运行 ablation/01_data_loading.py, 02_feature_engineering.py)
输出：
  outputs/ablation/results/feature_dim_contrast.csv
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ablation.common import load_data, RESULTS_DIR, K_REGIMES  # noqa: E402
from engine.hmm_offline import fit_hmm, decode_smoothed, decode_filtered, calibrate_state_exposures  # noqa: E402
from engine.evaluation import compute_backtest_metrics, print_metrics  # noqa: E402


def run_one(df: pd.DataFrame, feature_mode: str, decode_mode: str, k: int = K_REGIMES, seed: int = 0) -> dict:
    """
    拟合一个指定feature_mode的HMM，用指定decode_mode解码，套用该HMM自己的
    状态->暴露标定（按各自的状态划分各自算区制条件收益，不跨组合共用标定，
    因为z_only下的状态编号/含义本来就和z_vol下不是同一回事）。
    """
    model, feat = fit_hmm(df, k=k, seed=seed, feature_mode=feature_mode)
    state_seq = decode_smoothed(model, feat) if decode_mode == "smoothed" else decode_filtered(model, feat)
    state_stats = calibrate_state_exposures(state_seq, df["log_return"].values, k)

    out = df[["date", "ref_regime", "log_return"]].copy()
    out["w_held"] = [state_stats[f"state_{s}"]["target_exposure"] for s in state_seq]

    metrics = compute_backtest_metrics(out["log_return"], out["w_held"], regime_labels=out["ref_regime"])
    return metrics


if __name__ == "__main__":
    df = load_data()

    print("=== 特征维度对照实验：{z_vol, z_only} x {smoothed, filtered} ===\n")

    rows = []
    for feature_mode in ("z_vol", "z_only"):
        for decode_mode in ("smoothed", "filtered"):
            label = f"{feature_mode}+{decode_mode}"
            metrics = run_one(df, feature_mode, decode_mode)
            print_metrics(label, metrics)
            rows.append({
                "feature_mode": feature_mode, "decode_mode": decode_mode,
                "ann_return": metrics["ann_return"], "ann_vol": metrics["ann_vol"],
                "sharpe": metrics["sharpe"], "calmar": metrics["calmar"],
                "max_dd": metrics["max_dd"], "turnover_rate": metrics["turnover_rate"],
            })

    result_df = pd.DataFrame(rows)

    sharpe = {(r["feature_mode"], r["decode_mode"]): r["sharpe"] for r in rows}
    feat_gap_smoothed = sharpe[("z_vol", "smoothed")] - sharpe[("z_only", "smoothed")]
    feat_gap_filtered = sharpe[("z_vol", "filtered")] - sharpe[("z_only", "filtered")]
    lookahead_gap_zvol = sharpe[("z_vol", "smoothed")] - sharpe[("z_vol", "filtered")]
    lookahead_gap_zonly = sharpe[("z_only", "smoothed")] - sharpe[("z_only", "filtered")]

    print("\n" + "=" * 70)
    print(f"纯特征维度落差（smoothed下）  = {feat_gap_smoothed:.3f} 个夏普点")
    print(f"纯特征维度落差（filtered下）  = {feat_gap_filtered:.3f} 个夏普点")
    print(f"纯前视落差（z_vol下，对照lookahead_contrast.py） = {lookahead_gap_zvol:.3f} 个夏普点")
    print(f"纯前视落差（z_only下）        = {lookahead_gap_zonly:.3f} 个夏普点")
    print("=" * 70)
    print("S1实际配置是z_vol+smoothed；主链路(S2~S4)信息量对应z_only。"
          "z_vol+smoothed 相对 z_only+filtered 的总落差，可以按上面四个数"
          "拆解成'前视'和'特征维度'两部分，不再笼统地全算在'前视'头上。")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(RESULTS_DIR / "feature_dim_contrast.csv", index=False)
    print(f"\n结果已保存 -> {RESULTS_DIR / 'feature_dim_contrast.csv'}")
