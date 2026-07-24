"""
feature_validation/validate_features.py
==========================================
验证"强度/路径/透支"三轴、四个候选特征，是否真的对w*_k（未来收益）和H_k(r)（久期）
有区分度——对应 1.3区制分类的本质与约束.md 第9~13点的方法论。

这是一次性的诊断脚本，不接入S2~S5的正式流水线，产出纯粹是"这套特征值不值得用"
的经验证据，供后续决定是否替换现有[trend_W, vol_W]二维特征空间。

四个候选特征（全部直接从原始log_return滚动计算，不经过BOCPD后验，
对应第12点"BOCPD与区制分类解耦"的结论）：
  强度：t_stat = trend_W / (vol_W * sqrt(W))，W=21
  路径：ER（效率比）= |sum(r)| / sum(|r|)，W=21
  路径：autocorr（一阶自相关）= corr(r_t, r_{t-1})，W'=60（比强度轴窗口更长，
        自相关本身是统计量，21天样本对它来说噪声过大）
  透支：extremity = 过去M=60天累计收益，在过去L=500天同口径滚动值里的百分位排名

验证内容：
  1. ER与autocorr的相关性——两者都是"路径轴"候选，若高度相关，说明四维空间里
     "路径"这个概念被计了两次权重，需要在后续设计里处理（只留一个/降维），
     现在先如实报告相关系数，不擅自二选一
  2. 强度类特征（t_stat, ER）对未来收益的IC——用Spearman秩相关，分别测
     未来5/21/63个交易日的累计收益
  3. 路径类特征（autocorr）、透支类特征（extremity）对"这一段还能持续多久"
     的预测力——用ref_regime（zigzag规则标注，与本次候选特征的计算方式无关，
     可以当独立的历史分段参照）切出的连续段做样本，每段取段内特征均值，
     和该段真实持续天数做Spearman相关。段末尾如果是数据截止时仍未结束的那一段
     （右截断），按1.3文档第13点的处理方式剔除，不当完整段计入
  4. 标准化后跑KMeans：K=2..6做轮廓系数扫描（看K=3是否仍是合理选择），
     并报告K=3时四维特征的簇内均值、簇大小、与ref_regime的交叉分布（仅供参考，
     不作为聚类"对不对"的判断标准——按1.3文档第4点，判断标准是分组是否让
     w*_k/H_k(r)产生实质区分，不是跟ref_regime像不像）

数据: regime_timing_engine/data/features.csv（同ablation.common.load_data()）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, percentileofscore, mannwhitneyu, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.features import rolling_trend_vol_features  # noqa: E402

STRENGTH_WINDOW = 21     # 强度轴、ER窗口，沿用既有约定
PATH_WINDOW = 60         # 自相关窗口，比21天长，估计更稳
EXTREMITY_M = 60         # 透支轴：当前累计收益的窗口
EXTREMITY_L = 500        # 透支轴：历史参照基准的回看长度（约2年）
FORWARD_HORIZONS = (5, 21, 63)  # IC检验用的未来收益窗口（交易日）


# ------------------------------- 特征计算 -------------------------------

def compute_efficiency_ratio(log_returns: np.ndarray, window: int) -> np.ndarray:
    """
    效率比 ER = |窗口内净收益| / 窗口内每日收益绝对值之和。
    ER→1：窗口内几乎单调、路径直；ER→0：来回震荡、净进展很小。
    因果滚动窗口，min_periods与rolling_trend_vol_features保持一致的宽松策略
    （样本不足时用能取到的最早有效值填充，不产生NaN）。
    """
    s = pd.Series(log_returns)
    min_periods = max(5, window // 2)
    net = s.rolling(window, min_periods=min_periods).sum()
    gross = s.rolling(window, min_periods=min_periods).apply(lambda w: np.abs(w).sum(), raw=True)
    gross = gross.replace(0, np.nan)
    er = (net.abs() / gross).bfill().fillna(0.0)
    return er.to_numpy()


def compute_rolling_autocorr(log_returns: np.ndarray, window: int) -> np.ndarray:
    """
    一阶自相关 corr(r_t, r_{t-1})，滚动窗口计算，窗口需比强度轴更长
    （见模块docstring）。样本不足或窗口内方差为0时填0（视为"看不出自相关"，
    不是"负相关"，用0代表信息不足更保守）。
    """
    s = pd.Series(log_returns)
    lagged = s.shift(1)
    min_periods = max(10, window // 2)

    def _autocorr(x):
        a, b = x[:-1], x[1:]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    out = np.full(len(s), np.nan)
    valid = s.notna() & lagged.notna()
    combined = s.where(valid)
    for i in range(len(s)):
        if i < min_periods:
            continue
        start = max(0, i - window + 1)
        window_vals = combined.iloc[start:i + 1].to_numpy()
        if np.isnan(window_vals).any() or len(window_vals) < min_periods:
            continue
        out[i] = _autocorr(window_vals)
    out = pd.Series(out).bfill().fillna(0.0).to_numpy()
    return out


def compute_extremity_percentile(log_returns: np.ndarray, m: int, lookback: int) -> np.ndarray:
    """
    透支轴：过去M天累计对数收益，在过去L天内同口径滚动值的历史分布中的
    百分位排名（0~100）。用"这段涨幅相对历史上同类涨法通常有多大"衡量
    是否走过头，而不是用绝对价格水平（避免长期结构性上涨的指数一直贴着
    100%分位、看不出区分度——见1.3文档第13点）。

    因果：第t天的值只用cum_ret[t-lookback+1 : t+1]（含自身），不看未来。
    """
    s = pd.Series(log_returns)
    min_periods_cum = max(5, m // 2)
    cum_ret = s.rolling(m, min_periods=min_periods_cum).sum()

    out = np.full(len(s), np.nan)
    for i in range(len(s)):
        if np.isnan(cum_ret.iloc[i]):
            continue
        start = max(0, i - lookback + 1)
        window_vals = cum_ret.iloc[start:i + 1].dropna().to_numpy()
        if len(window_vals) < min_periods_cum:
            continue
        out[i] = percentileofscore(window_vals, cum_ret.iloc[i], kind="rank")
    out = pd.Series(out).bfill().fillna(50.0).to_numpy()
    return out


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    log_returns = df["log_return"].fillna(0.0).to_numpy()

    trend, vol = rolling_trend_vol_features(log_returns, window=STRENGTH_WINDOW)
    t_stat = trend / (vol * np.sqrt(STRENGTH_WINDOW))

    er = compute_efficiency_ratio(log_returns, window=STRENGTH_WINDOW)
    autocorr = compute_rolling_autocorr(log_returns, window=PATH_WINDOW)
    extremity = compute_extremity_percentile(log_returns, m=EXTREMITY_M, lookback=EXTREMITY_L)

    out = df.copy()
    out["feat_t_stat"] = t_stat
    out["feat_er"] = er
    out["feat_autocorr"] = autocorr
    out["feat_extremity"] = extremity
    return out


# ------------------------------- 验证1：路径轴内部相关性 -------------------------------

def check_path_axis_redundancy(df: pd.DataFrame) -> None:
    print("\n[验证1] ER 与 自相关 的相关性（两者都是路径轴候选，检查是否重复计权）")
    rho, p = spearmanr(df["feat_er"], df["feat_autocorr"])
    print(f"  Spearman(ER, autocorr) = {rho:.3f}  (p={p:.4g})")
    if abs(rho) > 0.7:
        print("  -> 相关性较高，四维空间里路径轴事实上占了两个维度，"
              "后续聚类前建议只保留一个，或对两者做进一步降维。")
    else:
        print("  -> 相关性不高，两者提供的信息有区分，保留两者问题不大。")


# ------------------------------- 验证2：强度/路径特征对未来收益的IC -------------------------------

def check_return_ic(df: pd.DataFrame) -> None:
    print("\n[验证2] 强度/路径类特征 对未来收益的IC（Spearman秩相关）")
    log_returns = df["log_return"].fillna(0.0).to_numpy()
    n = len(df)
    for horizon in FORWARD_HORIZONS:
        fwd = np.full(n, np.nan)
        cum = pd.Series(log_returns).rolling(horizon).sum().to_numpy()
        fwd[:n - horizon] = cum[horizon:]
        valid = ~np.isnan(fwd)
        print(f"  未来{horizon}日累计收益:")
        for col, label in [("feat_t_stat", "t_stat(强度)"), ("feat_er", "ER(路径)"),
                            ("feat_autocorr", "autocorr(路径)"), ("feat_extremity", "extremity(透支)")]:
            rho, p = spearmanr(df[col][valid], fwd[valid])
            flag = "  <-- 有意义" if abs(rho) > 0.05 and p < 0.01 else ""
            print(f"    IC({label:16s}) = {rho:+.4f}  (p={p:.4g}){flag}")


# ------------------------------- 验证3：用候选特征自己聚出来的簇，检验w*/久期区分度 -------------------------------
# 不再用ref_regime（zigzag规则划分）当真值——那是我们已经放弃的分类标准，用它测新
# 特征的预测力，测的其实是"新特征像不像旧分类"，不是"新特征能不能区分w*_k/H_k"。
# 正确做法：直接用这四个候选特征自己跑出来的KMeans簇（同一簇的连续天数当一段），
# 检验这些簇彼此之间的收益特征、久期特征是不是真的不同——这才是第10点要测的
# "联合充分性"本身。


def debounce_labels(labels: pd.Series, min_dwell: int) -> pd.Series:
    """
    去抖动平滑：验证4发现逐日硬聚类会疯狂抖动（段长中位数只有3~4天），
    久期因此测不出区分度——不是特征选得不好，是分类本身缺乏持续性机制。

    这里用最简单的"去抖动"处理，替代真正的hazard持续性/无交易带机制：
    只有当新标签连续出现满min_dwell天，才真正确认切换到新标签；在那之前
    （哪怕已经出现1~min_dwell-1天的新标签），仍然沿用上一个已确认的标签。
    """
    raw = labels.to_numpy()
    n = len(raw)
    confirmed = np.empty(n, dtype=raw.dtype)
    current = raw[0]
    confirmed[0] = current
    candidate, candidate_count = None, 0
    for i in range(1, n):
        if raw[i] == current:
            candidate, candidate_count = None, 0
        else:
            if raw[i] == candidate:
                candidate_count += 1
            else:
                candidate, candidate_count = raw[i], 1
            if candidate_count >= min_dwell:
                current = candidate
                candidate, candidate_count = None, 0
        confirmed[i] = current
    return pd.Series(confirmed, index=labels.index)


def extract_segments_from_cluster_labels(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """
    用聚类标签（而不是ref_regime）切出连续段：同一簇标签连续出现的这几天算一段。
    右截断处理同1.3文档第13点：数据末尾仍未结束的最后一段剔除，不当完整段计入。
    """
    labels = df[cluster_col]
    seg_id = (labels != labels.shift(1)).cumsum()
    records = []
    for sid, g in df.groupby(seg_id):
        start, end = g.index[0], g.index[-1]
        records.append({"cluster": labels.loc[start], "start": start, "end": end,
                         "duration": end - start + 1,
                         "mean_log_return": df["log_return"].loc[start:end].mean(),
                         "std_log_return": df["log_return"].loc[start:end].std()})
    seg_df = pd.DataFrame(records).sort_values("start").reset_index(drop=True)
    n_before = len(seg_df)
    last_idx = df.index[-1]
    seg_df = seg_df.loc[seg_df["end"] != last_idx].reset_index(drop=True)
    print(f"  (聚类标签切出{n_before}段，剔除数据末尾右截断的{n_before - len(seg_df)}段，"
          f"剩余{len(seg_df)}段)")
    return seg_df


def check_pairwise_returns(df: pd.DataFrame, cluster_col: str) -> tuple[int, int]:
    """
    Kruskal-Wallis只回答"K组里是不是至少有一组不一样"，不回答具体哪两组不一样。
    这里对所有C(K,2)对做两两Mann-Whitney U检验，并用Bonferroni校正
    （阈值除以对数，因为同时做多次检验会推高假阳性率），报告校正后有几对显著。
    """
    clusters = sorted(df[cluster_col].unique())
    pairs = [(a, b) for i, a in enumerate(clusters) for b in clusters[i + 1:]]
    n_pairs = len(pairs)
    alpha_corrected = 0.05 / n_pairs
    print(f"\n  两两比较（{len(clusters)}个簇，共{n_pairs}对，Bonferroni校正阈值={alpha_corrected:.4g}）:")
    n_sig = 0
    for a, b in pairs:
        ga = df.loc[df[cluster_col] == a, "log_return"].dropna()
        gb = df.loc[df[cluster_col] == b, "log_return"].dropna()
        stat, p = mannwhitneyu(ga, gb, alternative="two-sided")
        sig = p < alpha_corrected
        n_sig += int(sig)
        print(f"    簇{a} vs 簇{b}: p={p:.4g}{'  <-- 显著' if sig else ''}")
    print(f"  {n_sig}/{n_pairs} 对显著（校正后）")
    return n_sig, n_pairs


def check_daily_hazard_regression(df: pd.DataFrame, cluster_col: str) -> float:
    """
    久期检验的另一种办法：不再把每段压缩成一个"段长"数字（只有几十个样本、
    检验功效很低），改成"离散时间风险回归"——把每一个"仍在段内、尚未结束"的
    交易日都当一个观测：event=1表示"这段明天就要结束了"，event=0表示"还没结束"。
    这样样本量是全部在段内的天数（几千个），而不是段的数量（几十个），
    统计功效高得多，也更贴近H_k(r)本身的定义（给定活到今天，明天结束的条件概率）。

    数据末尾仍未结束的那一段，所有天都标记event=0（真实是否结束未知，
    right-censored），不整段剔除——这是充分利用截断信息的标准处理方式，
    比"分段检验"里直接丢弃末尾段更充分。

    预测变量只有cluster这一个分类变量、没有连续协变量，所以"event ~ C(cluster)"
    这个逻辑回归是饱和模型，等价于直接比较各簇"当日结束"这个二值结果的发生率
    是否相同——这就是标准的卡方齐性检验（K个组 × {结束,未结束} 列联表），
    数学上跟逻辑回归的似然比检验结果一致，不需要额外的回归框架。
    """
    labels = df[cluster_col]
    seg_id = (labels != labels.shift(1)).cumsum()
    last_idx = df.index[-1]

    records = []
    for sid, g in df.groupby(seg_id):
        start, end = g.index[0], g.index[-1]
        is_censored = end == last_idx
        cluster_val = labels.loc[start]
        for i, day_idx in enumerate(range(start, end + 1)):
            is_last_day = day_idx == end
            event = 0 if is_censored else int(is_last_day)
            records.append({"cluster": cluster_val, "age": i + 1, "event": event})
    hazard_df = pd.DataFrame(records)

    print(f"\n  离散时间风险回归（{len(hazard_df)}个'天'级观测，"
          f"其中{hazard_df['event'].sum()}个是'该段最后一天'的事件）:")

    contingency = pd.crosstab(hazard_df["cluster"], hazard_df["event"])
    chi2_stat, p_value, dof, _ = chi2_contingency(contingency)
    print(f"  卡方齐性检验（各簇当日结束风险是否相同，等价于event~C(cluster)逻辑回归的似然比检验）: "
          f"chi2={chi2_stat:.2f}, df={dof}, p={p_value:.4g}"
          f"{'  <-- 显著' if p_value < 0.05 else '  <-- 未见显著差异'}")

    implied_hazard = hazard_df.groupby("cluster")["event"].mean()
    print("  各簇的粗略日均结束风险（事件数/该簇天数，仅描述性参考）:")
    print(implied_hazard.round(5).to_string())
    return p_value


SMOOTH_WINDOW = 10  # 单特征平滑窗口（滚动均值），减少特征本身的逐日抖动


def smooth_series(s: pd.Series, window: int) -> pd.Series:
    """因果滚动均值平滑，min_periods宽松处理，跟其它特征保持一致的边界填充策略。"""
    min_periods = max(3, window // 2)
    return s.rolling(window, min_periods=min_periods).mean().bfill()


def single_feature_diagnostic(df: pd.DataFrame) -> None:
    """
    退一步，不再一次性把四个特征揉进一个KMeans——先看每个特征单独平滑之后，
    自己最多能在"对应目标"上撑起几类分位分组（两两显著区分的比例是否过半，
    对应用户"至少一半组对能显著区分"的门槛）。

    不预设"哪个特征该服务哪个目标"——4个候选特征（t_stat, ER, autocorr,
    extremity）全部各自对w*_k口径（同期log_return，复用check_pairwise_returns）
    和H_k口径（当日结束风险，复用check_daily_hazard_regression）都测一遍，
    交叉表补满，核心目标是"区分w*_k和H_k"本身，不是验证之前的轴划分假设。

    按分位数（qcut）切分，而不是KMeans——单变量场景下分位分组就是最自然的
    "分几类"方式，不需要引入聚类算法。
    """
    print("\n" + "=" * 70)
    print("单特征诊断：4个特征 × 2个目标(w*_k/H_k)，全部交叉测一遍")
    print("=" * 70)

    summary = []
    all_cols = ("feat_t_stat", "feat_er", "feat_autocorr", "feat_extremity")

    for col in all_cols:
        smoothed = smooth_series(df[col], SMOOTH_WINDOW)

        print(f"\n--- {col}（平滑窗口={SMOOTH_WINDOW}天）对 w*_k（同期收益） ---")
        best_k_return = None
        for k in (2, 3, 4, 5):
            bins = pd.qcut(smoothed, k, labels=False, duplicates="drop")
            if bins.nunique() < k:
                print(f"  K={k}: 分位点重复，实际只分出{bins.nunique()}类，跳过")
                continue
            tmp = df.copy()
            tmp["_bin"] = bins
            print(f"  K={k}:")
            n_sig, n_pairs = check_pairwise_returns(tmp, "_bin")
            if n_sig >= n_pairs / 2:
                best_k_return = k
        summary.append((col, "w*_k(过半组对显著)", best_k_return))

        print(f"\n--- {col}（平滑窗口={SMOOTH_WINDOW}天）对 H_k（当日结束风险） ---")
        best_k_duration = None
        for k in (2, 3, 4, 5):
            bins = pd.qcut(smoothed, k, labels=False, duplicates="drop")
            if bins.nunique() < k:
                print(f"  K={k}: 分位点重复，实际只分出{bins.nunique()}类，跳过")
                continue
            tmp = df.copy()
            tmp["_bin"] = bins
            print(f"  K={k}:")
            p_value = check_daily_hazard_regression(tmp, "_bin")
            if p_value < 0.05:
                best_k_duration = k
        summary.append((col, "H_k(卡方显著)", best_k_duration))

    print("\n--- 汇总：4特征 × 2目标，交叉表（None=任何K都不满足门槛） ---")
    for col, target, best_k in summary:
        print(f"  {col:20s} 对应{target:20s}: 最大可行K = {best_k}")


def check_joint_sufficiency_on_clusters(df: pd.DataFrame, cluster_col: str = "cluster",
                                         title: str = "") -> None:
    """
    直接检验：用候选特征聚出来的K个簇，彼此之间的w*相关统计量（均值/标准差收益）
    和久期分布，是不是真的不一样——这是"联合充分性"要求的直接检验，不再借道
    ref_regime或任何旧分类标准。
    """
    print(f"\n[联合充分性检验] {title}")

    print("  各簇的逐日收益统计（对应w*_k该怎么标定）:")
    ret_stats = df.groupby(cluster_col)["log_return"].agg(["mean", "std", "count"])
    ret_stats["annualized_return"] = ret_stats["mean"] * 252
    ret_stats["kelly_f"] = ret_stats["mean"] / (ret_stats["std"] ** 2)
    print(ret_stats.round(5).to_string())

    from scipy.stats import kruskal
    groups = [g["log_return"].dropna().to_numpy() for _, g in df.groupby(cluster_col)]
    stat, p = kruskal(*groups)
    print(f"  Kruskal-Wallis（各簇逐日收益分布是否不同）: H={stat:.2f}, p={p:.4g}"
          f"{'  <-- 显著不同' if p < 0.01 else '  <-- 未见显著差异'}")

    check_pairwise_returns(df, cluster_col)

    seg_df = extract_segments_from_cluster_labels(df, cluster_col)
    print("\n  各簇的段落久期统计（对应H_k(r)/g_k该怎么标定）:")
    dur_stats = seg_df.groupby("cluster")["duration"].agg(["mean", "median", "std", "count"])
    print(dur_stats.round(1).to_string())

    dur_groups = [g["duration"].to_numpy() for _, g in seg_df.groupby("cluster") if len(g) >= 3]
    if len(dur_groups) >= 2:
        stat, p = kruskal(*dur_groups)
        print(f"  Kruskal-Wallis（各簇段落久期分布是否不同，段级别，样本量=段数）: H={stat:.2f}, p={p:.4g}"
              f"{'  <-- 显著不同' if p < 0.05 else '  <-- 未见显著差异，样本量也偏小，见下方段数'}")
    else:
        print("  段数太少，无法做久期的Kruskal-Wallis检验")

    check_daily_hazard_regression(df, cluster_col)


# ------------------------------- 验证4：标准化 + KMeans -------------------------------

# 经过单特征筛选（收益IC + 事件式久期IC，均为非循环检验）之后确定的特征组合：
# t_stat同期收益分组检验里唯一站得住的（最大K=3）；extremity在"距回撤天数"这个
# 干净的久期代理目标上稳健显著（两个阈值都显著，10%阈值IC=0.10）。ER、autocorr
# 虽然各自在某些检验里有信号，但都没有同时通过"对应目标的非循环检验"，暂不纳入
# 聚类特征——聚类只用真正筛选通过的特征，不是把测过的都堆进去。
DEFAULT_FEAT_COLS = ["feat_t_stat", "feat_extremity"]


def run_kmeans(df: pd.DataFrame, k: int = 3, feat_cols: list[str] | None = None) -> pd.DataFrame:
    """
    标准化后跑KMeans，返回带有效行（非NaN）+ cluster列的DataFrame，供后续
    验证3（联合充分性直接检验）复用同一份聚类结果，不重复聚类。
    """
    feat_cols = feat_cols or DEFAULT_FEAT_COLS
    valid = df[feat_cols].notna().all(axis=1)
    result = df.loc[valid].reset_index(drop=True).copy()
    X = result[feat_cols].to_numpy()
    X_std = StandardScaler().fit_transform(X)

    print("  轮廓系数扫描 (K=2..6):")
    best_k, best_score = None, -np.inf
    for kk in range(2, 7):
        km = KMeans(n_clusters=kk, n_init=10, random_state=0)
        labels = km.fit_predict(X_std)
        score = silhouette_score(X_std, labels)
        marker = ""
        if score > best_score:
            best_score, best_k = score, kk
            marker = "  <-- 当前最高"
        print(f"    K={kk}: silhouette={score:.4f}{marker}")
    print(f"  轮廓系数建议的K = {best_k}（K={k}是否仍合理，供参考，不强制切换）")

    km_final = KMeans(n_clusters=k, n_init=10, random_state=0)
    result["cluster"] = km_final.fit_predict(X_std)

    print(f"\n  K={k} 具体结果:")
    print("  各簇大小:", result["cluster"].value_counts().sort_index().to_dict())
    print("  各簇特征均值（标准化前，原始量纲）:")
    print(result.groupby("cluster")[feat_cols].mean().round(3).to_string())

    if "ref_regime" in result.columns:
        print("\n  与ref_regime的交叉表（仅供参考，不是聚类对错的判断标准——见1.3文档第4点）:")
        print(pd.crosstab(result["cluster"], result["ref_regime"]).to_string())

    return result


MIN_DWELL = 5  # 去抖动最短确认天数，见debounce_labels


def multi_feature_kmeans_diagnostic(df: pd.DataFrame, feat_cols: list[str] | None = None) -> None:
    """
    联合KMeans（K=3/4，原始 vs 去抖动平滑）。

    这一版默认用DEFAULT_FEAT_COLS=[t_stat, extremity]——是经过单特征筛选
    （事件式久期IC + 同期收益分组检验，两者都是非循环检验）之后剩下的、
    真正各自证明过有用的特征，替换掉早前那版直接堆四个特征、久期端测不出
    区分度的尝试（旧结论：四特征在四种配置下久期都不显著，记在1.3文档里，
    但那次的久期检验方法本身后来也被发现有问题，所以旧结论本身也不能
    全信——这次feat_cols和检验方法都换了，是一次新的、更干净的尝试）。
    """
    feat_cols = feat_cols or DEFAULT_FEAT_COLS
    print(f"\n聚类特征: {feat_cols}")
    for k in (3, 4):
        print(f"\n{'=' * 70}\nK={k}\n{'=' * 70}")
        print(f"\n[标准化后KMeans, K={k}]（K=3/4对比时轮廓系数扫描结果相同，只重复展示一次亦可）")
        clustered = run_kmeans(df, k=k, feat_cols=feat_cols)

        check_joint_sufficiency_on_clusters(
            clustered, cluster_col="cluster",
            title=f"K={k}, 原始逐日硬聚类标签（未平滑，预期久期无区分度）")

        clustered["cluster_smoothed"] = debounce_labels(clustered["cluster"], min_dwell=MIN_DWELL)
        check_joint_sufficiency_on_clusters(
            clustered, cluster_col="cluster_smoothed",
            title=f"K={k}, 去抖动平滑后（min_dwell={MIN_DWELL}天）")


def main():
    from ablation.common import load_data  # noqa: E402
    df = load_data().reset_index(drop=True)
    print(f"数据: {len(df)}行, {df['date'].min().date()} ~ {df['date'].max().date()}")

    df = compute_all_features(df)

    check_path_axis_redundancy(df)
    check_return_ic(df)

    # single_feature_diagnostic(df)  # 单特征筛选已跑过，结论用于确定下面的DEFAULT_FEAT_COLS
    multi_feature_kmeans_diagnostic(df)  # 用筛选后的[t_stat, extremity]重新聚类，替换旧的四特征版本

    out_path = REPO_ROOT / "outputs" / "feature_validation"
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path / "candidate_features.csv", index=False)
    print(f"\n特征数据已保存 -> {out_path / 'candidate_features.csv'}")


if __name__ == "__main__":
    main()
