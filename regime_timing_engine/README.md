# 全天候指数仓位择时框架 —— 工程原型（真实数据阶段）

基于《贝叶斯在线变点检测与久期感知状态机》方法论报告的工程实现。
当前阶段使用**真实中证800数据**（`data/csi800.csv`，占位格式，见
`pipeline/01_data_loading.py`）。真实数据没有"哪天是牛市/哪天变盘"的
上帝视角标签，因此 pipeline 03-06 与 ablation 的诊断/评估改用
`engine/regime_labeling.py` 产出的**自动标注参照标签**（`ref_regime`/
`ref_regime_age`，离线全样本HMM给出，不是真值，详见该模块docstring）。
此前基于合成数据的验证阶段已下线（`engine/synthetic_data.py` 及相关
`synthetic_*.csv` 已删除）。

## 近期变更（严格对齐文档字面定义）

1. **`engine/emission.py`：一元NIG → 通用D维NIW共轭发射模型**（对应文档
   §3.3"多变量情形取NIW共轭"）。`NIWConjugateEmission`，D=1时数值上与原
   实现完全等价（已做回归测试），`from_nig(mu0,kappa0,alpha0,beta0)`是
   D=1的便捷构造入口，全仓库统一走这个入口。这也为以后如果要试文档§2/§8
   提到的"联合建模[z_t,sigma_t]"这个可选扩展打好了基础（把维度参数从1
   改成2即可，不需要另起一个类）。
2. **区制软分配的特征描述子改回文档字面定义**：§3.6写的是"run-length后验
   加权的发射描述子[μ̂t, σ̂t]"，两个维度都应来自BOCPD自己的后验。之前的
   实现第二维用的是外部的"原始已实现波动率"（判别力更强但不是文档字面
   定义），现在`engine/calibration.py`会先跑一遍"影子BOCPD"过历史数据、
   取得真正的[μ̂t, σ̂t]轨迹再聚类，`ablation/s2~s4`的在线循环也同步改用
   `bocpd.emission.posterior_weighted_mean_scale(...)`。
   **代价**：z_t经过波动率归一化后自身判别力较弱（`pipeline/03`已验证），
   S2~S4的绩效数字因此普遍下降，这是"字面对齐文档"的预期结果，不是bug。
3. **`pipeline/03/05/06`的NIG先验初始值`kappa0`统一为1.0**（原来0.5/1.0
   混用）；`pipeline/05`里独立实现的"检测滞后"逻辑（用`low_r_threshold=5`）
   已删除，改为直接复用`engine/bocpd.py`内置的`prob_recent_reset`(k=3)。
4. **已知问题，暂不处理**：`ablation/run_ablation_summary.py`因为改动2
   引入的"影子BOCPD"是每次重估都从头跑一遍历史（O(T²)），单次运行耗时
   显著增加（约10分钟+）；K固定为3（未做数据驱动的自动选择）——这两项
   连同其他尚未有明确文档依据的实现细节，会统一整理进一个单独的问题文档。

## 目录结构

```
.
├── README.md
├── requirements.txt
├── engine/                       # 核心可复用模块，唯一的公共依赖来源
│   ├── features.py               # 特征计算：r_t, sigma_t, z_t             （文档§2）
│   ├── regime_labeling.py        # 离线全样本HMM给出的自动标注参照标签（非真值）（诊断/评估用）
│   ├── emission.py               # NIG共轭发射模型 + Student-t预测分布 + 按矩标定先验（文档§3.3, §5.1）
│   ├── duration.py               # 久期分布/hazard/预期剩余久期 + 分段久期拟合（文档§3.4-3.5, §5.1）
│   ├── bocpd.py                  # 贝叶斯在线变点检测主递归引擎             （文档§3.2）
│   ├── regime.py                 # 区制原型与softmax软分配                 （文档§3.6）
│   ├── decision.py               # 目标暴露标定/久期折减/不确定性收缩/仓位映射/无交易带（文档§3.7, §5.3）
│   ├── calibration.py            # 因果walk-forward区制参数估计（不用ref_regime）（文档§5.1, §5.4）
│   ├── hmm_offline.py            # S1专用：离线平滑HMM（允许前视，仅作上限参照）（文档§4.2 S1）
│   ├── evaluation.py             # 统一回测评估口径 + 统计去伪工具          （文档§6）
│   └── plotting.py               # 绘图辅助（中文字体、区制背景着色）
├── pipeline/                     # 组件级验证脚本：只验证engine/里的数学/公式对不对
│   ├── 01_data_loading.py            # 加载真实中证800数据、标准化列名        （文档§2）
│   ├── 02_feature_engineering.py     # 特征工程 + 自动标注参照标签          （文档§2）
│   ├── 03_emission_validation.py     # 共轭发射模型正确性+行为验证          （文档§3.3）
│   ├── 04_duration_hazard_validation.py  # 久期分布拟合+hazard函数验证      （文档§3.4-3.5）
│   ├── 05_bocpd_validation.py        # BOCPD完整递归验证                   （文档§3.2, §5.2）
│   └── 06_regime_assignment.py       # 区制软分配+混合hazard验证           （文档§3.6）
├── ablation/                      # 策略级消融实验：S0~S5六级对比 + 第六节回测框架
│   ├── common.py                     # 六级共用的路径/数据加载/walk-forward节奏常量
│   ├── s0_baseline.py                # S0：恒定满仓 / 均线择时（基准）
│   ├── s1_offline_hmm.py             # S1：离线HMM（前视上界参照，不可执行）
│   ├── s2_causal_bocpd_map.py        # S2：因果BOCPD + MAP硬切离散仓位（首个可上线版）
│   ├── s3_full_posterior_band.py     # S3：全后验连续仓位 + 无交易带（尚无HSMM）
│   ├── s4_hsmm_duration.py           # S4：S3 + HSMM年龄相依hazard（终版核心）
│   ├── s5_multi_seed_robustness.py   # S5：单一真实序列多时间窗稳健性 + 统计去伪（"多指数"部分待真实多资产数据）
│   ├── run_ablation_summary.py       # 汇总S0~S5，产出统一对比表（消融实验最终交付物）
│   ├── lookahead_contrast.py         # §6.4：同一HMM「平滑vs滤波」对照实验
│   └── backtest_report.py            # §6.1-6.4 汇总报告（调仓频次分布图等）
├── data/                          # 真实中证800数据（csi800.csv，用户提供）+ pipeline/01、02生成的标准化中间文件
└── outputs/
    ├── figures/, results/             # pipeline/ 各阶段的诊断图与中间结果
    └── ablation/figures/, results/    # ablation/ 各级别的诊断图、结果csv、汇总表
```

## `pipeline/` 与 `ablation/` 的分工

两者都依赖 `engine/`，但目的完全不同，不要混为一谈：

- **`pipeline/`（组件级验证）**：验证 `engine/` 里每个理论组件本身对不对——
  NIG递归数值对不对、hazard公式对不对、BOCPD归一化对不对……跟"选哪种策略
  配置"无关。哪怕消融实验换了十种参数组合，这些数学正确性也不需要重新验证。
- **`ablation/`（策略级消融）**：在锁死数据、评估口径、walk-forward节奏的
  前提下，对应文档 §4「模型构建与逐步优化路径」，从 S0 到 S5 逐级只改一个
  变量，对比策略配置的真实回测表现。

**`pipeline/07_decision_backtest.py` 已退休**：它当初做的事情（因果BOCPD+
区制混合hazard+全后验仓位映射+walk-forward+回测）内容上已经和消融阶梯里的
"S4"完全重合，两份代码长得一样、逻辑重复。现在这部分逻辑已经在
`ablation/s4_hsmm_duration.py` 里以更严谨的、和其它五级严格对齐的形式重新
实现，07的历史版本不再保留。

## 消融实验：S0 ~ S5（对应文档 §4）

六级只允许 **S1** 使用整段历史（含"未来"）一次性拟合；S0、S2~S5 全部走
严格因果的 walk-forward（估参窗口只用当前时点之前的数据，每季度重估一次，
常数定义在 `ablation/common.py`，六级共用同一套节奏，不允许各定义各的）。
每一级只改一个变量：

| 级别 | 相对上一级改了什么 | 隔离的增益 |
|---|---|---|
| S0 | 恒定满仓 / 均线择时（零参数基准） | — |
| S1 | 整段历史一次性拟合HMM，平滑状态驱动仓位（**唯一允许前视**） | 信息增益上限（不可上线） |
| S2 | 换成因果BOCPD + 无监督聚类定区制，MAP硬切离散仓位；hazard用geometric（常数） | 因果去前视 |
| S3 | MAP硬切 → 全后验混合暴露 + 不确定性收缩ψ；加无交易带；hazard仍是geometric | 概率化 + 平滑 |
| S4 | hazard从geometric换成negbinom（HSMM，年龄相依）；加久期折减φ | 久期自适应 |
| S5 | S4不变，改为单一真实序列多时间窗(Purged K-Fold+Embargo)并行复算 + 统计去伪(BH-FDR, Deflated Sharpe) | 稳健性/泛化 |

**关于S1的HMM**：用 `hmmlearn.GaussianHMM` 在 `[z, log(realized_vol)]` 两维
特征上做EM估参+Viterbi解码，整段历史一次性喂入（刻意允许前视，这正是它
作为"理论上限参照"的意义）。这套机制同时被 `engine/regime_labeling.py`
复用来给真实数据产出 `ref_regime`/`ref_regime_age` 自动标注参照标签。

**关于S5的"多指数"**：本仓库目前只有一份真实中证800数据，没有真实多宽基/
行业数据可用；此前用"多个独立随机种子生成的合成指数路径"充当代理的做法
已随合成数据一起下线。`ablation/s5_multi_seed_robustness.py` 现在只覆盖
"多时间窗稳健性 + 统计去伪"这一半，"多指数并行"部分留待接入真实多资产
数据后再补上，该脚本 docstring 里对这个现状有明确说明。

### 运行消融实验

```bash
python pipeline/01_data_loading.py          # 先加载真实中证800数据（占位路径 data/csi800.csv）
python pipeline/02_feature_engineering.py

python ablation/run_ablation_summary.py     # 依次跑S0~S5，打印+保存六级对比表
```

也可以单独跑某一级（比如只看S4）：`python ablation/s4_hsmm_duration.py`。

第六节的回测框架与统计检验（§6.1-6.4）：

```bash
python ablation/lookahead_contrast.py       # §6.4：同一HMM，平滑vs滤波对照
python ablation/backtest_report.py          # 汇总§6.1~6.4，含调仓频次分布图
```

### 历史记录：合成数据阶段的一次实测结果（已过期，待真实数据重跑）

以下数据来自本仓库切换到真实中证800数据**之前**、基于合成三区制数据的一次
实测，仅作历史参考，不代表真实数据上的表现——真实数据跑通后应删除或替换
这张表。

| 级别 | 夏普 | 最大回撤 | 换手率 | 可执行 |
|---|---|---|---|---|
| S0-恒定满仓 | 0.28 | -36.7% | 0% | ✅ |
| S1-离线HMM(上限参照) | 0.60 | -16.6% | 56.1% | ❌不可执行 |
| S2-因果BOCPD+MAP硬切 | 0.34 | -35.6% | 21.0% | ✅ |
| S3-全后验+无交易带 | 0.28 | -18.5% | 56.6% | ✅ |
| S4-HSMM久期升级 | 0.29 | -17.7% | 51.6% | ✅ |
| S5-多指数×多窗口(15次试验均值，合成数据阶段的多种子机制，现已下线) | 0.22 | -14.5%(均值) | — | ✅ |

**S1→S2 前视偏差幅度 = 0.26个夏普点**（合成数据阶段的观测）：这是文档
"主流离线HMM择时绩效部分源于前视"这一核心主张的直接量化证据，方向性结论
预期在真实数据上依然成立，具体数值待重跑确认。

**诚实的意外发现（不是文档方法论的问题，是这次具体实现暴露出的问题，
是否在真实数据上依然存在待验证）**：
- **S2→S3 换手率不降反升**（21.0%→56.6%），和文档"验证：夏普提升且换手
  显著下降"的预期方向相反。根源是每季度重新做KMeans聚类后，"哪个簇是牛市"
  可能因局部样本波动而重新洗牌（聚类标签漂移），全后验混合暴露比MAP硬切
  对这种边界抖动更敏感。值得后续专门处理（比如给新旧聚类中心做匹配对齐）。
- **S4相对S3的改善方向正确但幅度有限**（回撤-18.5%→-17.7%，夏普0.28→0.29），
  说明HSMM久期升级本身的边际贡献在这份合成数据上并不大。
- **S5的统计去伪结果并不乐观**：15次独立试验里80%夏普为正，但经BH-FDR
  多重检验校正后，**没有任何一次试验单独维持统计显著**；Deflated Sharpe
  Ratio仅0.38（越接近1才说明"不是撞大运"）。诚实的结论是：当前这套causal
  walk-forward配置的边际优势，在现有的合成数据规模和窗口切分下，统计上
  还站不住脚。

## 第六节「回测框架与统计检验」的落地情况

文档§6的四项要求（§6.5方法论流程验证已由pipeline/+ablation/整体覆盖，不
单列）全部落地为可运行代码，汇总入口是 `ablation/backtest_report.py`：

```bash
python ablation/backtest_report.py
```

| 文档要求 | 落地位置 | 说明 |
|---|---|---|
| §6.1 防泄漏（walk-forward; Purged K-Fold + Embargo） | `engine/calibration.py`（walk-forward）+ `engine/evaluation.purged_embargo_windows`（Purged K-Fold+Embargo） | S2~S5全部走causal walk-forward；S5额外用Purged K-Fold+Embargo切多窗口 |
| §6.2 评估口径（年化收益/夏普/Calmar/最大回撤/换手率/调仓频次分布/分区制绩效/检测滞后） | `engine/evaluation.compute_backtest_metrics` + `detection_lag_stats` | 六级统一调用，`ablation/backtest_report.py`额外画出"调仓间隔天数分布"直方图 |
| §6.3 统计去伪（Deflated Sharpe Ratio; BH-FDR） | `engine/evaluation.deflated_sharpe_ratio` / `benjamini_hochberg` | 用于S5的多时间窗独立试验 |
| §6.4 前视偏差对照实验（同一HMM平滑vs滤波） | `engine/hmm_offline.py`（`decode_smoothed`/`decode_filtered`） + `ablation/lookahead_contrast.py` | 见下方专门说明 |

> 以下§6.4对照与"新发现"两小节的具体数值同样来自合成数据阶段的一次实测，
> 方向性结论预期在真实数据上依然成立，具体数值待重跑确认。

### §6.4 的两种对照实验，互补不重复

- **`run_ablation_summary.py` 里的 S1 vs S2**：模型类别（HMM→BOCPD）和因果性
  （平滑→滤波）**同时**变化，回答"换成本框架后总代价多大"——落差0.26个夏普点。
- **`lookahead_contrast.py`**：**同一个**HMM、同一组参数、同一套状态->暴露
  标定，只让"解码时看没看未来"这一个变量变化（Viterbi平滑 vs 前向算法滤波）
  ——落差只有0.09个夏普点。

两者一对比就能看出：S1→S2那0.26个点里，只有约1/3（0.09点）真的是"平滑"
这个动作本身的贡献，剩下2/3来自"换了模型类别+区制识别方式也变了"——
这是把"前视偏差"这个笼统说法拆解到更精确程度的结果。

### 新发现：调仓频次分布暴露了S3引入的换手问题

`backtest_report.py`画出的调仓间隔直方图显示：S4终版模型的调仓间隔**中位数
只有1天**，也就是说超过一半的交易日都在调仓——这和文档§5.3设想的"多数
交易日不动、调仓集中于区制切换附近"正好相反，和此前发现的"区制聚类标签
季度重估间漂移"是同一个根源。这是本次实现暴露出的问题，需要后续解决
（比如给新旧聚类中心做匹配对齐），不是文档方法论本身的缺陷。


## `pipeline/` 组件验证的核心发现（历史记录：合成数据阶段，待真实数据重跑）

01 现已改为真实数据加载脚本（`pipeline/01_data_loading.py`），不再生成合成
数据；下表 01 那一行是合成数据阶段的历史记录。03/04/05/06 涉及的"区制转移
点""区制标签"在真实数据上均指 `ref_regime`/`ref_regime_age`（自动标注参照，
非真值），具体数值待真实数据重跑确认。

| 脚本 | 验证内容 | 结论（合成数据阶段） |
|---|---|---|
| 01 | （历史）合成数据生成，现已下线 | 三区制、负二项久期（非几何），久期/波动特征符合设计 |
| 02 | 特征工程 + 自动标注参照标签 | z_t归一化确实压抑了跨区制波动差异（变异系数0.609→0.093） |
| 03 | 共轭发射模型 | 递归实现数值正确（误差<1e-13）；但z_t对多数区制切换的信号偏弱，只有约41%参照变点表现出似然骤降 |
| 04 | 久期/hazard | 负二项hazard随年龄变化、几何hazard恒为常数，理论对比清晰坐实 |
| 05 | BOCPD引擎 | 数值正确；常数hazard下P(r_t=0)恒等于hazard本身（数学性质非bug）；泛化hazard下检测率仅17.2% |
| 06 | 区制软分配 | bull/sideways混淆明显；MAP段龄跟踪从0.6%改善到7.2% |

## 依赖

见 `requirements.txt`（新增 `hmmlearn`，供 `ablation/s1_offline_hmm.py` 使用）。
中文绘图需要系统安装 Noto Sans CJK 字体（`engine/plotting.py` 中硬编码了
字体路径，不同环境可能需要调整）。
