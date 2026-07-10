# 全天候指数仓位择时框架 —— 工程原型（真实数据阶段）

基于《贝叶斯在线变点检测与久期感知状态机》方法论报告的工程实现。
当前阶段使用**真实中证800全收益指数数据**（`data/csi800_total_return.xlsx`，
2009-01-05 ~ 2026-07-09，日频收盘点位，对应文档§2"标的：中证800指数
（建议采用全收益口径）"，见 `ablation/01_data_loading.py`）。真实数据没有
"哪天是牛市/哪天变盘"的上帝视角标签，因此 `validation/` 与 `ablation/` 的
诊断/评估改用 `engine/regime_labeling.py` 产出的**自动标注参照标签**
（`ref_regime`/`ref_regime_age`，离线全样本HMM给出，不是真值，详见该模块
docstring）。此前基于合成数据的验证阶段已下线（`engine/synthetic_data.py`
及相关 `synthetic_*.csv` 已删除）。

## 六项工程改进（真实数据接入、跑通全流程之后完成）

逐行代码走查真实数据下的区制软分配/在线决策逻辑时发现六处值得改进的地方，
一次性做了如下改动：

1. **区制软分配距离度量可插拔**：`engine/regime.py` 的 `RegimeSoftAssigner`
   支持 `metric="mahalanobis"`（样本量加权的pooled协方差）和
   `metric="wasserstein"`（各原型自己的协方差，点-高斯W2距离闭式解）两种
   softmax软分配距离，`.assign_all_metrics()` 可并排对比。
2. **区制命名固定为 bull/sideways/bear**：K固定为3，不再用通用的
   `cluster_rank{i}` 命名。`engine/regime.py::name_clusters_by_return_rank`
   统一按收益排名映射，`engine/regime_labeling.py`（离线参照标签）与
   `engine/calibration.py`（因果walk-forward估参）共用同一份实现。
3. **季度walk-forward重估的7:3新旧平滑**：`engine/calibration.py::blend_assigners`
   与 `ablation/common.py::blend_prior` 对发射先验、区制原型（均值/协方差/
   目标暴露）、久期分布参数统一做"新7旧3"加权平滑，避免每季度重估造成
   仓位/区制判定的跳变。
4. **hazard/区制身份循环依赖的refine修复**：算hazard需要先知道区制身份，
   而区制身份的计算又需要先跑完用到这个hazard的那一步BOCPD递归——这是
   一个结构性的循环依赖。现在的处理方式：构造 `h_mix` 喂给 `bocpd.step()`
   本身仍只能用step前的区制概率（这一步避不开），但 `step()` 跑完后，
   用最新后验重新算一次更准的区制概率，专用于当天的仓位决策与
   `map_regime` 记录。
5. **仓位边界拆分为长仓/杠杆两种模式**：`engine/decision.py::calibrate_target_exposures`
   与 `ablation/s2~s4` 的 `generate_positions` 都支持可配置的
   `position_bounds=(lo, hi)`，默认 `(0,1)` 对应不允许做空/杠杆的长仓模式
   （数值上与早期硬编码`clip(0,1)`完全一致），另配 `(-1,2)` 做允许做空/
   杠杆的对比，两种模式都能跑，对比见 `ablation/leverage_contrast.py`。
6. **`engine/emission.py` 只留一元NIG**：删除了为未来可能的"联合建模
   [z_t,sigma_t]"（文档§2/§8提到的可选扩展）预留的通用D维NIW实现——全
   仓库只用到D=1，也没有近期做多维联合建模的计划，不为不存在的需求预留
   抽象。`NIWConjugateEmission`重命名为 `NIGConjugateEmission`，构造函数
   直接是 `(mu0, kappa0, alpha0, beta0, max_run_length=None)`。

**六项改进前后的实测对比**（真实中证800数据，`validation/regime_assignment_validation.py`
与 `ablation/s2~s4`）：区制识别与自动标注参照的一致率 50.2%→91.5%；S2夏普
0.04→0.24（换手率10.3%→9.7%）；S3夏普0.18→0.23（换手率16.5%→13.7%）；S4
夏普0.18→0.23（换手率10.7%→8.9%）；`leverage_contrast.py`：长仓(0,1)夏普
0.23/最大回撤-23.8%/换手率8.9%，杠杆(-1,2)夏普0.20/最大回撤-31.0%/换手率
14.1%。下方"真实数据实测结果"表格与"`validation/`组件验证的核心发现"表格
（现为`validation/`）成文时间早于这六项改进，数值是六项改进**之前**的
基线，仅作历史记录保留；六项改进后完整重跑 `run_ablation_summary.py` 得到
S0=0.304, S1=0.347, S2=0.236, S3=0.226, S4=0.234（夏普，与单独跑S2~S4时的
数字有细微差异，因为完整链路里前序阶段的随机性会传导），S1→S2前视偏差
缺口0.11（专门的`lookahead_contrast.py`给出更干净的缺口0.09），S5三个独立
时间窗口夏普均值0.03/标准差0.27，BH-FDR多重检验后0/3显著，Deflated Sharpe
Ratio=0.049——统计去伪结论方向与六项改进前一致，仍不乐观。

## 已知问题，暂不处理

`engine/calibration.py` 里为对齐文档§3.6字面定义引入的"影子BOCPD"（先跑
一遍历史获得真正的run-length后验加权发射描述子[μ̂t, σ̂t]再聚类）是每次
季度重估都从头跑一遍历史（O(T²)），单次运行耗时显著增加（真实数据17年
历史下单次完整S0~S5约15-20分钟）；K固定为3（未做数据驱动的自动选择）——
这两项连同其他尚未有明确文档依据的实现细节，会统一整理进一个单独的问题
文档。

## 目录结构

```
.
├── README.md
├── requirements.txt
├── engine/                       # 核心可复用模块，唯一的公共依赖来源
│   ├── features.py               # 特征计算：r_t, sigma_t, z_t             （文档§2）
│   ├── regime_labeling.py        # 离线全样本HMM给出的自动标注参照标签（非真值）（诊断/评估用）
│   ├── emission.py               # 一元NIG共轭发射模型 + Student-t预测分布 + 按矩标定先验（文档§3.3, §5.1）
│   ├── duration.py               # 久期分布/hazard/预期剩余久期 + 分段久期拟合（文档§3.4-3.5, §5.1）
│   ├── bocpd.py                  # 贝叶斯在线变点检测主递归引擎             （文档§3.2）
│   ├── regime.py                 # 区制原型 + 马氏/Wasserstein软分配 + 收益排名命名（文档§3.6）
│   ├── decision.py               # 目标暴露标定(可配置仓位边界)/久期折减/不确定性收缩/仓位映射/无交易带（文档§3.7, §5.3）
│   ├── calibration.py            # 因果walk-forward区制参数估计(含7:3新旧平滑)（不用ref_regime）（文档§5.1, §5.4）
│   ├── hmm_offline.py            # S1专用：离线平滑HMM（允许前视，仅作上限参照，多重启避免局部最优）（文档§4.2 S1）
│   ├── evaluation.py             # 统一回测评估口径 + 统计去伪工具          （文档§6）
│   └── plotting.py               # 绘图辅助（中文字体、区制背景着色）
├── ablation/                      # 真实数据处理 + 策略回测的**实际执行链路**
│   ├── 01_data_loading.py            # 加载真实中证800数据、标准化列名        （文档§2）
│   ├── 02_feature_engineering.py     # 特征工程 + 自动标注参照标签          （文档§2）
│   ├── common.py                     # 六级共用的路径/数据加载/walk-forward节奏常量
│   ├── s0_baseline.py                # S0：恒定满仓 / 均线择时（基准）
│   ├── s1_offline_hmm.py             # S1：离线HMM（前视上界参照，不可执行）
│   ├── s2_causal_bocpd_map.py        # S2：因果BOCPD + MAP硬切离散仓位（首个可上线版）
│   ├── s3_full_posterior_band.py     # S3：全后验连续仓位 + 无交易带（尚无HSMM）
│   ├── s4_hsmm_duration.py           # S4：S3 + HSMM年龄相依hazard（终版核心）
│   ├── s5_multi_seed_robustness.py   # S5：单一真实序列多时间窗稳健性 + 统计去伪（"多指数"部分待真实多资产数据）
│   ├── run_ablation_summary.py       # 汇总S0~S5，产出统一对比表（消融实验最终交付物）
│   ├── lookahead_contrast.py         # §6.4：同一HMM「平滑vs滤波」对照实验
│   ├── leverage_contrast.py          # 仓位边界对照实验：长仓clip(0,1) vs 允许做空/杠杆(-1,2)
│   └── backtest_report.py            # §6.1-6.4 汇总报告（调仓频次分布图等）
├── validation/                    # 组件级正确性/行为验证脚本，不是实际执行链路的一部分
│   ├── emission_validation.py            # 共轭发射模型正确性+行为验证          （文档§3.3）
│   ├── duration_hazard_validation.py     # 久期分布拟合+hazard函数验证         （文档§3.4-3.5）
│   ├── bocpd_validation.py               # BOCPD完整递归验证                   （文档§3.2, §5.2）
│   └── regime_assignment_validation.py   # 区制软分配+混合hazard验证           （文档§3.6）
├── data/                          # 真实中证800全收益数据（csi800_total_return.xlsx）+ ablation/01、02生成的标准化中间文件
└── outputs/
    ├── ablation/figures/, results/    # ablation/ 各阶段/各级别的诊断图、中间结果csv、汇总表
    └── validation/figures/, results/  # validation/ 各脚本的诊断图与验证结果
```

## `ablation/` 与 `validation/` 的分工

两者都依赖 `engine/`，但目的完全不同，不要混为一谈：

- **`validation/`（组件级正确性/行为验证）**：验证 `engine/` 里每个理论组件
  本身对不对——NIG递归数值对不对、hazard公式对不对、BOCPD归一化对不对……
  跟"选哪种策略配置"无关，不是实际执行链路的一部分。哪怕消融实验换了十种
  参数组合，这些数学正确性也不需要重新验证。
- **`ablation/`（真实数据处理 + 策略级消融实验的实际执行链路）**：从加载
  真实中证800数据开始（01/02），在锁死数据、评估口径、walk-forward节奏的
  前提下，对应文档 §4「模型构建与逐步优化路径」，从 S0 到 S5 逐级只改一个
  变量，对比策略配置的真实回测表现。

**目录曾经的组织方式（已废弃）**：01/02（真实数据加载、特征工程）与
03~06（组件正确性验证脚本）此前都放在同一个 `pipeline/` 目录下按数字编号
排列，但后续 S0~S5 实际运行都建立在01/02之上、却和01/02分处不同目录
（`ablation/`），而03~06只是验证脚本、根本不是实际运行链路的一部分——放
在同一个目录容易让人误以为03~06也是"下一步该跑什么"的流水线阶段。因此
把01/02并入 `ablation/`（它们是S0~S5这条实际执行链路的起点），03~06搬到
新的 `validation/` 目录并去掉数字前缀（改名为按内容命名，因为它们不是
顺序执行的流水线阶段，数字编号本身就是误导的来源）。

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
python ablation/01_data_loading.py          # 加载真实中证800全收益数据（data/csi800_total_return.xlsx）
python ablation/02_feature_engineering.py

python ablation/run_ablation_summary.py     # 依次跑S0~S5，打印+保存六级对比表
```

也可以单独跑某一级（比如只看S4）：`python ablation/s4_hsmm_duration.py`。

第六节的回测框架与统计检验（§6.1-6.4）：

```bash
python ablation/lookahead_contrast.py       # §6.4：同一HMM，平滑vs滤波对照
python ablation/backtest_report.py          # 汇总§6.1~6.4，含调仓频次分布图
```

### 真实数据实测结果（中证800全收益指数，2009-01-05~2026-07-09；六项工程改进之前的基线，仅作历史记录）

| 级别 | 年化收益 | 夏普 | 最大回撤 | 换手率 | 可执行 |
|---|---|---|---|---|---|
| S0-恒定满仓 | 6.9% | 0.30 | -48.5% | 0% | ✅ |
| S1-离线HMM(上限参照) | 4.1% | 0.35 | -37.5% | 2.6% | ❌不可执行 |
| S2-因果BOCPD+MAP硬切 | 0.4% | 0.04 | -27.6% | 10.3% | ✅ |
| S3-全后验+无交易带 | 1.7% | 0.18 | -24.5% | 16.5% | ✅ |
| S4-HSMM久期升级 | 1.7% | 0.18 | -23.8% | 10.7% | ✅ |
| S5-多时间窗(3次独立试验均值) | — | 0.04 | -14.8%(均值) | — | ✅ |

**S1→S2 前视偏差幅度 = 0.31个夏普点**：这是文档"主流离线HMM择时绩效部分
源于前视"这一核心主张在真实数据上的直接量化证据。

**诚实的发现（不是文档方法论的问题，是这次具体实现暴露出的问题）**：
- **S2→S3 换手率不降反升**（10.3%→16.5%），和文档"验证：夏普提升且换手
  显著下降"的预期方向相反，在合成数据阶段也观察到同样的方向。根源是每
  季度重新做KMeans聚类后，"哪个簇是牛市"可能因局部样本波动而重新洗牌
  （聚类标签漂移），全后验混合暴露比MAP硬切对这种边界抖动更敏感。值得
  后续专门处理（比如给新旧聚类中心做匹配对齐）。
- **S4相对S3的改善方向正确但幅度有限**（回撤-24.5%→-23.8%，夏普0.177→0.178，
  基本持平；换手率则从16.5%降到10.7%），说明HSMM久期升级本身在这份真实
  数据上主要体现在换手收敛而非收益/回撤的显著改善。
- **S5的统计去伪结果并不乐观**：3次独立时间窗试验里2/3夏普为正，但经
  BH-FDR多重检验校正后，**没有任何一次试验单独维持统计显著**；Deflated
  Sharpe Ratio仅0.05（越接近1才说明"不是撞大运"）。诚实的结论是：当前
  这套causal walk-forward配置的边际优势，在现有的窗口切分下，统计上还
  站不住脚——这与合成数据阶段的结论方向一致。

## 第六节「回测框架与统计检验」的落地情况

文档§6的四项要求（§6.5方法论流程验证已由validation/+ablation/整体覆盖，不
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

### §6.4 的两种对照实验，互补不重复

- **`run_ablation_summary.py` 里的 S1 vs S2**：模型类别（HMM→BOCPD）和因果性
  （平滑→滤波）**同时**变化，回答"换成本框架后总代价多大"——落差0.31个夏普点。
- **`lookahead_contrast.py`**：**同一个**HMM、同一组参数、同一套状态->暴露
  标定，只让"解码时看没看未来"这一个变量变化（Viterbi平滑 vs 前向算法滤波）
  ——落差只有0.09个夏普点。

两者一对比就能看出：S1→S2那0.31个点里，只有不到1/3（0.09点）真的是"平滑"
这个动作本身的贡献，剩下大部分来自"换了模型类别+区制识别方式也变了"——
这是把"前视偏差"这个笼统说法拆解到更精确程度的结果。

### 新发现：调仓频次分布暴露了S3引入的换手问题

`backtest_report.py`画出的调仓间隔直方图显示：S4终版模型的调仓间隔中位数
只有2天、均值8.5天，但最大间隔可达548天——这是一个双峰特征：整体换手率
只有10.7%（约89%的交易日不动，支持"多数交易日不动"），但一旦触发就容易
连续密集调仓（中位数间隔短，拉低了中位数），调仓在区制切换附近扎堆爆发，
和文档§5.3"多数交易日不动、调仓集中于区制切换附近"的设想方向一致，但
"扎堆爆发"这种连续密集调仓的模式比文档设想的更极端。换手率（S3 16.5%、
S4 10.7%）本身也仍偏高于文档"验证：夏普提升且换手显著下降"的预期，根源
和上面"S2→S3换手率不降反升"是同一个——区制聚类标签季度重估间漂移。这是
本次实现暴露出的问题，需要后续解决（比如给新旧聚类中心做匹配对齐），
不是文档方法论本身的缺陷。


## `validation/` 组件验证的核心发现（真实数据：中证800全收益指数；六项工程改进之前的基线，仅作历史记录）

下表涉及的"区制转移点""区制标签"均指 `ref_regime`/`ref_regime_age`
（`engine/regime_labeling.py` 给出的自动标注参照，不是真值）。

| 脚本（曾用编号） | 验证内容 | 结论 |
|---|---|---|
| `ablation/01_data_loading.py`（曾用01） | 真实数据加载 | 4253个交易日（2009-01-05~2026-07-09），无缺失/重复，标准化为[date, price] |
| `ablation/02_feature_engineering.py`（曾用02） | 特征工程 + 自动标注参照标签 | z_t归一化确实压抑了跨区制波动差异（变异系数0.478→0.018） |
| `validation/emission_validation.py`（曾用03） | 共轭发射模型 | 递归实现数值正确（误差<1e-13）；约51%自动标注参照变点表现出似然骤降 |
| `validation/duration_hazard_validation.py`（曾用04） | 久期/hazard | 负二项hazard随年龄变化、几何hazard恒为常数，理论对比清晰坐实 |
| `validation/bocpd_validation.py`（曾用05） | BOCPD引擎 | 数值正确；常数hazard下P(r_t=0)恒等于hazard本身（数学性质非bug）；检测率19.1% |
| `validation/regime_assignment_validation.py`（曾用06） | 区制软分配 | 与自动标注参照的整体一致率50.2%（六项改进后91.5%，见上方"六项工程改进"章节）；MAP段龄跟踪(<=5天占比)11.6% |

## 依赖

见 `requirements.txt`（新增 `hmmlearn`，供 `ablation/s1_offline_hmm.py` 使用）。
中文绘图需要系统安装 Noto Sans CJK 字体（`engine/plotting.py` 中硬编码了
字体路径，不同环境可能需要调整）。
