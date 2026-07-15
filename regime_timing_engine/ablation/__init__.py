"""
ablation
========
真实数据处理 + 策略回测的**实际执行链路**，从加载中证800数据开始，一路到
方法论文档 §4「模型构建与逐步优化路径（Baseline → Final）」的五级消融
路径 S0~S5（S0作为额外的第0级基准，共六级），再到 §6 回测框架的汇总报告。

目录内脚本按执行顺序：
  01_data_loading.py         —— 加载真实中证800全收益数据，标准化成[date, price]
  02_feature_engineering.py  —— 特征工程 + 自动标注参照标签(ref_regime)
  common.py                  —— 六级共用的路径/walk-forward节奏常量
  s0_baseline.py ~ s5_multi_seed_robustness.py —— S0~S5 六级消融
  run_ablation_summary.py    —— 汇总六级对比表（消融实验最终交付物）
  lookahead_contrast.py / leverage_contrast.py —— §6.4/仓位边界对照实验
  backtest_report.py         —— §6.1-6.4 汇总报告

与 validation/ 的关系：
  validation/ —— 验证 engine/ 里每个理论组件本身对不对（数值正确性、公式对
                 不对），是"组件级正确性/行为验证"，不是实际执行链路的一
                 部分，跟采用哪种策略配置无关。哪怕消融实验换了十种参数
                 组合，这些数学正确性也不需要重新验证。
  ablation/   —— 本目录，在锁死数据、评估口径、walk-forward节奏的前提下，
                 从真实数据出发、逐级对比不同策略配置的真实回测表现，是
                 "策略级消融实验"，依赖 engine/ 但不依赖 validation/ 的
                 任何脚本。

S0~S5 每一级的模块只暴露一个入口函数 generate_positions(df, ...) ->
pd.DataFrame，统一产出至少包含 ['date','ref_regime','log_return','w_held']
的表（ref_regime 是 engine.regime_labeling 给出的自动标注参照标签，不是
真值），供 run_ablation_summary.py 统一调用
engine.evaluation.compute_backtest_metrics 评估、汇总成六级对比表。

六级只允许 S1 使用整段历史（含"未来"）一次性拟合，S0、S2~S5 全部必须走
严格因果的 walk-forward（估参窗口只使用当前时点之前的数据），这是消融实验
能否成立的前提，详见各脚本内的文档字符串。
"""
