"""
ablation
========
对应方法论文档 §4「模型构建与逐步优化路径（Baseline → Final）」的五级消融
路径 S0~S5（S0作为额外的第0级基准，共六级）。

与 pipeline/ 的关系：
  pipeline/  —— 验证 engine/ 里每个理论组件本身对不对（数值正确性、公式对不
                 对），是"组件级单元测试"，跟采用哪种策略配置无关。
  ablation/  —— 在锁死数据、评估口径、walk-forward节奏的前提下，逐级对比不同
                 策略配置的真实回测表现，是"策略级消融实验"，依赖 engine/ 但
                 不依赖 pipeline/ 的任何脚本。

每一级的模块只暴露一个入口函数 generate_positions(df, ...) -> pd.DataFrame，
统一产出至少包含 ['date','ref_regime','log_return','w_held'] 的表（ref_regime
是 engine.regime_labeling 给出的自动标注参照标签，不是真值），供
run_ablation_summary.py 统一调用 engine.evaluation.compute_backtest_metrics
评估、汇总成六级对比表。

六级只允许 S1 使用整段历史（含"未来"）一次性拟合，S0、S2~S5 全部必须走
严格因果的 walk-forward（估参窗口只使用当前时点之前的数据），这是消融实验
能否成立的前提，详见各脚本内的文档字符串。
"""
