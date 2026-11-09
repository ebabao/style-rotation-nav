"""全天候策略模型 v1.0 工程包。

复用 all_weather_prototype.py 的既有数值内核（estimate_cov / blended_cov /
risk_contribution / solve_risk_budget / four_quadrant_budget / target_vol_leverage /
metrics），本包只做工程化扩展：真实数据清洗、因子层、自动学习、进化训练、
份额法执行、诊断报告、编排与递归测试。
"""
__version__ = "v1.0"
