"""
Prediction model configuration.

所有风险参数集中管理。
以后接入机器学习模型时，可以替换这里。
"""


class PredictionConfig:

    # 风险最高分
    MAX_SCORE = 100


    # 各因素权重
    # 总和约等于1

    HEIGHT_WEIGHT = 0.10       # 树高
    AGE_WEIGHT = 0.15          # 年龄
    DIAMETER_WEIGHT = 0.10     # 胸径
    WIND_WEIGHT = 0.25         # 风速
    RAIN_WEIGHT = 0.10         # 降雨
    HUMIDITY_WEIGHT = 0.05     # 湿度
    SLOPE_WEIGHT = 0.10        # 坡度
    DISTANCE_WEIGHT = 0.15     # 距离电线


    # 风险等级阈值

    LOW_RISK_LIMIT = 30

    MEDIUM_RISK_LIMIT = 60

    HIGH_RISK_LIMIT = 80