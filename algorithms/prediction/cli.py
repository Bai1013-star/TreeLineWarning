"""
TreeLineWarning CLI

命令行预测入口

用户输入:
    树木信息
    天气信息
    环境信息

输出:
    风险预测结果
"""


from algorithms.prediction.api import (
    PredictionAPI
)

from algorithms.prediction.predictor import (
    TreeInput
)



def run_cli():

    print("=" * 40)

    print("TreeLineWarning 树木风险预测系统")

    print("=" * 40)



    species = input(
        "请输入树种:"
    )


    height = float(
        input("请输入树高(m):")
    )


    age = int(
        input("请输入树龄:")
    )


    diameter = float(
        input("请输入胸径(cm):")
    )


    wind_speed = float(
        input("请输入风速(m/s):")
    )


    rainfall = float(
        input("请输入降雨(mm):")
    )


    humidity = float(
        input("请输入湿度(%):")
    )


    slope = float(
        input("请输入坡度(°):")
    )


    distance = float(
        input("请输入距离电线(m):")
    )



    tree = TreeInput(

        species,

        height,

        age,

        diameter,

        wind_speed,

        rainfall,

        humidity,

        slope,

        distance
    )



    result = PredictionAPI().predict_tree(
        tree
    )



    print("\n预测结果")
    print("-" * 40)

    print(
        "风险等级:",
        result.risk_level
    )

    print(
        "风险分数:",
        result.risk_score
    )

    print(
        "建议:",
        result.suggestion
    )



if __name__ == "__main__":

    run_cli()