"""
I1 Prediction Pipeline

功能:

I1目录
 ↓
读取树数据
 ↓
风险预测
 ↓
生成报告
 ↓
保存结果
"""


from algorithms.prediction.i1_loader import (
    I1Loader
)

from algorithms.prediction.pipeline import (
    PredictionPipeline
)



class I1PredictionPipeline:
    """
    I1完整预测流程
    """


    def __init__(self):

        self.predict_pipeline = PredictionPipeline()



    def run(self, i1_path):

        # 1. 读取I1数据

        trees = I1Loader(
            i1_path
        ).load_trees()


        if not trees:

            return {

                "results": [],

                "report": {

                    "total_tree": 0

                }

            }


        # 2. 执行预测

        result = self.predict_pipeline.run(
            trees
        )


        return result