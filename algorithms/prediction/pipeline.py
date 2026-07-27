"""
Prediction Pipeline

系统统一预测入口

流程:

I1数据
 ↓
批量预测
 ↓
风险报告
 ↓
保存结果
"""


from algorithms.prediction.i1_batch import (
    I1BatchPredictionService
)

from algorithms.prediction.risk_report import (
    RiskReport
)

from algorithms.prediction.storage import (
    PredictionStorage
)



class PredictionPipeline:
    """
    总预测流程
    """


    def __init__(self):

        self.batch_service = I1BatchPredictionService()

        self.report = RiskReport()

        self.storage = PredictionStorage()



    def run(self, trees):

        # 1. 批量预测

        results = self.batch_service.predict_batch(
            trees
        )


        # 2. 生成报告

        report = self.report.generate(
            results
        )


        # 3. 保存

        self.storage.save(
            {
                "results": results,
                "report": report
            }
        )


        return {

            "results": results,

            "report": report

        }