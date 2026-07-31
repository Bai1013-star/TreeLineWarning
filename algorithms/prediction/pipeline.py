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


import logging

from algorithms.prediction.i1_batch import I1BatchPredictionService
from algorithms.prediction.risk_report import RiskReport
from algorithms.prediction.storage import PredictionStorage

logger = logging.getLogger(__name__)



class PredictionPipeline:
    """
    总预测流程
    """


    def __init__(
        self,
        *,
        storage: PredictionStorage | None = None,
        persist: bool = False,
    ):
        self.batch_service = I1BatchPredictionService()
        self.report = RiskReport()
        if persist and storage is None:
            raise ValueError("persist=True requires an explicit PredictionStorage")
        self.storage = storage
        self.persist = persist



    def run(self, trees):


        # =========================
        # 1. 批量预测
        # =========================

        results = self.batch_service.predict_batch(
            trees
        )


        # =========================
        # 2. 生成风险报告
        # =========================

        report = self.report.generate(
            results
        )



        logger.info(
            "Prediction batch completed: trees=%d high=%d medium=%d low=%d",
            len(results),
            report.get("high_risk_count", 0),
            report.get("medium_risk_count", 0),
            report.get("low_risk_count", 0),
        )

        if self.persist:
            assert self.storage is not None
            self.storage.save({"results": results, "report": report})



        # =========================
        # 4. 返回
        # =========================

        return {


            "results": results,


            "report": report


        }
