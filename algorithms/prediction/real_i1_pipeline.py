"""
Real I1 Prediction Pipeline

真实I1数据完整预测流程

I1 Bundle
    |
    ↓
I1 Loader
    |
    ↓
Tree Adapter
    |
    ↓
Prediction Pipeline
"""


from pathlib import Path


from algorithms.integration.i1 import (
    load_i1_bundle
)


from algorithms.prediction.i1_real_adapter import (
    RealI1TreeAdapter
)


from algorithms.prediction.pipeline import (
    PredictionPipeline
)



class RealI1PredictionPipeline:
    """
    真实I1预测流水线
    """


    def __init__(self):

        self.adapter = RealI1TreeAdapter()

        self.pipeline = PredictionPipeline()



    def run(self, i1_path):

        root = Path(i1_path)


        # 1. 加载真实I1 bundle

        bundle = load_i1_bundle(
            root
        )


        # 2. 转换成预测输入

        trees = self.adapter.convert(
            bundle
        )


        # 3. 执行预测

        if not trees:

            return {
                "results": [],
                "report": {
                    "total_tree": 0
                }
            }


        return self.pipeline.run(
            trees
        )