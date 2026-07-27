"""
Prediction API

对外统一调用接口

提供:

1. 单树预测
2. 批量预测
3. I1目录预测
"""


from algorithms.prediction.predictor import (
    TreeInput,
    TreeRiskPredictor
)


from algorithms.prediction.i1_batch import (
    I1BatchPredictionService
)


from algorithms.prediction.real_i1_pipeline import (
    RealI1PredictionPipeline
)



class PredictionAPI:
    """
    预测系统统一接口
    """


    def __init__(self):

        self.predictor = TreeRiskPredictor()

        self.batch = I1BatchPredictionService()

        self.i1_pipeline = RealI1PredictionPipeline()



    def predict_tree(
        self,
        tree_input: TreeInput
    ):

        """
        单棵树预测
        """

        return self.predictor.predict(
            tree_input
        )



    def predict_batch(
        self,
        trees
    ):

        """
        多棵树预测
        """

        return self.batch.predict_batch(
            trees
        )



    def predict_i1(
        self,
        i1_path
    ):

        """
        真实I1预测
        """

        return self.i1_pipeline.run(
            i1_path
        )