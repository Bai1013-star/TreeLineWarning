"""
I1 Batch Prediction

作用:
批量预测多棵树风险
"""


from algorithms.prediction.i1_prediction import (
    I1PredictionService,
    I1TreeData
)



class I1BatchPredictionService:
    """
    I1批量预测服务
    """


    def __init__(self):

        self.service = I1PredictionService()



    def predict_batch(self, trees):

        results = []


        for tree in trees:

            result = self.service.predict_tree(tree)

            results.append(result)


        return results