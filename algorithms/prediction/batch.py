"""
Batch prediction

批量树木风险预测
"""

from .service import PredictionService


class BatchPredictionService:
    """
    多树预测服务
    """

    def __init__(self):
        self.service = PredictionService()


    def predict_batch(self, trees: list[dict]):
        """
        输入多棵树数据

        trees:
        [
            {
                "tree_id":1,
                "species":"松树",
                ...
            }
        ]
        """

        results = []

        for tree in trees:

            result = self.service.predict_tree(
                tree_id=tree["tree_id"],
                species=tree["species"],
                height=tree["height"],
                age=tree["age"],
                diameter=tree["diameter"],
                wind_speed=tree["wind_speed"],
                rainfall=tree["rainfall"],
                humidity=tree["humidity"],
                slope=tree["slope"],
            )

            results.append(result)

        return results