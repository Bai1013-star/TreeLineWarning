from algorithms.prediction.i1_adapter import convert_i1_tree
from algorithms.prediction.service import PredictionService


class I1PredictionService:
    """
    I1数据预测服务

    功能：
    I1树木记录
        ↓
    转换预测输入
        ↓
    调用预测模型
        ↓
    返回风险结果
    """

    def __init__(self):
        self.service = PredictionService()


    def predict(self, i1_tree):
        """
        输入:
            I1TreeRecord

        输出:
            风险预测结果
        """

        tree = convert_i1_tree(i1_tree)

        result = self.service.predict_tree(
            tree["tree_id"],
            tree["species"],
            tree["height"],
            tree["age"],
            tree["diameter"],
            tree["wind_speed"],
            tree["rainfall"],
            tree["humidity"],
            tree["slope"],
        )

        return result