"""
Prediction Visualization

将预测结果转换为展示格式
"""


class PredictionVisualizer:
    """
    风险结果可视化转换
    """


    def get_color(self, risk_level):

        colors = {

            "高风险": "red",

            "中风险": "orange",

            "低风险": "green"

        }


        return colors.get(
            risk_level,
            "gray"
        )



    def format_result(
        self,
        tree_id,
        result
    ):

        return {

            "tree_id": tree_id,

            "risk_level": result.risk_level,

            "risk_score": result.risk_score,

            "color": self.get_color(
                result.risk_level
            ),

            "suggestion": result.suggestion

        }



    def format_batch(
        self,
        results
    ):

        data = []


        for item in results:

            data.append(

                self.format_result(

                    item["tree_id"],

                    item["result"]

                )

            )


        return data