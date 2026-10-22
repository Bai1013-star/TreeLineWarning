"""
Prediction report

生成风险统计报告
"""


class PredictionReport:


    def generate(self, results: list[dict]):
        """
        根据批量预测结果生成报告
        """

        total = len(results)

        high = []
        medium = []
        low = []


        for item in results:

            result = item["result"]

            if result.risk_level == "高风险":
                high.append(item["tree_id"])

            elif result.risk_level == "中风险":
                medium.append(item["tree_id"])

            else:
                low.append(item["tree_id"])


        return {

            "total_tree": total,

            "high_risk_count": len(high),

            "medium_risk_count": len(medium),

            "low_risk_count": len(low),

            "high_risk_tree_ids": high,

            "medium_risk_tree_ids": medium,

            "low_risk_tree_ids": low,

            "suggestion":
                "建议立即巡检"
                if high
                else "正常维护"

        }