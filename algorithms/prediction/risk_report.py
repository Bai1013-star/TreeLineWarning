"""
Risk Report

作用:
整理批量预测结果
生成巡检报告
"""


class RiskReport:


    def generate(self, results):

        total = len(results)

        high = []
        medium = []
        low = []


        for item in results:

            tree_id = item["tree_id"]

            result = item["result"]


            if result.risk_level == "高风险":

                high.append(
                    tree_id
                )


            elif result.risk_level == "中风险":

                medium.append(
                    tree_id
                )


            else:

                low.append(
                    tree_id
                )



        return {

            "total_tree": total,

            "high_risk_count": len(high),

            "medium_risk_count": len(medium),

            "low_risk_count": len(low),


            "high_risk_tree_ids": high,

            "medium_risk_tree_ids": medium,

            "low_risk_tree_ids": low,


            "inspection_priority": high + medium

        }