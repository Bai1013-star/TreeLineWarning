"""
Risk Report Generator

负责把预测结果转换成最终风险报告
"""


class RiskReportGenerator:
    """
    风险报告生成器
    """


    def generate(self, results):

        if not results:

            return {
                "total_tree": 0,
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "low_risk_count": 0,
                "high_risk_tree_ids": [],
                "medium_risk_tree_ids": [],
                "low_risk_tree_ids": [],

                "risk_level": "未知",

                "risk_score": 0,

                "suggestion": "暂无数据"

            }


        max_score = 0

        risk_level = "低风险"

        suggestion = "树木状态正常"
        high_ids = []
        medium_ids = []
        low_ids = []



        for item in results:


            result = item.get(
                "result",
                {}
            )


            if isinstance(result, dict):

                score = result.get(
                    "risk_score",
                    0
                )

                level = result.get(
                    "risk_level",
                    "未知"
                )


            else:

                score = getattr(
                    result,
                    "risk_score",
                    0
                )

                level = getattr(
                    result,
                    "risk_level",
                    "未知"
                )

            tree_id = item.get("tree_id")
            if level == "高风险":
                high_ids.append(tree_id)
            elif level == "中风险":
                medium_ids.append(tree_id)
            else:
                low_ids.append(tree_id)


            if score > max_score:

                max_score = score

                risk_level = level



        if max_score >= 80:

            suggestion = (
                "高风险，请立即检查树木与线路距离"
            )


        elif max_score >= 50:

            suggestion = (
                "中风险，建议安排巡检"
            )


        else:

            suggestion = (
                "树木状态正常"
            )



        return {
            "total_tree": len(results),
            "high_risk_count": len(high_ids),
            "medium_risk_count": len(medium_ids),
            "low_risk_count": len(low_ids),
            "high_risk_tree_ids": high_ids,
            "medium_risk_tree_ids": medium_ids,
            "low_risk_tree_ids": low_ids,

            "risk_level": risk_level,

            "risk_score": max_score,

            "suggestion": suggestion

        }



# ================================
# 兼容旧代码
# pipeline.py 需要 RiskReport
# ================================


class RiskReport(RiskReportGenerator):

    pass
