"""
Prediction Report Export

功能:

1. JSON风险报告
2. CSV风险列表
3. 自动处理PredictionResult对象
"""


import json
import csv

from pathlib import Path
from datetime import datetime



def serialize_result(data):
    """
    将Python对象转换为JSON可保存格式
    """

    # list处理

    if isinstance(data, list):

        return [
            serialize_result(item)
            for item in data
        ]


    # dict处理

    if isinstance(data, dict):

        result = {}

        for key, value in data.items():

            result[key] = serialize_result(value)

        return result


    # dataclass / 普通对象处理

    if hasattr(data, "__dict__"):

        return {
            key: serialize_result(value)
            for key, value in data.__dict__.items()
        }


    # 基础类型

    return data




class ReportExporter:
    """
    风险报告导出器
    """



    def __init__(
        self,
        output_dir="reports"
    ):

        self.output_dir = Path(
            output_dir
        )


        self.output_dir.mkdir(
            exist_ok=True
        )



    def export_json(
        self,
        data,
        filename="risk_report.json"
    ):

        """
        导出JSON报告
        """


        path = (
            self.output_dir /
            filename
        )


        content = {

            "created_time":
                datetime.now().isoformat(),


            "data":
                serialize_result(data)

        }



        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:


            json.dump(

                content,

                f,

                ensure_ascii=False,

                indent=4

            )


        return str(path)




    def export_csv(
        self,
        results,
        filename="tree_risk.csv"
    ):

        """
        导出CSV巡检列表
        """


        path = (
            self.output_dir /
            filename
        )



        with open(

            path,

            "w",

            encoding="utf-8-sig",

            newline=""

        ) as f:



            writer = csv.DictWriter(

                f,

                fieldnames=[

                    "tree_id",

                    "risk_level",

                    "risk_score",

                    "suggestion"

                ]

            )



            writer.writeheader()



            for item in results:


                result = item["result"]



                writer.writerow({

                    "tree_id":

                        item["tree_id"],


                    "risk_level":

                        result.risk_level,


                    "risk_score":

                        result.risk_score,


                    "suggestion":

                        result.suggestion

                })



        return str(path)