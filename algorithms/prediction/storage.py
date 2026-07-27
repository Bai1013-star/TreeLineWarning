"""
Prediction Storage

保存和读取预测结果
"""

import json
from pathlib import Path
from datetime import datetime



class PredictionStorage:
    """
    预测结果存储器
    """


    def __init__(self, file_path="prediction_results.json"):

        self.file_path = Path(file_path)



    def save(self, results):

        data = {

            "created_time":
                datetime.now().isoformat(),

            "results":
                results

        }


        with self.file_path.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=4,
                default=str
            )


        return self.file_path



    def load(self):

        if not self.file_path.exists():

            return None


        with self.file_path.open(
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)