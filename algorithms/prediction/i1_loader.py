"""
I1 Loader

作用:
读取I1相关数据
转换为预测系统输入
"""

import json
from pathlib import Path

from algorithms.prediction.i1_prediction import (
    I1TreeData
)


class I1Loader:
    """
    I1数据读取器
    """


    def __init__(self, i1_path):

        self.root = Path(i1_path)



    def load_manifest(self):

        """
        读取I1清单
        """

        manifest = self.root / "i1_manifest.json"


        if not manifest.exists():

            raise FileNotFoundError(
                "i1_manifest.json not found"
            )


        with manifest.open(
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)



    def load_trees(self):

        """
        加载树木信息

        当前版本:
        支持I1目录中的tree JSON数据

        后续:
        接入真实V2结果
        """

        tree_file = self.root / "trees.json"


        if not tree_file.exists():

            return []



        with tree_file.open(
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)



        trees = []


        for item in data:


            trees.append(

                I1TreeData(

                    tree_id=item["tree_id"],

                    species=item.get(
                        "species",
                        "未知"
                    ),

                    height=item.get(
                        "height",
                        0
                    ),

                    age=item.get(
                        "age",
                        0
                    ),

                    diameter=item.get(
                        "diameter",
                        0
                    ),

                    distance_to_wire=item.get(
                        "distance_to_wire",
                        999
                    ),

                    wind_speed=item.get(
                        "wind_speed",
                        0
                    ),

                    rainfall=item.get(
                        "rainfall",
                        0
                    ),

                    humidity=item.get(
                        "humidity",
                        0
                    ),

                    slope=item.get(
                        "slope",
                        0
                    )
                )
            )


        return trees