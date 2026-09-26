"""
I1 Batch Prediction Service

作用:
1. 接收 API 输入的树木列表
2. 统一转换成 I1TreeData
3. 调用 I1PredictionService
"""

from algorithms.prediction.i1_prediction import I1PredictionService, I1TreeData


class I1BatchPredictionService:
    """
    I1 批量预测服务
    """

    def __init__(self):

        self.service = I1PredictionService()



    def predict_batch(self, trees):
        """
        批量预测

        参数:
            trees:
                list[dict]
                或
                list[I1TreeData]

        返回:
            list[
                {
                    tree_id,
                    result
                }
            ]
        """

        results = []


        for index, tree in enumerate(trees):

            # ===============================
            # 情况1:
            # API传入dict
            # ===============================

            if isinstance(tree, dict):

                tree_data = I1TreeData(

                    tree_id=tree.get(
                        "tree_id",
                        index + 1
                    ),


                    species=tree.get(
                        "species",
                        "未知"
                    ),


                    height=float(
                        tree.get(
                            "height",
                            0
                        )
                    ),


                    age=int(
                        tree.get(
                            "age",
                            0
                        )
                    ),


                    diameter=float(
                        tree.get(
                            "diameter",
                            0
                        )
                    ),


                    distance_to_wire=float(
                        tree.get(
                            "distance_to_wire",
                            tree.get("clearance_distance", 0)
                        )
                    ),

                    vertical_distance=float(
                        tree.get("vertical_distance", 999)
                    ),

                    horizontal_distance=float(
                        tree.get("horizontal_distance", 999)
                    ),

                    clearance_distance=float(
                        tree.get(
                            "clearance_distance",
                            tree.get("distance_to_wire", 999)
                        )
                    ),

                    wind_offset_distance=float(
                        tree.get("wind_offset_distance", 0)
                    ),

                    voltage_level_kv=float(
                        tree.get("voltage_level_kv", 110)
                    ),

                    maximum_sag_winter=float(
                        tree.get("maximum_sag_winter", 3.5)
                    ),

                    maximum_sag_summer=float(
                        tree.get("maximum_sag_summer", 5.0)
                    ),

                    observation_month=int(
                        tree.get("observation_month", 7)
                    ),


                    wind_speed=float(
                        tree.get(
                            "wind_speed",
                            0
                        )
                    ),


                    rainfall=float(
                        tree.get(
                            "rainfall",
                            0
                        )
                    ),


                    humidity=float(
                        tree.get(
                            "humidity",
                            0
                        )
                    ),

                    temperature=float(
                        tree.get(
                            "temperature",
                            0
                        )
                    ),

                    sunlight_hours=float(
                        tree.get(
                            "sunlight_hours",
                            tree.get("sunlight", 0)
                        )
                    ),


                    slope=float(
                        tree.get(
                            "slope",
                            0
                        )
                    ),

                    standards=tree.get("standards")

                )


            # ===============================
            # 情况2:
            # 已经是I1TreeData
            # ===============================

            elif isinstance(tree, I1TreeData):

                tree_data = tree


            else:

                raise TypeError(
                    f"不支持的数据类型: {type(tree)}"
                )



            # ===============================
            # 执行单树预测
            # ===============================

            result = self.service.predict_tree(
                tree_data
            )


            results.append(result)



        return results
