"""
Real I1 Adapter

连接:
algorithms.integration.i1

把真实I1 bundle
转换为预测输入
"""


from algorithms.prediction.i1_prediction import (
    I1TreeData
)



class RealI1TreeAdapter:
    """
    真实I1树木适配器
    """


    def convert(
        self,
        bundle
    ):

        trees = []


        # 当前先读取V2 clusters
        # 后续根据真实V2字段调整

        if not hasattr(
            bundle,
            "v2_result"
        ):

            return trees



        clusters = (
            bundle.v2_result.clusters
        )


        for cluster in clusters:


            tree_id = cluster.get(
                "cluster_id",
                len(trees)+1
            )


            tree = I1TreeData(

                tree_id=tree_id,

                species=cluster.get(
                    "species",
                    "未知"
                ),

                height=cluster.get(
                    "height",
                    0
                ),

                age=cluster.get(
                    "age",
                    0
                ),

                diameter=cluster.get(
                    "diameter",
                    0
                ),

                distance_to_wire=cluster.get(
                    "distance_to_wire",
                    999
                ),

                wind_speed=0,

                rainfall=0,

                humidity=0,

                slope=0
            )


            trees.append(tree)


        return trees