"""Backward-compatible I1 prediction pipeline entry points."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from algorithms.prediction.i1_loader import I1Loader
from algorithms.prediction.pipeline import PredictionPipeline

logger = logging.getLogger(__name__)


class I1PredictionPipeline:
    def __init__(self, pipeline: PredictionPipeline | None = None):
        self.pipeline = pipeline or PredictionPipeline()
        self.predict_pipeline = self.pipeline

    def run(self, i1_path: str | Path) -> dict[str, Any]:
        """Load an I1 directory and return the legacy batch result shape."""
        trees = I1Loader(i1_path).load_trees()
        if not trees:
            return {
                "results": [],
                "report": {
                    "total_tree": 0,
                    "high_risk_count": 0,
                    "medium_risk_count": 0,
                    "low_risk_count": 0,
                    "high_risk_tree_ids": [],
                    "medium_risk_tree_ids": [],
                    "low_risk_tree_ids": [],
                    "risk_level": "低风险",
                    "risk_score": 0,
                    "suggestion": "暂无数据",
                },
            }
        result = self.pipeline.run(trees)
        logger.info("I1 prediction completed: trees=%d", len(result["results"]))
        return result

    def predict(self, tree_data: Any) -> list[dict[str, Any]]:
        """Predict one already-materialized tree input without replacing ``run``."""
        result = self.pipeline.run([tree_data])
        logger.info("Single-tree prediction completed")
        return result["results"]
