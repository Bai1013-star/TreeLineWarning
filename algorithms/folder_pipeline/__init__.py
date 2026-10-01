"""FOLDER_TO_EXCEL_V1: folder -> manual span -> science -> editable trees -> Excel.

This package wires the existing authoritative science (Wire V2, high-recall
vegetation, exact original-index Wire/Tower exclusion, CROWN_STEM_HYBRID tree
segmentation, exact Tree-Wire clearance, danger, persistent identity) behind a
single local three-step browser workflow:

    STEP 1  /step1  manual span (tower A / tower B) selection
    STEP 2  /step2  human tree review (DELETE / MERGE / UNDO)
    STEP 3  /step3  final result + 树木风险结果.xlsx

The workspace layout is fixed:

    workspace/
      input/pointcloud/source.las|.laz      (exactly one source, else fail closed)
      source_cache/                          science staging (run checkpoints etc.)
      01_span_selection/span_selection.json
      02_tree_analysis/Prediction_Input.xlsx  internal final cache and metadata
      03_export/树木风险结果.xlsx               user workbook, 23 columns

Every science step calls the current authoritative implementation; nothing is
re-implemented.  Display sampling never enters science membership.
"""

from __future__ import annotations

FOLDER_PIPELINE_PACKAGE = "folder-to-excel-v1"

__all__ = ["FOLDER_PIPELINE_PACKAGE"]
