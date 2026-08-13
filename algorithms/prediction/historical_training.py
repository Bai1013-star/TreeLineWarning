"""Import tabular history and safely fine-tune the temporal attention model."""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from openpyxl import load_workbook

from algorithms.prediction.temporal_attention import (
    FEATURE_NAMES,
    NumpyAttentionLSTM,
    _temporal_value,
    load_temporal_model,
    resolve_temporal_model_path,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 50_000
TRAINING_LOCK = Lock()
LOGGER = logging.getLogger(__name__)


def _key(value: Any) -> str:
    return re.sub(r"[\s_（）()/%°.\-]+", "", str(value).strip().lower())


FIELD_ALIASES = {
    "tree_id": ("tree_id", "treeid", "树木编号", "树编号", "单树编号"),
    "species": ("species", "树种", "树木种类"),
    "observation_date": ("observation_date", "date", "观测日期", "巡检日期", "日期"),
    "observation_year": ("observation_year", "year", "观测年份", "年份"),
    "observation_month": ("observation_month", "month", "观测月份", "月份"),
    "observation_day": ("observation_day", "day", "观测日", "日"),
    "height": ("height", "tree_height", "树高", "树高m"),
    "age": ("age", "树龄"),
    "diameter": ("diameter", "dbh", "胸径", "胸径cm"),
    "vertical_distance": ("vertical_distance", "垂直距离", "垂直距离m"),
    "horizontal_distance": ("horizontal_distance", "水平距离", "水平距离m"),
    "clearance_distance": (
        "clearance_distance",
        "distance_to_wire",
        "净空距离",
        "树线距离",
        "净空距离m",
    ),
    "wind_offset_distance": ("wind_offset_distance", "风偏距离", "风偏距离m"),
    "voltage_level_kv": ("voltage_level_kv", "voltage", "电压等级", "电压等级kv"),
    "maximum_sag_winter": ("maximum_sag_winter", "冬季最大弧垂", "冬季弧垂"),
    "maximum_sag_summer": ("maximum_sag_summer", "夏季最大弧垂", "夏季弧垂"),
    "wind_speed": ("wind_speed", "风速", "风速ms"),
    "rainfall": ("rainfall", "降雨", "降雨量", "降雨量mm"),
    "humidity": ("humidity", "湿度"),
    "temperature": ("temperature", "温度", "温度c"),
    "sunlight_hours": ("sunlight_hours", "sunlight", "日照", "日照时长", "日照时长h"),
    "risk_score": ("risk_score", "score", "风险分", "风险评分"),
    "risk_level": ("risk_level", "level", "风险等级"),
}
ALIAS_LOOKUP = {
    _key(alias): canonical for canonical, aliases in FIELD_ALIASES.items() for alias in aliases
}

REQUIRED_FIELDS = (
    "tree_id",
    "species",
    "height",
    "diameter",
    "vertical_distance",
    "horizontal_distance",
    "clearance_distance",
    "wind_speed",
    "rainfall",
    "humidity",
    "temperature",
    "sunlight_hours",
)
NUMERIC_FIELDS = {
    "height",
    "age",
    "diameter",
    "vertical_distance",
    "horizontal_distance",
    "clearance_distance",
    "wind_offset_distance",
    "voltage_level_kv",
    "maximum_sag_winter",
    "maximum_sag_summer",
    "wind_speed",
    "rainfall",
    "humidity",
    "temperature",
    "sunlight_hours",
    "risk_score",
}
RISK_LEVEL_SCORES = {
    "低风险": 20.0,
    "低": 20.0,
    "low": 20.0,
    "中风险": 55.0,
    "中": 55.0,
    "medium": 55.0,
    "高风险": 85.0,
    "高": 85.0,
    "high": 85.0,
}


def parse_tabular_file(
    filename: str,
    content: bytes,
    *,
    max_upload_bytes: int,
    max_rows: int,
) -> dict[str, Any]:
    """Parse an uploaded Excel/text table without applying training requirements."""

    if len(content) > max_upload_bytes:
        raise ValueError(f"文件不能超过 {max_upload_bytes // (1024 * 1024)} MB")
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        rows, sheet_name = _rows_from_xlsx(content, max_rows=max_rows)
        source_format = suffix.lstrip(".")
    elif suffix in {".csv", ".tsv", ".txt"}:
        rows = _rows_from_delimited(_decode_text(content), "\t" if suffix == ".tsv" else None)
        sheet_name = None
        source_format = suffix.lstrip(".")
    elif suffix == ".json":
        rows = _rows_from_json(_decode_text(content))
        sheet_name = None
        source_format = "json"
    else:
        raise ValueError("支持 .xlsx、.xlsm、.csv、.tsv、.txt 和 .json 文件")
    if not rows:
        raise ValueError("文件中没有数据")
    if len(rows) > max_rows:
        raise ValueError(f"单次最多导入 {max_rows} 行")
    return {
        "source_name": Path(filename).name,
        "source_format": source_format,
        "sheet_name": sheet_name,
        "row_count": len(rows),
        "records": rows,
    }


def parse_history_file(filename: str, content: bytes) -> dict[str, Any]:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("文件不能超过 10 MB")
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        rows, sheet_name = _rows_from_xlsx(content, max_rows=MAX_ROWS)
        source_format = suffix.lstrip(".")
    elif suffix in {".csv", ".tsv", ".txt"}:
        text = _decode_text(content)
        rows = _rows_from_delimited(text, "\t" if suffix == ".tsv" else None)
        sheet_name = None
        source_format = suffix.lstrip(".")
    elif suffix == ".json":
        rows = _rows_from_json(_decode_text(content))
        sheet_name = None
        source_format = "json"
    else:
        raise ValueError("支持 .xlsx、.xlsm、.csv、.tsv、.txt 和 .json 文件")
    return normalize_history_rows(
        rows,
        source_name=Path(filename).name,
        source_format=source_format,
        sheet_name=sheet_name,
    )


def parse_history_text(text: str, source_format: str = "auto") -> dict[str, Any]:
    if not text.strip():
        raise ValueError("粘贴内容不能为空")
    if len(text.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise ValueError("粘贴内容不能超过 10 MB")
    stripped = text.lstrip()
    if source_format == "json" or (source_format == "auto" and stripped.startswith(("[", "{"))):
        rows = _rows_from_json(text)
        resolved = "json"
    else:
        rows = _rows_from_delimited(text, "\t" if source_format == "tsv" else None)
        resolved = "tsv" if "\t" in text.splitlines()[0] else "csv"
    return normalize_history_rows(
        rows,
        source_name="粘贴数据",
        source_format=resolved,
        sheet_name=None,
    )


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("文本文件编码无法识别，请使用 UTF-8 或 GB18030")


def _rows_from_xlsx(content: bytes, *, max_rows: int) -> tuple[list[dict[str, Any]], str]:
    try:
        workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    except Exception as exc:
        raise ValueError("Excel 文件无法读取或文件已损坏") from exc
    try:
        worksheet = next(
            (sheet for sheet in workbook.worksheets if sheet.max_row > 0 and sheet.max_column > 0),
            workbook.active,
        )
        values = []
        for row in worksheet.iter_rows(values_only=True):
            if any(value is not None and str(value).strip() for value in row):
                values.append(tuple(row))
                if len(values) > max_rows + 1:
                    break
        return _matrix_to_rows(values), worksheet.title
    finally:
        workbook.close()


def _rows_from_delimited(text: str, delimiter: str | None = None) -> list[dict[str, Any]]:
    sample = text[:4096]
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
        except csv.Error:
            delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        raise ValueError("没有识别到表头")
    return [dict(row) for row in reader]


def _rows_from_json(text: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 格式错误：第 {exc.lineno} 行") from exc
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("history"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("JSON 必须是对象数组，或包含 records/history 数组")
    return payload


def _matrix_to_rows(values: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    nonempty = [
        tuple(row)
        for row in values
        if any(value is not None and str(value).strip() for value in row)
    ]
    if not nonempty:
        raise ValueError("Excel 工作表为空")
    headers = [str(value).strip() if value is not None else "" for value in nonempty[0]]
    if not any(headers):
        raise ValueError("Excel 第一行必须是表头")
    return [
        {header: value for header, value in zip(headers, row, strict=False) if header}
        for row in nonempty[1:]
    ]


def normalize_history_rows(
    rows: Iterable[dict[str, Any]],
    *,
    source_name: str,
    source_format: str,
    sheet_name: str | None,
) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("文件中没有历史数据")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"单次最多导入 {MAX_ROWS} 行")

    valid: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    detected_columns: set[str] = set()
    for row_number, raw in enumerate(rows, start=2):
        mapped: dict[str, Any] = {}
        for header, value in raw.items():
            canonical = ALIAS_LOOKUP.get(_key(header))
            if canonical:
                mapped[canonical] = value
                detected_columns.add(canonical)
        try:
            valid.append(_normalize_row(mapped, row_number))
        except (TypeError, ValueError) as exc:
            if len(errors) < 200:
                errors.append({"row": row_number, "message": str(exc)})

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        groups[str(row["tree_id"])].append(row)
    for records in groups.values():
        records.sort(key=lambda item: item["observation_date"])
    trainable_sequences = sum(max(0, len(records) - 1) for records in groups.values())
    eligible_trees = sum(len(records) >= 2 for records in groups.values())
    preview = sorted(valid, key=lambda item: (str(item["tree_id"]), item["observation_date"]))[:10]
    return {
        "source_name": source_name,
        "source_format": source_format,
        "sheet_name": sheet_name,
        "row_count": len(rows),
        "valid_row_count": len(valid),
        "invalid_row_count": len(rows) - len(valid),
        "tree_count": len(groups),
        "eligible_tree_count": eligible_trees,
        "trainable_sequence_count": trainable_sequences,
        "detected_columns": sorted(detected_columns),
        "records": valid,
        "preview": preview,
        "errors": errors,
        "trainable": trainable_sequences >= 2,
        "requirements": [
            "同一树木编号至少需要两个按日期排序的观测点",
            "每行必须提供风险分，或提供可映射的风险等级",
            "树高、胸径、四类距离、风雨湿温和日照必须完整",
        ],
    }


def _normalize_row(row: dict[str, Any], row_number: int) -> dict[str, Any]:
    missing = [
        field
        for field in REQUIRED_FIELDS
        if row.get(field) is None or str(row.get(field)).strip() == ""
    ]
    if missing:
        raise ValueError(f"缺少字段：{', '.join(missing)}")
    observed_on = _parse_date(row)
    result: dict[str, Any] = {
        "tree_id": str(row["tree_id"]).strip(),
        "species": str(row["species"]).strip() or "其他",
        "observation_date": observed_on.isoformat(),
        "observation_year": observed_on.year,
        "observation_month": observed_on.month,
        "observation_day": observed_on.day,
        "month": observed_on.month,
    }
    for field in NUMERIC_FIELDS:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        try:
            result[field] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} 必须是数值") from exc

    result.setdefault("age", 0.0)
    result.setdefault("wind_offset_distance", 0.0)
    result.setdefault("voltage_level_kv", 110.0)
    result.setdefault("maximum_sag_winter", 3.5)
    result.setdefault("maximum_sag_summer", 5.0)
    if "risk_score" not in result:
        level = str(row.get("risk_level", "")).strip().lower()
        if level not in RISK_LEVEL_SCORES:
            raise ValueError("缺少有效风险分或风险等级")
        result["risk_score"] = RISK_LEVEL_SCORES[level]
    if not 0 <= result["risk_score"] <= 100:
        raise ValueError("风险分必须在 0–100 之间")
    for field in (
        "height",
        "diameter",
        "vertical_distance",
        "horizontal_distance",
        "clearance_distance",
    ):
        if result[field] <= 0:
            raise ValueError(f"{field} 必须大于 0")
    if not 0 <= result["humidity"] <= 100:
        raise ValueError("湿度必须在 0–100% 之间")
    if not -60 <= result["temperature"] <= 60:
        raise ValueError("温度必须在 -60–60℃ 之间")
    if not 0 <= result["sunlight_hours"] <= 24:
        raise ValueError("日照时长必须在 0–24 小时之间")
    result["_source_row"] = row_number
    return result


def _parse_date(row: dict[str, Any]) -> date:
    value = row.get("observation_date")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is not None and str(value).strip():
        text = str(value).strip().replace("年", "-").replace("月", "-").replace("日", "")
        text = text.replace("/", "-").split(" ")[0]
        parts = text.split("-")
        if len(parts) == 3:
            try:
                return date(int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError as exc:
                raise ValueError("观测日期无效") from exc
        raise ValueError("观测日期必须为 YYYY-MM-DD")
    try:
        return date(
            int(float(row["observation_year"])),
            int(float(row["observation_month"])),
            int(float(row.get("observation_day", 1) or 1)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("缺少有效观测日期或年/月/日") from exc


def fine_tune_with_history(
    records: list[dict[str, Any]],
    *,
    model_path: str | Path | None = None,
    output_path: str | Path | None = None,
    epochs: int = 15,
    learning_rate: float = 0.0005,
    seed: int = 20260731,
    allow_replace: bool = False,
) -> dict[str, Any]:
    """Fine-tune a source model and write only an explicit independent output."""
    if not 1 <= int(epochs) <= 50:
        raise ValueError("训练轮数必须在 1–50 之间")
    if not 0 < float(learning_rate) <= 0.01:
        raise ValueError("学习率必须在 0–0.01 之间")
    model_path = resolve_temporal_model_path(model_path)
    if output_path is None or not str(output_path).strip():
        raise ValueError("微调必须显式提供 output_path，且不得覆盖输入模型。")
    output_path = Path(output_path)
    if output_path == model_path:
        raise ValueError("微调 output_path 必须与输入 model_path 不同。")
    if output_path.exists() and not allow_replace:
        raise FileExistsError(f"微调输出已存在：{output_path}")

    with TRAINING_LOCK:
        model, metadata = load_temporal_model(model_path)
        samples = _build_training_samples(records, metadata)
        if len(samples) < 2:
            raise ValueError("至少需要两个可训练时序样本")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(samples))
        validation_count = max(1, int(round(len(samples) * 0.2)))
        validation_idx = order[-validation_count:]
        training_idx = order[:-validation_count]
        if not len(training_idx):
            raise ValueError("有效训练样本不足")

        features = np.stack([sample[0] for sample in samples]).astype(np.float32)
        species_ids = np.asarray([sample[1] for sample in samples], dtype=np.int64)
        months = np.asarray([sample[2] for sample in samples], dtype=np.float32)
        targets = np.asarray([sample[3] for sample in samples], dtype=np.float32)
        before_mae = _mae(
            model,
            features[validation_idx],
            species_ids[validation_idx],
            months[validation_idx],
            targets[validation_idx],
        )
        best_mae = before_mae
        best_params = deepcopy(model.params)
        epoch_history = []
        for epoch in range(1, int(epochs) + 1):
            shuffled = rng.permutation(training_idx)
            losses = []
            for start in range(0, len(shuffled), 16):
                batch = shuffled[start : start + 16]
                losses.append(
                    model.train_batch(
                        features[batch],
                        species_ids[batch],
                        months[batch],
                        targets[batch],
                        learning_rate=float(learning_rate),
                    )
                )
            validation_mae = _mae(
                model,
                features[validation_idx],
                species_ids[validation_idx],
                months[validation_idx],
                targets[validation_idx],
            )
            epoch_history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(float(np.mean(losses)), 7),
                    "validation_mae": round(validation_mae, 4),
                }
            )
            if validation_mae < best_mae:
                best_mae = validation_mae
                best_params = deepcopy(model.params)

        model.params = best_params
        deployed = best_mae < before_mae
        trained_at = datetime.now().isoformat(timespec="seconds")
        if deployed:
            metadata = deepcopy(metadata)
            fine_tuning_history = list(metadata.get("fine_tuning_history", []))
            fine_tuning_history.append(
                {
                    "trained_at": trained_at,
                    "training_samples": int(len(training_idx)),
                    "validation_samples": int(len(validation_idx)),
                    "before_validation_mae": round(before_mae, 4),
                    "after_validation_mae": round(best_mae, 4),
                }
            )
            metadata["fine_tuning_history"] = fine_tuning_history[-20:]
            metadata["last_fine_tuning"] = fine_tuning_history[-1]
            model.save(output_path, metadata, allow_replace=allow_replace)
            LOGGER.info(
                "Temporal model fine-tuned: samples=%d, validation_mae=%.4f",
                len(samples),
                best_mae,
            )

        return {
            "status": "deployed" if deployed else "not_deployed",
            "deployed": deployed,
            "message": (
                "验证误差下降，增量训练模型已发布"
                if deployed
                else "验证误差未改善，已保留原线上模型"
            ),
            "trained_at": trained_at,
            "sequence_samples": len(samples),
            "training_samples": int(len(training_idx)),
            "validation_samples": int(len(validation_idx)),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "before_validation_mae": round(before_mae, 4),
            "after_validation_mae": round(best_mae, 4),
            "improvement": round(before_mae - best_mae, 4),
            "output_path": str(output_path) if deployed else None,
            "epoch_history": epoch_history,
        }


def _build_training_samples(
    records: list[dict[str, Any]], metadata: dict[str, Any]
) -> list[tuple[np.ndarray, int, float, float]]:
    sequence_length = int(metadata["sequence_length"])
    mean = np.asarray(metadata["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(metadata["normalization"]["std"], dtype=np.float32)
    species_map = dict(metadata["species_map"])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["tree_id"])].append(record)
    samples = []
    for tree_records in groups.values():
        tree_records.sort(key=lambda item: item["observation_date"])
        for end in range(1, len(tree_records)):
            sequence = tree_records[max(0, end - sequence_length + 1) : end + 1]
            values = np.asarray(
                [[_temporal_value(record, name) for name in FEATURE_NAMES] for record in sequence],
                dtype=np.float32,
            )
            if len(values) < sequence_length:
                pad = np.repeat(values[:1], sequence_length - len(values), axis=0)
                values = np.concatenate((pad, values), axis=0)
            normalized = (values - mean) / np.where(std < 1e-6, 1, std)
            latest = sequence[-1]
            species_id = species_map.get(latest["species"], species_map.get("其他", 0))
            samples.append(
                (
                    normalized,
                    int(species_id),
                    float(latest["observation_month"]),
                    float(latest["risk_score"]),
                )
            )
    return samples


def _mae(
    model: NumpyAttentionLSTM,
    features: np.ndarray,
    species_ids: np.ndarray,
    months: np.ndarray,
    targets: np.ndarray,
) -> float:
    predictions = model.forward(features, species_ids, months)["risk_score"]
    return float(np.mean(np.abs(predictions - targets)))


def training_template_csv() -> str:
    headers = [
        "tree_id",
        "species",
        "observation_date",
        "height",
        "age",
        "diameter",
        "vertical_distance",
        "horizontal_distance",
        "clearance_distance",
        "wind_offset_distance",
        "voltage_level_kv",
        "maximum_sag_winter",
        "maximum_sag_summer",
        "wind_speed",
        "rainfall",
        "humidity",
        "temperature",
        "sunlight_hours",
        "risk_score",
        "risk_level",
    ]
    examples = [
        [
            1,
            "杨树",
            "2026-01-15",
            17.2,
            20,
            33.2,
            6.4,
            6.2,
            5.4,
            0.8,
            110,
            3.5,
            5,
            6,
            20,
            60,
            5,
            5,
            32,
            "低风险",
        ],
        [
            1,
            "杨树",
            "2026-02-15",
            17.22,
            20.1,
            33.24,
            6.39,
            6.19,
            5.39,
            0.8,
            110,
            3.5,
            5,
            7,
            28,
            62,
            8,
            6,
            35,
            "低风险",
        ],
        [
            1,
            "杨树",
            "2026-03-15",
            17.26,
            20.2,
            33.31,
            6.37,
            6.18,
            5.37,
            0.9,
            110,
            3.5,
            5,
            9,
            42,
            66,
            14,
            7,
            43,
            "中风险",
        ],
    ]
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(examples)
    return "\ufeff" + stream.getvalue()
