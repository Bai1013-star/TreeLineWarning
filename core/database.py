"""SQLite persistence for TreeLineWarning predictions."""

from __future__ import annotations

import ast
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from algorithms.prediction.standards import PredictionStandards

DATABASE_PATH_ENV = "TREELINE_WARNING_DATABASE_PATH"

EXTENDED_INPUT_FIELDS = (
    "line_name",
    "tower_section",
    "province",
    "city",
    "district",
    "gps_longitude",
    "gps_latitude",
    "tree_code",
    "crown_width",
    "growth_direction",
    "historical_patrol_distance",
    "crown_expansion_speed",
    "pruning_record",
    "span_length",
    "conductor_height",
    "line_direction",
    "wind_direction",
    "weather_source",
    "weather_observed_at",
    "season_stage",
)


def _scalar_text(value: Any, default: str) -> str:
    """Return a database-safe scalar string, never a serialized dict/list."""
    if value is None:
        return default
    if isinstance(value, (dict, list, tuple, set)):
        return default
    return str(value)


def _read_legacy_text(value: Any, field: str, default: str) -> str:
    """Extract a scalar from rows written by older dict-serializing versions."""
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            decoded = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return default
        if isinstance(decoded, dict):
            nested = decoded.get("result")
            if isinstance(nested, dict):
                decoded = nested
            aliases = (field, "suggestion") if field == "prediction" else (field,)
            for key in aliases:
                if key in decoded:
                    return _scalar_text(decoded[key], default)
            return default
    return _scalar_text(value, default)


def _serialize_risk_snapshot(value: Any) -> str | None:
    """Serialize the deliberately small auditable D/G/M/H snapshot."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("risk_snapshot 必须是对象")
    allowed = {
        "prediction_mode",
        "prediction_mode_label",
        "comprehensive_risk",
        "comprehensive_risk_score",
        "hazard_level",
        "risk_escalation_indicator",
        "calculation_ledger",
        "model_status",
        "parameter_source",
        "clearance_defect",
        "attention_prior",
    }
    snapshot = {key: value[key] for key in allowed if key in value}
    try:
        return json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("risk_snapshot 必须可 JSON 序列化") from exc


def _decode_risk_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {"available": False, "reason": "missing_risk_snapshot"}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {"available": False, "reason": "invalid_risk_snapshot"}
    if not isinstance(decoded, dict):
        return {"available": False, "reason": "invalid_risk_snapshot"}
    return decoded


def _decoded_prediction_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["risk_level"] = _read_legacy_text(item.get("risk_level"), "risk_level", "未知")
    item["prediction"] = _read_legacy_text(item.get("prediction"), "prediction", "暂无建议")
    for field in ("standard_profile", "standard_analysis"):
        value = item.get(field)
        if isinstance(value, str) and value:
            try:
                item[field] = json.loads(value)
            except json.JSONDecodeError:
                item[field] = {}
        elif not isinstance(value, dict):
            item[field] = {}
    item["risk_snapshot"] = _decode_risk_snapshot(item.get("risk_snapshot_json"))
    raw_report = item.get("full_report_json")
    try:
        full_report = json.loads(raw_report) if isinstance(raw_report, str) and raw_report else None
    except json.JSONDecodeError:
        full_report = None
    item["full_report"] = full_report if isinstance(full_report, dict) else None
    raw_features = item.get("input_features_json")
    try:
        features = json.loads(raw_features) if isinstance(raw_features, str) else {}
    except json.JSONDecodeError:
        features = {}
    if not isinstance(features, dict):
        features = {}
    item["input_features"] = features
    for field, value in features.items():
        if field in EXTENDED_INPUT_FIELDS:
            item[field] = value
    return item


class Database:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = Lock()
        self._closed = False
        self.create_tables()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Database is closed")

    def create_tables(self) -> None:
        self._ensure_open()
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tree_id INTEGER,
                    species TEXT,
                    height REAL,
                    age INTEGER,
                    diameter REAL,
                    wind_speed REAL,
                    rainfall REAL,
                    humidity REAL,
                    temperature REAL,
                    sunlight_hours REAL,
                    slope REAL,
                    distance_to_wire REAL,
                    vertical_distance REAL,
                    horizontal_distance REAL,
                    clearance_distance REAL,
                    wind_offset_distance REAL,
                    voltage_level_kv REAL,
                    maximum_sag_winter REAL,
                    maximum_sag_summer REAL,
                    observation_date TEXT,
                    observation_year INTEGER,
                    observation_month INTEGER,
                    standard_profile TEXT,
                    standard_analysis TEXT,
                    input_features_json TEXT,
                    risk_snapshot_json TEXT,
                    full_report_json TEXT,
                    risk_level TEXT,
                    prediction TEXT,
                    created_at TEXT
                )
                """
            )
            columns = {
                row["name"] for row in self.conn.execute("PRAGMA table_info(prediction_history)")
            }
            migrations = {
                "temperature": "REAL DEFAULT 0",
                "sunlight_hours": "REAL DEFAULT 0",
                "vertical_distance": "REAL DEFAULT 0",
                "horizontal_distance": "REAL DEFAULT 0",
                "clearance_distance": "REAL DEFAULT 0",
                "wind_offset_distance": "REAL DEFAULT 0",
                "voltage_level_kv": "REAL DEFAULT 0",
                "maximum_sag_winter": "REAL DEFAULT 0",
                "maximum_sag_summer": "REAL DEFAULT 0",
                "observation_date": "TEXT DEFAULT ''",
                "observation_year": "INTEGER DEFAULT 0",
                "observation_month": "INTEGER DEFAULT 0",
                "standard_profile": "TEXT DEFAULT ''",
                "standard_analysis": "TEXT DEFAULT ''",
                "input_features_json": "TEXT DEFAULT ''",
                "risk_snapshot_json": "TEXT DEFAULT ''",
                "full_report_json": "TEXT DEFAULT ''",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    self.conn.execute(
                        f"ALTER TABLE prediction_history ADD COLUMN {name} {definition}"
                    )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prediction_history_tree_id_id
                ON prediction_history(tree_id, id)
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_standards (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    settings_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    source_format TEXT NOT NULL,
                    sheet_name TEXT,
                    row_count INTEGER NOT NULL,
                    valid_row_count INTEGER NOT NULL,
                    invalid_row_count INTEGER NOT NULL,
                    tree_count INTEGER NOT NULL,
                    trainable_sequence_count INTEGER NOT NULL,
                    records_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS weather_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    deployed INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(dataset_id) REFERENCES training_datasets(id)
                )
                """
            )
            existing = self.conn.execute(
                "SELECT id FROM prediction_standards WHERE id = 1"
            ).fetchone()
            if existing is None:
                self.conn.execute(
                    """
                    INSERT INTO prediction_standards (id, settings_json, updated_at)
                    VALUES (1, ?, ?)
                    """,
                    (
                        json.dumps(PredictionStandards().to_dict(), ensure_ascii=False),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
            self.conn.execute("PRAGMA optimize")
            self.conn.commit()

    def save_prediction(self, prediction: dict[str, Any]) -> int:
        """Save one normalized result and return its row id.

        Both the API's flat shape and the pipeline's ``{"result": ...}`` shape
        are accepted. Only scalar risk text is persisted.
        """
        self._ensure_open()
        nested = prediction.get("result")
        result = nested if isinstance(nested, dict) else prediction
        risk_level = _scalar_text(result.get("risk_level"), "未知")
        suggestion = _scalar_text(
            result.get("suggestion", result.get("prediction")),
            "暂无建议",
        )
        standard_profile = prediction.get("standard_profile")
        standard_analysis = prediction.get("standard_analysis")
        if not isinstance(standard_profile, dict):
            standard_profile = {}
        if not isinstance(standard_analysis, dict):
            standard_analysis = {}
        risk_snapshot = _serialize_risk_snapshot(prediction.get("risk_snapshot"))
        full_report = prediction.get("full_report")
        if full_report is not None and not isinstance(full_report, dict):
            raise ValueError("full_report 必须是对象")
        try:
            full_report_json = (
                json.dumps(full_report, ensure_ascii=False, allow_nan=False)
                if full_report
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("full_report 必须可 JSON 序列化") from exc
        input_features = {
            field: prediction[field]
            for field in EXTENDED_INPUT_FIELDS
            if field in prediction and prediction[field] not in (None, "")
        }

        values = (
            prediction.get("tree_id", 0),
            prediction.get("species", ""),
            prediction.get("height", 0),
            prediction.get("age", 0),
            prediction.get("diameter", 0),
            prediction.get("wind_speed", 0),
            prediction.get("rainfall", 0),
            prediction.get("humidity", 0),
            prediction.get("temperature", 0),
            prediction.get("sunlight_hours", prediction.get("sunlight", 0)),
            prediction.get("slope", 0),
            prediction.get("distance_to_wire", 0),
            prediction.get("vertical_distance", 0),
            prediction.get("horizontal_distance", 0),
            prediction.get("clearance_distance", prediction.get("distance_to_wire", 0)),
            prediction.get("wind_offset_distance", 0),
            prediction.get("voltage_level_kv", 0),
            prediction.get("maximum_sag_winter", 0),
            prediction.get("maximum_sag_summer", 0),
            prediction.get("observation_date", ""),
            prediction.get("observation_year", 0),
            prediction.get("observation_month", 0),
            json.dumps(standard_profile, ensure_ascii=False),
            json.dumps(standard_analysis, ensure_ascii=False),
            json.dumps(input_features, ensure_ascii=False, allow_nan=False),
            risk_snapshot,
            full_report_json,
            risk_level,
            suggestion,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        with self._lock:
            cursor = self.conn.execute(
                """
                INSERT INTO prediction_history (
                    tree_id, species, height, age, diameter, wind_speed,
                    rainfall, humidity, temperature, sunlight_hours,
                    slope, distance_to_wire,
                    vertical_distance, horizontal_distance, clearance_distance,
                    wind_offset_distance, voltage_level_kv,
                    maximum_sag_winter, maximum_sag_summer,
                    observation_date, observation_year, observation_month,
                    standard_profile, standard_analysis, input_features_json,
                    risk_snapshot_json,
                    full_report_json,
                    risk_level, prediction, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                values,
            )
            self.conn.commit()
            return int(cursor.lastrowid)

    def get_predictions(self, limit: int | None = None) -> list[dict[str, Any]]:
        self._ensure_open()
        sql = "SELECT * FROM prediction_history ORDER BY id DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (max(0, int(limit)),)
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [_decoded_prediction_row(row) for row in rows]

    def get_prediction(self, prediction_id: int) -> dict[str, Any] | None:
        """Return one decoded history row for the interactive detail view."""
        self._ensure_open()
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM prediction_history WHERE id = ?",
                (int(prediction_id),),
            ).fetchone()
        return _decoded_prediction_row(row) if row is not None else None

    def delete_prediction(self, prediction_id: int) -> dict[str, Any] | None:
        """Permanently erase one prediction, including its saved full report."""
        self._ensure_open()
        normalized_id = int(prediction_id)
        with self._lock:
            row = self.conn.execute(
                "SELECT id, tree_id, species, created_at FROM prediction_history WHERE id = ?",
                (normalized_id,),
            ).fetchone()
            if row is None:
                return None
            self.conn.execute("DELETE FROM prediction_history WHERE id = ?", (normalized_id,))
            self.conn.commit()
        return dict(row)

    def get_predictions_by_tree(self, tree_id: int, limit: int = 36) -> list[dict[str, Any]]:
        """Return one tree's observations in chronological order for LSTM input."""
        self._ensure_open()
        requested = max(0, int(limit))
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT * FROM prediction_history
                WHERE tree_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(tree_id), requested),
            ).fetchall()
        return [_decoded_prediction_row(row) for row in reversed(rows)]

    def get_standards(self) -> dict[str, Any]:
        self._ensure_open()
        with self._lock:
            row = self.conn.execute(
                "SELECT settings_json FROM prediction_standards WHERE id = 1"
            ).fetchone()
        if row is None:
            return PredictionStandards().to_dict()
        try:
            return PredictionStandards.from_mapping(json.loads(row["settings_json"])).to_dict()
        except (json.JSONDecodeError, TypeError, ValueError):
            return PredictionStandards().to_dict()

    def update_standards(self, settings: dict[str, Any]) -> dict[str, Any]:
        self._ensure_open()
        validated = PredictionStandards.from_mapping(settings).to_dict()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO prediction_standards (id, settings_json, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    settings_json = excluded.settings_json,
                    updated_at = excluded.updated_at
                """,
                (
                    json.dumps(validated, ensure_ascii=False),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            self.conn.commit()
        return validated

    def save_training_dataset(self, parsed: dict[str, Any]) -> int:
        self._ensure_open()
        summary = {key: value for key, value in parsed.items() if key not in {"records"}}
        with self._lock:
            cursor = self.conn.execute(
                """
                INSERT INTO training_datasets (
                    source_name, source_format, sheet_name, row_count,
                    valid_row_count, invalid_row_count, tree_count,
                    trainable_sequence_count, records_json, summary_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    parsed["source_name"],
                    parsed["source_format"],
                    parsed.get("sheet_name"),
                    int(parsed["row_count"]),
                    int(parsed["valid_row_count"]),
                    int(parsed["invalid_row_count"]),
                    int(parsed["tree_count"]),
                    int(parsed["trainable_sequence_count"]),
                    json.dumps(parsed["records"], ensure_ascii=False),
                    json.dumps(summary, ensure_ascii=False),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            self.conn.commit()
            return int(cursor.lastrowid)

    def get_training_dataset(self, dataset_id: int) -> dict[str, Any] | None:
        self._ensure_open()
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM training_datasets WHERE id = ?",
                (int(dataset_id),),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["records"] = json.loads(item.pop("records_json"))
        item["summary"] = json.loads(item.pop("summary_json"))
        return item

    def list_training_datasets(self, limit: int = 20) -> list[dict[str, Any]]:
        self._ensure_open()
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id, source_name, source_format, sheet_name, row_count,
                       valid_row_count, invalid_row_count, tree_count,
                       trainable_sequence_count, created_at
                FROM training_datasets
                ORDER BY id DESC LIMIT ?
                """,
                (max(0, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_training_records(self) -> list[dict[str, Any]]:
        """Return records imported by users, excluding generated pool snapshots."""

        self._ensure_open()
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT records_json
                FROM training_datasets
                WHERE source_format <> 'pool_snapshot'
                ORDER BY id ASC
                """
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            parsed = json.loads(row["records_json"])
            if isinstance(parsed, list):
                records.extend(item for item in parsed if isinstance(item, dict))
        return records

    def delete_training_dataset(self, dataset_id: int) -> dict[str, Any] | None:
        """Delete one imported dataset and its linked training-run audit rows.

        The caller is responsible for deciding whether generated pool snapshots
        are eligible for deletion.  Removing an imported dataset immediately
        removes its observations from subsequent calls to ``list_training_records``.
        """

        self._ensure_open()
        normalized_id = int(dataset_id)
        with self._lock:
            row = self.conn.execute(
                """
                SELECT id, source_name, source_format, sheet_name, row_count,
                       valid_row_count, invalid_row_count, tree_count,
                       trainable_sequence_count, created_at
                FROM training_datasets
                WHERE id = ?
                """,
                (normalized_id,),
            ).fetchone()
            if row is None:
                return None
            runs = self.conn.execute(
                "SELECT id, metrics_json FROM training_runs WHERE dataset_id = ?", (normalized_id,)
            ).fetchall()
            try:
                self.conn.execute(
                    "DELETE FROM training_runs WHERE dataset_id = ?",
                    (normalized_id,),
                )
                self.conn.execute(
                    "DELETE FROM training_datasets WHERE id = ?",
                    (normalized_id,),
                )
                self.conn.commit()
            except sqlite3.Error:
                self.conn.rollback()
                raise
        result = dict(row)
        result["runs"] = [
            {"id": int(run["id"]), "metrics": json.loads(run["metrics_json"])} for run in runs
        ]
        return result

    def save_training_run(self, dataset_id: int, metrics: dict[str, Any]) -> int:
        self._ensure_open()
        with self._lock:
            cursor = self.conn.execute(
                """
                INSERT INTO training_runs (
                    dataset_id, status, deployed, metrics_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(dataset_id),
                    str(metrics["status"]),
                    int(bool(metrics["deployed"])),
                    json.dumps(metrics, ensure_ascii=False),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            self.conn.commit()
            return int(cursor.lastrowid)

    def list_training_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        self._ensure_open()
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id, dataset_id, status, deployed, metrics_json, created_at
                FROM training_runs ORDER BY id DESC LIMIT ?
                """,
                (max(0, int(limit)),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["deployed"] = bool(item["deployed"])
            item["metrics"] = json.loads(item.pop("metrics_json"))
            result.append(item)
        return result

    def delete_training_run(self, run_id: int) -> dict[str, Any] | None:
        """Erase one incremental-training audit row and return its model path metadata."""
        self._ensure_open()
        normalized_id = int(run_id)
        with self._lock:
            row = self.conn.execute(
                "SELECT id, dataset_id, metrics_json FROM training_runs WHERE id = ?",
                (normalized_id,),
            ).fetchone()
            if row is None:
                return None
            self.conn.execute("DELETE FROM training_runs WHERE id = ?", (normalized_id,))
            self.conn.commit()
        item = dict(row)
        item["metrics"] = json.loads(item.pop("metrics_json"))
        return item

    def save_weather_template(self, name: str, plan: dict[str, Any]) -> int:
        """Persist editable weather scenarios independently of prediction runs."""
        self._ensure_open()
        with self._lock:
            cursor = self.conn.execute(
                "INSERT INTO weather_templates(name,plan_json,created_at) VALUES(?,?,?)",
                (name, json.dumps(plan, ensure_ascii=False), datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            )
            self.conn.commit()
            return int(cursor.lastrowid)

    def list_weather_templates(self) -> list[dict[str, Any]]:
        self._ensure_open()
        with self._lock:
            rows = self.conn.execute(
                'SELECT id,name,plan_json,created_at FROM weather_templates ORDER BY id DESC LIMIT 100'
            ).fetchall()
        return [{**dict(row), 'plan': json.loads(row['plan_json'])} for row in rows]

    def delete_weather_template(self, template_id: int) -> bool:
        self._ensure_open()
        with self._lock:
            cursor = self.conn.execute('DELETE FROM weather_templates WHERE id=?', (template_id,))
            self.conn.commit()
            return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.conn.close()
            self._closed = True


def default_database_path() -> Path:
    """Return the configured database path without opening or creating it."""
    configured_path = os.getenv(DATABASE_PATH_ENV)
    if configured_path:
        return Path(configured_path).expanduser()
    application_data = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
    if application_data:
        return Path(application_data) / "TreeLineWarning" / "prediction_history.db"
    return Path.home() / ".local" / "share" / "TreeLineWarning" / "prediction_history.db"


def create_database(db_path: str | Path | None = None) -> Database:
    """Create a database only when an application explicitly requests one."""
    return Database(db_path if db_path is not None else default_database_path())


PredictionDatabase = Database
