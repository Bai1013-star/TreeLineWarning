"""On-demand llama-server lifecycle for optional local tree AI-QC."""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.ai_qc.contracts import AiQcResult, parse_ai_qc_result

CLASSIFICATION_PROMPT = """You are a conservative quality-control assistant for LiDAR
single-tree segmentation. The supplied image is a six-panel deterministic point-cloud
rendering. Panels are TOP_XY, FRONT_XZ, SIDE_YZ, OBLIQUE_A, OBLIQUE_B, and CONTEXT.
Gold highlighted points are one algorithmically segmented tree. Muted blue points in
CONTEXT are nearby vegetation or neighboring trees. Inspect every panel. Identify only
obvious segmentation-quality problems. A false PASS is more harmful than requesting review.
When evidence is weak, return UNCERTAIN. Do not speculate about species and do not modify geometry.
Visible warning signs include multiple separated crown centers, multiple stem-like structures,
long flat or linear sheets that do not resemble a tree, interwoven neighboring vegetation,
a missing crown, or a missing stem. PASS only when the gold points coherently resemble one
tree across every view. Choose semantic reason codes that describe the observed evidence;
do not copy field descriptions or placeholder text. Return only the required JSON object.
Use issue NONE only with PASS. Use a non-NONE issue with REVIEW_REQUIRED or UNCERTAIN.
The tree_id for this case is: {tree_id}
"""


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    model_load_seconds: float
    inference_seconds: float
    peak_rss_bytes: int
    raw_text: str


def _free_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError("AI_QC_RUNTIME_RESPONSE_NOT_OBJECT")
    return parsed


class LlamaServerSession:
    """Start for one bounded batch, then terminate to release model memory."""

    def __init__(
        self,
        *,
        server_exe: Path | str,
        model_path: Path | str,
        mmproj_path: Path | str,
        timeout_seconds: float = 180.0,
        threads: int | None = None,
    ) -> None:
        self.server_exe = Path(server_exe)
        self.model_path = Path(model_path)
        self.mmproj_path = Path(mmproj_path)
        self.timeout_seconds = timeout_seconds
        self.threads = threads or max(1, min(12, (os.cpu_count() or 2) // 2))
        self.port = _free_port()
        self.process: subprocess.Popen[bytes] | None = None
        self.model_load_seconds = 0.0

    def __enter__(self) -> LlamaServerSession:
        for path, code in (
            (self.server_exe, "AI_QC_RUNTIME_MISSING"),
            (self.model_path, "AI_QC_MODEL_MISSING"),
            (self.mmproj_path, "AI_QC_MMPROJ_MISSING"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{code}:{path}")
        command = [
            str(self.server_exe),
            "--model",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            "8192",
            "--threads",
            str(self.threads),
            "--n-gpu-layers",
            "0",
            "--parallel",
            "1",
            "--no-webui",
        ]
        started = time.perf_counter()
        self.process = subprocess.Popen(
            command,
            cwd=self.server_exe.parent,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = started + self.timeout_seconds
        health = f"http://127.0.0.1:{self.port}/health"
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"AI_QC_RUNTIME_LAUNCH_FAILED:{self.process.returncode}")
            try:
                with urllib.request.urlopen(health, timeout=2.0) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if payload.get("status") == "ok":
                    self.model_load_seconds = time.perf_counter() - started
                    return self
            except (OSError, ValueError, urllib.error.URLError):
                time.sleep(0.25)
        raise TimeoutError("AI_QC_MODEL_LOAD_TIMEOUT")

    def classify(
        self, *, tree_id: str, contact_sheet: Path | str
    ) -> tuple[AiQcResult, RuntimeMetrics]:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("AI_QC_RUNTIME_NOT_AVAILABLE")
        image_bytes = Path(contact_sheet).read_bytes()
        image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": "tree-qc-local",
            "temperature": 0.0,
            "seed": 1,
            "max_tokens": 180,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "tree_ai_qc_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "tree_id": {"type": "string", "const": tree_id},
                            "status": {
                                "type": "string",
                                "enum": ["PASS", "REVIEW_REQUIRED", "UNCERTAIN"],
                            },
                            "issue": {
                                "type": "string",
                                "enum": [
                                    "NONE",
                                    "MERGED_MULTI_TREE",
                                    "OVER_SEGMENTED",
                                    "NEIGHBOR_CONTAMINATION",
                                    "CROWN_INCOMPLETE",
                                    "STEM_MISSING",
                                    "NOISE_CONTAMINATION",
                                    "OTHER",
                                ],
                            },
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reason_codes": {
                                "type": "array",
                                "maxItems": 12,
                                "items": {"type": "string", "pattern": "^[A-Z0-9_]{1,64}$"},
                            },
                        },
                        "required": [
                            "tree_id",
                            "status",
                            "issue",
                            "confidence",
                            "reason_codes",
                        ],
                    },
                },
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CLASSIFICATION_PROMPT.format(tree_id=tree_id)},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
        started = time.perf_counter()
        response = _post_json(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            payload,
            self.timeout_seconds,
        )
        elapsed = time.perf_counter() - started
        raw = str(response["choices"][0]["message"]["content"])
        result = parse_ai_qc_result(raw, expected_tree_id=tree_id)
        if result.status == "PASS" and result.confidence < 0.65:
            result = AiQcResult(
                tree_id=tree_id,
                status="UNCERTAIN",
                issue="OTHER",
                confidence=result.confidence,
                reason_codes=("LOW_CONFIDENCE_PASS",),
            )
        peak_rss = 0
        if self.process is not None:
            try:
                import psutil

                parent = psutil.Process(self.process.pid)
                peak_rss = parent.memory_info().rss + sum(
                    child.memory_info().rss for child in parent.children(recursive=True)
                )
            except (ImportError, OSError):
                pass
        return result, RuntimeMetrics(self.model_load_seconds, elapsed, peak_rss, raw)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
