"""On-demand llama-server lifecycle for optional local tree AI-QC."""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.ai_qc.contracts import AiQcResult, parse_ai_qc_result_v2

CLASSIFICATION_PROMPT = """You are a conservative segmentation quality-control assistant.
These are deterministic LiDAR renderings of one algorithmically segmented candidate tree.
Gold points belong to the current candidate tree. Muted blue points, where shown, are
surrounding vegetation or neighboring trees. The HEIGHT_CODED view colors candidate points
from blue at low height through yellow to red at high height. Inspect every supplied view,
including orthographic, oblique, context, and lower-stem views. Judge segmentation quality,
not tree species. A false PASS is more harmful than an unnecessary REVIEW_REQUIRED. If the
visual evidence is weak or contradictory, return UNCERTAIN. Warning signs include multiple
separated crowns or stems, non-tree linear or flat sheets, neighboring vegetation included
in gold, an incomplete crown, or a missing stem. PASS only when the highlighted points
coherently resemble one complete tree across all views. Return only the required four-field
JSON object. Use issue NONE only with PASS. Use a non-NONE issue with REVIEW_REQUIRED or
UNCERTAIN. Reason codes must be short uppercase semantic observations, never prose.
"""


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    model_load_seconds: float
    inference_seconds: float
    peak_rss_bytes: int
    peak_vram_bytes: int
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
        gpu_layers: int = 0,
        ctx_size: int = 8_192,
    ) -> None:
        self.server_exe = Path(server_exe)
        self.model_path = Path(model_path)
        self.mmproj_path = Path(mmproj_path)
        self.timeout_seconds = timeout_seconds
        self.threads = threads or max(1, min(12, (os.cpu_count() or 2) // 2))
        self.gpu_layers = gpu_layers
        self.ctx_size = ctx_size
        self.port = _free_port()
        self.process: subprocess.Popen[bytes] | None = None
        self.model_load_seconds = 0.0
        self.peak_rss_bytes = 0
        self.peak_vram_bytes = 0
        self._gpu_baseline_bytes = 0
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None

    def _sample_resources(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        while not self._monitor_stop.wait(0.2):
            process = self.process
            if process is None or process.poll() is not None:
                return
            try:
                parent = psutil.Process(process.pid)
                processes = [parent, *parent.children(recursive=True)]
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes,
                    sum(item.memory_info().rss for item in processes if item.is_running()),
                )
                pids = {item.pid for item in processes}
            except (OSError, psutil.Error):
                continue
            if self.gpu_layers <= 0:
                continue
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                used_mb = 0
                for line in completed.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) in pids:
                        used_mb += int(fields[1])
                if used_mb:
                    sampled_vram = used_mb * 1024 * 1024
                else:
                    sampled_vram = max(0, self._total_gpu_used_bytes() - self._gpu_baseline_bytes)
                self.peak_vram_bytes = max(self.peak_vram_bytes, sampled_vram)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass

    @staticmethod
    def _total_gpu_used_bytes() -> int:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return sum(int(line.strip()) for line in completed.stdout.splitlines()) * 1024 * 1024
        except (OSError, subprocess.SubprocessError, ValueError):
            return 0

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
            str(self.ctx_size),
            "--threads",
            str(self.threads),
            "--n-gpu-layers",
            str(self.gpu_layers),
            "--parallel",
            "1",
            "--no-webui",
        ]
        if self.gpu_layers > 0:
            self._gpu_baseline_bytes = self._total_gpu_used_bytes()
        started = time.perf_counter()
        self.process = subprocess.Popen(
            command,
            cwd=self.server_exe.parent,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(target=self._sample_resources, daemon=True)
        self._monitor_thread.start()
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
        self.__exit__(None, None, None)
        raise TimeoutError("AI_QC_MODEL_LOAD_TIMEOUT")

    def classify(
        self,
        *,
        tree_id: str,
        image_paths: Sequence[Path | str] | None = None,
        contact_sheet: Path | str | None = None,
    ) -> tuple[AiQcResult, RuntimeMetrics]:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("AI_QC_RUNTIME_NOT_AVAILABLE")
        if image_paths is not None and contact_sheet is not None:
            raise ValueError("AI_QC_IMAGE_INPUT_AMBIGUOUS")
        selected_paths = list(image_paths or ([] if contact_sheet is None else [contact_sheet]))
        if not selected_paths:
            raise ValueError("AI_QC_IMAGE_INPUT_MISSING")
        content: list[dict[str, Any]] = [{"type": "text", "text": CLASSIFICATION_PROMPT}]
        for image_path in selected_paths:
            image_bytes = Path(image_path).read_bytes()
            image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": image_url}})
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
                    "content": content,
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
        result = parse_ai_qc_result_v2(raw, expected_tree_id=tree_id)
        if result.status == "PASS" and result.confidence < 0.65:
            result = AiQcResult(
                tree_id=tree_id,
                status="UNCERTAIN",
                issue="OTHER",
                confidence=result.confidence,
                reason_codes=("LOW_CONFIDENCE_PASS",),
            )
        return result, RuntimeMetrics(
            self.model_load_seconds,
            elapsed,
            self.peak_rss_bytes,
            self.peak_vram_bytes,
            raw,
        )

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            self._monitor_stop.set()
            if self._monitor_thread is not None:
                self._monitor_thread.join(timeout=2)
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        finally:
            self._monitor_stop.set()
            if self._monitor_thread is not None:
                self._monitor_thread.join(timeout=2)
