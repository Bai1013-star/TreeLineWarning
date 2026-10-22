"""Offline BioCLIP image recognition for the bundled Beijing tree catalog.

The bundled scores are relative matches inside the supplied catalog.  They are
not calibrated probabilities and must not be used as the sole field
identification evidence.
"""

from __future__ import annotations

import importlib.util
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from algorithms.prediction.beijing_species import BEIJING_TREE_PROFILES, species_profile

MAX_MODEL_IMAGES = 5
IMAGE_SIZE = 224
MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
ASSET_FILENAMES = (
    "bioclip_image_encoder.onnx",
    "beijing_text_features.npy",
    "beijing_species.json",
)


def model_directory() -> Path:
    configured = os.environ.get("TREELINE_WARNING_BIOCLIP_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data" / "bioclip"


def status() -> dict[str, Any]:
    directory = model_directory()
    missing = [name for name in ASSET_FILENAMES if not (directory / name).is_file()]
    runtime_available = importlib.util.find_spec("onnxruntime") is not None
    catalog_size = 0
    labels_path = directory / "beijing_species.json"
    if labels_path.is_file():
        try:
            catalog_size = len(json.loads(labels_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            missing.append(labels_path.name + "（无法读取）")
    return {
        "available": not missing and runtime_available and catalog_size > 0,
        "engine": "BioCLIP ViT-B/16 ONNX",
        "mode": "offline_zero_shot_beijing_catalog",
        "catalog_size": catalog_size,
        "runtime_available": runtime_available,
        "missing_assets": sorted(set(missing)),
    }


def _prepare_image(content: bytes, filename: str) -> np.ndarray:
    try:
        with Image.open(BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = ImageOps.fit(
                image,
                (IMAGE_SIZE, IMAGE_SIZE),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )
            pixels = np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"无法读取图片：{filename}") from exc
    pixels = (pixels - MEAN) / STD
    return np.transpose(pixels, (2, 0, 1)).astype(np.float32, copy=False)


def _select_images(images: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    if len(images) <= MAX_MODEL_IMAGES:
        return images
    indices = np.linspace(0, len(images) - 1, MAX_MODEL_IMAGES, dtype=int)
    return [images[int(index)] for index in indices]


def _softmax(values: np.ndarray, temperature: float = 0.025) -> np.ndarray:
    scaled = values.astype(np.float64) / temperature
    scaled -= scaled.max()
    exponential = np.exp(scaled)
    return (exponential / exponential.sum()).astype(np.float32)


class OfflineBioClip:
    def __init__(self, directory: Path | None = None) -> None:
        directory = directory or model_directory()
        availability = status()
        if directory != model_directory():
            missing = [name for name in ASSET_FILENAMES if not (directory / name).is_file()]
            availability = {**availability, "available": not missing, "missing_assets": missing}
        if not availability["available"]:
            details = "、".join(availability["missing_assets"])
            if not availability["runtime_available"]:
                details = (details + "、" if details else "") + "onnxruntime"
            raise ValueError(f"离线树种识别组件不完整：{details}")

        try:
            import onnxruntime as ort
        except (ImportError, OSError) as exc:
            raise ValueError("离线树种识别运行库无法加载，请重新安装软件运行组件") from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self.session = ort.InferenceSession(
            str(directory / "bioclip_image_encoder.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.text_features = np.load(directory / "beijing_text_features.npy").astype(
            np.float32, copy=False
        )
        self.labels: list[dict[str, str]] = json.loads(
            (directory / "beijing_species.json").read_text(encoding="utf-8")
        )
        if self.text_features.shape != (len(self.labels), 512):
            raise ValueError("离线树种特征与北京树种目录不匹配，请重新安装模型文件")

    def identify(self, images: list[tuple[str, bytes]]) -> dict[str, Any]:
        if not images:
            raise ValueError("离线 BioCLIP 识别需要至少一张照片或视频代表画面")
        selected = _select_images(images)
        batch = np.stack([_prepare_image(content, name) for name, content in selected])
        image_features = self.session.run(None, {"images": batch})[0].astype(np.float32)
        combined = image_features.mean(axis=0)
        norm = float(np.linalg.norm(combined))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("离线模型没有生成有效图像特征，请更换清晰照片")
        combined /= norm

        similarities = self.text_features @ combined
        probabilities = _softmax(similarities)
        order = np.argsort(probabilities)[::-1][:5]
        candidates = []
        for index in order:
            label = self.labels[int(index)]
            name = label["chinese_name"]
            profile = species_profile(name)
            candidates.append(
                {
                    "species": name,
                    "scientific_name": label.get("scientific_name")
                    or profile["scientific_name"],
                    "identification_features": (
                        profile["identification_features"]
                        if name in BEIJING_TREE_PROFILES
                        else "请结合叶、树皮、芽、花果及现场档案进行人工复核"
                    ),
                    "confidence": round(float(probabilities[int(index)]) * 100, 1),
                    "sample_count": 0,
                    "beijing_category": label.get("category", "北京参考树种"),
                }
            )
        return {
            "candidates": candidates,
            "top_species": candidates[0]["species"],
            "confidence": candidates[0]["confidence"],
            "engine": "BioCLIP ViT-B/16 ONNX",
            "recognition_mode": "offline_zero_shot_beijing_catalog",
            "catalog_size": len(self.labels),
            "images_received": len(images),
            "images_used": len(selected),
            "score_type": "relative_catalog_match_unvalidated",
            "field_validated": False,
            "review_required": True,
        }


_ENGINE: OfflineBioClip | None = None


def predict(images: list[tuple[str, bytes]]) -> dict[str, Any]:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = OfflineBioClip()
    return _ENGINE.identify(images)
