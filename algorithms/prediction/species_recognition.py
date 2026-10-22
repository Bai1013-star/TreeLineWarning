"""Local, labelled-sample tree species recognition support.

This module deliberately produces an auxiliary candidate ranking only.  It never
pretends that a colour or point-cloud shape alone is a botanical determination.
Users first save labelled field samples, then train an offline nearest-centroid
model from their own photos and LAS/LAZ measurements.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import laspy
import numpy as np
from PIL import Image, UnidentifiedImageError

from algorithms.prediction.beijing_species import (
    BEIJING_TREE_SPECIES,
    CATALOG_SOURCES,
    species_profile,
)
from algorithms.prediction.offline_bioclip import predict as predict_offline_bioclip
from algorithms.prediction.offline_bioclip import status as offline_bioclip_status

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_POINT_CLOUD_BYTES = 80 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_POINT_CLOUD_SUFFIXES = {".las", ".laz"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_SPECIES = BEIJING_TREE_SPECIES


def _paths(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    return root / "samples.json", root / "model.json"


def _read_json(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def image_feature(content: bytes, filename: str) -> list[float]:
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("单张照片不能超过 10 MB")
    if Path(filename).suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError("照片仅支持 JPG、PNG、WEBP 或 BMP")
    try:
        with Image.open(BytesIO(content)) as image:
            rgb = image.convert("RGB").resize((160, 160))
            values = np.asarray(rgb, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("无法读取照片文件") from exc
    mean = values.mean(axis=(0, 1))
    deviation = values.std(axis=(0, 1))
    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 1e-6,
    ).mean()
    green_dominance = (values[:, :, 1] - (values[:, :, 0] + values[:, :, 2]) / 2).mean()
    return [*mean.tolist(), *deviation.tolist(), float(saturation), float(green_dominance)]


def point_cloud_feature(content: bytes, filename: str) -> list[float]:
    if len(content) > MAX_POINT_CLOUD_BYTES:
        raise ValueError("点云文件不能超过 80 MB")
    if Path(filename).suffix.lower() not in ALLOWED_POINT_CLOUD_SUFFIXES:
        raise ValueError("点云仅支持 Share2 导出的 LAS 或 LAZ 文件")
    try:
        cloud = laspy.read(BytesIO(content))
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError("无法读取 LAS/LAZ 点云；请确认文件完整且为单株点云") from exc
    if len(cloud.x) < 100:
        raise ValueError("点云点数过少，至少需要 100 个点")
    x, y, z = (np.asarray(axis, dtype=np.float64) for axis in (cloud.x, cloud.y, cloud.z))
    extent_x = max(float(np.ptp(x)), 1e-6)
    extent_y = max(float(np.ptp(y)), 1e-6)
    height = max(float(np.ptp(z)), 1e-6)
    crown_width = max(extent_x, extent_y)
    z_normalized = (z - z.min()) / height
    return [
        height,
        crown_width,
        min(extent_x, extent_y) / crown_width,
        float(np.std(z_normalized)),
        float(np.quantile(z_normalized, 0.25)),
        float(np.quantile(z_normalized, 0.75)),
        float(len(z) / max(extent_x * extent_y * height, 1e-6)),
    ]


def _mean_feature(features: list[list[float]]) -> list[float] | None:
    if not features:
        return None
    return np.mean(np.asarray(features, dtype=np.float64), axis=0).round(8).tolist()


def _feature_scale(features: list[list[float]]) -> list[float] | None:
    if not features:
        return None
    values = np.asarray(features, dtype=np.float64)
    scale = np.std(values, axis=0)
    # Avoid a zero-variance feature dominating after a tiny numerical change.
    scale = np.where(scale < 1e-6, 1.0, scale)
    return scale.round(8).tolist()


def _normalized_distance(
    feature: list[float], centroid: list[float], scale: list[float] | None = None
) -> float:
    current = np.asarray(feature, dtype=np.float64)
    target = np.asarray(centroid, dtype=np.float64)
    if current.shape != target.shape:
        raise ValueError("识别模型特征维度与输入不一致，请重新训练模型")
    divisor = np.ones_like(current) if scale is None else np.asarray(scale, dtype=np.float64)
    if divisor.shape != current.shape or not np.all(np.isfinite(divisor)):
        raise ValueError("识别模型特征缩放参数无效，请重新训练模型")
    divisor = np.where(divisor < 1e-6, 1.0, divisor)
    return float(
        np.linalg.norm((current - target) / divisor) / max(np.sqrt(len(current)), 1.0)
    )


def status(root: Path) -> dict:
    samples_path, model_path = _paths(root)
    samples = _read_json(samples_path, {"samples": []}).get("samples", [])
    counts = {species: 0 for species in SUPPORTED_SPECIES}
    for sample in samples:
        if sample.get("species") in counts:
            counts[sample["species"]] += 1
    model = _read_json(model_path, {})
    offline = offline_bioclip_status()
    return {
        "supported_species": list(SUPPORTED_SPECIES),
        "supported_species_count": len(SUPPORTED_SPECIES) - 1,
        "catalog_sources": list(CATALOG_SOURCES),
        "sample_counts": counts,
        "sample_count": len(samples),
        "trained": bool(model.get("species")),
        "trained_species": sorted(model.get("species", {}).keys()),
        "model_status": "experimental",
        "field_validated": False,
        "score_type": "heuristic_similarity_unvalidated",
        "offline_recognition": offline,
        "samples": [
            {
                "id": sample.get("id"),
                "species": sample.get("species"),
                "photo_count": sample.get("photo_count", 0),
                "video_count": sample.get("video_count", 0),
                "video_frame_count": sample.get("video_frame_count", 0),
                "has_video": sample.get("has_video", False),
                "has_point_cloud": sample.get("has_point_cloud", False),
            }
            for sample in samples
        ],
        "message": (
            "照片和视频可直接使用内置离线 BioCLIP 北京树种库；点云或现场样本校准仍需先保存"
            "已知树种样本再训练。所有相似度均未经现场独立验证，结果必须人工复核。"
            if offline["available"]
            else "离线 BioCLIP 组件不可用；请先保存已知树种样本再训练。所有结果必须人工复核。"
        ),
    }


def save_sample(
    root: Path,
    species: str,
    photos: list[tuple[str, bytes]],
    video_count: int,
    video_frame_count: int,
    point_cloud: tuple[str, bytes] | None,
) -> dict:
    if species not in SUPPORTED_SPECIES:
        raise ValueError("请选择支持的已知树种标签")
    photo_features = [image_feature(content, name) for name, content in photos]
    cloud_feature = point_cloud_feature(point_cloud[1], point_cloud[0]) if point_cloud else None
    if not photo_features and cloud_feature is None:
        raise ValueError("至少选择照片、视频或一个 LAS/LAZ 点云")
    samples_path, model_path = _paths(root)
    data = _read_json(samples_path, {"samples": []})
    record = {
        "id": uuid4().hex,
        "species": species,
        "photo_feature": _mean_feature(photo_features),
        "point_cloud_feature": cloud_feature,
        "photo_count": max(0, len(photo_features) - video_frame_count),
        "video_count": video_count,
        "video_frame_count": video_frame_count,
        "has_video": video_count > 0,
        "has_point_cloud": cloud_feature is not None,
    }
    data.setdefault("samples", []).append(record)
    _write_json(samples_path, data)
    # A centroid model is a snapshot of the labelled sample set.  Any sample
    # change invalidates it; continuing to serve the old file would silently
    # ignore the newly added observation.
    model_path.unlink(missing_ok=True)
    return {"record": record, "status": status(root)}


def train(root: Path) -> dict:
    samples_path, model_path = _paths(root)
    samples = _read_json(samples_path, {"samples": []}).get("samples", [])
    model_species: dict[str, dict] = {}
    all_photo_features = [
        sample["photo_feature"] for sample in samples if sample.get("photo_feature")
    ]
    all_cloud_features = [
        sample["point_cloud_feature"]
        for sample in samples
        if sample.get("point_cloud_feature")
    ]
    feature_scales = {
        "photo": _feature_scale(all_photo_features),
        "point_cloud": _feature_scale(all_cloud_features),
    }
    for species in SUPPORTED_SPECIES:
        species_samples = [sample for sample in samples if sample.get("species") == species]
        photo = [
            sample["photo_feature"] for sample in species_samples if sample.get("photo_feature")
        ]
        cloud = [
            sample["point_cloud_feature"]
            for sample in species_samples
            if sample.get("point_cloud_feature")
        ]
        if photo or cloud:
            model_species[species] = {
                "sample_count": len(species_samples),
                "photo_centroid": _mean_feature(photo),
                "point_cloud_centroid": _mean_feature(cloud),
            }
    if len(model_species) < 2:
        raise ValueError("至少要有两个树种的已标注样本，才能训练候选分类器")
    payload = {
        "method": "local_nearest_centroid_auxiliary",
        "model_status": "experimental",
        "field_validated": False,
        "score_type": "heuristic_similarity_unvalidated",
        "feature_scales": feature_scales,
        "species": model_species,
    }
    _write_json(model_path, payload)
    return {"status": status(root), "method": payload["method"]}


def delete_sample(root: Path, sample_id: str) -> dict:
    samples_path, model_path = _paths(root)
    data = _read_json(samples_path, {"samples": []})
    samples = data.get("samples", [])
    remaining = [sample for sample in samples if sample.get("id") != sample_id]
    if len(remaining) == len(samples):
        raise ValueError("未找到该树种训练样本")
    data["samples"] = remaining
    _write_json(samples_path, data)
    if model_path.exists():
        model_path.unlink()
    return status(root)


def predict(
    root: Path,
    photos: list[tuple[str, bytes]],
    video_count: int,
    video_frame_count: int,
    point_cloud: tuple[str, bytes] | None,
) -> dict:
    photo_features = [image_feature(content, name) for name, content in photos]
    cloud_feature = point_cloud_feature(point_cloud[1], point_cloud[0]) if point_cloud else None
    photo_feature = _mean_feature(photo_features)
    if photo_feature is None and cloud_feature is None:
        raise ValueError("至少选择照片、视频或一个 LAS/LAZ 点云")
    _, model_path = _paths(root)
    model = _read_json(model_path, {})
    species_model = model.get("species", {})
    if not species_model:
        if photos:
            result = predict_offline_bioclip(photos)
            result["input_summary"] = {
                "photo_count": max(0, len(photos) - video_frame_count),
                "video_count": video_count,
                "video_frame_count": video_frame_count,
                "has_point_cloud": cloud_feature is not None,
            }
            result["message"] = (
                "结果由内置离线 BioCLIP 模型在北京参考树种目录中进行相对匹配；百分比不是"
                "现场准确率。请用叶、树皮、芽、花果和现场档案人工复核。"
            )
            return result
        raise ValueError(
            "仅使用点云时需要本地样本模型；请先保存至少两个已知树种的点云样本并训练"
        )
    candidates = []
    feature_scales = model.get("feature_scales", {})
    for species, entry in species_model.items():
        distances = []
        if photo_feature is not None and entry.get("photo_centroid"):
            distances.append(
                _normalized_distance(
                    photo_feature,
                    entry["photo_centroid"],
                    feature_scales.get("photo"),
                )
            )
        if cloud_feature is not None and entry.get("point_cloud_centroid"):
            distances.append(
                _normalized_distance(
                    cloud_feature,
                    entry["point_cloud_centroid"],
                    feature_scales.get("point_cloud"),
                )
            )
        if distances:
            candidates.append((species, float(np.mean(distances)), entry["sample_count"]))
    if not candidates:
        raise ValueError("当前模型没有与所选文件类型相匹配的训练样本")
    candidates.sort(key=lambda item: item[1])
    best_distance = candidates[0][1]
    second_distance = candidates[1][1] if len(candidates) > 1 else None
    margin = (
        max(0.0, 1.0 - best_distance / max(second_distance, 1e-9))
        if second_distance is not None
        else 0.0
    )
    ranked = [
        {
            "species": species,
            "scientific_name": species_profile(species)["scientific_name"],
            "identification_features": species_profile(species)["identification_features"],
            "distance": round(distance, 4),
            "confidence": float(
                round(
                    max(
                        0.0,
                        min(100.0, 70 * np.exp(-distance) + (30 * margin if rank == 0 else 0)),
                    ),
                    1,
                )
            ),
            "sample_count": count,
        }
        for rank, (species, distance, count) in enumerate(candidates[:3])
    ]
    top = ranked[0]
    return {
        "candidates": ranked,
        "top_species": top["species"],
        "confidence": top["confidence"],
        "score_type": "heuristic_similarity_unvalidated",
        "field_validated": False,
        "input_summary": {
            "photo_count": max(0, len(photo_features) - video_frame_count),
            "video_count": video_count,
            "video_frame_count": video_frame_count,
            "has_point_cloud": cloud_feature is not None,
        },
        "review_required": bool(top["confidence"] < 80 or top["sample_count"] < 3),
        "message": (
            "结果由所选照片、视频代表画面和/或点云与本地样本比对得到；"
            "百分比是未校准的辅助相似度，不是现场准确率，请用叶、树皮、花果和现场档案复核。"
        ),
    }
