from io import BytesIO

import laspy
import numpy as np
from algorithms.prediction.species import annual_height_growth, get_species_factor
from api.prediction_api import create_app
from fastapi.testclient import TestClient
from PIL import Image


def _image_bytes(rgb: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (40, 40), rgb)
    stream = BytesIO()
    image.save(stream, format="JPEG")
    return stream.getvalue()


def _point_cloud_bytes(tmp_path, name: str, height: float, width: float) -> bytes:
    header = laspy.LasHeader(point_format=3, version="1.2")
    cloud = laspy.LasData(header)
    count = 200
    cloud.x = np.linspace(0, width, count)
    cloud.y = np.mod(np.arange(count), 17) / 17 * width
    cloud.z = np.linspace(0, height, count)
    path = tmp_path / name
    cloud.write(path)
    return path.read_bytes()


def test_local_photo_samples_train_and_rank_candidates(tmp_path):
    client = TestClient(create_app(training_output_dir=tmp_path))
    page = client.get("/").text
    assert 'id="recognitionPhotos"' in page
    assert 'id="recognitionVideos"' in page
    assert 'id="recognitionPointCloud"' in page
    assert 'id="inputRecognitionVideos"' in page
    assert "照片和视频可直接使用内置离线 BioCLIP" in page
    assert "/species-recognition/predict" in page
    assert client.get("/species-recognition/status").json()["sample_count"] == 0
    assert client.get("/species-recognition/status").json()["supported_species_count"] >= 100

    first = client.post(
        "/species-recognition/samples",
        data={"species": "杨树"},
        files=[("photos", ("poplar.jpg", _image_bytes((30, 160, 40)), "image/jpeg"))],
    )
    assert first.status_code == 200
    second = client.post(
        "/species-recognition/samples",
        data={"species": "榆树"},
        files=[("photos", ("elm.jpg", _image_bytes((170, 80, 30)), "image/jpeg"))],
    )
    assert second.status_code == 200

    trained = client.post("/species-recognition/train")
    assert trained.status_code == 200
    assert trained.json()["status"]["trained"] is True
    assert trained.json()["status"]["field_validated"] is False

    prediction = client.post(
        "/species-recognition/predict",
        files=[("photos", ("unknown.jpg", _image_bytes((32, 158, 42)), "image/jpeg"))],
    )
    assert prediction.status_code == 200
    assert prediction.json()["top_species"] == "杨树"
    assert prediction.json()["candidates"][0]["scientific_name"]
    assert prediction.json()["candidates"][0]["identification_features"]
    assert prediction.json()["score_type"] == "heuristic_similarity_unvalidated"

    sample_id = first.json()["record"]["id"]
    deleted = client.delete(f"/species-recognition/samples/{sample_id}")
    assert deleted.status_code == 200
    assert deleted.json()["trained"] is False


def test_adding_a_sample_invalidates_the_previous_centroid_model(tmp_path):
    client = TestClient(create_app(training_output_dir=tmp_path))
    for species, color in (("杨树", (30, 160, 40)), ("榆树", (170, 80, 30))):
        response = client.post(
            "/species-recognition/samples",
            data={"species": species},
            files=[("photos", (f"{species}.jpg", _image_bytes(color), "image/jpeg"))],
        )
        assert response.status_code == 200
    assert client.post("/species-recognition/train").json()["status"]["trained"] is True

    saved = client.post(
        "/species-recognition/samples",
        data={"species": "国槐"},
        files=[("photos", ("locust.jpg", _image_bytes((60, 130, 55)), "image/jpeg"))],
    )

    assert saved.status_code == 200
    assert saved.json()["status"]["trained"] is False
    prediction = client.post(
        "/species-recognition/predict",
        files=[("photos", ("unknown.jpg", _image_bytes((60, 130, 55)), "image/jpeg"))],
    )
    assert prediction.status_code == 200
    assert prediction.json()["recognition_mode"] == "offline_zero_shot_beijing_catalog"
    assert prediction.json()["engine"] == "BioCLIP ViT-B/16 ONNX"
    assert prediction.json()["review_required"] is True


def test_bundled_bioclip_is_reported_as_available_without_local_training(tmp_path):
    client = TestClient(create_app(training_output_dir=tmp_path))

    status = client.get("/species-recognition/status")

    assert status.status_code == 200
    offline = status.json()["offline_recognition"]
    assert offline["available"] is True
    assert offline["catalog_size"] > 0
    assert offline["runtime_available"] is True


def test_video_frame_metadata_is_saved_and_used_with_extracted_frames(tmp_path):
    client = TestClient(create_app(training_output_dir=tmp_path))

    first = client.post(
        "/species-recognition/samples",
        data={"species": "杨树", "video_count": "1", "video_frame_count": "5"},
        files=[
            (
                "photos",
                (f"poplar-frame-{index}.jpg", _image_bytes((30, 160, 40)), "image/jpeg"),
            )
            for index in range(5)
        ],
    )
    assert first.status_code == 200
    assert first.json()["record"]["video_count"] == 1
    assert first.json()["record"]["video_frame_count"] >= 1

    second = client.post(
        "/species-recognition/samples",
        data={"species": "榆树"},
        files=[("photos", ("elm.jpg", _image_bytes((170, 80, 30)), "image/jpeg"))],
    )
    assert second.status_code == 200
    assert client.post("/species-recognition/train").status_code == 200

    prediction = client.post(
        "/species-recognition/predict",
        data={"video_count": "1", "video_frame_count": "5"},
        files=[
            (
                "photos",
                (f"unknown-frame-{index}.jpg", _image_bytes((30, 160, 40)), "image/jpeg"),
            )
            for index in range(5)
        ],
    )
    assert prediction.status_code == 200
    assert prediction.json()["top_species"] == "杨树"
    assert prediction.json()["input_summary"]["video_count"] == 1
    assert prediction.json()["input_summary"]["photo_count"] == 0


def test_point_cloud_can_be_used_without_photo_or_video(tmp_path):
    client = TestClient(create_app(training_output_dir=tmp_path / "training"))
    poplar = _point_cloud_bytes(tmp_path, "poplar.las", height=12, width=3)
    elm = _point_cloud_bytes(tmp_path, "elm.las", height=6, width=8)

    for species, filename, content in (
        ("杨树", "poplar.las", poplar),
        ("榆树", "elm.las", elm),
    ):
        saved = client.post(
            "/species-recognition/samples",
            data={"species": species},
            files={"point_cloud": (filename, content, "application/octet-stream")},
        )
        assert saved.status_code == 200
        assert saved.json()["record"]["photo_count"] == 0
        assert saved.json()["record"]["has_point_cloud"] is True

    assert client.post("/species-recognition/train").status_code == 200
    prediction = client.post(
        "/species-recognition/predict",
        files={"point_cloud": ("unknown.las", poplar, "application/octet-stream")},
    )
    assert prediction.status_code == 200
    assert prediction.json()["top_species"] == "杨树"
    assert prediction.json()["input_summary"]["has_point_cloud"] is True


def test_photo_video_and_point_cloud_can_be_combined(tmp_path):
    client = TestClient(create_app(training_output_dir=tmp_path / "training"))
    files = [
        ("photos", (f"photo-{index}.jpg", _image_bytes((40, 140, 60)), "image/jpeg"))
        for index in range(3)
    ]
    files.extend(
        (
            "photos",
            (f"video-frame-{index}.jpg", _image_bytes((45, 145, 65)), "image/jpeg"),
        )
        for index in range(5)
    )
    files.append(
        (
            "point_cloud",
            (
                "combined.las",
                _point_cloud_bytes(tmp_path, "combined.las", height=10, width=4),
                "application/octet-stream",
            ),
        )
    )

    response = client.post(
        "/species-recognition/samples",
        data={"species": "国槐", "video_count": "1", "video_frame_count": "5"},
        files=files,
    )
    assert response.status_code == 200
    record = response.json()["record"]
    assert record["photo_count"] == 3
    assert record["video_count"] == 1
    assert record["video_frame_count"] == 5
    assert record["has_point_cloud"] is True


def test_beijing_tree_catalog_is_shared_by_warning_and_recognition(tmp_path):
    client = TestClient(create_app(training_output_dir=tmp_path))
    catalog = client.get("/species/catalog")
    assert catalog.status_code == 200
    body = catalog.json()
    names = {item["name"] for item in body["species"]}
    assert body["count"] >= 100
    assert {"油松", "白皮松", "毛白杨", "国槐", "七叶树", "流苏树"} <= names
    assert len(body["sources"]) >= 4

    saved = client.post(
        "/species-recognition/samples",
        data={"species": "七叶树"},
        files=[("photos", ("aesculus.jpg", _image_bytes((50, 130, 50)), "image/jpeg"))],
    )
    assert saved.status_code == 200
    assert saved.json()["record"]["species"] == "七叶树"

    growth = annual_height_growth("七叶树")
    assert growth["source"] == "北京树种目录习性分组工程初值"
    assert 0 < float(growth["minimum"]) <= float(growth["maximum"])
    assert get_species_factor("七叶树") >= 1.0
