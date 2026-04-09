from functools import lru_cache
from pathlib import Path
import json

import cv2
import numpy as np

from core.face_dataset import (
    count_images_per_employee,
    load_training_data,
    preprocess_face_image,
)
from db.database import get_employee_codes

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "lbph_model.yml"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
IMAGE_SIZE = (200, 200)
MIN_MATCH_THRESHOLD = 45.0
SINGLE_PERSON_MIN_THRESHOLD = 60.0
MAX_MATCH_THRESHOLD = 75.0
THRESHOLD_MARGIN = 12.0
DEFAULT_CAMERA_PROFILES = {
    "0": {
        "name": "Laptop Camera",
        "threshold_offset": 0.0,
        "min_threshold": 50.0,
    },
    "1": {
        "name": "Rappo C200",
        "threshold_offset": 10.0,
        "min_threshold": 55.0,
    },
}


def ensure_model_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def create_recognizer():
    return cv2.face.LBPHFaceRecognizer_create()


def clear_model_cache():
    _load_model_cached.cache_clear()


def _normalize_camera_profiles(camera_profiles):
    normalized_profiles = {}

    for camera_key, profile in DEFAULT_CAMERA_PROFILES.items():
        source_profile = dict(profile)
        if camera_profiles and camera_key in camera_profiles:
            source_profile.update(camera_profiles[camera_key])

        normalized_profiles[str(camera_key)] = {
            "name": str(source_profile.get("name", DEFAULT_CAMERA_PROFILES[camera_key]["name"])),
            "threshold_offset": float(source_profile.get("threshold_offset", 0.0)),
            "min_threshold": float(source_profile.get("min_threshold", MIN_MATCH_THRESHOLD)),
        }

    return normalized_profiles


@lru_cache(maxsize=1)
def _load_model_cached(model_mtime_ns, label_map_mtime_ns, metadata_mtime_ns):
    recognizer = create_recognizer()
    recognizer.read(str(MODEL_PATH))

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as file:
        label_to_code_raw = json.load(file)

    metadata = {
        "default_threshold": MAX_MATCH_THRESHOLD,
        "code_thresholds": {},
        "images_per_employee": {},
        "camera_profiles": DEFAULT_CAMERA_PROFILES,
    }

    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as file:
            metadata.update(json.load(file))

    label_to_code = {int(key): value for key, value in label_to_code_raw.items()}
    metadata["default_threshold"] = float(
        metadata.get("default_threshold", MAX_MATCH_THRESHOLD)
    )
    metadata["code_thresholds"] = {
        str(key): float(value)
        for key, value in metadata.get("code_thresholds", {}).items()
    }
    metadata["camera_profiles"] = _normalize_camera_profiles(
        metadata.get("camera_profiles")
    )

    return recognizer, label_to_code, metadata


def _build_model_metadata(recognizer, images, labels, label_to_code, images_per_employee):
    distance_by_code = {}
    min_threshold = (
        SINGLE_PERSON_MIN_THRESHOLD
        if len(label_to_code) == 1
        else MIN_MATCH_THRESHOLD
    )

    for image, label in zip(images, labels):
        predicted_label, confidence = recognizer.predict(image)

        if predicted_label != int(label):
            continue

        employee_code = label_to_code[int(label)]
        distance_by_code.setdefault(employee_code, []).append(float(confidence))

    code_thresholds = {}

    for employee_code in label_to_code.values():
        distances = distance_by_code.get(employee_code, [])

        if not distances:
            code_thresholds[employee_code] = MAX_MATCH_THRESHOLD
            continue

        max_distance = max(distances)
        mean_distance = float(np.mean(distances))
        std_distance = float(np.std(distances))
        calibrated_threshold = max(
            max_distance + THRESHOLD_MARGIN,
            mean_distance + (2 * std_distance) + THRESHOLD_MARGIN,
        )
        code_thresholds[employee_code] = min(
            MAX_MATCH_THRESHOLD,
            max(min_threshold, calibrated_threshold),
        )

    return {
        "default_threshold": MAX_MATCH_THRESHOLD,
        "code_thresholds": code_thresholds,
        "images_per_employee": images_per_employee,
        "camera_profiles": _normalize_camera_profiles(None),
    }


def train_model():
    valid_employee_codes = get_employee_codes()
    images, labels, label_to_code, code_to_label = load_training_data(
        IMAGE_SIZE,
        valid_employee_codes=valid_employee_codes,
    )

    if len(valid_employee_codes) == 0:
        raise ValueError("No employees found in database.")

    if len(images) == 0:
        raise ValueError("No training images found for valid employees in dataset.")

    recognizer = create_recognizer()
    recognizer.train(images, labels)

    ensure_model_dir()
    recognizer.save(str(MODEL_PATH))

    images_per_employee = count_images_per_employee(valid_employee_codes)
    metadata = _build_model_metadata(
        recognizer,
        images,
        labels,
        label_to_code,
        images_per_employee,
    )

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as file:
        json.dump(label_to_code, file, ensure_ascii=False, indent=2)

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    clear_model_cache()

    return {
        "num_images": len(images),
        "num_people": len(label_to_code),
        "label_to_code": label_to_code,
        "images_per_employee": images_per_employee,
        "code_thresholds": metadata["code_thresholds"],
        "camera_profiles": metadata["camera_profiles"],
    }


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Please train model first.")

    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError("Label map file not found. Please train model first.")

    model_mtime_ns = MODEL_PATH.stat().st_mtime_ns
    label_map_mtime_ns = LABEL_MAP_PATH.stat().st_mtime_ns
    metadata_mtime_ns = (
        MODEL_METADATA_PATH.stat().st_mtime_ns if MODEL_METADATA_PATH.exists() else 0
    )
    return _load_model_cached(model_mtime_ns, label_map_mtime_ns, metadata_mtime_ns)


def _apply_camera_profile(match_threshold, metadata, camera_index):
    if camera_index is None:
        return match_threshold, None

    camera_profiles = metadata.get("camera_profiles", {})
    camera_profile = camera_profiles.get(str(camera_index))
    if camera_profile is None:
        return match_threshold, None

    adjusted_threshold = max(
        match_threshold + float(camera_profile.get("threshold_offset", 0.0)),
        float(camera_profile.get("min_threshold", MIN_MATCH_THRESHOLD)),
    )
    adjusted_threshold = min(MAX_MATCH_THRESHOLD, adjusted_threshold)
    return adjusted_threshold, camera_profile


def predict_face(face_image, camera_index=None):
    recognizer, label_to_code, metadata = load_model()

    processed_face = preprocess_face_image(face_image, IMAGE_SIZE)
    predicted_label, confidence = recognizer.predict(processed_face)
    predicted_code = label_to_code.get(predicted_label, "Unknown")
    confidence = float(confidence)
    default_threshold = float(metadata.get("default_threshold", MAX_MATCH_THRESHOLD))
    code_thresholds = metadata.get("code_thresholds", {})
    base_match_threshold = float(code_thresholds.get(predicted_code, default_threshold))
    match_threshold, camera_profile = _apply_camera_profile(
        base_match_threshold,
        metadata,
        camera_index,
    )
    is_match = predicted_code != "Unknown" and confidence <= match_threshold

    return {
        "predicted_code": predicted_code,
        "display_code": predicted_code if is_match else "Unknown",
        "confidence": confidence,
        "match_threshold": match_threshold,
        "base_match_threshold": base_match_threshold,
        "is_match": is_match,
        "camera_profile": camera_profile,
    }

