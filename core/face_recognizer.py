from functools import lru_cache
from pathlib import Path
import json

import cv2

from core.face_dataset import count_images_per_employee, load_training_data
from db.database import get_employee_codes

# Train model, load model da train, du doan 1 khuon mat moi
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "lbph_model.yml"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
IMAGE_SIZE = (200, 200)


def ensure_model_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def create_recognizer():
    """
    Tao model LBPH Face Recognizer.
    """
    return cv2.face.LBPHFaceRecognizer_create()


def clear_model_cache():
    _load_model_cached.cache_clear()


@lru_cache(maxsize=1)
def _load_model_cached(model_mtime_ns, label_map_mtime_ns):
    recognizer = create_recognizer()
    recognizer.read(str(MODEL_PATH))

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as file:
        label_to_code_raw = json.load(file)

    label_to_code = {int(key): value for key, value in label_to_code_raw.items()}
    return recognizer, label_to_code


def train_model():
    """
    Train model tu dataset hien co va luu model xuong file.
    Chi train cac thu muc anh co employee_code hop le trong database.
    """
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

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as file:
        json.dump(label_to_code, file, ensure_ascii=False, indent=2)

    clear_model_cache()

    return {
        "num_images": len(images),
        "num_people": len(label_to_code),
        "label_to_code": label_to_code,
        "images_per_employee": count_images_per_employee(valid_employee_codes),
    }


def load_model():
    """
    Load model da train va label map.
    Dung cache de tranh doc lai file model lien tuc khi recognition live.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Please train model first.")

    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError("Label map file not found. Please train model first.")

    model_mtime_ns = MODEL_PATH.stat().st_mtime_ns
    label_map_mtime_ns = LABEL_MAP_PATH.stat().st_mtime_ns
    return _load_model_cached(model_mtime_ns, label_map_mtime_ns)


def predict_face(face_image):
    """
    Du doan khuon mat thuoc nhan vien nao.

    Dau vao:
    - face_image: anh mat da crop

    Tra ve:
    - predicted_code: ma nhan vien du doan
    - confidence: do tin cay cua LBPH (cang thap thuong cang tot)
    """
    recognizer, label_to_code = load_model()

    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, IMAGE_SIZE)

    predicted_label, confidence = recognizer.predict(gray_face)
    predicted_code = label_to_code.get(predicted_label, "Unknown")

    return predicted_code, confidence
