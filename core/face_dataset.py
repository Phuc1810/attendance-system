from pathlib import Path

import cv2
import numpy as np

# Doc toan bo anh trong data/faces, chuyen du lieu anh thanh dang model de train
# Tao mapping giua label so va ma nhan vien
DATASET_DIR = Path("data/faces")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


def list_employee_folders(valid_employee_codes=None):
    """
    Tra ve danh sach thu muc nhan vien trong data/faces.
    Chi lay cac thu muc hop le neu co danh sach employee_code tu database.
    """
    if not DATASET_DIR.exists():
        return []

    valid_codes = set(valid_employee_codes or [])
    folders = [folder for folder in DATASET_DIR.iterdir() if folder.is_dir()]

    if valid_codes:
        folders = [folder for folder in folders if folder.name in valid_codes]

    return sorted(folders, key=lambda folder: folder.name)


def list_image_files(folder_path):
    """
    Tra ve danh sach file anh hop le trong 1 thu muc nhan vien.
    """
    folder = Path(folder_path)
    if not folder.exists():
        return []

    return sorted(
        [
            file_path
            for file_path in folder.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS
        ],
        key=lambda file_path: file_path.name.lower(),
    )


def preprocess_face_image(image, image_size=(200, 200)):
    """
    Chuan hoa anh khuon mat truoc khi train/predict de giam khac biet giua camera.
    """
    if image is None:
        return None

    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    gray_image = cv2.resize(gray_image, image_size)
    gray_image = cv2.equalizeHist(gray_image)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    return clahe.apply(gray_image)


def load_training_data(image_size=(200, 200), valid_employee_codes=None):
    """
    Doc toan bo dataset khuon mat de train model.

    Tra ve:
    - images: list anh grayscale
    - labels: list nhan so tuong ung
    - label_to_code: dict anh xa so -> ma nhan vien
    - code_to_label: dict anh xa ma nhan vien -> so
    """
    images = []
    labels = []
    label_to_code = {}
    code_to_label = {}

    employee_folders = list_employee_folders(valid_employee_codes)

    for label_index, employee_folder in enumerate(employee_folders):
        employee_code = employee_folder.name
        label_to_code[label_index] = employee_code
        code_to_label[employee_code] = label_index

        image_files = list_image_files(employee_folder)

        for image_file in image_files:
            image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Cannot read image: {image_file}")
                continue

            processed_image = preprocess_face_image(image, image_size)
            images.append(processed_image)
            labels.append(label_index)

    return images, np.array(labels), label_to_code, code_to_label


def count_images_per_employee(valid_employee_codes=None):
    """
    Thong ke so anh cua tung nhan vien trong dataset hop le.
    """
    stats = {}

    for employee_folder in list_employee_folders(valid_employee_codes):
        employee_code = employee_folder.name
        stats[employee_code] = len(list_image_files(employee_folder))

    return stats