from pathlib import Path
import shutil

import cv2

#Cắt ảnh, chỉnh kích thước và lưu vào đúng ổ cứng
DATASET_DIR = Path("data/faces")
FACE_SIZE = (200, 200)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

#Kiểm tra xem thư mục data/face có mã NV đã tồn tại, nếu chưa thì tạo
def ensure_employee_folder(employee_code):
    employee_folder = DATASET_DIR / str(employee_code)
    ensure_dir(employee_folder)
    return employee_folder


def get_employee_folder(employee_code):
    return ensure_employee_folder(employee_code)


def _list_image_files(folder_path):
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


def get_next_image_number(employee_code):
    employee_folder = get_employee_folder(employee_code)
    image_numbers = []

    for file_path in _list_image_files(employee_folder):
        if file_path.stem.isdigit():
            image_numbers.append(int(file_path.stem))

    if not image_numbers:
        return 1

    return max(image_numbers) + 1


def sync_employee_face_folders(employee_rows):
    ensure_dir(DATASET_DIR)

    for employee_id, employee_code in employee_rows:
        if not employee_code:
            continue

        target_folder = ensure_employee_folder(employee_code)
        legacy_folder = DATASET_DIR / str(employee_id)

        if not legacy_folder.exists() or legacy_folder == target_folder:
            continue

        for file_path in _list_image_files(legacy_folder):
            next_number = get_next_image_number(employee_code)
            target_path = target_folder / f"{next_number}{file_path.suffix.lower()}"
            shutil.move(str(file_path), target_path)

        if legacy_folder.exists() and not any(legacy_folder.iterdir()):
            legacy_folder.rmdir()

# Cắt ô chứa khuôn mặt và ép về kích thước 
def crop_and_resize_face(image, face_box):
    x, y, w, h = face_box
    face_crop = image[y:y + h, x:x + w]
    face_crop = cv2.resize(face_crop, FACE_SIZE)
    return face_crop

#Lưu xuống ổ cứng
def save_face_image(employee_code, face_image):
    employee_folder = get_employee_folder(employee_code)
    next_number = get_next_image_number(employee_code)
    save_path = employee_folder / f"{next_number}.jpg"
    cv2.imwrite(str(save_path), face_image)
    return str(save_path)
