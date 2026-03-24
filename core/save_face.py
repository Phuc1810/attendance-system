import os

import cv2

DATASET_DIR = "data/faces"
FACE_SIZE = (200, 200)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_employee_folder(employee_id):
    employee_folder = os.path.join(DATASET_DIR, str(employee_id))
    ensure_dir(employee_folder)
    return employee_folder


def get_next_image_number(employee_id):
    employee_folder = get_employee_folder(employee_id)
    image_numbers = []

    for file_name in os.listdir(employee_folder):
        name, ext = os.path.splitext(file_name)
        if ext.lower() in [".jpg", ".jpeg", ".png"] and name.isdigit():
            image_numbers.append(int(name))

    if not image_numbers:
        return 1

    return max(image_numbers) + 1


def crop_and_resize_face(image, face_box):
    x, y, w, h = face_box
    face_crop = image[y:y + h, x:x + w]
    face_crop = cv2.resize(face_crop, FACE_SIZE)
    return face_crop


def save_face_image(employee_id, face_image):
    employee_folder = get_employee_folder(employee_id)
    next_number = get_next_image_number(employee_id)
    save_path = os.path.join(employee_folder, f"{next_number}.jpg")
    cv2.imwrite(save_path, face_image)
    return save_path
