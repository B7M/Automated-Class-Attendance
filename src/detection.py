# Face detection, cropping, and data augmentation.

import os
import shutil

import cv2
import dlib
import numpy as np

from config import (
    FACE_DETECTOR_PATH,
    SHAPE_PREDICTOR_PATH,
    FACE_BOXES_DIR,
    FACE_CROPS_DIR,
    FACE_MARGIN,
    AUGMENTATION_COUNT,
)


def load_detector() -> tuple:
    """Load and return the Dlib CNN face detector and shape predictor."""
    detector = dlib.cnn_face_detection_model_v1(FACE_DETECTOR_PATH)
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    return detector, predictor


def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to a face crop.
    Transforms: rotation (+-10 deg), brightness jitter, Gaussian blur (50%), horizontal flip (50%).
    """
    # Rotation
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Brightness
    factor = np.random.uniform(0.9, 1.1)
    image = np.clip(image * factor, 0, 255).astype(np.uint8)

    # Gaussian blur
    if np.random.rand() > 0.5:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    # Horizontal flip
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)

    return image


def detect_and_crop_faces(
    image_path: str,
    detector: dlib.cnn_face_detection_model_v1,
    output_crops_dir: str = FACE_CROPS_DIR,
    output_boxes_dir: str = FACE_BOXES_DIR,
    frame_tag: str = "",
    n_augmentations: int = AUGMENTATION_COUNT,
    margin: int = FACE_MARGIN,
) -> list[tuple[str, int, int]]:
    """
    Detect faces in an image, crop them (with augmentation), and save outputs.

    Args:
        image_path:        Path to the input image.
        detector:          Loaded Dlib CNN face detector.
        output_crops_dir:  Directory to save face crops.
        output_boxes_dir:  Directory to save annotated images.
        frame_tag:         Prefix tag for filenames (e.g. 'frame_100').
        n_augmentations:   Number of augmented copies per face.
        margin:            Pixel margin added around each detected face.

    Returns:
        List of (crop_path, face_index, aug_index) tuples.
    """
    os.makedirs(output_crops_dir, exist_ok=True)
    os.makedirs(output_boxes_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image '{image_path}'. Skipping.")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    results = []
    for i, face in enumerate(faces):
        x1 = max(0, face.rect.left() - margin)
        y1 = max(0, face.rect.top() - margin)
        x2 = min(image.shape[1], face.rect.right() + margin)
        y2 = min(image.shape[0], face.rect.bottom() + margin)

        face_crop = image[y1:y2, x1:x2]
        augmented = [face_crop] + [augment_image(face_crop) for _ in range(n_augmentations)]

        for j, aug_face in enumerate(augmented):
            filename = f"{frame_tag}_face_{i + 1}_aug_{j}.jpg"
            crop_path = os.path.join(output_crops_dir, filename)
            cv2.imwrite(crop_path, aug_face)
            results.append((crop_path, i + 1, j))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    boxes_path = os.path.join(output_boxes_dir, f"{frame_tag}_face_boxes.jpg")
    cv2.imwrite(boxes_path, image)

    return results


def extract_faces_from_image(
    image_path: str,
    output_folder: str,
    margin: int = FACE_MARGIN,
) -> list[str]:
    """
    Detect and crop faces from a single image for inference (no augmentation).

    Args:
        image_path:    Path to the input image.
        output_folder: Directory to save cropped faces and annotated image.
        margin:        Pixel margin around each detected face.

    Returns:
        List of saved face crop file paths.
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    detector, _ = load_detector()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = detector(gray, 1)

    face_paths = []
    for i, face in enumerate(detected_faces):
        x1 = max(0, face.rect.left() - margin)
        y1 = max(0, face.rect.top() - margin)
        x2 = min(image.shape[1], face.rect.right() + margin)
        y2 = min(image.shape[0], face.rect.bottom() + margin)

        face_crop = image[y1:y2, x1:x2]
        face_path = os.path.join(output_folder, f"face_{i}.jpg")
        cv2.imwrite(face_path, face_crop)
        face_paths.append(face_path)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_folder, "face_detected.png"), image)
    return face_paths