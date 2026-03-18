# FaceNet embedding extraction and loading from disk.

import os
import json

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from config import EMBEDDINGS_DIR, CLUSTER_MAPPING_PATH, EMBEDDING_VARIANCE_THRESHOLD


def load_facenet_model() -> InceptionResnetV1:
    """Load FaceNet (InceptionResnetV1, VGGFace2) and move to GPU if available."""
    model = InceptionResnetV1(pretrained="vggface2").eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Resize and normalize a BGR face crop to a FaceNet-compatible tensor (1, 3, 160, 160)."""
    resized = cv2.resize(image, (160, 160))
    tensor = torch.tensor(resized).permute(2, 0, 1).float().div(255).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def extract_embedding(model: InceptionResnetV1, image: np.ndarray) -> np.ndarray:
    """
    Extract a 512-d FaceNet embedding from a single face image.

    Args:
        model: Loaded FaceNet model.
        image: BGR face crop as a numpy array.

    Returns:
        1-D numpy array of shape (512,).
    """
    tensor = image_to_tensor(image)
    with torch.no_grad():
        embedding = model(tensor).cpu().numpy().flatten()
    return embedding


def extract_and_save_embeddings(
    model: InceptionResnetV1,
    crop_results: list[tuple[str, int, int]],
    frame_tag: str,
    output_dir: str = EMBEDDINGS_DIR,
    variance_threshold: float = EMBEDDING_VARIANCE_THRESHOLD,
) -> list[str]:
    """
    Extract embeddings from face crops (enrollment phase) and save qualifying ones.

    Args:
        model:              Loaded FaceNet model.
        crop_results:       List of (crop_path, face_idx, aug_idx) from detection.
        frame_tag:          Frame identifier prefix for filenames.
        output_dir:         Directory to save .npy embedding files.
        variance_threshold: Minimum embedding variance; low-variance embeddings are discarded.

    Returns:
        List of saved .npy file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    for crop_path, face_idx, aug_idx in crop_results:
        image = cv2.imread(crop_path)
        if image is None:
            print(f"Warning: Cannot load crop '{crop_path}'. Skipping.")
            continue

        embedding = extract_embedding(model, image)

        if np.var(embedding) < variance_threshold:
            print(f"Low-quality embedding: {frame_tag}, face {face_idx} (aug {aug_idx}). Discarded.")
            continue

        filename = f"{frame_tag}_face_{face_idx}_aug_{aug_idx}.npy"
        out_path = os.path.join(output_dir, filename)
        np.save(out_path, embedding)
        saved.append(out_path)

    return saved


def extract_embeddings_from_faces(
    model: InceptionResnetV1,
    face_paths: list[str],
) -> np.ndarray:
    """
    Extract embeddings for a list of face crop paths (inference phase).

    Args:
        model:      Loaded FaceNet model.
        face_paths: Paths to face crop images.

    Returns:
        Array of shape (N, 512).
    """
    embeddings = []
    for face_path in face_paths:
        image = cv2.imread(face_path)
        if image is None:
            print(f"Warning: Cannot load '{face_path}'. Skipping.")
            continue
        try:
            embedding = extract_embedding(model, image)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error extracting embedding for '{face_path}': {e}")
    return np.array(embeddings)


def load_enrollment_embeddings(
    embeddings_dir: str = EMBEDDINGS_DIR,
    cluster_mapping_path: str = CLUSTER_MAPPING_PATH,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load saved embeddings and their cluster labels from disk.

    Args:
        embeddings_dir:       Directory containing .npy embedding files.
        cluster_mapping_path: JSON file mapping filenames to cluster IDs.

    Returns:
        Tuple of (embeddings array, labels array, filenames list).
    """
    with open(cluster_mapping_path, "r") as f:
        cluster_mapping = json.load(f)

    embeddings, labels, filenames = [], [], []
    for filename, cluster_label in cluster_mapping.items():
        path = os.path.join(embeddings_dir, filename)
        if os.path.exists(path):
            embeddings.append(np.load(path))
            labels.append(cluster_label)
            filenames.append(filename)

    return np.array(embeddings), np.array(labels), filenames