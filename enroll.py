# Enrollment pipeline: extract frames, detect faces, extract embeddings, cluster identities.
# Run once per cohort, then manually populate map/cluster_to_student.json.

import os
import shutil

from config import (
    VIDEO_PATH,
    FRAME_NUMBERS,
    FRAMES_DIR,
    FACE_BOXES_DIR,
    FACE_CROPS_DIR,
    EMBEDDINGS_DIR,
)
from video_utils import extract_frames
from detection import load_detector, detect_and_crop_faces
from embeddings import load_facenet_model, extract_and_save_embeddings, load_enrollment_embeddings
from clustering import plot_elbow, plot_k_distance, run_dbscan, plot_pca, plot_tsne


def reset_enrollment_dirs() -> None:
    """Clear all intermediate enrollment directories to start fresh."""
    for d in [FACE_BOXES_DIR, FACE_CROPS_DIR, EMBEDDINGS_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)


def main():
    # 1. Extract frames from video
    print("--- Step 1: Extracting frames ---")
    extract_frames(VIDEO_PATH, FRAME_NUMBERS, output_dir=FRAMES_DIR)

    # 2. Reset output directories
    reset_enrollment_dirs()

    # 3. Load models
    print("--- Step 2: Loading models ---")
    detector, _ = load_detector()
    facenet = load_facenet_model()

    # 4. Detect faces, augment, and extract embeddings
    print("--- Step 3: Detecting faces and extracting embeddings ---")
    for frame_number in FRAME_NUMBERS:
        image_path = os.path.join(FRAMES_DIR, f"frame_{frame_number}.jpg")
        frame_tag = f"frame_{frame_number}"

        crop_results = detect_and_crop_faces(
            image_path=image_path,
            detector=detector,
            frame_tag=frame_tag,
        )
        extract_and_save_embeddings(
            model=facenet,
            crop_results=crop_results,
            frame_tag=frame_tag,
        )

    print("Embedding extraction complete.")

    # 5. Load all saved embeddings
    print("--- Step 4: Loading embeddings for clustering ---")
    import numpy as np
    embeddings, _, filenames = load_enrollment_embeddings()

    if len(embeddings) == 0:
        print("No embeddings found. Check detection and quality threshold.")
        return

    # 6. Diagnostic plots (optional — comment out if running headless)
    print("--- Step 5: Cluster diagnostics ---")
    plot_elbow(embeddings)
    plot_k_distance(embeddings, k=4)

    # 7. Run DBSCAN clustering
    print("--- Step 6: Clustering ---")
    labels = run_dbscan(embeddings, filenames)

    # 8. Visualize clusters
    plot_pca(embeddings, labels)
    plot_tsne(embeddings, labels)

    print("\nEnrollment complete.")
    print("Next step: review cluster folders in './frames/clusters/' and populate 'map/cluster_to_student.json'.")
    print("Example format:")
    print('  { "0": "Jane Smith", "1": "John Doe", ... }')


if __name__ == "__main__":
    main()