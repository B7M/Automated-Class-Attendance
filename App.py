# Inference entry point: detect faces in a new image and mark attendance.

from config import NEW_SESSION_DIR, EMBEDDINGS_DIR, CLUSTER_MAPPING_PATH, STUDENT_MAP_PATH
from detection import extract_faces_from_image
from embeddings import load_facenet_model, extract_embeddings_from_faces, load_enrollment_embeddings
from inference import load_student_map, match_faces_to_students, mark_attendance


def run(image_path: str) -> list[str]:
    """
    Run the full inference pipeline on a single image.

    Args:
        image_path: Path to the classroom image to process.

    Returns:
        List of unique present student names.
    """
    # 1. Detect and crop faces
    print(f"Processing: {image_path}")
    face_paths = extract_faces_from_image(image_path, output_folder=NEW_SESSION_DIR)
    if not face_paths:
        print("No faces detected.")
        return []

    # 2. Extract embeddings for detected faces
    facenet = load_facenet_model()
    new_embeddings = extract_embeddings_from_faces(facenet, face_paths)
    if len(new_embeddings) == 0:
        print("Embedding extraction failed for all detected faces.")
        return []

    # 3. Load enrollment database
    enrolled_embeddings, enrolled_labels, _ = load_enrollment_embeddings(
        embeddings_dir=EMBEDDINGS_DIR,
        cluster_mapping_path=CLUSTER_MAPPING_PATH,
    )

    # 4. Load student name mapping
    student_map = load_student_map(STUDENT_MAP_PATH)

    # 5. Match and record attendance
    matched_names = match_faces_to_students(
        new_embeddings, enrolled_embeddings, enrolled_labels, student_map
    )
    return mark_attendance(matched_names)


if __name__ == "__main__":
    IMAGE_PATH = "frame_100.jpg"  # Update as needed
    present = run(IMAGE_PATH)