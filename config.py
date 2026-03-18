# Central configuration for paths and hyperparameters.

# --- Paths ---
VIDEO_PATH = "classroom.mp4"

MODELS_DIR = "./models"
FACE_DETECTOR_PATH = f"{MODELS_DIR}/mmod_human_face_detector.dat"
SHAPE_PREDICTOR_PATH = f"{MODELS_DIR}/shape_predictor_68_face_landmarks.dat"

FRAMES_DIR = "./frames"
FACE_BOXES_DIR = f"{FRAMES_DIR}/face_boxes"
FACE_CROPS_DIR = f"{FRAMES_DIR}/face_crops"
EMBEDDINGS_DIR = f"{FRAMES_DIR}/embeddings"
CLUSTERS_DIR = f"{FRAMES_DIR}/clusters"
CLUSTER_MAPPING_PATH = f"{FRAMES_DIR}/cluster_mapping.json"

MAP_DIR = "./map"
STUDENT_MAP_PATH = f"{MAP_DIR}/cluster_to_student.json"

# --- Enrollment ---
FRAME_NUMBERS = [100, 200, 300, 400, 500]
FACE_MARGIN = 20               # Pixels added around each detected face crop
AUGMENTATION_COUNT = 3         # Number of augmented copies per face
EMBEDDING_VARIANCE_THRESHOLD = 0.00194  # Embeddings below this are discarded

# --- Clustering ---
DBSCAN_EPS = 0.55
DBSCAN_MIN_SAMPLES = 5
KMEANS_K_RANGE = range(1, 10)  # Range for elbow plot

# --- Inference ---
KNN_NEIGHBORS = 4
NEW_SESSION_DIR = "./new_session"