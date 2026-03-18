# Face Recognition Attendance System

Automated classroom attendance from video — no manual roll call. Detects faces with Dlib's CNN detector, extracts 512-d FaceNet embeddings, clusters identities via DBSCAN/KMeans, and matches new faces at inference time using KNN.

---

## How It Works

**Enrollment (once per cohort):** Sample frames from reference video → detect and crop faces → extract augmented FaceNet embeddings → cluster with DBSCAN/KMeans → validate clusters with t-SNE → map cluster IDs to student names in a JSON file.

**Inference:** Detect faces in a new image → extract embeddings → KNN match against enrolled database → output attendance set.

---

## Stack

| Component | Tool |
|---|---|
| Face Detection | Dlib CNN (`mmod_human_face_detector`) |
| Embeddings | FaceNet `InceptionResnetV1` — VGGFace2, 512-d |
| Clustering | DBSCAN + KMeans (scikit-learn) |
| Identity Matching | KNN, k = number of students, Euclidean distance |
| Visualization | PCA / t-SNE |
| Acceleration | CUDA with automatic CPU fallback |

---

## Project Structure

```
face-recognition-attendance/
├── app.py                            # Inference entry point
├── enroll.py                         # Enrollment entry point
├── config.py                         # All paths and hyperparameters
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── video_utils.py                # Frame extraction and video preview
│   ├── detection.py                  # Face detection, cropping, augmentation
│   ├── embeddings.py                 # FaceNet embedding extraction and I/O
│   ├── clustering.py                 # DBSCAN, elbow/k-distance plots, PCA/t-SNE
│   └── inference.py                  # KNN matching and attendance recording
├── models/
│   ├── mmod_human_face_detector.dat
│   └── shape_predictor_68_face_landmarks.dat
├── frames/
│   ├── embeddings/                   # Per-face .npy files
│   └── cluster_mapping.json          # filename → cluster ID
└── map/
    └── cluster_to_student.json       # cluster ID → student name
```

---

## Installation

```bash
git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition-attendance
pip install -r requirements.txt
```

Download pretrained weights into `models/`:
- [`mmod_human_face_detector.dat`](http://dlib.net/files/mmod_human_face_detector.dat.bz2)
- [`shape_predictor_68_face_landmarks.dat`](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

> If dlib installation fails, ensure cmake is available: `brew install cmake` or `sudo apt install cmake`.

---

## Usage

**Enrollment** — run once per cohort, then populate the student name mapping:

```bash
python enroll.py
```

```json
{ "0": "Student 1", "1": "Student 2", "2": "Student 3" }
```

**Inference:**

```bash
python app.py
```

Set a custom input image in `app.py`:

```python
IMAGE_PATH = "path/to/image.jpg"
```

Annotated output and detected faces are saved to `new_session/`.

---

## Design Notes

- **Augmentation during enrollment:** each face generates 4 embeddings (rotation ±10°, brightness jitter, Gaussian blur, horizontal flip), expanding the database and improving robustness to lighting and pose variation at inference time.
- **Unsupervised identity discovery:** since ground-truth labels are unavailable from raw video, DBSCAN handles noise-robust cluster discovery before KMeans refinement; t-SNE plots serve as a visual sanity check prior to manual labeling.
- **Quality filtering:** embeddings with variance below threshold are discarded to prevent occluded or blurry faces from polluting the enrollment database.

---

## License

MIT