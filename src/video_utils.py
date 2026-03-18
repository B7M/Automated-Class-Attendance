# Utilities for video playback and frame extraction.

import os
import cv2

from config import FRAMES_DIR


def preview_video(video_path: str) -> None:
    """Display a video in a window. Press 'q' to quit."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video Preview", frame)
        if cv2.waitKey(30) == ord("q"):
            print("Preview closed.")
            break

    cap.release()
    cv2.destroyAllWindows()


def extract_frames(video_path: str, frame_numbers: list[int], output_dir: str = FRAMES_DIR) -> list[str]:
    """
    Extract specific frames from a video and save them as JPEG files.

    Args:
        video_path:    Path to the video file.
        frame_numbers: List of frame indices to extract.
        output_dir:    Directory to save extracted frames.

    Returns:
        List of saved frame file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    saved_paths = []
    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"frame_{frame_number}.jpg")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)
        else:
            print(f"Warning: Could not read frame {frame_number}.")

    cap.release()
    print(f"Extracted {len(saved_paths)} frames to '{output_dir}'.")
    return saved_paths