import cv2
import os
import uuid

def extract_frames(video_path, output_dir, max_frames=30):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"{uuid.uuid4().hex}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_frames.append(frame_filename)
        frame_count += 1

    cap.release()
    return saved_frames
