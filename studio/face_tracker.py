"""Scene-Aware Smart Face Tracking for PeakCut.
Detects camera scene cuts and tracks active speakers to dynamically center 9:16 vertical crop.
"""
import os
import cv2
import numpy as np
from pathlib import Path

CASCADE_FILE = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
_classifier = None


def get_face_detector():
    global _classifier
    if _classifier is None:
        if os.path.exists(CASCADE_FILE):
            _classifier = cv2.CascadeClassifier(CASCADE_FILE)
    return _classifier


def analyze_scene_and_faces(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    src_w: int,
    src_h: int,
    out_w: int = 1080,
    out_h: int = 1920
) -> str:
    """
    Analyzes video segment between start_sec and end_sec.
    Detects scene cuts and speaker faces.
    Returns an FFmpeg crop expression for 'x' parameter mapped to scaled video coordinates:
      - scale=1080:1920:force_original_aspect_ratio=increase
      - crop=1080:1920:x='<returned_expr>':y=0
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return ""

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Video is scaled so height becomes out_h (1920)
        # width becomes scaled_w (e.g. 3413 for 16:9)
        scale_ratio = out_h / float(src_h) if src_h > 0 else 1.0
        scaled_w = int(round(src_w * scale_ratio))
        max_crop_x = max(0, scaled_w - out_w)
        default_crop_x = max_crop_x // 2

        # Fast forward to start_sec
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_sec) * 1000.0)

        # Sample at ~4 frames per second
        sample_interval_sec = 0.25
        sample_interval_frames = max(1, int(fps * sample_interval_sec))

        classifier = get_face_detector()
        prev_gray_thumb = None
        duration = max(0.1, end_sec - start_sec)

        # Samples: (rel_time, crop_x, is_scene_cut)
        samples = []
        prev_center_scaled_x = scaled_w // 2

        while True:
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            rel_t = (pos_msec / 1000.0) - start_sec
            if rel_t > duration:
                break

            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # 1. Scene cut detection using downscaled 160x90 grayscale difference
            thumb = cv2.resize(frame, (160, 90))
            gray_thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
            is_cut = False

            if prev_gray_thumb is not None:
                diff = float(np.mean(np.abs(gray_thumb.astype(float) - prev_gray_thumb.astype(float))))
                if diff > 30.0:  # Camera cut threshold
                    is_cut = True

            prev_gray_thumb = gray_thumb

            # 2. Face detection
            detect_frame = frame
            downscale = 1.0
            if h > 480:
                downscale = 480.0 / h
                detect_frame = cv2.resize(frame, (int(w * downscale), 480))

            gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
            detected_x_orig = None

            if classifier and not classifier.empty():
                faces = classifier.detectMultiScale(
                    gray,
                    scaleFactor=1.18,
                    minNeighbors=4,
                    minSize=(int(gray.shape[0] * 0.12), int(gray.shape[0] * 0.12))
                )
                if len(faces) > 0:
                    largest = max(faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = largest
                    detected_x_orig = int((fx + fw // 2) / downscale)

            # 3. Map to scaled output coordinates
            if detected_x_orig is not None:
                target_scaled_x = int(round(detected_x_orig * scale_ratio))
                if is_cut or prev_center_scaled_x is None:
                    # Scene Cut: Immediate snap to new speaker, zero dragging
                    curr_center_x = target_scaled_x
                else:
                    # Same shot: Smooth cinematic exponential moving average
                    curr_center_x = int(0.3 * target_scaled_x + 0.7 * prev_center_scaled_x)
                prev_center_scaled_x = curr_center_x
            else:
                if is_cut:
                    curr_center_x = scaled_w // 2
                    prev_center_scaled_x = curr_center_x
                else:
                    curr_center_x = prev_center_scaled_x if prev_center_scaled_x is not None else scaled_w // 2

            crop_x = max(0, min(curr_center_x - out_w // 2, max_crop_x))
            samples.append((round(rel_t, 2), crop_x, is_cut))

            # Skip ahead
            for _ in range(sample_interval_frames - 1):
                if not cap.grab():
                    break

        cap.release()

        if not samples:
            return ""

        # If variance is very low, center is already good
        all_x = [s[1] for s in samples]
        if max(all_x) - min(all_x) < 45:
            avg_x = int(np.mean(all_x))
            return f"{avg_x}"

        # Group samples into piecewise timeline segments
        segments = []
        current_x = samples[0][1]
        seg_start = 0.0

        for i in range(1, len(samples)):
            t, x, is_cut = samples[i]
            if is_cut or abs(x - current_x) > 60 or (t - seg_start) > 3.0:
                segments.append((t, current_x))
                current_x = x
                seg_start = t

        segments.append((duration + 1.0, current_x))

        # Build clean nested if-expression for FFmpeg
        expr = str(segments[-1][1])
        for end_t, x_val in reversed(segments[:-1]):
            expr = f"if(lt(t,{end_t:.2f}),{x_val},{expr})"

        return expr
    except Exception as e:
        print(f"[FaceTracker] Analysis error: {e}")
        return ""
