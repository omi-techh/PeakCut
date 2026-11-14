"""Cinematic Scene-Aware Deep Learning Face Tracking for PeakCut.
Uses OpenCV YuNet Deep Neural Network for full 360-degree pose face detection.
Features:
- Instant camera snap on scene cuts (zero-drag transitions).
- Gimbal deadzone to eliminate micro-jitter.
- Smooth cinematic exponential gliding within shots.
- Persistent anchor (never drifts to center when speaker turns sideways).
"""
import os
import cv2
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

CASCADE_FILE = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
_yunet_detector = None
_cascade_detector = None


def ensure_yunet_model():
    """Download YuNet ONNX model (~232 KB) if not already present."""
    if not YUNET_PATH.exists() or YUNET_PATH.stat().st_size < 100000:
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        except Exception as e:
            print(f"[FaceTracker] YuNet download notice: {e}")


def get_detector(width: int = 320, height: int = 320):
    """Load YuNet Deep Learning face detector with fallback to Haar Cascade."""
    global _yunet_detector, _cascade_detector
    ensure_yunet_model()

    if YUNET_PATH.exists() and hasattr(cv2, "FaceDetectorYN"):
        try:
            if _yunet_detector is None:
                _yunet_detector = cv2.FaceDetectorYN.create(
                    str(YUNET_PATH), "", (width, height), 0.45, 0.3, 5000
                )
            else:
                _yunet_detector.setInputSize((width, height))
            return ("yunet", _yunet_detector)
        except Exception:
            pass

    # Fallback to Haar Cascade
    if _cascade_detector is None and os.path.exists(CASCADE_FILE):
        _cascade_detector = cv2.CascadeClassifier(CASCADE_FILE)
    return ("cascade", _cascade_detector)


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
    Detects scene cuts and speaker faces using deep learning.
    Returns a smooth, cinematic FFmpeg crop expression for 'x' parameter.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return ""

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Scale factor so output height is 1920
        scale_ratio = out_h / float(src_h) if src_h > 0 else 1.0
        scaled_w = int(round(src_w * scale_ratio))
        max_crop_x = max(0, scaled_w - out_w)
        default_crop_x = max_crop_x // 2

        # Fast forward to start_sec
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_sec) * 1000.0)

        # Sample at ~6 frames per second for smooth trajectory tracking
        sample_interval_sec = 0.16
        sample_interval_frames = max(1, int(fps * sample_interval_sec))

        # Analysis resolution: 480p for high-speed & high-accuracy face detection
        det_h = 360
        det_w = int(round(src_w * (det_h / float(src_h)))) if src_h > 0 else 640
        det_type, detector = get_detector(det_w, det_h)

        prev_gray_thumb = None
        duration = max(0.2, end_sec - start_sec)

        # Tracking state
        curr_camera_x = default_crop_x
        last_known_face_x = default_crop_x
        has_face_in_current_shot = False

        # Timeline of smooth camera keyframes: [(t_rel, crop_x, is_cut)]
        raw_timeline = []

        while True:
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            rel_t = (pos_msec / 1000.0) - start_sec
            if rel_t > duration:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # 1. Scene cut detection (camera angle change)
            thumb = cv2.resize(frame, (120, 68))
            gray_thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
            is_cut = False

            if prev_gray_thumb is not None:
                diff = float(np.mean(np.abs(gray_thumb.astype(float) - prev_gray_thumb.astype(float))))
                if diff > 27.0:  # Camera cut threshold
                    is_cut = True
                    has_face_in_current_shot = False

            prev_gray_thumb = gray_thumb

            # 2. Deep Learning Face Detection (YuNet / Cascade)
            det_frame = cv2.resize(frame, (det_w, det_h))
            detected_scaled_center_x = None

            if det_type == "yunet" and detector is not None:
                _, faces = detector.detect(det_frame)
                if faces is not None and len(faces) > 0:
                    # Pick largest face by bounding box area (w * h)
                    largest = max(faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = largest[:4]
                    # Center X in original video scale
                    norm_center_x = (fx + fw / 2.0) / float(det_w)
                    detected_scaled_center_x = int(round(norm_center_x * scaled_w))
            elif det_type == "cascade" and detector is not None:
                gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.15, 4, minSize=(int(det_h * 0.1), int(det_h * 0.1)))
                if len(faces) > 0:
                    largest = max(faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = largest
                    norm_center_x = (fx + fw / 2.0) / float(det_w)
                    detected_scaled_center_x = int(round(norm_center_x * scaled_w))

            # 3. Target crop X calculation
            if detected_scaled_center_x is not None:
                target_crop_x = max(0, min(detected_scaled_center_x - out_w // 2, max_crop_x))
                last_known_face_x = target_crop_x
                has_face_in_current_shot = True
            else:
                # If face is temporarily lost (e.g. turned head), LOCK to last known face in this shot!
                # NEVER drift to center unless it's a brand new scene cut with no face!
                if is_cut and not has_face_in_current_shot:
                    target_crop_x = default_crop_x
                    last_known_face_x = default_crop_x
                else:
                    target_crop_x = last_known_face_x

            # 4. Cinematic Gimbal Movement Logic:
            if is_cut or not raw_timeline:
                # SCENE CUT: Snap immediately to new speaker! Zero drag across cuts
                curr_camera_x = target_crop_x
            else:
                # WITHIN CONTINUOUS SHOT:
                # Gimbal Deadzone: Micro-movements (< 35px) don't twitch the camera
                dx = target_crop_x - curr_camera_x
                if abs(dx) > 35:
                    # Smooth exponential glide (cameraman easing)
                    curr_camera_x = int(round(curr_camera_x + 0.22 * dx))

            raw_timeline.append((round(rel_t, 2), curr_camera_x, is_cut))

            # Skip ahead
            for _ in range(sample_interval_frames - 1):
                if not cap.grab():
                    break

        cap.release()

        if not raw_timeline:
            return ""

        # Check total movement across clip
        all_x = [s[1] for s in raw_timeline]
        if max(all_x) - min(all_x) < 40:
            return f"{int(np.mean(all_x))}"

        # 5. Build Smooth Linear-Interpolated FFmpeg Expression
        # Filter down into keyframes at cuts and movement inflection points
        keyframes = [(raw_timeline[0][0], raw_timeline[0][1])]
        last_k_x = raw_timeline[0][1]

        for i in range(1, len(raw_timeline) - 1):
            t, x, is_cut = raw_timeline[i]
            prev_t, prev_x, _ = raw_timeline[i - 1]
            next_t, next_x, _ = raw_timeline[i + 1]

            if is_cut:
                # Hard cut keyframe: finish previous shot, snap to new
                keyframes.append((round(max(0, t - 0.04), 2), last_k_x))
                keyframes.append((t, x))
                last_k_x = x
            elif abs(x - last_k_x) >= 45 or (t - keyframes[-1][0]) >= 2.0:
                keyframes.append((t, x))
                last_k_x = x

        keyframes.append((round(duration + 0.5, 2), raw_timeline[-1][1]))

        # Generate piecewise linear interpolation expression:
        # For between(t, t1, t2): x1 + (x2 - x1) * (t - t1) / (t2 - t1)
        expr = str(keyframes[-1][1])
        for i in reversed(range(len(keyframes) - 1)):
            t1, x1 = keyframes[i]
            t2, x2 = keyframes[i + 1]
            dt = max(0.01, t2 - t1)
            dx = x2 - x1

            if dt < 0.08 or abs(dx) < 2:
                # Instant cut / static step
                seg_expr = str(x1)
            else:
                # Smooth continuous linear glide
                seg_expr = f"({x1}+{dx}*(t-{t1:.2f})/{dt:.2f})"

            expr = f"if(lt(t,{t2:.2f}),{seg_expr},{expr})"

        return expr

    except Exception as e:
        print(f"[FaceTracker] Cinematic tracking error: {e}")
        return ""
