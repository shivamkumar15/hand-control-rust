#!/usr/bin/env python3
"""
Minimal MediaPipe hand landmark streamer using the new Tasks API.
Outputs one JSON object per line on stdout:
    {"landmarks": [[x,y,z], [x,y,z], ...]}   # one hand, 21 points
    {"landmarks": null}                       # no hand detected
"""
import argparse
import json
import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode


def open_camera(pref: int) -> cv2.VideoCapture:
    # Try the requested index first, then scan a few common ones.
    for idx in [pref, *range(0, 5)]:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                if idx != pref:
                    print(f"camera {pref} unavailable, using camera {idx}", file=sys.stderr)
                return cap
        cap.release()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "hand_landmarker.task")
    if not os.path.exists(model_path):
        # Try the zip we extracted
        alt = os.path.join(script_dir, "models", "hand_landmarker.task.zip")
        if os.path.exists(alt):
            model_path = alt
        else:
            print(f"ERROR: model not found at {model_path}", file=sys.stderr)
            sys.exit(1)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)

    cap = open_camera(args.camera)
    if cap is None:
        print("ERROR: no working camera found", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always analyze the freshest frame

    start_time = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)  # don't burn CPU if the camera hiccups
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            hands = []
            if result.hand_landmarks:
                for i, lm in enumerate(result.hand_landmarks):
                    landmarks = [[p.x, p.y, p.z] for p in lm]
                    handedness = None
                    if result.handedness and i < len(result.handedness):
                        categories = result.handedness[i]
                        if categories:
                            handedness = categories[0].category_name
                    hands.append({"landmarks": landmarks, "handedness": handedness})

            print(json.dumps({"hands": hands}), flush=True)

            if args.preview:
                cv2.imshow("tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
