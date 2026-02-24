#!/usr/bin/env python3
"""Simple camera viewer to test OpenCV CUDA GPU support."""

import time

import argparse
import pathlib

import cv2
import numpy as np

from capture_ui import CaptureUI
from lens_controller import LensController
from lightning import Lightning

# ----------------------------
# Config
# ----------------------------
DISPLAY_W, DISPLAY_H = 1280, 800  # your screen resolution
CAPTURE_W, CAPTURE_H = 1280, 800  # capture size (lower = less latency)
CAPTURE_FPS = 60
DATA_DIR = "captures"

CLASS_NAMES = [
    "background",
    "black_stain",
    "corrosion",
    "crack",
    "deformation",
    "missing_part",
    "ok",
    "other",
    "silicate_stain",
    "water_stain"
]


# ----------------------------
# Camera pipeline
# ----------------------------
def gstreamer_pipeline(
        sensor_id: int = 0,
        sensor_mode: int = 1,
        capture_width: int = 2464,
        capture_height: int = 2064,
        flip_method: int = 2,
        framerate: int = 60
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} aelock=true awblock=false wbmode=2 tnr-mode=0 tnr-strength=-1 ee-mode=2 ee-strength=0 saturation=0.75 gainrange=\"1 1\" ispdigitalgainrange=\"1 1\" "
        f"! video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, framerate={framerate}/1, format=NV12 "
        f"! nvvidconv flip-method={flip_method} "
        "! video/x-raw, format=BGRx "
        "! queue max-size-buffers=1 leaky=downstream "
        "! appsink drop=1 max-buffers=1 sync=false"
    )

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Camera stream with TensorRT inference")
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Directory to store captured images",
    )
    parser.add_argument(
        "--lens-adjust-interval-s",
        type=float,
        default=30.0,
        help="Interval in seconds for lens focus adjustment",
    )
    parser.add_argument(
        "--lens-target-focus-mm",
        type=float,
        default=50.0,
        help="Target focus in mm for lens adjustment",
    )
    parser.add_argument(
        "--temp-path",
        type=str,
        default="/sys/class/hwmon/hwmon1/temp1_input",
        help="Path to temperature input file for lens adjustment",
    )
    args = parser.parse_args()

    # Check OpenCV CUDA support
    print(f"OpenCV version: {cv2.__version__}")
    print(f"CUDA enabled: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
        cv2.cuda.printShortCudaDeviceInfo(0)
    print()

    l = Lightning()
    l.start()

    lens_controller = LensController(
        args.lens_adjust_interval_s,
        args.lens_target_focus_mm,
        args.temp_path,
    )
    lens_controller.adjust_once()
    lens_controller.start()

    delay = 2
    print(f"Waiting {delay} seconds for camera to initialize...")
    time.sleep(delay)
    
    cap = None
    pipeline = gstreamer_pipeline(capture_width=CAPTURE_W, capture_height=CAPTURE_H, framerate=CAPTURE_FPS)
    try:
        candidate = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if candidate.isOpened():
            cap = candidate
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception as e:
        print(f"Error opening pipeline: {e}")

    if cap is None or not cap.isOpened():
        raise RuntimeError("Failed to open camera with CSI GStreamer pipelines")

    window_name = "Camera Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H)

    frame_toggle = 0
    ui = CaptureUI(CLASS_NAMES, args.data_dir, DISPLAY_W)
    prev_frame: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    last_saved_paths: list[pathlib.Path] | None = None
    last_saved_time = 0.0

    print("Press ESC to exit. Click a class, increment instrument, then Capture.\n")

    def frame_luma_mean(frame: np.ndarray) -> float:
        if frame.ndim == 3 and frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def on_mouse(event, x, y, _flags, _param):
        nonlocal prev_frame, last_frame, last_saved_paths, last_saved_time
        if event == cv2.EVENT_LBUTTONDOWN:
            action = ui.handle_click(x, y)
            if action == "capture" and last_frame is not None and prev_frame is not None:
                last_luma = frame_luma_mean(last_frame)
                prev_luma = frame_luma_mean(prev_frame)
                if last_luma >= prev_luma:
                    bright_frame, dark_frame = last_frame, prev_frame
                else:
                    bright_frame, dark_frame = prev_frame, last_frame
                dark_path, bright_path = ui.save_frames(dark_frame, bright_frame)
                last_saved_paths = [bright_path, dark_path]
                last_saved_time = time.time()

    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame")
                break
            frame_toggle = (frame_toggle + 1) % 2
            prev_frame = last_frame
            last_frame = frame.copy()
            if frame_toggle == 0:
                continue
            ui.draw(frame)

            if last_saved_paths is not None and (time.time() - last_saved_time) < 1.5:
                cv2.putText(
                    frame,
                    "Saved:",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                for idx, path in enumerate(last_saved_paths):
                    cv2.putText(
                        frame,
                        path.name,
                        (20, 50 + (idx + 1) * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            # Exit on ESC
            if key == 27:
                break

    finally:
        lens_controller.stop()
        lens_controller.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        print("\nCapture session ended.")


if __name__ == "__main__":
    main()