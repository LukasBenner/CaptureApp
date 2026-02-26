# main.py
import argparse
import os
import pathlib
import re
import subprocess
import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

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
    "water_stain",
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
    framerate: int = 60,
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
# Async USB copy support
# ----------------------------
@dataclass
class CopyProgress:
    running: bool = False
    done: bool = False
    ok: bool = False

    phase: str = ""       # mounting/scanning/copying/unmounting/finished/error
    message: str = ""
    error: str = ""

    total_bytes: int = 0
    copied_bytes: int = 0
    total_files: int = 0
    copied_files: int = 0

    current_file: str = ""

    started_at: float = 0.0
    finished_at: float = 0.0

    # show a "toast" overlay after completion/error
    toast_until: float = 0.0

    lock: threading.Lock = field(default_factory=threading.Lock)


def reset_progress(p: CopyProgress) -> None:
    p.running = False
    p.done = False
    p.ok = False
    p.phase = ""
    p.message = ""
    p.error = ""
    p.total_bytes = 0
    p.copied_bytes = 0
    p.total_files = 0
    p.copied_files = 0
    p.current_file = ""
    p.started_at = 0.0
    p.finished_at = 0.0
    p.toast_until = 0.0


def _read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def _is_mounted(mount_point: str) -> bool:
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == mount_point:
                    return True
    except Exception:
        pass
    return False


def _ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_files(root: pathlib.Path) -> list[pathlib.Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def _safe_remove_empty_dirs(root: pathlib.Path) -> None:
    if not root.exists():
        return
    dirs = sorted(
        [p for p in root.rglob("*") if p.is_dir()],
        key=lambda x: len(str(x)),
        reverse=True,
    )
    for d in dirs:
        try:
            next(d.iterdir())
        except StopIteration:
            try:
                d.rmdir()
            except Exception:
                pass


def _copy_file_chunked(
    src: pathlib.Path,
    dst: pathlib.Path,
    chunk_size: int,
    on_bytes: Callable[[int], None],
) -> None:
    _ensure_dir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + ".part")

    with open(src, "rb") as fsrc, open(tmp, "wb") as fdst:
        while True:
            buf = fsrc.read(chunk_size)
            if not buf:
                break
            fdst.write(buf)
            on_bytes(len(buf))
        fdst.flush()
        os.fsync(fdst.fileno())

    os.replace(tmp, dst)

    # Preserve timestamps if possible
    try:
        st = src.stat()
        os.utime(dst, (st.st_atime, st.st_mtime))
    except Exception:
        pass


def copy_tree_to_usb(
    src_root: pathlib.Path,
    dst_root: pathlib.Path,
    progress: CopyProgress,
    remove_source_files: bool = True,
    chunk_size: int = 1024 * 1024,  # 1 MiB
) -> None:
    progress.phase = "scanning"
    progress.message = "Scanning files…"

    files = _iter_files(src_root)

    total_bytes = 0
    for p in files:
        try:
            total_bytes += p.stat().st_size
        except Exception:
            pass

    with progress.lock:
        progress.total_files = len(files)
        progress.total_bytes = total_bytes
        progress.copied_files = 0
        progress.copied_bytes = 0

    if progress.total_files == 0:
        progress.phase = "finished"
        progress.message = "No files to copy."
        progress.ok = True
        return

    progress.phase = "copying"
    progress.message = "Copying…"

    def on_bytes(n: int) -> None:
        with progress.lock:
            progress.copied_bytes += n

    for src in files:
        rel = src.relative_to(src_root)
        dst = dst_root / rel

        progress.current_file = str(rel)

        _copy_file_chunked(src, dst, chunk_size=chunk_size, on_bytes=on_bytes)
        with progress.lock:
            progress.copied_files += 1

        if remove_source_files:
            try:
                src.unlink()
            except Exception:
                # Keep going, but report last such issue
                progress.error = f"Could not remove source file: {src}"

    if remove_source_files:
        _safe_remove_empty_dirs(src_root)

    progress.phase = "finished"
    progress.message = "Copy finished."
    progress.ok = True


def _list_candidate_usb_partitions() -> list[str]:
    """
    Return a list of /dev/<partition> paths that look like removable USB partitions.
    Uses /sys/class/block and checks parent disk removable == 1.
    """
    sys_block = pathlib.Path("/sys/class/block")
    if not sys_block.exists():
        return []

    candidates: list[str] = []

    for dev in sys_block.iterdir():
        name = dev.name

        # Skip obvious non-USB block devices
        if name.startswith(("loop", "ram", "mmcblk", "nvme")):
            continue

        # Only partitions
        if not (dev / "partition").exists():
            continue

        # Most common case: sda1, sdb2, ...
        m = re.match(r"^(sd[a-z]+)(\d+)$", name)
        if not m:
            continue

        parent = m.group(1)
        parent_path = sys_block / parent
        if not parent_path.exists():
            continue

        removable = _read_text(parent_path / "removable")
        if removable != "1":
            continue

        candidates.append(f"/dev/{name}")

    return candidates


def _has_filesystem(dev_path: str) -> bool:
    """
    If blkid exists, prefer partitions with detectable filesystem.
    If blkid isn't present, return True and let mount attempt decide.
    """
    try:
        r = subprocess.run(
            ["blkid", "-o", "value", "-s", "TYPE", dev_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if r.returncode != 0:
            return False
        fs = (r.stdout.decode("utf-8", errors="replace") or "").strip()
        return bool(fs)
    except FileNotFoundError:
        return True
    except Exception:
        return False


def find_usb_partition() -> str:
    cands = _list_candidate_usb_partitions()
    if not cands:
        raise RuntimeError("No USB storage partition detected. Please insert a USB stick.")
    with_fs = [d for d in cands if _has_filesystem(d)]
    return with_fs[0] if with_fs else cands[0]


def start_async_copy_job(
    progress: CopyProgress,
    device: str,
    mount_point: pathlib.Path,
    src_root: pathlib.Path,
    dst_root: pathlib.Path,
) -> threading.Thread:
    def worker():
        with progress.lock:
            progress.running = True
            progress.done = False
            progress.ok = False
            progress.error = ""
            progress.message = ""
            progress.current_file = ""
            progress.phase = "mounting"
            progress.started_at = time.time()

        _ensure_dir(mount_point)

        mounted_by_us = False
        try:
            if _is_mounted(str(mount_point)):
                progress.message = "USB already mounted."
            else:
                progress.message = f"Mounting {device}…"
                r = subprocess.run(
                    ["mount", device, str(mount_point)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if r.returncode != 0:
                    out = (r.stdout.decode("utf-8", errors="replace") if r.stdout else "").strip()
                    raise RuntimeError(out or "Error mounting USB drive.")
                mounted_by_us = True

            copy_tree_to_usb(src_root=src_root, dst_root=dst_root, progress=progress)

        except Exception as e:
            with progress.lock:
                progress.error = str(e)
                progress.ok = False
                progress.phase = "error"

        finally:
            if mounted_by_us:
                progress.phase = "unmounting"
                progress.message = "Unmounting…"
                try:
                    r = subprocess.run(
                        ["umount", str(mount_point)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    if r.returncode != 0 and not progress.error:
                        out = (r.stdout.decode("utf-8", errors="replace") if r.stdout else "").strip()
                        with progress.lock:
                            progress.error = out or "Error unmounting USB drive."
                            progress.ok = False
                            progress.phase = "error"
                except Exception as e:
                    with progress.lock:
                        if not progress.error:
                            progress.error = str(e)
                        progress.ok = False
                        progress.phase = "error"

            now = time.time()
            with progress.lock:
                progress.running = False
                progress.done = True
                progress.finished_at = now
                progress.toast_until = now + 2.5
                if not progress.error:
                    progress.ok = True
                    progress.phase = "finished"
                    progress.message = "Copy to USB completed."

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


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
        "--meta-data-path",
        default="station_metadata.json",
        help="Path to station metadata JSON file",
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

    delay = 1
    print(f"Waiting {delay} seconds for camera to initialize...")
    time.sleep(delay)

    cap = None
    pipeline = gstreamer_pipeline(
        capture_width=CAPTURE_W,
        capture_height=CAPTURE_H,
        framerate=CAPTURE_FPS,
    )
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
    ui = CaptureUI(CLASS_NAMES, args.data_dir, args.meta_data_path, DISPLAY_W)
    prev_frame: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    last_luma = 0.0
    prev_luma = 0.0
    last_saved_paths: list[pathlib.Path] | None = None
    last_saved_time = 0.0

    # Cache the "Saved:" label width — it never changes
    _saved_label = "Saved:"
    _saved_label_w = cv2.getTextSize(_saved_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]

    copy_progress = CopyProgress()
    copy_thread: Optional[threading.Thread] = None

    save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    print("Press ESC to exit. Click a class, increment instrument, then Capture.\n")

    def on_mouse(event, x, y, _flags, _param):
        nonlocal prev_frame, last_frame, last_saved_paths, last_saved_time, copy_progress, copy_thread
        if event == cv2.EVENT_LBUTTONDOWN:
            action = ui.handle_click(x, y)

            if action == "capture" and last_frame is not None and prev_frame is not None:
                if last_luma >= prev_luma:
                    bright_frame, dark_frame = last_frame, prev_frame
                else:
                    bright_frame, dark_frame = prev_frame, last_frame
                dark_path, bright_path = ui.save_frames_async(dark_frame, bright_frame, save_executor)
                last_saved_paths = [bright_path, dark_path]
                last_saved_time = time.time()

            if action == "save_to_drive":
                # Prevent overlapping copy jobs
                if copy_progress.running:
                    copy_progress.toast_until = time.time() + 1.5
                    copy_progress.message = "Copy already in progress…"
                    return

                reset_progress(copy_progress)

                # Detect the inserted USB partition dynamically
                try:
                    device = find_usb_partition()
                except Exception as e:
                    copy_progress.error = str(e)
                    copy_progress.phase = "error"
                    copy_progress.toast_until = time.time() + 2.5
                    return

                mount_point = pathlib.Path("/mnt/usb")
                src_root = pathlib.Path(args.data_dir)
                dst_root = mount_point / "data_capture"

                copy_thread = start_async_copy_job(
                    progress=copy_progress,
                    device=device,
                    mount_point=mount_point,
                    src_root=src_root,
                    dst_root=dst_root,
                )

    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame")
                break

            frame_toggle = (frame_toggle + 1) % 2
            prev_luma = last_luma
            last_luma = float(frame[::4, ::4, 1].mean())  # green-channel subsample, no alloc
            prev_frame = last_frame
            last_frame = frame.copy()
            if frame_toggle == 0:
                continue

            ui.draw(frame)

            # Center overlay for copy progress/errors/finished toast
            ui.draw_copy_overlay(frame, copy_progress)

            # Draw last saved paths in the bottom right, above the button bar
            if last_saved_paths is not None and (time.time() - last_saved_time) < 1.5:
                button_bar_h = max(90, int(DISPLAY_H * 0.16))
                x = DISPLAY_W - _saved_label_w - 20
                y = DISPLAY_H - button_bar_h + 30
                cv2.putText(
                    frame,
                    _saved_label,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                for idx, path in enumerate(last_saved_paths):
                    path_text = path.name
                    path_width = cv2.getTextSize(path_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
                    px = DISPLAY_W - path_width - 20
                    py = y + (idx + 1) * 28
                    cv2.putText(
                        frame,
                        path_text,
                        (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        save_executor.shutdown(wait=True)
        lens_controller.stop()
        lens_controller.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        print("\nCapture session ended.")


if __name__ == "__main__":
    main()