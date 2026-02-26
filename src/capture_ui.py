# capture_ui.py
import concurrent.futures
import contextlib
import datetime
import hashlib
import json
import pathlib
import time
import uuid

import cv2
import numpy as np


def _fmt_bytes(n: int) -> str:
    n_f = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_f < 1024:
            return f"{n_f:.0f}{unit}" if unit == "B" else f"{n_f:.1f}{unit}"
        n_f /= 1024
    return f"{n_f:.1f}PB"


class CaptureUI:
    def __init__(self, class_names: list[str], data_dir: str, meta_data_path: str, display_w: int):
        self.class_names = class_names
        self.selected_idx = 0
        self.instance_id = uuid.uuid4()
        self.capture_id = 0
        self.instance_counter = 0
        self.data_dir = pathlib.Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.display_w = display_w
        self.buttons: list[dict[str, int | str]] = []
        self._scratch = None  # reusable overlay buffer (np.ndarray once allocated)

        self.station_metadata = self._load_metadata(pathlib.Path(meta_data_path))

    def layout(self, w: int, h: int) -> None:
        margin = 12
        class_gap = 6
        top_bar_h = max(70, int(h * 0.12))
        bottom_bar_h = max(90, int(h * 0.16))
        class_btn_h = top_bar_h - 2 * margin
        action_btn_h = bottom_bar_h - 2 * margin
        top_y = margin
        bottom_y = h - bottom_bar_h + margin
        x = margin
        self.buttons = []

        total_w = max(self.display_w, w) - 2 * margin
        class_count = max(1, len(self.class_names))
        class_btn_w = int((total_w - class_gap * (class_count - 1)) / class_count)

        for idx, name in enumerate(self.class_names):
            btn_w = class_btn_w
            self.buttons.append(
                {
                    "x": x,
                    "y": top_y,
                    "w": btn_w,
                    "h": class_btn_h,
                    "type": "class",
                    "idx": idx,
                }
            )
            x += btn_w + class_gap

        self.buttons.append(
            {"x": margin, "y": bottom_y, "w": 180, "h": action_btn_h, "type": "counter"}
        )
        self.buttons.append(
            {
                "x": margin + 180 + margin,
                "y": bottom_y,
                "w": 180,
                "h": action_btn_h,
                "type": "capture",
            }
        )
        self.buttons.append(
            {
                "x": margin + 180 + margin + 180 + margin,
                "y": bottom_y,
                "w": 220,
                "h": action_btn_h,
                "type": "save_to_drive",
            }
        )

    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if not self.buttons:
            self.layout(w, h)

        top_bar_h = max(70, int(h * 0.12))
        bottom_bar_h = max(90, int(h * 0.16))
        if self._scratch is None or self._scratch.shape != frame.shape:
            self._scratch = np.empty_like(frame)
        np.copyto(self._scratch, frame)
        cv2.rectangle(self._scratch, (0, 0), (w, top_bar_h), (10, 10, 10), -1)
        cv2.rectangle(self._scratch, (0, h - bottom_bar_h), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(self._scratch, 0.65, frame, 0.35, 0, frame)

        for btn in self.buttons:
            x = int(btn["x"])
            y = int(btn["y"])
            bw = int(btn["w"])
            bh = int(btn["h"])
            btype = btn["type"]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2

            if btype == "class":
                idx = int(btn["idx"])
                is_selected = idx == self.selected_idx
                color = (0, 200, 255) if is_selected else (50, 50, 50)
                text = self.class_names[idx]
            elif btype == "counter":
                color = (50, 50, 50)
                text = f"Instrument: {self.instance_counter}"
            elif btype == "capture":
                color = (0, 150, 0)
                text = "Capture"
            elif btype == "save_to_drive":
                color = (0, 150, 0)
                text = "Save to Drive"
            else:
                color = (50, 50, 50)
                text = ""

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, -1)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 255), 2)
            if btype == "class":
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = x + (bw - text_w) // 2
                text_y = y + (bh + text_h) // 2
            else:
                text_x = x + 12
                text_y = y + int(bh * 0.65)

            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    def handle_click(self, x: int, y: int) -> str | None:
        for btn in self.buttons:
            bx = int(btn["x"])
            by = int(btn["y"])
            bw = int(btn["w"])
            bh = int(btn["h"])
            if bx <= x <= bx + bw and by <= y <= by + bh:
                btype = btn["type"]
                if btype == "class":
                    self.selected_idx = int(btn["idx"])
                    return "class"
                if btype == "counter":
                    self.instance_id = uuid.uuid4()
                    self.instance_counter += 1
                    self.capture_id = 0
                    return "counter"
                if btype == "capture":
                    return "capture"
                if btype == "save_to_drive":
                    return "save_to_drive"
        return None

    def save_frames_async(
        self,
        dark_frame: np.ndarray,
        bright_frame: np.ndarray,
        executor: concurrent.futures.Executor,
    ) -> tuple[pathlib.Path, pathlib.Path]:
        """Return paths immediately and perform disk I/O on the executor thread."""
        class_name = self.class_names[self.selected_idx]
        class_dir = self.data_dir / class_name
        capture_id = self.capture_id
        instance_id = self.instance_id
        self.capture_id += 1

        dark_frame_id = f"{capture_id}_dark"
        bright_frame_id = f"{capture_id}_bright"
        dark_path = class_dir / f"{instance_id}_{dark_frame_id}.jpg"
        bright_path = class_dir / f"{instance_id}_{bright_frame_id}.jpg"

        def _do_io() -> None:
            class_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dark_path), dark_frame)
            self._save_metadata(dark_frame, class_name, dark_frame_id, "dark", dark_path)
            cv2.imwrite(str(bright_path), bright_frame)
            self._save_metadata(bright_frame, class_name, bright_frame_id, "bright", bright_path)

        executor.submit(_do_io)
        return dark_path, bright_path

    def _save_metadata(
        self,
        frame: np.ndarray,
        class_name: str,
        frame_id: str,
        illumination: str,
        path: pathlib.Path,
    ) -> pathlib.Path:
        utc_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
        class_dir = self.data_dir / class_name
        sha256_hash = hashlib.sha256(frame.tobytes()).hexdigest()
        meta_name = f"{self.instance_id}_{frame_id}_meta.json"
        meta_path = class_dir / meta_name
        payload = {
            "frame_id": str(frame_id),
            "capture_id": str(self.capture_id),
            "instrument_id": str(self.instance_id),
            "illumination": illumination,
            "timestamp": utc_timestamp,
            "file": path.name,
            "width": frame.shape[1],
            "height": frame.shape[0],
            "sha256": sha256_hash,
            "label": {
                "class": class_name,
                "source": self.station_metadata.get("source", "unknown"),
                "review_status": "new",
            },
            "station_id": self.station_metadata.get("station_id", "unknown"),
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return meta_path

    def _load_metadata(self, meta_path: pathlib.Path) -> dict:
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    # ----------------------------
    # Copy progress overlay
    # ----------------------------
    def draw_copy_overlay(self, frame: np.ndarray, progress) -> None:
        """
        Draw a centered overlay when copying is running, and a short toast when done/error.

        Expects an object with fields similar to:
          running, ok, phase, message, error,
          total_bytes, copied_bytes, total_files, copied_files,
          current_file, toast_until
        """
        # Take an atomic snapshot of all progress fields before rendering
        lock = getattr(progress, "lock", None)
        with (lock if lock is not None else contextlib.nullcontext()):
            running = bool(getattr(progress, "running", False))
            ok = bool(getattr(progress, "ok", False))
            phase = str(getattr(progress, "phase", ""))
            msg = str(getattr(progress, "message", ""))
            err = str(getattr(progress, "error", ""))
            total_b = int(getattr(progress, "total_bytes", 0) or 0)
            done_b = int(getattr(progress, "copied_bytes", 0) or 0)
            total_f = int(getattr(progress, "total_files", 0) or 0)
            done_f = int(getattr(progress, "copied_files", 0) or 0)
            cur = str(getattr(progress, "current_file", ""))
            toast_until = float(getattr(progress, "toast_until", 0.0))

        now = time.time()
        show_toast = toast_until > now
        show_full = running

        if not show_full and not show_toast:
            return

        h, w = frame.shape[:2]
        if self._scratch is None or self._scratch.shape != frame.shape:
            self._scratch = np.empty_like(frame)

        # Center box
        box_w = int(w * 0.62)
        box_h = int(h * 0.22)
        x0 = (w - box_w) // 2
        y0 = (h - box_h) // 2
        x1 = x0 + box_w
        y1 = y0 + box_h

        # Dim background — copy frame into scratch, fill with black, blend back
        np.copyto(self._scratch, frame)
        cv2.rectangle(self._scratch, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(self._scratch, 0.25, frame, 0.75, 0, frame)

        # Box background — reuse scratch (still all-black), fill box region and blend in-place
        self._scratch[y0:y1, x0:x1] = (20, 20, 20)
        cv2.addWeighted(self._scratch[y0:y1, x0:x1], 0.75, frame[y0:y1, x0:x1], 0.25, 0, frame[y0:y1, x0:x1])

        # Border
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Title
        if running:
            title = "Saving to USB…"
        else:
            title = "Finished" if ok and not err else "Error"

        cv2.putText(frame, title, (x0 + 18, y0 + 40), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if err:
            err_short = err.strip().replace("\n", " ")
            if len(err_short) > 90:
                err_short = err_short[:90] + "…"
            cv2.putText(frame, err_short, (x0 + 18, y0 + 75), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            line = f"{phase}: {msg}".strip(": ")
            cv2.putText(frame, line, (x0 + 18, y0 + 75), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        ratio = (done_b / total_b) if total_b > 0 else (done_f / total_f if total_f > 0 else 0.0)
        ratio = max(0.0, min(1.0, ratio))

        if cur:
            cur_short = cur.replace("\n", " ")
            if len(cur_short) > 70:
                cur_short = "…" + cur_short[-69:]
            cv2.putText(frame, f"File: {cur_short}", (x0 + 18, y0 + 105), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if total_b > 0:
            p_line = f"{int(ratio * 100)}%  ({_fmt_bytes(done_b)} / {_fmt_bytes(total_b)})   Files: {done_f}/{total_f}"
        else:
            p_line = f"{int(ratio * 100)}%   Files: {done_f}/{total_f}"

        cv2.putText(frame, p_line, (x0 + 18, y0 + 135), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Progress bar
        bar_x0 = x0 + 18
        bar_y0 = y0 + box_h - 42
        bar_x1 = x1 - 18
        bar_y1 = bar_y0 + 18

        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x1, bar_y1), (255, 255, 255), 2)
        fill_w = int((bar_x1 - bar_x0 - 4) * ratio)
        if fill_w > 0:
            cv2.rectangle(
                frame,
                (bar_x0 + 2, bar_y0 + 2),
                (bar_x0 + 2 + fill_w, bar_y1 - 2),
                (0, 200, 255),
                -1,
            )