import pathlib
import time

import cv2
import numpy as np


class CaptureUI:
    def __init__(self, class_names: list[str], data_dir: str, display_w: int):
        self.class_names = class_names
        self.selected_idx = 0
        self.instance_id = 1
        self.instance_image_id = 1
        self.data_dir = pathlib.Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.display_w = display_w
        self.buttons: list[dict[str, int | str]] = []

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

    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if not self.buttons:
            self.layout(w, h)

        top_bar_h = max(70, int(h * 0.12))
        bottom_bar_h = max(90, int(h * 0.16))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (10, 10, 10), -1)
        cv2.rectangle(overlay, (0, h - bottom_bar_h), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

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
                text = f"Instrument: {self.instance_id}"
            else:
                color = (0, 150, 0)
                text = "Capture"

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
                    self.instance_id += 1
                    self.instance_image_id = 1
                    return "counter"
                if btype == "capture":
                    return "capture"
        return None

    def save_frames(self, dark_frame: np.ndarray, bright_frame: np.ndarray) -> tuple[pathlib.Path, pathlib.Path]:
        dark_path = self._save_frame(dark_frame, "dark")
        bright_path = self._save_frame(bright_frame, "bright")
        self.instance_image_id += 1
        return dark_path, bright_path

    def _save_frame(self, frame: np.ndarray, suffix: str | None = None) -> pathlib.Path:
        class_name = self.class_names[self.selected_idx]
        stamp = time.strftime("%Y%m%d")
        class_dir = self.data_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        if suffix:
            filename = f"{stamp}_inst{self.instance_id:04d}_img{self.instance_image_id:04d}_{suffix}.jpg"
        else:
            filename = f"{stamp}_inst{self.instance_id:04d}_img{self.instance_image_id:04d}.jpg"
        out_path = class_dir / filename
        cv2.imwrite(str(out_path), frame)
        return out_path
