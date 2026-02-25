import time
from threading import Event, Thread


class LensController(Thread):
    def __init__(self, interval_s: float, target_focus_mm: float, temp_path: str):
        super().__init__(daemon=True, name="LensController")
        self.interval_s = interval_s
        self.target_focus_mm = target_focus_mm
        self.temp_path = temp_path
        self._stop_event = Event()
        self._lens_available = False
        self._focus_calibrator = None

        try:
            from llens.lens import Lens
            from llens.focus import Focus, SafePolyError
        except Exception as exc:
            print(f"Lens support disabled: {exc}")
            return

        self._Lens = Lens
        self._SafePolyError = SafePolyError
        self._focus_calibrator = Focus(var_names=("temp", "dist"))
        self._lens_available = True

    def stop(self) -> None:
        self._stop_event.set()

    def adjust_once(self) -> None:
        if not self._lens_available:
            return
        try:
            temp_c = self._read_temperature()
            with self._Lens() as lens:
                bits = self._focus_calibrator.estimate_focus(
                    temp=temp_c,
                    dist=self.target_focus_mm,
                )
                lens.set_focus(bits)
        except self._SafePolyError as exc:
            print(f"Lens calibration error: {exc}")
        except Exception as exc:
            print(f"Lens operation error: {exc}")

    def run(self) -> None:
        if not self._lens_available:
            return
        while not self._stop_event.is_set():
            self.adjust_once()
            time.sleep(self.interval_s)

    def _read_temperature(self) -> float:
        with open(self.temp_path, "r", encoding="utf-8") as handle:
            temp_milli_c = int(handle.read().strip())
        return temp_milli_c / 1000.0
