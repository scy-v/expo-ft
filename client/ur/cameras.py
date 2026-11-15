"""RGB RealSense capture without a dependency on a particular LeRobot release."""
import threading


class RealSenseCamera:
    def __init__(self, serial, fps, width, height):
        self.serial, self.fps, self.width, self.height = str(serial), fps, width, height
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._frame = self._error = self._pipeline = self._thread = None

    def connect(self):
        import pyrealsense2 as rs
        import numpy as np
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self._pipeline.start(cfg)

        def capture():
            try:
                while not self._stop.is_set():
                    frame = self._pipeline.wait_for_frames(2000).get_color_frame()
                    if frame:
                        with self._condition:
                            self._frame = np.asanyarray(frame.get_data()).copy()
                            self._condition.notify_all()
            except Exception as error:
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
        self._thread = threading.Thread(target=capture, daemon=True)
        self._thread.start()

    def async_read(self, **kwargs):
        with self._condition:
            if not self._condition.wait_for(lambda: self._frame is not None or self._error, 3):
                raise TimeoutError(f"No RGB image from camera {self.serial}")
            if self._error:
                raise RuntimeError(f"Camera {self.serial} failed") from self._error
            return self._frame.copy()

    read = async_read

    def disconnect(self):
        self._stop.set()
        if self._pipeline is not None:
            self._pipeline.stop()
        if self._thread is not None:
            self._thread.join(timeout=3)
