"""
FRAME_SOURCE — luôn đưa cho bộ nhận diện FRAME MỚI NHẤT, bỏ frame cũ.

VÌ SAO
    Camera ra 30 fps, MediaPipe trên Pi 5 chỉ xử lý được ~10-20 fps. Nếu đọc
    camera và xử lý trong CÙNG một vòng lặp, mỗi lần cap.read() trả về frame đã
    nằm trong bộ đệm từ lúc xử lý frame trước -> người điều khiển hạ tay mà drone
    còn "thấy" tay giơ thêm vài trăm ms. BUFFERSIZE=1 không đảm bảo trên mọi
    driver V4L2.

    Ở đây một thread riêng đọc camera liên tục và chỉ giữ frame cuối. Vòng xử lý
    lấy frame mới nhất, frame nào bị ghi đè trước khi kịp xử lý thì đếm vào
    `dropped`. Độ trễ = tuổi frame lúc bắt đầu xử lý + thời gian xử lý, không cộng
    dồn theo thời gian.

HAI NGUỒN
    camera     cv2.VideoCapture đã mở (camera_util.open_camera)
    video      danh sách file, PHÁT THEO NHỊP THẬT của video (fps trong file).
               Frame vẫn bị bỏ nếu xử lý chậm -> đo đúng cảm giác trên Pi,
               khác với chấm điểm từng frame trong training/7_replay_check.py.

KHÔNG dính ROS: test được trên Windows.
"""

import threading
import time
from pathlib import Path


class LatestFrame:
    def __init__(self, cap=None, video_paths=(), loop=True, stamp_fn=None):
        """
        cap: VideoCapture đã mở (camera). Hoặc để None và truyền video_paths.
        stamp_fn: gọi ngay khi có frame, kết quả trả kèm frame (vd đồng hồ ROS).
        """
        if cap is None and not video_paths:
            raise ValueError("can cap hoac video_paths")
        self._cap = cap
        self._videos = [str(p) for p in video_paths]
        self._loop = bool(loop)
        self._stamp_fn = stamp_fn or (lambda: None)
        self._cv = threading.Condition()
        self._item = None          # (frame, t_mono, stamp, seq, source_name)
        self._seq = 0
        self._consumed_seq = 0
        self.grabbed = 0
        self.dropped = 0
        self.ended = False
        self.source_name = "camera" if cap is not None else ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---- thread đọc --------------------------------------------------------
    def _publish(self, frame, name):
        t = time.monotonic()
        stamp = self._stamp_fn()
        with self._cv:
            if self._seq > self._consumed_seq:
                self.dropped += 1          # frame trước chưa ai lấy -> bị bỏ
            self._seq += 1
            self.grabbed += 1
            self._item = (frame, t, stamp, self._seq, name)
            self._cv.notify_all()

    def _run(self):
        if self._cap is not None:
            self._run_camera()
        else:
            self._run_videos()
        with self._cv:
            self.ended = True
            self._cv.notify_all()

    def _run_camera(self):
        fails = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 50:             # ~2.5 s không có hình
                    return
                time.sleep(0.05)
                continue
            fails = 0
            self._publish(frame, "camera")

    def _run_videos(self):
        import cv2
        while not self._stop.is_set():
            for path in self._videos:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS)
                period = 1.0 / fps if 1.0 < fps < 240.0 else 1.0 / 30.0
                name = Path(path).name
                self.source_name = name
                next_t = time.monotonic()
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # phát theo nhịp thật; chậm quá thì bắt kịp, không tua nhanh
                    now = time.monotonic()
                    if next_t > now:
                        time.sleep(next_t - now)
                    next_t = max(next_t + period, time.monotonic() - period)
                    self._publish(frame, name)
                cap.release()
                if self._stop.is_set():
                    return
            if not self._loop:
                return

    # ---- phía xử lý --------------------------------------------------------
    def get(self, timeout=1.0):
        """
        Chờ frame MỚI hơn frame đã lấy lần trước.
        -> (frame, t_mono, stamp, seq, source_name) hoặc None (hết giờ / nguồn đã hết).
        """
        end = time.monotonic() + timeout
        with self._cv:
            while self._seq <= self._consumed_seq and not self.ended and not self._stop.is_set():
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self._cv.wait(left)
            if self._seq <= self._consumed_seq:
                return None
            self._consumed_seq = self._seq
            return self._item

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
