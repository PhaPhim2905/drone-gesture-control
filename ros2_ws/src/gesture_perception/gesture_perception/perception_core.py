"""
PERCEPTION_CORE — toàn bộ chuỗi nhận diện của MỘT frame, KHÔNG dính ROS.

    frame BGR -> MediaPipe Pose -> BODY25 pixel -> One-Euro -> PCK -> TFLite

perception_node chỉ là lớp vỏ: mở camera, gọi process(), đóng gói message.
Tách như vậy để:
  - test được trên laptop Windows không có ROS (test/test_perception_core.py)
  - doctor.py đo FPS thật trên Pi mà không cần khởi động ROS
"""

import time
from dataclasses import dataclass, field

import numpy as np

from . import pck_format as pf
from . import settings as S
from .classifier import GestureClassifier
from .one_euro import Body25Filter


@dataclass
class FrameResult:
    label: str = S.NO_OPERATOR
    confidence: float = 0.0
    probs: np.ndarray = None
    kp_raw: np.ndarray = None          # (25, 3) pixel, trước lọc
    kp_norm: np.ndarray = None         # (25, 2) sau lọc + chuẩn hoá PCK
    landmarks: object = None           # để vẽ
    pose_ms: float = 0.0
    clf_ms: float = 0.0
    top_labels: list = field(default_factory=list)
    top_probs: list = field(default_factory=list)


class GesturePerception:
    def __init__(self, model_path, one_euro=True, tau_override=None,
                 complexity=S.POSE_MODEL_COMPLEXITY, detect_conf=S.POSE_DETECT_CONF,
                 track_conf=S.POSE_TRACK_CONF, min_visibility=S.LM_MIN_VISIBILITY,
                 flip=S.FLIP_FRAME):
        self.clf = GestureClassifier(model_path, tau_override)
        self.filter = Body25Filter()
        self.one_euro = bool(one_euro)
        self.flip = bool(flip)
        self.min_visibility = float(min_visibility)
        self._pose_args = dict(
            model_complexity=int(complexity),
            smooth_landmarks=S.POSE_SMOOTH_LANDMARKS,
            min_detection_confidence=float(detect_conf),
            min_tracking_confidence=float(track_conf),
        )
        self._pose = None

    # ---- MediaPipe mở muộn: test không có camera vẫn tạo được object -----
    def _pose_obj(self):
        if self._pose is None:
            import mediapipe as mp
            self._mp = mp
            self._pose = mp.solutions.pose.Pose(**self._pose_args)
        return self._pose

    def set_one_euro(self, enabled):
        self.one_euro = bool(enabled)
        self.filter.reset()

    def classify_body25(self, kp, t):
        """(25, 3) pixel -> FrameResult. Nửa sau của chuỗi, không cần MediaPipe."""
        r = FrameResult(kp_raw=kp)
        if self.one_euro:
            kp = self.filter(kp, t)
        # strict=False: lúc bay thà phân loại một tư thế hơi lệch chuẩn hoá còn
        # hơn im lặng. Loại mẫu chỉ đúng ở khâu DỰNG TẬP HUẤN LUYỆN.
        n = pf.normalize_body25(kp, strict=False)
        if n is None:
            self.filter.reset()
            return r
        t0 = time.perf_counter()
        r.label, r.confidence, r.probs = self.clf.predict(n)
        r.clf_ms = (time.perf_counter() - t0) * 1000
        r.kp_norm = n
        r.top_labels, r.top_probs = self.clf.top(r.probs)
        return r

    def process(self, frame_bgr, t):
        """frame BGR (đã lật nếu cần) -> FrameResult."""
        import cv2
        pose = self._pose_obj()
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        res = pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        pose_ms = (time.perf_counter() - t0) * 1000
        if not res.pose_landmarks:
            self.filter.reset()
            return FrameResult(pose_ms=pose_ms)
        kp = pf.mediapipe_to_body25(res.pose_landmarks.landmark, w, h,
                                    self.min_visibility)
        r = self.classify_body25(kp, t)
        r.landmarks = res.pose_landmarks
        r.pose_ms = pose_ms
        return r

    def draw(self, frame, r, fps=0.0, extra=""):
        """Vẽ khung xương + nhãn + top-3 lên frame (tại chỗ)."""
        import cv2
        if r.landmarks is not None and self._pose is not None:
            self._mp.solutions.drawing_utils.draw_landmarks(
                frame, r.landmarks, self._mp.solutions.pose.POSE_CONNECTIONS)
        col = (0, 255, 255) if r.label not in (S.NO_OPERATOR, S.UNKNOWN_GESTURE) \
            else (0, 0, 255)
        cv2.putText(frame, f"{r.label} {r.confidence*100:.0f}%", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        for i, (lb, p) in enumerate(zip(r.top_labels, r.top_probs)):
            cv2.putText(frame, f"{lb[:16]:<16} {p*100:5.1f}%", (12, 62 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"{fps:4.1f} fps  pose {r.pose_ms:4.0f} ms  "
                    f"one-euro {'on' if self.one_euro else 'off'}",
                    (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 220, 150), 1)
        if extra:
            cv2.putText(frame, extra, (12, frame.shape[0] - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        return frame

    def close(self):
        if self._pose is not None:
            self._pose.close()
            self._pose = None
