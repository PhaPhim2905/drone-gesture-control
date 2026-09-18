"""
MODE_GATE — chỉ NHẬN DIỆN khi pilot đã gạt drone sang GUIDED.

    Pilot gạt công tắc sang GUIDED  = pilot trao quyền cho cử chỉ
    Mode khác (LOITER, ALT_HOLD, LAND, ...) = camera vẫn mở nhưng KHÔNG chạy
    MediaPipe, KHÔNG phát cử chỉ nào.

Được hai thứ:
  1. An toàn: người đi ngang qua camera lúc pilot đang lái tay không thể tạo lệnh.
  2. Pi 5 nhàn: MediaPipe là khâu tốn CPU nhất, chỉ chạy khi cần.

Không chắc thì KHÔNG nhận diện: chưa từng nhận /mavros/state, mất state quá
state_timeout giây, FCU chưa kết nối -> đóng cổng.

Không dính ROS: perception_node chuyển /mavros/state vào on_state().
"""


class ModeGate:
    def __init__(self, require_mode="GUIDED", state_timeout=2.0):
        self.require_mode = (require_mode or "").strip()
        self.state_timeout = float(state_timeout)
        self._t = None
        self._connected = False
        self._mode = ""

    @property
    def enabled(self):
        return bool(self.require_mode)

    def on_state(self, connected, mode, t):
        self._connected = bool(connected)
        self._mode = mode or ""
        self._t = t

    def check(self, t):
        """-> (được nhận diện?, lý do nếu không)."""
        if not self.enabled:
            return True, ""
        if self._t is None:
            return False, "chua nhan /mavros/state"
        if t - self._t > self.state_timeout:
            return False, f"mat /mavros/state {t - self._t:.1f}s"
        if not self._connected:
            return False, "FCU chua ket noi"
        if self._mode != self.require_mode:
            return False, f"dang {self._mode or '?'}, cho pilot gat {self.require_mode}"
        return True, ""
