"""
COMMAND_LOGIC — cử chỉ theo từng frame  ->  lệnh drone an toàn.

KHÔNG import ROS. command_node chỉ chuyển message vào đây và chuyển kết quả ra
MAVROS. Toàn bộ quyết định an toàn nằm trong file này để test được trên laptop:

    python ros2_ws/src/gesture_command/test/test_command_logic.py

──────────────────────────────────────────────────────────────────────────────
NGUYÊN TẮC (chốt 2026-09-10, xem memory "lệnh không rút lại được")

  1. Thao tác KHÔNG RÚT LẠI ĐƯỢC nằm trên tay cầm pilot, không nằm sau camera.
       - file này KHÔNG BAO GIỜ tạo lệnh ARM / DISARM. Test kiểm cả mã nguồn.
       - STOP = đứng yên + khoá + trả LOITER. KHÔNG phải Motor Emergency Stop.
  2. DỪNG thì NGAY, ĐI thì phải GIỮ. Đổi sang bất kỳ cử chỉ nào khác lệnh đang
     chạy -> vận tốc về 0 ngay frame đó. Lệnh mới chỉ áp sau thời gian giữ.
  3. Không chắc thì đứng yên: AMBIGUOUS, NEGATIVE, NO_OPERATOR, cử chỉ lạ đều
     là HOVER.
  4. Pilot gạt mode ra khỏi GUIDED = pilot lấy lại quyền -> khoá, ngừng gửi.

──────────────────────────────────────────────────────────────────────────────
THỜI GIAN GIỮ — đo ngày 2026-09-09 trên 3.765 mẫu / 31 take, mô phỏng luật phát
lệnh trên dự đoán out-of-fold:

    giữ (s)   lệnh phát   lệnh SAI   take bắt được
      0.62        1825          4       23/26
      0.99        1624          0       23/26     <- chọn 1.0 s
      1.23        1513          0       22/26

    STOP 0.6 s: cố ý ngắn hơn. STOP nhầm chỉ làm drone đứng yên, STOP muộn thì
    không cứu được gì.

Thay cho k_vote = 8 frame của model: 8 frame ở 8 fps = 1.0 s, nhưng ở 15 fps
chỉ còn 0.53 s. Tính theo giây thì Pi nhanh hay chậm đều cùng một cảm giác.
"""

from dataclasses import dataclass, field

# ---- nhãn đặc biệt (khớp gesture_perception/settings.py) --------------------
UNKNOWN_GESTURE = "AMBIGUOUS"
NO_OPERATOR = "NO_OPERATOR"

# ---- mavros_msgs/ExtendedState.landed_state ---------------------------------
LANDED_UNDEFINED, LANDED_ON_GROUND, LANDED_IN_AIR, LANDED_TAKEOFF, LANDED_LANDING = range(5)

LOCKED, ENABLED = "LOCKED", "ENABLED"

# Cử chỉ nhận được cả khi đang KHOÁ. Còn lại chỉ nghe khi ENABLED.
GESTURE_ALWAYS_ON = ("ASSUME_GUIDANCE", "STOP", "LAND")

# Cử chỉ giữ tư thế = lệnh DUY TRÌ. Hạ tay là dừng.
CONTINUOUS = ("HOVER", "ROLL_LEFT", "ROLL_RIGHT")


def gesture_command_map(mirror_left_right=False):
    """
    Cử chỉ -> tên lệnh. LEFT / RIGHT luôn là hướng CỦA DRONE (hệ thân máy bay).

    QUY ƯỚC ĐÃ CHỐT 2026-09-14 (mirror_left_right = False): tên cử chỉ tính theo
    hướng CỦA DRONE, drone đứng đối diện người điều khiển.
        ROLL_RIGHT -> drone sang phải CỦA NÓ  = sang TRÁI của người điều khiển
        ROLL_LEFT  -> drone sang trái CỦA NÓ  = sang PHẢI của người điều khiển
    Tay dang ngang chỉ về phía drone đi, nhìn từ người điều khiển.

    mirror_left_right = True đảo lại (tên theo hướng người điều khiển). Chỉ bật
    khi đã QUAY LẠI dữ liệu theo quy ước đó.
    """
    return {
        "ASSUME_GUIDANCE": "ENABLE",
        "HOVER": "HOVER",
        "TAKEOFF": "TAKEOFF",
        "LAND": "LAND",
        "STOP": "STOP",
        "NEGATIVE": "HOVER",
        "ROLL_RIGHT": "LEFT" if mirror_left_right else "RIGHT",
        "ROLL_LEFT": "RIGHT" if mirror_left_right else "LEFT",
    }


def velocity_frd(command, speed_xy):
    """
    Lệnh -> (vx, vy, vz) m/s, hệ thân máy bay FRD (x tiến, y PHẢI, z XUỐNG).
    Đây là quy ước MAVLink/ArduPilot, giống config.py cũ.
    """
    return {
        "RIGHT": (0.0, +speed_xy, 0.0),
        "LEFT": (0.0, -speed_xy, 0.0),
    }.get(command, (0.0, 0.0, 0.0))


def frd_to_flu(v):
    """
    FRD (MAVLink) -> FLU (ROS REP-103).

    ⚠️ CHỖ ĐỔI DẤU NGUY HIỂM NHẤT CỦA BẢN ROS 2. MAVROS nhận FLU rồi TỰ đổi sang
    FRD khi coordinate_frame = FRAME_BODY_NED. Gửi thẳng số FRD vào MAVROS là
    drone bay NGƯỢC chiều ngang và NGƯỢC chiều cao.
        LEFT  FRD (0, -1, 0)  ->  FLU (0, +1, 0)   y FLU dương = trái
    """
    return (v[0], -v[1], -v[2])


@dataclass
class Params:
    control_mode: str = "GUIDED"
    handback_mode: str = "LOITER"
    speed_xy: float = 0.5
    takeoff_alt: float = 2.0
    hold_sec: float = 1.0
    hold_override: dict = field(default_factory=lambda: {
        "ASSUME_GUIDANCE": 1.0, "LAND": 0.8, "STOP": 0.6})
    watchdog_sec: float = 0.5        # không có message cử chỉ mới -> HOVER
    operator_lost_sec: float = 0.8   # NO_OPERATOR liên tục -> khoá
    mode_grace_sec: float = 3.0      # chờ FCU đổi mode sau khi mình yêu cầu
    request_guided_on_enable: bool = False
    mirror_left_right: bool = False   # tên ROLL theo hướng CỦA DRONE

    def hold_for(self, gesture):
        return float(self.hold_override.get(gesture, self.hold_sec))


@dataclass
class Vehicle:
    connected: bool = False
    armed: bool = False
    mode: str = ""
    landed_state: int = LANDED_UNDEFINED


@dataclass
class Action:
    kind: str            # "set_mode" | "takeoff" | "event"
    value: object = None
    reason: str = ""


class CommandLogic:
    def __init__(self, params=None):
        self.p = params or Params()
        self.cmd_map = gesture_command_map(self.p.mirror_left_right)
        self.state = LOCKED
        self.candidate = NO_OPERATOR
        self.cand_since = 0.0
        self.cand_fired = False
        self.motion = "HOVER"
        self.last_gesture_t = None
        self.no_operator_since = None
        self.mode_request_t = None
        self.was_in_control_mode = False
        self.last_event = ""
        self.blocked_reason = "LOCKED - gio ASSUME_GUIDANCE"

    # ---- tiện ích ---------------------------------------------------------
    def _event(self, text, t):
        self.last_event = text
        return Action("event", text)

    def _lock(self, why, t):
        self.state = LOCKED
        self.motion = "HOVER"
        self.was_in_control_mode = False
        return self._event(f"KHOA: {why}", t)

    def hold_progress(self, t):
        h = self.p.hold_for(self.candidate)
        return 0.0 if h <= 0 else max(0.0, min(1.0, (t - self.cand_since) / h))

    # ---- đầu vào 1: một frame cử chỉ -------------------------------------
    def on_gesture(self, label, t, v):
        """Nạp một quan sát. -> danh sách Action cần thực hiện ngay."""
        acts = []
        self.last_gesture_t = t

        # Mất người quá lâu khi đang điều khiển -> khoá.
        if label == NO_OPERATOR:
            if self.no_operator_since is None:
                self.no_operator_since = t
            elif self.state == ENABLED and t - self.no_operator_since >= self.p.operator_lost_sec:
                acts.append(self._lock(f"mat nguoi {self.p.operator_lost_sec:.1f}s", t))
        else:
            self.no_operator_since = None

        if label != self.candidate:
            self.candidate = label
            self.cand_since = t
            self.cand_fired = False
            # NGUYÊN TẮC 2: tư thế đổi -> dừng NGAY, không chờ giữ.
            self.motion = "HOVER"

        if self.cand_fired or t - self.cand_since < self.p.hold_for(label):
            return acts
        if label not in self.cmd_map:          # AMBIGUOUS, NO_OPERATOR, lạ
            return acts
        if self.state == LOCKED and label not in GESTURE_ALWAYS_ON:
            return acts

        self.cand_fired = True
        acts.extend(self._confirmed(label, t, v))
        return acts

    def _confirmed(self, label, t, v):
        cmd = self.cmd_map[label]
        acts = []

        if cmd == "ENABLE":
            if self.state == ENABLED:
                return acts
            if not v.connected:
                acts.append(self._event("ASSUME_GUIDANCE bo qua: chua ket noi FCU", t))
                return acts
            self.state = ENABLED
            self.was_in_control_mode = v.mode == self.p.control_mode
            acts.append(self._event("MO QUYEN DIEU KHIEN", t))
            if self.p.request_guided_on_enable and v.armed and v.mode != self.p.control_mode:
                self.mode_request_t = t
                acts.append(Action("set_mode", self.p.control_mode, "ASSUME_GUIDANCE"))
            return acts

        if cmd == "STOP":
            self.motion = "HOVER"
            if self.state == ENABLED:
                acts.append(self._lock("STOP", t))
                # Chỉ trả mode khi drone đang ở mode của mình. Pilot đã gạt sang
                # mode khác thì không giành lại.
                if v.armed and v.mode == self.p.control_mode:
                    acts.append(Action("set_mode", self.p.handback_mode, "STOP"))
            return acts

        if cmd == "LAND":
            if not v.armed or v.landed_state == LANDED_ON_GROUND:
                acts.append(self._event("LAND bo qua: drone dang o duoi dat", t))
                return acts
            if self.state == ENABLED:
                acts.append(self._lock("LAND", t))
            acts.append(Action("set_mode", "LAND", "LAND"))
            acts.append(self._event("HA CANH", t))
            return acts

        if cmd == "TAKEOFF":
            why = None
            if not v.armed:
                why = "chua ARM - pilot phai arm bang tay cam"
            elif v.mode != self.p.control_mode:
                why = f"dang {v.mode}, can {self.p.control_mode}"
            elif v.landed_state == LANDED_UNDEFINED:
                why = "chua biet duoi dat hay dang bay (FCU chua gui EXTENDED_SYS_STATE)"
            elif v.landed_state != LANDED_ON_GROUND:
                why = "dang bay roi"
            if why:
                acts.append(self._event(f"TAKEOFF bo qua: {why}", t))
                return acts
            acts.append(Action("takeoff", self.p.takeoff_alt, "TAKEOFF"))
            acts.append(self._event(f"CAT CANH {self.p.takeoff_alt:.1f} m", t))
            return acts

        if label in CONTINUOUS:
            self.motion = cmd
        return acts

    # ---- đầu vào 2: nhịp điều khiển --------------------------------------
    def tick(self, t, v):
        """
        Gọi ở CONTROL_HZ. -> (vận tốc FLU hoặc None, danh sách Action).
        None = KHÔNG gửi setpoint (lý do ở self.blocked_reason).
        """
        acts = []
        in_mode = v.mode == self.p.control_mode

        if self.state == ENABLED:
            if not v.connected:
                acts.append(self._lock("mat ket noi FCU", t))
            elif in_mode:
                self.was_in_control_mode = True
                self.mode_request_t = None
            elif self.was_in_control_mode:
                # NGUYÊN TẮC 4
                acts.append(self._lock(f"pilot doi sang {v.mode}", t))
            elif self.mode_request_t is not None and \
                    t - self.mode_request_t > self.p.mode_grace_sec:
                self.mode_request_t = None
                acts.append(self._lock(f"FCU khong chuyen sang {self.p.control_mode}", t))

        reason = None
        if self.state != ENABLED:
            # Camera chỉ nhận diện khi đã ở GUIDED: chưa GUIDED thì việc tiếp theo là
            # của pilot, không phải của người đứng trước camera.
            reason = ("LOCKED - gio ASSUME_GUIDANCE" if in_mode
                      else f"LOCKED - cho pilot gat {self.p.control_mode}")
        elif not v.armed:
            reason = "chua ARM (pilot arm bang tay cam)"
        elif not in_mode:
            reason = f"dang {v.mode or '?'}, can {self.p.control_mode}"
        elif v.landed_state != LANDED_IN_AIR:
            # Gửi vận tốc trong lúc ArduCopter đang TAKEOFF sẽ HUỶ lệnh cất cánh
            # (GUIDED chuyển submode sang velocity). Nên im lặng tới khi IN_AIR.
            # UNDEFINED cũng im lặng: đo 2026-09-15 trên SITL, FCU không gửi
            # EXTENDED_SYS_STATE -> vận tốc 0 gửi liên tục đã huỷ lệnh TAKEOFF.
            reason = {LANDED_ON_GROUND: "duoi dat - gio TAKEOFF",
                      LANDED_TAKEOFF: "dang cat canh",
                      LANDED_LANDING: "dang ha canh"}.get(
                          v.landed_state, "chua biet duoi dat hay dang bay (cho EXTENDED_SYS_STATE)")
        self.blocked_reason = reason or ""
        if reason:
            return None, acts

        motion = self.motion
        if self.last_gesture_t is None or t - self.last_gesture_t > self.p.watchdog_sec:
            motion = "HOVER"       # perception treo / chết -> đứng yên
            self.blocked_reason = ""
        return frd_to_flu(velocity_frd(motion, self.p.speed_xy)), acts
