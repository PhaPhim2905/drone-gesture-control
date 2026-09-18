#!/usr/bin/env python3
"""
Kiểm LOGIC LỆNH và CÁC LỚP AN TOÀN — không cần ROS, camera hay drone.

    python ros2_ws/src/gesture_command/test/test_command_logic.py
    colcon test --packages-select gesture_command        # trên Pi, qua pytest

Thay cho tests/test_control_logic.py của v2. Sửa command_logic.py hay config
command_*.yaml thì CHẠY LẠI FILE NÀY trước khi nối MAVROS.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gesture_command.command_logic import (  # noqa: E402
    CommandLogic, Params, Vehicle, ENABLED, LOCKED,
    LANDED_ON_GROUND, LANDED_IN_AIR, LANDED_TAKEOFF,
    frd_to_flu, velocity_frd, gesture_command_map)

DT = 0.1   # 10 fps, gần với Pi


def flying():
    return Vehicle(connected=True, armed=True, mode="GUIDED", landed_state=LANDED_IN_AIR)


class Sim:
    """Chạy logic theo thời gian: mỗi frame gọi on_gesture rồi tick."""

    def __init__(self, params=None, vehicle=None):
        self.L = CommandLogic(params or Params())
        self.v = vehicle or flying()
        self.t = 0.0
        self.acts = []
        self.vel = None

    def hold(self, label, sec):
        for _ in range(int(round(sec / DT))):
            self.t += DT
            self.acts += self.L.on_gesture(label, self.t, self.v)
            self.vel, a = self.L.tick(self.t, self.v)
            self.acts += a
        return self

    def kinds(self, kind):
        return [a.value for a in self.acts if a.kind == kind]

    def enable(self):
        return self.hold("ASSUME_GUIDANCE", 1.2)


# ─────────────────────────────────────────────────────────────────────────────
def test_frame_conversion():
    # ⚠️ chỗ đổi dấu nguy hiểm nhất
    assert frd_to_flu((0.0, -1.0, 0.0)) == (0.0, 1.0, -0.0)
    assert frd_to_flu((1.0, 0.0, 0.5)) == (1.0, -0.0, -0.5)
    assert velocity_frd("LEFT", 0.5) == (0.0, -0.5, 0.0)
    assert velocity_frd("XYZ", 0.5) == (0.0, 0.0, 0.0)


def test_mirror_mapping():
    # Mặc định: tên theo hướng CỦA DRONE (chốt 2026-09-14)
    m = gesture_command_map()
    assert m["ROLL_RIGHT"] == "RIGHT" and m["ROLL_LEFT"] == "LEFT"
    assert m["STOP"] == "STOP" and m["NEGATIVE"] == "HOVER"
    assert Params().mirror_left_right is False
    m = gesture_command_map(True)
    assert m["ROLL_RIGHT"] == "LEFT"


def test_roll_direction_in_flu():
    """
    ROLL_RIGHT -> drone sang PHẢI của nó -> FRD vy > 0 -> FLU y ÂM (y FLU = trái).
    Người điều khiển đứng đối diện thấy drone đi sang TRÁI của mình.
    """
    s = Sim().enable().hold("ROLL_RIGHT", 1.2)
    assert s.L.motion == "RIGHT"
    assert s.vel[1] < 0 and s.vel[0] == 0 and s.vel[2] == 0
    s.hold("ROLL_LEFT", 1.2)
    assert s.L.motion == "LEFT" and s.vel[1] > 0


def test_locked_reason_points_to_pilot_before_guided():
    """Chưa GUIDED: camera chưa nhận diện, việc tiếp theo là của pilot."""
    v = Vehicle(connected=True, armed=True, mode="LOITER", landed_state=LANDED_IN_AIR)
    s = Sim(vehicle=v).hold("NEGATIVE", 0.3)
    assert s.vel is None and "cho pilot gat GUIDED" in s.L.blocked_reason
    s.v.mode = "GUIDED"
    s.hold("NEGATIVE", 0.3)
    assert s.vel is None and "gio ASSUME_GUIDANCE" in s.L.blocked_reason


def test_unknown_landed_state_never_streams_or_takes_off():
    """
    FCU không gửi EXTENDED_SYS_STATE -> không biết dưới đất hay đang bay.
    Đo trên SITL 2026-09-15: vận tốc 0 gửi liên tục đã HUỶ lệnh TAKEOFF.
    """
    from gesture_command.command_logic import LANDED_UNDEFINED
    v = Vehicle(connected=True, armed=True, mode="GUIDED", landed_state=LANDED_UNDEFINED)
    s = Sim(vehicle=v).enable().hold("TAKEOFF", 1.3)
    assert s.vel is None and "EXTENDED_SYS_STATE" in s.L.blocked_reason
    assert not s.kinds("takeoff"), "khong duoc cat canh khi chua biet trang thai"


def test_locked_ignores_motion():
    s = Sim().hold("ROLL_LEFT", 3.0)
    assert s.L.state == LOCKED and s.vel is None and s.L.motion == "HOVER"


def test_enable_needs_hold():
    s = Sim().hold("ASSUME_GUIDANCE", 0.8)
    assert s.L.state == LOCKED
    # Frame đầu tiên là mốc bắt đầu giữ, nên cần thêm quá 1.0 s tính từ mốc đó.
    s.hold("ASSUME_GUIDANCE", 0.4)
    assert s.L.state == ENABLED


def test_stop_immediately_on_change():
    """NGUYÊN TẮC 2: đổi tư thế -> vận tốc 0 NGAY frame đó, không chờ giữ."""
    s = Sim().enable().hold("ROLL_LEFT", 1.2)
    assert s.vel[1] != 0
    s.hold("NEGATIVE", DT)
    assert s.vel == (0.0, -0.0, -0.0) or s.vel == (0.0, 0.0, 0.0)
    s = Sim().enable().hold("ROLL_LEFT", 1.2).hold("AMBIGUOUS", DT)
    assert s.vel[1] == 0


def test_new_motion_needs_hold():
    s = Sim().enable().hold("ROLL_LEFT", 0.5)
    assert s.vel[1] == 0, "chua giu du 1 s ma da bay"


def test_negative_never_moves():
    s = Sim().enable().hold("NEGATIVE", 5.0)
    assert s.vel[1] == 0 and s.L.state == ENABLED


def test_stop_locks_and_hands_back_loiter():
    s = Sim().enable().hold("ROLL_LEFT", 1.2).hold("STOP", 0.7)
    assert s.L.state == LOCKED
    assert "LOITER" in s.kinds("set_mode")
    assert s.vel is None


def test_stop_does_not_steal_mode_from_pilot():
    s = Sim().enable()
    s.v.mode = "ALT_HOLD"          # pilot đã gạt sang mode khác
    s.hold("STOP", 0.7)
    assert s.kinds("set_mode") == []


def test_land_always_on_and_locks():
    s = Sim().hold("LAND", 1.0)     # đang LOCKED vẫn nghe LAND
    assert "LAND" in s.kinds("set_mode")
    s = Sim().enable().hold("LAND", 1.0)
    assert s.L.state == LOCKED and "LAND" in s.kinds("set_mode")


def test_land_ignored_on_ground():
    s = Sim(vehicle=Vehicle(True, True, "GUIDED", LANDED_ON_GROUND)).hold("LAND", 1.0)
    assert s.kinds("set_mode") == []


def test_land_fires_once_per_hold():
    s = Sim().hold("LAND", 3.0)
    assert s.kinds("set_mode").count("LAND") == 1


def test_takeoff_never_arms():
    """Nguyên tắc 1: TAKEOFF khi chưa arm -> KHÔNG làm gì ngoài báo."""
    v = Vehicle(connected=True, armed=False, mode="GUIDED", landed_state=LANDED_ON_GROUND)
    s = Sim(vehicle=v).enable().hold("TAKEOFF", 1.5)
    assert s.kinds("takeoff") == []
    assert all(a.kind in ("event",) for a in s.acts if a.kind != "set_mode")
    assert any("pilot phai arm" in str(a.value) for a in s.acts)


def test_takeoff_when_armed_on_ground():
    v = Vehicle(connected=True, armed=True, mode="GUIDED", landed_state=LANDED_ON_GROUND)
    s = Sim(Params(takeoff_alt=2.5), v).enable().hold("TAKEOFF", 1.2)
    assert s.kinds("takeoff") == [2.5]


def test_takeoff_ignored_in_air():
    s = Sim().enable().hold("TAKEOFF", 1.2)
    assert s.kinds("takeoff") == []


def test_no_velocity_while_taking_off():
    """Vận tốc gửi trong lúc TAKEOFF sẽ huỷ cất cánh của ArduCopter."""
    v = Vehicle(connected=True, armed=True, mode="GUIDED", landed_state=LANDED_TAKEOFF)
    s = Sim(vehicle=v).enable().hold("HOVER", 1.2)
    assert s.vel is None and "cat canh" in s.L.blocked_reason


def test_pilot_mode_switch_locks():
    s = Sim().enable().hold("ROLL_LEFT", 1.2)
    s.v.mode = "LOITER"
    s.hold("ROLL_LEFT", DT)
    assert s.L.state == LOCKED and s.vel is None


def test_not_armed_blocks():
    s = Sim(vehicle=Vehicle(True, False, "GUIDED", LANDED_ON_GROUND)).enable().hold("ROLL_LEFT", 1.2)
    assert s.vel is None and "ARM" in s.L.blocked_reason


def test_operator_lost_locks():
    s = Sim().enable().hold("ROLL_LEFT", 1.2).hold("NO_OPERATOR", 1.0)
    assert s.L.state == LOCKED


def test_watchdog_hover():
    s = Sim().enable().hold("ROLL_LEFT", 1.2)
    vel, _ = s.L.tick(s.t + 0.6, s.v)          # không có frame nào trong 0.6 s
    assert vel[1] == 0


def test_enable_without_fcu_refused():
    s = Sim(vehicle=Vehicle(connected=False)).enable()
    assert s.L.state == LOCKED


def test_request_guided_on_enable():
    v = Vehicle(True, True, "LOITER", LANDED_IN_AIR)
    s = Sim(Params(request_guided_on_enable=True), v).enable()
    assert "GUIDED" in s.kinds("set_mode") and s.L.state == ENABLED
    s.v.mode = "GUIDED"
    s.hold("HOVER", 0.2)
    assert s.L.state == ENABLED
    # FCU không chịu đổi -> hết thời gian chờ thì khoá
    v = Vehicle(True, True, "LOITER", LANDED_IN_AIR)
    s = Sim(Params(request_guided_on_enable=True, mode_grace_sec=1.0), v).enable().hold("HOVER", 1.5)
    assert s.L.state == LOCKED


def test_source_never_arms():
    """Kiểm MÃ NGUỒN: không file nào của gesture_command gọi được lệnh arm."""
    pkg = HERE.parent / "gesture_command"
    bad = re.compile(r"cmd/arming|CommandBool|COMPONENT_ARM_DISARM|arming", re.I)
    for f in pkg.glob("*.py"):
        hits = [ln for ln in f.read_text(encoding="utf8").splitlines()
                if bad.search(ln) and not ln.lstrip().startswith("#")
                and "KHÔNG có client" not in ln]
        assert not hits, f"{f.name} co dong nghi ARM: {hits}"
        # /mavros/cmd/command gửi được MỌI lệnh MAVLink, kể cả ARM (400). Chỉ cho
        # đúng MAV_CMD_SET_MESSAGE_INTERVAL (511).
        cmds = re.findall(r"CommandLong\.Request\([^)]*command\s*=\s*(\d+)", f.read_text(encoding="utf8"))
        assert set(cmds) <= {"511"}, f"{f.name} gui CommandLong ngoai 511: {cmds}"


# ─────────────────────────────────────────────────────────────────────────────
def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [ok]   {name}")
        except AssertionError as e:
            fails += 1
            print(f"  [FAIL] {name}: {e}")
    print("=" * 60)
    print("TAT CA DEU QUA" if not fails else f"THAT BAI {fails}/{len(tests)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
