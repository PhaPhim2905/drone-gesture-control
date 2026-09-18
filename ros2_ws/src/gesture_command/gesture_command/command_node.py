"""
COMMAND_NODE — /gesture/detected  ->  MAVROS.

    nghe   /gesture/detected         gesture_interfaces/Gesture
           /mavros/state             mavros_msgs/State
           /mavros/extended_state    mavros_msgs/ExtendedState
    phát   /mavros/setpoint_raw/local   mavros_msgs/PositionTarget  (vận tốc)
           /drone/gesture_command       gesture_interfaces/CommandStatus
    gọi    /mavros/set_mode          ENABLE->GUIDED (tuỳ chọn), STOP->LOITER, LAND
           /mavros/cmd/takeoff       TAKEOFF, chỉ khi đã armed

KHÔNG có client /mavros/cmd/arming trong file này, và test kiểm điều đó trên mã
nguồn. Arm là việc của pilot trên tay cầm (SITL: gõ "arm throttle" ở MAVProxy).

Mọi quyết định nằm trong command_logic.py. File này chỉ chuyển dữ liệu.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Vector3
from mavros_msgs.msg import State, ExtendedState, PositionTarget
from mavros_msgs.srv import SetMode, CommandTOL, CommandLong

from gesture_interfaces.msg import Gesture, CommandStatus

from .command_logic import CommandLogic, Params, Vehicle

# Chỉ dùng 3 trường vận tốc. Bỏ qua vị trí (1|2|4), gia tốc (64|128|256), yaw
# (1024) và yaw_rate (2048). = 3527, đúng giá trị đã test ở v2.
VELOCITY_ONLY_MASK = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                      PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX |
                      PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                      PositionTarget.IGNORE_YAW | PositionTarget.IGNORE_YAW_RATE)


class CommandNode(Node):
    def __init__(self):
        super().__init__("command_node")
        d = Params()
        p = self.declare_parameter
        p("control_mode", d.control_mode)
        p("handback_mode", d.handback_mode)
        p("speed_xy", d.speed_xy)
        p("takeoff_alt", d.takeoff_alt)
        p("hold_sec", d.hold_sec)
        p("hold_assume_guidance", d.hold_override["ASSUME_GUIDANCE"])
        p("hold_land", d.hold_override["LAND"])
        p("hold_stop", d.hold_override["STOP"])
        p("watchdog_sec", d.watchdog_sec)
        p("operator_lost_sec", d.operator_lost_sec)
        p("mode_grace_sec", d.mode_grace_sec)
        p("request_guided_on_enable", d.request_guided_on_enable)
        p("mirror_left_right", d.mirror_left_right)
        p("control_hz", 20.0)
        g = lambda k: self.get_parameter(k).value  # noqa: E731

        params = Params(
            control_mode=g("control_mode"), handback_mode=g("handback_mode"),
            speed_xy=g("speed_xy"), takeoff_alt=g("takeoff_alt"),
            hold_sec=g("hold_sec"),
            hold_override={"ASSUME_GUIDANCE": g("hold_assume_guidance"),
                           "LAND": g("hold_land"), "STOP": g("hold_stop")},
            watchdog_sec=g("watchdog_sec"), operator_lost_sec=g("operator_lost_sec"),
            mode_grace_sec=g("mode_grace_sec"),
            request_guided_on_enable=g("request_guided_on_enable"),
            mirror_left_right=g("mirror_left_right"))
        if params.speed_xy > 2.0:
            raise SystemExit(f"speed_xy = {params.speed_xy} m/s qua nhanh cho giai doan test")
        self.logic = CommandLogic(params)
        self.vehicle = Vehicle()
        self.last_gesture = None
        # Trễ camera -> command_node: header.stamp của Gesture là lúc CHỤP frame.
        self._lat_ms = []
        self._lat_t0 = time.monotonic()
        self._lat_period = 5.0

        self.create_subscription(Gesture, "/gesture/detected", self._on_gesture,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                            history=HistoryPolicy.KEEP_LAST))
        # Best-effort nhận được cả publisher reliable lẫn best-effort của MAVROS.
        self.create_subscription(State, "/mavros/state", self._on_state,
                                 qos_profile_sensor_data)
        self.create_subscription(ExtendedState, "/mavros/extended_state",
                                 self._on_ext, qos_profile_sensor_data)

        self.pub_sp = self.create_publisher(PositionTarget, "/mavros/setpoint_raw/local", 10)
        self.pub_status = self.create_publisher(CommandStatus, "/drone/gesture_command", 10)
        self.cli_mode = self.create_client(SetMode, "/mavros/set_mode")
        self.cli_takeoff = self.create_client(CommandTOL, "/mavros/cmd/takeoff")
        self.cli_cmd = self.create_client(CommandLong, "/mavros/cmd/command")
        self._ext_req_t = -1e9
        self._ext_req_n = 0

        self._hz = float(g("control_hz"))
        self.create_timer(1.0 / self._hz, self._tick)
        self.get_logger().info(
            f"san sang: {params.control_mode}, {params.speed_xy} m/s, "
            f"mirror={params.mirror_left_right}. Trang thai LOCKED - gio ASSUME_GUIDANCE.")

    # ---- thời gian: đồng hồ của node (sim time nếu use_sim_time) ------------
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_state(self, msg):
        self.vehicle.connected = msg.connected
        self.vehicle.armed = msg.armed
        self.vehicle.mode = msg.mode

    def _on_ext(self, msg):
        self.vehicle.landed_state = msg.landed_state

    def _on_gesture(self, msg):
        self.last_gesture = msg.label
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp > 0:
            self._lat_ms.append((self._now() - stamp) * 1000.0)
        prev = (self.logic.state, self.logic.motion)
        self._run(self.logic.on_gesture(msg.label, self._now(), self.vehicle))
        # Lệnh đổi (dừng / đi / khoá) -> gửi setpoint NGAY, không đợi nhịp control_hz
        # (tiết kiệm tới 50 ms ở 20 Hz). tick() không có trạng thái thời gian nên gọi thêm an toàn.
        if (self.logic.state, self.logic.motion) != prev:
            self._tick()
        now = time.monotonic()
        if now - self._lat_t0 >= self._lat_period and self._lat_ms:
            s = sorted(self._lat_ms)
            self.get_logger().info(
                f"tre chup frame -> command_node: {s[len(s) // 2]:.0f}/{s[int(.95 * (len(s) - 1))]:.0f} ms "
                f"(trung vi/p95, {len(s)} cu chi) | lenh ra MAVROS ngay khi doi, giu nhip {1000 / self._hz:.0f} ms")
            self._lat_ms, self._lat_t0 = [], now

    def _request_extended_state(self):
        """
        ArduPilot không tự gửi EXTENDED_SYS_STATE trên cổng MAVROS nối vào (SITL
        SERIAL0, TELEM2 nếu SR2_* chưa bật). Thiếu nó thì landed_state = UNDEFINED
        và logic không cho TAKEOFF / không gửi vận tốc. Xin 5 Hz bằng
        MAV_CMD_SET_MESSAGE_INTERVAL (511), message 245. Thử lại mỗi 3 s, tối đa 10 lần.
        """
        mono = time.monotonic()
        if (not self.vehicle.connected or self.vehicle.landed_state != 0
                or mono - self._ext_req_t < 3.0 or self._ext_req_n >= 10):
            return
        self._ext_req_t = mono
        self._ext_req_n += 1
        req = CommandLong.Request(command=511, param1=245.0, param2=200000.0)
        self._call(self.cli_cmd, req, f"xin EXTENDED_SYS_STATE 5 Hz (lan {self._ext_req_n})",
                   lambda r: r.success)

    def _tick(self):
        self._request_extended_state()
        t = self._now()
        vel, acts = self.logic.tick(t, self.vehicle)
        self._run(acts)
        if vel is not None:
            sp = PositionTarget()
            sp.header.stamp = self.get_clock().now().to_msg()
            sp.header.frame_id = "base_link"
            sp.coordinate_frame = PositionTarget.FRAME_BODY_NED
            sp.type_mask = VELOCITY_ONLY_MASK
            sp.velocity = Vector3(x=vel[0], y=vel[1], z=vel[2])
            self.pub_sp.publish(sp)

        st = CommandStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.state = self.logic.state
        st.candidate = self.logic.candidate
        st.hold_progress = float(self.logic.hold_progress(t))
        st.motion = self.logic.motion
        st.streaming = vel is not None
        if vel is not None:
            st.velocity_flu = Vector3(x=vel[0], y=vel[1], z=vel[2])
        st.blocked_reason = self.logic.blocked_reason
        st.last_event = self.logic.last_event
        st.fcu_connected = self.vehicle.connected
        st.armed = self.vehicle.armed
        st.mode = self.vehicle.mode
        st.landed_state = int(self.vehicle.landed_state)
        self.pub_status.publish(st)

    # ---- thực hiện Action ------------------------------------------------
    def _run(self, acts):
        for a in acts:
            if a.kind == "event":
                self.get_logger().info(a.value)
            elif a.kind == "set_mode":
                self._call(self.cli_mode, SetMode.Request(custom_mode=a.value),
                           f"set_mode {a.value} ({a.reason})",
                           lambda r: r.mode_sent)
            elif a.kind == "takeoff":
                self._call(self.cli_takeoff, CommandTOL.Request(altitude=float(a.value)),
                           f"takeoff {a.value} m", lambda r: r.success)

    def _call(self, client, req, what, ok_of):
        if not client.service_is_ready():
            self.get_logger().error(f"{what}: MAVROS chua san sang ({client.srv_name})")
            return
        fut = client.call_async(req)

        def done(f):
            try:
                ok = ok_of(f.result())
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"{what}: loi {e!r}")
                return
            (self.get_logger().info if ok else self.get_logger().warn)(
                f"{what}: {'FCU nhan' if ok else 'FCU TU CHOI'}")
        fut.add_done_callback(done)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CommandNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
