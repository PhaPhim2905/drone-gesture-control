"""sim/chain_test: ghi lai /gesture/detected (dem/giay), /drone/gesture_command (doi trang thai), /mavros/state (mode)."""
import sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from mavros_msgs.msg import State
from gesture_interfaces.msg import Gesture, CommandStatus

T0 = time.time()
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 240


def log(s):
    print(f"{time.time() - T0:7.1f} MON {s}", flush=True)


class Mon(Node):
    def __init__(self):
        super().__init__("chain_monitor")
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Gesture, "/gesture/detected", self.on_g, be)
        self.create_subscription(CommandStatus, "/drone/gesture_command", self.on_c, 10)
        self.create_subscription(State, "/mavros/state", self.on_s, qos_profile_sensor_data)
        self.n = 0
        self.labels = {}
        self.last = None
        self.mode = None
        self.create_timer(2.0, self.tick)

    def on_g(self, m):
        self.n += 1
        self.labels[m.label] = self.labels.get(m.label, 0) + 1
        self.src = m.header.frame_id

    def on_c(self, m):
        key = (m.state, m.last_event, m.blocked_reason, m.motion, m.streaming)
        if key != self.last:
            log(f"CMD state={m.state} event='{m.last_event}' blocked='{m.blocked_reason}' motion={m.motion} "
                f"streaming={m.streaming} mode={m.mode} armed={m.armed} landed={m.landed_state}")
            self.last = key

    def on_s(self, m):
        if m.mode != self.mode:
            log(f"FCU mode {self.mode} -> {m.mode} connected={m.connected} armed={m.armed}")
            self.mode = m.mode

    def tick(self):
        top = sorted(self.labels.items(), key=lambda x: -x[1])[:2]
        log(f"GESTURE {self.n} msg/2s mode={self.mode} {top} {getattr(self, 'src', '')}")
        self.n = 0
        self.labels = {}


rclpy.init()
node = Mon()
end = time.time() + DUR
try:
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
except Exception:  # noqa: BLE001  - bi kill luc don dep
    pass
node.destroy_node()
if rclpy.ok():
    rclpy.shutdown()
