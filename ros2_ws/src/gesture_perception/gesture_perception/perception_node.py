"""
PERCEPTION_NODE — camera -> cử chỉ, phát lên /gesture/detected.

    ros2 launch gesture_bringup camera_only.launch.py      # cách chạy thường dùng
    ros2 run gesture_perception perception_node --ros-args -p model_path:=...

NODE NÀY KHÔNG BIẾT GÌ VỀ DRONE. Nó chỉ nói "frame này trông như ROLL_LEFT với
độ tin cậy 0.97". Giữ bao lâu mới thành lệnh, có được phép bay không — việc của
command_node.

VÌ SAO CAMERA NẰM TRONG NODE NÀY, KHÔNG TÁCH NODE CAMERA RIÊNG
    Một frame 640x480 là ~0.9 MB. Đẩy qua DDS 30 lần/giây trên Pi 5 tốn vài fps
    chỉ để tuần tự hoá. Ở đây ảnh không bao giờ rời tiến trình; ra ngoài chỉ có
    50 số thực. Muốn xem hình thì bật debug_image_hz (JPEG nén, nhịp thấp).

CHỈ NHẬN DIỆN KHI GUIDED (require_mode, mặc định bật trong sitl/real launch)
    Mode khác GUIDED: camera vẫn chạy, MediaPipe KHÔNG chạy, không phát gì.
    Xem mode_gate.py.

ĐỘ TRỄ
    Thread riêng đọc camera, vòng xử lý luôn lấy frame MỚI NHẤT (frame_source.py).
    header.stamp của /gesture/detected = LÚC CHỤP frame, nên command_node đo được
    trễ thật từ camera tới lệnh. Mỗi stats_period_sec giây in một dòng:
        cam 30 fps | xu ly 15 fps | tuoi frame | pose | clf | tong (chup->phat) | bo frame

Đổi tham số lúc đang chạy, không cần khởi động lại:
    ros2 param set /perception_node one_euro true
    ros2 param set /perception_node tau_override 0.90

Thử trên Pi / WSL không cần đứng trước camera (video take, phát theo nhịp thật):
    ros2 launch gesture_bringup sitl.launch.py video_path:=/duong/dan/TAKEOFF__x.mp4
"""

import glob
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CompressedImage

from gesture_interfaces.msg import Gesture

from . import camera_util as cu
from . import settings as S
from .frame_source import LatestFrame
from .mode_gate import ModeGate
from .perception_core import GesturePerception


def _pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        p = self.declare_parameter
        p("model_path", "")
        p("camera_index", -1)
        p("camera_backend", "auto")
        p("camera_width", S.CAM_WIDTH)
        p("camera_height", S.CAM_HEIGHT)
        p("flip", S.FLIP_FRAME)
        p("pose_complexity", S.POSE_MODEL_COMPLEXITY)
        p("pose_detect_conf", S.POSE_DETECT_CONF)
        p("pose_track_conf", S.POSE_TRACK_CONF)
        p("min_visibility", S.LM_MIN_VISIBILITY)
        p("one_euro", True)
        p("tau_override", -1.0)          # < 0 = dùng tau trong labels.json
        p("publish_keypoints", True)
        p("debug_image_hz", 0.0)         # 0 = tắt. 2 là đủ để xem trên laptop
        p("show_window", False)          # cv2.imshow, chỉ khi Pi có màn hình
        p("require_mode", "")            # "GUIDED" = chỉ nhận diện khi FCU ở GUIDED
        p("state_timeout_sec", 2.0)      # mất /mavros/state lâu hơn -> ngừng nhận diện
        p("video_path", "")              # thay camera: file, glob, hoặc nhiều file cách dấu phẩy
        p("video_loop", True)
        p("stats_period_sec", 5.0)       # 0 = không in thống kê độ trễ

        g = lambda k: self.get_parameter(k).value  # noqa: E731
        model_path = g("model_path")
        if not model_path:
            raise SystemExit("Thieu tham so model_path. Chay qua launch file.")

        tau = g("tau_override")
        self.core = GesturePerception(
            model_path, one_euro=g("one_euro"),
            tau_override=None if tau < 0 else tau,
            complexity=g("pose_complexity"), detect_conf=g("pose_detect_conf"),
            track_conf=g("pose_track_conf"), min_visibility=g("min_visibility"),
            flip=g("flip"))
        clf = self.core.clf
        self.get_logger().info(
            f"model {clf.name}: {len(clf.labels)} lop {clf.labels}, "
            f"tau={clf.tau}, T={clf.temperature:.2f}, doc bang {clf.backend}")
        if clf.unsafe:
            self.get_logger().error("MODEL TRAIN BANG --force TREN DU LIEU KHONG DAT. "
                                    "KHONG DEM RA BAY.")
        # Khởi tạo MediaPipe NGAY, không đợi frame GUIDED đầu tiên: đo trên WSL
        # lần gọi đầu mất ~2.7 s, rơi đúng lúc pilot vừa gạt GUIDED.
        t0 = time.monotonic()
        self.core.process(np.zeros((S.CAM_HEIGHT, S.CAM_WIDTH, 3), np.uint8), 0.0)
        self.core.filter.reset()
        self.get_logger().info(f"MediaPipe san sang ({(time.monotonic() - t0) * 1000:.0f} ms khoi tao)")

        stamp_fn = lambda: self.get_clock().now().to_msg()  # noqa: E731
        video = g("video_path").strip()
        if video:
            paths = []
            for part in video.split(","):
                paths.extend(sorted(glob.glob(part.strip())) or [part.strip()])
            self.src = LatestFrame(video_paths=paths, loop=g("video_loop"), stamp_fn=stamp_fn)
            self.get_logger().info(f"nguon hinh: {len(paths)} video, phat theo nhip that "
                                   f"(loop={g('video_loop')}). flip={self.core.flip}")
        else:
            cap, be = cu.open_camera(g("camera_index"), g("camera_backend"),
                                     g("camera_width"), g("camera_height"))
            if cap is None:
                raise SystemExit(cu.fail_message(g("camera_index"), g("camera_backend")))
            self.src = LatestFrame(cap=cap, stamp_fn=stamp_fn)
            self.get_logger().info(f"camera: {be}")

        # Cử chỉ: best-effort, chỉ giữ cái mới nhất. Mất một frame không sao,
        # xử lý frame cũ thì có sao.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(Gesture, "/gesture/detected", qos)
        self.pub_img = self.create_publisher(
            CompressedImage, "/gesture/debug_image/compressed", qos)

        self.gate = ModeGate(g("require_mode"), g("state_timeout_sec"))
        if self.gate.enabled:
            from mavros_msgs.msg import State
            self.create_subscription(State, "/mavros/state", self._on_state,
                                     qos_profile_sensor_data)
            self.get_logger().info(f"CHI NHAN DIEN KHI FCU O {self.gate.require_mode}")

        self.add_on_set_parameters_callback(self._on_params)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _on_params(self, params):
        for prm in params:
            if prm.name == "one_euro" and prm.type_ == Parameter.Type.BOOL:
                self.core.set_one_euro(prm.value)
                self.get_logger().info(f"one_euro = {prm.value}")
            elif prm.name == "tau_override":
                v = float(prm.value)
                if v < 0:
                    tau = self.core.clf.meta.get("threshold")
                    self.core.clf.tau = float(tau) if tau is not None else None
                else:
                    self.core.clf.set_tau(v)
                self.get_logger().info(f"tau = {self.core.clf.tau}")
        return SetParametersResult(successful=True)

    def _on_state(self, msg):
        self.gate.on_state(msg.connected, msg.mode, time.monotonic())

    def _log_stats(self, st, period, grabbed0, dropped0):
        n = len(st["total"])
        grabbed = self.src.grabbed - grabbed0
        dropped = self.src.dropped - dropped0
        act = 100.0 * st["active"] / max(1, st["loops"])
        if n == 0:
            self.get_logger().info(
                f"nhan dien: cam {grabbed / period:.0f} fps | KHONG xu ly ({st['reason'] or 'khong co frame'})")
            return
        self.get_logger().info(
            f"nhan dien: cam {grabbed / period:.0f} fps | xu ly {n / period:.1f} fps | "
            f"tuoi frame {_pct(st['age'], .5):.0f}/{_pct(st['age'], .95):.0f} ms | "
            f"pose {_pct(st['pose'], .5):.0f}/{_pct(st['pose'], .95):.0f} ms | "
            f"clf {_pct(st['clf'], .5):.1f} ms | "
            f"tong chup->phat {_pct(st['total'], .5):.0f}/{_pct(st['total'], .95):.0f} ms "
            f"(trung vi/p95) | bo {dropped} frame cu | {self.gate.require_mode or 'luon'} {act:.0f}%")

    def _loop(self):
        import cv2
        frame_dt = deque(maxlen=30)
        last = time.monotonic()
        last_img = 0.0
        was_active = None
        period = float(self.get_parameter("stats_period_sec").value)
        st = None
        st_t0 = time.monotonic()
        grabbed0 = dropped0 = 0

        def new_stats():
            return {"age": [], "pose": [], "clf": [], "total": [], "loops": 0, "active": 0, "reason": ""}

        st = new_stats()
        while not self._stop.is_set() and rclpy.ok():
            item = self.src.get(timeout=1.0)
            now = time.monotonic()
            if period > 0 and now - st_t0 >= period:
                self._log_stats(st, now - st_t0, grabbed0, dropped0)
                st, st_t0 = new_stats(), now
                grabbed0, dropped0 = self.src.grabbed, self.src.dropped
            if item is None:
                if self.src.ended:
                    self.get_logger().warn("nguon hinh da het (camera mat hinh hoac video het)",
                                           throttle_duration_sec=5.0)
                continue
            frame, t_cap, stamp, _seq, src_name = item
            st["loops"] += 1

            active, reason = self.gate.check(now)
            if active != was_active:
                if active:
                    self.core.filter.reset()
                    self.get_logger().info(f"BAT DAU NHAN DIEN ({self.gate.require_mode or 'luon bat'})")
                else:
                    self.get_logger().warn(f"NGUNG NHAN DIEN: {reason}")
                was_active = active
            if not active:
                st["reason"] = reason
                if self.get_parameter("show_window").value:
                    cv2.putText(frame, f"CHO {self.gate.require_mode}: {reason}", (12, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow("perception_node", frame)
                    cv2.waitKey(1)
                continue
            st["active"] += 1

            if self.core.flip:
                frame = cv2.flip(frame, 1)
            frame_dt.append(now - last)
            last = now
            fps = 1.0 / max(1e-6, float(np.mean(frame_dt)))

            r = self.core.process(frame, t_cap)

            msg = Gesture()
            # Lúc CHỤP, không phải lúc phát: command_node trừ ra là trễ thật.
            msg.header.stamp = stamp
            msg.header.frame_id = "camera" if src_name == "camera" else f"video:{src_name}"
            msg.label = r.label
            msg.confidence = float(r.confidence)
            msg.top_labels = r.top_labels
            msg.top_probs = [float(x) for x in r.top_probs]
            if r.kp_norm is not None and self.get_parameter("publish_keypoints").value:
                msg.keypoints_norm = [float(x) for x in np.asarray(r.kp_norm).reshape(-1)]
            msg.fps = float(fps)
            msg.pose_ms = float(r.pose_ms)
            self.pub.publish(msg)
            done = time.monotonic()
            st["age"].append((now - t_cap) * 1000)
            st["pose"].append(r.pose_ms)
            st["clf"].append(r.clf_ms)
            st["total"].append((done - t_cap) * 1000)

            img_hz = self.get_parameter("debug_image_hz").value
            show = self.get_parameter("show_window").value
            if show or (img_hz > 0 and now - last_img >= 1.0 / img_hz):
                self.core.draw(frame, r, fps)
                if img_hz > 0 and now - last_img >= 1.0 / img_hz:
                    last_img = now
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    if ok:
                        im = CompressedImage()
                        im.header = msg.header
                        im.format = "jpeg"
                        im.data.frombytes(jpg.tobytes())   # giống cv_bridge
                        self.pub_img.publish(im)
                if show:
                    cv2.imshow("perception_node", frame)
                    cv2.waitKey(1)

    def destroy_node(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.src.stop()
        self.core.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit) as e:
        if isinstance(e, SystemExit) and e.code not in (None, 0):
            print(e.code)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
