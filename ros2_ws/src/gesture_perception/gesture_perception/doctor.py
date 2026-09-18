"""
DOCTOR — kiểm toàn bộ máy trước khi chạy, in ĐẠT / LỖI cho từng mục.

    ros2 run gesture_perception doctor               # kiểm đủ, có đo camera 10 s
    ros2 run gesture_perception doctor --no-camera   # chỉ thư viện + model
    ros2 run gesture_perception doctor --probe       # dò mọi camera
    python -m gesture_perception.doctor              # không cần ROS cũng chạy

Thay cho tools/test_tflite.py và tools/list_cams.py (đã xoá 2026-09-14). Sáu con
số cần gửi lại sau khi flash Pi đều in ra ở đây.
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

OK, FAIL, WARN = "DAT ", "LOI ", "CHU Y"
_results = []


def report(status, name, detail=""):
    _results.append((status, name))
    print(f"  [{status}] {name:<34} {detail}")


def find_model(arg):
    if arg:
        return Path(arg)
    try:
        from ament_index_python.packages import get_package_share_directory
        d = Path(get_package_share_directory("gesture_bringup")) / "models"
    except Exception:
        # Chạy thẳng từ repo, chưa colcon build
        d = Path(__file__).resolve().parents[2] / "gesture_bringup" / "models"
    found = sorted(d.glob("*.tflite"))
    return found[0] if found else d / "v3_pck_body25.tflite"


def check_libs():
    print("\n1. THU VIEN")
    report(OK, "python", f"{platform.python_version()} ({sys.executable})")
    report(OK, "numpy", np.__version__)
    try:
        import cv2
        report(OK, "opencv", cv2.__version__)
    except ImportError as e:
        report(FAIL, "opencv", str(e))
    try:
        import mediapipe as mp
        has = hasattr(mp, "solutions")
        report(OK if has else FAIL, "mediapipe",
               mp.__version__ + ("" if has else "  <- KHONG CO mp.solutions, can 0.10.14"))
    except ImportError as e:
        report(FAIL, "mediapipe", str(e))
    try:
        import rclpy  # noqa: F401
        report(OK, "rclpy", "import duoc")
    except ImportError:
        report(WARN, "rclpy", "khong import duoc (chua source ROS / venv thieu "
               "--system-site-packages)")
    try:
        import mavros_msgs  # noqa: F401
        report(OK, "mavros_msgs", "import duoc")
    except ImportError:
        report(WARN, "mavros_msgs", "chua cai ros-jazzy-mavros")


def check_model(path):
    print("\n2. MODEL")
    from .classifier import GestureClassifier
    from . import pck_format as pf
    if not path.exists():
        report(FAIL, "file model", str(path))
        return None
    try:
        clf = GestureClassifier(path)
    except Exception as e:
        report(FAIL, "doc model", repr(e))
        return None
    report(OK, "doc model", f"{clf.name} qua {clf.backend}")
    report(OK, "lop", ", ".join(clf.labels))
    report(OK if clf.tau else WARN, "tau / T", f"{clf.tau} / {clf.temperature:.2f}")
    if clf.unsafe:
        report(FAIL, "an toan", "model train bang --force, KHONG DEM RA BAY")

    kp = pf._fake_body25(200.0, "t")
    n = pf.normalize_body25(kp)
    for _ in range(20):
        clf.probs(n)
    t0 = time.perf_counter()
    for _ in range(300):
        clf.probs(n)
    ms = (time.perf_counter() - t0) / 300 * 1000
    report(OK, "suy luan", f"{ms:.3f} ms/frame")
    return clf


def check_system():
    print("\n3. MAY")
    try:
        import psutil
        vm = psutil.virtual_memory()
        report(OK, "RAM", f"tong {vm.total/2**30:.1f} GB, trong {vm.available/2**30:.1f} GB")
        du = psutil.disk_usage(str(Path.home()))
        report(OK, "o dia", f"trong {du.free/2**30:.1f} GB / {du.total/2**30:.1f} GB")
    except ImportError:
        report(WARN, "psutil", "chua cai")
    tz = Path("/sys/class/thermal/thermal_zone0/temp")
    if tz.exists():
        c = int(tz.read_text()) / 1000
        report(OK if c < 75 else WARN, "nhiet do CPU", f"{c:.1f} °C (ha xung o 85)")
    thr = Path("/sys/devices/platform/soc/soc:firmware/get_throttled")
    if thr.exists():
        v = thr.read_text().strip()
        report(OK if v in ("0", "0x0") else WARN, "throttled", v + " (khac 0 = nguon yeu/nong)")
    if platform.system() == "Linux":
        for dev in ("/dev/ttyACM0", "/dev/serial0"):
            if os.path.exists(dev):
                report(OK if os.access(dev, os.R_OK | os.W_OK) else FAIL, dev,
                       "doc/ghi duoc" if os.access(dev, os.R_OK | os.W_OK)
                       else "khong co quyen: sudo usermod -aG dialout $USER roi dang nhap lai")
        venv = sys.prefix != sys.base_prefix
        report(OK if venv else WARN, "venv", sys.prefix if venv else "dang chay python he thong")
        disc = os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE", "(mac dinh SUBNET)")
        report(OK, "ROS_AUTOMATIC_DISCOVERY_RANGE", disc)


def check_camera(clf, index, backend, seconds):
    print(f"\n4. CAMERA + MEDIAPIPE ({seconds:.0f} s, dung truoc camera)")
    import cv2
    from . import camera_util as cu
    from .perception_core import GesturePerception
    cap, be = cu.open_camera(index, backend, 640, 480)
    if cap is None:
        report(FAIL, "mo camera", f"index {index} backend {backend}")
        return
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    report(OK, "mo camera", f"{be}, {fcc}")
    core = GesturePerception(_model_path)
    n, t0, pose_ms, seen = 0, time.monotonic(), [], {}
    rss_peak = 0
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:
        proc = None
    while time.monotonic() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        r = core.process(frame, time.monotonic())
        n += 1
        pose_ms.append(r.pose_ms)
        seen[r.label] = seen.get(r.label, 0) + 1
        if proc:
            rss_peak = max(rss_peak, proc.memory_info().rss)
    el = time.monotonic() - t0
    cap.release()
    core.close()
    fps = n / el
    report(OK if fps >= 8 else FAIL, "fps toan chuoi", f"{fps:.1f} (can >= 8)")
    report(OK, "MediaPipe", f"trung vi {np.median(pose_ms):.1f} ms")
    if proc:
        report(OK, "RAM dinh cua tien trinh", f"{rss_peak/2**20:.0f} MB")
    report(OK, "nhan thay", ", ".join(f"{k} x{v}" for k, v in
                                      sorted(seen.items(), key=lambda kv: -kv[1])))


def probe():
    from . import camera_util as cu
    print("\nDO CAMERA")
    found = cu.probe(max_index=6, width=640, height=480)
    if not found:
        print("  khong thay camera nao")
    for nm, i, w, h in found:
        print(f"  index {i}  backend {nm:<6} {w}x{h}")


_model_path = None


def main(argv=None):
    global _model_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--cam", type=int, default=-1, help="-1 = tu do webcam USB")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--probe", action="store_true")
    args, _ = ap.parse_known_args(argv)

    if args.probe:
        probe()
        return 0

    print("=" * 72)
    print("DOCTOR — kiem may truoc khi chay")
    print("=" * 72)
    check_libs()
    _model_path = find_model(args.model)
    clf = check_model(_model_path)
    check_system()
    if clf is not None and not args.no_camera:
        check_camera(clf, args.cam, args.backend, args.seconds)

    n_fail = sum(1 for s, _ in _results if s == FAIL)
    print("\n" + "=" * 72)
    print("KET QUA: " + ("TAT CA DAT" if n_fail == 0 else f"{n_fail} MUC LOI"))
    print("=" * 72)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
