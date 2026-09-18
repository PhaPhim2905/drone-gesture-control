#!/usr/bin/env python3
"""
BƯỚC 6 — xem model nhận ra cử chỉ gì, TRỰC TIẾP từ webcam laptop. Không cần ROS.

    python training\\6_live_check.py                  # model mới train trong models/
    python training\\6_live_check.py --cam 1
    python training\\6_live_check.py --model ros2_ws\\src\\gesture_bringup\\models\\v3_pck_body25.tflite

Phím (bấm khi cửa sổ camera đang được chọn):
    r   xoá bộ đếm, bắt đầu đếm lại cho cử chỉ kế tiếp
    o   bật / tắt One-Euro
    q   thoát, in bảng NHAN THAY

Dùng cùng chuỗi xử lý với perception_node trên Pi (perception_core.py), nên thấy
gì ở đây thì Pi thấy y như vậy, chỉ khác fps.

──────────────────────────────────────────────────────────────────────────────
PHÉP THỬ TRÁI / PHẢI
    Đứng: tay PHẢI THẬT dang ngang, tay TRÁI THẬT giơ thẳng lên. Bấm r, giữ 10 s.
    Phải ra ROLL_RIGHT chiếm đa số.

    Dòng "MediaPipe:" cho biết MediaPipe gọi tay nào là tay đang GIƠ CAO. Ảnh đã
    lật như gương, nên nếu anh giơ tay TRÁI thật mà dòng này ghi "tay PHAI cao"
    thì MediaPipe đang đảo tên hai tay — bình thường, miễn là lúc quay dữ liệu
    và lúc chạy cùng lật.
"""

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "gesture_perception"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gesture_perception import camera_util as cu  # noqa: E402
from gesture_perception import pck_format as pf  # noqa: E402
from gesture_perception import settings as S  # noqa: E402
from gesture_perception.perception_core import GesturePerception  # noqa: E402


def pick_model(arg):
    if arg:
        return Path(arg)
    for d in (ROOT / "models", ROOT / "ros2_ws" / "src" / "gesture_bringup" / "models"):
        c = sorted(d.glob("v3_*.tflite"), key=lambda p: p.stat().st_mtime)
        if c:
            return c[-1]
    raise SystemExit("Khong tim thay model .tflite nao. Train truoc (training/4_train.py).")


def which_arm_high(kp):
    """Theo TÊN của MediaPipe: cổ tay nào cao hơn vai của nó rõ rệt."""
    if kp is None:
        return ""
    def up(w, s):
        if kp[w, 2] <= 0 or kp[s, 2] <= 0:
            return None
        return kp[w, 1] < kp[s, 1] - 0.5 * abs(kp[pf.B25_RSHOULDER, 0] - kp[pf.B25_LSHOULDER, 0])
    r = up(pf.B25_RWRIST, pf.B25_RSHOULDER)
    l = up(pf.B25_LWRIST, pf.B25_LSHOULDER)
    if r is None or l is None:
        return "MediaPipe: khong thay du 2 tay"
    return {(True, True): "MediaPipe: HAI tay cao", (True, False): "MediaPipe: tay PHAI cao",
            (False, True): "MediaPipe: tay TRAI cao", (False, False): "MediaPipe: hai tay thap"}[(r, l)]


def print_table(counts, since):
    total = sum(counts.values())
    print("\n" + "=" * 56)
    print(f"NHAN THAY  ({total} frame, {time.monotonic() - since:.1f} s)")
    print("=" * 56)
    for lab, n in counts.most_common():
        bar = "#" * int(40 * n / max(1, total))
        print(f"  {lab:<16}{n:>6}  {100 * n / max(1, total):5.1f}%  {bar}")
    print("=" * 56)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--tau", type=float, default=None, help="ghi de nguong, 0 = tat")
    ap.add_argument("--seconds", type=float, default=0, help="tu thoat sau N giay (0 = khong)")
    ap.add_argument("--no-window", action="store_true", help="khong mo cua so, chi in log")
    args = ap.parse_args()

    model = pick_model(args.model)
    core = GesturePerception(model, tau_override=args.tau)
    clf = core.clf
    print(f"  model  {model}")
    print(f"  lop    {', '.join(clf.labels)}")
    print(f"  tau    {clf.tau}   T {clf.temperature:.2f}   flip {S.FLIP_FRAME}")
    if clf.unsafe:
        print("  !!! MODEL TRAIN BANG --force - CHI DE THU")

    cap, be = cu.open_camera(args.cam, args.backend, S.CAM_WIDTH, S.CAM_HEIGHT)
    if cap is None:
        print(cu.fail_message(args.cam, args.backend))
        return 1
    print(f"  camera {be}")
    print("  phim: r dem lai   o bat/tat One-Euro   q thoat\n")

    counts, since = Counter(), time.monotonic()
    dts, last, last_log = deque(maxlen=30), time.monotonic(), 0.0
    t_start = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if S.FLIP_FRAME:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            dts.append(now - last)
            last = now
            fps = 1.0 / max(1e-6, float(np.mean(dts)))

            r = core.process(frame, now)
            counts[r.label] += 1
            arm = which_arm_high(r.kp_raw)

            if now - last_log >= 1.0:
                last_log = now
                top = ", ".join(f"{a} {p*100:.0f}%" for a, p in zip(r.top_labels, r.top_probs))
                print(f"  {now - since:5.1f}s  {r.label:<16} {r.confidence*100:5.1f}%  "
                      f"| {top:<48} | {arm:<28} | {fps:4.1f} fps")

            if not args.no_window:
                core.draw(frame, r, fps, extra=arm)
                y = 150
                cv2.putText(frame, "NHAN THAY (r = dem lai)", (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                total = max(1, sum(counts.values()))
                for i, (lab, n) in enumerate(counts.most_common(5)):
                    cv2.putText(frame, f"{lab[:16]:<16} {100*n/total:5.1f}%",
                                (12, y + 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0) if i == 0 else (180, 180, 180), 1)
                cv2.imshow("6_live_check", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("r"):
                    print_table(counts, since)
                    counts, since = Counter(), time.monotonic()
                    print("  -- DEM LAI --")
                if k == ord("o"):
                    core.set_one_euro(not core.one_euro)
                    print(f"  one-euro = {core.one_euro}")
            if args.seconds and now - t_start >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        core.close()
        cv2.destroyAllWindows()
    print_table(counts, since)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
