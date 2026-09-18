#!/usr/bin/env python3
"""
BUILD_FROM_VIDEO — chay MediaPipe tren file video, sinh ra .jsonl nhu la vua
thu bang camera.

    python training/3b_build_from_video.py                 # moi video trong data/video/
    python training/3b_build_from_video.py --dir D:\\quay_ngoai_san
    python training/3b_build_from_video.py --flip          # video quay bang dien thoai

Chay o MOI TRUONG CHINH. Khong can camera, khong can TensorFlow.

──────────────────────────────────────────────────────────────────────────────
NHÃN NẰM TRONG TÊN FILE

    HOVER__20260907_1.mp4        ->  nhan HOVER,   take 20260907_1
    MOVE_LEFT__ngoai_san_3.mp4   ->  nhan MOVE_LEFT

Phần trước dấu "__" là nhãn, phần sau là mã take. Đổi tên file là đổi nhãn —
không có file cấu hình nào phải sửa theo. Video không có "__" thì bỏ qua kèm
cảnh báo, chứ không đoán bừa.

──────────────────────────────────────────────────────────────────────────────
VÌ SAO LƯU VIDEO CHỨ KHÔNG CHỈ LƯU KHỚP

JSONL đã là dữ liệu thô ở tầng KHỚP, đủ để đổi phép chuẩn hoá mà không phải
thu lại. Nhưng có ba thứ nó KHÔNG cứu được, và cả ba đều sẽ xảy ra:

  1. Đổi POSE_MODEL_COMPLEXITY từ 0 (Lite) sang 1. Khớp sẽ khác. Không có
     video thì phải mời người ra đứng thu lại toàn bộ.
  2. Đổi LM_MIN_VISIBILITY, hoặc đổi hẳn sang một bộ ước lượng tư thế khác.
  3. Dữ liệu quay ở NGOÀI SÂN, ở cự ly bay thật 5-10 m, quay bằng điện thoại
     hoặc bằng chính camera trên drone. Không có đường nào khác để đưa loại
     dữ liệu đó vào tập huấn luyện.

Cái giá là dung lượng: 640x480 mp4v khoảng 1-2 MB mỗi take 10 giây. Toàn bộ
9 lớp x 5 take rơi vào 50-100 MB. Vì vậy data/video/ nằm trong .gitignore.

──────────────────────────────────────────────────────────────────────────────
⚠️ SOI GƯƠNG — chỗ sai thì drone bay ngược hướng ra lệnh

record_dataset.py lật ảnh (FLIP_FRAME) TRƯỚC khi ghi video, nên video nó lưu
ra ĐÃ lật rồi. Mặc định ở đây là KHÔNG lật thêm.

Video quay bằng điện thoại hoặc camera ngoài thì tuỳ: camera trước của điện
thoại thường đã tự soi gương, camera sau thì không. Xem lại video, nếu người
trong đó giơ tay phải mà anh thấy nó ở phía trái màn hình thì KHÔNG cần --flip;
ngược lại thì cần. Sai chỗ này là MOVE_LEFT và MOVE_RIGHT hoán đổi cho nhau.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# pck_format, settings, camera_util nam trong goi ROS gesture_perception -
# MOT ban duy nhat cho ca train lan luc bay.
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "gesture_perception" / "gesture_perception"))

try:
    import cv2
except ImportError:
    raise SystemExit(
        "\n  THIEU cv2 -> dang chay NHAM python.\n\n"
        "  File nay chay o MOI TRUONG CHINH:\n"
        "      python training/3b_build_from_video.py\n\n"
        "  Neu terminal dang bat .venv-train thi go:  deactivate\n"
    )

import pck_format as pf     # noqa: E402
import settings as C       # noqa: E402

VIDEO_DIR = ROOT / "data" / "video"
OUT_DIR = ROOT / "data" / "gestures"
EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")

# Cùng luật với record_dataset.py: tám lớp cử chỉ bắt buộc thấy đủ hai cổ tay.
NEED_WRISTS_EXCEPT = {"NEGATIVE"}


def parse_name(p):
    """HOVER__take3.mp4 -> ("HOVER", "take3"). Khong co "__" thi tra (None, None)."""
    stem = p.stem
    if "__" not in stem:
        return None, None
    lab, take = stem.split("__", 1)
    return lab.strip(), take.strip()


def process(path, pose, args):
    """Mot video -> danh sach ban ghi jsonl. Tra ve (rows, thong_ke)."""
    lab, take = parse_name(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], {"loi": 1}

    rows = []
    st = Counter()
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1.0 < fps < 240.0 else 30.0
    step_t = args.period          # giây giữa hai mẫu, tính theo THỜI GIAN VIDEO
    next_t = 0.0
    n_frame = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n_frame / fps
        n_frame += 1
        if t < next_t:
            continue
        next_t = t + step_t

        if args.flip:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            st["khong thay nguoi"] += 1
            continue

        kp = pf.mediapipe_to_body25(res.pose_landmarks.landmark, w, h,
                                    C.LM_MIN_VISIBILITY)
        if lab not in NEED_WRISTS_EXCEPT and not (kp[pf.B25_LWRIST, 2] > 0
                                                  and kp[pf.B25_RWRIST, 2] > 0):
            st["thieu co tay"] += 1
            continue

        st["giu"] += 1
        rows.append({
            "label": lab, "take": take, "t": round(t, 3), "w": w, "h": h,
            "kp": [[round(float(v), 2) for v in p] for p in kp],
            "src": path.name,
        })

    cap.release()
    st["tong frame"] = n_frame
    return rows, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(VIDEO_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--period", type=float, default=0.10,
                    help="giay giua hai mau, tinh theo thoi gian VIDEO")
    ap.add_argument("--flip", action="store_true",
                    help="lat guong. Video do record_dataset.py ghi thi KHONG can")
    ap.add_argument("--force", action="store_true",
                    help="lam lai ca nhung video da trich roi")
    args = ap.parse_args()

    import mediapipe as mp

    src = Path(args.dir)
    vids = sorted(p for p in src.glob("*") if p.suffix.lower() in EXTS)
    if not vids:
        print(f"  Khong co video nao trong {src}")
        print("  Thu bang:  python training/1_record.py --cam 1 --save-video")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("BUILD_FROM_VIDEO")
    print("=" * 76)
    print(f"  {len(vids)} video trong {src}")
    print(f"  lay mau moi {args.period}s thoi gian video, "
          f"lat guong: {'CO' if args.flip else 'KHONG'}")

    pose = mp.solutions.pose.Pose(
        model_complexity=C.POSE_MODEL_COMPLEXITY,
        smooth_landmarks=C.POSE_SMOOTH_LANDMARKS,
        min_detection_confidence=C.POSE_DETECT_CONF,
        min_tracking_confidence=C.POSE_TRACK_CONF,
    )

    total = Counter()
    print("\n" + "-" * 76)
    print(f"  {'video':<34}{'giu':>7}{'thieu tay':>11}{'khong nguoi':>13}")
    print("-" * 76)

    for p in vids:
        lab, take = parse_name(p)
        if lab is None:
            print(f"  {p.name:<34}BO QUA - ten thieu dau '__' de tach nhan")
            total["bo qua"] += 1
            continue

        dst = out_dir / f"video_{p.stem}.jsonl"
        if dst.exists() and not args.force:
            print(f"  {p.name:<34}da co {dst.name}, bo qua (--force de lam lai)")
            continue

        rows, st = process(p, pose, args)
        if st.get("loi"):
            print(f"  {p.name:<34}KHONG DOC DUOC")
            total["loi"] += 1
            continue

        dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                               for r in rows), encoding="utf8")
        flag = ""
        if st["thieu co tay"] > st["giu"]:
            flag = "  <- CO TAY RA NGOAI KHUNG"
        print(f"  {p.name:<34}{st['giu']:>7}{st['thieu co tay']:>11}"
              f"{st['khong thay nguoi']:>13}{flag}")
        total["giu"] += st["giu"]
        total["thieu co tay"] += st["thieu co tay"]
        total[f"lop:{lab}"] += st["giu"]

    pose.close()

    print("-" * 76)
    print(f"  Tong: {total['giu']} mau giu, {total['thieu co tay']} bo vi thieu co tay")
    labs = sorted(k[4:] for k in total if k.startswith("lop:"))
    for lab in labs:
        print(f"    {lab:<20}{total['lop:' + lab]:>6} mau")

    print("\n" + "=" * 76)
    print("  Da ghi .jsonl vao data/gestures/ - tu day di tiep nhu binh thuong:")
    print("      python training/3_build_dataset.py")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
