#!/usr/bin/env python3
"""
BƯỚC 5 — đưa model vừa train vào gói ROS, kèm kiểm hợp đồng.

    python training/5_export_to_ros.py                     # model v3_* mới nhất
    python training/5_export_to_ros.py --stem v3_pck_body25_20260915

Chép <stem>.tflite + <stem>.labels.json từ models/ (nơi 4_train.py ghi ra)
sang ros2_ws/src/gesture_bringup/models/ dưới tên CỐ ĐỊNH v3_pck_body25, để
launch file không phải sửa gì. Sau đó: git commit + push, trên Pi git pull +
scripts/build.sh.

TỪ CHỐI chép nếu:
  - model train bằng --force trên dữ liệu không đạt (an_toan = false)
  - thiếu tau hoặc T trong labels.json
  - có lớp mà command_node không biết -> lớp đó sẽ rơi về HOVER mà không báo,
    đúng loại "lỗi câm" từng làm TAKEOFF không bao giờ phát ra
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models"
DST = ROOT / "ros2_ws" / "src" / "gesture_bringup" / "models"
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "gesture_command"))

from gesture_command.command_logic import gesture_command_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=None)
    ap.add_argument("--name", default="v3_pck_body25", help="ten dich trong goi ROS")
    args = ap.parse_args()

    if args.stem:
        stem = args.stem
    else:
        cands = sorted(SRC.glob("v3_*.tflite"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit(f"Khong co model v3_*.tflite trong {SRC}. Chay 4_train.py truoc.")
        stem = cands[-1].stem
    tfl, lab = SRC / f"{stem}.tflite", SRC / f"{stem}.labels.json"
    for f in (tfl, lab):
        if not f.exists():
            raise SystemExit(f"Thieu {f}")

    meta = json.loads(lab.read_text(encoding="utf8"))
    errors = []
    if meta.get("an_toan") is False or stem.endswith("_UNSAFE"):
        errors.append("model train bang --force tren du lieu KHONG DAT")
    if meta.get("threshold") is None or meta.get("temperature") is None:
        errors.append("labels.json thieu threshold (tau) hoac temperature (T)")
    known = set(gesture_command_map(True))
    unknown = [c for c in meta["labels"] if c not in known]
    if unknown:
        errors.append(f"lop command_node khong biet: {unknown} -> them vao "
                      "gesture_command_map() trong command_logic.py truoc")

    print(f"  model   {stem}")
    print(f"  lop     {meta['labels']}")
    print(f"  tau/T   {meta.get('threshold')} / {meta.get('temperature')}")
    if meta.get("cv_acc_mean") is not None:
        print(f"  cv acc  {meta['cv_acc_mean']*100:.2f}% +/- {(meta.get('cv_acc_std') or 0)*100:.2f}")
    if errors:
        print("\n  TU CHOI:")
        for e in errors:
            print(f"    x  {e}")
        return 1

    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tfl, DST / f"{args.name}.tflite")
    shutil.copy2(lab, DST / f"{args.name}.labels.json")
    print(f"\n  DA CHEP -> {DST.relative_to(ROOT)}/{args.name}.*")
    print("\n  Buoc tiep:")
    print("    git add ros2_ws/src/gesture_bringup/models && git commit -m \"model moi\" && git push")
    print("    (Pi)  git pull && ./scripts/build.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
