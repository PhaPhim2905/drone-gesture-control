#!/usr/bin/env python3
"""
BƯỚC 7 — chấm model bằng VIDEO TAKE, qua ĐÚNG chuỗi chạy trên Pi. Không cần ROS.

    python training/7_replay_check.py                         # mọi video trong data/video/
    python training/7_replay_check.py --proc-fps 12           # giả lập Pi chỉ xử lý 12 fps
    python training/7_replay_check.py --model models/v3_moi.tflite --csv ket_qua.csv

Khác 4_train.py thế nào
    4_train.py chấm trên TỌA ĐỘ đã trích sẵn (dataset .npz), chia fold theo take.
    File này chạy lại TỪ ẢNH: MediaPipe -> One-Euro -> PCK -> TFLite -> luật giữ
    của command_logic, y như perception_node + command_node. Nên nó trả lời thẳng
    câu hỏi "đứng trước camera làm cử chỉ này thì drone có nhận lệnh đúng không,
    sau bao lâu", và đo luôn thời gian xử lý mỗi frame TRÊN MÁY ĐANG CHẠY.

    Chạy trên Pi 5 = số độ trễ thật của Pi.

Tên video = nhãn thật: TAKEOFF__20260909_151101_0.mp4 (1_record.py đặt sẵn).
Video 1_record.py quay ĐÃ LẬT gương -> mặc định KHÔNG lật lại. Video điện thoại: --flip.

⚠️ Take đã dùng để TRAIN thì điểm ở đây lạc quan. Muốn điểm công bằng: quay take
MỚI (1_record.py), để riêng một thư mục, chấm bằng --dir trước khi đưa vào train.

Cách đọc
    frame dung  % frame model ra đúng nhãn (sau cửa chặn tau)
    lenh        lệnh command_logic phát ra trong take, kèm giây phát
    dat         lệnh đúng đầu tiên có phát không (NEGATIVE: không phát lệnh nguy hiểm nào)
    SAI         số lệnh khác nhãn thật mà không phải HOVER  -> con số phải bằng 0
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "gesture_perception"))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "gesture_command"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gesture_perception import settings as S  # noqa: E402
from gesture_perception.perception_core import GesturePerception  # noqa: E402
from gesture_command.command_logic import (  # noqa: E402
    CommandLogic, Params, Vehicle, ENABLED, LANDED_ON_GROUND, LANDED_IN_AIR)

DEPLOYED = ROOT / "ros2_ws" / "src" / "gesture_bringup" / "models" / "v3_pck_body25.tflite"


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def run_take(core, path, truth, proc_fps, flip):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1.0 < fps < 240.0 else 30.0

    logic = CommandLogic(Params())
    cmd_of = logic.cmd_map
    # Pilot đã gạt GUIDED. Take ASSUME_GUIDANCE bắt đầu KHOÁ (đó là việc của nó),
    # các take khác bắt đầu đã mở khoá để chấm đúng cử chỉ đó.
    v = Vehicle(connected=True, armed=True, mode="GUIDED",
                landed_state=LANDED_ON_GROUND if truth == "TAKEOFF" else LANDED_IN_AIR)
    if truth != "ASSUME_GUIDANCE":
        logic.state = ENABLED
        logic.was_in_control_mode = True

    core.filter.reset()
    labels, ms, fired = Counter(), [], []
    i, next_t = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if proc_fps > 0:
            if t + 1e-9 < next_t:
                continue                       # Pi còn đang bận frame trước
            next_t = t + 1.0 / proc_fps
        if flip:
            frame = cv2.flip(frame, 1)
        r = core.process(frame, t)
        ms.append(r.pose_ms + r.clf_ms)
        labels[r.label] += 1
        was = logic.cand_fired
        logic.on_gesture(r.label, t, v)
        logic.tick(t, v)
        if logic.cand_fired and not was:
            fired.append((logic.candidate, t))
    cap.release()

    want = cmd_of.get(truth)
    n = sum(labels.values())
    wrong = [(g, t) for g, t in fired if cmd_of.get(g) not in (want, "HOVER")]
    right = [(g, t) for g, t in fired if cmd_of.get(g) == want]
    if truth == "NEGATIVE":
        ok_take = not wrong
        first = None
    else:
        ok_take = bool(right)
        first = right[0][1] if right else None
    return {
        "take": path.name, "truth": truth, "frames": n,
        "frame_acc": labels[truth] / max(1, n),
        "ambiguous": labels[S.UNKNOWN_GESTURE] / max(1, n),
        "no_operator": labels[S.NO_OPERATOR] / max(1, n),
        "fired": fired, "ok": ok_take, "first_s": first, "wrong": len(wrong),
        "ms": ms, "top_wrong": [lb for lb, _ in labels.most_common() if lb != truth][:2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "video"))
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--model", default=str(DEPLOYED))
    ap.add_argument("--proc-fps", type=float, default=0.0,
                    help="gia lap may chi xu ly N fps (bo frame giua). 0 = moi frame")
    ap.add_argument("--flip", action="store_true", help="video CHUA lat guong (quay dien thoai)")
    ap.add_argument("--no-one-euro", action="store_true")
    ap.add_argument("--tau", type=float, default=-1.0,
                    help="thu nguong tin cay khac tau trong labels.json (<0 = giu nguyen)")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    vids = sorted(Path(args.dir).glob(args.glob))
    if not vids:
        raise SystemExit(f"Khong co video {args.glob} trong {args.dir}")
    core = GesturePerception(args.model, one_euro=not args.no_one_euro, flip=args.flip,
                             tau_override=None if args.tau < 0 else args.tau)
    clf = core.clf
    print(f"model {Path(args.model).name}: {clf.labels}, tau={clf.tau}")
    print(f"{len(vids)} video, {'moi frame' if args.proc_fps <= 0 else f'xu ly {args.proc_fps:g} fps'}, "
          f"one-euro {'tat' if args.no_one_euro else 'bat'}, lat {'co' if args.flip else 'khong'}\n")

    rows = []
    for p in vids:
        truth = p.name.split("__")[0]
        if truth not in clf.labels:
            print(f"  bo qua {p.name}: nhan '{truth}' khong co trong model")
            continue
        r = run_take(core, p, truth, args.proc_fps, args.flip)
        rows.append(r)
        fired = " ".join(f"{g}@{t:.1f}s" for g, t in r["fired"]) or "-"
        print(f"  {'dat ' if r['ok'] else 'TRUOT'} {'SAI=' + str(r['wrong']) if r['wrong'] else '     '} "
              f"{r['take']:<42} frame dung {100 * r['frame_acc']:5.1f}%  "
              f"mo {100 * r['ambiguous']:4.1f}%  lenh: {fired}")
    core.close()
    if not rows:
        return 1

    print("\n" + "=" * 92)
    print(f"{'nhan':<16}{'take':>5}{'dat':>6}{'lenh SAI':>10}{'frame dung':>12}{'lenh dau (s)':>14}   nham nhieu nhat")
    by = defaultdict(list)
    for r in rows:
        by[r["truth"]].append(r)
    for lb in clf.labels:
        rs = by.get(lb)
        if not rs:
            continue
        firsts = [r["first_s"] for r in rs if r["first_s"] is not None]
        conf = Counter(x for r in rs for x in r["top_wrong"]).most_common(1)
        print(f"{lb:<16}{len(rs):>5}{sum(r['ok'] for r in rs):>6}{sum(r['wrong'] for r in rs):>10}"
              f"{100 * np.mean([r['frame_acc'] for r in rs]):>11.1f}%"
              f"{(f'{np.median(firsts):.1f}' if firsts else '-'):>14}   {conf[0][0] if conf else '-'}")
    ms = [x for r in rows for x in r["ms"]]
    total_wrong = sum(r["wrong"] for r in rows)
    print("-" * 92)
    print(f"TONG: {sum(r['ok'] for r in rows)}/{len(rows)} take dat, {total_wrong} lenh SAI, "
          f"frame dung {100 * np.mean([r['frame_acc'] for r in rows]):.1f}%")
    print(f"XU LY tren may nay: {pct(ms, 50):.0f} ms/frame (trung vi), p95 {pct(ms, 95):.0f} ms "
          f"-> toi da ~{1000 / max(1e-6, pct(ms, 50)):.0f} fps")
    print("  (lenh dau tinh tu dau video, da gom thoi gian GIU 1 s cua command_logic)")
    if total_wrong:
        print("\n  !! CO LENH SAI. Xem take co SAI= o tren, quay them take cho cap nhan bi nham.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(["take", "truth", "frames", "frame_acc", "ambiguous", "no_operator",
                        "ok", "wrong", "first_s", "fired", "ms_median"])
            for r in rows:
                w.writerow([r["take"], r["truth"], r["frames"], f"{r['frame_acc']:.4f}",
                            f"{r['ambiguous']:.4f}", f"{r['no_operator']:.4f}", int(r["ok"]),
                            r["wrong"], r["first_s"] if r["first_s"] is not None else "",
                            " ".join(f"{g}@{t:.2f}" for g, t in r["fired"]),
                            f"{pct(r['ms'], 50):.1f}"])
        print(f"\n  ghi {args.csv}")
    return 1 if total_wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
