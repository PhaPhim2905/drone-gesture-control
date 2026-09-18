"""
Log Health - Doc file log .BIN cua Pixhawk, bao board co dau hieu hu khong
==========================================================================
CHI DOC FILE LOG. Khong ket noi, khong ghi gi xuong Pixhawk.

Kiem tra 4 nhom, moi nhom tach LUC DONG CO TAT va LUC DONG CO QUAY - loi do
nhieu dien / rung chi hien khi dong co keo dong, board dung yen van bao khoe:
    IMU     IMU.GH/AH (suc khoe), IMU.EG/EA (dem loi), 2 IMU co doc giong nhau
    RUNG    VIBE.VibeX/Y/Z, VIBE.Clip (so lan cam bien bi bao hoa)
    IOMCU   IOMC.Nerr/Nerr2/RSErr - chip phu xuat PWM MAIN OUT va doc RC IN
    MSG     "Internal errors", "not healthy", "inconsistent", "Crash"

CACH DUNG
    python pixhawk_tools/log_health.py "sitl_logs/flight_log.bin"

Muon ket luan duoc thi log phai co doan DONG CO QUAY (THAO CANH, arm, tang ga
30-60 s). Log chi co luc dung yen khong chung minh duoc board tot.
"""

import argparse
import io
import math
import platform
import sys

if platform.system() == "Windows":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pymavlink import mavutil

# Nguong theo huong dan ArduPilot "Measuring Vibration": duoi 30 m/s/s tot,
# tren 60 thuong gay mat vi tri/do cao; Clip phai bang 0.
VIBE_WARN, VIBE_BAD = 30.0, 60.0
MOTOR_ON_PWM = 1150          # RCOU > muc nay = dong co dang quay
GYRO_DIFF_WARN_DPS = 10.0    # 2 IMU lech nhau hon muc nay (dps, trung vi) la dang ngo


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    args = ap.parse_args()

    m = mavutil.mavlink_connection(args.log, dialect="ardupilotmega")
    types = ["IMU", "VIBE", "IOMC", "MSG", "ERR", "RCOU", "PARM"]

    motor_on = False
    t_motor_on = None
    imu = {}        # I -> dict
    vibe = {}       # IMU -> dict
    iomc = {"off": None, "on_first": None, "last": None, "t_first_err": None}
    msgs = []
    parm = {}
    last_gyro = {}
    gyro_diff = []

    while True:
        msg = m.recv_match(type=types)
        if msg is None:
            break
        tp = msg.get_type()
        if tp == "PARM":
            parm[msg.Name] = msg.Value
            continue
        t = msg.TimeUS / 1e6

        if tp == "RCOU":
            on = max(getattr(msg, f"C{i}", 0) for i in range(1, 9)) > MOTOR_ON_PWM
            if on and t_motor_on is None:
                t_motor_on = t
            motor_on = on

        elif tp == "IMU":
            d = imu.setdefault(msg.I, {"n": 0, "bad": 0, "t_bad": None, "eg": 0, "ea": 0,
                                        "bad_off": 0, "n_off": 0})
            # "dung yen" = truoc lan dau dong co quay (sau khi roi thi board da hong, khong tinh)
            before = t_motor_on is None
            d["n"] += 1
            d["n_off"] += 1 if before else 0
            if msg.GH != 1 or msg.AH != 1:
                d["bad"] += 1
                d["bad_off"] += 1 if before else 0
                d["t_bad"] = d["t_bad"] or t
            d["eg"], d["ea"] = max(d["eg"], msg.EG), max(d["ea"], msg.EA)
            g = (msg.GyrX, msg.GyrY, msg.GyrZ)
            last_gyro[msg.I] = (t, g)
            if msg.I == 1 and 0 in last_gyro and motor_on:
                t0, g0 = last_gyro[0]
                # bo doan nhao lon (roi/lat) - khi do con quay co the bao hoa
                if abs(t - t0) < 0.01 and max(map(abs, g + g0)) < 3.0:
                    gyro_diff.append(math.degrees(math.dist(g, g0)))

        elif tp == "VIBE":
            d = vibe.setdefault(msg.IMU, {"on": [], "clip_on0": None, "clip": 0})
            if motor_on:
                d["on"].append(max(msg.VibeX, msg.VibeY, msg.VibeZ))
                if d["clip_on0"] is None:
                    d["clip_on0"] = msg.Clip
            d["clip"] = msg.Clip

        elif tp == "IOMC":
            s = (msg.Nerr, msg.Nerr2, msg.RSErr)
            if not motor_on and iomc["on_first"] is None:
                iomc["off"] = s
            if motor_on and iomc["on_first"] is None:
                iomc["on_first"] = s
            if iomc["last"] and s != iomc["last"] and iomc["t_first_err"] is None:
                iomc["t_first_err"] = t
            if iomc["last"] is None and any(s):
                iomc["t_first_err"] = t
            iomc["last"] = s

        elif tp == "MSG":
            txt = msg.Message
            if any(k in txt.lower() for k in ("internal err", "not healthy", "unhealthy", "inconsistent",
                                               "crash", "calibration", "iomcu", "clip", "vibration")):
                msgs.append((t, txt))
            if txt.startswith(("ArduCopter", "Pixhawk", "Frame")):
                msgs.append((t, txt))

        elif tp == "ERR":
            msgs.append((t, f"ERR Subsys={msg.Subsys} ECode={msg.ECode}"))

    verdict = []

    def say(level, text):
        verdict.append(level)
        print(f"  [{level:^8s}] {text}")

    print(f"\nLOG: {args.log}")
    print(f"Dong co bat dau quay: {t_motor_on:.1f} s" if t_motor_on else
          "Dong co KHONG quay trong log nay -> chi kiem tra duoc luc dung yen")

    print("\n== IMU ==")
    for i in sorted(imu):
        d = imu[i]
        gid = parm.get("INS_GYR_ID" if i == 0 else f"INS_GYR{i + 1}_ID")
        tag = f"(INS_GYR{'' if i == 0 else i + 1}_ID={int(gid)})" if gid else ""
        if d["bad"] == 0 and d["eg"] == 0 and d["ea"] == 0:
            say("OK", f"IMU{i} {tag}: luon khoe, 0 loi")
        else:
            where = "ca luc dung yen" if d["bad_off"] else "CHI khi dong co quay"
            say("LOI", f"IMU{i} {tag}: {d['bad']}/{d['n']} mau bao hong tu {d['t_bad']:.1f} s ({where}), "
                       f"dem loi gyro={d['eg']} accel={d['ea']}")
    if gyro_diff:
        med, p99 = pct(gyro_diff, 0.5), pct(gyro_diff, 0.99)
        lvl = "CANH BAO" if med > GYRO_DIFF_WARN_DPS else "OK"
        say(lvl, f"2 IMU so voi nhau khi bay: lech trung vi {med:.1f} dps, 99% {p99:.1f} dps")

    print("\n== RUNG (khi dong co quay) ==")
    for i in sorted(vibe):
        d = vibe[i]
        if not d["on"]:
            print(f"  IMU{i}: khong co mau khi dong co quay")
            continue
        mx, p50 = max(d["on"]), pct(d["on"], 0.5)
        clip_new = d["clip"] - (d["clip_on0"] or 0)
        lvl = "LOI" if p50 > VIBE_BAD else "CANH BAO" if (p50 > VIBE_WARN or clip_new > 0) else "OK"
        say(lvl, f"IMU{i}: rung trung vi {p50:.0f}, max {mx:.0f} m/s/s (tot <{VIBE_WARN:.0f}); "
                 f"clipping tang {clip_new}")

    print("\n== IOMCU (chip xuat MAIN OUT + doc RC IN) ==")
    if iomc["last"] is None:
        print("  Khong co message IOMC (board khong co IOMCU hoac BRD_IO_ENABLE=0)")
    else:
        off = iomc["off"] or (0, 0, 0)
        on0 = iomc["on_first"] or off
        last = iomc["last"]
        grew = last[1] - on0[1]
        if last == (0, 0, 0):
            say("OK", "0 loi truyen thong suot log")
        else:
            note = "bat dau SAU khi dong co quay" if off == (0, 0, 0) and iomc["t_first_err"] and t_motor_on \
                and iomc["t_first_err"] >= t_motor_on else "co tu luc dung yen"
            say("LOI", f"loi truyen thong Nerr={last[0]} Nerr2={last[1]} RSErr={last[2]}, "
                       f"tang {grew} khi dong co quay; loi dau tien {iomc['t_first_err']:.1f} s ({note})")

    print("\n== THONG BAO TU PIXHAWK ==")
    for t, txt in msgs:
        bad = any(k in txt.lower() for k in ("internal err", "not healthy", "unhealthy", "inconsistent", "crash"))
        print(f"  {t:7.1f} s  {'!! ' if bad else '   '}{txt}")
        if "internal err" in txt.lower():
            verdict.append("LOI")

    print("\n== KET LUAN ==")
    if "LOI" in verdict:
        print("  BOARD CO DAU HIEU LOI PHAN CUNG / NHIEU DIEN. Khong bay drone nang voi board nay.")
    elif "CANH BAO" in verdict:
        print("  Chua thay loi phan cung, nhung co canh bao (rung/lech IMU) can xu ly truoc khi bay.")
    elif t_motor_on is None:
        print("  Khong thay loi, NHUNG log khong co doan dong co quay -> chua ket luan duoc.")
    else:
        print("  Khong thay dau hieu loi trong log nay.")
    return 1 if "LOI" in verdict else 0


if __name__ == "__main__":
    raise SystemExit(main())
