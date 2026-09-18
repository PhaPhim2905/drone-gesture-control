#!/usr/bin/env python3
"""
Lọc file .param lưu từ Pixhawk 6C thành bộ tham số dùng được trong SITL.

    python sim/make_sitl_params.py
    python sim/make_sitl_params.py --src sim/params/Parameters_Pixhawk_6c_V2.param

Ghi ra:
    sim/params/pixhawk6c_sitl.parm      nạp bằng  run_sitl.sh --real-params
    sim/params/pixhawk6c_sitl_report.md  giữ gì, bỏ gì, sửa gì, và vì sao

CHỈ ĐỌC file Pixhawk, không kết nối Pixhawk, không ghi gì xuống drone.

──────────────────────────────────────────────────────────────────────────────
NGUYÊN TẮC LỌC
    GIỮ   hành vi bay: mode, failsafe, tốc độ, gia tốc, PID, EKF nguồn, RC option
          -> đây là thứ quyết định drone phản ứng với lệnh GUIDED ra sao
    BỎ    phần cứng: hiệu chỉnh cảm biến, ID chip, cổng serial, chân ADC pin,
          dải PWM tay cầm, log, đèn/còi, CAN
          -> gắn với board thật; nạp vào SITL gây lỗi cảm biến giả
    SỬA   chỉ những chỗ SITL KHÔNG THỂ chạy nếu giữ nguyên, ghi rõ trong report
"""

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

KEEP_PREFIX = (
    "ACRO_", "ARMING_", "ATC_", "AUTOTUNE_", "AVOID_", "CIRCLE_", "EK3_",
    "FENCE_", "FLTMODE", "FRAME_", "FS_", "GUID_", "LAND_", "LOIT_", "MOT_",
    "PHLD_", "PILOT_", "PSC_", "RCMAP_", "RTL_", "SRTL_", "SURFTRAK_",
    "THROW_", "TKOFF_", "WP_", "BATT_FS_", "BATT_LOW_", "BATT_CRT_",
    "BATT_ARM_", "COMPASS_USE", "AHRS_EKF_TYPE", "AHRS_GPS_",
)
KEEP_EXACT = {
    "DISARM_DELAY", "FLIGHT_OPTIONS", "SIMPLE", "SUPER_SIMPLE", "THR_DZ",
    "RC_OPTIONS", "RC_OVERRIDE_TIME", "RC_FS_TIMEOUT", "BATT_CAPACITY",
    "INITIAL_MODE", "AUTO_OPTIONS",
}
KEEP_REGEX = (re.compile(r"^RC\d+_OPTION$"), re.compile(r"^SERVO\d+_FUNCTION$"))

# Trong nhóm GIỮ nhưng vẫn là phần cứng
DROP_EXACT = {"MOT_PWM_TYPE", "ARMING_CRSDP_IGN"}

MOTOR_FN = {33: 1, 34: 2, 35: 3, 36: 4}   # SERVOn_FUNCTION -> số động cơ


def load(path):
    out = {}
    for line in Path(path).read_text(encoding="utf8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, v = re.split(r"[,\s]+", line, maxsplit=1)
        out[k] = v.strip()
    return out


def keep(name):
    if name in DROP_EXACT:
        return False
    return (name.startswith(KEEP_PREFIX) or name in KEEP_EXACT
            or any(r.match(name) for r in KEEP_REGEX))


def num(v):
    return float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "params" / "pixhawk_export.param"),
                    help="file .param xuat tu Mission Planner (khong kem trong repo)")
    ap.add_argument("--out", default=str(HERE / "params" / "pixhawk6c_sitl.parm"))
    ap.add_argument("--batt-ah", type=float, default=16.0, help="dung luong pin that (Ah)")
    args = ap.parse_args()

    real = load(args.src)
    kept = {k: v for k, v in real.items() if keep(k)}

    # Firmware trên Pixhawk và SITL lệch nhau vài commit: cùng tham số, khác tên.
    # Đối chiếu ngày 2026-09-15 với SITL ArduCopter V4.8.0-dev (35356c656c).
    RENAMED = {"PSC_JERK_D": "PSC_D_JERK", "PSC_JERK_NE": "PSC_NE_JERK"}
    for old, new in RENAMED.items():
        if old in kept:
            kept[new] = kept.pop(old)
    dropped = sorted(k for k in real if k not in kept)
    changes, warn = [], []

    # ---- SỬA 1: động cơ 4 nằm ở output 5 -------------------------------
    # SITL/Gazebo đọc động cơ theo thứ tự output 1..4. Để động cơ 4 ở output 5
    # thì mô phỏng chỉ có 3 động cơ quay -> lật ngay khi cất cánh.
    motor_on = {}
    for k, v in real.items():
        m = re.match(r"^SERVO(\d+)_FUNCTION$", k)
        if m and int(float(v)) in MOTOR_FN:
            motor_on[MOTOR_FN[int(float(v))]] = int(m.group(1))
    if motor_on and any(motor_on[i] != i for i in motor_on):
        for k in list(kept):
            if re.match(r"^SERVO\d+_FUNCTION$", k):
                kept[k] = "0"
        for motor in sorted(motor_on):
            kept[f"SERVO{motor}_FUNCTION"] = str(32 + motor)
        changes.append(("SERVOn_FUNCTION",
                        "that: " + ", ".join(f"dong co {m} -> output {o}" for m, o in sorted(motor_on.items())),
                        "SITL: dong co n -> output n",
                        "mo phong doc dong co theo output 1..4; giu nguyen thi chi 3 dong co quay"))

    # ---- SỬA 2: pin 12S trong mô phỏng ---------------------------------
    kept["SIM_BATT_VOLTAGE"] = "50.4"
    kept["SIM_BATT_CAP_AH"] = f"{args.batt_ah:g}"
    changes.append(("SIM_BATT_VOLTAGE / SIM_BATT_CAP_AH", "(khong co tren Pixhawk)",
                    f"50.4 V / {args.batt_ah:g} Ah", "pin 12S cua drone that"))

    # ---- CẢNH BÁO: cấu hình thật đáng xem lại (KHÔNG sửa trong SITL) ----
    def g(k, d=None):
        return num(real[k]) if k in real else d

    modes = {0: "STABILIZE", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED", 5: "LOITER",
             6: "RTL", 9: "LAND", 16: "POSHOLD", 17: "BRAKE"}
    fm = [modes.get(int(g(f"FLTMODE{i}", -1)), str(int(g(f"FLTMODE{i}", -1)))) for i in range(1, 7)]
    if "GUIDED" not in fm:
        warn.append(f"Cong tac mode (kenh {int(g('FLTMODE_CH', 5))}) co {fm} - KHONG CO GUIDED. "
                    "Tren drone that pilot khong gat duoc sang GUIDED, cu chi se khong dieu khien duoc.")
    if "LOITER" not in fm:
        warn.append("Cong tac mode KHONG CO LOITER. STOP cua he cu chi tra ve LOITER; "
                    "can mot vi tri LOITER (hoac POSHOLD) de pilot lay lai quyen.")
    if g("FS_THR_ENABLE") == 0:
        warn.append("FS_THR_ENABLE = 0: MAT TIN HIEU TAY CAM KHONG CO FAILSAFE. Drone 12 kg nen bat (vd 1 = RTL / 3 = LAND).")
    if g("BATT_FS_LOW_ACT") == 0 and g("BATT_FS_CRT_ACT") == 0:
        warn.append("BATT_FS_LOW_ACT = BATT_FS_CRT_ACT = 0: failsafe pin TAT.")
    if g("BATT_LOW_VOLT", 0) and g("BATT_LOW_VOLT") < 30:
        warn.append(f"BATT_LOW_VOLT = {real['BATT_LOW_VOLT']} V la nguong cua pin 3S; pin 12S can ~42 V.")
    if g("BATT_CAPACITY", 0) and g("BATT_CAPACITY") < 8000:
        warn.append(f"BATT_CAPACITY = {real['BATT_CAPACITY']} mAh qua nho cho drone 12 kg 12S - kiem lai dung luong pin that.")
    if all(g(k, 1) == 0 for k in ("COMPASS_USE", "COMPASS_USE2", "COMPASS_USE3")):
        warn.append("COMPASS_USE = 0 tren ca 3 la ban, trong khi EK3_SRC1_YAW = 1 (la ban). "
                    "DA DO TREN SITL 2026-09-15: bo tham so nay bao 'Need Position Estimate', KHONG arm duoc o GUIDED "
                    "tren mat dat -> cu chi TAKEOFF se khong bao gio chay. Bat COMPASS_USE (sau khi hieu chinh la ban) "
                    "thi arm + takeoff binh thuong. SITL tam dung --compass-on.")
    if g("FENCE_ENABLE") == 0:
        warn.append("FENCE_ENABLE = 0: khong co hang rao dia ly.")
    if g("LOIT_SPEED_MS", 0) > 5:
        warn.append(f"LOIT_SPEED_MS = {real['LOIT_SPEED_MS']} m/s (mac dinh) - nhanh cho giai doan thu cu chi.")
    if g("RC5_OPTION") == 31:
        warn.append("RC5_OPTION = 31 (Motor Emergency Stop tren kenh 5) - DUNG nhu thiet ke an toan, giu nguyen.")

    out = Path(args.out)
    lines = [f"# Tu {Path(args.src).name} qua sim/make_sitl_params.py - KHONG sua tay, chay lai script."]
    lines += [f"{k},{v}" for k, v in sorted(kept.items())]
    out.write_text("\n".join(lines) + "\n", encoding="utf8")

    rep = out.with_name("pixhawk6c_sitl_report.md")
    r = [f"# Tham so Pixhawk 6C -> SITL", "",
         f"Nguon: `{Path(args.src).name}` ({len(real)} tham so). "
         f"Giu {len(kept) - 2}, bo {len(dropped)}, them 2 tham so SIM_.", "",
         "## Da sua cho SITL chay duoc", "", "| Tham so | Tren Pixhawk | Trong SITL | Ly do |", "|:--|:--|:--|:--|"]
    r += [f"| `{a}` | {b} | {c} | {d} |" for a, b, c, d in changes]
    r += ["", "## Can xem lai tren drone THAT (khong sua trong SITL, khong ghi xuong Pixhawk)", ""]
    r += [f"- {w}" for w in warn] or ["- (khong co)"]
    r += ["", "## Da bo (phan cung / hieu chinh)", "", ", ".join(f"`{d}`" for d in dropped)]
    rep.write_text("\n".join(r) + "\n", encoding="utf8")

    print(f"  giu {len(kept)} tham so -> {out}")
    print(f"  bo  {len(dropped)} tham so phan cung")
    print(f"  bao cao -> {rep}")
    print("\n  CAN XEM LAI TREN DRONE THAT:")
    for w in warn:
        print(f"    ! {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
