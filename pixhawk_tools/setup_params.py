#!/usr/bin/env python3
"""
Pixhawk Parameter Setup - doc / tim / ghi param, chay duoc tren Windows va Pi 5.

Thay cho fix_params.py cu (ban cu hardcode COM11, ghi mu khong kiem tra param
co ton tai khong, va tat sach moi arming check).

BA VIEC TOOL NAY LAM DUOC:

  1. NOI CHO ANH BIET DANG CHAY FIRMWARE GI.
     Neu board dang chay PX4 (firmware mac dinh cua nhieu Pixhawk ban san) thi
     KHONG BAO GIO co param ten ARMING_CHECK - do la ten cua ArduPilot. PX4
     dung COM_ARM_* / CBRK_*. Day la nguyen nhan so 1 cua viec "tim trong full
     parameter list ma khong thay ARMING_CHECK".

  2. TAI TOAN BO BANG PARAM ROI MOI GHI.
     Co bang roi thi: tim duoc param theo chuoi con, biet param nao KHONG ton
     tai tren firmware nay, va tu chon dung ten khi ArduPilot doi ten giua cac
     phien ban (BRD_SAFETYENABLE -> BRD_SAFETY_DEFLT, GPS_TYPE -> GPS1_TYPE).

  3. TU BAT KIP VIEC ARDUPILOT DOI TEN VA DOI DON VI PARAM.
     ArduPilot 4.6+ doi ten hang loat VA doi don vi sang he SI:
         ARMING_CHECK (bat check nao)  ->  ARMING_SKIPCHK (BO QUA check nao)
         PILOT_SPEED_UP  150 cm/s      ->  PILOT_SPD_UP    1.5 m/s
         PILOT_ACCEL_Z   150 cm/s/s    ->  PILOT_ACC_Z     1.5 m/s/s
         ANGLE_MAX      1500 centi-do  ->  ATC_ANGLE_MAX    15 do
         LOITER_SPEED   1250 cm/s      ->  LOIT_SPEED_MS  12.5 m/s
         SYSID_MYGCS                   ->  MAV_GCS_SYSID
     Ghi nham don vi la sai 100 lan. Nen profile luu SAN gia tri rieng cho
     tung ten, va tool chon theo ten nao that su co tren firmware dang chay.

CACH DUNG
    python pixhawk_tools/setup_params.py --info
    python pixhawk_tools/setup_params.py --find ARMING
    python pixhawk_tools/setup_params.py --get ARMING_SKIPCHK --get PILOT_SPD_UP
    python pixhawk_tools/setup_params.py --profile althold --dry-run
    python pixhawk_tools/setup_params.py --profile althold
    python pixhawk_tools/setup_params.py --arming all      # chay du moi kiem tra
    python pixhawk_tools/setup_params.py --set PILOT_SPD_UP=1.5 --set ATC_ANGLE_MAX=15
    python pixhawk_tools/setup_params.py --reset-default        # XOA SACH, phai go xac nhan

    --port /dev/ttyACM0     (mac dinh: tu do)
    --baud 115200           (mac dinh: thu lan luot 115200, 57600, 921600)
"""

import argparse
import platform
import sys
import time

from pymavlink import mavutil

try:
    import serial.tools.list_ports as list_ports
except ImportError:
    list_ports = None


# ============================================================================
#  ARMING CHECK - HAI SO DO KHAC NHAU TUY PHIEN BAN
# ============================================================================
#   <= 4.5 :  ARMING_CHECK    bitmask "CHAY nhung check nao".  1 = chay het.
#   >= 4.6 :  ARMING_SKIPCHK  bitmask "BO QUA nhung check nao". 0 = chay het.
#
# Hai cai NGUOC NGHIA NHAU. Viet nham so cua cai nay vao cai kia la tu tay tat
# het kiem tra an toan ma van tuong dang bat.
#
# Bang bit duoi day CHI dung cho ARMING_CHECK (ban cu). Voi ARMING_SKIPCHK tool
# CO Y khong doan bit: chi nhan 'all' (= 0, chay du moi check) hoac mot so cu
# the do nguoi dung tu tra trong Mission Planner. Doan bit tren mot tham so an
# toan la cach nhanh nhat de tat nham dung cai check dang bao ve minh.
ARMING_BITS = {
    "all":          1 << 0,
    "baro":         1 << 1,
    "compass":      1 << 2,
    "gps":          1 << 3,    # GPS lock
    "ins":          1 << 4,    # accel + gyro
    "params":       1 << 5,
    "rc":           1 << 6,    # RC channels da calibrate chua
    "board_volt":   1 << 7,
    "battery":      1 << 8,
    "logging":      1 << 10,
    "safety":       1 << 11,   # nut safety switch tren board
    "gps_config":   1 << 12,
    "system":       1 << 13,
    "mission":      1 << 14,
    "rangefinder":  1 << 15,
    "camera":       1 << 16,
    "aux_auth":     1 << 17,
    "vision":       1 << 18,
    "fft":          1 << 19,
}
# bit 9 (512) la Airspeed - chi co y nghia voi ArduPlane, bo qua o day.

_ALL_CHECK_BITS = sum(v for k, v in ARMING_BITS.items() if k != "all")


def arming_mask(spec):
    """
    'all'                      -> 1
    'none'                     -> 0
    'all-except gps,compass'   -> bitmask moi check tru 2 cai do
    'baro,ins,rc'              -> chi bat dung 3 check nay
    """
    spec = spec.strip().lower()
    if spec == "all":
        return 1
    if spec in ("none", "0", "off"):
        return 0

    if spec.startswith("all-except"):
        names = spec[len("all-except"):].lstrip(" =:,")
        mask = _ALL_CHECK_BITS
        for n in [x.strip() for x in names.split(",") if x.strip()]:
            if n not in ARMING_BITS:
                raise ValueError("Khong biet check '%s'. Co: %s" % (n, ", ".join(ARMING_BITS)))
            mask &= ~ARMING_BITS[n]
        return mask

    mask = 0
    for n in [x.strip() for x in spec.split(",") if x.strip()]:
        if n not in ARMING_BITS:
            raise ValueError("Khong biet check '%s'. Co: %s" % (n, ", ".join(ARMING_BITS)))
        mask |= ARMING_BITS[n]
    return mask


def describe_skipchk(value):
    """Giai ma ARMING_SKIPCHK (ban 4.6+, nghia NGUOC voi ARMING_CHECK)."""
    v = int(value)
    if v == 0:
        return "0 = KHONG bo qua check nao, chay DU moi kiem tra (an toan nhat)"
    return ("%d = dang BO QUA mot so kiem tra. Tra y nghia tung bit trong "
            "Mission Planner > Full Parameter List > ARMING_SKIPCHK" % v)


def describe_arming(value):
    """Giai ma mot gia tri ARMING_CHECK ra danh sach check dang bat."""
    v = int(value)
    if v == 0:
        return "0 = TAT HET MOI KIEM TRA (chi dung khi da thao canh quat)"
    if v & ARMING_BITS["all"]:
        return "1 = BAT TAT CA cac kiem tra"
    on = [k for k, b in ARMING_BITS.items() if k != "all" and (v & b)]
    off = [k for k, b in ARMING_BITS.items() if k != "all" and not (v & b)]
    return "%d = bat [%s] | TAT [%s]" % (v, ", ".join(on), ", ".join(off) or "khong co")


# ============================================================================
#  PROFILE - nhom param theo muc dich
# ============================================================================
# Moi entry: (tuple alias ten param, gia tri, giai thich). Tool dung alias nao
# thuc su ton tai tren firmware dang chay.

# Moi entry: ([(ten, gia tri), (ten cu, gia tri cu)], giai thich).
# Tool dung UNG VIEN DAU TIEN thuc su co tren firmware dang chay. Moi ung vien
# mang gia tri RIENG vi don vi khac nhau giua cac phien ban.

PROFILES = {
    # ------------------------------------------------------------------
    "althold": [
        ([("PILOT_SPD_UP", 1.5),          # 4.6+  m/s
          ("PILOT_SPEED_UP", 150.0)],     # <=4.5 cm/s
         "Toc do leo toi da = 1.5 m/s"),
        ([("PILOT_SPD_DN", 1.0),
          ("PILOT_SPEED_DN", 100.0)],
         "Toc do ha toi da = 1.0 m/s"),
        ([("PILOT_ACC_Z", 1.5),
          ("PILOT_ACCEL_Z", 150.0)],
         "Gia toc len/xuong = 1.5 m/s/s. Nho = muot, phan hoi cham hon"),
        ([("THR_DZ", 150.0)],
         "Vung chet quanh giua can ga trong ALT_HOLD (PWM, khong doi don vi)"),
        ([("MOT_HOVER_LEARN", 2.0)],
         "2 = tu hoc ga hover va LUU. ALT_HOLD on dinh hay khong la o day"),
        ([("ATC_ANGLE_MAX", 15.0),        # 4.6+  do
          ("ANGLE_MAX", 1500.0)],         # <=4.5 centi-do
         "Goc nghieng toi da = 15 do, cham va an toan cho giai doan test"),
    ],
    # ------------------------------------------------------------------
    # Bench: da THAO CANH QUAT, de tren ban, chi can ARM duoc.
    # KHONG dung arming check o day - dung --arming de ro rang la minh dang
    # co y noi long kiem tra an toan, khong phai lam roi quen mat.
    "bench": [
        ([("BRD_SAFETY_DEFLT", 0.0), ("BRD_SAFETYENABLE", 0.0)],
         "Bo nut safety switch tren board"),
        ([("FS_THR_ENABLE", 0.0)],
         "Tat failsafe mat song tay cam khi test tren ban"),
    ],
    # ------------------------------------------------------------------
    # Bay trong nha / khong GPS. ALT_HOLD khong can GPS (chi can baro + IMU),
    # nhung EKF van keu neu no dang trong cho nguon vi tri ngang.
    "indoor-nogps": [
        ([("EK3_SRC1_POSXY", 0.0)], "Khong co nguon vi tri ngang"),
        ([("EK3_SRC1_VELXY", 0.0)], "Khong co nguon van toc ngang"),
        ([("EK3_SRC1_POSZ", 1.0)], "Cao do lay tu BARO - day la cai ALT_HOLD can"),
        ([("EK3_SRC1_VELZ", 0.0)], "Khong co nguon van toc doc rieng"),
        ([("EK3_SRC1_YAW", 1.0)], "Yaw lay tu la ban. Dat 0 neu da tat han compass"),
    ],
}


# ============================================================================
#  KET NOI
# ============================================================================

def candidate_ports():
    """Do cong co kha nang la Pixhawk, khac nhau giua Windows va Pi."""
    found = []
    if list_ports is not None:
        for p in list_ports.comports():
            desc = ("%s %s" % (p.description, p.manufacturer or "")).lower()
            # Cong COM ao cua Bluetooth luon co san tren Windows va khong bao gio
            # la Pixhawk. Thu chung ton 15s moi cong -> loai thang tay.
            if "bluetooth" in desc:
                continue
            score = 0
            # Pixhawk qua USB hien ra la CDC ACM. Cac VID duoi day la nha san
            # xuat flight controller quen thuoc.
            if p.vid in (0x26AC, 0x1209, 0x2DAE, 0x3162):
                score += 10
            if any(k in desc for k in ("px4", "pixhawk", "ardupilot", "cube", "acm")):
                score += 5
            found.append((score, p.device, p.description))
    if platform.system() != "Windows":
        # UART tren GPIO cua Pi khong hien trong list_ports nhu thiet bi USB.
        known = [f[1] for f in found]
        for dev in ("/dev/ttyACM0", "/dev/ttyACM1", "/dev/serial0",
                    "/dev/ttyAMA0", "/dev/ttyUSB0"):
            if dev not in known:
                found.append((1, dev, "UART/serial co dinh"))
    found.sort(key=lambda x: -x[0])
    return [(d, desc) for _s, d, desc in found]


def connect(port, bauds):
    ports = [(port, "do nguoi dung chi dinh")] if port else candidate_ports()
    if not ports:
        print("Khong thay cong serial nao.")
        return None

    print("Cong se thu:")
    for d, desc in ports:
        print("     %-20s %s" % (d, desc))

    for dev, _desc in ports:
        for baud in bauds:
            print("\n[*] %s @ %d ..." % (dev, baud), end=" ")
            sys.stdout.flush()
            try:
                m = mavutil.mavlink_connection(dev, baud=baud)
                hb = m.wait_heartbeat(timeout=5)
            except Exception as e:
                print("loi: %s" % e)
                continue
            if hb is None:
                print("khong co heartbeat")
                try:
                    m.close()
                except Exception:
                    pass
                continue
            print("OK")
            return m
    print("\nKhong ket noi duoc.")
    return None


def vehicle_heartbeat(master, timeout=6.0):
    """
    Lay HEARTBEAT CUA XE BAY, bo qua heartbeat cua tram mat dat.

    Mission Planner / MAVProxy cung phat HEARTBEAT, voi type=MAV_TYPE_GCS(6) va
    autopilot=MAV_AUTOPILOT_INVALID(8). ArduPilot dinh tuyen goi tin giua cac
    cong cua no, nen heartbeat cua GCS cam o cong khac VAN XUAT HIEN tren cong
    minh dang mo. Bat nham cai do thi:
      - bao nham "day khong phai ArduPilot"
      - te hon: target_system bi dat thanh 255 (id cua GCS), moi lenh gui di
        deu sai dia chi va board lang le bo qua het.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb is None:
            continue
        if hb.type == mavutil.mavlink.MAV_TYPE_GCS:
            continue
        if hb.autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
            continue
        return hb
    return None


def lock_onto_vehicle(master):
    """Dam bao target_system tro dung xe bay, khong phai GCS."""
    hb = vehicle_heartbeat(master)
    if hb is None:
        return None
    src_sys, src_comp = hb.get_srcSystem(), hb.get_srcComponent()
    if master.target_system != src_sys or master.target_component != src_comp:
        print("  (chinh lai muc tieu: system %s comp %s)" % (src_sys, src_comp))
        master.target_system = src_sys
        master.target_component = src_comp
    return hb


AUTOPILOT_NAMES = {
    3:  "ArduPilot (ArduCopter/ArduPlane)  -> param la ARMING_CHECK",
    12: "PX4                               -> KHONG co ARMING_CHECK, dung COM_ARM_*",
    0:  "Generic",
}


def show_info(master):
    """In firmware dang chay. Buoc dau tien phai lam khi param 'bien mat'."""
    hb = lock_onto_vehicle(master)
    print("\n" + "=" * 66)
    print("  FIRMWARE DANG CHAY TREN BOARD")
    print("=" * 66)
    if hb is None:
        print("  Khong doc duoc HEARTBEAT.")
        return
    ap = hb.autopilot
    print("  System %s / Component %s" % (master.target_system, master.target_component))
    print("  autopilot = %s  ->  %s" % (ap, AUTOPILOT_NAMES.get(ap, "khong ro")))
    print("  type      = %s   base_mode = %s" % (hb.type, hb.base_mode))

    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES,
        0, 1, 0, 0, 0, 0, 0, 0)
    v = master.recv_match(type="AUTOPILOT_VERSION", blocking=True, timeout=3)
    if v:
        fv = v.flight_sw_version
        print("  firmware  = %d.%d.%d" % ((fv >> 24) & 0xFF, (fv >> 16) & 0xFF, (fv >> 8) & 0xFF))
    if ap != 3:
        print("")
        print("  !! DAY KHONG PHAI ARDUPILOT. Muon dung ARMING_CHECK va cac param")
        print("     PILOT_SPEED_* thi phai nap lai firmware ArduCopter")
        print("     (Mission Planner > Setup > Install Firmware).")
    print("=" * 66)


# ============================================================================
#  DOC / GHI PARAM
# ============================================================================

def fetch_all(master, timeout=60.0):
    """
    Tai toan bo bang param ve. Cham (vai chuc giay, hang nghin param) nhung lam
    mot lan roi moi thao tac sau do biet chac param nao co that.
    """
    print("\n[*] Dang tai bang param (mat 15-60s)...", end=" ")
    sys.stdout.flush()
    master.mav.param_request_list_send(master.target_system, master.target_component)

    params = {}
    expected = None
    t0 = time.time()
    last_rx = time.time()
    while time.time() - t0 < timeout:
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=2)
        if msg is None:
            # Im lang 3s lien tiep thi coi nhu het - mot so board khong gui dung
            # param_count, cho tiep chi to treo tool.
            if time.time() - last_rx > 3:
                break
            continue
        last_rx = time.time()
        name = msg.param_id
        if isinstance(name, bytes):
            name = name.decode("utf-8", "ignore")
        params[name.strip("\x00")] = msg.param_value
        expected = msg.param_count
        if expected and len(params) >= expected:
            break

    print("xong: %d/%s param" % (len(params), expected or "?"))
    return params


def resolve(candidates, table):
    """
    Chon ung vien dau tien thuc su co tren firmware nay -> (ten, gia tri).

    Quan trong: tra ve CA gia tri, vi cung mot y nghia nhung don vi khac nhau
    giua cac phien ban (1.5 m/s so voi 150 cm/s).
    """
    for name, value in candidates:
        if name in table:
            return name, value
    return None, None


def set_param(master, name, value, table, dry=False):
    if table and name not in table:
        print("  [x] %-22s KHONG ton tai tren firmware nay - bo qua" % name)
        return False
    old = table.get(name) if table else None
    old_s = ("%g" % old) if old is not None else "?"
    if dry:
        print("  [ ] %-22s %10s -> %g   (dry-run, chua ghi)" % (name, old_s, value))
        return True

    master.mav.param_set_send(
        master.target_system, master.target_component,
        name.encode("utf-8"), float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    t0 = time.time()
    while time.time() - t0 < 4:
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        if msg is None:
            continue
        got = msg.param_id
        if isinstance(got, bytes):
            got = got.decode("utf-8", "ignore")
        if got.strip("\x00") == name:
            ok = abs(msg.param_value - float(value)) < 1e-4
            print("  [%s] %-22s %10s -> %g" % ("OK" if ok else "??", name, old_s, msg.param_value))
            if table is not None:
                table[name] = msg.param_value
            return ok
    print("  [??] %-22s da gui nhung KHONG co xac nhan - kiem lai bang --get" % name)
    return False


def reset_to_default(master):
    """
    Xoa sach param ve mac dinh nha may.

    Cach cua ArduPilot: dat FORMAT_VERSION = 0 roi reboot. Luc khoi dong lai,
    firmware thay phien ban format sai nen khoi tao lai toan bo vung nho param.

    MAT LUON CALIBRATION: accel, compass, RC, ESC. Sau khi reset phai calibrate
    lai tu dau trong Mission Planner moi bay duoc.
    """
    print("\n" + "!" * 66)
    print("  RESET TOAN BO PARAM VE MAC DINH")
    print("  Mat het: hieu chuan accel, compass, RC, ESC, moi tinh chinh.")
    print("  Sau do BAT BUOC calibrate lai moi bay duoc.")
    print("!" * 66)
    if input("  Go dung chu  RESET  de xac nhan: ").strip() != "RESET":
        print("  Da huy.")
        return

    master.mav.param_set_send(
        master.target_system, master.target_component,
        b"FORMAT_VERSION", 0.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(1.0)
    print("  Dang reboot board...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
        0, 1, 0, 0, 0, 0, 0, 0)
    time.sleep(2.0)
    print("  Da gui. Rut nguon, cho 5s, cam lai roi ket noi kiem tra.")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Doc/ghi param Pixhawk (ArduPilot)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--port", help="COM11 / /dev/ttyACM0 / /dev/serial0. Bo trong = tu do")
    ap.add_argument("--baud", type=int, action="append",
                    help="Baud (lap lai duoc). Mac dinh: 115200, 57600, 921600")
    ap.add_argument("--info", action="store_true", help="Chi in firmware dang chay roi thoat")
    ap.add_argument("--find", metavar="CHUOI", action="append", default=[],
                    help="Tim param chua chuoi nay (vd: --find ARMING)")
    ap.add_argument("--get", metavar="TEN", action="append", default=[], help="Doc gia tri 1 param")
    ap.add_argument("--set", metavar="TEN=GIATRI", action="append", default=[], help="Ghi 1 param")
    ap.add_argument("--profile", action="append", default=[],
                    choices=sorted(PROFILES), help="Ghi ca nhom param theo muc dich")
    ap.add_argument("--arming", metavar="SPEC",
                    help="Dat arming check. 4.6+ (ARMING_SKIPCHK): 'all' = chay "
                         "du moi kiem tra, hoac mot so cu the. Ban cu "
                         "(ARMING_CHECK): all | none | 'all-except gps' | baro,ins,rc")
    ap.add_argument("--reset-default", action="store_true", help="XOA SACH param ve mac dinh")
    ap.add_argument("--dry-run", action="store_true", help="In ra se ghi gi, khong ghi that")
    args = ap.parse_args()

    bauds = args.baud or [115200, 57600, 921600]

    master = connect(args.port, bauds)
    if master is None:
        return 1

    try:
        show_info(master)
        if args.info:
            return 0

        if args.reset_default:
            reset_to_default(master)
            return 0

        need_table = bool(args.find or args.profile or args.set or args.arming or args.get)
        table = fetch_all(master) if need_table else {}

        # ---- tim ----
        for needle in args.find:
            hits = sorted(k for k in table if needle.upper() in k.upper())
            print("\n[?] Param chua '%s': %d ket qua" % (needle, len(hits)))
            for k in hits:
                print("     %-24s = %g" % (k, table[k]))
                if k == "ARMING_CHECK":
                    print("        %s" % describe_arming(table[k]))
                elif k == "ARMING_SKIPCHK":
                    print("        %s" % describe_skipchk(table[k]))
            if not hits:
                print("     (khong co - xem lai firmware o phan tren co phai ArduPilot khong)")

        # ---- doc ----
        for name in args.get:
            if name in table:
                print("\n[=] %s = %g" % (name, table[name]))
                if name == "ARMING_CHECK":
                    print("    %s" % describe_arming(table[name]))
                elif name == "ARMING_SKIPCHK":
                    print("    %s" % describe_skipchk(table[name]))
            else:
                print("\n[=] %s: KHONG ton tai tren firmware nay" % name)

        # ---- ghi theo profile ----
        for prof in args.profile:
            print("\n[W] PROFILE '%s'" % prof)
            for candidates, why in PROFILES[prof]:
                real, value = resolve(candidates, table)
                if real is None:
                    names = "/".join(n for n, _v in candidates)
                    print("  [x] %-22s khong co tren firmware nay - bo qua" % names)
                    continue
                print("      - %s" % why)
                set_param(master, real, value, table, args.dry_run)

        # ---- ghi ARMING_CHECK ----
        if args.arming:
            spec = args.arming.strip().lower()
            if "ARMING_SKIPCHK" in table:
                # ArduPilot 4.6+ : bitmask "BO QUA check nao", 0 = chay du.
                if spec == "all":
                    mask = 0
                elif spec.lstrip("-").isdigit():
                    mask = int(spec)
                else:
                    print("\nFirmware nay dung ARMING_SKIPCHK (nghia NGUOC voi")
                    print("ARMING_CHECK: bitmask nay la 'BO QUA check nao').")
                    print("Tool chi nhan:")
                    print("    --arming all      -> 0, chay DU moi kiem tra")
                    print("    --arming <so>     -> ghi thang so do")
                    print("Tra y nghia tung bit trong Mission Planner >")
                    print("Full Parameter List > ARMING_SKIPCHK, roi truyen so.")
                    print("Tool CO Y khong doan bit tren tham so an toan.")
                    return 2
                print("\n[W] ARMING_SKIPCHK -> %d" % mask)
                print("    %s" % describe_skipchk(mask))
                set_param(master, "ARMING_SKIPCHK", mask, table, args.dry_run)
            elif "ARMING_CHECK" in table:
                try:
                    mask = arming_mask(spec)
                except ValueError as e:
                    print("\nLOI: %s" % e)
                    return 2
                print("\n[W] ARMING_CHECK -> %d" % mask)
                print("    %s" % describe_arming(mask))
                set_param(master, "ARMING_CHECK", mask, table, args.dry_run)
            else:
                print("\nFirmware nay khong co ARMING_CHECK lan ARMING_SKIPCHK.")
                print("Kiem lai o phan FIRMWARE ben tren xem co phai ArduPilot khong.")
                return 2

        # ---- ghi thu cong ----
        for item in args.set:
            if "=" not in item:
                print("\nLOI: '%s' phai co dang TEN=GIATRI" % item)
                continue
            name, val = item.split("=", 1)
            print("\n[W] %s" % name.strip())
            set_param(master, name.strip().upper(), float(val), table, args.dry_run)

        if not args.dry_run and (args.profile or args.arming or args.set):
            print("\n" + "-" * 66)
            print("  Param da ghi vao flash ngay. Nen rut nguon cam lai mot lan")
            print("  cho chac, roi chay lai voi --get de xac nhan.")
            print("-" * 66)

    finally:
        master.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
