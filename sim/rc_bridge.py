#!/usr/bin/env python3
"""
Tay cầm THẬT điều khiển drone ẢO: đọc kênh RC từ Pixhawk thật, đẩy sang SITL.

    receiver tay cầm ─► RC IN Pixhawk 6X ─USB─► máy này ─► SITL (RC override)

    python sim/rc_bridge.py                       # tự dò cổng Pixhawk, SITL ở 127.0.0.1:5762
    python sim/rc_bridge.py --port COM7
    python sim/rc_bridge.py --show                # chỉ xem kênh, KHÔNG gửi sang SITL

Chạy trên Windows (Python có pymavlink) hoặc trong WSL/Pi. WSL dùng
networkingMode=mirrored nên Windows gọi 127.0.0.1:5762 là tới SITL trong WSL.

AN TOÀN
    Pixhawk thật ở đây CHỈ là bộ đọc receiver. Script KHÔNG gửi lệnh nào tới
    Pixhawk thật ngoài yêu cầu tốc độ gửi message; không arm, không đổi mode,
    không ghi tham số. Vẫn nên rút nguồn ESC / tháo cánh.

ÁNH XẠ KÊNH
    Dải PWM tay cầm (RCn_MIN/TRIM/MAX đọc từ Pixhawk thật) được co giãn sang dải
    RCn_MIN/TRIM/MAX của SITL, để cần ga thấp nhất trên tay = ga thấp nhất trong
    SITL, cần giữa = giữa. Kênh ga và các kênh công tắc (5..8) co giãn tuyến tính
    MIN..MAX; roll/pitch/yaw co giãn hai nửa quanh TRIM.

MẤT SÓNG
    Pixhawk thật báo RC receiver không khoẻ (SYS_STATUS) hoặc mất RC_CHANNELS
    > 0.5 s  ->  ngừng override + đặt SIM_RC_FAIL=1 trên SITL. Sau RC_OVERRIDE_TIME
    SITL vào failsafe radio y như drone thật mất sóng. Có sóng lại -> SIM_RC_FAIL=0.
"""

import argparse
import sys
import time

from pymavlink import mavutil

NCH = 8
# VID USB: ArduPilot chung, Holybro, 3DR/PX4, CubePilot
PIXHAWK_VIDS = {0x1209, 0x3162, 0x26AC, 0x2DAE}
RC_RX_BIT = mavutil.mavlink.MAV_SYS_STATUS_SENSOR_RC_RECEIVER


def find_port():
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    hits = [p for p in ports if p.vid in PIXHAWK_VIDS
            or any(k in (p.description or "") for k in ("Pixhawk", "ArduPilot", "PX4"))]
    for p in ports:
        print(f"    {p.device:14s} {p.description}  VID={p.vid and hex(p.vid)}")
    if not hits:
        sys.exit("Khong thay Pixhawk. Cam USB hoac dung --port.")
    # Pixhawk có 2 cổng ảo (MAVLink + SLCAN): cổng số nhỏ hơn là MAVLink
    return sorted(hits, key=lambda p: p.device)[0].device


def read_params(conn, names, timeout=10.0):
    """Đọc từng tham số bằng PARAM_REQUEST_READ. Thiếu tham số nào thì bỏ qua."""
    out = {}
    for name in names:
        end = time.time() + timeout / len(names) + 0.5
        conn.mav.param_request_read_send(conn.target_system, conn.target_component, name.encode(), -1)
        while time.time() < end:
            msg = conn.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.3)
            if msg and msg.param_id == name:
                out[name] = msg.param_value
                break
    return out


RCMAP_NAMES = ("RCMAP_ROLL", "RCMAP_PITCH", "RCMAP_THROTTLE", "RCMAP_YAW")


def rc_ranges(conn, who):
    names = ([f"RC{i}_{k}" for i in range(1, NCH + 1) for k in ("MIN", "TRIM", "MAX")]
             + list(RCMAP_NAMES) + [f"RC{i}_REVERSED" for i in range(1, NCH + 1)])
    p = read_params(conn, names)
    missing = [n for n in names[:3 * NCH] if n not in p]
    if missing:
        print(f"  ! {who}: khong doc duoc {len(missing)} tham so RC, dung mac dinh 1100/1500/1900")
    rng = {i: (p.get(f"RC{i}_MIN", 1100), p.get(f"RC{i}_TRIM", 1500), p.get(f"RC{i}_MAX", 1900))
           for i in range(1, NCH + 1)}
    rcmap = {n: int(p.get(n, d)) for n, d in zip(RCMAP_NAMES, (1, 2, 3, 4))}
    rev = {i for i in range(1, NCH + 1) if p.get(f"RC{i}_REVERSED", 0) >= 0.5}
    if rev:
        print(f"  {who}: kenh dao chieu (RCn_REVERSED=1): {sorted(rev)}")
    rcmap["reversed"] = rev
    return rng, rcmap


# Cùng ngưỡng với pixhawk_tools/lua/csc_arm.lua
CSC_EDGE = 0.85


def norm_trim(v, rng):
    """RC_Channel::norm_input của ArduPilot: -1..0..+1 quanh TRIM."""
    vmin, trim, vmax = rng
    if v < trim:
        return (v - trim) / max(1.0, trim - vmin)
    return (v - trim) / max(1.0, vmax - trim)


def norm_linear(v, rng):
    """RC_Channel::norm_input_ignore_trim: MIN..MAX -> -1..+1."""
    vmin, _, vmax = rng
    return 2.0 * (v - vmin) / max(1.0, vmax - vmin) - 1.0


def csc_status(out, rng, rcmap):
    """Hiện từng điều kiện chụm 2 cần mà csc_arm.lua kiểm tra, trên giá trị SITL nhận."""
    val = lambda name: out[rcmap[name] - 1]
    rn = lambda name: rng[rcmap[name]]
    parts = [
        ("ga",    norm_linear(val("RCMAP_THROTTLE"), rn("RCMAP_THROTTLE")), -1),
        ("yaw",   norm_trim(val("RCMAP_YAW"), rn("RCMAP_YAW")), +1),
        ("pitch", norm_trim(val("RCMAP_PITCH"), rn("RCMAP_PITCH")), +1),
        ("roll",  norm_trim(val("RCMAP_ROLL"), rn("RCMAP_ROLL")), -1),
    ]
    cells = []
    for name, x, want in parts:
        ok = x * want >= CSC_EDGE
        cells.append(f"{name}{x:+.2f}{'OK' if ok else ('NGUOC' if x * want <= -CSC_EDGE else '--')}")
    return "CSC[" + " ".join(cells) + "]"


def rescale(v, src, dst, linear):
    smin, strim, smax = src
    dmin, dtrim, dmax = dst
    if v <= 0:
        return 0
    if linear or not (smin < strim < smax):
        u = (v - smin) / max(1.0, smax - smin)
        out = dmin + u * (dmax - dmin)
    elif v < strim:
        out = dtrim - (strim - v) / max(1.0, strim - smin) * (dtrim - dmin)
    else:
        out = dtrim + (v - strim) / max(1.0, smax - strim) * (dmax - dtrim)
    return int(round(min(max(out, dmin), dmax)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="cong Pixhawk that (COM7, /dev/ttyACM0). Bo trong = tu do")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--sitl", default="tcp:127.0.0.1:5762", help="SITL SERIAL1 (5762) - khong dung cong cua MAVProxy")
    ap.add_argument("--rate", type=float, default=25.0, help="Hz gui override")
    ap.add_argument("--show", action="store_true", help="chi in kenh, khong gui SITL")
    args = ap.parse_args()

    port = args.port or find_port()
    print(f">>> Pixhawk that: {port}")
    real = mavutil.mavlink_connection(port, baud=args.baud, source_system=250)
    if not real.wait_heartbeat(timeout=15):
        sys.exit("Pixhawk that khong gui HEARTBEAT (sai cong? Mission Planner dang giu cong?)")
    for msg_id, hz in ((mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS, 50),
                       (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 5)):
        real.mav.command_long_send(real.target_system, real.target_component,
                                   mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                   msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)
    src_rng, src_map = rc_ranges(real, "Pixhawk that")
    thr_ch = src_map["RCMAP_THROTTLE"]

    sitl = None
    if not args.show:
        print(f">>> SITL: {args.sitl}")
        # SITL chỉ nhận RC override từ GCS có sysid = MAV_GCS_SYSID (mặc định 255)
        sitl = mavutil.mavlink_connection(args.sitl, source_system=255, source_component=190)
        if not sitl.wait_heartbeat(timeout=20):
            sys.exit("SITL khong tra loi. Da chay sim/run_sitl.sh chua?")
        dst_rng, dst_map = rc_ranges(sitl, "SITL")
        sitl.param_set_send("SIM_RC_FAIL", 0)
    else:
        dst_rng, dst_map = src_rng, src_map

    # RCn_REVERSED=1 trên Pixhawk thật đảo chiều bên trong ArduPilot, RC_CHANNELS vẫn gửi
    # PWM thô. SITL không đảo -> lật PWM ở đây cho drone ảo nhận đúng chiều đã chỉnh.
    def unreverse(v, ch):
        if v <= 0 or ch not in src_map["reversed"] or args.show:
            return v
        vmin, trim, vmax = src_rng[ch]
        mid = (vmin + vmax) / 2 if ch == thr_ch else trim
        return int(round(2 * mid - v))

    print("\n  kenh:     " + "  ".join(f"{i:>5d}" for i in range(1, NCH + 1)) + "     (T = kenh ga)")
    last_rc_t, rc_ok, lost = 0.0, True, False
    chans = [0] * NCH
    period, next_send, next_print = 1.0 / args.rate, 0.0, 0.0
    try:
        while True:
            msg = real.recv_match(type=["RC_CHANNELS", "SYS_STATUS"], blocking=True, timeout=0.05)
            now = time.time()
            if msg is not None and msg.get_type() == "RC_CHANNELS":
                chans = [getattr(msg, f"chan{i}_raw") for i in range(1, NCH + 1)]
                last_rc_t = now
            elif msg is not None:
                present = msg.onboard_control_sensors_present & RC_RX_BIT
                rc_ok = not present or bool(msg.onboard_control_sensors_health & RC_RX_BIT)

            now_lost = (not rc_ok) or (now - last_rc_t > 0.5)
            if now_lost != lost:
                lost = now_lost
                print(f"\n  {'!!! MAT SONG tay cam -> SITL SIM_RC_FAIL=1 (failsafe sau RC_OVERRIDE_TIME)' if lost else '>>> co song lai -> SIM_RC_FAIL=0'}")
                if sitl:
                    sitl.param_set_send("SIM_RC_FAIL", 1 if lost else 0)

            out = [rescale(unreverse(v, i + 1), src_rng[i + 1], dst_rng[i + 1], linear=(i + 1 == thr_ch or i >= 4))
                   for i, v in enumerate(chans)]
            if sitl and not lost and now >= next_send:
                next_send = now + period
                # kênh 9..18 = 0: không override, giữ nguyên
                sitl.mav.rc_channels_override_send(sitl.target_system, sitl.target_component, *out, *([0] * 10))
            if sitl:
                sitl.recv_match(blocking=False)  # xả bộ đệm, tránh đầy socket
            if now >= next_print:
                next_print = now + 0.2
                cells = " ".join(f"{v:>4d}{'T' if i + 1 == thr_ch else ' '}" for i, v in enumerate(chans))
                tag = "MAT SONG" if lost else ("xem" if args.show else "-> SITL")
                csc = csc_status(out, dst_rng, dst_map) if any(out) else ""
                print(f"\r  tay cam: {cells}  {tag}  {csc}   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n>>> dung. SITL ngung nhan override, sau RC_OVERRIDE_TIME quay ve RC gia lap cua no.")
        if sitl:
            sitl.mav.rc_channels_override_send(sitl.target_system, sitl.target_component, *([0] * 18))


if __name__ == "__main__":
    main()
