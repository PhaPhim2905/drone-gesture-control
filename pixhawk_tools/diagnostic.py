"""
Pixhawk Diagnostic v3 - Kiem tra ket noi serial (Windows + Raspberry Pi)
========================================================================
1. Liet ke cac cong serial dang co
2. Doc RAW BYTES o nhieu baud - buoc quan trong nhat khi noi bang UART:
   0 byte  = day TX/RX dau nguoc, hoac SERIALx_PROTOCOL tren Pixhawk chua bat
   Co byte nhung MAVLink fail = dung day, SAI BAUD
3. Doc message + STATUSTEXT (ly do khong ARM duoc nam o day)
4. Thu ARM - CHI KHI co co --try-arm

CACH DUNG
    python pixhawk_tools/diagnostic.py                      # tu do cong
    python pixhawk_tools/diagnostic.py --port /dev/ttyACM0
    python pixhawk_tools/diagnostic.py --port /dev/serial0 --baud 921600
    python pixhawk_tools/diagnostic.py --try-arm            # DA THAO CANH QUAT
"""

import argparse
import sys
import io
import platform
import time
import serial
import serial.tools.list_ports

# Fix unicode cho Windows console (terminal Linux von da UTF-8)
if platform.system() == "Windows":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pymavlink import mavutil

# ============ CAU HINH ============
# Cong mac dinh khi khong truyen --port. Tren Pi: /dev/ttyACM0 la Pixhawk cam
# USB, /dev/serial0 la UART tren chan GPIO.
DEFAULT_PORT = "COM11" if platform.system() == "Windows" else "/dev/ttyACM0"
BAUD_RATES = [115200, 57600, 921600, 38400, 9600]
# ==================================


def list_com_ports():
    """Liet ke tat ca cong COM dang co."""
    print("\n[0] Cac cong serial (USB) tren may:")
    print("-" * 60)
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("  Khong tim thay cong COM nao!")
        return []
    for p in ports:
        print(f"  {p.device:8s} | {p.description}")
        if p.manufacturer:
            print(f"           | Manufacturer: {p.manufacturer}")
        if p.vid is not None:
            print(f"           | VID:PID = {p.vid:04X}:{p.pid:04X}")
    print("-" * 60)
    return [p.device for p in ports]


def test_raw_serial(port, baud, duration=3):
    """Doc raw bytes tu cong COM de xem co du lieu khong."""
    print(f"\n  Thu raw serial {port} @ {baud}...")
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(0.5)  # Cho port on dinh

        total_bytes = 0
        start = time.time()
        while time.time() - start < duration:
            data = ser.read(256)
            if data:
                total_bytes += len(data)

        ser.close()

        if total_bytes > 0:
            print(f"  --> Nhan duoc {total_bytes} bytes trong {duration}s! Port HOAT DONG.")
            return True
        else:
            print(f"  --> 0 bytes. Khong co du lieu.")
            return False

    except serial.SerialException as e:
        print(f"  --> LOI: {e}")
        return False


def try_mavlink_connect(port, baud, timeout=8):
    """Thu ket noi MAVLink voi baud rate cu the."""
    print(f"\n  Thu MAVLink {port} @ {baud}...")
    try:
        master = mavutil.mavlink_connection(port, baud=baud)
        hb = master.wait_heartbeat(timeout=timeout)
        if hb:
            armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            print(f"  --> THANH CONG! System={master.target_system}, "
                  f"Component={master.target_component}")
            print(f"      Type={hb.type}, Autopilot={hb.autopilot}, "
                  f"Armed={'CO' if armed else 'KHONG'}")
            return master
        else:
            print(f"  --> Khong nhan duoc heartbeat (timeout {timeout}s)")
            master.close()
            return None
    except Exception as e:
        print(f"  --> LOI: {e}")
        return None


def read_messages(master, duration=5):
    """Doc tat ca message trong duration giay."""
    print(f"\n[3] Doc message ({duration} giay)...")
    print("-" * 60)

    start = time.time()
    msg_types_seen = {}
    important_msgs = []

    while time.time() - start < duration:
        msg = master.recv_match(blocking=True, timeout=1)
        if msg:
            msg_type = msg.get_type()
            msg_types_seen[msg_type] = msg_types_seen.get(msg_type, 0) + 1

            if msg_type == "STATUSTEXT":
                text = msg.text
                print(f"  *** STATUSTEXT: {text}")
                important_msgs.append(text)
            elif msg_type == "SYS_STATUS":
                print(f"  SYS_STATUS: battery={msg.voltage_battery}mV, "
                      f"current={msg.current_battery}cA, "
                      f"remaining={msg.battery_remaining}%")
            elif msg_type == "GPS_RAW_INT":
                print(f"  GPS: fix_type={msg.fix_type}, satellites={msg.satellites_visible}")
            elif msg_type == "HEARTBEAT" and msg.get_srcSystem() == master.target_system:
                armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                print(f"  HEARTBEAT: armed={armed}, custom_mode={msg.custom_mode}")

    print("-" * 60)
    print(f"  Tong: {sum(msg_types_seen.values())} msg, "
          f"cac loai: {', '.join(sorted(msg_types_seen.keys()))}")
    return important_msgs


def try_arm(master, force=False):
    """Thu ARM drone."""
    mode = "FORCE" if force else "NORMAL"
    param2 = 21196 if force else 0
    print(f"\n[{'5' if force else '4'}] Thu ARM ({mode})...")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, param2, 0, 0, 0, 0, 0
    )

    print(f"  Cho phan hoi (5 giay)...")
    print("-" * 60)

    start = time.time()
    got_ack = False

    while time.time() - start < 5:
        msg = master.recv_match(blocking=True, timeout=1)
        if msg:
            msg_type = msg.get_type()

            if msg_type == "COMMAND_ACK":
                got_ack = True
                results = {
                    0: "ACCEPTED", 1: "TEMPORARILY_REJECTED",
                    2: "DENIED", 3: "UNSUPPORTED",
                    4: "FAILED", 5: "IN_PROGRESS", 6: "CANCELLED",
                }
                r = results.get(msg.result, f"UNKNOWN({msg.result})")
                print(f"  >>> ACK: {r} (command={msg.command})")

            elif msg_type == "STATUSTEXT":
                print(f"  >>> STATUSTEXT: {msg.text}")

            elif msg_type == "HEARTBEAT":
                armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                if armed:
                    print(f"  >>> DA ARM THANH CONG!")

    if not got_ack:
        print("  >>> Khong nhan duoc ACK!")
    print("-" * 60)


def main():
    ap = argparse.ArgumentParser(description="Chan doan ket noi Pixhawk")
    ap.add_argument("--port", default=DEFAULT_PORT,
                    help="Cong serial (mac dinh: %s)" % DEFAULT_PORT)
    ap.add_argument("--baud", type=int, help="Chi thu dung baud nay")
    ap.add_argument("--try-arm", action="store_true",
                    help="Thu ARM va FORCE-ARM. CHI DUNG KHI DA THAO CANH QUAT.")
    args = ap.parse_args()

    port = args.port
    bauds = [args.baud] if args.baud else BAUD_RATES

    print("=" * 60)
    print("  PIXHAWK DIAGNOSTIC TOOL v3")
    print("=" * 60)

    # 0. Liet ke cong serial
    available = list_com_ports()

    if available and port.upper() not in [p.upper() for p in available]:
        print(f"\n  CHU Y: {port} khong co trong danh sach USB o tren.")
        print("  Binh thuong neu day la UART tren chan GPIO (/dev/serial0,")
        print("  /dev/ttyAMA0) - loai nay khong hien ra nhu thiet bi USB.")

    # 1. Thu raw serial voi nhieu baud rate
    print(f"\n[1] Kiem tra raw serial {port}...")
    working_baud = None
    for baud in bauds:
        if test_raw_serial(port, baud, duration=2):
            working_baud = baud
            break

    if not working_baud:
        print(f"\n  LOI: Khong doc duoc byte nao tu {port}!")
        print("  Neu cam USB:")
        print("  - Cong co dung khong? (xem danh sach o tren)")
        print("  - Co chuong trinh khac dang giu cong khong?")
        print("    (Mission Planner, QGroundControl, mavproxy...)")
        print("  - Tren Linux, user da o trong nhom dialout chua?")
        print("      sudo usermod -aG dialout $USER   roi dang xuat / dang nhap lai")
        print("  Neu noi UART tren chan GPIO:")
        print("  - TX cua Pixhawk phai vao RX cua Pi va nguoc lai (rat hay dau nguoc)")
        print("  - GND hai ben phai noi chung")
        print("  - Tren Pixhawk: SERIALx_PROTOCOL = 2 (MAVLink2), SERIALx_BAUD dung")
        print("  - Tren Pi: da tat serial console va bat UART chua?")
        sys.exit(1)

    # 2. Thu MAVLink voi baud rate da tim
    print(f"\n[2] Ket noi MAVLink (baud={working_baud})...")
    master = try_mavlink_connect(port, working_baud)

    # Neu khong duoc, thu cac baud khac
    if not master:
        for baud in bauds:
            if baud == working_baud:
                continue
            master = try_mavlink_connect(port, baud)
            if master:
                working_baud = baud
                break

    if not master:
        print("\n  LOI: Ket noi MAVLink THAT BAI voi tat ca baud rates!")
        print("  Co the day khong phai Pixhawk, hoac firmware loi.")
        sys.exit(1)

    # 3. Doc messages
    read_messages(master, duration=5)

    # Request data streams
    master.mav.request_data_stream_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1
    )
    time.sleep(1)

    # 4-6. Thu ARM. MAC DINH KHONG LAM: force-arm (magic 21196) bo qua MOI kiem
    # tra an toan - canh quat quay that neu chua thao. Phai co co --try-arm.
    if args.try_arm:
        print("\n  CHI TIEP TUC NEU DA THAO CANH QUAT.")
        if input("  Go  YES  de thu ARM: ").strip() != "YES":
            print("  Bo qua buoc ARM.")
        else:
            try_arm(master, force=False)
            try_arm(master, force=True)

            print("\n[6] DISARM...")
            master.mav.command_long_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 21196, 0, 0, 0, 0, 0
            )
            time.sleep(2)

            msg = master.recv_match(type='STATUSTEXT', blocking=True, timeout=2)
            if msg:
                print(f"  STATUSTEXT: {msg.text}")
    else:
        print("\n[4] Bo qua thu ARM. Them --try-arm neu muon (phai thao canh quat).")

    master.close()
    print("\n" + "=" * 60)
    print("  HOAN TAT! Xem cac dong STATUSTEXT de biet ly do.")
    print("=" * 60)


if __name__ == "__main__":
    main()
