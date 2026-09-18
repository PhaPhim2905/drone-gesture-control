"""sim/chain_test: pymavlink 'pilot' tren SERIAL1 cua SITL: cho GPS, gat GUIDED, arm; in do cao; ket thuc khi da bay len va ha xuong."""
import sys, time
from pymavlink import mavutil

T0 = time.time()


def log(s):
    print(f"{time.time() - T0:7.1f} PIL {s}", flush=True)


m = mavutil.mavlink_connection(sys.argv[1], source_system=255, source_component=191)
m.wait_heartbeat(timeout=120)
m.mav.request_data_stream_send(m.target_system, m.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)
texts = []
alt = 0.0
mode = ""
armed = False


def pump(dur):
    global alt, mode, armed
    end = time.time() + dur
    while time.time() < end:
        x = m.recv_match(blocking=True, timeout=0.2)
        if x is None:
            continue
        t = x.get_type()
        if t == "STATUSTEXT":
            texts.append(x.text)
            log(f"AP: {x.text}")
        elif t == "GLOBAL_POSITION_INT":
            alt = x.relative_alt / 1000
        elif t == "HEARTBEAT" and x.type == mavutil.mavlink.MAV_TYPE_QUADROTOR:
            mode = mavutil.mode_string_v10(x)
            armed = bool(x.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)


end = time.time() + 150
while time.time() < end and not any("using GPS" in t for t in texts):
    pump(1)
pump(20)
log(f"PHASE A xong (mode={mode}): perception phai IM LANG. Gat LOITER 10 s")
m.set_mode(m.mode_mapping()["LOITER"])
pump(10)
log("PHASE B: pilot gat GUIDED + arm")
m.set_mode(m.mode_mapping()["GUIDED"])
pump(2)
for _ in range(10):
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    pump(2)
    if armed:
        break
log(f"armed={armed} mode={mode}")
peak = 0.0
nxt = 0
end = time.time() + float(sys.argv[2] if len(sys.argv) > 2 else 200)
while time.time() < end:
    pump(0.5)
    peak = max(peak, alt)
    if time.time() > nxt:
        nxt = time.time() + 3
        log(f"alt={alt:.2f} m mode={mode} armed={armed} peak={peak:.2f}")
    if peak > 1.5 and alt < 0.3 and not armed:
        log("DA CAT CANH VA HA CANH XONG")
        break
log(f"KET THUC peak={peak:.2f} alt={alt:.2f} mode={mode} armed={armed}")
