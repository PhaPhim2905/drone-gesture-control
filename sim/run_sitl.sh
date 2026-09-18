#!/usr/bin/env bash
# ArduCopter SITL + MAVProxy. Chạy ở terminal RIÊNG vì cần gõ lệnh vào MAVProxy.
#
#   ./sim/run_sitl.sh                        # nối Gazebo (chạy gazebo.launch.py trước)
#   ./sim/run_sitl.sh --out 192.168.1.50     # + gửi MAVLink sang Pi ở IP đó
#   ./sim/run_sitl.sh --no-gazebo            # SITL trần, nhẹ, chạy được trên Pi
#   ./sim/run_sitl.sh --wipe                 # xoá param SITL về mặc định
#   ./sim/run_sitl.sh --real-params          # + tham số Pixhawk 6C thật (đã lọc)
#   ./sim/run_sitl.sh --no-gazebo --real-params
#                                            # vật lý drone 12 kg thật, không 3D
#
# --real-params: sim/params/pixhawk6c_sitl.parm (sinh bằng sim/make_sitl_params.py)
#   Với Gazebo: tham số thật nhưng VẬT LÝ vẫn là iris 1.5 kg -> PID/ga treo lệch.
#   Với --no-gazebo: vật lý sim/models/quad12kg.json -> gần drone thật nhất.
#   --compass-on: CHỈ trong SITL bật la bàn. Bộ tham số thật có COMPASS_USE=0 nên
#   GUIDED báo "Need Position Estimate" và không arm được trên mặt đất.
#   --proposal: CHỈ trong SITL, đề xuất cho Pixhawk 6X (sim/params/sitl_6x_proposal.parm):
#   kênh 5 = LOITER / ALT_HOLD / GUIDED, kênh 6 = E-stop, mất sóng -> LAND,
#   ARM/DISARM = chụm 2 cần (pixhawk_tools/lua/csc_arm.lua). Thử trước khi ghi xuống board.
#
# Tay cầm THẬT lái drone ảo: chạy thêm sim/rc_bridge.py (xem RUNBOOK bước 5b).
#
# Ở dấu nhắc MAVProxy, PILOT (anh) arm bằng tay — không có node nào arm hộ:
#   mode guided
#   arm throttle
# rồi giơ ASSUME_GUIDANCE -> TAKEOFF trước camera.
set -euo pipefail

ARDUPILOT_DIR="${ARDUPILOT_DIR:-$HOME/ardupilot}"
GAZEBO=1
OUT_IP=""
REAL=0
COMPASS_ON=0
PROPOSAL=0
EXTRA=()
REPO="$(cd "$(dirname "$0")/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_IP="$2"; shift 2 ;;
    --no-gazebo) GAZEBO=0; shift ;;
    --wipe) EXTRA+=(-w); shift ;;
    --real-params) REAL=1; shift ;;
    --compass-on) COMPASS_ON=1; shift ;;
    --proposal) PROPOSAL=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "tham so la: $1"; exit 1 ;;
  esac
done

if [[ ! -x "$ARDUPILOT_DIR/Tools/autotest/sim_vehicle.py" ]]; then
  echo "Khong thay ArduPilot o $ARDUPILOT_DIR. Chay scripts/sim_setup.sh truoc."
  exit 1
fi

CMD=("$ARDUPILOT_DIR/Tools/autotest/sim_vehicle.py" -v ArduCopter --speedup 1)
DP="$ARDUPILOT_DIR/Tools/autotest/default_params"
if [[ $GAZEBO == 1 ]]; then
  # gazebo-iris + JSON: vật lý do Gazebo tính, SITL chỉ chạy firmware. Lockstep.
  #
  # PHẢI nạp tay file tham số khung. ArduPilot bản 2026 để binary SITL tự tìm
  # tham số mặc định theo --model, mà --model JSON không có trong vehicleinfo
  # -> FRAME_CLASS = 0 -> "PreArm: Motors: Check frame class and type", không
  # arm được. Đo ngày 2026-09-15 trên commit 35356c656c.
  CMD+=(-f gazebo-iris --model JSON
        --add-param-file="$DP/copter.parm"
        --add-param-file="$DP/gazebo-iris.parm")
elif [[ $REAL == 1 ]]; then
  # Vật lý của chính drone 12 kg: SITL đọc khối lượng, cánh, ga treo từ json.
  # Đường dẫn json PHẢI tương đối với thư mục chạy: SITL panic "failed to load"
  # với đường dẫn tuyệt đối (đo 2026-09-15). File được chép vào SITL_DIR bên dưới.
  CMD+=(-f quad --model "quad:quad12kg.json"
        --add-param-file="$DP/copter.parm")
else
  CMD+=(-f quad)
fi
if [[ $REAL == 1 ]]; then
  RP="$REPO/sim/params/pixhawk6c_sitl.parm"
  [[ -f "$RP" ]] || { echo "Thieu $RP. Chay: python3 sim/make_sitl_params.py"; exit 1; }
  CMD+=(--add-param-file="$RP")
  [[ $COMPASS_ON == 1 ]] && CMD+=(--add-param-file="$REPO/sim/params/sitl_compass_on.parm")
fi
# nạp SAU tham số thật để ghi đè FLTMODE / FS_THR của file Pixhawk
[[ $PROPOSAL == 1 ]] && CMD+=(--add-param-file="$REPO/sim/params/sitl_6x_proposal.parm")
[[ -n "$OUT_IP" ]] && CMD+=(--out "udp:${OUT_IP}:14550")
CMD+=("${EXTRA[@]}")

# --console / --map cần màn hình. WSL2 trên Windows 11 có WSLg nên mở được.
if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  CMD+=(--console --map)
fi

# install-prereqs của ArduPilot trên 24.04 tạo venv riêng chứa MAVProxy/pymavlink.
# Nếu terminal đang ở .venv của dự án (scripts/env.sh) thì sim_vehicle.py sẽ chạy
# nhầm python đó -> "No module named pymavlink". Đổi sang venv của ArduPilot.
if [[ -f "$HOME/venv-ardupilot/bin/activate" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$HOME/venv-ardupilot/bin/activate"
  set -u
fi

# SITL ghi eeprom.bin, logs/, terrain/ vào thư mục hiện tại. Chạy ở thư mục
# riêng để không rải file vào repo.
# Mỗi cấu hình một thư mục: eeprom.bin của bộ tham số này không lẫn sang bộ kia.
CFG=$([[ $GAZEBO == 1 ]] && echo gazebo || echo nogazebo)$([[ $REAL == 1 ]] && echo _pixhawk6c || echo _default)
SITL_DIR="${SITL_DIR:-$HOME/sitl_run/$CFG}"
mkdir -p "$SITL_DIR" && cd "$SITL_DIR"
[[ $GAZEBO == 0 && $REAL == 1 ]] && cp "$REPO/sim/models/quad12kg.json" "$SITL_DIR/"
# SITL chạy Lua trong ./scripts của thư mục chạy, giống APM/scripts trên thẻ nhớ Pixhawk
if [[ $PROPOSAL == 1 ]]; then
  mkdir -p "$SITL_DIR/scripts" && cp "$REPO/pixhawk_tools/lua/csc_arm.lua" "$SITL_DIR/scripts/"
fi

echo ">>> ${CMD[*]}"
exec "${CMD[@]}"
