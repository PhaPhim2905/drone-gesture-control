#!/usr/bin/env bash
# CÀI MỘT LẦN máy MÔ PHỎNG: Ubuntu 24.04 (WSL2 trên laptop, PC Ubuntu, hoặc Pi 5).
#
#   ./scripts/sim_setup.sh             # Gazebo Harmonic + ArduPilot SITL + plugin
#   ./scripts/sim_setup.sh --no-gazebo # chỉ SITL (đủ để thử command_node trên Pi)
#
# Mất 30-60 phút lần đầu (build ArduPilot). Cần ~6 GB đĩa.
#
# WSL2 (PowerShell, một lần):
#   wsl --install -d Ubuntu-24.04
#   # %UserProfile%\.wslconfig:
#   #   [wsl2]
#   #   networkingMode=mirrored      <- Pi trong LAN thấy được SITL trong WSL
#   wsl --shutdown
set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GAZEBO=1
[[ "${1:-}" == "--no-gazebo" ]] && GAZEBO=0

. /etc/os-release
[[ "$VERSION_ID" == "24.04" ]] || { echo "CAN Ubuntu 24.04, dang $PRETTY_NAME"; exit 1; }

ARDUPILOT_DIR="${ARDUPILOT_DIR:-$HOME/ardupilot}"
ARDUPILOT_GAZEBO="${ARDUPILOT_GAZEBO:-$HOME/ardupilot_gazebo}"

echo ">>> 1. ROS 2 Jazzy (can cho ros2 launch gazebo.launch.py va rqt)"
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  "$REPO/scripts/pi_setup.sh" || true    # dùng lại phần cài ROS; bỏ qua lỗi camera
fi

if [[ $GAZEBO == 1 ]]; then
  echo ">>> 2. Gazebo Harmonic"
  if ! command -v gz >/dev/null; then
    sudo apt install -y curl lsb-release gnupg
    sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
      -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
      | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
    sudo apt update
    sudo apt install -y gz-harmonic
  fi
  gz sim --versions || true
fi

echo ">>> 3. ArduPilot SITL"
if [[ ! -d "$ARDUPILOT_DIR" ]]; then
  git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git "$ARDUPILOT_DIR"
fi
cd "$ARDUPILOT_DIR"
if [[ ! -f "$HOME/.ardupilot_prereqs_done" ]]; then
  Tools/environment_install/install-prereqs-ubuntu.sh -y
  touch "$HOME/.ardupilot_prereqs_done"
fi
# install-prereqs ghi PATH vào ~/.profile
# shellcheck disable=SC1091
. "$HOME/.profile" || true
./waf configure --board sitl
./waf copter

if [[ $GAZEBO == 1 ]]; then
  echo ">>> 4. ardupilot_gazebo (plugin noi Gazebo <-> SITL)"
  sudo apt install -y libgz-sim8-dev rapidjson-dev libopencv-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl
  if [[ ! -d "$ARDUPILOT_GAZEBO" ]]; then
    git clone https://github.com/ArduPilot/ardupilot_gazebo "$ARDUPILOT_GAZEBO"
  fi
  mkdir -p "$ARDUPILOT_GAZEBO/build" && cd "$ARDUPILOT_GAZEBO/build"
  GZ_VERSION=harmonic cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
  make -j"$(nproc)"
fi

cat <<EOF

XONG. Thu mo phong (3 terminal, deu "source $REPO/scripts/env.sh"):
  T1  ros2 launch gesture_bringup gazebo.launch.py
  T2  $REPO/sim/run_sitl.sh --out <IP Pi>
  T3  (tren Pi) ros2 launch gesture_bringup sitl.launch.py
Chi tiet: docs/RUNBOOK_ROS2.md
EOF
