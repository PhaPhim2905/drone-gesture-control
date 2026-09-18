#!/usr/bin/env bash
# CÀI MỘT LẦN trên Raspberry Pi 5 — Ubuntu 24.04 LTS (64-bit).
#
#   cd ~/drone-gesture-control && ./scripts/pi_setup.sh
#
# Làm theo thứ tự, mỗi bước in ">>> ". Chạy lại được: bước nào xong rồi thì bỏ qua.
#   1. kiểm đúng Ubuntu 24.04 arm64
#   2. ROS 2 Jazzy ros-base + MAVROS + dữ liệu GeographicLib
#   3. .venv (--system-site-packages) + requirements.txt
#   4. quyền cổng serial Pixhawk, zram
#   5. build workspace + test
set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo ">>> 1. kiem he dieu hanh"
. /etc/os-release
if [[ "$VERSION_ID" != "24.04" ]]; then
  echo "CAN Ubuntu 24.04, dang la $PRETTY_NAME. ROS 2 Jazzy chi co goi cho 24.04."
  exit 1
fi
echo "    $PRETTY_NAME $(uname -m)"

echo ">>> 2. ROS 2 Jazzy + MAVROS"
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  sudo apt update
  sudo apt install -y software-properties-common curl
  sudo add-apt-repository -y universe
  ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
    | grep -F '"tag_name"' | awk -F'"' '{print $4}')
  curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${UBUNTU_CODENAME}_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
  sudo apt update
fi
# ros-base: KHÔNG có rviz/gazebo/GUI -> nhẹ cho Pi. Xem đồ hoạ từ laptop.
sudo apt install -y ros-jazzy-ros-base ros-dev-tools \
  ros-jazzy-mavros ros-jazzy-mavros-extras \
  python3-venv python3-pip v4l-utils
# MAVROS không chạy được nếu thiếu bộ dữ liệu này (lỗi GeographicLib lúc khởi động).
if [[ ! -d /usr/share/GeographicLib/geoids ]]; then
  sudo /opt/ros/jazzy/lib/mavros/install_geographiclib_datasets.sh
fi

echo ">>> 3. venv + thu vien Python"
if [[ ! -d "$REPO/.venv" ]]; then
  # --system-site-packages: venv vẫn thấy rclpy, mavros_msgs của ROS.
  python3 -m venv --system-site-packages "$REPO/.venv"
fi
"$REPO/.venv/bin/pip" install --upgrade pip
"$REPO/.venv/bin/pip" install -r "$REPO/requirements.txt"
"$REPO/.venv/bin/python" - <<'EOF'
import cv2, mediapipe, numpy
print(f"    opencv {cv2.__version__}  mediapipe {mediapipe.__version__}  numpy {numpy.__version__}")
assert hasattr(mediapipe, "solutions"), "mediapipe khong co mp.solutions - phai la 0.10.14"
EOF

echo ">>> 4. quyen serial + zram"
if ! id -nG "$USER" | grep -qw dialout; then
  sudo usermod -aG dialout "$USER"
  echo "    DA THEM $USER vao nhom dialout. DANG XUAT roi dang nhap lai."
fi
if ! systemctl is-active --quiet zramswap 2>/dev/null; then
  sudo apt install -y zram-tools || true
  echo -e "ALGO=zstd\nPERCENT=50" | sudo tee /etc/default/zramswap >/dev/null
  sudo systemctl restart zramswap || true
fi

echo ">>> 5. build + test"
"$REPO/scripts/build.sh"

cat <<EOF

XONG. Moi terminal moi:
    source $REPO/scripts/env.sh
Kiem may:
    ros2 run gesture_perception doctor
Buoc 1:
    ros2 launch gesture_bringup camera_only.launch.py
EOF
