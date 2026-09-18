# Nạp môi trường cho MỌI terminal chạy dự án. Dùng "source", không chạy trực tiếp:
#
#   source scripts/env.sh
#
# Thứ tự quan trọng:
#   1. ROS 2 Jazzy        rclpy, mavros_msgs
#   2. .venv              mediapipe 0.10.14, opencv, ai-edge-litert
#                         (tạo với --system-site-packages để vẫn thấy rclpy)
#   3. install/ của ws    4 gói gesture_*
# Sai thứ tự 2 thì node chạy bằng python hệ thống và báo "No module mediapipe".

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  source /opt/ros/jazzy/setup.bash
else
  echo "[env] CHUA CAI ROS 2 Jazzy (/opt/ros/jazzy). Chay scripts/pi_setup.sh"
fi

if [[ -f "$_REPO/.venv/bin/activate" ]]; then
  source "$_REPO/.venv/bin/activate"
else
  echo "[env] chua co .venv - chay scripts/pi_setup.sh"
fi

if [[ -f "$_REPO/ros2_ws/install/setup.bash" ]]; then
  source "$_REPO/ros2_ws/install/setup.bash"
else
  echo "[env] chua build workspace - chay scripts/build.sh"
fi

# Cùng số domain trên Pi và laptop thì rqt_graph ở laptop thấy node trên Pi.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
# SUBNET: thấy máy khác trong mạng LAN (cần khi xem từ laptop).
# LOCALHOST: chỉ trong máy — đặt khi bay thật để DDS không dò mạng.
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"

export ARDUPILOT_DIR="${ARDUPILOT_DIR:-$HOME/ardupilot}"
export ARDUPILOT_GAZEBO="${ARDUPILOT_GAZEBO:-$HOME/ardupilot_gazebo}"

echo "[env] repo=$_REPO  python=$(command -v python3)  ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
