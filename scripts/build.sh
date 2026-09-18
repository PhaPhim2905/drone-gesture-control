#!/usr/bin/env bash
# Build workspace + chạy test logic. Chạy lại sau MỖI lần git pull.
#
#   ./scripts/build.sh
set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/env.sh"

echo ">>> test logic an toan (khong can ROS)"
python3 "$REPO/ros2_ws/src/gesture_command/test/test_command_logic.py" | tail -1
python3 "$REPO/ros2_ws/src/gesture_perception/test/test_perception_core.py" | tail -1

echo ">>> colcon build"
cd "$REPO/ros2_ws"
# Build bằng python của venv: setup.py thấy đúng môi trường, shebang /usr/bin/env
# python3 (setup.cfg) chạy node bằng venv.
python3 -m colcon build --symlink-install --event-handlers console_cohesion+

source "$REPO/ros2_ws/install/setup.bash"
echo ">>> xong. Kiem:"
ros2 pkg list | grep gesture_ || true
