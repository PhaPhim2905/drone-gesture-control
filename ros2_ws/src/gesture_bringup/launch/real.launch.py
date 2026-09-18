"""
BƯỚC 3 — Pixhawk THẬT. THÁO CÁNH QUẠT cho lần đầu.

    ros2 launch gesture_bringup real.launch.py                  # TELEM2 -> UART GPIO Pi 5
    ros2 launch gesture_bringup real.launch.py fcu_url:=serial:///dev/ttyACM0:115200   # cáp USB

Mặc định nối Pixhawk 6X cổng TELEM2 vào UART GPIO của Pi 5 (/dev/ttyAMA0, 921600).
Cài một lần: scripts/pi_uart_setup.sh (có sơ đồ dây và tham số SERIAL2_*).
USB của Pixhawk để trống cho Mission Planner theo dõi CÙNG LÚC.

Trước khi chạy:
  - python3 ros2_ws/src/gesture_command/test/test_command_logic.py -> TAT CA DEU QUA
  - đã chạy sitl.launch.py và thấy drone ảo đi ĐÚNG CHIỀU
  - tay cầm bật, công tắc mode có GUIDED và LOITER, kill switch hoạt động
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    L = LaunchConfiguration
    core = PathJoinSubstitution([FindPackageShare("gesture_bringup"), "launch", "core.launch.py"])
    return LaunchDescription([
        DeclareLaunchArgument("fcu_url", default_value="serial:///dev/ttyAMA0:921600"),
        DeclareLaunchArgument("camera_index", default_value="-1"),
        DeclareLaunchArgument("show_window", default_value="false"),
        DeclareLaunchArgument("debug_image_hz", default_value="0.0"),
        IncludeLaunchDescription(core, launch_arguments={
            "fcu_url": L("fcu_url"),
            "command_config": "command_real.yaml",
            "camera_index": L("camera_index"),
            "show_window": L("show_window"),
            "debug_image_hz": L("debug_image_hz"),
        }.items()),
    ])
