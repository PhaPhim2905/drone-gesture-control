"""
BƯỚC 2 — điều khiển drone ẢO (ArduPilot SITL, có hoặc không Gazebo).

Chạy SAU KHI SITL đã lên (sim/run_sitl.sh). Xem docs/RUNBOOK_ROS2.md.

    # SITL ở laptop (run_sitl.sh --out <IP Pi>) hoặc ngay máy này — cùng một lệnh,
    # vì MAVProxy luôn gửi 127.0.0.1:14550 và thêm <IP>:14550 nếu có --out:
    ros2 launch gesture_bringup sitl.launch.py

    # không camera, bắn cử chỉ bằng tay để thử logic:
    ros2 launch gesture_bringup sitl.launch.py use_camera:=false
    ros2 topic pub -r 10 /gesture/detected gesture_interfaces/msg/Gesture "{label: ASSUME_GUIDANCE}"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    L = LaunchConfiguration
    core = PathJoinSubstitution([FindPackageShare("gesture_bringup"), "launch", "core.launch.py"])
    return LaunchDescription([
        # udp://:14550@  = nghe cổng 14550 trên mọi card mạng. sim/run_sitl.sh
        # gửi MAVLink tới <IP Pi>:14550.
        DeclareLaunchArgument("fcu_url", default_value="udp://:14550@"),
        DeclareLaunchArgument("camera_index", default_value="-1"),
        DeclareLaunchArgument("show_window", default_value="false"),
        DeclareLaunchArgument("debug_image_hz", default_value="0.0"),
        DeclareLaunchArgument("use_camera", default_value="true"),
        # command_real.yaml = hành xử như drone thật (pilot tự gạt GUIDED, 0.5 m/s),
        # dùng khi tay cầm thật lái drone ảo qua sim/rc_bridge.py
        DeclareLaunchArgument("command_config", default_value="command_sitl.yaml"),
        IncludeLaunchDescription(core, launch_arguments={
            "fcu_url": L("fcu_url"),
            "command_config": L("command_config"),
            "camera_index": L("camera_index"),
            "show_window": L("show_window"),
            "debug_image_hz": L("debug_image_hz"),
            "use_camera": L("use_camera"),
        }.items()),
    ])
