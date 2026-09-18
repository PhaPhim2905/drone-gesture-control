"""
LÕI CHUNG của sitl.launch.py và real.launch.py. Không gọi trực tiếp.

Khởi động 3 node theo thứ tự:
    1. mavros          nối FCU (SITL qua UDP hoặc Pixhawk qua USB)
    2. perception_node camera -> /gesture/detected, CHỈ khi FCU ở GUIDED
    3. command_node    LOCKED cho tới khi FCU connected VÀ có ASSUME_GUIDANCE

Chuỗi trao quyền: pilot gạt GUIDED (camera bắt đầu nhận diện) -> người điều
khiển giơ ASSUME_GUIDANCE 1 s (mở khoá) -> TAKEOFF / ROLL / STOP / LAND.

Thứ tự khởi động KHÔNG phải lớp an toàn: command_node tự kiểm connected, armed,
mode, landed_state ở MỖI nhịp (command_logic.tick), nên node nào lên trước cũng
không phát lệnh sai.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("gesture_bringup")
    cfg = lambda f: PathJoinSubstitution([share, "config", f])  # noqa: E731
    L = LaunchConfiguration
    # Chuỗi có thể rỗng phải ép kiểu str: launch_ros đọc giá trị theo YAML, "" -> None.
    S_ = lambda name: ParameterValue(L(name), value_type=str)  # noqa: E731
    model =PathJoinSubstitution([share, "models", [L("model_name"), ".tflite"]])

    return LaunchDescription([
        DeclareLaunchArgument("fcu_url"),
        DeclareLaunchArgument("command_config"),
        DeclareLaunchArgument("gcs_url", default_value=""),
        DeclareLaunchArgument("model_name", default_value="v3_pck_body25"),
        DeclareLaunchArgument("camera_index", default_value="-1"),
        DeclareLaunchArgument("show_window", default_value="false"),
        DeclareLaunchArgument("debug_image_hz", default_value="0.0"),
        DeclareLaunchArgument("use_camera", default_value="true",
                              description="false = chỉ mavros + command_node, "
                                          "gửi cử chỉ giả bằng ros2 topic pub"),
        DeclareLaunchArgument("require_mode", default_value="GUIDED",
                              description="chỉ nhận diện khi FCU ở mode này; rỗng = luôn nhận diện"),
        DeclareLaunchArgument("video_path", default_value="",
                              description="thay camera bằng video take (file, glob, hoặc a.mp4,b.mp4)"),
        DeclareLaunchArgument("flip", default_value="true",
                              description="video quay bằng training/1_record.py ĐÃ lật -> flip:=false"),

        # KHÔNG đặt namespace="mavros": mavros_node của MAVROS 2 tự đặt topic dưới
        # /mavros. Thêm namespace thì thành /mavros/mavros/state và command_node,
        # perception_node không bao giờ nhận được state (đo 2026-09-15, MAVROS 2.15.1).
        Node(
            package="mavros", executable="mavros_node",
            output="screen",
            parameters=[cfg("mavros.yaml"), {
                "fcu_url": L("fcu_url"),
                "gcs_url": S_("gcs_url"),
                "tgt_system": 1,
                "tgt_component": 1,
                "fcu_protocol": "v2.0",
            }],
        ),
        Node(
            package="gesture_perception", executable="perception_node",
            name="perception_node", output="screen",
            condition=IfCondition(L("use_camera")),
            parameters=[cfg("perception.yaml"), {
                "model_path": model,
                "camera_index": L("camera_index"),
                "show_window": L("show_window"),
                "debug_image_hz": L("debug_image_hz"),
                "require_mode": S_("require_mode"),
                "video_path": S_("video_path"),
                "flip": L("flip"),
            }],
        ),
        Node(
            package="gesture_command", executable="command_node",
            name="command_node", output="screen",
            parameters=[PathJoinSubstitution([share, "config", L("command_config")])],
        ),
    ])
