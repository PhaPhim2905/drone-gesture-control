"""
BƯỚC 1 — chỉ camera + nhận diện. KHÔNG nối drone.

    ros2 launch gesture_bringup camera_only.launch.py
    ros2 launch gesture_bringup camera_only.launch.py show_window:=true camera_index:=1

Xem kết quả ở terminal khác:
    ros2 topic echo /gesture/detected --field label
    ros2 topic hz /gesture/detected          # = fps thật của cả chuỗi

Đo độ trễ trên Pi bằng video take thay camera (luôn nhận diện, không cần FCU):
    ros2 launch gesture_bringup camera_only.launch.py video_path:="$HOME/data/video/*.mp4" flip:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("gesture_bringup")
    model = PathJoinSubstitution([share, "models",
                                  [LaunchConfiguration("model_name"), ".tflite"]])
    return LaunchDescription([
        DeclareLaunchArgument("model_name", default_value="v3_pck_body25"),
        DeclareLaunchArgument("camera_index", default_value="-1"),
        DeclareLaunchArgument("show_window", default_value="false"),
        DeclareLaunchArgument("debug_image_hz", default_value="0.0"),
        DeclareLaunchArgument("video_path", default_value=""),
        DeclareLaunchArgument("flip", default_value="true"),
        Node(
            package="gesture_perception", executable="perception_node",
            name="perception_node", output="screen",
            parameters=[PathJoinSubstitution([share, "config", "perception.yaml"]), {
                "model_path": model,
                "camera_index": LaunchConfiguration("camera_index"),
                "show_window": LaunchConfiguration("show_window"),
                "debug_image_hz": LaunchConfiguration("debug_image_hz"),
                "video_path": ParameterValue(LaunchConfiguration("video_path"), value_type=str),
                "flip": LaunchConfiguration("flip"),
            }],
        ),
    ])
