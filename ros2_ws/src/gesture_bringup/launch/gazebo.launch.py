"""
GAZEBO HARMONIC + drone iris của ardupilot_gazebo. Chạy TRƯỚC sim/run_sitl.sh.

    ros2 launch gesture_bringup gazebo.launch.py                 # có cửa sổ
    ros2 launch gesture_bringup gazebo.launch.py headless:=true  # Pi 5: không GUI

Vì sao Pi 5 nên headless: giao diện Gazebo Harmonic cần OpenGL 3.3, driver V3D
của Pi 5 báo 3.1. Server vật lý không cần OpenGL khi world không có camera.

Đồng bộ với SITL: plugin ArduPilotPlugin chạy LOCKSTEP qua giao thức JSON — mỗi
bước vật lý của Gazebo đợi đúng một bước của SITL, nên máy chậm thì cả hai cùng
chậm chứ không lệch nhau. Đừng dùng --speedup khác 1 khi có camera thật trong
vòng lặp: người điều khiển sống theo thời gian thật.

GPU TRONG WSL (gpu:=auto, mặc định)
    Mesa trên Ubuntu 24.04 trong WSL tự rơi về llvmpipe (vẽ bằng CPU) dù có
    /dev/dxg. Đặt GALLIUM_DRIVER=d3d12 thì vẽ qua GPU của Windows. Đo 2026-09-15
    trên MSI Cyborg 15 (Intel UHD, RTX 4050 đang tắt):
        world shapes, có GUI        CPU 0.88  ->  GPU 1.00 real-time factor
        iris_runway + SITL, GUI     CPU 0.30  ->  GPU 0.30   (giới hạn ở CPU vật lý)
    Tức GPU làm hình mượt, KHÔNG làm mô phỏng drone nhanh hơn trên máy này.
    Card NVIDIA bật lại được thì MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA chọn nó.

        ros2 launch gesture_bringup gazebo.launch.py gpu:=false   # ép vẽ bằng CPU

Cài một lần: scripts/sim_setup.sh.
"""

import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction,
                            SetEnvironmentVariable)


def _gpu_env(mode):
    """-> dict biến môi trường cho gz. Chỉ đụng tới khi đang trong WSL (có /dev/dxg)."""
    if mode == "false":
        return {"LIBGL_ALWAYS_SOFTWARE": "1"}
    in_wsl = os.path.exists("/dev/dxg")
    if mode == "true" or (mode == "auto" and in_wsl and "GALLIUM_DRIVER" not in os.environ):
        # Không có card NVIDIA thì Mesa tự dùng GPU mặc định (đã thử: ra Intel UHD).
        return {"GALLIUM_DRIVER": "d3d12",
                "MESA_D3D12_DEFAULT_ADAPTER_NAME":
                    os.environ.get("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")}
    return {}


def _gz(context):
    get = lambda k: context.launch_configurations[k]  # noqa: E731
    cmd = ["gz", "sim", "-v", "3", "-r", get("world")]
    if get("headless").lower() == "true":
        cmd.insert(2, "-s")
    env = _gpu_env(get("gpu").lower())
    actions = [LogInfo(msg=f"gazebo do hoa: {env or 'mac dinh cua he thong'}")]
    return actions + [ExecuteProcess(cmd=cmd, output="screen", additional_env=env)]


def generate_launch_description():
    agz = os.path.expanduser(os.environ.get("ARDUPILOT_GAZEBO", "~/ardupilot_gazebo"))

    def join(var, *paths):
        old = os.environ.get(var, "")
        return os.pathsep.join([p for p in (*paths, old) if p])

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="iris_runway.sdf"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("gpu", default_value="auto",
                              description="auto = WSL thì vẽ bằng GPU (d3d12); true; false = CPU"),
        SetEnvironmentVariable("GZ_SIM_SYSTEM_PLUGIN_PATH",
                               join("GZ_SIM_SYSTEM_PLUGIN_PATH", f"{agz}/build")),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH",
                               join("GZ_SIM_RESOURCE_PATH", f"{agz}/models", f"{agz}/worlds")),
        OpaqueFunction(function=_gz),
    ])
