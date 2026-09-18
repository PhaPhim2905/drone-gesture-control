<h1 align="center">🚁 Drone Gesture Control</h1>

<p align="center">
  <b>Điều khiển drone ArduPilot bằng cử chỉ toàn thân, trên ROS 2</b><br>
  MediaPipe Pose → One-Euro → PCK → TFLite → MAVROS → Pixhawk 6X / SITL + Gazebo
</p>

<p align="center">
  <img alt="ROS 2" src="https://img.shields.io/badge/ROS%202-Jazzy-22314E?logo=ros">
  <img alt="Ubuntu" src="https://img.shields.io/badge/Ubuntu-24.04-E95420?logo=ubuntu&logoColor=white">
  <img alt="Gazebo" src="https://img.shields.io/badge/Gazebo-Harmonic-orange">
  <img alt="ArduPilot" src="https://img.shields.io/badge/ArduPilot-GUIDED-red">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow">
</p>

---

## Bắt đầu từ đâu

| Anh muốn | Đọc / chạy |
|:---|:---|
| **Cài Pi 5 và chạy thử theo từng bước** | [`docs/RUNBOOK_ROS2.md`](docs/RUNBOOK_ROS2.md) |
| **Hiểu luồng bên trong**: node, topic, service, máy trạng thái | [`docs/KIEN_TRUC_ROS2.md`](docs/KIEN_TRUC_ROS2.md) |
| Quay dữ liệu, train model mới | [`docs/RUNBOOK_TRAINING.md`](docs/RUNBOOK_TRAINING.md) + thư mục [`training/`](training/) |
| Đọc / ghi param Pixhawk | [`docs/PIXHAWK_PARAM.md`](docs/PIXHAWK_PARAM.md) |
| Hiểu bộ lọc One-Euro | [`docs/ONE_EURO.md`](docs/ONE_EURO.md) |

---

## Cấu trúc

Hai thế giới tách rời: **chạy trên Pi (ROS 2)** và **train trên laptop (không ROS)**.
Chúng chỉ chia sẻ hai thứ: định dạng vector 50 chiều (`pck_format.py`) và file `.tflite`.

```
drone-gesture-control/
│
├── ros2_ws/src/                         ◀ CHẠY TRÊN PI — ROS 2 Jazzy
│   ├── gesture_interfaces/                 message: Gesture, CommandStatus
│   ├── gesture_perception/                 NHIỆM VỤ: camera → cử chỉ
│   │   ├── perception_node.py              vỏ ROS: camera, publish /gesture/detected
│   │   ├── perception_core.py              chuỗi 1 frame, không ROS (test được trên Windows)
│   │   ├── one_euro.py                     lọc 25 khớp (Body25Filter) + tự kiểm
│   │   ├── pck_format.py                   MediaPipe 33 → BODY25 → chuẩn hoá PCK
│   │   ├── classifier.py                   .tflite + tau + nhiệt độ T
│   │   ├── settings.py                     tham số thu hình DÙNG CHUNG với training/
│   │   ├── camera_util.py                  mở camera (DSHOW/MSMF/V4L2+MJPG)
│   │   ├── doctor.py                       kiểm máy: thư viện, model, fps, RAM, nhiệt
│   │   └── test/test_perception_core.py
│   ├── gesture_command/                    NHIỆM VỤ: cử chỉ → lệnh drone AN TOÀN
│   │   ├── command_logic.py                máy trạng thái, không ROS — MỌI chốt an toàn ở đây
│   │   ├── command_node.py                 vỏ ROS: MAVROS setpoint / set_mode / takeoff
│   │   └── test/test_command_logic.py      24 kiểm, gồm "mã nguồn không gọi được ARM"
│   └── gesture_bringup/                    NHIỆM VỤ: khởi động hệ
│       ├── launch/  camera_only · sitl · real · gazebo · core
│       ├── config/  perception · command_sitl · command_real · mavros
│       └── models/  v3_pck_body25.tflite + .labels.json
│
├── training/                            ◀ LAPTOP — chạy theo số thứ tự
│   ├── 1_record.py                         quay take → data/gestures/*.jsonl
│   ├── 2_check_poses.py                    chấm tư thế bằng tiêu chí hình học
│   ├── 3_build_dataset.py                  .jsonl → data/dataset_v3.npz
│   ├── 3b_build_from_video.py              (tuỳ chọn) video → .jsonl
│   ├── 4_train.py                          train K-fold, hiệu chuẩn T, tau (.venv-train)
│   └── 5_export_to_ros.py                  kiểm hợp đồng → chép model vào gesture_bringup
│
├── sim/run_sitl.sh                      ◀ ArduCopter SITL (+ Gazebo), terminal riêng
├── scripts/                             ◀ cài đặt & môi trường
│   ├── pi_setup.sh                         Pi 5: ROS 2 + MAVROS + venv + build
│   ├── sim_setup.sh                        WSL2/PC: Gazebo Harmonic + ArduPilot + plugin
│   ├── env.sh                              source ở MỌI terminal
│   └── build.sh                            test + colcon build, chạy sau git pull
├── pixhawk_tools/                       ◀ chẩn đoán, đọc/ghi param (luôn --dry-run trước)
├── data/gestures/                          dữ liệu thô đã quay (khớp xương, không phải ảnh)
└── docs/
```

---

## Luồng dữ liệu

```
 webcam ─► perception_node ──/gesture/detected──► command_node ──► MAVROS ──► Pixhawk 6X
           Pose Lite            nhãn + conf        LOCKED/ENABLED    │           hoặc
           One-Euro             theo TỪNG frame    giữ 0.6-1.0 s     │        SITL ◄► Gazebo
           PCK → TFLite                            chốt an toàn      │
                                                         │           ▼
                                           /drone/gesture_command   /mavros/state
```

**Ảnh không bao giờ rời perception_node.** Ra ngoài chỉ có nhãn và 50 số thực. Đẩy
ảnh 640×480 qua DDS sẽ tốn vài fps trên Pi chỉ để tuần tự hoá.

---

## Cử chỉ → lệnh

| Cử chỉ | Lệnh | Giữ | Khi nào nghe |
|:---|:---|:---:|:---|
| **ASSUME_GUIDANCE** | mở quyền điều khiển | 1.0 s | luôn |
| **TAKEOFF** | `NAV_TAKEOFF` 2 m, **chỉ khi pilot đã arm** | 1.0 s | đã mở quyền |
| **HOVER** | vận tốc 0 | 1.0 s | đã mở quyền |
| **ROLL_RIGHT** — tay trái dang ngang, tay phải giơ thẳng | drone sang phải **của nó** = sang **trái** người điều khiển | 1.0 s | đã mở quyền |
| **ROLL_LEFT** — tay phải dang ngang, tay trái giơ thẳng | drone sang trái **của nó** = sang **phải** người điều khiển | 1.0 s | đã mở quyền |
| **STOP** | vận tốc 0 + khoá + trả **LOITER** | 0.6 s | luôn |
| **LAND** | mode LAND + khoá | 0.8 s | luôn |
| **NEGATIVE** / không rõ / mất người | vận tốc 0 **ngay** | — | luôn |

> Tên ROLL tính theo **hướng của drone** (đứng đối diện người điều khiển); tay dang ngang
> chỉ về phía drone sẽ đi. Model `v3_pck_body25` (2026-09-14): 8 lớp, 90.8% theo frame,
> 0 lệnh sai/phút trong mô phỏng, tau 0.93.

## Chốt an toàn

1. **Không node nào arm.** Arm là việc của pilot trên tay cầm (SITL: gõ `arm throttle`).
   Test kiểm cả mã nguồn.
2. **STOP không cắt động cơ.** Kill switch thật nằm trên tay cầm.
3. **Dừng thì ngay, đi thì phải giữ.** Đổi tư thế → vận tốc 0 trong frame đó.
4. **Pilot gạt mode ra khỏi GUIDED** → khoá, ngừng gửi.
5. **Watchdog 0.5 s**: không có cử chỉ mới → đứng yên. Chương trình chết thì ArduCopter tự phanh sau 3 s.
6. **Không gửi vận tốc lúc đang cất/hạ cánh**: làm vậy sẽ huỷ lệnh cất cánh của ArduCopter.

---

## Chạy nhanh

```bash
source scripts/env.sh
ros2 run gesture_perception doctor                         # kiểm máy
ros2 launch gesture_bringup camera_only.launch.py          # chỉ nhận diện
ros2 launch gesture_bringup sitl.launch.py                 # drone ảo
ros2 launch gesture_bringup real.launch.py                 # Pixhawk thật — THÁO CÁNH
```

Test logic không cần ROS (chạy được trên Windows):

```bash
python ros2_ws/src/gesture_command/test/test_command_logic.py
python ros2_ws/src/gesture_perception/test/test_perception_core.py
```

---

## Phần cứng

| Thành phần | Cấu hình |
|:---|:---|
| Flight controller | Pixhawk 6X, ArduCopter 4.7 |
| Máy tính bay | Raspberry Pi 5 4 GB, Ubuntu 24.04, ROS 2 Jazzy |
| Camera | Webcam USB 640×480 MJPG |
| Máy mô phỏng / train | Windows 11 + WSL2 Ubuntu 24.04 |


## 📄 Giấy phép

[MIT](LICENSE), kèm **cảnh báo an toàn bay**. Phần mềm này điều khiển thiết bị bay thật.
Luôn có pilot cầm tay cầm, luôn tuân thủ quy định UAV tại địa phương.
