# RUNBOOK ROS 2 — từ thẻ nhớ trắng tới drone ảo bay theo cử chỉ

Làm **đúng thứ tự**. Mỗi bước có **CỔNG**: chưa qua cổng thì không làm bước sau.

```
 BƯỚC 0  cài Pi              scripts/pi_setup.sh
 BƯỚC 1  kiểm máy            ros2 run gesture_perception doctor
 BƯỚC 2  chỉ camera          ros2 launch gesture_bringup camera_only.launch.py
 BƯỚC 2b chấm model bằng video   training/7_replay_check.py   (Pi hoặc laptop)
 BƯỚC 3  cài máy mô phỏng    scripts/sim_setup.sh          (laptop WSL2)
 BƯỚC 4  logic + SITL trần   sitl.launch.py use_camera:=false
 BƯỚC 4b cả chuỗi tự động    sim/chain_test/run.sh         (WSL, video thay camera)
 BƯỚC 5  Gazebo + camera     gazebo.launch.py → run_sitl.sh → sitl.launch.py
 BƯỚC 6  Pixhawk thật        real.launch.py                (THÁO CÁNH QUẠT)
```

---

## Sơ đồ node

```
 Pi 5 ─────────────────────────────────────────────────────────────────────────
  webcam ─► perception_node ──/gesture/detected──► command_node ──► mavros ──┐
            MediaPipe Pose       (label, conf)      máy trạng thái           │
            One-Euro 25 khớp                        chốt an toàn             │
            PCK → TFLite                            /drone/gesture_command   │
                                                                 MAVLink UDP │ 14550
 Laptop (WSL2) ───────────────────────────────────────────────────────────────┤
  gz sim (iris_runway) ◄──JSON lockstep──► ArduCopter SITL ◄── MAVProxy ◄────┘
                                                                ▲
                                                 anh gõ: mode guided / arm throttle
```

Pixhawk thật thay đúng khung dưới cùng: `fcu_url` đổi từ `udp://:14550@` sang
`serial:///dev/ttyAMA0:921600` (TELEM2). Không node nào khác thay đổi.

**Chuỗi trao quyền** (sitl / real launch):

```
pilot gạt GUIDED ──► perception_node BẮT ĐẦU nhận diện ──► giơ ASSUME_GUIDANCE 1 s (mở khoá)
                                                        ──► TAKEOFF / ROLL / STOP / LAND
mode khác GUIDED ──► camera vẫn chạy, MediaPipe KHÔNG chạy, KHÔNG phát cử chỉ nào
```

`camera_only.launch.py` không nối FCU nên luôn nhận diện (để thử camera / model).

---

## BƯỚC 0 — cài Pi (một lần)

Thẻ nhớ: **Ubuntu Desktop 24.04 LTS (64-bit)**. Boot, nối mạng, rồi:

```bash
sudo apt install -y git
git clone https://github.com/PhaPhim2905/drone-gesture-control.git
cd drone-gesture-control
./scripts/pi_setup.sh          # 20-40 phút. Hỏi mật khẩu sudo vài lần
```

Xong thì **đăng xuất rồi đăng nhập lại**, để quyền `dialout` có hiệu lực.

**Mỗi terminal mới** đều phải chạy:

```bash
source ~/drone-gesture-control/scripts/env.sh
```

> **CỔNG 0:** `scripts/build.sh` in `TAT CA DEU QUA` + `CHUOI DAY DU CHAY THONG`,
> và `ros2 pkg list | grep gesture_` ra đủ 4 gói.

---

## BƯỚC 1 — kiểm máy

```bash
ros2 run gesture_perception doctor           # đứng trước camera 10 giây
```

Gửi lại cho em **toàn bộ** phần in ra. Các số cần có:

| Mục | Cần |
|:---|:---|
| mediapipe | `0.10.14`, không có dòng "KHONG CO mp.solutions" |
| mo camera | `qua v4l2, MJPG` |
| fps toan chuoi | **≥ 8** (model đã hiệu chuẩn ở 8 fps) |
| RAM dinh | < 600 MB |
| nhiet do CPU | < 75 °C sau 10 s |
| throttled | `0x0` (khác 0 = nguồn yếu, dùng sạc 5V 5A) |

> **CỔNG 1:** `KET QUA: TAT CA DAT`.
>
> fps < 8 thì DỪNG, gửi số cho em. Đừng chạy tiếp: thời gian giữ và ngưỡng tau
> đều hiệu chuẩn ở 8 fps.

---

## BƯỚC 2 — chỉ camera, không drone

```bash
# terminal 1
ros2 launch gesture_bringup camera_only.launch.py show_window:=true

# terminal 2
ros2 topic echo /gesture/detected --field label
ros2 topic hz /gesture/detected
```

Làm lần lượt 6 cử chỉ, mỗi cái giữ 3 giây. Rồi đi ra khỏi khung hình: phải ra
`NO_OPERATOR`.

So bật / tắt One-Euro ngay lúc đang chạy:

```bash
ros2 param set /perception_node one_euro false
ros2 param set /perception_node one_euro true
```

Xem hình từ laptop (khi Pi không gắn màn hình):

```bash
# Pi
ros2 launch gesture_bringup camera_only.launch.py debug_image_hz:=2.0
# laptop (cùng ROS_DOMAIN_ID=42, cùng mạng LAN)
ros2 run rqt_image_view rqt_image_view /gesture/debug_image/compressed
```

**Độ trễ**: `perception_node` in mỗi 5 s một dòng, đọc từ trái sang:

```
nhan dien: cam 30 fps | xu ly 14 fps | tuoi frame 0/5 ms | pose 60/75 ms | clf 0.4 ms | tong chup->phat 65/82 ms | bo 80 frame cu
```

| Mục | Nghĩa | Pi 5 nên có |
|:--|:--|:--|
| `xu ly` | số frame nhận diện được mỗi giây | ≥ 8 |
| `tuoi frame` | frame nằm chờ bao lâu trước khi xử lý (luôn lấy frame MỚI NHẤT) | < 40 ms |
| `pose` | MediaPipe, khâu nặng nhất | — |
| `tong chup->phat` | từ lúc chụp tới lúc phát cử chỉ = **trễ nhận diện** | < 150 ms |
| `bo N frame cu` | frame camera bị bỏ vì Pi chậm hơn camera: **bình thường**, là cách giữ trễ thấp | — |

Đo trên Pi **không cần đứng trước camera**: phát video take theo nhịp thật.

```bash
ros2 launch gesture_bringup camera_only.launch.py video_path:="$HOME/video/*.mp4" flip:=false
```

(Video của `1_record.py` đã lật gương sẵn nên `flip:=false`. Chép thư mục `data/video`
từ laptop sang Pi bằng `scp`.)

> **CỔNG 2:** mỗi cử chỉ ra đúng nhãn ≥ 80% số frame khi giữ yên. `NEGATIVE`
> (buông tay) **không bao giờ** ra một cử chỉ lệnh.

---

## BƯỚC 2b — chấm model bằng video take

Chạy video qua **đúng** chuỗi của Pi (MediaPipe → One-Euro → PCK → TFLite → luật giữ
của `command_logic`). Chạy trên Pi thì dòng `XU LY` là thời gian thật của Pi.

```bash
python3 training/7_replay_check.py                 # mọi video trong data/video
python3 training/7_replay_check.py --proc-fps 10   # giả lập máy chỉ xử lý 10 fps
python3 training/7_replay_check.py --tau 0.90      # thử ngưỡng tin cậy khác
```

Đo 15/09 trên 30 take có sẵn (laptop):

| tau | take đạt | lệnh SAI | frame đúng |
|:--|:--|:--|:--|
| 0.93 (đang dùng) | 27/30 (HOVER trượt 3) | 0 | 81.4 % |
| 0.90 | 28/30 | 0 | 86.3 % |
| 0.85 | 30/30 | 0 | 95.6 % |

Giả lập 8 và 12 fps: độ chính xác không đổi, vì thời gian giữ tính bằng giây.

⚠️ 30 take này **đã dùng để train** nên điểm lạc quan, và **chưa có video ROLL**.
Quy trình thêm take để model đáng tin hơn:

1. Quay take mới **có video**: `python training/1_record.py --save-video --classes HOVER,ROLL_LEFT,ROLL_RIGHT`
2. **Trước khi train**, chép video mới sang thư mục riêng, chấm model hiện tại:
   `python training/7_replay_check.py --dir data/video_moi` → đây là điểm **công bằng**
3. Train lại (`2_check_poses` → `3_build_dataset` → `4_train`), chấm lại **cùng thư mục** đó
4. Chỉ `5_export_to_ros.py` khi `lenh SAI = 0` và số take đạt không giảm
5. Đổi tau lúc chạy nếu cần: `ros2 param set /perception_node tau_override 0.90`

> **CỔNG 2b:** `lenh SAI = 0` trên take mới. Có lệnh sai thì quay thêm take cho cặp nhãn bị nhầm.

---

## BƯỚC 3 — cài máy mô phỏng (laptop, một lần)

PowerShell:

```powershell
wsl --install -d Ubuntu-24.04
notepad $env:USERPROFILE\.wslconfig
```

Ghi vào `.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

```powershell
wsl --shutdown
```

Trong Ubuntu (WSL):

```bash
git clone https://github.com/PhaPhim2905/drone-gesture-control.git
cd drone-gesture-control
./scripts/sim_setup.sh          # 30-60 phút, ~6 GB
```

> **CỔNG 3:** `gz sim --versions` ra `8.x`, và `~/ardupilot/build/sitl/bin/arducopter` tồn tại.

---

## BƯỚC 4 — logic với SITL trần, chưa camera

Mục đích: kiểm chiều bay và chốt an toàn **mà không có sai số nhận diện xen vào**.

```bash
# laptop T1 — SITL trần, không Gazebo
./sim/run_sitl.sh --no-gazebo --out <IP Pi>

# Pi T1
ros2 launch gesture_bringup sitl.launch.py use_camera:=false

# Pi T2 — theo dõi
ros2 topic echo /drone/gesture_command
```

Bắn cử chỉ bằng tay (Pi T3). Mỗi lệnh chạy **liên tục** (`-r 10`), Ctrl+C để
đổi cử chỉ. Đó chính là "giữ tư thế":

```bash
G() { ros2 topic pub -r 10 /gesture/detected gesture_interfaces/msg/Gesture "{label: $1, confidence: 0.99}"; }
```

| # | Làm | Ở đâu | Phải thấy |
|:-:|:---|:---|:---|
| 1 | `G ROLL_LEFT` | Pi | `state: LOCKED`, `streaming: false` |
| 2 | `mode guided` rồi `arm throttle` | MAVProxy | `armed: true` |
| 3 | `G ASSUME_GUIDANCE` (2 s) | Pi | `last_event: MO QUYEN DIEU KHIEN` |
| 4 | `G TAKEOFF` (2 s) | Pi | log `takeoff 2.0 m: FCU nhan`, drone lên 2 m |
| 5 | `G ROLL_RIGHT` (3 s) | Pi | `velocity_flu.y: -1.0` (**âm** = drone sang PHẢI của nó = sang trái của anh) |
| 6 | Ctrl+C, `G NEGATIVE` | Pi | `velocity_flu.y: 0.0` **ngay** |
| 7 | `G STOP` (1 s) | Pi | `LOCKED`, MAVProxy báo `LOITER` |
| 8 | `G LAND` (1 s) | Pi | MAVProxy báo `LAND`, drone hạ |

> ArduCopter **tự disarm sau ~10 s** nếu arm mà không cất cánh. Làm bước 3-4
> ngay sau bước 2; lỡ disarm thì arm lại.

Kiểm trên MAVProxy: `status LOCAL_POSITION_NED`. Ở bước 5, `vy` phải **dương**
(NED: y dương = đông; drone mũi hướng bắc sang phải của nó = sang đông).

> **CỔNG 4:** đủ 8 dòng. **Sai chiều ở bước 5 thì DỪNG**, không bay thật.

---

## BƯỚC 4b — cả chuỗi tự động bằng video (WSL)

Một lệnh, không cần camera, không cần người: SITL riêng + MAVROS + `perception_node`
đọc video take + `command_node`, pilot giả gạt GUIDED và arm. Không đụng SITL/Gazebo
anh đang mở.

```bash
./sim/chain_test/run.sh          # ~4 phút
```

Kết quả 15/09 (7/7 đạt):

```
  DAT   chua GUIDED: 0 cu chi duoc phat
  DAT   GUIDED: bat dau nhan dien
  DAT   ASSUME_GUIDANCE mo khoa
  DAT   TAKEOFF bang cu chi, len > 1.5 m (2.08 m)
  DAT   LAND bang cu chi
  DAT   sang LAND: ngung nhan dien
  DAT   ha canh xong, disarm
```

Trễ đo được trong WSL: chụp frame → `command_node` **≈ 20–24 ms**.

Hai lỗi chỉ lộ ra khi chạy cả chuỗi, đã sửa:
- MAVROS bị đặt namespace thừa → topic thành `/mavros/mavros/state`, không node nào nhận được state.
- ArduPilot không gửi `EXTENDED_SYS_STATE` → không biết drone dưới đất → vận tốc 0 huỷ lệnh TAKEOFF.
  `command_node` giờ tự xin message này.

> **CỔNG 4b:** `7 dat, 0 truot`. Chạy lại sau MỖI lần train, đổi tau, sửa perception / command.

---

## BƯỚC 5 — Gazebo + camera thật

Ba terminal, **đúng thứ tự**:

```bash
# laptop T1 — thế giới Gazebo (đợi thấy drone trên đường băng)
source scripts/env.sh
ros2 launch gesture_bringup gazebo.launch.py

# laptop T2 — SITL nối Gazebo (đợi "EKF3 IMU0 is using GPS")
./sim/run_sitl.sh --out <IP Pi>

# Pi T1 — toàn bộ hệ, có camera
ros2 launch gesture_bringup sitl.launch.py show_window:=true
```

Trình tự bay:

1. **Gạt GUIDED**: tay cầm kênh 5 nấc cao (bước 5b), hoặc MAVProxy `mode guided`.
   Log Pi phải hiện `BAT DAU NHAN DIEN (GUIDED)`. Trước đó camera **không** nhận diện.
2. **Arm** bằng tay cầm (chụm 2 cần) hoặc MAVProxy `arm throttle`. Không node nào arm.
3. Trước camera: **ASSUME_GUIDANCE** giữ 1 s → **TAKEOFF** giữ 1 s → drone lên 2 m
4. **ROLL_RIGHT** (tay trái dang ngang) giữ → drone Gazebo sang **phải của nó**. Anh đứng đối diện drone thì thấy nó đi về phía **tay trái** của anh
5. Buông tay (**NEGATIVE**) → drone dừng ngay
6. **STOP** → drone đứng yên, chuyển LOITER → camera **ngừng** nhận diện.
   Muốn điều khiển tiếp: pilot gạt kênh 5 xuống rồi lên lại GUIDED.
7. **LAND** → hạ cánh, camera ngừng nhận diện

Xem trễ thật trên Pi trong log: dòng `nhan dien: ... tong chup->phat` (bước 2) và
`tre chup frame -> command_node`.

Chạy toàn bộ trên Pi, không cần laptop (chậm, chỉ để thử):

```bash
ros2 launch gesture_bringup gazebo.launch.py headless:=true &
./sim/run_sitl.sh &
ros2 launch gesture_bringup sitl.launch.py
```

> **CỔNG 5:** làm trọn 7 bước 3 lần liên tiếp không lệnh sai. Ghi video màn hình
> làm bản dự phòng cho buổi demo.

---

## BƯỚC 5b — tay cầm THẬT lái drone ẢO

Mục đích: pilot tập đúng thao tác của ngày bay thật — arm bằng cần, gạt mode,
E-stop, mất sóng — trên drone ảo, cùng lúc với cử chỉ.

```
receiver UniRC7 ─► RC IN Pixhawk 6X ─USB─► laptop: sim/rc_bridge.py ─► SITL
```

Pixhawk 6X ở bước này **chỉ là bộ đọc receiver**:

- cắm USB, **KHÔNG cắm pin / không cấp nguồn ESC**, tháo cánh
- đã **Radio Calibration** trong Mission Planner
- **đóng Mission Planner** (script cần cổng COM)

```bash
# laptop T1 (WSL) — --proposal = kênh 5 LOITER/ALT_HOLD/GUIDED, kênh 6 E-stop,
./sim/run_sitl.sh --no-gazebo --real-params --compass-on --proposal --wipe

# laptop T2 (PowerShell Windows) — xem trước kênh có nhảy theo tay cầm không
python sim\rc_bridge.py --show
# rồi chạy thật (Ctrl+C để dừng)
python sim\rc_bridge.py

# T3 — hệ cử chỉ, hành xử như drone THẬT: pilot tự gạt GUIDED
ros2 launch gesture_bringup sitl.launch.py command_config:=command_real.yaml
```

| # | Làm trên tay cầm | Phải thấy |
|:-:|:---|:---|
| 1 | Gạt từng cần, từng công tắc | `rc_bridge.py`: số kênh tương ứng đổi |
| 2 | Công tắc kênh 5: thấp / giữa / cao | MAVProxy: `LOITER` / `ALT_HOLD` / `GUIDED` |
| 3 | Kênh 5 thấp. **Chụm 2 cần vào trong** (trái xuống-phải, phải xuống-trái) 1.5 s → thấy `CSC: buong can de ARM` → **buông cần về giữa**, ga vẫn thấp | `CSC: ARM` |
| 4 | Đẩy ga lên giữa | drone ảo cất cánh, giữ độ cao |
| 5 | Kênh 5 cao → ASSUME_GUIDANCE → ROLL_RIGHT | cử chỉ lái drone ảo |
| 6 | Kênh 5 thấp giữa chừng | drone về LOITER **ngay**, tay cầm thắng cử chỉ |
| 7 | **Tắt tay cầm** | `MAT SONG`; ~3 s sau MAVProxy `Radio Failsafe` → `LAND` |
| 8 | Bật lại, **E-stop kênh 6** khi drone ảo đang trên đất | `RC6: MotorEStop HIGH`, không arm được tới khi nhả |

> Bỏ `--proposal` ở bước 7 thì drone ảo **không làm gì** khi mất sóng: đó là
> `FS_THR_ENABLE = 0` trong file Pixhawk cũ. Chính vì vậy đề xuất đổi.
>
> **CỔNG 5b:** đủ 10 dòng. Giá trị trong `sim/params/sitl_6x_proposal.parm` chỉ
> ghi xuống Pixhawk 6X **sau khi** qua cổng này và anh đồng ý.

---

## BƯỚC 6 — Pixhawk 6X thật, Pi 5 qua TELEM2

> ### 🔴 THÁO CÁNH QUẠT. Lượt đầu KHÔNG cắm nguồn ESC.

```
Pixhawk 6X TELEM2 ──3 dây──► UART GPIO Pi 5 (/dev/ttyAMA0, 921600)   hệ cử chỉ
Pixhawk 6X USB ────────────► laptop Mission Planner                    theo dõi CÙNG LÚC
```

**Pixhawk 6X** (anh đặt trong Mission Planner, xem lại trước khi Write):
đã hiệu chỉnh accel / compass / radio; `FRAME_CLASS 1`; `SERIAL2_PROTOCOL 2`,
`SERIAL2_BAUD 921`, `BRD_SER2_RTSCTS 0`; các giá trị trong `sim/params/sitl_6x_proposal.parm`
đã qua cổng 5b (kênh 5 mode, `RC6_OPTION 31`, `RC5_OPTION 0`, `FS_THR_ENABLE`,
`ARMING_RUDDER 0`, `SCR_ENABLE 1`); chép `pixhawk_tools/lua/csc_arm.lua` vào thẻ nhớ
`APM/scripts/` rồi khởi động lại, Messages phải có `CSC: chum 2 can vao trong...`;
`SERVO1..4_FUNCTION = 33..36` (board mới, động cơ 4 về lại MAIN OUT 4).

**Pi, một lần** (sơ đồ dây nằm đầu file):

```bash
./scripts/pi_uart_setup.sh && sudo reboot
python3 pixhawk_tools/diagnostic.py --port /dev/ttyAMA0 --baud 921600   # phải thấy HEARTBEAT
```

**Chạy** — ngoài trời, đợi Mission Planner báo GPS 3D fix (GUIDED cần vị trí):

```bash
ros2 launch gesture_bringup real.launch.py show_window:=true
ros2 topic echo /drone/gesture_command        # Pi T2
```

Khác SITL: `request_guided_on_enable: false` (pilot tự gạt), `speed_xy: 0.5`.

| # | Làm | Phải thấy |
|:-:|:---|:---|
| 1 | `ros2 topic echo /mavros/state --once` | `connected: true`, mode trùng Mission Planner |
| 2 | Kênh 5 LOITER, giơ ASSUME_GUIDANCE | `LOCKED`, mode **không** đổi (cử chỉ không giành mode) |
| 3 | Kênh 5 GUIDED, giơ ASSUME_GUIDANCE | `ENABLED` |
| 4 | ROLL_RIGHT | `ros2 topic echo /mavros/setpoint_raw/local`: `velocity.y: -0.5` |
| 5 | Che camera | trong 0.8 s về HOVER, vận tốc 0 |
| 6 | STOP | Mission Planner báo `LOITER` |
| 7 | Ctrl+C node khi đang ENABLED | Mission Planner vẫn thấy FCU; drone không nhận lệnh mới |
| 8 | Gạt E-stop kênh 6 | Mission Planner: `RC6: MotorEStop HIGH` |
| 9 | Rút dây TELEM2 | `/mavros/state` `connected: false`; Mission Planner qua USB vẫn nối |

Lượt này **không arm**. Arm, TAKEOFF và LAND trên Pixhawk thật chỉ làm ở lượt sau,
có nguồn ESC, vẫn tháo cánh, khi cả 9 dòng trên đã đúng.

---

## Lệnh hay dùng

| Việc | Lệnh |
|:---|:---|
| sơ đồ node đang chạy | `ros2 run rqt_graph rqt_graph` (laptop) |
| trạng thái lệnh | `ros2 topic echo /drone/gesture_command` |
| FCU có nối không | `ros2 topic echo /mavros/state --once` |
| chỉnh ngưỡng khi đang chạy | `ros2 param set /perception_node tau_override 0.9` |
| ghi lại buổi thử | `ros2 bag record /gesture/detected /drone/gesture_command /mavros/state` |
| build lại sau git pull | `./scripts/build.sh` |
| test logic (không cần ROS) | `python3 ros2_ws/src/gesture_command/test/test_command_logic.py` |

## Sự cố

| Triệu chứng | Nguyên nhân | Xử lý |
|:---|:---|:---|
| `No module named mediapipe` khi `ros2 launch` | terminal chưa `source scripts/env.sh` | source rồi chạy lại |
| `No module named pymavlink` ở run_sitl.sh | nhầm python | script tự đổi sang `~/venv-ardupilot`; nếu vẫn lỗi: `source ~/venv-ardupilot/bin/activate` |
| MAVROS: `GeographicLib exception` | thiếu dữ liệu geoid | `sudo /opt/ros/jazzy/lib/mavros/install_geographiclib_datasets.sh` |
| `/mavros/state` `connected: false` | sai IP `--out` / tường lửa / WSL chưa mirrored | `ping <IP Pi>` từ WSL; kiểm `.wslconfig` |
| `takeoff: FCU TU CHOI` | chưa GUIDED, chưa arm, hoặc EKF chưa sẵn sàng | đợi "EKF3 ... using GPS" rồi thử lại |
| `set_mode: MAVROS chua san sang` | plugin chưa nạp | xem log mavros có `sys_status` / `command` không |
| `$'\r': command not found` | script bị đổi sang CRLF | `git config core.autocrlf input` rồi clone lại |
| rqt ở laptop không thấy node Pi | khác `ROS_DOMAIN_ID` hoặc discovery `LOCALHOST` | cả hai máy `source scripts/env.sh` |
