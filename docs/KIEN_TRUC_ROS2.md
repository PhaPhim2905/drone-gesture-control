# KIẾN TRÚC ROS 2 — luồng sự kiện bên trong `ros2_ws`

RUNBOOK trả lời *"gõ lệnh gì"*. File này trả lời *"bên trong đang chảy ra sao"*:
node nào làm việc gì, topic nào mang cái gì, ai là client ai là server.

Mục có đánh số để tiện hỏi đúng chỗ: "mục 5 chưa hiểu".

---

## 1. Bản đồ một trang

Toàn hệ chỉ có **3 node**. Không có node camera riêng, không có action server.

```
                     ┌─ /gesture/detected ─────────────┐
 camera ──► perception_node                         command_node ──► /mavros/setpoint_raw/local
 (thread đọc)   │  ▲                                  │  ▲   │              │
                │  └── /mavros/state ◄────┐           │  │   └► /drone/gesture_command (HUD)
                │                         │           │  │
                └── /gesture/debug_image  │           │  └── /mavros/state, /mavros/extended_state
                                          │           │
                                      ┌───┴───────────▼────┐  service: set_mode, cmd/takeoff, cmd/command
                                      │    mavros_node     │
                                      └─────────┬──────────┘
                                                │ MAVLink
                                   UDP 14550 (SITL)  |  serial /dev/ttyAMA0 (Pixhawk thật)
```

| Node | Gói | File | Việc duy nhất của nó |
|:--|:--|:--|:--|
| `perception_node` | `gesture_perception` | `perception_node.py` | frame → tên cử chỉ. **Không biết gì về drone** |
| `command_node` | `gesture_command` | `command_node.py` | cử chỉ → lệnh bay an toàn |
| `mavros_node` | `mavros` (của ROS) | — | dịch ROS ↔ MAVLink. **Là server của mọi service** |

Ranh giới quan trọng: `perception_node` nói *"frame này trông như ROLL_LEFT, 0.97"*.
Nó không biết drone đã arm chưa, đang ở mode nào cũng không quan tâm — trừ đúng
một việc: có được phép nhận diện không (mục 7). `command_node` mới là nơi quyết
định có bay hay không.

---

## 2. Topic — ai phát, ai nghe

| Topic | Kiểu | Phát | Nghe | QoS | Nhịp |
|:--|:--|:--|:--|:--|:--|
| `/gesture/detected` | `gesture_interfaces/Gesture` | perception | command | **best-effort, depth 1** | mỗi frame xử lý xong (8–15 Hz) |
| `/gesture/debug_image/compressed` | `sensor_msgs/CompressedImage` | perception | `rqt_image_view` | best-effort, depth 1 | `debug_image_hz` (mặc định 0 = tắt) |
| `/mavros/state` | `mavros_msgs/State` | mavros | **cả hai node** | sensor_data | 1 Hz |
| `/mavros/extended_state` | `mavros_msgs/ExtendedState` | mavros | command | sensor_data | 5 Hz (phải xin, mục 6) |
| `/mavros/setpoint_raw/local` | `mavros_msgs/PositionTarget` | command | mavros | reliable, depth 10 | 20 Hz khi đang bay |
| `/drone/gesture_command` | `gesture_interfaces/CommandStatus` | command | anh xem | reliable, depth 10 | 20 Hz, **luôn luôn** |

**Ảnh không bao giờ đi qua DDS.** Frame 640×480 ≈ 0.9 MB; đẩy 30 lần/giây chỉ để
tuần tự hoá đã mất vài fps trên Pi 5. Camera nằm **trong** `perception_node`, ra
ngoài chỉ còn nhãn + 50 số thực. Đó là lý do không tách node camera. Muốn xem hình
thì bật `debug_image_hz` (JPEG nén, nhịp thấp).

**`/gesture/detected` cố ý best-effort depth 1.** Mất một frame cử chỉ thì không
sao; xử lý một frame **cũ** mới nguy hiểm.

`/drone/gesture_command` là **topic duy nhất cần xem** để biết drone đang được bảo
làm gì và vì sao: `state`, `candidate`, `hold_progress`, `motion`, `streaming`,
`blocked_reason`, `last_event`, `armed`, `mode`, `landed_state`.

---

## 3. Service — client là ai, server là ai

**Server luôn là `mavros_node`. Client luôn là `command_node`.**
Hai node của mình không mở service nào, không có action nào.

| Service | Kiểu | Gọi khi |
|:--|:--|:--|
| `/mavros/set_mode` | `mavros_msgs/SetMode` | STOP → `LOITER`; LAND → `LAND`; (tuỳ chọn) ENABLE → `GUIDED` |
| `/mavros/cmd/takeoff` | `mavros_msgs/CommandTOL` | cử chỉ TAKEOFF, chỉ khi armed + GUIDED + đang dưới đất |
| `/mavros/cmd/command` | `mavros_msgs/CommandLong` | xin `EXTENDED_SYS_STATE` 5 Hz (mục 6) |

Mọi lời gọi đều **bất đồng bộ** (`call_async` + callback), nên vòng 20 Hz không bao
giờ bị chặn. Service chưa sẵn sàng thì log `MAVROS chua san sang`, không treo.

> **Không có client `/mavros/cmd/arming`** trong toàn bộ mã nguồn —
> `test/test_command_logic.py` kiểm chính văn bản mã nguồn để giữ điều đó.
> Arm/disarm là việc của pilot trên tay cầm.

MAVROS chỉ nạp 6 plugin (`config/mavros.yaml`): `sys_status`, `sys_time`, `command`,
`setpoint_raw`, `local_position`, `home_position`. Mặc định nó nạp ~50 plugin, mỗi
cái là subscriber/timer riêng — tốn CPU Pi mà không ai đọc.

---

## 4. Đường đi của một frame, theo thời gian

```
t=0    thread đọc camera chụp frame, đóng dấu thời gian   (LatestFrame._publish)
       └─ frame trước chưa ai lấy → dropped++    (bỏ frame cũ là CỐ Ý)
t≈0    vòng xử lý lấy frame MỚI NHẤT
       ├─ mode_gate.check(): FCU có đang GUIDED không?  KHÔNG → dừng tại đây
       ├─ MediaPipe Pose        ~60–75 ms   ← khâu nặng nhất
       ├─ 33 landmark → BODY25 pixel
       ├─ One-Euro lọc rung     25 khớp
       ├─ chuẩn hoá PCK         → 50 số
       └─ TFLite                ~0.4 ms → nhãn + độ tin cậy
t≈80   publish /gesture/detected
       header.stamp = LÚC CHỤP (t=0), KHÔNG phải lúc phát
       ↓
       command_node._on_gesture()
       ├─ đo trễ = now − header.stamp      → dòng log "tre chup frame -> command_node"
       ├─ logic.on_gesture(): đếm thời gian giữ, có thể đổi state / motion
       └─ nếu state hoặc motion ĐỔI → gọi _tick() NGAY, không đợi nhịp 20 Hz
                                        (tiết kiệm tới 50 ms)
```

Hai lớp thống kê tách biệt:

| Dòng log | Ở đâu | Đo cái gì |
|:--|:--|:--|
| `nhan dien: ... tong chup->phat` | perception | trong một node, chưa qua mạng |
| `tre chup frame -> command_node` | command | đã qua DDS + mạng LAN |

Số thứ hai luôn lớn hơn; chênh lệch chính là DDS và mạng. Đo trong WSL ngày
15/09: ≈ 20–24 ms.

---

## 5. Hai vòng lặp chạy song song trong `command_node`

Đây là chỗ hay nhầm nhất.

| | `on_gesture()` — theo sự kiện | `tick()` — 20 Hz đều đặn |
|:--|:--|:--|
| Kích hoạt bởi | có message cử chỉ mới | timer `control_hz` |
| Làm gì | đếm **thời gian giữ**, chốt lệnh, đổi `state` / `motion` | quyết định **có gửi vận tốc không**, phát `CommandStatus` |
| Không có nó | không bao giờ có lệnh mới | drone mất setpoint, ArduPilot tự huỷ GUIDED |

`tick()` không nhớ gì về thời gian, nên gọi thêm một nhịp ngoài lịch là an toàn —
đó là lý do `on_gesture` được phép gọi thẳng nó khi lệnh vừa đổi.

Trong `perception_node` cũng có hai luồng: **thread đọc camera** (chạy hết tốc độ,
chỉ giữ frame cuối) và **thread xử lý** (lấy frame mới nhất). Chúng không chạy trong
executor của ROS; `rclpy` cho phép publish từ thread khác.

---

## 6. `EXTENDED_SYS_STATE` — cái bẫy đã làm hỏng TAKEOFF

ArduPilot **không tự gửi** `EXTENDED_SYS_STATE` trên cổng MAVROS nối vào (SITL
SERIAL0; TELEM2 nếu `SR2_*` chưa bật). Thiếu nó thì `landed_state = UNDEFINED`.

Khi `UNDEFINED`, `tick()` cố tình **im lặng** thay vì gửi vận tốc 0 — vì gửi vận tốc
liên tục trong lúc ArduCopter đang cất cánh sẽ **huỷ lệnh TAKEOFF** (GUIDED chuyển
submode sang velocity). Lỗi này chỉ lộ ra khi chạy cả chuỗi, ngày 15/09.

`_request_extended_state()` xin message 245 ở 5 Hz bằng `MAV_CMD_SET_MESSAGE_INTERVAL`
(511), thử lại mỗi 3 s, tối đa 10 lần.

---

## 7. Máy trạng thái — chỉ có 2 trạng thái

Toàn bộ quyết định nằm trong `command_logic.py`. **File đó không import ROS**, nên
test được trên laptop không cần drone, không cần cài ROS.

```
LOCKED ──ASSUME_GUIDANCE giữ 1.0 s──► ENABLED ──STOP / LAND / mất người 0.8 s──► LOCKED
                                              ──pilot gạt khỏi GUIDED──────────► LOCKED
                                              ──mất kết nối FCU───────────────► LOCKED
```

Khi **LOCKED**, chỉ 3 cử chỉ được nghe: `ASSUME_GUIDANCE`, `STOP`, `LAND`.
Mọi cử chỉ khác bị bỏ qua hoàn toàn.

Thời gian giữ (đo 09/09 trên 3.765 mẫu / 31 take): mặc định **1.0 s**, `LAND` 0.8 s,
`STOP` **0.6 s** — STOP cố ý ngắn hơn, vì *STOP nhầm chỉ làm drone đứng yên, STOP
muộn thì không cứu được gì*.

Hai nguyên tắc quyết định cách viết:

- **Dừng thì ngay, đi thì phải giữ.** Đổi sang bất kỳ cử chỉ nào khác lệnh đang chạy
  → `motion = HOVER` ngay frame đó. Lệnh mới chỉ áp sau khi giữ đủ.
- **Không chắc thì đứng yên.** `AMBIGUOUS`, `NEGATIVE`, `NO_OPERATOR`, nhãn lạ đều
  thành HOVER.

---

## 8. Năm cửa chặn trước khi vận tốc được gửi

`tick()` trả `None` (= không gửi setpoint) và ghi lý do vào `blocked_reason`:

| Điều kiện | `blocked_reason` |
|:--|:--|
| chưa `ENABLED` | `LOCKED - gio ASSUME_GUIDANCE` / `cho pilot gat GUIDED` |
| chưa armed | `chua ARM (pilot arm bang tay cam)` |
| mode ≠ GUIDED | `dang LOITER, can GUIDED` |
| `landed_state ≠ IN_AIR` | `duoi dat - gio TAKEOFF` / `dang cat canh` / `cho EXTENDED_SYS_STATE` |
| 0.5 s không có cử chỉ mới | vẫn gửi, nhưng **vận tốc 0** (watchdog: perception treo = drone đứng) |

Cộng với cổng ở tầng trên (`mode_gate`), hệ có **3 lớp xếp chồng**: pilot chưa gạt
GUIDED thì MediaPipe còn không chạy → không có cử chỉ nào ra đời → `command_node`
cũng không có gì để xử lý.

---

## 9. Chỗ đổi dấu nguy hiểm nhất

```
ROLL_RIGHT  →  lệnh RIGHT  →  FRD (0, +v, 0)  →  FLU (0, −v, 0)  →  MAVROS  →  FCU
```

MAVROS nhận **FLU** (x tiến, y **trái**, z lên) rồi tự đổi sang FRD khi
`coordinate_frame = FRAME_BODY_NED`. Gửi thẳng số FRD vào MAVROS là drone bay
**ngược** cả chiều ngang lẫn chiều cao. Hàm `frd_to_flu()` tồn tại chỉ vì chuyện này.

Quy ước tên: **LEFT / RIGHT luôn là hướng của drone**. `ROLL_RIGHT` = drone sang phải
*của nó* = sang **trái** của người điều khiển đứng đối diện. `mirror_left_right = true`
đảo lại, nhưng chỉ được bật sau khi đã **quay lại dữ liệu** theo quy ước đó.

---

## 10. Ba cách chạy, khác nhau ở đâu

| Launch | Node được bật | `require_mode` | Dùng khi |
|:--|:--|:--|:--|
| `camera_only.launch.py` | chỉ perception | **rỗng → luôn nhận diện** | thử camera / model, không có FCU |
| `sitl.launch.py` | cả 3, `fcu_url=udp://:14550@` | `GUIDED` | RUNBOOK bước 4, 4b, 5 |
| `real.launch.py` | cả 3, `fcu_url=serial:///dev/ttyAMA0:921600` | `GUIDED` | RUNBOOK bước 6 |

Hai cái sau chỉ là vỏ mỏng quanh `core.launch.py`, khác đúng **một dòng `fcu_url`**
và file config. `use_camera:=false` bỏ `perception_node` để bắn cử chỉ giả bằng
`ros2 topic pub`.

Tham số đổi được **lúc đang chạy**, không cần khởi động lại:

```bash
ros2 param set /perception_node one_euro false
ros2 param set /perception_node tau_override 0.90
```

---

## 11. Cách tự nhìn luồng khi đang chạy

```bash
ros2 node list                                 # phải đúng 3 node
ros2 node info /command_node                   # pub / sub / client của nó
ros2 topic echo /drone/gesture_command         # xem TẤT CẢ quyết định
ros2 topic hz /gesture/detected                # fps thật của cả chuỗi
ros2 service list | grep mavros                # server do mavros mở
ros2 run rqt_graph rqt_graph                   # vẽ lại sơ đồ mục 1
ros2 bag record /gesture/detected /drone/gesture_command /mavros/state
```

Test không cần ROS, chạy được cả trên Windows:

```bash
python ros2_ws/src/gesture_command/test/test_command_logic.py
python ros2_ws/src/gesture_perception/test/test_perception_core.py
```

---

## 12. Vì sao chia file như vậy

Mỗi gói tách làm đôi: **phần logic không dính ROS** và **vỏ ROS mỏng**. Lý do là để
test trên laptop Windows không cài ROS, và để `doctor.py` đo được fps thật trên Pi
mà không phải khởi động cả hệ.

| Không dính ROS (test được ở đâu cũng được) | Vỏ ROS (chỉ chuyển dữ liệu) |
|:--|:--|
| `perception_core.py` — chuỗi 1 frame | `perception_node.py` |
| `one_euro.py`, `pck_format.py`, `classifier.py` | |
| `frame_source.py` — lấy frame mới nhất | |
| `mode_gate.py` — cổng GUIDED | |
| `command_logic.py` — **mọi chốt an toàn** | `command_node.py` |

Nguyên tắc: file vỏ không được chứa quyết định nào. Đọc `command_node.py` sẽ thấy nó
chỉ đọc message → gọi logic → gửi kết quả đi.
