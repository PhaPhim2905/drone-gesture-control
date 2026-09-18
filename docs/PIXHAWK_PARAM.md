# Param Pixhawk — đổi tên, đổi đơn vị, và cách ghi an toàn

Kiến thức rút ra khi làm với board 6C — vẫn đúng cho **Pixhawk 6X** đang dùng.
Gộp từ `docs/pi5_pixhawk_setup.md` cũ, đã bỏ phần trùng với
[`RUNBOOK_ROS2.md`](RUNBOOK_ROS2.md) (đấu dây TELEM2 → xem bước 6) và phần nói về
code v2 đã xoá.

> 🔴 **Luôn `--dry-run` trước khi ghi, rồi đọc lại để chứng minh đã ghi đúng.**
> Đây là drone thật.

Công cụ: [`pixhawk_tools/setup_params.py`](../pixhawk_tools/setup_params.py) —
tải **toàn bộ** bảng param rồi mới ghi, nên biết param nào không tồn tại trên
firmware này và tự chọn đúng tên khi ArduPilot đổi tên giữa các phiên bản.

---

## 1. Vì sao không thấy `ARMING_CHECK` trong full parameter list

**ArduPilot 4.6 trở lên đã đổi tên nó thành `ARMING_SKIPCHK`** — đã kiểm chứng
trực tiếp: trong 1033 param của board **không hề có** `ARMING_CHECK`.

Và **nghĩa của nó ngược lại**:

| | Bitmask nghĩa là gì | Giá trị an toàn nhất |
|:--|:--|:--:|
| `ARMING_CHECK` (≤ 4.5) | **CHẠY** những check nào | `1` = chạy hết |
| `ARMING_SKIPCHK` (≥ 4.6) | **BỎ QUA** những check nào | `0` = không bỏ qua cái nào |

Viết nhầm số của cái này vào cái kia là tự tay tắt hết kiểm tra an toàn mà vẫn tưởng
đang bật.

Ba nguyên nhân còn lại nếu vẫn không thấy param:

1. **Board đang chạy PX4, không phải ArduPilot.** PX4 dùng `COM_ARM_*`, `CBRK_*`.
   `--info` in ra `autopilot = 3` là ArduPilot, `= 12` là PX4.
2. **Bảng param tải chưa xong** trong Mission Planner — bấm *Refresh Params*.
3. **Nhìn nhầm tab** — `Full Parameter List`, không phải `Full Parameter Tree`.

```bash
python pixhawk_tools/setup_params.py --port /dev/ttyAMA0 --info
python pixhawk_tools/setup_params.py --port /dev/ttyAMA0 --find ARMING
python pixhawk_tools/setup_params.py --port /dev/ttyAMA0 --get ARMING_SKIPCHK
```

---

## 2. ArduPilot 4.6+ đổi tên **và** đổi đơn vị

| Tên cũ (≤ 4.5) | Tên mới (4.6+) | Đổi đơn vị |
|:--|:--|:--|
| `ARMING_CHECK` | `ARMING_SKIPCHK` | **nghĩa ngược** |
| `PILOT_SPEED_UP` cm/s | `PILOT_SPD_UP` | → **m/s** |
| `PILOT_ACCEL_Z` | `PILOT_ACC_Z` | → **m/s²** |
| `ANGLE_MAX` centi-độ | `ATC_ANGLE_MAX` | → **độ** |
| `LOITER_SPEED` cm/s | `LOIT_SPEED_MS` | → **m/s** |
| `SYSID_MYGCS` | `MAV_GCS_SYSID` | |

> ⚠️ Ghi `150` vào `PILOT_SPD_UP` = ra lệnh leo **150 m/s**.
> `--profile` lưu sẵn giá trị riêng cho từng tên nên không nhầm được.

---

## 3. `ARMING_CHECK` là bitmask, không phải on/off

Giá trị `1` = **bật tất cả**. Muốn bật chọn lọc thì phải liệt kê từng bit —
ArduPilot không có kiểu "1 trừ đi GPS". Tool nhận tên check và tự cộng bit:

```bash
python pixhawk_tools/setup_params.py --arming "all-except gps,gps_config"   # bay trong nhà
python pixhawk_tools/setup_params.py --arming none                          # bench, ĐÃ THÁO CÁNH
python pixhawk_tools/setup_params.py --arming all                           # trước khi bay thật
```

| Tên trong tool | Bit | Giá trị | Kiểm tra gì |
|:--|:--:|--:|:--|
| `all` | 0 | 1 | bật toàn bộ |
| `baro` | 1 | 2 | khí áp kế |
| `compass` | 2 | 4 | la bàn |
| `gps` | 3 | 8 | GPS đã lock chưa |
| `ins` | 4 | 16 | accel + gyro |
| `params` | 5 | 32 | param hợp lệ |
| `rc` | 6 | 64 | RC đã calibrate chưa |
| `board_volt` | 7 | 128 | điện áp board |
| `battery` | 8 | 256 | mức pin |
| `logging` | 10 | 1024 | ghi log được không |
| `safety` | 11 | 2048 | nút safety switch |
| `gps_config` | 12 | 4096 | cấu hình GPS |
| `system` | 13 | 8192 | hệ thống |
| `mission` | 14 | 16384 | mission |
| `rangefinder` | 15 | 32768 | cảm biến khoảng cách |
| `camera` | 16 | 65536 | camera |
| `aux_auth` | 17 | 131072 | uỷ quyền phụ |
| `vision` | 18 | 262144 | visual odometry |
| `fft` | 19 | 524288 | phân tích rung |

> ⚠️ Tắt **mọi** kiểm tra là tắt cả những cái đang bảo vệ anh khỏi gyro chưa
> calibrate hay pin sắp cạn. Chỉ dùng khi **đã tháo cánh quạt**, và trả về `all`
> trước khi lắp cánh.

---

## 4. Param cho ALT_HOLD (giữ độ cao)

ALT_HOLD **không cần GPS** — chỉ cần khí áp kế + IMU. Cái quyết định nó đứng yên
được hay không là `MOT_HOVER_LEARN`.

```bash
python pixhawk_tools/setup_params.py --profile althold --dry-run   # xem trước
python pixhawk_tools/setup_params.py --profile althold             # ghi thật
```

| Param (4.6+) | Đặt | Ý nghĩa |
|:--|--:|:--|
| `PILOT_SPD_UP` | 1.5 **m/s** | Tốc độ leo tối đa |
| `PILOT_SPD_DN` | 1.0 **m/s** | Tốc độ hạ tối đa |
| `PILOT_ACC_Z` | 1.5 **m/s²** | Gia tốc lên/xuống. Nhỏ = mượt hơn |
| `THR_DZ` | 150 | Vùng chết quanh giữa cần ga (PWM) |
| `MOT_HOVER_LEARN` | 2 | Tự học ga hover **và lưu lại** |
| `ATC_ANGLE_MAX` | 15 **độ** | Nghiêng tối đa 15° |

Bay trong nhà không GPS thì EKF sẽ kêu thiếu nguồn vị trí ngang:
`--profile indoor-nogps`.

---

## 5. Reset về mặc định — và cái giá của nó

**Reset mất luôn hiệu chuẩn**: accel, compass, RC, ESC. Sau khi reset, drone **không
arm được** cho tới khi calibrate lại hết.

Trước khi reset: Mission Planner → Full Parameter List → **Save to file**.

```bash
python pixhawk_tools/setup_params.py --reset-default    # đặt FORMAT_VERSION = 0 rồi reboot
```

Làm lại theo **đúng thứ tự này**, bỏ bước nào là không arm được:

1. `FRAME_CLASS` và `FRAME_TYPE` — sau reset `FRAME_CLASS = 0` nghĩa là *chưa khai
   báo khung*, ArduCopter từ chối arm với thông báo "Check FRAME_CLASS".
   Quadcopter chữ X: `FRAME_CLASS = 1`, `FRAME_TYPE = 1`.
   (Đây cũng chính là lỗi làm SITL không arm được.)
2. Accel calibration (6 mặt)
3. Compass calibration
4. RC calibration
5. Battery monitor
6. ESC calibration
7. Rồi mới đến `--profile althold` và `--arming`

---

## 6. Param cho hệ cử chỉ (TELEM2 → Pi 5)

Đấu dây và trình tự chạy: [`RUNBOOK_ROS2.md`](RUNBOOK_ROS2.md) bước 6.

| Param | Đặt | Vì sao |
|:--|--:|:--|
| `SERIAL2_PROTOCOL` | 2 | MAVLink 2 trên TELEM2 |
| `SERIAL2_BAUD` | 921 | = 921600 |
| `BRD_SER2_RTSCTS` | 0 | không đấu RTS/CTS thì phải tắt flow control |
| `SR2_EXTRA1`… | — | nếu thiếu `EXTENDED_SYS_STATE`, `command_node` tự xin bằng `MAV_CMD_SET_MESSAGE_INTERVAL` |

Giá trị đề xuất cho 6X (kênh 5 mode, `RC6_OPTION 31` E-stop, `FS_THR_ENABLE`,
`ARMING_RUDDER 0`, `SCR_ENABLE 1`) nằm trong `sim/params/sitl_6x_proposal.parm`,
**chỉ ghi xuống board sau khi qua cổng 5b** của RUNBOOK và anh đồng ý.

---

## 7. Bài học đắt nhất

**Mô phỏng không bắt được lỗi phần cứng.** Board 6C hỏng (IMU2 chết, IOMCU lỗi, OUT 4
chết) chỉ lộ ra qua **log bay thật** và MAVLink Inspector, không bao giờ lộ qua SITL.
Công cụ đọc log: [`pixhawk_tools/log_health.py`](../pixhawk_tools/log_health.py).
