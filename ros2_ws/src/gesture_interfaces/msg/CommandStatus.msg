# Trạng thái tầng lệnh, do command_node phát lên /drone/gesture_command mỗi nhịp.
# Topic DUY NHẤT cần xem để biết drone đang được bảo làm gì và vì sao.

std_msgs/Header header

# LOCKED   chưa có quyền điều khiển (chờ ASSUME_GUIDANCE)
# ENABLED  đang nghe cử chỉ
string state

string candidate          # cử chỉ đang giữ, chưa đủ thời gian
float32 hold_progress     # 0..1 tiến độ giữ của candidate
string motion             # lệnh chuyển động đang áp: HOVER / LEFT / RIGHT

# Vận tốc ĐANG gửi, hệ thân máy bay FLU (x tiến, y TRÁI, z lên), m/s.
# Chỉ có nghĩa khi streaming = true.
geometry_msgs/Vector3 velocity_flu
bool streaming

# Vì sao không gửi setpoint. Rỗng = đang gửi.
string blocked_reason

# Sự kiện gần nhất (ENABLE, TAKEOFF, LAND, khoá vì mất người...), để HUD hiện.
string last_event

bool fcu_connected
bool armed
string mode
uint8 landed_state        # mavros_msgs/ExtendedState: 0 ?, 1 đất, 2 bay, 3 cất, 4 hạ
