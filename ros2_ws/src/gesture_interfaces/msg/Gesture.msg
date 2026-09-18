# Kết quả nhận diện của MỘT frame, do perception_node phát lên /gesture/detected.
#
# Đây là quan sát THÔ theo từng frame — CHƯA qua thời gian giữ. Việc "giữ đủ lâu
# mới thành lệnh" nằm ở command_node, để perception không cần biết gì về drone.

std_msgs/Header header

# Tên lớp của model (HOVER, ROLL_LEFT, ...) hoặc một trong hai nhãn đặc biệt:
#   AMBIGUOUS    thấy người nhưng độ tin cậy dưới ngưỡng tau
#   NO_OPERATOR  không thấy người
string label

# Độ tin cậy của lớp cao nhất, SAU hiệu chuẩn nhiệt độ T. 0 khi NO_OPERATOR.
float32 confidence

# Ba lớp cao nhất, để hiện lên màn hình / rqt khi dò lỗi.
string[] top_labels
float32[] top_probs

# Vector 50 chiều đã chuẩn hoá PCK (x0,y0,...,x24,y24). Rỗng nếu tắt
# publish_keypoints hoặc không thấy người.
float32[] keypoints_norm

# Đo hiệu năng tại chỗ.
float32 fps
float32 pose_ms
