"""
SETTINGS — tham số THU HÌNH dùng chung cho HAI nơi: quay dữ liệu (training/) và
chạy thật (perception_node).

Vì sao phải chung một file: model học trên khớp xương sinh ra với ĐÚNG bộ tham số
này. Quay dữ liệu với FLIP_FRAME = True mà chạy thật với False là tay trái thành
tay phải, ROLL_LEFT thành ROLL_RIGHT, và drone bay ngược hướng — không lỗi, không
log, chỉ sai.

    training/1_record.py        import settings as C
    perception_node             đọc làm giá trị MẶC ĐỊNH của tham số ROS

config/perception.yaml có thể ghi đè lúc chạy. Đổi ở yaml mà KHÔNG đổi ở đây là
tự tạo lệch giữa dữ liệu train và lúc bay — chỉ làm khi đo thử.

Chắt từ src/config_v3.py mục 1-2 (đã xoá 2026-09-14). Phần hình học cử chỉ bằng
ngưỡng tay (mục 3, 6) không dùng nữa vì model TFLite đã thay; phần One-Euro nằm
trong one_euro.py; phần thời gian giữ và bản đồ lệnh chuyển sang gesture_command.
"""

import os

# --- MediaPipe Pose ---------------------------------------------------------
POSE_MODEL_COMPLEXITY = int(os.environ.get("POSE_COMPLEXITY", "0"))
# 0 = Lite. Đòn bẩy tốc độ lớn nhất trên Pi 5. Resize ảnh KHÔNG giúp: đã đo,
# MediaPipe tự resize nội bộ về kích thước cố định của model.

POSE_DETECT_CONF = 0.6
POSE_TRACK_CONF = 0.5
POSE_SMOOTH_LANDMARKS = False
# TẮT bộ làm mượt nội bộ của MediaPipe: không chỉnh được tham số, không biết nó
# đo dt thế nào. Hai tầng lọc chồng nhau thì trễ cộng dồn không ai kiểm soát.
# Ta tự lọc bằng One-Euro, nơi đo được và chỉnh được.

LM_MIN_VISIBILITY = 0.5
# MediaPipe vẫn trả toạ độ cho khớp bị che, chỉ hạ visibility. Không lọc thì cổ
# tay khuất sau lưng vẫn sinh ra một vị trí "hợp lệ" và tạo lệnh ma.

# --- Camera -----------------------------------------------------------------
CAM_INDEX = int(os.environ.get("CAM_INDEX", "0"))
CAM_BACKEND = os.environ.get("CAM_BACKEND", "auto")
CAM_WIDTH = 640
CAM_HEIGHT = 480

FLIP_FRAME = True
# Soi gương. Dữ liệu v3 đã quay với True. ĐỔI LÀ PHẢI QUAY LẠI TOÀN BỘ DỮ LIỆU.

# --- Nhãn đặc biệt, không phải lớp của model --------------------------------
UNKNOWN_GESTURE = "AMBIGUOUS"     # thấy người nhưng độ tin cậy dưới ngưỡng tau
NO_OPERATOR = "NO_OPERATOR"       # không thấy người
