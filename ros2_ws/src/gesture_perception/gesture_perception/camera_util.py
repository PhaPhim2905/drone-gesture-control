"""
CAMERA_UTIL — mở camera trên Windows cho đúng, gọn về một chỗ.

VÌ SAO CẦN FILE NÀY

Trên Windows, OpenCV có hai đường vào camera và CHÚNG ĐÁNH SỐ KHÁC NHAU. Đo
được trên chính máy này, khi cắm thêm một webcam USB:

    CAP_DSHOW   thay index 0    khong thay index 1
    CAP_MSMF    thay index 1    khong thay index 0

Hai camera có thật (Windows liệt kê đủ cả hai), nhưng mỗi backend chỉ với tới
được một cái. Code khoá cứng một backend thì nửa số camera trên máy trở thành
vô hình, và người dùng nhận được đúng một câu "khong mo duoc camera" mà không
có manh mối gì.

Vì vậy mặc định là "auto": thử lần lượt, lấy cái nào mở được VÀ đọc được frame.
Mở được mà không đọc được frame là chuyện thường gặp khi camera đang bị chương
trình khác giữ — nên phép thử phải bao gồm đọc thật một frame.
"""

import sys

import cv2

# Tên -> hằng số backend của OpenCV. "auto" xử lý riêng bên dưới.
BACKENDS = {
    "dshow": getattr(cv2, "CAP_DSHOW", 0),   # DirectShow, cu, on dinh
    "msmf": getattr(cv2, "CAP_MSMF", 0),     # Media Foundation, mac dinh cua Win10+
    "v4l2": getattr(cv2, "CAP_V4L2", 0),     # Linux / Pi 5 (Ubuntu)
    "any": cv2.CAP_ANY,                      # de OpenCV tu chon
}

# Thứ tự thử khi để "auto". DSHOW trước vì nó ít giở chứng hơn với webcam rẻ
# tiền; MSMF sau vì đó là thứ thấy được camera mà DSHOW bỏ sót.
AUTO_ORDER = ("dshow", "msmf", "any") if sys.platform == "win32" else ("v4l2", "any")


def _try_open(index, be, width, height):
    cap = cv2.VideoCapture(index, be)
    if not cap.isOpened():
        cap.release()
        return None
    if be == BACKENDS["v4l2"] and be != 0:
        # Webcam USB trên Linux mặc định gửi YUYV không nén: 640x480 qua USB 2.0
        # thường bị giới hạn ~10-15 fps. MJPG nén trong camera, đủ 30 fps.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # Chỉ giữ 1 frame trong bộ đệm: MediaPipe chậm hơn camera, bộ đệm dài thì
    # ta xử lý frame CŨ và người điều khiển thấy trễ tích luỹ.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None            # mo duoc nhung khong doc duoc -> coi nhu that bai
    return cap


def linux_video_devices():
    """
    -> [(index, tên, driver)] của mọi /dev/videoN, đọc từ sysfs.

    VÌ SAO CẦN: trên Pi 5, /dev/video0.. KHÔNG phải webcam mà là khối xử lý ảnh
    nội bộ (rp1-cfe, pispbe, hevc). Đo ngày 2026-09-15: index 0 báo "Not a video
    capture device". Webcam USB có driver "uvcvideo" và nằm ở số lớn hơn, số nào
    thì tuỳ thứ tự nạp driver — nên phải dò, không đoán.
    """
    import glob
    import os
    import re
    out = []
    for p in glob.glob("/sys/class/video4linux/video*"):
        m = re.search(r"(\d+)$", p)
        if not m:
            continue
        try:
            with open(os.path.join(p, "name")) as f:
                name = f.read().strip()
        except OSError:
            name = "?"
        drv = os.path.join(p, "device", "driver")
        driver = os.path.basename(os.path.realpath(drv)) if os.path.exists(drv) else "?"
        out.append((int(m.group(1)), name, driver))
    return sorted(out)


def _candidates(index):
    """index >= 0: đúng số đó. index < 0: tự dò, webcam USB (uvcvideo) trước."""
    if index >= 0:
        return [index]
    if sys.platform.startswith("linux"):
        devs = linux_video_devices()
        uvc = [i for i, _, d in devs if d == "uvcvideo"]
        return uvc or [i for i, _, _ in devs]
    return list(range(4))


def open_camera(index, backend="auto", width=None, height=None):
    """
    -> (VideoCapture, ten_backend)  hoặc  (None, None)

    index: số camera, hoặc -1 để tự dò (Pi: webcam USB đầu tiên đọc được frame)
    backend: "auto" | "dshow" | "msmf" | "v4l2" | "any"

    Index thực tế mở được ghi vào cap.index_used.
    """
    names = AUTO_ORDER if backend == "auto" else (backend,)
    for i in _candidates(index):
        for nm in names:
            be = BACKENDS.get(nm, cv2.CAP_ANY)
            cap = _try_open(i, be, width, height)
            if cap is not None:
                try:
                    cap.index_used = i
                except AttributeError:
                    pass
                return cap, f"{nm} index {i}"
    return None, None


def probe(max_index=4, width=None, height=None):
    """Dò mọi (backend, index) mở được. -> danh sách (ten_backend, index, w, h)."""
    found = []
    if sys.platform.startswith("linux"):
        devs = linux_video_devices()
        for i, name, drv in devs:
            print(f"  /dev/video{i:<3} {drv:<10} {name}")
        indices = [i for i, _, _ in devs]
    else:
        indices = range(max_index)
    for nm in AUTO_ORDER:
        be = BACKENDS.get(nm, cv2.CAP_ANY)
        for i in indices:
            cap = _try_open(i, be, width, height)
            if cap is None:
                continue
            ok, f = cap.read()
            if ok and f is not None:
                found.append((nm, i, f.shape[1], f.shape[0]))
            cap.release()
    return found


def fail_message(index, backend):
    """Thông báo lỗi có ích, thay vì một dòng cụt lủn."""
    return (
        f"\n  KHONG MO DUOC camera index {index} (backend {backend}).\n\n"
        "  Do xem may co nhung camera nao:\n"
        "      ros2 run gesture_perception doctor --probe\n"
        "      (Pi: v4l2-ctl --list-devices)\n\n"
        "  Ly do hay gap:\n"
        "    - Pi 5: /dev/video0.. la bo xu ly anh noi bo, KHONG phai webcam.\n"
        "      De index -1 (tu do) hoac lay so tu v4l2-ctl --list-devices.\n"
        "    - webcam chua cam / cam qua hub thieu dien. Cam thang vao Pi.\n"
        "    - sai index. Cam tich hop thuong la 0, cam USB thuong la 1.\n"
        "    - sai backend. Tren Windows, DSHOW va MSMF danh so KHAC NHAU;\n"
        "      camera MSMF thay index 1 thi DSHOW khong thay index 1.\n"
        "    - camera dang bi chuong trinh khac giu (Teams, Zoom, Camera app).\n"
        "      Dong het roi thu lai.\n"
    )
