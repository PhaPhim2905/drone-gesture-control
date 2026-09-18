"""
PCK_FORMAT — cầu nối MediaPipe Pose  <->  định dạng BODY25 của Pose-Classification-Kit.

VÌ SAO CÓ FILE NÀY

PCK sinh dữ liệu bằng OpenPose (25 điểm, BODY25). Ta chạy MediaPipe Pose (33
điểm). Muốn dùng chung dataset, chung model, chung phép chuẩn hoá thì phải quy
về một định dạng. File này làm đúng việc đó, bằng numpy thuần:

    MediaPipe 33 điểm  ->  BODY25  ->  chuẩn hoá PCK  ->  vector 50 chiều

KHÔNG import tensorflow, KHÔNG import pose_classification_kit. Chạy được ở môi
trường chính (numpy 2.2.6 / mediapipe 0.10.14) lẫn ở venv train.

Đây là điểm mấu chốt của việc "đồng bộ thư viện": tầng huấn luyện và tầng suy
luận chỉ chia sẻ MỘT thứ duy nhất là định dạng vector 50 chiều này. Không
package nào phải nói chuyện với package nào.

──────────────────────────────────────────────────────────────────────────────
PHÉP CHUẨN HOÁ — theo MÃ NGUỒN, KHÔNG theo slide

Slide 22 của báo cáo PCK viết:

    f(x) = (x - <x>) / max_{j} (delta_j * gamma_j)          <- MAX, tâm = trung bình

Mã nguồn openpose_thread.getBodyData() lại làm:

    f(x) = (x - tâm_bbox) / mean(delta_j * gamma_j)         <- MEAN, tâm = bbox

Hai công thức KHÁC NHAU. Đã kiểm bằng chính BodyPose_Dataset.csv họ ship:

    tâm bbox      -> |lệch| trung vị = 0.00000   <- khớp
    tâm trung bình-> |lệch| trung vị = 0.04257

    nếu là MAX thì mọi chi sau chuẩn hoá phải <= ratio/16. Thực tế:
    thân vượt trần 71.4% mẫu, đùi phải vượt 82.2%  -> KHÔNG THỂ là MAX.

=> Slide viết gọn/sai. Mã nguồn mới là thứ đã tạo ra dữ liệu. File này theo
   mã nguồn, nếu không thì mọi mẫu ta sinh ra sẽ lệch phân bố so với dataset
   của họ và model sẽ học nhầm.
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
#  BODY25 — định dạng của OpenPose, cũng là định dạng dataset PCK
# ══════════════════════════════════════════════════════════════════════════

BODY25_MAPPING = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "mid_hip",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
    "left_bigtoe", "left_smalltoe", "left_heel",
    "right_bigtoe", "right_smalltoe", "right_heel",
)
N_BODY25 = 25

# Chỉ số cho tiện đọc
B25_NOSE, B25_NECK = 0, 1
B25_RSHOULDER, B25_RELBOW, B25_RWRIST = 2, 3, 4
B25_LSHOULDER, B25_LELBOW, B25_LWRIST = 5, 6, 7
B25_MIDHIP = 8
B25_RHIP, B25_RKNEE, B25_RANKLE = 9, 10, 11
B25_LHIP, B25_LKNEE, B25_LANKLE = 12, 13, 14

# Bảy chi dùng để tính hệ số tỷ lệ, kèm hằng số nhân — chép nguyên từ
# openpose_thread.py. Con số 16 là hằng số chuẩn hoá của PCK; tử số là tỷ lệ
# nhân trắc của từng chi so với chiều cao người.
SCALE_LIMBS = (
    (B25_NECK, B25_MIDHIP, 5.2),      # thân
    (B25_NOSE, B25_NECK, 2.5),        # cổ
    (B25_RHIP, B25_RKNEE, 3.6),       # đùi phải
    (B25_RKNEE, B25_RANKLE, 3.5),     # cẳng chân phải
    (B25_LHIP, B25_LKNEE, 3.6),       # đùi trái
    (B25_LKNEE, B25_LANKLE, 3.5),     # cẳng chân trái
    (B25_RSHOULDER, B25_LSHOULDER, 3.4),  # vai
)
SCALE_CONST = 16.0


# ══════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE POSE 33 -> BODY25
# ══════════════════════════════════════════════════════════════════════════
# MediaPipe Pose trả 33 landmark. BODY25 cần 25. Hai điểm BODY25 KHÔNG có sẵn
# trong MediaPipe (neck, mid_hip) nên phải TÍNH bằng trung điểm. Hai điểm
# MediaPipe không có (ngón út chân trái/phải) để 0 — không ảnh hưởng cử chỉ tay.

MP_NOSE = 0
MP_LEYE, MP_REYE = 2, 5
MP_LEAR, MP_REAR = 7, 8
MP_LSHOULDER, MP_RSHOULDER = 11, 12
MP_LELBOW, MP_RELBOW = 13, 14
MP_LWRIST, MP_RWRIST = 15, 16
MP_LHIP, MP_RHIP = 23, 24
MP_LKNEE, MP_RKNEE = 25, 26
MP_LANKLE, MP_RANKLE = 27, 28
MP_LHEEL, MP_RHEEL = 29, 30
MP_LFOOT, MP_RFOOT = 31, 32

# BODY25 index -> MediaPipe index. None = phải tính, hoặc không có.
_DIRECT = {
    B25_NOSE: MP_NOSE,
    B25_RSHOULDER: MP_RSHOULDER, B25_RELBOW: MP_RELBOW, B25_RWRIST: MP_RWRIST,
    B25_LSHOULDER: MP_LSHOULDER, B25_LELBOW: MP_LELBOW, B25_LWRIST: MP_LWRIST,
    B25_RHIP: MP_RHIP, B25_RKNEE: MP_RKNEE, B25_RANKLE: MP_RANKLE,
    B25_LHIP: MP_LHIP, B25_LKNEE: MP_LKNEE, B25_LANKLE: MP_LANKLE,
    15: MP_REYE, 16: MP_LEYE, 17: MP_REAR, 18: MP_LEAR,
    19: MP_LFOOT,   # left_bigtoe   ~ left_foot_index
    21: MP_LHEEL,   # left_heel
    22: MP_RFOOT,   # right_bigtoe  ~ right_foot_index
    24: MP_RHEEL,   # right_heel
}
# 20 = left_smalltoe, 23 = right_smalltoe: MediaPipe không có -> để 0.


def mediapipe_to_body25(landmarks, image_w, image_h, min_visibility=0.5):
    """
    33 landmark MediaPipe -> mảng (25, 3) kiểu BODY25: [x_pixel, y_pixel, conf].

    ⚠️ PHẢI NHÂN VỚI KÍCH THƯỚC ẢNH.

    MediaPipe trả toạ độ chuẩn hoá: x chia cho CHIỀU RỘNG, y chia cho CHIỀU CAO.
    Khung 640x480 thì hai trục bị chia hai số khác nhau -> cơ thể bị BÓP MÉO
    theo tỷ lệ khung hình. OpenPose làm việc trên pixel nên không có chuyện đó.
    Không nhân lại thì mọi tỷ lệ chi đều sai ~33% ở khung 4:3, và model học
    trên dataset PCK sẽ đoán bậy hoàn toàn.

    conf: MediaPipe cho `visibility`, dùng thẳng làm confidence. Landmark dưới
    ngưỡng bị coi như KHÔNG THẤY (đặt 0) — MediaPipe vẫn trả toạ độ cho khớp bị
    che, chỉ hạ visibility, nên không lọc thì khớp khuất sau lưng vẫn sinh ra
    một vị trí "hợp lệ".
    """
    out = np.zeros((N_BODY25, 3), dtype=np.float64)

    def _get(mp_idx):
        lm = landmarks[mp_idx]
        v = getattr(lm, "visibility", 1.0)
        if v < min_visibility:
            return None
        return np.array([lm.x * image_w, lm.y * image_h, v])

    for b25, mp_idx in _DIRECT.items():
        p = _get(mp_idx)
        if p is not None:
            out[b25] = p

    # neck = trung điểm hai vai (MediaPipe không có khớp cổ)
    ls, rs = _get(MP_LSHOULDER), _get(MP_RSHOULDER)
    if ls is not None and rs is not None:
        out[B25_NECK] = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2,
                                  min(ls[2], rs[2])])

    # mid_hip = trung điểm hai hông
    lh, rh = _get(MP_LHIP), _get(MP_RHIP)
    if lh is not None and rh is not None:
        out[B25_MIDHIP] = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2,
                                    min(lh[2], rh[2])])

    return out


# ══════════════════════════════════════════════════════════════════════════
#  CHUẨN HOÁ — tái hiện openpose_thread.getBodyData()
# ══════════════════════════════════════════════════════════════════════════

def _limb_length(kp, a, b):
    """Độ dài chi, hoặc 0 nếu một trong hai đầu không nhìn thấy."""
    if kp[a, 2] > 0.0 and kp[b, 2] > 0.0:
        return float(np.linalg.norm(kp[a, 0:2] - kp[b, 0:2]))
    return 0.0


def normalize_body25(kp, strict=True):
    """
    (25, 3) toạ độ pixel  ->  (25, 2) đã chuẩn hoá, hoặc None nếu không hợp lệ.

    Bốn bước, đúng thứ tự mã nguồn PCK:

      1. bbox các khớp NHÌN THẤY -> tâm
      2. x' = x - tâm_x                 (giữ chiều)
         y' = tâm_y - y                 (LẬT TRỤC: y dương là hướng LÊN)
      3. scale = TRUNG BÌNH các chi có mặt, mỗi chi nhân 16/tỷ_lệ_riêng
      4. chia cả hai trục cho scale; loại mẫu nếu bất kỳ |giá trị| > 1.0

    Khớp không nhìn thấy giữ nguyên (0, 0) — đó là quy ước của PCK, và cũng là
    thứ hàm data augmentation của họ mô phỏng khi "xoá khớp".

    strict=False thì bỏ bước loại ở mục 4 (dùng khi khảo sát dữ liệu).
    """
    kp = np.asarray(kp, dtype=np.float64)
    vis = kp[:, 2] > 0.0
    if vis.sum() < 5:
        return None

    pts = kp[vis, 0:2]
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2.0

    out = np.zeros((N_BODY25, 2), dtype=np.float64)
    out[vis, 0] = kp[vis, 0] - cx
    out[vis, 1] = cy - kp[vis, 1]          # lật trục y

    # Hệ số tỷ lệ: trung bình các chi CÓ MẶT. Lấy trung bình nhiều chi chứ
    # không lấy một chi, vì chiếu 2D co ngắn chi tuỳ hướng 3D của nó —
    # đo trên chính dataset PCK: tỷ lệ tay/vai biến thiên 1.39..2.11 (52%).
    lengths = np.array([
        _limb_length(kp, a, b) * (SCALE_CONST / ratio)
        for a, b, ratio in SCALE_LIMBS
    ])
    lengths = lengths[lengths > 0.0]
    if len(lengths) == 0:
        return None
    scale = float(np.mean(lengths))
    if scale < 1e-9:
        return None

    out /= scale
    if strict and np.any(np.abs(out) > 1.0):
        return None
    return out


def to_flat(kp_norm):
    """(25, 2) -> (50,) theo đúng thứ tự cột CSV: x0,y0,x1,y1,...,x24,y24."""
    return np.asarray(kp_norm, dtype=np.float32).reshape(-1)


def from_flat(vec):
    """(50,) -> (25, 2). Nghịch đảo của to_flat."""
    return np.asarray(vec, dtype=np.float64).reshape(N_BODY25, 2)


# ══════════════════════════════════════════════════════════════════════════
#  TỰ KIỂM — không cần TensorFlow, không cần camera
# ══════════════════════════════════════════════════════════════════════════

def _fake_body25(scale_px=200.0, arms="down"):
    """Dựng một bộ khớp giả có tỷ lệ người thật, để kiểm phép biến đổi."""
    s = scale_px
    kp = np.zeros((N_BODY25, 3))

    def put(i, x, y):
        kp[i] = (500.0 + x * s, 400.0 - y * s, 1.0)

    put(B25_NECK, 0.0, 0.0)
    put(B25_NOSE, 0.0, 0.30)
    put(B25_RSHOULDER, -0.20, 0.0)
    put(B25_LSHOULDER, +0.20, 0.0)
    put(B25_MIDHIP, 0.0, -0.62)
    put(B25_RHIP, -0.10, -0.62); put(B25_LHIP, +0.10, -0.62)
    put(B25_RKNEE, -0.10, -1.05); put(B25_LKNEE, +0.10, -1.05)
    put(B25_RANKLE, -0.10, -1.47); put(B25_LANKLE, +0.10, -1.47)

    if arms == "down":
        put(B25_RELBOW, -0.24, -0.32); put(B25_RWRIST, -0.26, -0.62)
        put(B25_LELBOW, +0.24, -0.32); put(B25_LWRIST, +0.26, -0.62)
    elif arms == "t":
        put(B25_RELBOW, -0.42, 0.0); put(B25_RWRIST, -0.64, 0.0)
        put(B25_LELBOW, +0.42, 0.0); put(B25_LWRIST, +0.64, 0.0)
    elif arms == "up":
        put(B25_RELBOW, -0.22, 0.32); put(B25_RWRIST, -0.20, 0.64)
        put(B25_LELBOW, +0.22, 0.32); put(B25_LWRIST, +0.20, 0.64)
    return kp


def _selftest():
    ok = True
    print("=" * 74)
    print("pck_format — TU KIEM")
    print("=" * 74)

    # --- 1. bat bien ty le: phong to nguoi len 3 lan -> vector KHONG DOI ----
    print("\n  1. Bat bien ty le (nguoi xa/gan camera)")
    ref = None
    for s in (80.0, 200.0, 600.0):
        n = normalize_body25(_fake_body25(s, "t"))
        if n is None:
            print(f"     scale_px={s:<6} -> None  SAI"); ok = False; continue
        if ref is None:
            ref = n
        d = float(np.abs(n - ref).max())
        good = d < 1e-9
        ok &= good
        print(f"     scale_px={s:<6.0f} lech toi da so voi mau dau = {d:.2e}  "
              f"{'OK' if good else 'SAI'}")

    # --- 2. bat bien tinh tien: doi cho nguoi trong khung -----------------
    print("\n  2. Bat bien tinh tien (nguoi dung lech khung hinh)")
    a = _fake_body25(200.0, "t")
    b = a.copy(); b[b[:, 2] > 0, 0] += 137.0; b[b[:, 2] > 0, 1] -= 89.0
    na, nb = normalize_body25(a), normalize_body25(b)
    d = float(np.abs(na - nb).max()); good = d < 1e-9; ok &= good
    print(f"     lech toi da = {d:.2e}  {'OK' if good else 'SAI'}")

    # --- 3. lat truc y: gio tay len phai ra y DUONG ------------------------
    print("\n  3. Lat truc y (tay gio len -> y DUONG trong he PCK)")
    up = normalize_body25(_fake_body25(200.0, "up"))
    dn = normalize_body25(_fake_body25(200.0, "down"))
    good = up[B25_LWRIST, 1] > 0 and dn[B25_LWRIST, 1] < 0
    ok &= good
    print(f"     tay gio len  y = {up[B25_LWRIST,1]:+.3f}  (mong doi > 0)")
    print(f"     tay buong    y = {dn[B25_LWRIST,1]:+.3f}  (mong doi < 0)  "
          f"{'OK' if good else 'SAI'}")

    # --- 4. doi chieu voi so do that tren dataset PCK ---------------------
    print("\n  4. Do dai chi sau chuan hoa, doi chieu dataset PCK that")
    n = normalize_body25(_fake_body25(200.0, "t"))
    kp = _fake_body25(200.0, "t")
    for nm, a_, b_, ratio, real in (("than", B25_NECK, B25_MIDHIP, 5.2, 0.3354),
                                    ("vai", B25_RSHOULDER, B25_LSHOULDER, 3.4, 0.1802)):
        got = float(np.linalg.norm(n[a_] - n[b_]))
        print(f"     {nm:<6} tinh duoc = {got:.4f}   ty le PCK khai bao = "
              f"{ratio/16:.4f}   do tren dataset that = {real:.4f}")

    # --- 5. to_flat / from_flat khu hoi ------------------------------------
    print("\n  5. to_flat <-> from_flat")
    f = to_flat(n)
    good = f.shape == (50,) and np.abs(from_flat(f) - n).max() < 1e-6
    ok &= good
    print(f"     shape {f.shape}, khu hoi {'OK' if good else 'SAI'}")

    # --- 6. anh khong vuong: bay ty le khung hinh --------------------------
    print("\n  6. Bay ty le khung hinh (vi sao phai nhan voi w,h)")
    kp_px = _fake_body25(200.0, "t")
    kp_norm_wrong = kp_px.copy()
    kp_norm_wrong[:, 0] /= 640.0      # gia lam toa do chuan hoa MediaPipe
    kp_norm_wrong[:, 1] /= 480.0      # ... roi KHONG nhan lai
    n_wrong = normalize_body25(kp_norm_wrong, strict=False)
    n_right = normalize_body25(kp_px)
    err = float(np.abs(n_wrong - n_right).max())
    good = err > 0.05
    ok &= good
    print(f"     lech toi da neu QUEN nhan (w,h) = {err:.3f}")
    print(f"     -> sai lech dang ke, dung nhu canh bao. {'OK' if good else 'SAI'}")

    print("\n" + "=" * 74)
    print("KET QUA: " + ("TAT CA DAT" if ok else "CO TEST SAI"))
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
