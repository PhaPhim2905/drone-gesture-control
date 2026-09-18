#!/usr/bin/env python3
"""
TRAIN_PCK — huấn luyện bộ phân loại tư thế theo kiến trúc Pose-Classification-Kit.

    # kiểm dữ liệu TRƯỚC khi train (chạy được ở MÔI TRƯỜNG CHÍNH, không cần TF)
    python training/4_train.py --npz data/dataset_v3.npz --check

    # train thật (VENV RIÊNG)
    .venv-train/Scripts/python.exe training/4_train.py --npz data/dataset_v3.npz

Vì sao venv riêng: TensorFlow kéo theo cả cây phụ thuộc riêng. Môi trường
chính đang có numpy 2.2.6 + mediapipe 0.10.14 + opencv 4.13 chạy tốt, không có
lý do gì để mạo hiểm. Hai môi trường chỉ chia sẻ ĐÚNG MỘT thứ:

    file .tflite  +  vector 50 chiều theo định dạng src/pck_format.py

──────────────────────────────────────────────────────────────────────────────
BỘ SÁU CỬ CHỈ — ROSTER "v1"

    ASSUME_GUIDANCE   hai tay giơ THẲNG lên, dang bằng vai (chữ V)   -> mở quyền
    HOVER             hai tay dang NGANG, duỗi hết biên (chữ T)      -> giữ chỗ
    TAKEOFF           dang ngang, GẬP KHUỶU 90 độ (chữ U)            -> cất cánh
    LAND              hai tay bắt CHÉO trước BỤNG                    -> hạ cánh
    STOP              hai tay bắt CHÉO trên ĐẦU                      -> dừng khẩn
    NEGATIVE          mọi thứ khác                                   -> không lệnh

Bốn lớp hướng bay (MOVE_UP/DOWN/LEFT/RIGHT) để giai đoạn sau: chúng nằm trên
cùng một cung, chỉ cách nhau 45 độ, nên cần dữ liệu dày hơn nhiều.

NEGATIVE KHÔNG PHẢI LỚP PHỤ. Softmax luôn cộng lại bằng 1, nên không có nó thì
mọi tư thế lạ đều bị ép vào một trong năm lệnh. Đã đo trên chính hệ này: entropy
của tư thế thật 0.553, của nhiễu ngẫu nhiên 0.183 — mạng TỰ TIN HƠN khi gặp thứ
nó chưa từng thấy. Đó là lý do bằng số cho việc bắt buộc có NEGATIVE, và cho
ngưỡng từ chối ở mục dưới.

──────────────────────────────────────────────────────────────────────────────
BỐN THỨ BẢN NÀY LÀM KHÁC BẢN CŨ

1. CHẶN TRƯỚC KHI TRAIN (mục KIỂM ĐỊNH). Thiếu lớp, thiếu take, lệch lớp, dữ
   liệu chưa chuẩn hoá, hai take trùng điều kiện — dừng ngay và nói phải thu gì.
   Muốn bỏ qua thì --force, nhưng khi đó TÊN FILE MODEL mang hậu tố _UNSAFE để
   chính nó tự khai ra, không dựa vào trí nhớ người dùng.

2. KIỂM CHỨNG CHÉO THEO TAKE (K-fold) thay vì một lần chia. Với 5-8 take mỗi
   lớp, một tập test 1 take là một con số may rủi. K-fold cho MỌI take được làm
   test đúng một lần -> ma trận nhầm lẫn phủ toàn bộ dữ liệu, kèm độ lệch chuẩn
   giữa các fold. Lệch chuẩn lớn nghĩa là con số trung bình không đáng tin.

3. NGƯỠNG TỪ CHỐI + HIỆU CHUẨN NHIỆT ĐỘ. Mô hình phải được phép nói "không
   biết". Ngưỡng tau chọn trên dự đoán out-of-fold sao cho ĐỘ CHÍNH XÁC CỦA LỆNH
   ĐÃ PHÁT đạt mục tiêu (mặc định 99%), rồi ghi vào labels.json để tầng chạy
   thật dùng lại. Nhiệt độ T hiệu chuẩn xác suất trước khi so ngưỡng.

4. ĐO THEO LUẬT PHÁT LỆNH THẬT, không chỉ đo theo frame. Hệ thật đòi k frame
   liên tiếp cùng nhãn (hold_sec trong gesture_bringup/config/command_*.yaml). Số đáng quan tâm không phải
   "accuracy" mà là LỆNH SAI MỖI PHÚT — vì một lệnh sai là drone làm sai.

──────────────────────────────────────────────────────────────────────────────
KIẾN TRÚC — lấy từ slide 25 của báo cáo PCK

    Hybrid:  2D Keypoints -> 1CBR(k=3,16) -> 1CBR(k=3,16)
                          -> DBR(128) -> DBR(128) -> Dense(n_class, softmax)

VÌ SAO CHỌN HYBRID chứ không phải Light CNN (nhỏ hơn, nhanh hơn) — slide 26/27:

                        tập test GỐC   tập test ĐÃ NHIỄU LOẠN
        Light CNN          98.25%          50.95%     <- sụp một nửa
        Hybrid             98.30%          95.05%     <- bền

Người đứng xa, chân ra ngoài khung, gió rung camera — đó là cột phải. Với thiết
bị bay thì cột phải mới là cột thật. Cả hai đều dưới 1 ms trên CPU.

Mặc định vẫn là --arch pck (kiến trúc đọc thẳng từ file .h5 của PCK, KHÔNG có
BatchNorm) cho tới khi tập tự thu đủ dày. Lý do trong docstring build_hybrid.

──────────────────────────────────────────────────────────────────────────────
TĂNG CƯỜNG DỮ LIỆU — bảng slide 23, chép nguyên, CỘNG THÊM phép soi gương

    tỷ lệ   sigma_scale  sigma_rot  sigma_noise   xoá khớp
     10%       0.08         0.0        0.0        không
     10%       0.0         10.0        0.0        không
     15%       0.0          0.0        0.03       hai chân
     15%       0.0          0.0        0.03       hai chân + hông
     20%       0.0          0.0        0.03       2 khớp ngẫu nhiên

"Xoá hai chân" đặc biệt hợp với ta: camera gắn trên drone, người đứng gần thì
chân ra ngoài khung là chuyện thường.

SOI GƯƠNG (mới): lật x rồi hoán đổi chỉ số trái/phải. Sáu lớp của roster v1 đều
đối xứng nên nhãn giữ nguyên, và nó vá đúng một thói quen của người thu: LAND và
STOP lúc nào cũng bắt chéo cùng một tay ở trên. Không có phép này thì mạng học
luôn cả thói quen đó. MOVE_LEFT/MOVE_RIGHT thì PHẢI đổi nhãn cho nhau — bảng
MIRROR_LABEL giữ việc đó, và lớp nào không có trong bảng thì từ chối soi gương
chứ không đoán.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# pck_format, settings, camera_util nam trong goi ROS gesture_perception -
# MOT ban duy nhat cho ca train lan luc bay.
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "gesture_perception" / "gesture_perception"))

N_KP = 25
N_COORD = 2

# Chỉ số BODY25 của chân và hông — dùng cho phép xoá khớp có chủ đích.
IDX_LEGS = (10, 11, 13, 14, 19, 20, 21, 22, 23, 24)
IDX_HIPS = (8, 9, 12)


# ══════════════════════════════════════════════════════════════════════════
#  ROSTER — bộ lớp mà model NÀY cam kết nhận
# ══════════════════════════════════════════════════════════════════════════
# Đây là một HỢP ĐỒNG, không phải gợi ý. Tập dữ liệu phải khớp đúng danh sách
# này: thiếu một lớp thì lệnh đó không bao giờ ra được; thừa một lớp thì model
# xuất ra một nhãn mà tầng quyết định (gesture_command (bang GESTURE_COMMAND trong command_logic.py)) không biết
# dịch, và nó rơi im lặng về HOVER.

ROSTERS = {
    "v1": ["ASSUME_GUIDANCE", "HOVER", "TAKEOFF", "LAND", "STOP", "NEGATIVE"],
    # v2 (2026-09-15): thêm hai cử chỉ chữ L. Thay bản v2 cũ MOVE_* chưa từng
    # được quay. Phải khớp gesture_command_map() trong command_logic.py.
    "v2": ["ASSUME_GUIDANCE", "HOVER", "TAKEOFF", "LAND", "STOP", "NEGATIVE",
           "ROLL_RIGHT", "ROLL_LEFT"],
}
DEFAULT_ROSTER = "v2"

# Cặp dễ nhầm — hai cử chỉ chỉ khác nhau ở MỘT đại lượng hình học. Ma trận nhầm
# lẫn chung dễ che mất chúng (5 lần nhầm trên 1200 mẫu trông như bụi), nhưng
# chính chúng mới là chỗ hệ gãy khi bay. Báo cáo riêng.
CAP_DE_NHAM = [
    ("TAKEOFF", "HOVER", "cang tay DUNG hay NGANG"),
    ("TAKEOFF", "ASSUME_GUIDANCE", "khuyu GAP hay tay THANG"),
    ("LAND", "STOP", "cheo DUOI BUNG hay TREN DAU"),
]

# Lệnh hệ trọng: nhầm sang đây tốn kém hơn hẳn nhầm giữa hai lệnh bay.
LOP_HE_TRONG = ("LAND", "STOP", "ASSUME_GUIDANCE")


# ══════════════════════════════════════════════════════════════════════════
#  SOI GƯƠNG
# ══════════════════════════════════════════════════════════════════════════
# BODY25: 0 nose, 1 neck, 2-4 vai/khuỷu/cổ tay PHẢI, 5-7 TRÁI, 8 midhip,
# 9-11 hông/gối/cổ chân PHẢI, 12-14 TRÁI, 15-18 mắt/tai (P,T,P,T),
# 19-21 ngón/gót TRÁI, 22-24 PHẢI.
MIRROR_PAIRS = ((2, 5), (3, 6), (4, 7), (9, 12), (10, 13), (11, 14),
                (15, 16), (17, 18), (19, 22), (20, 23), (21, 24))

# Lớp nào soi gương ra lớp nào. Lớp KHÔNG có tên ở đây thì không được soi —
# thà bỏ phép tăng cường còn hơn sinh ra mẫu gắn sai nhãn.
MIRROR_LABEL = {
    "ASSUME_GUIDANCE": "ASSUME_GUIDANCE",
    "HOVER": "HOVER",
    "TAKEOFF": "TAKEOFF",
    "LAND": "LAND",
    "STOP": "STOP",
    "NEGATIVE": "NEGATIVE",
    "MOVE_UP": "MOVE_UP",
    "MOVE_DOWN": "MOVE_DOWN",
    "MOVE_LEFT": "MOVE_RIGHT",     # <- đổi nhãn, đây là chỗ dễ sai nhất
    "MOVE_RIGHT": "MOVE_LEFT",
    "ROLL_LEFT": "ROLL_RIGHT",     # <- cũng đổi nhãn. Hai lớp này là ảnh gương
    "ROLL_RIGHT": "ROLL_LEFT",     #    của nhau, soi gương mà giữ nguyên nhãn
                                   #    là dạy mạng rằng trái = phải.
}


def mirror(X, y_idx, labels):
    """
    Lật ngang. -> (X_mirror, y_mirror) hoặc (rỗng, rỗng) nếu có lớp không lật được.

    Toạ độ đã chuẩn hoá quanh tâm bbox nên đổi dấu x là đủ; đổi dấu xong phải
    hoán đổi chỉ số trái/phải, nếu không thì "vai phải" nằm ở nửa trái người và
    bộ xương thành vặn xoắn.
    """
    thieu = [l for l in labels if l not in MIRROR_LABEL]
    if thieu:
        print(f"  BO soi guong: khong biet lat nhan {', '.join(thieu)}")
        return (np.empty((0, N_KP, N_COORD), np.float32),
                np.empty((0,), np.int32))

    M = X.copy()
    M[:, :, 0] *= -1.0
    for a, b in MIRROR_PAIRS:
        M[:, [a, b]] = M[:, [b, a]]

    lut = {l: i for i, l in enumerate(labels)}
    doi = np.array([lut[MIRROR_LABEL[l]] for l in labels], dtype=np.int32)
    return M.astype(np.float32), doi[y_idx]


# ══════════════════════════════════════════════════════════════════════════
#  NẠP DỮ LIỆU
# ══════════════════════════════════════════════════════════════════════════

def find_pck_wheel():
    """Tìm file .whl của PCK đã tải về, để lấy dataset mà không cần cài."""
    for pat in ("**/pose_classification_kit-*.whl",):
        for base in (ROOT, Path(os.environ.get("TEMP", "/tmp"))):
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0]
    return None


def load_pck_dataset(csv_path=None, wheel=None, min_accuracy=10.0):
    """
    -> X (n, 25, 2), y (n,) nhãn chuỗi, labels (danh sách đã sắp xếp)

    min_accuracy: PCK lọc mẫu theo tổng độ tin cậy OpenPose. File JSON dataset
    của họ ghi threshold_value = 10.0, nên dùng lại số đó.
    """
    if csv_path and Path(csv_path).exists():
        txt = Path(csv_path).read_text(encoding="utf8")
    else:
        if wheel is None:
            wheel = find_pck_wheel()
        if wheel is None:
            raise SystemExit(
                "Khong tim thay dataset. Tai ve bang:\n"
                "  pip download pose-classification-kit==1.1.5 --no-deps -d <thu_muc>"
            )
        with zipfile.ZipFile(wheel) as z:
            txt = z.read(
                "pose_classification_kit/datasets/BodyPose_Dataset.csv"
            ).decode("utf8")

    rows = list(csv.reader(io.StringIO(txt)))[1:]
    keep = [r for r in rows if float(r[1]) >= min_accuracy]
    X = np.array([[float(v) for v in r[2:]] for r in keep],
                 dtype=np.float32).reshape(len(keep), N_KP, N_COORD)
    y = np.array([r[0] for r in keep])
    labels = sorted(set(y.tolist()))
    print(f"  Nap {len(rows)} mau, giu {len(keep)} sau loc accuracy >= {min_accuracy}")
    print(f"  {len(labels)} lop")
    return X, y, labels


def load_npz_dataset(path):
    """
    Nạp tập TỰ THU (data/dataset_v3.npz do build_dataset.py sinh ra).

    -> X (n,25,2), y (n,) chuỗi, labels, groups (n,) mã take, t (n,) thời điểm

    `groups` là thứ tập của PCK KHÔNG có và là thứ quan trọng nhất ở đây: nó cho
    phép chia train/test theo TAKE thay vì theo mẫu. Xem record_dataset.py —
    chia theo mẫu thì hai frame liền nhau của cùng một lần đứng rơi vào hai bên,
    và accuracy đo được là accuracy của trí nhớ.

    `t` có thể vắng (tập build bằng bản build_dataset cũ) -> trả None, và phép
    quy đổi "lệnh sai mỗi phút" sẽ dùng --fps thay vì đo từ dữ liệu.
    """
    # Cảnh báo tập CŨ. Đã dính đúng lỗi này: xoá file jsonl đi thu lại, nhưng
    # dataset_v3.npz cũ vẫn nằm nguyên, build_dataset báo "khong co file nao"
    # rồi thoát mà KHÔNG ghi đè, và train chạy ngon lành trên dữ liệu đã chết.
    npz = Path(path)
    raw = (sorted((ROOT / "data" / "gestures").glob("*.jsonl"))
           if npz.resolve() == (ROOT / "data" / "dataset_v3.npz").resolve()
           else None)
    if raw is None:
        pass
    elif raw:
        newest = max(p.stat().st_mtime for p in raw)
        if newest > npz.stat().st_mtime:
            print("  CANH BAO: co file .jsonl MOI HON dataset_v3.npz.")
            print("            Chay lai:  python training\\3_build_dataset.py")
    elif npz.exists():
        print("  CANH BAO: data/gestures/ RONG nhung dataset_v3.npz van con.")
        print("            Tap nay dung tu du lieu DA BI XOA.")

    d = np.load(path, allow_pickle=False)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(str)
    labels = sorted(set(y.tolist()))
    groups = d["groups"].astype(str)
    t = d["t"].astype(np.float64) if "t" in d.files else None
    print(f"  Nap {len(X)} mau tu {path}")
    print(f"  {len(labels)} lop, {len(set(groups.tolist()))} take"
          f"{'' if t is not None else '   (khong co moc thoi gian)'}")
    return X, y, labels, groups, t


def van_tay(X, y, groups):
    """Mã 12 ký tự nhận dạng ĐÚNG tập dữ liệu này, ghi kèm model.

    Để sau này nhìn labels.json là biết model sinh ra từ tập nào, không phải
    dựa vào ngày giờ file hay trí nhớ.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    h.update("|".join(y.tolist()).encode())
    h.update("|".join(groups.tolist()).encode())
    return h.hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════
#  KIỂM ĐỊNH — chạy TRƯỚC khi nạp TensorFlow
# ══════════════════════════════════════════════════════════════════════════
# Chia làm hai mức, và ranh giới giữa chúng có ý nghĩa:
#
#   CHAN  = train xong sẽ ra một model KHÔNG DÙNG ĐƯỢC hoặc một con số KHÔNG ĐO
#           ĐƯỢC GÌ. Dừng lại rẻ hơn nhiều so với phát hiện sau khi đã tin nó.
#   LUU Y = dùng được nhưng yếu ở một chỗ cụ thể, và ta phải biết chỗ đó ở đâu.
#
# Cả hai đều chạy bằng numpy thuần, nên `--check` gọi được từ môi trường chính.

MIN_TAKE_MOI_LOP = 4        # dưới mức này không chia nổi K-fold
MIN_MAU_MOI_LOP = 150       # dưới mức này lớp đó không đủ để học
MAX_LECH_LOP = 5.0          # lớp đông nhất / lớp thưa nhất
MAX_CHONG_LAN = 0.02        # tỷ lệ mẫu có láng giềng gần nhất KHÁC lớp cử chỉ
TY_LE_NEGATIVE_MIN = 0.12   # NEGATIVE phải chiếm ít nhất bấy nhiêu tập


def kiem_dinh(X, y_str, labels, groups, roster, k_fold):
    """-> (danh sách CHAN, danh sách LUU Y). Không in gì, chỉ trả về."""
    chan, luu_y = [], []
    n = len(X)

    # ---- 1. hợp đồng roster ------------------------------------------------
    if roster:
        thieu = [l for l in roster if l not in labels]
        thua = [l for l in labels if l not in roster]
        if thieu:
            chan.append(f"THIEU LOP: {', '.join(thieu)}  "
                        f"-> lenh tuong ung khong bao gio ra duoc")
        if thua:
            chan.append(f"THUA LOP: {', '.join(thua)}  "
                        f"-> tang quyet dinh khong biet dich nhan nay")

    # ---- 2. NEGATIVE ------------------------------------------------------
    if "NEGATIVE" not in labels:
        chan.append("KHONG CO LOP NEGATIVE. Softmax luon cong bang 1 nen moi "
                    "tu the la deu bi ep thanh mot trong cac lenh.")
    else:
        r = float((y_str == "NEGATIVE").sum()) / max(1, n)
        if r < TY_LE_NEGATIVE_MIN:
            luu_y.append(f"NEGATIVE chi chiem {r*100:.0f}% tap "
                         f"(nen >= {TY_LE_NEGATIVE_MIN*100:.0f}%). "
                         f"Thu them tu the trung gian, di lai, cam dien thoai.")

    # ---- 3. take & mẫu mỗi lớp --------------------------------------------
    for lab in labels:
        sel = (y_str == lab)
        nt = len(set(groups[sel].tolist()))
        nm = int(sel.sum())
        if nt < k_fold:
            chan.append(f"{lab}: {nt} take, can >= {k_fold} thi moi con danh "
                        f"gia duoc (nen 8)")
        elif nt < 6:
            luu_y.append(f"{lab}: {nt} take. Du chay nhung moi fold chi 1 take "
                         f"test -> ket qua dao dong manh. Nen len 8.")
        if nm < MIN_MAU_MOI_LOP:
            chan.append(f"{lab}: chi {nm} mau, can >= {MIN_MAU_MOI_LOP}")

    # ---- 4. lệch lớp -------------------------------------------------------
    dem = np.array([int((y_str == l).sum()) for l in labels], dtype=float)
    if dem.min() > 0:
        lech = dem.max() / dem.min()
        if lech > MAX_LECH_LOP:
            luu_y.append(f"lech lop {lech:.1f}x ({labels[int(dem.argmax())]} "
                         f"vs {labels[int(dem.argmin())]}). Da bu bang trong so "
                         f"mau, nhung thu them cho lop thua van tot hon.")

    # ---- 5. dữ liệu có thật sự đã chuẩn hoá không --------------------------
    # normalize_body25(strict=True) loại mẫu |giá trị| > 1. Nếu ở đây vẫn còn
    # thì tập được build bằng --no-strict, hoặc bằng một phép chuẩn hoá khác —
    # và model sẽ nhận đầu vào khác hẳn lúc chạy thật.
    vmax = float(np.abs(X).max())
    if vmax > 1.001:
        luu_y.append(f"|gia tri| lon nhat = {vmax:.3f} > 1. Tap nay build bang "
                     f"--no-strict? Luc chay that pck_format loai cac mau do.")

    # ---- 6. hai take TRÙNG ĐIỀU KIỆN --------------------------------------
    # Mục đích của nhiều take là nhiều ĐIỀU KIỆN khác nhau, không phải nhiều
    # lần bấm SPACE ở cùng một chỗ. Hai take đứng cùng vị trí thì K-fold vẫn
    # chạy, vẫn ra số đẹp, mà số đó không nói được gì về vị trí thứ ba.
    F = X.reshape(n, -1)
    for lab in labels:
        sel = np.where(y_str == lab)[0]
        gs = sorted(set(groups[sel].tolist()))
        if len(gs) < 2:
            continue
        tam, toa = [], []
        for g in gs:
            idx = sel[groups[sel] == g]
            c = F[idx].mean(0)
            tam.append(c)
            toa.append(float(np.linalg.norm(F[idx] - c, axis=1).mean()))
        tam = np.stack(tam)
        D = np.linalg.norm(tam[:, None, :] - tam[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        i, j = np.unravel_index(D.argmin(), D.shape)
        rong_trong_take = float(np.mean(toa))
        if D[i, j] < rong_trong_take:
            luu_y.append(
                f"{lab}: take {gs[i]} va {gs[j]} gan nhu TRUNG DIEU KIEN "
                f"(cach {D[i,j]:.3f} < do rong noi bo {rong_trong_take:.3f}). "
                f"Doi vi tri / goc dung roi thu lai mot trong hai.")

    # ---- 7. chồng lấn giữa hai lớp CỬ CHỈ ----------------------------------
    # cử chỉ <-> NEGATIVE là BÌNH THƯỜNG (NEGATIVE phải giáp ranh mọi lớp).
    # cử chỉ <-> cử chỉ là hỏng: hai lệnh khác nhau mà hình giống nhau.
    rng = np.random.default_rng(0)
    idx = rng.choice(n, min(800, n), replace=False)
    S, L = F[idx], y_str[idx]
    D = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    xau = [(L[i], L[nn[i]]) for i in range(len(idx))
           if L[nn[i]] != L[i] and "NEGATIVE" not in (L[i], L[nn[i]])]
    if xau:
        r = len(xau) / len(idx)
        top = {}
        for a, b in xau:
            top[(a, b)] = top.get((a, b), 0) + 1
        goi = ", ".join(f"{a}->{b} ({c})" for (a, b), c in
                        sorted(top.items(), key=lambda kv: -kv[1])[:3])
        msg = f"chong lan cu chi<->cu chi {r*100:.1f}%: {goi}"
        (chan if r > MAX_CHONG_LAN else luu_y).append(
            msg + ("  -> hai lenh khac nhau ma hinh giong nhau, model khong tach "
                   "duoc" if r > MAX_CHONG_LAN else "  (duoi nguong, chap nhan)"))

    return chan, luu_y


def in_bang_lop(y_str, labels, groups):
    print(f"\n  {'lop':<20}{'mau':>7}{'take':>7}{'ty le':>8}")
    print("  " + "-" * 42)
    n = len(y_str)
    for lab in labels:
        sel = (y_str == lab)
        print(f"  {lab:<20}{int(sel.sum()):>7}"
              f"{len(set(groups[sel].tolist())):>7}"
              f"{sel.sum()/max(1,n)*100:>7.1f}%")


# ══════════════════════════════════════════════════════════════════════════
#  CHIA THEO TAKE
# ══════════════════════════════════════════════════════════════════════════

def folds_by_take(y, groups, labels, k, seed=0):
    """
    K-fold nhóm theo TAKE, cân THEO TỪNG LỚP. -> [(tr, va, te), ...] chỉ số.

    Vì sao không dùng GroupKFold của sklearn: nó bốc take mù nhãn. Với 5 take
    mỗi lớp, chuyện nó dồn trọn cả 5 take của một lớp vào train là bình thường
    — và lớp đó biến mất khỏi tập test. Đã gặp đúng lỗi này: 9 lớp mà test chỉ
    còn 7, classification_report ném ValueError giữa chừng.

    Cách ở đây: TỪNG LỚP chia take của chính nó vòng tròn vào k rổ. Fold thứ i
    lấy rổ i làm TEST, rổ (i+1)%k làm VAL, phần còn lại làm TRAIN. Mọi lớp có
    mặt ở cả ba tập trong mọi fold, và mỗi take được làm test ĐÚNG MỘT LẦN —
    nên gộp dự đoán của k fold lại là được dự đoán out-of-fold cho TOÀN BỘ tập.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    groups = np.asarray(groups)
    ro = [[] for _ in range(k)]          # ro[i] = danh sách take của rổ i

    for cls in range(len(labels)):
        sel = np.where(y == cls)[0]
        gs = list(dict.fromkeys(groups[sel].tolist()))
        rng.shuffle(gs)
        for j, g in enumerate(gs):
            ro[j % k].append(g)

    folds = []
    for i in range(k):
        te_g = set(ro[i])
        va_g = set(ro[(i + 1) % k]) - te_g
        tr, va, te = [], [], []
        for idx, g in enumerate(groups):
            (te if g in te_g else va if g in va_g else tr).append(idx)
        tr = np.array(tr, dtype=int)
        va = np.array(va, dtype=int)
        te = np.array(te, dtype=int)
        # Rò rỉ là loại lỗi không bao giờ lộ ra qua con số — nó chỉ làm con số
        # ĐẸP LÊN. Nên khẳng định thẳng ở đây thay vì tin vào vòng lặp trên.
        assert not (set(groups[tr]) & set(groups[te])), "ro ri take train/test"
        assert not (set(groups[va]) & set(groups[te])), "ro ri take val/test"
        folds.append((tr, va, te))
    return folds


# ══════════════════════════════════════════════════════════════════════════
#  TĂNG CƯỜNG DỮ LIỆU — slide 23
# ══════════════════════════════════════════════════════════════════════════

def _augment_once(X, y, ratio, sigma_scale, sigma_rot, sigma_noise,
                  remove_idx, remove_rand, rng):
    n = int(len(X) * ratio)
    if n <= 0:
        return np.empty((0, N_KP, N_COORD), np.float32), np.empty((0,), y.dtype)
    pick = rng.choice(len(X), n, replace=False)
    A = X[pick].copy()

    if sigma_scale > 0:
        f = 1.0 + rng.normal(0.0, sigma_scale, (n, 1, 1))
        A *= f

    if sigma_rot > 0:
        ang = np.deg2rad(rng.normal(0.0, sigma_rot, n))
        c, s = np.cos(ang), np.sin(ang)
        R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (n,2,2)
        A = np.einsum("nij,nkj->nki", R, A)

    if sigma_noise > 0:
        # chỉ thêm nhiễu vào khớp CÓ MẶT — khớp vắng phải giữ đúng (0,0),
        # vì đó là mã hiệu "không thấy" chứ không phải một toạ độ
        mask = (np.abs(A).sum(-1, keepdims=True) > 1e-9)
        A += rng.normal(0.0, sigma_noise, A.shape) * mask

    if remove_idx:
        A[:, list(remove_idx), :] = 0.0

    if remove_rand:
        for i in range(n):
            A[i, rng.choice(N_KP, remove_rand, replace=False), :] = 0.0

    return A.astype(np.float32), y[pick]


def augment(X, y, seed=0):
    """Bảng tham số slide 23, chép nguyên. Trả về phần THÊM VÀO."""
    rng = np.random.default_rng(seed)
    recipes = [
        # ratio, sig_scale, sig_rot, sig_noise, remove_idx,          remove_rand
        (0.10, 0.08, 0.0, 0.00, (), 0),
        (0.10, 0.00, 10.0, 0.00, (), 0),
        (0.15, 0.00, 0.0, 0.03, IDX_LEGS, 0),
        (0.15, 0.00, 0.0, 0.03, IDX_LEGS + IDX_HIPS, 0),
        (0.20, 0.00, 0.0, 0.03, (), 2),
    ]
    Xs, ys = [], []
    for r in recipes:
        a, b = _augment_once(X, y, *r, rng)
        Xs.append(a); ys.append(b)
    return np.concatenate(Xs), np.concatenate(ys)


# ══════════════════════════════════════════════════════════════════════════
#  MÔ HÌNH
# ══════════════════════════════════════════════════════════════════════════

def build_hybrid(n_class, dropout=0.2):
    """
    Hybrid theo slide 25, NHƯNG ĐỔI THỨ TỰ TRONG KHỐI. Đã đo, không phải sở thích.

    Slide viết   1CBR = Conv1D + BatchNorm + Dropout + ReLU   (dropout TRƯỚC relu)
    Ở đây làm    Conv1D + BatchNorm + ReLU + Dropout          (dropout SAU relu)

    SỐ ĐO, trên tập giả 9 lớp / 45 take / tách được hoàn hảo (1-NN = 100%):

                                    tap gia std=0.005   tap gia std=0.013   PCK that
        thu tu slide (drop->relu)        11.11%              (khong do)      99.44%
        thu tu chuan (relu->drop)        35.00%              100.00%         98.94%

    Đọc bảng này cho đúng:

    a) Cột đầu là một tập SUY BIẾN — phương sai trong lớp chỉ 0.005, gần như
       không có nhiễu. Ở đó Dropout-trước-ReLU sập xuống đúng 1/9 = đoán bừa,
       còn thứ tự chuẩn cũng chỉ gượng lên 35%. Cả hai đều hỏng; thứ tự chuẩn
       chỉ hỏng ít hơn.

    b) Nâng nhiễu lên mức của người thật (cột giữa) thì hybrid đạt 100%. Trên
       dữ liệu PCK thật hai thứ tự chênh 0.5%, tức là ngang nhau.

    => Chọn thứ tự chuẩn vì nó không thua ở đâu và bền hơn ở chỗ khắc nghiệt.
       Nguyên nhân của cột đầu: Dropout nhân đầu ra với 1/(1-p) lúc train mà
       không nhân lúc suy luận; BatchNorm của khối SAU học thống kê trên phương
       sai đã bị thổi phồng rồi áp lên phương sai thật lúc suy luận. Sai lệch
       cộng dồn qua 4 khối BN. (Li et al., "Understanding the Disharmony
       between Dropout and Batch Normalization by Variance Shift".)

    BÀI HỌC THẬT SỰ, quan trọng hơn thứ tự lớp: BatchNorm hỏng khi một lớp có
    QUÁ ÍT BIẾN THIÊN. Thu một take đứng yên một chỗ cho mỗi lớp thì tạo ra
    đúng cột đầu tiên. Nhiều take, mỗi take một điều kiện — đó không phải lời
    khuyên cho đẹp, đó là điều kiện để kiến trúc này chạy được. Tập tự thu còn
    nhỏ thì dùng --arch pck (không có BatchNorm) trước.
    """
    from tensorflow import keras
    from tensorflow.keras import layers as L

    def cbr(x, filters, k):
        x = L.Conv1D(filters, k, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        return L.Dropout(dropout)(x)

    def dbr(x, units):
        x = L.Dense(units, use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        return L.Dropout(dropout)(x)

    inp = keras.Input(shape=(N_KP, N_COORD), name="keypoints")
    x = cbr(inp, 16, 3)
    x = cbr(x, 16, 3)
    x = L.Flatten()(x)
    x = dbr(x, 128)
    x = dbr(x, 128)
    out = L.Dense(n_class, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name="PCK_Hybrid")


def build_pck_actual(n_class, dropout=0.25):
    """
    Kiến trúc ĐỌC TRỰC TIẾP từ file .h5 mà PCK ship, bằng h5py.

    Ba nguồn nói ba kiểu — đây là lý do phải mở file ra xem thay vì tin slide:

        tên file   CNN-2Conv1D-64x3filter-2dense-2x128
        slide 25   1CBR(3,16) -> 1CBR(3,16) -> DBR(128) -> DBR(128)
        .h5 THAT   Conv1D(16,3) -> Conv1D(32,3) -> Conv1D(32,3)
                   -> Dense(128) -> Dense(64) -> Dense(20)
                   dropout 0.25, KHONG co BatchNorm

    Thứ duy nhất cả ba nguồn đồng ý, và cũng là thứ quan trọng nhất với ta:
    ĐẦU VÀO LÀ (25, 2) — 25 khớp, 2 toạ độ, Conv1D trượt dọc TRỤC KHỚP.
    """
    from tensorflow import keras
    from tensorflow.keras import layers as L
    inp = keras.Input(shape=(N_KP, N_COORD), name="keypoints")
    x = inp
    for f in (16, 32, 32):
        x = L.Conv1D(f, 3, activation="relu")(x)   # padding='valid', mac dinh Keras
        x = L.Dropout(dropout)(x)
    x = L.Flatten()(x)
    x = L.Dense(128, activation="relu")(x)
    x = L.Dense(64, activation="relu")(x)
    out = L.Dense(n_class, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name="PCK_Actual")


def build_light_cnn(n_class, dropout=0.2):
    """Bản nhỏ của slide 25, để đối chứng độ bền."""
    from tensorflow import keras
    from tensorflow.keras import layers as L
    inp = keras.Input(shape=(N_KP, N_COORD), name="keypoints")
    x = inp
    for _ in range(2):
        x = L.Conv1D(12, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(dropout)(x)
        x = L.ReLU()(x)
    x = L.Flatten()(x)
    out = L.Dense(n_class, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name="PCK_LightCNN")


BUILDERS = {"pck": build_pck_actual, "hybrid": build_hybrid,
            "light": build_light_cnn}


# ══════════════════════════════════════════════════════════════════════════
#  MỘT LƯỢT TRAIN
# ══════════════════════════════════════════════════════════════════════════

def chuan_bi_train(X, y, n_class, seed, dung_mirror, labels, khong_augment):
    """Nhân bản tập TRAIN: soi gương trước, rồi tăng cường trên tất cả."""
    Xt, yt = X, y
    if dung_mirror:
        Xm, ym = mirror(Xt, yt, labels)
        if len(Xm):
            Xt = np.concatenate([Xt, Xm])
            yt = np.concatenate([yt, ym])
    if not khong_augment:
        Xa, ya = augment(Xt, yt, seed=seed)
        Xt = np.concatenate([Xt, Xa])
        yt = np.concatenate([yt, ya])
    return Xt, yt


def trong_so_mau(y, n_class):
    """Bù lệch lớp bằng trọng số mẫu.

    Dùng sample_weight chứ không dùng class_weight của Keras: label smoothing
    cần nhãn one-hot, mà class_weight thì đòi nhãn số nguyên. Hai thứ đó không
    đi cùng nhau được.
    """
    dem = np.bincount(y, minlength=n_class).astype(np.float64)
    w = np.where(dem > 0, len(y) / (n_class * np.maximum(dem, 1)), 0.0)
    return w[y].astype(np.float32)


def train_mot_lan(tf, arch, n_class, Xtr, ytr, Xva, yva, epochs, batch,
                  seed, label_smooth, verbose=0):
    """-> (model, so epoch tot nhat)"""
    tf.keras.utils.set_random_seed(seed)
    model = BUILDERS[arch](n_class)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        # Label smoothing 0.05: không cho mạng đẩy xác suất về đúng 1.0. Với
        # tập nhỏ nó là thứ giữ cho ngưỡng tau ở dưới còn phân biệt được —
        # softmax bão hoà thì mọi mẫu đều p_max = 1.000 và ngưỡng vô dụng.
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth),
        metrics=["accuracy"])

    Ytr = tf.keras.utils.to_categorical(ytr, n_class)
    Yva = tf.keras.utils.to_categorical(yva, n_class)
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=4, min_lr=1e-5),
    ]
    h = model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=epochs,
                  batch_size=batch, callbacks=cb, verbose=verbose,
                  sample_weight=trong_so_mau(ytr, n_class))
    best = int(np.argmin(h.history["val_loss"])) + 1
    return model, best


# ══════════════════════════════════════════════════════════════════════════
#  HIỆU CHUẨN + NGƯỠNG TỪ CHỐI
# ══════════════════════════════════════════════════════════════════════════
# Softmax cho ra thứ TRÔNG NHƯ xác suất nhưng không phải: mạng train tới hội tụ
# gần như luôn TỰ TIN QUÁ MỨC. Nếu lấy thẳng p_max so với 0.9 thì ngưỡng đó
# không tương ứng với "đúng 90% số lần".
#
# Chia logit cho một nhiệt độ T rồi mới softmax là phép hiệu chuẩn rẻ nhất và
# hầu như luôn hiệu quả (Guo et al. 2017, "On Calibration of Modern Neural
# Networks"). Ở đây chỉ có xác suất chứ không có logit, nhưng không sao:
#
#     p_i^(1/T) / SUM_j p_j^(1/T)  ==  softmax(z/T)
#
# nên nâng luỹ thừa 1/T rồi chuẩn hoá lại là ĐÚNG BẰNG phép chia logit.

def ap_nhiet_do(P, T):
    if abs(T - 1.0) < 1e-9:
        return P
    Q = np.power(np.clip(P, 1e-12, 1.0), 1.0 / T)
    return Q / Q.sum(1, keepdims=True)


def khop_nhiet_do(P, y):
    """Quét T, chọn T cho NLL nhỏ nhất trên chính dự đoán out-of-fold."""
    best_T, best_nll = 1.0, np.inf
    for T in np.concatenate([np.arange(0.50, 1.00, 0.02),
                             np.arange(1.00, 5.05, 0.05)]):
        Q = ap_nhiet_do(P, float(T))
        nll = float(-np.log(np.clip(Q[np.arange(len(y)), y], 1e-12, 1)).mean())
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T, best_nll


def ece(P, y, bins=10):
    """Expected Calibration Error — chênh giữa 'tự tin' và 'đúng thật'."""
    conf, pred = P.max(1), P.argmax(1)
    dung = (pred == y).astype(float)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf >= lo) & (conf < lo + 1.0 / bins)
        if m.sum():
            e += m.mean() * abs(dung[m].mean() - conf[m].mean())
    return float(e)


def chon_nguong(P, y, labels, muc_tieu=0.99, P_la=None, muc_tieu_la=0.90):
    """
    Chọn tau thoả ĐỒNG THỜI hai điều kiện. -> (tau, bang, ly_do)

    bang = [(tau, do_chinh_xac, do_phu, so_lenh, tu_choi_dau_vao_la), ...]

    HAI điều kiện, không phải một. Bản đầu của hàm này chỉ có điều kiện thứ
    nhất, và đã đo được hậu quả trên tập kiểm: dữ liệu tách hoàn hảo -> mọi tau
    đều đạt 100% -> chọn tau nhỏ nhất = 0.30 -> tỷ lệ từ chối nhiễu ngẫu nhiên
    tụt còn 15.2%. Ngưỡng "tối ưu" trên dữ liệu quen là ngưỡng VÔ DỤNG trước
    dữ liệu lạ, mà lúc bay thì dữ liệu lạ mới là thứ hay gặp.

    Định nghĩa cho đúng, vì đây là chỗ dễ tự lừa:
      LỆNH ĐÃ PHÁT = p_max >= tau  VÀ  nhãn đoán khác NEGATIVE.
      ĐÚNG         = nhãn đoán trùng nhãn thật.
      -> đoán ra cử chỉ trong khi sự thật là NEGATIVE bị tính LÀ SAI. Phải vậy:
         đó chính là lệnh ma, loại lỗi nguy hiểm nhất của cả hệ.
      ĐỘ PHỦ       = trong các frame THẬT SỰ là cử chỉ, bao nhiêu % phát ra lệnh.
         Đây là cái giá phải trả cho độ chính xác, và người dùng cảm nhận nó
         thành "phải giữ tay lâu hơn".
    """
    neg = labels.index("NEGATIVE") if "NEGATIVE" in labels else -1
    pred, pmax = P.argmax(1), P.max(1)
    la_cu_chi = (y != neg)
    bang = []
    for tau in np.round(np.arange(0.30, 1.00, 0.01), 2):
        phat = (pmax >= tau) & (pred != neg)
        if phat.sum() == 0:
            continue
        cx = float((pred[phat] == y[phat]).mean())
        phu = float((phat & la_cu_chi).sum()) / max(1, int(la_cu_chi.sum()))
        tc = ty_le_tu_choi(P_la, labels, float(tau)) if P_la is not None else None
        bang.append((float(tau), cx, phu, int(phat.sum()), tc))

    dat_cx = [r for r in bang if r[1] >= muc_tieu]
    dat_ca_hai = [r for r in dat_cx
                  if r[4] is None or r[4] >= muc_tieu_la]
    if dat_ca_hai:
        return dat_ca_hai[0][0], bang, "dat ca hai muc tieu"
    if dat_cx:
        # Đạt độ chính xác nhưng không chặn nổi đầu vào lạ. Lấy tau CAO NHẤT
        # còn đạt độ chính xác — mỗi nấc tau cao thêm là một nấc chặn tốt hơn,
        # mà ở đây không mất gì về độ chính xác cả.
        return dat_cx[-1][0], bang, "dat do chinh xac, CHUA chan du dau vao la"
    return None, bang, "khong dat do chinh xac o bat ky nguong nao"


# ══════════════════════════════════════════════════════════════════════════
#  MÔ PHỎNG LUẬT PHÁT LỆNH THẬT
# ══════════════════════════════════════════════════════════════════════════

def mo_phong_phat_lenh(P, y, groups, labels, tau, k_vote, fps):
    """
    Chạy đúng luật của tầng quyết định trên dự đoán out-of-fold.

    Luật: phát lệnh khi k frame LIÊN TIẾP cùng thoả (p_max >= tau và nhãn khác
    NEGATIVE) và cùng một nhãn. Đây là bản rời rạc của GESTURE_HOLD_SEC trong
    command_logic.py — hệ thật không nghe theo một frame đơn lẻ bao giờ.

    Vì sao phải đo riêng: accuracy theo frame và số lệnh sai KHÔNG tỷ lệ với
    nhau. Nhầm rải rác từng frame một thì bộ đếm k không bao giờ đầy và không
    lệnh sai nào phát ra; nhầm thành CHÙM k frame thì một lệnh sai đi thẳng ra
    drone. Chỉ có mô phỏng theo thời gian mới phân biệt được hai trường hợp.
    """
    neg = labels.index("NEGATIVE") if "NEGATIVE" in labels else -1
    pred, pmax = P.argmax(1), P.max(1)
    n_frame = n_phat = n_sai = 0
    tre = []
    sai_chi_tiet = {}

    for g in dict.fromkeys(groups.tolist()):
        idx = np.where(groups == g)[0]        # đã đúng thứ tự thời gian
        buf = deque(maxlen=k_vote)
        dau_tien = None
        for pos, i in enumerate(idx):
            ok = (pmax[i] >= tau) and (pred[i] != neg)
            buf.append(int(pred[i]) if ok else -1)
            n_frame += 1
            if len(buf) == k_vote and buf[0] != -1 and len(set(buf)) == 1:
                n_phat += 1
                if dau_tien is None:
                    dau_tien = pos
                if buf[0] != y[i]:
                    n_sai += 1
                    key = (labels[int(y[i])], labels[int(buf[0])])
                    sai_chi_tiet[key] = sai_chi_tiet.get(key, 0) + 1
        if y[idx[0]] != neg:
            tre.append(dau_tien if dau_tien is not None else -1)

    bat_duoc = sum(1 for v in tre if v >= 0)
    return {
        "k_vote": k_vote,
        "tau": tau,
        "n_frame": n_frame,
        "n_phat": n_phat,
        "n_sai": n_sai,
        "ty_le_sai": n_sai / max(1, n_phat),
        "lenh_sai_moi_phut": n_sai / max(1e-9, n_frame / fps) * 60.0,
        "take_bat_duoc": f"{bat_duoc}/{len(tre)}",
        "tre_frame_trung_vi": float(np.median([v for v in tre if v >= 0]))
                              if bat_duoc else None,
        "sai_chi_tiet": {f"{a}->{b}": c for (a, b), c in
                         sorted(sai_chi_tiet.items(), key=lambda kv: -kv[1])},
    }


def chon_nguong_theo_lenh(Pc, y, groups, labels, k_vote, fps,
                          P_la=None, muc_tieu_la=0.90,
                          ngan_sach=0.0, bien=0.01):
    """
    Chọn tau NHỎ NHẤT mà luật phát lệnh không còn lệnh sai. -> (tau, bien_thuc, ly_do)

    Vì sao không dùng độ chính xác theo frame (hàm chon_nguong ở trên):

        tau    lệnh phát   lệnh SAI   take bắt được
        0.95        1624          0          23/26
        0.98         306          0           8/26

    Hai ngưỡng cùng 0 lệnh sai. Chỉ tiêu theo frame vẫn chấm 0.98 cao hơn, vì
    nó đếm cả những frame nhầm LẺ TẺ — mà luật k_vote đòi k frame liên tiếp
    cùng ý kiến, nên nhầm lẻ tẻ không bao giờ thành lệnh. Tối ưu theo frame ở
    đây là tối ưu nhầm thứ, và cái giá phải trả là 15 take câm.

    `bien` cộng thêm sau khi tìm được điểm hết lỗi: điểm đó tìm trên CHÍNH dữ
    liệu dùng để đo, nên nó lạc quan. Biên là khoản dự phòng, không phải phép
    đo.
    """
    grid = np.round(np.arange(0.50, 0.9951, 0.005), 3)
    rows = []
    for t in grid:
        m = mo_phong_phat_lenh(Pc, y, groups, labels, float(t), k_vote, fps)
        tc = ty_le_tu_choi(P_la, labels, float(t)) if P_la is not None else None
        rows.append((float(t), m, tc))

    dat = [r for r in rows
           if r[1]["lenh_sai_moi_phut"] <= ngan_sach
           and (r[2] is None or r[2] >= muc_tieu_la)]
    if not dat:
        # Không ngưỡng nào vừa hết lỗi vừa chặn đủ đầu vào lạ. Ưu tiên HẾT LỖI
        # — chặn đầu vào lạ còn có NEGATIVE làm lớp phòng thủ thứ hai, còn lệnh
        # sai thì không có gì đỡ.
        het_loi = [r for r in rows if r[1]["lenh_sai_moi_phut"] <= ngan_sach]
        if not het_loi:
            return None, 0.0, "khong nguong nao het lenh sai"
        t0 = het_loi[0][0]
        return (min(0.995, t0 + bien), 0.0,
                "het lenh sai nhung CHUA chan du dau vao la")

    t0 = dat[0][0]
    # biên thực: khoảng cách từ điểm hết lỗi xuống ngưỡng gần nhất CÒN lỗi
    con_loi = [r[0] for r in rows if r[0] < t0
               and r[1]["lenh_sai_moi_phut"] > ngan_sach]
    bien_thuc = (t0 - max(con_loi)) if con_loi else t0 - 0.50
    return min(0.995, t0 + bien), bien_thuc, f"het lenh sai tu tau={t0:.3f}"


def cham_tung_take(P, y, groups, labels, tau, k_vote):
    """
    Chấm RIÊNG từng take. -> [(take, lop, n, acc, ty_le_phat), ...]

    Vì sao cần: báo cáo theo lớp cộng gộp mọi take lại, nên một take hỏng bị
    năm take tốt che đi. Mà cái ta phải sửa lại chính là take hỏng đó — biết
    tên nó thì chỉ phải quay lại một lần 23 giây, không biết thì phải quay lại
    cả lớp.

    `ty_le_phat` là tỷ lệ frame của take đó phát được lệnh dưới đúng luật thật
    (tau + k_vote frame liên tiếp). Take có acc cao mà ty_le_phat = 0 nghĩa là
    model đoán ĐÚNG nhưng KHÔNG ĐỦ TỰ TIN — lúc bay take đó câm.
    """
    neg = labels.index("NEGATIVE") if "NEGATIVE" in labels else -1
    pred, pmax = P.argmax(1), P.max(1)
    out = []
    for gname in dict.fromkeys(groups.tolist()):
        idx = np.where(groups == gname)[0]
        acc = float((pred[idx] == y[idx]).mean())
        buf, n_phat = deque(maxlen=k_vote), 0
        for i in idx:
            ok = (pmax[i] >= tau) and (pred[i] != neg)
            buf.append(int(pred[i]) if ok else -1)
            if len(buf) == k_vote and buf[0] != -1 and len(set(buf)) == 1:
                n_phat += 1
        out.append((gname, labels[int(y[idx[0]])], len(idx), acc,
                    n_phat / max(1, len(idx))))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  KIỂM ĐẦU VÀO LẠ (out-of-distribution)
# ══════════════════════════════════════════════════════════════════════════

# Ba họ đầu vào lạ, nhưng CHỈ HAI trong đó có thể xảy ra thật.
#
# MediaPipe luôn xuất ra một bộ xương có hình dạng người — nó không bao giờ trả
# về toạ độ ngẫu nhiên đều. Nên "nhieu ngau nhien" là phép thử ÁP LỰC, đọc để
# biết mạng cứng tới đâu, chứ KHÔNG được dùng làm điều kiện chọn ngưỡng.
#
# Đã trả giá cho chỗ này: gộp cả ba họ lại thì tỷ lệ chặn ở tau=0.95 là 88.1%,
# trượt mục tiêu 90%, nên ngưỡng bị đẩy lên 0.98 — và số take phát được lệnh
# rơi từ 23/26 xuống 8/26. Hai họ thực tế ở tau=0.95 đã là 98.7% và 100%.
LA_THUC_TE = ("dao khop", "mat hai co tay")


def ty_le_tu_choi(P, labels, tau):
    """Tỷ lệ mẫu bị CHẶN: hoặc dưới ngưỡng, hoặc đoán ra NEGATIVE."""
    neg = labels.index("NEGATIVE") if "NEGATIVE" in labels else -1
    return float(((P.max(1) < tau) | (P.argmax(1) == neg)).mean())


def sinh_dau_vao_la(X, seed=0, n=600):
    """
    Ba loại đầu vào KHÔNG PHẢI cử chỉ, xem hệ có từ chối không.

    Đây là phép thử cho chính cái lỗi đã đo được ở hệ này: entropy của nhiễu
    ngẫu nhiên (0.183) THẤP HƠN entropy của tư thế thật (0.553) — mạng tự tin
    hơn khi gặp thứ nó chưa từng thấy. NEGATIVE + tau là hai lớp phòng thủ; đây
    là chỗ kiểm chúng có thật sự đứng vững không.

      nhieu    toạ độ ngẫu nhiên đều trong [-1,1]. Phép thử dễ nhất.
      dao khop hoán vị chỉ số khớp của tư thế THẬT. Giữ nguyên phân bố từng
               toạ độ, chỉ phá CẤU TRÚC. Khó hơn hẳn, và giống lỗi thật của
               MediaPipe khi nó gán nhầm khớp lúc người quay lưng.
      mat tay  tư thế thật nhưng xoá hai cổ tay. Giống hệt lúc người đứng quá
               gần camera — trường hợp đã xảy ra thật, 67% mẫu của một phiên.
    """
    rng = np.random.default_rng(seed)
    base = X[rng.choice(len(X), n, replace=True)]
    nhieu = rng.uniform(-1, 1, (n, N_KP, N_COORD)).astype(np.float32)
    dao = np.stack([b[rng.permutation(N_KP)] for b in base]).astype(np.float32)
    mat = base.copy(); mat[:, [4, 7], :] = 0.0
    return {"nhieu ngau nhien": nhieu, "dao khop": dao,
            "mat hai co tay": mat.astype(np.float32)}


# ══════════════════════════════════════════════════════════════════════════
#  KIỂM TRA MODEL CÓ SẴN CỦA PCK
# ══════════════════════════════════════════════════════════════════════════

def inspect_pck_model(wheel=None):
    """Xác nhận shape đầu vào thật của model PCK, thay vì đoán từ slide."""
    import h5py
    if wheel is None:
        wheel = find_pck_wheel()
    path = "pose_classification_kit/models/Body/20Class_CNN_BODY25/" \
           "CNN-2Conv1D-64x3filter-2dense-2x128_body25.h5"
    tmp = ROOT / ".venv-train" / "_pck_model.h5"
    with zipfile.ZipFile(wheel) as z:
        tmp.write_bytes(z.read(path))
    with h5py.File(tmp, "r") as f:
        cfg = f.attrs.get("model_config")
        if isinstance(cfg, bytes):
            cfg = cfg.decode("utf8")
        cfg = json.loads(cfg)
    layers = cfg["config"]["layers"]
    print("  Kien truc model 20 lop cua PCK:")
    for l in layers:
        c, t = l["config"], l["class_name"]
        bits = []
        if "batch_input_shape" in c:
            bits.append(f"input={c['batch_input_shape']}")
        for k in ("filters", "kernel_size", "units", "rate", "activation"):
            if k in c and c[k] is not None:
                bits.append(f"{k}={c[k]}")
        print(f"    {t:<20}{'  '.join(str(b) for b in bits)}")
    tmp.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="chi doc kien truc model co san cua PCK roi thoat")
    ap.add_argument("--check", action="store_true",
                    help="CHI kiem dinh du lieu roi thoat. Khong can TensorFlow, "
                         "chay duoc o moi truong chinh.")
    ap.add_argument("--csv", default=None, help="duong dan BodyPose_Dataset.csv")
    ap.add_argument("--npz", default=None,
                    help="tap TU THU (data/dataset_v3.npz). Co cai nay thi bo qua --csv")
    ap.add_argument("--roster", default=DEFAULT_ROSTER,
                    choices=tuple(ROSTERS) + ("none",),
                    help="bo lop bat buoc phai co. v1 = 6 cu chi (mac dinh)")
    ap.add_argument("--arch", choices=tuple(BUILDERS), default="pck")
    ap.add_argument("--cv", type=int, default=4,
                    help="so fold kiem chung cheo theo take (0 = tat, chia 1 lan)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-mirror", action="store_true",
                    help="tat phep soi guong (mac dinh BAT)")
    ap.add_argument("--no-quant", action="store_true",
                    help="xuat tflite khong luong tu hoa trong so")
    ap.add_argument("--muc-tieu", type=float, default=0.99,
                    help="do chinh xac toi thieu cua LENH DA PHAT, dung de chon tau")
    ap.add_argument("--ngan-sach-sai", type=float, default=0.0,
                    help="so LENH SAI moi phut con chap nhan duoc, dung de chon "
                         "tau. Mac dinh 0.")
    ap.add_argument("--muc-tieu-la", type=float, default=0.90,
                    help="ty le toi thieu phai TU CHOI duoc dau vao la, "
                         "dung de chon tau cung voi --muc-tieu")
    ap.add_argument("--k-vote", type=int, default=8,
                    help="so frame lien tiep phai dong y truoc khi phat lenh. "
                         "8 frame @ 8 fps = 1.0 giay.")
    ap.add_argument("--fps", type=float, default=8.0,
                    help="nhip vong lap thuc te, chi dung khi tap khong co moc "
                         "thoi gian")
    ap.add_argument("--force", action="store_true",
                    help="bo qua cac loi CHAN. Model xuat ra se mang hau to _UNSAFE.")
    ap.add_argument("--smoke", action="store_true",
                    help="CHAY THU: train tren TOAN BO du lieu, KHONG danh gia. "
                         "Chi de kiem chuoi chay thong.")
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.inspect:
        inspect_pck_model()
        return 0

    print("=" * 78)
    print("TRAIN_PCK")
    print("=" * 78)

    # ---- nạp ---------------------------------------------------------------
    t_moc = None
    if args.npz:
        X, y_str, labels, groups, t_moc = load_npz_dataset(args.npz)
        source = Path(args.npz).name
    else:
        X, y_str, labels = load_pck_dataset(args.csv)
        # Tập PCK không có khái niệm take. Gom 200 dòng liền nhau thành một
        # "take" giả để K-fold chạy được. File CSV sắp theo lớp nên mỗi nhóm
        # nằm trọn trong một lớp — không hoàn hảo, nhưng vẫn chặt hơn chia
        # ngẫu nhiên theo mẫu.
        groups = np.array([f"pck_{i//200}" for i in range(len(X))])
        source = "BodyPose_Dataset.csv"
    roster = None if args.roster == "none" or not args.npz else ROSTERS[args.roster]
    fp = van_tay(X, y_str, groups)
    print(f"  van tay du lieu   {fp}")

    in_bang_lop(y_str, labels, groups)

    # fps đo từ dữ liệu nếu có mốc thời gian — chính xác hơn hẳn tham số --fps
    fps = args.fps
    if t_moc is not None and len(t_moc) > 8:
        dt = np.diff(t_moc)
        dt = dt[(dt > 1e-4) & (dt < 1.0)]
        if len(dt) > 8:
            fps = float(np.clip(1.0 / np.median(dt), 1.0, 60.0))
            print(f"\n  nhip vong lap DO TU DU LIEU: {fps:.1f} fps")

    # ---- kiểm định ---------------------------------------------------------
    k_fold = max(0, args.cv)
    chan, luu_y = kiem_dinh(X, y_str, labels, groups, roster,
                            max(k_fold, MIN_TAKE_MOI_LOP) if k_fold else MIN_TAKE_MOI_LOP)

    print("\n" + "-" * 78)
    print("KIEM DINH DU LIEU")
    print("-" * 78)
    if chan:
        print("\n  CHAN - train trong tinh trang nay se ra mot so KHONG DO DUOC GI:")
        for m in chan:
            print(f"    x  {m}")
    if luu_y:
        print("\n  LUU Y - dung duoc, nhung yeu o cho nay:")
        for m in luu_y:
            print(f"    !  {m}")
    if not chan and not luu_y:
        print("\n  Khong co van de nao.")

    if chan and not args.force:
        print("\n" + "=" * 78)
        print("  DUNG LAI. Sua nhung dong 'x' o tren roi chay lai.")
        print()
        print("  Thu them du lieu:")
        print("      python training\\1_record.py --cam 1 --classes "
              + ",".join(roster or ROSTERS[DEFAULT_ROSTER]) + " --save-video")
        print("      python training\\2_check_poses.py")
        print("      python training\\3_build_dataset.py")
        print()
        print("  Van muon train (chi de thu chuoi chay):  them --force")
        print("  Khi do model se ten *_UNSAFE.tflite de tu no khai ra.")
        print("=" * 78)
        return 2

    if args.check:
        print("\n" + "=" * 78)
        print("  --check: chi kiem dinh, khong train. "
              + ("CO loi CHAN." if chan else "Du lieu dung duoc."))
        print("=" * 78)
        return 1 if chan else 0

    khong_an_toan = bool(chan)     # đã --force qua

    # ---- TensorFlow --------------------------------------------------------
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError:
        raise SystemExit(
            "\n  THIEU tensorflow -> dang chay NHAM python.\n\n"
            "  File nay chay bang VENV TRAIN, khong phai python chinh:\n"
            "      .venv-train\\Scripts\\python.exe training\\4_train.py "
            "--npz data\\dataset_v3.npz\n\n"
            "  Moi truong chinh CO Y khong cai TensorFlow. Xem phan dau file.\n"
            "  (Muon kiem du lieu bang python chinh thi dung --check.)\n"
        )
    from sklearn.metrics import confusion_matrix, classification_report

    tf.keras.utils.set_random_seed(args.seed)
    print(f"\n  tensorflow {tf.__version__}   numpy {np.__version__}")

    lut = {l: i for i, l in enumerate(labels)}
    y = np.array([lut[s] for s in y_str], dtype=np.int32)
    n_class = len(labels)
    if n_class < 2:
        raise SystemExit(f"\n  CHI CO {n_class} LOP. Khong co gi de phan biet.\n")

    # Tập đầu vào LẠ, dựng một lần ở đây. Mỗi fold chấm 1/k tập này bằng model
    # của chính nó, nên điểm số của nó cũng là out-of-fold — không mẫu lạ nào
    # được chấm bởi model đã nhìn thấy dữ liệu sinh ra nó.
    LA = sinh_dau_vao_la(X, args.seed)
    ten_la = list(LA)
    X_la = np.concatenate([LA[k] for k in ten_la])
    ho_la = np.concatenate([np.full(len(LA[k]), i) for i, k in enumerate(ten_la)])
    P_la = None

    dung_mirror = not args.no_mirror
    print(f"  kien truc {args.arch}   soi guong {'BAT' if dung_mirror else 'TAT'}"
          f"   tang cuong {'TAT' if args.no_augment else 'BAT'}")

    # ══════════════════════════════════════════════════════════════════════
    #  KIỂM CHỨNG CHÉO
    # ══════════════════════════════════════════════════════════════════════
    P_oof = None
    acc_fold = []
    best_epochs = []

    if args.smoke:
        print("\n  *** CHE DO CHAY THU (--smoke) ***")
        print("  Train tren TOAN BO du lieu, khong giu lai tap danh gia nao.")
        print("  Moi con so sau day do TREN CHINH DU LIEU DA HOC.")
        best_epochs = [args.epochs]
    elif k_fold >= 2:
        print("\n" + "-" * 78)
        print(f"KIEM CHUNG CHEO {k_fold}-FOLD THEO TAKE")
        print("-" * 78)
        print("  Moi take duoc lam TEST dung mot lan. Gop lai duoc du doan")
        print("  out-of-fold cho TOAN BO tap - khong mau nao tu cham chinh minh.\n")
        P_oof = np.zeros((len(X), n_class), dtype=np.float64)
        P_la = np.zeros((len(X_la), n_class), dtype=np.float64)
        manh_la = np.array_split(
            np.random.default_rng(args.seed).permutation(len(X_la)), k_fold)
        folds = folds_by_take(y, groups, labels, k_fold, args.seed)
        for fi, (itr, iva, ite) in enumerate(folds):
            Xtr, ytr = chuan_bi_train(X[itr], y[itr], n_class, args.seed + fi,
                                      dung_mirror, labels, args.no_augment)
            m, be = train_mot_lan(tf, args.arch, n_class, Xtr, ytr,
                                  X[iva], y[iva], args.epochs, args.batch,
                                  args.seed + fi, args.label_smooth)
            P_oof[ite] = m.predict(X[ite], verbose=0)
            P_la[manh_la[fi]] = m.predict(X_la[manh_la[fi]], verbose=0)
            a = float((P_oof[ite].argmax(1) == y[ite]).mean())
            acc_fold.append(a)
            best_epochs.append(be)
            print(f"  fold {fi+1}/{k_fold}   train {len(Xtr):>6}  val {len(iva):>5}"
                  f"  test {len(ite):>5}   acc {a*100:6.2f}%   epoch tot nhat {be}")
            tf.keras.backend.clear_session()

        mu, sd = float(np.mean(acc_fold)), float(np.std(acc_fold))
        print(f"\n  accuracy theo frame   {mu*100:6.2f}%  +/- {sd*100:.2f}")
        if sd > 0.06:
            print("  !  do lech giua cac fold LON -> con so trung binh chua on dinh.")
            print("     Nguyen nhan thuong gap: cac take khong du khac nhau, hoac")
            print("     mot vi tri dung chi xuat hien o dung mot take.")
    else:
        print("\n  --cv 0: bo qua kiem chung cheo. Khong co du doan out-of-fold")
        print("  nen KHONG chon duoc nguong tau va KHONG do duoc lenh sai/phut.")

    # ══════════════════════════════════════════════════════════════════════
    #  HIỆU CHUẨN, NGƯỠNG, MÔ PHỎNG — tất cả trên dự đoán out-of-fold
    # ══════════════════════════════════════════════════════════════════════
    T = 1.0
    tau = None
    mo_phong = None
    if P_oof is not None:
        e0 = ece(P_oof, y)
        T, _ = khop_nhiet_do(P_oof, y)
        Pc = ap_nhiet_do(P_oof, T)
        e1 = ece(Pc, y)
        print("\n" + "-" * 78)
        print("HIEU CHUAN VA NGUONG TU CHOI")
        print("-" * 78)
        print(f"  nhiet do T = {T:.2f}   "
              f"({'giam' if T > 1 else 'tang'} do tu tin)")
        print(f"  sai so hieu chuan ECE   {e0:.4f} -> {e1:.4f}   "
              f"({'tot hon' if e1 < e0 else 'khong doi'})")

        Pc_la = ap_nhiet_do(P_la, T) if P_la is not None else None
        # Chon nguong CHI nhin cac ho co the xay ra that (xem LA_THUC_TE).
        if Pc_la is not None:
            m_tt = np.isin(ho_la, [i for i, n in enumerate(ten_la)
                                   if n in LA_THUC_TE])
            Pc_la_gate = Pc_la[m_tt] if m_tt.any() else Pc_la
        else:
            Pc_la_gate = None
        _, bang, _ = chon_nguong(Pc, y, labels, args.muc_tieu,
                                 Pc_la, args.muc_tieu_la)
        # Chon THEO LENH, khong theo frame. Bang duoi chi de xem.
        tau, bien, ly_do = chon_nguong_theo_lenh(
            Pc, y, groups, labels, args.k_vote, fps, Pc_la_gate,
            args.muc_tieu_la, args.ngan_sach_sai)
        print(f"    {'tau':>6}{'lenh dung':>12}{'do phu':>10}{'so lenh':>10}{'chan dau vao la':>18}")
        for r in bang:
            if abs(r[0] * 100 - round(r[0] * 100 / 5) * 5) < 1e-6:
                cot = (f"{r[4]*100:>17.1f}%" if r[4] is not None
                       else f"{'-':>18}")
                print(f"  {r[0]:>6.2f}{r[1]*100:>11.2f}%{r[2]*100:>9.1f}%"
                      f"{r[3]:>10}{cot}")
        print("  (bang tren dem SAI THEO FRAME - chi de tham khao. Nguong that chon o duoi.)")
        if tau is None:
            print("\n  !  KHONG nguong nao het lenh sai. Hai cu chi dang chong len nhau -")
            print("     xem CAP DE NHAM ben duoi.")
            tau = 0.95
            print(f"     Tam dat tau = {tau:.2f} de con chay tiep.")
        else:
            print(f"\n  -> tau = {tau:.3f}   ({ly_do}, cong bien 0.01)")
            if bien < 0.02:
                print(f"     !  BIEN MONG: chi {bien:.3f} tu day toi nguong con phat lenh sai.")
                print("     Diem lam viec nay do tren chinh du lieu nay nen lac quan.")
                print("     Them take roi do lai truoc khi tin no.")
            else:
                print(f"     bien toi nguong con phat lenh sai: {bien:.3f}")

        mo_phong = mo_phong_phat_lenh(Pc, y, groups, labels, tau,
                                      args.k_vote, fps)
        print("\n" + "-" * 78)
        print(f"MO PHONG LUAT PHAT LENH  ({args.k_vote} frame lien tiep, "
              f"{fps:.1f} fps = {args.k_vote/fps:.2f} s)")
        print("-" * 78)
        print(f"  lenh da phat        {mo_phong['n_phat']}")
        print(f"  lenh SAI            {mo_phong['n_sai']}   "
              f"({mo_phong['ty_le_sai']*100:.2f}%)")
        print(f"  LENH SAI MOI PHUT   {mo_phong['lenh_sai_moi_phut']:.2f}"
              f"   <- day moi la con so quyet dinh")
        print(f"  take bat duoc       {mo_phong['take_bat_duoc']}")
        if mo_phong["tre_frame_trung_vi"] is not None:
            tf_ = mo_phong["tre_frame_trung_vi"]
            print(f"  tre den lenh dau    {tf_:.0f} frame = {tf_/fps:.2f} s")
        if mo_phong["sai_chi_tiet"]:
            print("  cac lenh sai:")
            for k_, v_ in list(mo_phong["sai_chi_tiet"].items())[:6]:
                print(f"    {k_:<44}{v_} lan")

        # ---- quet nguong theo LUAT PHAT LENH, khong theo frame ----------
        # Bang nguong o tren dem SAI THEO FRAME. Bang nay dem SAI THEO LENH.
        # Hai con so nay khac nhau rat xa vi luat k_vote doi k frame LIEN
        # TIEP cung y kien: nham rai rac thi bo dem khong bao gio day.
        # Chon diem lam viec phai nhin BANG NAY.
        print("\n  QUET NGUONG theo luat phat lenh:")
        print(f"    {'tau':>6}{'lenh phat':>11}{'lenh SAI':>10}{'sai/phut':>10}{'take bat duoc':>16}")
        for tt in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99):
            m = mo_phong_phat_lenh(Pc, y, groups, labels, tt, args.k_vote, fps)
            dau = "  <--" if abs(tt - tau) < 1e-9 else ""
            print(f"    {tt:>6.2f}{m['n_phat']:>11}{m['n_sai']:>10}"
                  f"{m['lenh_sai_moi_phut']:>10.2f}{m['take_bat_duoc']:>16}{dau}")

    # ══════════════════════════════════════════════════════════════════════
    #  BÁO CÁO THEO LỚP
    # ══════════════════════════════════════════════════════════════════════
    cm = None
    if P_oof is not None:
        pred = ap_nhiet_do(P_oof, T).argmax(1)
        all_idx = np.arange(n_class)
        rep = classification_report(y, pred, labels=all_idx, target_names=labels,
                                    output_dict=True, zero_division=0)
        print("\n" + "-" * 78)
        print("TUNG LOP (out-of-fold, toan bo tap)")
        print("-" * 78)
        print(f"  {'lop':<20}{'recall':>9}{'precision':>11}{'F1':>8}{'mau':>7}")
        for l in labels:
            r = rep[l]
            co = "  <- HE TRONG" if l in LOP_HE_TRONG else ""
            print(f"  {l:<20}{r['recall']*100:>8.1f}%{r['precision']*100:>10.1f}%"
                  f"{r['f1-score']:>8.3f}{int(r['support']):>7}{co}")

        cm = confusion_matrix(y, pred, labels=all_idx)
        print("\n  Ma tran nham lan (hang = that, cot = doan):")
        w = max(len(l) for l in labels)
        print(" " * (w + 4) + "".join(f"{l[:6]:>7}" for l in labels))
        for i, l in enumerate(labels):
            print(f"  {l:<{w+2}}" + "".join(
                f"{cm[i,j]:>7}" if i != j else f"{'.':>7}" for j in range(n_class)))

        print("\n  CAP DE NHAM - hai cu chi chi khac nhau o MOT dai luong:")
        for a, b, mo in CAP_DE_NHAM:
            if a in lut and b in lut:
                i, j = lut[a], lut[b]
                n_ab, n_ba = int(cm[i, j]), int(cm[j, i])
                tot = int(cm[i].sum()) + int(cm[j].sum())
                r = (n_ab + n_ba) / max(1, tot)
                trang = "TACH TOT" if r < 0.01 else "LAN NHAU"
                print(f"    {a:<18}<-> {b:<18}{n_ab + n_ba:>4} lan "
                      f"= {r*100:5.2f}%   {trang}   ({mo})")

        # ---- tung take ---------------------------------------------------
        bang_take = cham_tung_take(ap_nhiet_do(P_oof, T), y, groups, labels,
                                   tau if tau else 0.5, args.k_vote)
        yeu = sorted(bang_take, key=lambda r: (r[4], r[3]))
        cam = [r for r in yeu if r[4] < 0.02 and r[1] != "NEGATIVE"]
        print("\n  TUNG TAKE - 10 take yeu nhat (sap theo ty le phat lenh):")
        print(f"    {'take':<26}{'lop':<18}{'mau':>6}{'acc':>8}{'phat lenh':>11}")
        for gname, lab, n, acc, phat in yeu[:10]:
            co = ("   <-- CAM, khong bao gio phat lenh"
                  if phat < 0.02 and lab != "NEGATIVE" else
                  "   <-- yeu" if phat < 0.30 and lab != "NEGATIVE" else "")
            print(f"    {gname:<26}{lab:<18}{n:>6}{acc*100:>7.1f}%{phat*100:>10.1f}%{co}")
        if cam:
            print(f"\n    {len(cam)} take CAM HOAN TOAN. Day la nguyen nhan do phu thap,")
            print("    khong phai do nguong dat cao. Thu lai dung nhung take nay:")
            for gname, lab, n, acc, phat in cam:
                print(f"      {gname:<26}{lab}")

    # ══════════════════════════════════════════════════════════════════════
    #  MODEL CUỐI — train trên TOÀN BỘ dữ liệu
    # ══════════════════════════════════════════════════════════════════════
    # K-fold đã cho con số. Model đem đi dùng thì phải học từ MỌI take: bỏ phí
    # 1/k dữ liệu chỉ để giữ một tập test cho model cuối là không cần thiết khi
    # con số đã đo xong bằng out-of-fold.
    #
    # Không có tập val thì không dùng EarlyStopping được, nên lấy TRUNG VỊ số
    # epoch tốt nhất của các fold — nó đo trên dữ liệu cùng phân bố, cùng cỡ.
    print("\n" + "-" * 78)
    print("MODEL CUOI")
    print("-" * 78)
    n_ep = int(np.median(best_epochs)) if best_epochs else args.epochs
    Xall, yall = chuan_bi_train(X, y, n_class, args.seed, dung_mirror, labels,
                                args.no_augment)
    print(f"  train tren TOAN BO {len(Xall)} mau ({len(X)} goc), {n_ep} epoch")
    tf.keras.utils.set_random_seed(args.seed)
    model = BUILDERS[args.arch](n_class)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=args.label_smooth),
        metrics=["accuracy"])
    model.fit(Xall, tf.keras.utils.to_categorical(yall, n_class),
              epochs=n_ep, batch_size=args.batch, verbose=2,
              sample_weight=trong_so_mau(yall, n_class))
    print(f"  {model.count_params():,} tham so")

    # ---- đầu vào lạ, đo trên ĐÚNG model sắp xuất --------------------------
    # Bảng ngưỡng ở trên đo bằng các model của từng fold. Model xuất ra lại học
    # trên toàn bộ dữ liệu nên hành vi có thể khác. Đây là số của thứ THẬT SỰ
    # đem đi dùng.
    ood = None
    if tau is not None:
        Pf = ap_nhiet_do(model.predict(X_la, verbose=0), T)
        ood = {ten: ty_le_tu_choi(Pf[ho_la == i], labels, tau)
               for i, ten in enumerate(ten_la)}
        print(f"\n  Ty le TU CHOI dau vao la o tau={tau:.2f} (cang cao cang tot):")
        for k_, v_ in ood.items():
            co = ("" if v_ >= args.muc_tieu_la else
                  "  <-- THAP" if v_ >= 0.70 else "  <-- QUA THAP, nguy hiem")
            print(f"    {k_:<22}{v_*100:>6.1f}%{co}")
        if min(ood.values()) < 0.70:
            print("    Cach chua: thu them NEGATIVE, dac biet la cac tu the")
            print("    TRUNG GIAN tren duong dua tay vao vi tri.")

    # ══════════════════════════════════════════════════════════════════════
    #  XUẤT
    # ══════════════════════════════════════════════════════════════════════
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        stem = f"v3_{args.arch}_smoke"
    elif args.npz:
        stem = f"v3_{args.arch}_body25"
    else:
        stem = f"pck_{args.arch}_body25"
    if khong_an_toan:
        # Tên file tự khai ra. Không dựa vào trí nhớ của ai.
        stem += "_UNSAFE"

    # Luu du doan out-of-fold. Phan tich lai (doi tau, doi k_vote, soi mot
    # take cu the) khong con phai train lai K lan nua.
    if P_oof is not None:
        np.savez_compressed(out / f"{stem}.oof.npz", P=P_oof, y=y,
                            groups=groups, labels=np.array(labels),
                            temperature=np.array([T]))

    model.save(out / f"{stem}.keras")

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if not args.no_quant:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tfl = conv.convert()
    (out / f"{stem}.tflite").write_bytes(tfl)

    # ---- BẮT BUỘC: tflite phải khớp keras ----------------------------------
    # Lượng tử hoá trọng số là một phép ĐỔI MODEL. Không có gì trong quy trình
    # kiểm nó, và thứ thật sự chạy trên Pi là file .tflite chứ không phải .keras
    # đã đo ở trên. Nên đo lại ngay tại đây.
    itp = tf.lite.Interpreter(model_content=tfl)
    itp.allocate_tensors()
    di, do = itp.get_input_details()[0], itp.get_output_details()[0]
    n_thu = min(400, len(X))
    Pk = model.predict(X[:n_thu], verbose=0)
    Pt = np.zeros_like(Pk)
    for i in range(n_thu):
        itp.set_tensor(di["index"], X[i:i+1].astype(np.float32))
        itp.invoke()
        Pt[i] = itp.get_tensor(do["index"])[0]
    khop = float((Pk.argmax(1) == Pt.argmax(1)).mean())
    lech = float(np.abs(Pk - Pt).max())
    print(f"\n  tflite vs keras   nhan trung {khop*100:.2f}%   "
          f"lech xac suat lon nhat {lech:.4f}")
    if khop < 0.995:
        print("  !  LUONG TU HOA LAM DOI KET QUA. Xuat lai bang --no-quant,")
        print("     hoac chap nhan va do lai bang chinh file .tflite.")

    meta = {
        "labels": labels,
        "roster": args.roster,
        "input_shape": [N_KP, N_COORD],
        "normalization": "pck_format.normalize_body25",
        "arch": args.arch,
        "dataset": source,
        "data_fingerprint": fp,
        # Ba số dưới đây là HỢP ĐỒNG với tầng chạy thật. gesture_perception/
        # classifier.py đọc temperature + threshold, không cứng hoá ngưỡng.
        # k_vote chỉ còn để tham khảo: command_node dùng thời gian giữ theo giây.
        "temperature": float(T),
        "threshold": float(tau) if tau is not None else None,
        "k_vote": args.k_vote,
        "fps_do_duoc": float(fps),
        "cv_folds": k_fold,
        "cv_acc_mean": float(np.mean(acc_fold)) if acc_fold else None,
        "cv_acc_std": float(np.std(acc_fold)) if acc_fold else None,
        "mo_phong_phat_lenh": mo_phong,
        "tu_choi_dau_vao_la": ood,
        "tflite_khop_keras": khop,
        "an_toan": not khong_an_toan,
        "canh_bao": luu_y,
        "loi_chan_da_bo_qua": chan if khong_an_toan else [],
    }
    (out / f"{stem}.labels.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf8")

    print(f"\n  Da xuat vao {out}/")
    print(f"    {stem}.keras        "
          f"{(out/f'{stem}.keras').stat().st_size/1024:.0f} KB")
    print(f"    {stem}.tflite       {len(tfl)/1024:.0f} KB   <- chay tren Pi")
    print(f"    {stem}.labels.json  <- chua tau, T, k_vote. Tang chay PHAI doc.")

    print("\n" + "=" * 78)
    if khong_an_toan:
        print("  MODEL NAY KHONG AN TOAN (--force). Ten file da ghi _UNSAFE.")
        print("  Chi dung de kiem chuoi chay. Khong dem ra bay.")
    elif mo_phong:
        print(f"  Ket qua: {np.mean(acc_fold)*100:.1f}% theo frame, "
              f"{mo_phong['lenh_sai_moi_phut']:.2f} lenh sai/phut o tau={tau:.2f}")
    print("\n  Buoc tiep:  python training\\5_export_to_ros.py")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
