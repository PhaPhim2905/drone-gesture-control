#!/usr/bin/env python3
"""
CHECK_POSES — chấm điểm tư thế đã thu, theo tiêu chí hình học của TỪNG cử chỉ.

    python training/2_check_poses.py                 # cham toan bo
    python training/2_check_poses.py --lop TAKEOFF   # chi mot lop
    python training/2_check_poses.py --chi-tiet      # in them so do tho

Chạy ở MÔI TRƯỜNG CHÍNH. Không cần camera, không cần TensorFlow.

──────────────────────────────────────────────────────────────────────────────
VÌ SAO CẦN CÔNG CỤ NÀY

build_dataset.py trả lời "dữ liệu có dùng được không" — đủ mẫu chưa, đủ take
chưa, có mất khớp không. Nó KHÔNG trả lời "tư thế có đúng không".

Đã trả giá cho khoảng trống đó: ba vòng thu liên tiếp đều sai tư thế mà chỉ
phát hiện được khi ngồi phân tích thủ công. Vòng 1 làm động tác lặp thay vì
giữ tư thế (19% frame đúng). Vòng 2 khuỷu tay thấp hơn vai 0.5 đơn vị. Vòng 3
LAND chéo quá nông, tín hiệu chỉ gấp 2.2 lần nhiễu.

File này biến phép chấm đó thành một lệnh chạy được.

──────────────────────────────────────────────────────────────────────────────
ĐƠN VỊ

Mọi số đo quy về BỀ RỘNG VAI sau khi chuẩn hoá bằng pck_format. Nhờ vậy chúng
bất biến với khoảng cách tới camera và vị trí trong khung — so sánh được giữa
các take đứng xa gần khác nhau.

Ngưỡng trong TIEU_CHI suy từ hình học và từ số đo thật của các vòng thu trước.
Chúng là ngưỡng CẢNH BÁO, không phải luật: lệch một chút mà nhất quán vẫn tốt
hơn là đúng tâm nhưng dao động lớn.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# pck_format, settings, camera_util nam trong goi ROS gesture_perception -
# MOT ban duy nhat cho ca train lan luc bay.
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "gesture_perception" / "gesture_perception"))

import pck_format as pf   # noqa: E402

DATA_DIR = ROOT / "data" / "gestures"


# ══════════════════════════════════════════════════════════════════════════
#  ĐẶC TRƯNG HÌNH HỌC
# ══════════════════════════════════════════════════════════════════════════

def _angle(a, b, c):
    """Góc tại b, giữa đoạn b->a và b->c, tính bằng độ."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return np.nan
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


def features(N):
    """
    N: (n, 25, 2) đã chuẩn hoá  ->  dict tên đặc trưng -> mảng (n,)

    Mọi khoảng cách chia cho BỀ RỘNG VAI của chính frame đó.
    """
    sw = np.maximum(np.abs(N[:, pf.B25_LSHOULDER, 0] - N[:, pf.B25_RSHOULDER, 0]), 1e-9)
    sy = (N[:, pf.B25_LSHOULDER, 1] + N[:, pf.B25_RSHOULDER, 1]) / 2
    wy = (N[:, pf.B25_LWRIST, 1] + N[:, pf.B25_RWRIST, 1]) / 2
    ey = (N[:, pf.B25_LELBOW, 1] + N[:, pf.B25_RELBOW, 1]) / 2
    ny = N[:, pf.B25_NOSE, 1]

    aL = np.array([_angle(N[i, pf.B25_LSHOULDER], N[i, pf.B25_LELBOW],
                          N[i, pf.B25_LWRIST]) for i in range(len(N))])
    aR = np.array([_angle(N[i, pf.B25_RSHOULDER], N[i, pf.B25_RELBOW],
                          N[i, pf.B25_RWRIST]) for i in range(len(N))])

    def forearm_tilt(el, wr):
        v = N[:, wr] - N[:, el]
        return np.degrees(np.arctan2(np.abs(v[:, 0]), v[:, 1]))

    lw, rw = N[:, pf.B25_LWRIST, 0], N[:, pf.B25_RWRIST, 0]
    ls, rs = N[:, pf.B25_LSHOULDER, 0], N[:, pf.B25_RSHOULDER, 0]

    # --- ĐẶC TRƯNG THEO TỪNG BÊN -------------------------------------------
    # Bảy đặc trưng gốc ở dưới đều ĐỐI XỨNG: chúng lấy trung bình trái/phải
    # hoặc lấy trị tuyệt đối. Với sáu lớp roster v1 thì đủ, vì cả sáu đều là tư
    # thế hai tay giống nhau. Nhưng ROLL_LEFT và ROLL_RIGHT là ẢNH GƯƠNG của
    # nhau: mọi đặc trưng đối xứng cho ra CÙNG MỘT SỐ cho cả hai lớp. Chấm bằng
    # bảng cũ thì hai lớp này luôn cùng đạt hoặc cùng trượt, và ta không bao giờ
    # biết mình đã quay nhầm bên.
    #
    # "vươn ngang" đo theo HƯỚNG RA NGOÀI của chính bên đó, nên nó không phụ
    # thuộc vào việc ảnh có bị lật hay không — dương luôn nghĩa là cổ tay đã ra
    # ngoài khớp vai cùng bên.
    huong_P = np.sign(rs - ls)
    vuon_P = (rw - rs) * huong_P / sw
    vuon_T = (lw - ls) * (-huong_P) / sw

    lwy, rwy = N[:, pf.B25_LWRIST, 1], N[:, pf.B25_RWRIST, 1]

    return {
        "tay P vuon ngang": vuon_P,
        "tay T vuon ngang": vuon_T,
        "co tay P vs vai":  (rwy - sy) / sw,
        "co tay T vs vai":  (lwy - sy) / sw,
        "co tay P vs mui":  (rwy - ny) / sw,
        "co tay T vs mui":  (lwy - ny) / sw,
        "goc khuyu P":      aR,
        "goc khuyu T":      aL,
        "goc khuyu":        np.nanmean([aL, aR], axis=0),
        "khuyu vs vai":     (ey - sy) / sw,
        "cang tay lech":    np.nanmean([forearm_tilt(pf.B25_LELBOW, pf.B25_LWRIST),
                                        forearm_tilt(pf.B25_RELBOW, pf.B25_RWRIST)], axis=0),
        "co tay vs vai":    (wy - sy) / sw,
        "co tay vs mui":    (wy - ny) / sw,
        "hai co tay cach":  np.abs(lw - rw) / sw,
        "bat cheo":         (((lw - rw) * (ls - rs)) < 0).astype(float),
    }


# ══════════════════════════════════════════════════════════════════════════
#  TIÊU CHÍ TỪNG CỬ CHỈ
# ══════════════════════════════════════════════════════════════════════════
# (tên đặc trưng, cận dưới, cận trên, mô tả ngắn)   None = không giới hạn

TIEU_CHI = {
    "TAKEOFF": [
        ("goc khuyu",      65, 115, "khuyu gap vuong ~90 do"),
        ("khuyu vs vai",  -0.20, 0.20, "khuyu NGANG VAI"),
        ("cang tay lech",  None, 30, "cang tay DUNG THANG"),
        ("co tay vs vai",  0.10, None, "co tay CAO HON vai"),
        ("bat cheo",       None, 0.10, "khong bat cheo"),
    ],
    "HOVER": [
        ("goc khuyu",      145, None, "tay DUOI THANG"),
        ("co tay vs vai", -0.40, 0.40, "co tay NGANG VAI"),
        ("hai co tay cach", 3.0, None, "dang HET BIEN sang hai ben"),
        ("cang tay lech",  60, None, "cang tay NAM NGANG, khong dung"),
        ("bat cheo",       None, 0.10, "khong bat cheo"),
    ],
    "ASSUME_GUIDANCE": [
        ("goc khuyu",      145, None, "tay DUOI THANG"),
        ("co tay vs mui",  0.30, None, "co tay CAO HON MUI"),
        ("hai co tay cach", 0.7, 2.6, "chu V, khong chum cung khong qua rong"),
        ("bat cheo",       None, 0.10, "khong bat cheo"),
    ],
    "LAND": [
        ("bat cheo",       0.85, None, "BAT CHEO"),
        ("hai co tay cach", 0.80, None, "cheo SAU, hai co tay tach xa"),
        ("co tay vs vai",  None, -0.30, "co tay THAP HON vai"),
    ],
    "STOP": [
        ("bat cheo",       0.85, None, "BAT CHEO"),
        ("co tay vs mui",  0.20, None, "cheo TREN DAU"),
    ],
    # ROLL_* : chu L. Mot tay dang NGANG, tay kia gio THANG LEN, hai tay
    # vuong goc nhau. Tieu chi phai neu ro TUNG BEN, vi doi ben la doi lop.
    #
    # Ten theo huong CUA DRONE (chot 2026-09-14): ROLL_RIGHT = drone sang phai
    # cua no = sang TRAI cua nguoi dieu khien -> tay TRAI dang ngang chi huong.
    # Ban dau tieu chi viet nguoc lai; 23/23 take that deu "truot" dung doi xung.
    "ROLL_RIGHT": [
        ("goc khuyu T",    150, None, "tay TRAI duoi THANG"),
        ("tay T vuon ngang", 0.70, None, "tay TRAI vuon HET ra ngoai"),
        ("co tay T vs vai", -0.35, 0.35, "co tay TRAI NGANG VAI"),
        ("goc khuyu P",    150, None, "tay PHAI duoi THANG"),
        ("co tay P vs mui", 0.30, None, "co tay PHAI CAO HON MUI"),
    ],
    "ROLL_LEFT": [
        ("goc khuyu P",    150, None, "tay PHAI duoi THANG"),
        ("tay P vuon ngang", 0.70, None, "tay PHAI vuon HET ra ngoai"),
        ("co tay P vs vai", -0.35, 0.35, "co tay PHAI NGANG VAI"),
        ("goc khuyu T",    150, None, "tay TRAI duoi THANG"),
        ("co tay T vs mui", 0.30, None, "co tay TRAI CAO HON MUI"),
    ],
    "NEGATIVE": [],     # khong co tieu chi — cang da dang cang tot
}

# Cặp dễ nhầm: hai cử chỉ chỉ khác nhau ở MỘT đại lượng.
CAP_DE_NHAM = [
    ("TAKEOFF", "HOVER", "cang tay lech",
     "TAKEOFF cang tay DUNG (<30), HOVER cang tay NGANG (>60)"),
    ("TAKEOFF", "ASSUME_GUIDANCE", "goc khuyu",
     "TAKEOFF khuyu GAP (~90), ASSUME_GUIDANCE tay THANG (>145)"),
    ("LAND", "STOP", "co tay vs mui",
     "LAND cheo DUOI BUNG (am), STOP cheo TREN DAU (duong)"),
    # ROLL_* la nua HOVER cong nua ASSUME_GUIDANCE. Ba cap duoi day khong phai
    # ly thuyet: dung len tu the ASSUME_GUIDANCE bang cach gio LAN LUOT tung
    # tay thi cac frame o giua CHINH LA ROLL. Phai gio HAI TAY CUNG LUC.
    ("ROLL_RIGHT", "ROLL_LEFT", "co tay P vs mui",
     "ROLL_RIGHT tay PHAI gio cao (duong), ROLL_LEFT tay PHAI ngang (am)"),
    ("ROLL_RIGHT", "HOVER", "co tay P vs mui",
     "ROLL_RIGHT tay PHAI gio cao (>0.30), HOVER ca hai tay ngang vai"),
    ("ROLL_RIGHT", "ASSUME_GUIDANCE", "co tay T vs mui",
     "ROLL_RIGHT tay TRAI con NGANG (am), ASSUME_GUIDANCE ca hai tay qua dau"),
]


def load():
    rows = []
    for p in sorted(DATA_DIR.glob("*.jsonl")):
        for line in p.read_text(encoding="utf8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lop", default=None, help="chi cham mot lop")
    ap.add_argument("--chi-tiet", action="store_true", help="in them so do tho")
    args = ap.parse_args()

    rows = load()
    if not rows:
        print(f"  Khong co du lieu trong {DATA_DIR}")
        return 1

    by = defaultdict(list)
    for r in rows:
        by[(r["label"], r["take"])].append(r)

    print("=" * 84)
    print("CHECK_POSES - cham tu the theo tieu chi hinh hoc")
    print("=" * 84)

    trung_binh = {}     # lop -> {dac trung: gia tri trung vi}
    n_fail = 0

    for lab in sorted({k[0] for k in by}):
        if args.lop and lab != args.lop:
            continue
        takes = sorted(t for (l, t) in by if l == lab)
        print(f"\n  {lab}   {len(takes)} take, "
              f"{sum(len(by[(lab, t)]) for t in takes)} mau")
        print("  " + "-" * 80)

        # gop toan bo lop de lay trung vi
        allN = np.stack([pf.normalize_body25(np.array(r["kp"]), strict=False)
                         for t in takes for r in by[(lab, t)]])
        F = features(allN)
        trung_binh[lab] = {k: float(np.nanmedian(v)) for k, v in F.items()}

        crit = TIEU_CHI.get(lab)
        if crit is None:
            print("    (khong co tieu chi cho lop nay)")
            continue
        if not crit:
            std = allN.reshape(len(allN), -1).std(0).mean()
            print(f"    lop phu dinh - khong cham tu the.  do da dang std = {std:.4f}"
                  f"   {'tot' if std > 0.05 else 'nen da dang hon'}")
            continue

        print(f"    {'dac trung':<18}{'do duoc':>10}{'nguong':>18}   ket qua")
        for feat, lo, hi, mota in crit:
            v = float(np.nanmedian(F[feat]))
            ok = (lo is None or v >= lo) and (hi is None or v <= hi)
            n_fail += (not ok)
            ng = (f">= {lo}" if hi is None else
                  f"<= {hi}" if lo is None else f"{lo} .. {hi}")
            print(f"    {feat:<18}{v:>10.2f}{ng:>18}   "
                  f"{'DAT' if ok else 'CHUA DAT'}   {mota}")

        # ---- CHAM TUNG TAKE ------------------------------------------------
        # Trung vi cua CA LOP che mat take hong. Do duoc tren chinh du lieu
        # nay: TAKEOFF co trung vi 87.8 do (DAT), nhung hai trong sau take lai
        # o 111 va 122 do — mot cai da ra ngoai khoang 65..115. Chung keo lop
        # do lai gan ASSUME_GUIDANCE va sinh ra 5.08% nham lan, ma bang tren
        # bao "moi tieu chi DAT".
        xau = []
        for t in takes:
            Ft = features(np.stack([pf.normalize_body25(np.array(r["kp"]),
                                                        strict=False)
                                    for r in by[(lab, t)]]))
            for feat, lo, hi, mota in crit:
                v = float(np.nanmedian(Ft[feat]))
                if (lo is not None and v < lo) or (hi is not None and v > hi):
                    ng = (f">= {lo}" if hi is None else
                          f"<= {hi}" if lo is None else f"{lo} .. {hi}")
                    xau.append((t, feat, v, ng, mota))
        if xau:
            n_fail += len(xau)
            print(f"    TAKE LECH ({len(xau)} cho) - trung vi ca lop che mat:")
            for t, feat, v, ng, mota in xau:
                print(f"      {t:<24}{feat:<18}{v:>8.2f}   can {ng:<14}{mota}")
            print("      -> thu lai dung nhung take nay, hoac bo bang:")
            print("         python training\\3_build_dataset.py --drop-takes "
                  + ",".join(sorted({t for t, *_ in xau})))

        if args.chi_tiet:
            print(f"\n    {'dac trung':<18}{'trung vi':>10}{'10%':>9}{'90%':>9}")
            for k, v in F.items():
                v = v[~np.isnan(v)]
                print(f"    {k:<18}{np.median(v):>10.2f}"
                      f"{np.percentile(v,10):>9.2f}{np.percentile(v,90):>9.2f}")

    # ---- cac cap de nham -------------------------------------------------
    co_cap = [(a, b, f, m) for a, b, f, m in CAP_DE_NHAM
              if a in trung_binh and b in trung_binh]
    if co_cap:
        print("\n" + "=" * 84)
        print("CAC CAP DE NHAM - hai cu chi chi khac nhau o MOT dai luong")
        print("=" * 84)
        for a, b, feat, mota in co_cap:
            va, vb = trung_binh[a][feat], trung_binh[b][feat]
            gap = abs(va - vb)
            print(f"\n  {a}  vs  {b}   theo '{feat}'")
            print(f"    {a:<20}{va:>8.2f}")
            print(f"    {b:<20}{vb:>8.2f}")
            print(f"    khoang cach{'':<9}{gap:>8.2f}   {mota}")
            if feat == "goc khuyu" or feat == "cang tay lech":
                good = gap > 40          # do
            else:
                good = gap > 0.5         # don vi be rong vai
            print(f"    -> {'TACH DU RO' if good else 'QUA GAN, de nham'}")

    print("\n" + "=" * 84)
    if n_fail:
        print(f"  {n_fail} tieu chi CHUA DAT. Xem cot cuoi de biet phai sua gi.")
    else:
        print("  Moi tieu chi DAT.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
