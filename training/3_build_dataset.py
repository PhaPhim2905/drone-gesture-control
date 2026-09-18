#!/usr/bin/env python3
"""
BUILD_DATASET — phát lại dữ liệu thô đã thu, sinh tập huấn luyện.

    python training/3_build_dataset.py

Chạy ở MÔI TRƯỜNG CHÍNH (numpy + pck_format, không cần TensorFlow, không cần
camera). Đầu vào là data/gestures/*.jsonl do record_dataset.py ghi ra, đầu ra
là data/dataset_v3.npz.

    JSONL (25 khớp pixel)  ->  normalize_body25  ->  X (n, 25, 2)

Tách khỏi bước thu là có chủ đích: mỗi lần sửa phép chuẩn hoá trong
src/pck_format.py chỉ cần chạy lại file này, không phải thu lại dữ liệu.

──────────────────────────────────────────────────────────────────────────────
BA THỨ FILE NÀY KIỂM TRA, VÀ VÌ SAO

1. TỶ LỆ BỊ LOẠI theo từng lớp.
   normalize_body25(strict=True) loại mẫu có |giá trị| > 1.0. Nếu một lớp bị
   loại nhiều bất thường thì cử chỉ đó đang vượt ra ngoài quy ước chuẩn hoá của
   PCK — thường là do đứng quá gần camera nên tay ra ngoài khung.

2. SỐ TAKE mỗi lớp.
   Một lớp chỉ có một take thì con số accuracy đo được sau này vô nghĩa, dù nó
   có đẹp đến đâu. Xem giải thích trong record_dataset.py.

3. TRÙNG LẶP giữa các lớp.
   Nếu hai lớp có mẫu gần y hệt nhau thì đó là lỗi lúc thu (bấm nhầm lớp, hoặc
   hai cử chỉ thật sự quá giống nhau). Biết trước ở đây rẻ hơn nhiều so với
   phát hiện qua ma trận nhầm lẫn sau khi train xong.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# pck_format, settings, camera_util nam trong goi ROS gesture_perception -
# MOT ban duy nhat cho ca train lan luc bay.
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "gesture_perception" / "gesture_perception"))

import pck_format as pf   # noqa: E402

# Bo cu chi model cam ket nhan. Phai KHOP ROSTERS["v2"] trong 4_train.py —
# train se CHAN neu tap khong du bo nay (train lai 6 lop cu: --roster v1).
ROSTER_V1 = ["ASSUME_GUIDANCE", "HOVER", "TAKEOFF", "LAND", "STOP", "NEGATIVE",
             "ROLL_RIGHT", "ROLL_LEFT"]

DATA_DIR = ROOT / "data" / "gestures"
OUT = ROOT / "data" / "dataset_v3.npz"


def load_raw(paths):
    """Đọc mọi file jsonl -> danh sách bản ghi. Bỏ qua dòng cuối bị cắt."""
    rows, bad = [], 0
    for p in paths:
        for line in p.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  bo {bad} dong hong (thuong la dong cuoi khi thoat dot ngot)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-strict", action="store_true",
                    help="giu ca mau |gia tri| > 1.0 (mac dinh: loai)")
    ap.add_argument("--drop-takes", default="",
                    help="ma take can BO, cach nhau bang dau phay. "
                         "Vi du: --drop-takes 20260908_215719_1")
    args = ap.parse_args()

    paths = sorted(Path(args.dir).glob("*.jsonl"))
    if not paths:
        print(f"  Khong co file nao trong {args.dir}")
        print("  Chay truoc:  python training/1_record.py")
        # Không im lặng bỏ qua file .npz cũ: nó vẫn train được, vẫn ra model,
        # và không chỗ nào lộ ra rằng dữ liệu sinh ra nó đã bị xoá.
        old = Path(args.out)
        if old.exists():
            print(f"\n  CANH BAO: {old.name} van con, dung tu du lieu DA BI XOA.")
            print("            Dung train bang no. Xoa di cho chac:")
            print(f"                del {old}")
        return 1

    print("=" * 74)
    print("BUILD_DATASET")
    print("=" * 74)
    print(f"  {len(paths)} file phien:")
    for p in paths:
        print(f"    {p.name}")

    rows = load_raw(paths)
    print(f"\n  {len(rows)} ban ghi tho")

    # Bỏ take hỏng mà KHÔNG xoá dữ liệu thô. Bấm nhầm SPACE, đứng sai tư thế,
    # có người đi ngang — những chuyện đó xảy ra. Xoá dòng trong .jsonl thì mất
    # luôn, còn loại ở đây thì hồi lại được chỉ bằng cách bỏ cờ này đi.
    drop = {s.strip() for s in args.drop_takes.split(",") if s.strip()}
    if drop:
        before = len(rows)
        seen = {r.get("take") for r in rows}
        missing = drop - seen
        if missing:
            print(f"  CANH BAO: khong tim thay take {', '.join(sorted(missing))}")
        rows = [r for r in rows if r.get("take") not in drop]
        print(f"  bo {before - len(rows)} mau cua {len(drop & seen)} take: "
              f"{', '.join(sorted(drop & seen))}")

    X, y, groups, times = [], [], [], []
    n_seen, n_drop, n_nowrist = Counter(), Counter(), Counter()
    takes = defaultdict(set)

    for r in rows:
        lab = r["label"]
        n_seen[lab] += 1
        kp = np.asarray(r["kp"], dtype=np.float64)

        # Cùng cửa chặn với record_dataset.py, đặt lại ở đây để dữ liệu thu
        # TRƯỚC khi có cửa chặn đó cũng được lọc. Mẫu thiếu cổ tay ở lớp cử chỉ
        # không phải mẫu kém — nó là mẫu SAI: thứ định nghĩa cử chỉ không có
        # trong đó, nên nó dạy mạng liên hệ "vắng cổ tay" với nhãn đó.
        if lab != "NEGATIVE" and not (kp[pf.B25_LWRIST, 2] > 0
                                      and kp[pf.B25_RWRIST, 2] > 0):
            n_nowrist[lab] += 1
            continue

        n = pf.normalize_body25(kp, strict=not args.no_strict)
        if n is None:
            n_drop[lab] += 1
            continue
        X.append(n.astype(np.float32))
        y.append(lab)
        groups.append(r.get("take", "?"))
        times.append(float(r.get("t", 0.0)))
        takes[lab].add(r.get("take", "?"))

    if not X:
        print("\n  KHONG CON MAU NAO.")
        if sum(n_nowrist.values()):
            print(f"  Toan bo {sum(n_nowrist.values())} mau bi chan vi thieu co tay.")
            print("  Lui ra xa camera hoac ngua man hinh laptop len roi thu lai.")
        else:
            print("  Kiem tra lai du lieu tho.")
        return 1

    X = np.stack(X)
    y = np.array(y)
    groups = np.array(groups)
    times = np.array(times, dtype=np.float64)
    labels = sorted(set(y.tolist()))

    # ---- 1 & 2: bảng kiểm theo lớp -----------------------------------------
    print("\n" + "-" * 74)
    print(f"  {'lop':<18}{'giu':>7}{'thieu tay':>11}{'|max|>1':>9}{'take':>7}")
    print("-" * 74)
    warn = []
    for lab in sorted(set(list(labels) + list(n_seen))):
        keep = n_seen[lab] - n_drop[lab] - n_nowrist[lab]
        rate = n_drop[lab] / max(1, n_seen[lab])
        wrate = n_nowrist[lab] / max(1, n_seen[lab])
        nt = len(takes[lab])
        flag = ""
        if wrate > 0.10:
            flag += "  <- CO TAY RA NGOAI KHUNG"
            warn.append(f"{lab}: {wrate*100:.0f}% mau thieu co tay -> lui ra xa camera")
        if rate > 0.15:
            flag += "  <- loai nhieu"
            warn.append(f"{lab}: bi loai {rate*100:.0f}%, kiem tra lai lop nay")
        if nt < 4:
            # Dưới 4 take thì train_pck.folds_by_take không chia nổi val+test
            # cho lớp này — nó bị đẩy hết vào train và KHÔNG được đánh giá.
            flag += "  <- IT TAKE, khong danh gia duoc"
            warn.append(f"{lab}: chi co {nt} take, can toi thieu 4 (nen 8)")
        elif nt < 6:
            warn.append(f"{lab}: {nt} take -> chi 1 take test. Nen len 8 take.")
        if keep < 200:
            warn.append(f"{lab}: chi {keep} mau giu duoc, nen thu them")
        print(f"  {lab:<18}{keep:>7}{n_nowrist[lab]:>11}{n_drop[lab]:>9}{nt:>7}{flag}")

    # ---- 2b: còn thiếu cử chỉ nào của bộ sáu -------------------------------
    # Đặt ở đây chứ không để tới lúc train: người thu vừa đứng dậy khỏi chỗ
    # quay xong là chạy lệnh này. Biết ngay còn thiếu gì thì quay bù luôn,
    # chứ phát hiện lúc train là phải dựng máy, dựng đèn, đứng lại từ đầu.
    thieu = [l for l in ROSTER_V1 if l not in labels]
    thua = [l for l in labels if l not in ROSTER_V1]
    print(f"\n  Bo {len(ROSTER_V1)} cu chi (roster v2):")
    for l in ROSTER_V1:
        nt = len(takes[l])
        if l not in labels:
            print(f"    {l:<20}CHUA CO")
        else:
            print(f"    {l:<20}{nt} take"
                  f"{'   <- can them, toi thieu 4' if nt < 4 else ''}")
    if thieu:
        warn.append("con THIEU cu chi: " + ", ".join(thieu))
    if thua:
        warn.append("co lop NGOAI roster: " + ", ".join(thua)
                    + " -> train se chan, dung --roster none neu co y")

    # ---- 3: trùng lặp giữa các lớp -----------------------------------------
    # Với mỗi mẫu, tìm mẫu gần nhất KHÁC LỚP. Nếu khoảng cách bé hơn khoảng
    # cách trung bình TRONG lớp thì hai lớp đang chồng lên nhau.
    print("\n  Do chong lan giua cac lop (lay mau ngau nhien):")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), min(600, len(X)), replace=False)
    F = X[idx].reshape(len(idx), -1)
    L = y[idx]
    D = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    conf = Counter()
    for i in range(len(idx)):
        j = int(D[i].argmin())
        if L[j] != L[i]:
            conf[(L[i], L[j])] += 1
    # Tách hai loại chồng lấn — chúng có ý nghĩa NGƯỢC NHAU:
    #
    #   cử chỉ <-> NEGATIVE   BÌNH THƯỜNG. NEGATIVE theo định nghĩa phải giáp
    #                         ranh mọi lớp; nó bao trọn khoảng giữa các cử chỉ.
    #                         Bằng 0 mới đáng lo — nghĩa là NEGATIVE thu chưa đủ
    #                         gần vùng cử chỉ, và lúc bay sẽ không chặn được tư
    #                         thế trung gian.
    #
    #   cử chỉ <-> cử chỉ     ĐÁNG LO. Hai lệnh khác nhau mà hình giống nhau thì
    #                         model không có cách nào tách được, và drone nhận
    #                         nhầm lệnh này thành lệnh kia.
    g2g = {k: v for k, v in conf.items() if "NEGATIVE" not in k}
    g2n = {k: v for k, v in conf.items() if "NEGATIVE" in k}
    n_g2g = sum(g2g.values())
    tot = len(idx)
    if g2g:
        print(f"    CU CHI <-> CU CHI   {n_g2g}/{tot} = {n_g2g/tot*100:.1f}%  "
              f"<- {'chap nhan duoc' if n_g2g/tot < 0.02 else 'CAO, xem lai'}")
        for (a, b), c in sorted(g2g.items(), key=lambda t: -t[1])[:4]:
            print(f"      {a:<18}-> {b:<18}{c:>4} lan")
    else:
        print("    CU CHI <-> CU CHI   0  <- tot")
    if g2n:
        print(f"    cu chi <-> NEGATIVE {sum(g2n.values())}/{tot} = "
              f"{sum(g2n.values())/tot*100:.1f}%  <- binh thuong, NEGATIVE phai "
              f"giap ranh")
    elif "NEGATIVE" in labels:
        print("    cu chi <-> NEGATIVE 0  <- DANG NGO: NEGATIVE chua bao duoc "
              "vung giap ranh cu chi")

    # ---- xuất ---------------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # t = moc thoi gian tung mau. train_pck.py dung no de DO nhip vong lap
    # thay vi doan, roi quy doi ty le nham thanh 'lenh sai moi phut'.
    np.savez_compressed(out, X=X, y=y, groups=groups, t=times,
                        labels=np.array(labels))

    print("\n" + "=" * 74)
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out            # --out tro ra ngoai cay du an
    print(f"  Da ghi {shown}")
    print(f"    X {X.shape}   {len(labels)} lop   {len(set(groups.tolist()))} take")
    if warn:
        print("\n  CAN LUU Y:")
        for w in warn:
            print(f"    - {w}")
    print("\n  Buoc tiep:")
    print("    .venv-train/Scripts/python.exe training/4_train.py "
          "--npz data/dataset_v3.npz --arch pck")
    print("    (dung --arch pck cho lan dau: kien truc nay khong co BatchNorm")
    print("     nen khong sap khi tap con nho. Doi sang --arch hybrid khi da co")
    print("     nhieu take va muon so sanh do ben.)")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
