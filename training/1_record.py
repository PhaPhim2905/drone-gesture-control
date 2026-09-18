#!/usr/bin/env python3
"""
RECORD_DATASET — thu bộ dữ liệu cử chỉ bằng camera, ở MÔI TRƯỜNG CHÍNH.

    python training/1_record.py

Không cần TensorFlow. Chỉ cần opencv + mediapipe + numpy, đúng bộ đang chạy.

──────────────────────────────────────────────────────────────────────────────
GHI DỮ LIỆU THÔ, KHÔNG GHI ĐẶC TRƯNG

File này ghi 25 khớp ở toạ độ PIXEL kèm độ tin cậy, chứ KHÔNG ghi vector 50
chiều đã chuẩn hoá. Lý do: phép chuẩn hoá còn có thể đổi (đổi mốc, đổi cách
tính scale, đổi ngưỡng visibility). Ghi thô thì đổi xong chỉ cần chạy lại
build_dataset.py; ghi đặc trưng thì phải mời người ra đứng thu lại từ đầu.

──────────────────────────────────────────────────────────────────────────────
KHÁI NIỆM "TAKE" — và vì sao nó quyết định con số accuracy có thật hay không

Camera 30 FPS: hai frame liền nhau của cùng một người đứng yên gần như y hệt.
Nếu trộn tất cả rồi chia ngẫu nhiên train/test, frame 101 vào train và frame
102 vào test — model chỉ cần NHỚ chứ không cần HỌC, và ta đọc được 99.9% hoàn
toàn giả.

Mỗi lần bấm SPACE là một TAKE. build_dataset ghi lại take của từng mẫu, và
train chia tập theo TAKE chứ không theo mẫu. Muốn số đo có ý nghĩa thì mỗi lớp
phải có NHIỀU take, mỗi take một điều kiện khác: đứng gần / đứng xa, lệch
trái / lệch phải, xoay người 20-30 độ, đổi áo, đổi nền, đổi đèn.

    5 take x 100 mẫu   >>>   1 take x 500 mẫu

──────────────────────────────────────────────────────────────────────────────
PHÍM

    1..9, 0   chọn lớp (theo thứ tự hiện trên màn hình)
    SPACE     bắt đầu một take (đếm ngược rồi tự ghi, tự dừng)
    u         xoá take vừa thu (chỉ trong phiên này)
    q / ESC   thoát

──────────────────────────────────────────────────────────────────────────────
ĐẾM NGƯỢC VÀ TỰ DỪNG — vì sao phải có

Người thu đứng cách máy 2-4 m. Không có tay nào rảnh để bấm SPACE khi hai tay
đang giữ tư thế. Bản đầu của file này không tính đến chuyện đó, và dữ liệu thu
được dính đúng cái lỗi ấy: một take chứa cả đoạn đi ra vị trí và đoạn đi vào
bấm dừng. Đo được trên phiên thật — một take 5 giây toàn cảnh mặt người sát
camera (lúc đang ở bàn phím), và một take khác có góc khuỷu 41 độ ở đoạn đầu
rồi mới ổn định về 70 độ.

Nên vòng đời một take giờ là:

    SPACE  ->  ĐẾM NGƯỢC --delay giây  ->  GHI --duration giây  ->  tự dừng
               (đi ra, vào tư thế)         (đứng yên trong tư thế)

Bíp báo: 3 tiếng ngắn ở 3 giây cuối đếm ngược, 1 tiếng cao dài khi bắt đầu
ghi, 2 tiếng thấp khi dừng. Ở 4 m thì tai nghe được còn mắt đọc chữ không nổi,
mà cũng không thể vừa giữ tư thế vừa nhìn xuống màn hình.
"""

import argparse
import json
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit(
        "\n  THIEU cv2 -> dang chay NHAM python.\n\n"
        "  File nay chay o MOI TRUONG CHINH (co camera):\n"
        "      python training/1_record.py\n\n"
        "  .venv-train KHONG co cv2 va mediapipe. Do la co y, khong phai thieu\n"
        "  sot: no chi chua TensorFlow de train. Hai moi truong tach roi nhau\n"
        "  chinh la thu bao dam TensorFlow khong keo do numpy/mediapipe cua\n"
        "  moi truong chay that.\n\n"
        "      thu / dung tap / xem realtime  ->  python\n"
        "      train                          ->  .venv-train/Scripts/python.exe\n"
    )

ROOT = Path(__file__).resolve().parent.parent
# pck_format, settings, camera_util nam trong goi ROS gesture_perception -
# MOT ban duy nhat cho ca train lan luc bay.
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "gesture_perception" / "gesture_perception"))

import pck_format as pf          # noqa: E402
import settings as C            # noqa: E402
import camera_util as cu         # noqa: E402

DATA_DIR = ROOT / "data" / "gestures"
VIDEO_DIR = ROOT / "data" / "video"

# Danh mục đầy đủ. Chín lớp đầu là cử chỉ, lớp cuối NEGATIVE bắt buộc phải có
# (xem ghi chú cuối file). Phím chọn lớp chạy theo THỨ TỰ trong danh sách này.
CATALOG = [
    ("ASSUME_GUIDANCE", "hai tay gio THANG len qua dau"),
    ("HOVER",           "hai tay dang NGANG bang vai (chu T)"),
    ("TAKEOFF",         "hai tay dang ngang, GAP KHUYU 90 do (chu U)"),
    ("MOVE_UP",         "hai tay dang ngang, chech LEN ~45 do"),
    ("MOVE_DOWN",       "hai tay dang ngang, chech XUONG ~45 do"),
    ("MOVE_LEFT",       "tay TRAI dang ngang, tay phai buong xuoi"),
    ("MOVE_RIGHT",      "tay PHAI dang ngang, tay trai buong xuoi"),
    ("LAND",            "hai tay bat CHEO truoc BUNG"),
    ("STOP",            "hai tay bat CHEO tren DAU"),
    # Tên theo hướng CỦA DRONE (đối diện người điều khiển). Tay dang ngang chỉ
    # về phía drone phải đi, nhìn từ phía người điều khiển.
    ("ROLL_RIGHT",      "tay TRAI dang NGANG, tay PHAI gio THANG (drone sang TRAI cua anh)"),
    ("ROLL_LEFT",       "tay PHAI dang NGANG, tay TRAI gio THANG (drone sang PHAI cua anh)"),
    ("NEGATIVE",        "moi thu KHAC: dung yen, gai dau, cam dt, di lai"),
]
CATALOG_NAMES = [lab for lab, _ in CATALOG]

# Phím chọn lớp: 1..9, 0, rồi a..f. Tránh 'q' (thoát) và 'u' (xoá take).
KEYS = "1234567890abcdef"
assert len(CATALOG) <= len(KEYS), "them lop thi phai noi rong KEYS"

# ══════════════════════════════════════════════════════════════════════════
#  CỬA CHẶN: KHÔNG THẤY CỔ TAY THÌ KHÔNG GHI
# ══════════════════════════════════════════════════════════════════════════
# Đo được ở phiên thu thử đầu tiên (233 mẫu ASSUME_GUIDANCE, webcam laptop):
#
#     co tay phai  nhin thay  42.1% so frame
#     co tay trai  nhin thay  42.5%
#     vi tri co tay trung binh sau chuan hoa:  y = -0.039  (tuc NGANG VAI)
#
# Giơ tay quá đầu ở cự ly webcam laptop thì cổ tay ra ngoài khung. MediaPipe
# hạ visibility, pck_format đặt khớp về (0,0), và mẫu ghi được KHÔNG CÒN chứa
# thứ định nghĩa cử chỉ. Train bằng tập đó thì mạng học đúng một luật:
# "không thấy cổ tay => ASSUME_GUIDANCE" — mà đó là lệnh MỞ QUYỀN ĐIỀU KHIỂN.
#
# Vì vậy: tám lớp cử chỉ BẮT BUỘC thấy đủ hai cổ tay mới ghi. Riêng NEGATIVE
# thì không, vì "khuất tay" chính là một phần của cái mà NEGATIVE phải bao.
NEED_WRISTS = set(CATALOG_NAMES) - {"NEGATIVE"}

# Khung xương để vẽ ô xem trước hệ đã chuẩn hoá.
LINKS = [
    (pf.B25_NOSE, pf.B25_NECK), (pf.B25_NECK, pf.B25_MIDHIP),
    (pf.B25_NECK, pf.B25_RSHOULDER), (pf.B25_RSHOULDER, pf.B25_RELBOW),
    (pf.B25_RELBOW, pf.B25_RWRIST),
    (pf.B25_NECK, pf.B25_LSHOULDER), (pf.B25_LSHOULDER, pf.B25_LELBOW),
    (pf.B25_LELBOW, pf.B25_LWRIST),
    (pf.B25_MIDHIP, pf.B25_RHIP), (pf.B25_RHIP, pf.B25_RKNEE),
    (pf.B25_RKNEE, pf.B25_RANKLE),
    (pf.B25_MIDHIP, pf.B25_LHIP), (pf.B25_LHIP, pf.B25_LKNEE),
    (pf.B25_LKNEE, pf.B25_LANKLE),
]


def existing_counts():
    """Đếm mẫu và take đã có trong data/gestures/, để biết còn thiếu lớp nào."""
    n, takes = Counter(), Counter()
    seen = set()
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # dòng cuối bị cắt do thoát đột ngột
            n[r["label"]] += 1
            key = (r["label"], r.get("take"))
            if key not in seen:
                seen.add(key)
                takes[r["label"]] += 1
    return n, takes


def draw_norm_panel(img, kp_norm, x0, y0, size=170):
    """Vẽ tư thế ĐÃ CHUẨN HOÁ. Đây là thứ model thật sự nhìn thấy.

    Ô này không phải trang trí: nó là cách duy nhất thấy bằng mắt rằng phép
    đổi hệ toạ độ chạy đúng — y đã lật (giơ tay lên thì điểm đi LÊN trong ô),
    và người luôn nằm gọn trong ô bất kể đứng gần hay xa camera.
    """
    cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (60, 60, 60), 1)
    cv2.line(img, (x0, y0 + size // 2), (x0 + size, y0 + size // 2), (45, 45, 45), 1)
    cv2.line(img, (x0 + size // 2, y0), (x0 + size // 2, y0 + size), (45, 45, 45), 1)
    if kp_norm is None:
        return

    def P(i):
        # hệ chuẩn hoá: y DƯƠNG là hướng lên -> đảo lại khi vẽ lên ảnh
        return (int(x0 + size / 2 + kp_norm[i, 0] * size / 2),
                int(y0 + size / 2 - kp_norm[i, 1] * size / 2))

    vis = np.abs(kp_norm).sum(1) > 1e-9
    for a, b in LINKS:
        if vis[a] and vis[b]:
            cv2.line(img, P(a), P(b), (0, 200, 255), 1, cv2.LINE_AA)
    for i in range(pf.N_BODY25):
        if vis[i]:
            cv2.circle(img, P(i), 2, (255, 255, 255), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=float, default=0.10,
                    help="giay giua hai mau duoc ghi (mac dinh 0.10 = 10 Hz)")
    ap.add_argument("--target", type=int, default=500,
                    help="so mau muc tieu moi lop")
    ap.add_argument("--cam", type=int, default=C.CAM_INDEX)
    ap.add_argument("--backend", default=C.CAM_BACKEND,
                    choices=("auto", "dshow", "msmf", "any"),
                    help="tren Windows, dshow va msmf danh so camera KHAC NHAU")
    ap.add_argument("--min-move", type=float, default=0.004,
                    help="mau giong het mau truoc thi BO. 0 = ghi tat ca")
    ap.add_argument("--classes", default=None,
                    help="chi thu MOT SO lop, cach nhau bang dau phay. "
                         "Vi du: --classes TAKEOFF,LAND")
    ap.add_argument("--delay", type=float, default=8.0,
                    help="giay dem nguoc sau khi bam SPACE, du de di ra vi tri "
                         "va vao tu the. 0 = ghi ngay")
    ap.add_argument("--duration", type=float, default=15.0,
                    help="giay moi take, tu dong dung. 0 = bam SPACE de dung")
    ap.add_argument("--save-video", action="store_true",
                    help="luu them video tho moi take vao data/video/. "
                         "Cho phep trich lai dac trung sau nay bang "
                         "training/3b_build_from_video.py")
    args = ap.parse_args()

    if args.classes:
        want = [s.strip().upper() for s in args.classes.split(",") if s.strip()]
        bad = [s for s in want if s not in CATALOG_NAMES]
        if bad:
            print(f"  Khong co lop: {', '.join(bad)}")
            print(f"  Cac lop hop le: {', '.join(CATALOG_NAMES)}")
            return 1
        if len(want) > len(KEYS):
            print(f"  Nhieu nhat {len(KEYS)} lop mot phien.")
            return 1
        CLASSES = [c for c in CATALOG if c[0] in want]
        CLASSES.sort(key=lambda c: want.index(c[0]))   # giu thu tu nguoi dung go
    else:
        CLASSES = list(CATALOG)

    import mediapipe as mp
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    n_have, t_have = existing_counts()
    print("=" * 70)
    print("RECORD_DATASET")
    print("=" * 70)
    print("  du lieu da co trong data/gestures/:")
    for lab, _ in CLASSES:
        print(f"    {lab:<18}{n_have[lab]:>5} mau / {t_have[lab]:>2} take")
    print(f"\n  muc tieu: {args.target} mau moi lop, it nhat 4-5 take moi lop")
    keyhelp = KEYS[0] if len(CLASSES) == 1 else f"{KEYS[0]}..{KEYS[len(CLASSES)-1]}"
    print(f"  SPACE bat/tat thu, {keyhelp} chon lop, u xoa take vua thu, q thoat\n")

    cap, be = cu.open_camera(args.cam, args.backend, C.CAM_WIDTH, C.CAM_HEIGHT)
    if cap is None:
        print(cu.fail_message(args.cam, args.backend))
        return 1
    print(f"  camera index {args.cam} qua backend {be}\n")

    pose = mp.solutions.pose.Pose(
        model_complexity=C.POSE_MODEL_COMPLEXITY,
        smooth_landmarks=C.POSE_SMOOTH_LANDMARKS,
        min_detection_confidence=C.POSE_DETECT_CONF,
        min_tracking_confidence=C.POSE_TRACK_CONF,
    )
    drawer = mp.solutions.drawing_utils

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DATA_DIR / f"session_{stamp}.jsonl"
    fh = open(path, "a", encoding="utf8", buffering=1)

    idx = 0                 # lớp đang chọn
    recording = False
    take_no = 0
    take_start_offset = 0   # vị trí byte đầu take, dùng cho phím undo
    take_count = 0
    last_write = 0.0
    last_vec = None
    skipped = 0
    no_wrist = 0
    session_n = Counter()

    writer = None            # VideoWriter cua take dang thu, None neu tat
    video_path = None
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_fps = cam_fps if 5.0 < cam_fps < 120.0 else 20.0
    if args.save_video:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  luu video vao data/video/ ({vid_fps:.0f} fps)\n")

    def close_writer():
        nonlocal writer
        if writer is not None:
            writer.release()
            writer = None

    # ---- tieng bip -------------------------------------------------------
    # Dung o cach may 2-4 m thi khong doc noi chu tren man hinh, va cung khong
    # the vua giu tu the vua nhin xuong. Tai thi van nghe duoc. Bip chay trong
    # thread rieng vi winsound.Beep CHAN, ma chan 300 ms la mat 6 frame.
    try:
        import winsound

        def beep(freq, ms):
            threading.Thread(target=winsound.Beep, args=(freq, ms),
                             daemon=True).start()
    except ImportError:
        def beep(freq, ms):
            print("\a", end="", flush=True)

    counting = False        # dang dem nguoc, CHUA ghi
    t_phase = 0.0           # moc thoi gian cua pha hien tai
    last_tick = -1          # giay dem nguoc da bip lan cuoi
    loop_t = deque(maxlen=60)   # moc thoi gian vai giay gan nhat, de do fps that

    def start_take():
        """SPACE lan 1: bat dau dem nguoc (hoac ghi ngay neu --delay 0)."""
        nonlocal counting, recording, t_phase, take_count, last_vec, last_tick
        nonlocal take_start_offset
        fh.flush()
        take_start_offset = path.stat().st_size
        take_count = 0
        last_vec = None
        last_tick = -1
        t_phase = time.time()
        if args.delay > 0:
            counting = True
            recording = False
            print(f"  ... dem nguoc {args.delay:.0f}s -> take {take_no}, "
                  f"lop {label}")
        else:
            counting = False
            begin_recording()

    def begin_recording():
        """Het dem nguoc: mo VideoWriter va bat dau ghi that."""
        nonlocal recording, counting, t_phase, writer, video_path
        counting = False
        recording = True
        t_phase = time.time()
        beep(1400, 250)
        if args.save_video:
            # fps GHI VAO FILE phai la nhip THAT cua vong lap, khong phai fps
            # danh nghia cua camera. Camera bao 20 fps, nhung moi vong con phai
            # chay MediaPipe (~30 ms) va ve khung xuong, nen thuc te chi ~7.5
            # fps. Ghi nhan 20 thi video 15 giay phat het trong 5.7 giay — nhanh
            # 2.7 lan. Do duoc tren file that truoc khi sua.
            #
            # Sai fps khong lam hong toa do khop (du lieu that nam o .jsonl),
            # nhung build_from_video.py lay mau theo THOI GIAN VIDEO — sai fps
            # thi mat do lay mau sai theo dung ty le do.
            #
            # Doan dem nguoc la cua so do mien phi: no vua chay xong dung nhip
            # nay, voi dung tai nay.
            fps_do = (1.0 / float(np.median(np.diff(loop_t)))
                      if len(loop_t) >= 8 else vid_fps)
            fps_do = float(np.clip(fps_do, 1.0, 60.0))
            video_path = VIDEO_DIR / f"{label}__{stamp}_{take_no}.mp4"
            writer = cv2.VideoWriter(str(video_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_do, (w, h))
            print(f"      (video {fps_do:.1f} fps do duoc)")
            if not writer.isOpened():
                print("  CANH BAO: khong mo duoc VideoWriter, bo qua video")
                writer = None
        print(f"  ... DANG GHI take {take_no}, lop {label}")

    def stop_take(auto=False):
        nonlocal recording, counting, take_no
        was = recording
        recording = counting = False
        close_writer()
        if was:
            beep(700, 120); beep(500, 200)
            print(f"  take {take_no} [{label}] xong: {take_count} mau"
                  f"{'  (tu dong dung)' if auto else ''}")
            take_no += 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if C.FLIP_FRAME:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Bản SẠCH, chụp TRƯỚC khi vẽ khung xương và chữ lên. Video phải là
        # cảnh thật, không có vạch vẽ đè lên người — nếu không thì trích lại
        # đặc trưng từ video sau này sẽ ra kết quả khác lúc thu.
        if writer is not None:
            writer.write(frame.copy())

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        kp = kp_norm = None
        if res.pose_landmarks:
            drawer.draw_landmarks(frame, res.pose_landmarks,
                                  mp.solutions.pose.POSE_CONNECTIONS)
            # ⚠️ phải nhân w,h — xem cảnh báo trong pck_format.mediapipe_to_body25
            kp = pf.mediapipe_to_body25(res.pose_landmarks.landmark, w, h,
                                        C.LM_MIN_VISIBILITY)
            kp_norm = pf.normalize_body25(kp, strict=False)

        label = CLASSES[idx][0]
        now = time.time()
        loop_t.append(now)      # de do nhip THAT cua vong lap, xem begin_recording

        wrists_ok = kp is not None and (kp[pf.B25_LWRIST, 2] > 0
                                        and kp[pf.B25_RWRIST, 2] > 0)
        blocked = (label in NEED_WRISTS) and not wrists_ok

        # ---- may trang thai: dem nguoc -> ghi -> tu dong dung ---------------
        if counting:
            left = args.delay - (now - t_phase)
            tick = int(np.ceil(left))
            if 0 < tick <= 3 and tick != last_tick:
                beep(900, 90)          # 3 tieng ngan bao sap ghi
                last_tick = tick
            if left <= 0:
                begin_recording()
        elif recording and args.duration > 0 and now - t_phase >= args.duration:
            stop_take(auto=True)

        # ---- ghi mẫu ------------------------------------------------------
        wrote = False
        if recording and blocked:
            no_wrist += 1
        elif recording and kp is not None and now - last_write >= args.period:
            vec = None if kp_norm is None else kp_norm.reshape(-1)
            moved = (last_vec is None or vec is None or
                     float(np.abs(vec - last_vec).max()) >= args.min_move)
            if moved:
                rec = {
                    "label": label, "take": f"{stamp}_{take_no}",
                    "t": round(now, 3), "w": w, "h": h,
                    "kp": [[round(float(v), 2) for v in p] for p in kp],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                last_write = now
                last_vec = vec
                take_count += 1
                session_n[label] += 1
                wrote = True
            else:
                skipped += 1     # đứng yên tuyệt đối -> mẫu trùng, không ghi

        # ---- bảng thông tin bên phải ---------------------------------------
        panel = np.zeros((h, 330, 3), np.uint8)
        cv2.putText(panel, f"LOP  ({KEYS[0]}..{KEYS[len(CLASSES)-1]})", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        for i, (lab, _) in enumerate(CLASSES):
            tot = n_have[lab] + session_n[lab]
            y = 45 + i * 22
            on = (i == idx)
            if on:
                col = (0, 255, 255)
            elif tot >= args.target:
                col = (0, 220, 0)
            else:
                col = (150, 150, 150)
            cv2.putText(panel, f"{KEYS[i]} {lab:<16}{tot:>4}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2 if on else 1)
            bw = int(min(1.0, tot / max(1, args.target)) * 300)
            cv2.rectangle(panel, (10, y + 4), (10 + bw, y + 6), col, -1)

        cv2.putText(panel, CLASSES[idx][1][:46], (10, 45 + len(CLASSES) * 22 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        draw_norm_panel(panel, kp_norm, 80, 285)

        if kp_norm is None:
            st, col = "KHONG THAY NGUOI", (0, 0, 255)
        else:
            mx = float(np.abs(kp_norm).max())
            st = f"|max| = {mx:.2f}"
            col = (0, 220, 0) if mx <= 1.0 else (0, 165, 255)
            if mx > 1.0:
                st += "  > 1.0 -> se bi LOAI"
        cv2.putText(panel, st, (10, min(h - 6, 478)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        if kp is not None:
            wl = kp[pf.B25_LWRIST, 2] > 0
            wr = kp[pf.B25_RWRIST, 2] > 0
            wtxt = f"co tay T {'OK' if wl else '--'}   co tay P {'OK' if wr else '--'}"
            wcol = (0, 220, 0) if (wl and wr) else (0, 0, 255)
            cv2.putText(panel, wtxt, (10, min(h - 28, 456)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, wcol, 1)

        if blocked:
            # Chặn ở đây rẻ hơn nhiều so với phát hiện sau khi đã train xong.
            cv2.rectangle(frame, (0, h // 2 - 40), (w, h // 2 + 24), (0, 0, 140), -1)
            cv2.putText(frame, "CO TAY RA NGOAI KHUNG", (20, h // 2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            cv2.putText(frame, "lui ra xa / ngua man hinh len - KHONG GHI",
                        (20, h // 2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 200, 255), 1)

        # Chu phai TO. Nguoi dung dung cach may 2-4 m, chu 0.6 doc khong noi.
        if counting:
            left = max(0.0, args.delay - (now - t_phase))
            txt = str(int(np.ceil(left)))
            sz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 6.0, 12)[0]
            cv2.putText(frame, txt, ((w - sz[0]) // 2, (h + sz[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 200, 255), 12)
            cv2.putText(frame, "VAO TU THE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)
        elif recording:
            left = (args.duration - (now - t_phase)) if args.duration > 0 else 0
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            cv2.circle(frame, (32, 34), 13, (0, 0, 255), -1)
            head = (f"GHI  {left:4.1f}s" if args.duration > 0
                    else f"GHI  {take_count} mau")
            cv2.putText(frame, head, (55, 46), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)
            if args.duration > 0:   # thanh tien do chay het chieu ngang khung
                cv2.rectangle(frame, (0, h - 10), (int(w * (1 - left / args.duration)),
                              h), (0, 0, 255), -1)
        else:
            cv2.putText(frame, "SPACE de bat dau", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 3)
        cv2.putText(frame, label, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)

        cv2.imshow("record_dataset", np.hstack([frame, panel]))

        # ---- phím ---------------------------------------------------------
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif 0 <= KEYS.find(chr(k) if 32 <= k < 127 else "\0") < len(CLASSES):
            if recording or counting:       # đổi lớp thì kết thúc take hiện tại
                stop_take()
                counting = False
            idx = KEYS.find(chr(k))
        elif k == ord(" "):
            if recording or counting:
                stop_take()
                counting = False
            else:
                start_take()
        elif k == ord("u"):
            # cắt file về đúng vị trí trước take vừa rồi
            recording = counting = False
            close_writer()
            fh.close()
            with open(path, "r+b") as f:
                f.truncate(take_start_offset)
            fh = open(path, "a", encoding="utf8", buffering=1)
            session_n[label] -= take_count
            if video_path is not None and video_path.exists():
                video_path.unlink()
                video_path = None
            print(f"  da xoa take vua thu ({take_count} mau)")
            take_count = 0

    close_writer()
    fh.close()
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    print("\n" + "=" * 70)
    print(f"  ghi vao data/gestures/{path.name}")
    for lab, _ in CLASSES:
        if session_n[lab]:
            print(f"    {lab:<18}+{session_n[lab]:>4} mau")
    if skipped:
        print(f"  bo {skipped} frame trung lap (dung yen tuyet doi)")
    if no_wrist:
        print(f"  CHAN {no_wrist} frame vi khong thay du hai co tay")
        print("    -> lui ra xa hoac ngua man hinh laptop len cho camera nhin cao hon")
    if path.stat().st_size == 0:
        path.unlink()
        print("  phien rong -> da xoa file")
    if args.save_video:
        vids = sorted(VIDEO_DIR.glob(f"*__{stamp}_*.mp4"))
        mb = sum(v.stat().st_size for v in vids) / 1e6
        print(f"  {len(vids)} video trong data/video/  ({mb:.0f} MB)")
    print("\n  Buoc tiep:  python training/3_build_dataset.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ══════════════════════════════════════════════════════════════════════════
#  VÌ SAO BẮT BUỘC PHẢI CÓ LỚP NEGATIVE
# ══════════════════════════════════════════════════════════════════════════
# Đo được trên chính model 20 lớp đã train (test_perception_core.py):
#
#     tư thế thật        entropy trung bình = 0.553
#     25 điểm ngẫu nhiên entropy trung bình = 0.183   <- THẤP HƠN
#
# Tức là đưa rác vào thì mạng trả lời TỰ TIN HƠN là đưa tư thế thật. Softmax
# luôn phải cộng lại bằng 1: nó chỉ biết "giống lớp nào nhất", không có chỗ để
# nói "không giống lớp nào cả". Vì vậy KHÔNG thể dùng ngưỡng entropy hay ngưỡng
# confidence làm cửa chặn đầu vào lạ.
#
# Cách duy nhất chắc chắn là DẠY nó lớp "không phải cử chỉ nào" bằng dữ liệu
# thật. Lớp NEGATIVE cần ĐA DẠNG hơn hẳn tám lớp kia — đứng yên, đi lại, gãi
# đầu, cầm điện thoại, khoanh tay, cúi nhặt đồ, xoay lưng, một tay giơ nửa
# chừng (tư thế TRUNG GIAN trên đường đưa tay từ cử chỉ này sang cử chỉ khác —
# đây mới là nhóm nguy hiểm nhất, vì nó xuất hiện trong MỌI lần đổi lệnh).
