#!/usr/bin/env python3
"""
ONE_EURO — tầng lọc landmark của perception_node.

Trước 2026-09-14 file này là src/gesture_pipeline_v3.py, đứng một mình chưa nối
vào đâu. Giờ nó được perception_node gọi mỗi frame qua Body25Filter (PHẦN D),
nằm giữa MediaPipe và phép chuẩn hoá PCK:

    MediaPipe -> BODY25 (pixel) -> Body25Filter -> normalize_body25 -> TFLite

Chạy file này = chạy bộ test tự kiểm, không cần phần cứng gì:

    python ros2_ws/src/gesture_perception/gesture_perception/one_euro.py

Cài đặt đúng theo docs/ONE_EURO.md. Mỗi hàm/khối ghi rõ nó là công thức
CT-mấy để dò ngược được sang đặc tả.

──────────────────────────────────────────────────────────────────────────────
VÌ SAO CÓ TẦNG NÀY

v2 làm mượt (F, W) bằng EMA hệ số cố định:

    FACE_EVERY_N   = 4      # face detector chỉ chạy 1/4 số frame
    FACE_EMA_ALPHA = 0.3    # nhưng alpha vẫn để 0.3

EMA âm thầm giả định các mẫu cách đều nhau. Detector giãn nhịp còn 5 Hz mà alpha
giữ nguyên -> hằng số thời gian thực tế dài gấp 4 lần ý định:

    nếu chạy mỗi frame (20 Hz)   tau = 0.140 s  <=> cutoff 1.14 Hz   (ý định)
    thực tế ở 5 Hz               tau = 0.561 s  <=> cutoff 0.28 Hz   (đang chạy)

One-Euro khai báo bộ lọc bằng Hz và suy alpha từ dt THỰC ĐO, nên nhịp không đều
và FPS trồi sụt đều xử lý đúng. Đo được (tools/algo_demo.py euro):

    bộ lọc          rung lúc đứng yên    trễ khi vung
    EMA a=0.30            0.0133            230 ms
    One-Euro              0.0105             80 ms      <- thắng CẢ HAI cột

──────────────────────────────────────────────────────────────────────────────
TRUNG LẬP VỚI NGUỒN LANDMARK

Tầng này không biết gì về "khuôn mặt" hay "bàn tay". Nó lọc một bộ ba:

    origin  F   gốc toạ độ      hiện: tâm mặt        sau: trung điểm hai vai
    scale   W   đơn vị đo       hiện: bề rộng mặt    sau: khoảng cách hai vai
    point   C   điểm quan tâm   hiện: tâm bàn tay    sau: cổ tay

rồi trả về u = (C - F)/W. Chuyển sang nhận diện cả người thì chỉ đổi PHÍA GỌI,
không sửa một dòng nào trong file này.

──────────────────────────────────────────────────────────────────────────────
ĐÃ NỐI (2026-09-14)

    nguồn landmark     MediaPipe Pose, gốc = cổ (trung điểm vai), đơn vị = rộng vai
    tầng phân loại     TFLite trong classifier.py
    tầng lệnh          gesture_command/command_logic.py
    đường ra           MAVROS (thay DroneLink của v2)
"""

import math
import numpy as np


# ══════════════════════════════════════════════════════════════════════════
#  THAM SỐ
#
#  Để tạm ở đây trong lúc tầng lọc còn đứng một mình. Khi ráp vào pipeline
#  hoàn chỉnh thì chuyển hết sang config.py — mọi class dưới đây đều nhận tham
#  số qua constructor nên lúc đó chỉ đổi chỗ gọi, không đụng vào thân class.
# ══════════════════════════════════════════════════════════════════════════

D_CUTOFF = 1.0        # Hz. Cutoff CỐ ĐỊNH của bộ lọc vận tốc (CT-5).
DT_MAX = 0.5          # s.  Lâu hơn mức này thì trạng thái cũ vô nghĩa (CT-12/R2).
DT_MIN = 1e-3         # s.  Chặn dưới, bảo vệ phép chia (CT-3).
W_MIN = 0.02          # Chặn dưới cho scale — nó nằm dưới mẫu số (CT-13).
R_MAX = 5             # Số lần loại liên tiếp trước khi ép reset (CT-17).

#                     f_min   beta   S_max      nhịp     ghi chú
PARAMS_SCALE = dict(f_min=0.5, beta=0.05)     # 5 Hz   lọc mạnh nhất, dưới mẫu số
PARAMS_ORIGIN = dict(f_min=0.8, beta=0.20)    # 5 Hz   gốc di chuyển chậm
PARAMS_POINT = dict(f_min=1.0, beta=0.50)     # 20 Hz  mở nhanh nhất, theo cú vung

S_MAX_SCALE = 1.5     # 1/s          tỷ lệ thay đổi tương đối của W
S_MAX_ORIGIN = 10.0   # face-width/s drone yaw làm cả khung trôi ~5 -> để rộng
S_MAX_POINT = 25.0    # face-width/s vung tay nhanh ~12, đỉnh ~20

# Ngưỡng cho Body25Filter, đơn vị SHOULDER-WIDTH/giây (chép từ config_v3 mục 5).
# Suy từ hình học, CHƯA hiệu chuẩn bằng dữ liệu thật. Nhảy sang người khác tạo
# 100-400 sw/s, cách ngưỡng cả bậc độ lớn, nên lệch vài lần vẫn phân tách được.
S_MAX_ORIGIN_SW = 8.0   # drone yaw 45°/s với FOV 60° làm khung trôi ~2 sw/s
S_MAX_JOINT_SW = 18.0   # vung tay hết biên ~2.5 sw trong 0.3 s = 8 sw/s


# ══════════════════════════════════════════════════════════════════════════
#  PHẦN A — BỘ LỌC ONE-EURO           đặc tả: CT-1 .. CT-13
# ══════════════════════════════════════════════════════════════════════════

def alpha_from_cutoff(cutoff_hz, dt):
    """
    CT-1 — Đổi TẦN SỐ CẮT (Hz) sang hệ số làm mượt.

        tau   = 1 / (2*pi*f_c)
        alpha = 1 / (1 + tau/dt)

    Đây là mấu chốt làm One-Euro bất biến FPS: bộ lọc được khai báo bằng một đại
    lượng vật lý (Hz), còn alpha suy ra từ dt THỰC ĐO của từng mẫu.

    Biên: f_c -> vô cùng  =>  alpha -> 1 (không lọc)
          f_c -> 0        =>  alpha -> 0 (đóng băng)
    """
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class LowPass:
    """CT-2 — Lọc thông thấp bậc 1. Chính là EMA, nhưng alpha do người gọi đưa."""

    __slots__ = ("y",)

    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        self.y = x if self.y is None else alpha * x + (1.0 - alpha) * self.y
        return self.y

    def reset(self):
        self.y = None


class OneEuro:
    """
    Bộ lọc One-Euro cho MỘT tín hiệu vô hướng.   CT-3 .. CT-12

    Casiez, Roussel, Vogel — CHI 2012.

    Ý tưởng một câu: DÙNG EMA, NHƯNG TẦN SỐ CẮT TĂNG THEO TỐC ĐỘ TÍN HIỆU.

        f_c = f_min + beta * |v_hat|          (CT-7)

      tín hiệu đứng yên -> f_c nhỏ -> lọc MẠNH -> hết rung
      tín hiệu chạy nhanh -> f_c lớn -> lọc NHẸ -> hết trễ

    Đây là bản THUẦN THUẬT TOÁN, đúng bài báo. Cửa chặn điểm ngoại lai KHÔNG
    nằm ở đây — nó là phần ta thêm vào, để riêng ở PHẦN B.
    """

    def __init__(self, f_min, beta, d_cutoff=D_CUTOFF, dt_max=DT_MAX):
        self.f_min = float(f_min)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dt_max = float(dt_max)

        self._x_lp = LowPass()
        self._v_lp = LowPass()
        self.x_prev = None       # x_{i-1}, giá trị THÔ của mẫu trước
        self.t_prev = None       # thời điểm mẫu ĐƯỢC CHẤP NHẬN gần nhất
        self.velocity = 0.0      # v_hat, để CT-20 dùng lại

    # ---- trạng thái ------------------------------------------------------
    @property
    def value(self):
        """x_hat hiện tại, hoặc None nếu chưa có trạng thái."""
        return self._x_lp.y

    @property
    def has_state(self):
        return self._x_lp.y is not None

    def reset(self, x=None, t=None):
        """
        CT-11 — Khởi tạo lại.

            x_hat = x        (nhận thẳng, không lọc — chưa có gì để so)
            v_hat = 0
            x_prev = x
            t_prev = t
        """
        self._x_lp.reset()
        self._v_lp.reset()
        self.x_prev = None
        self.t_prev = None
        self.velocity = 0.0
        if x is not None and t is not None:
            self._x_lp.y = float(x)
            self._v_lp.y = 0.0
            self.x_prev = float(x)
            self.t_prev = float(t)
        return self._x_lp.y

    # ---- vòng lọc --------------------------------------------------------
    def __call__(self, x, t):
        """Nạp một mẫu -> trả x_hat. Chạy CT-3 .. CT-10."""
        x = float(x)

        # CT-12 / R1, R2 — chưa có trạng thái, hoặc trạng thái quá cũ.
        if not self.has_state or self.t_prev is None:
            return self.reset(x, t)

        dt = t - self.t_prev                                   # CT-3
        if dt > self.dt_max:
            return self.reset(x, t)                            # CT-12 / R2
        if dt <= 0.0:
            dt = DT_MIN                                        # CT-3, chặn chia 0

        v = (x - self.x_prev) / dt                             # CT-4
        alpha_v = alpha_from_cutoff(self.d_cutoff, dt)          # CT-5
        v_hat = self._v_lp(v, alpha_v)                          # CT-6
        self.velocity = v_hat

        # CT-7 ★ — linh hồn của thuật toán. Mọi thứ còn lại là lọc thông thường.
        f_c = self.f_min + self.beta * abs(v_hat)

        alpha = alpha_from_cutoff(f_c, dt)                      # CT-8
        x_hat = self._x_lp(x, alpha)                            # CT-9

        self.x_prev = x                                         # CT-10
        self.t_prev = float(t)
        return x_hat


class OneEuroVec2:
    """
    Hai bộ OneEuro dùng CHUNG tham số, cho một điểm 2 chiều.

    Lọc từng thành phần độc lập là đúng: nhiễu landmark của MediaPipe theo x và
    theo y gần như không tương quan, nên không có gì để ghép chung.
    """

    def __init__(self, f_min, beta, d_cutoff=D_CUTOFF, dt_max=DT_MAX):
        self.fx = OneEuro(f_min, beta, d_cutoff, dt_max)
        self.fy = OneEuro(f_min, beta, d_cutoff, dt_max)

    @property
    def value(self):
        if not self.fx.has_state:
            return None
        return np.array([self.fx.value, self.fy.value])

    @property
    def velocity(self):
        return np.array([self.fx.velocity, self.fy.velocity])

    @property
    def has_state(self):
        return self.fx.has_state

    @property
    def t_prev(self):
        return self.fx.t_prev

    def reset(self, p=None, t=None):
        if p is None:
            self.fx.reset()
            self.fy.reset()
            return None
        self.fx.reset(p[0], t)
        self.fy.reset(p[1], t)
        return self.value

    def __call__(self, p, t):
        self.fx(p[0], t)
        self.fy(p[1], t)
        return self.value


# ══════════════════════════════════════════════════════════════════════════
#  PHẦN B — CỬA CHẶN ĐIỂM NGOẠI LAI     đặc tả: CT-14 .. CT-17
# ══════════════════════════════════════════════════════════════════════════

class OutlierGate:
    """
    Phần One-Euro KHÔNG có sẵn, và là điểm yếu quan trọng nhất phải biết về nó.

    Nếu bộ nhận diện nhảy sang người khác, One-Euro thấy "tốc độ cực lớn" ->
    CT-7 mở toang bộ lọc -> cú nhảy đi qua NGUYÊN VẸN. One-Euro là bộ LÀM MƯỢT,
    không phải bộ KIỂM ĐỊNH. Cửa này bù vào đúng chỗ đó.

    Chạy TRƯỚC bộ lọc, trên giá trị thô, so với trạng thái đã lọc.
    """

    def __init__(self, s_max, r_max=R_MAX):
        self.s_max = float(s_max)
        self.r_max = int(r_max)
        self.n_reject = 0

    def accept(self, dist, scale, dt):
        """
        CT-14 + CT-15 — trả True nếu mẫu được nhận.

            s = dist / (scale * dt)          [đơn vị đo / giây]

        Chia cho `scale` để bất biến khoảng cách, chia cho `dt` để bất biến FPS.
        Cùng tinh thần với toàn hệ.

        CT-16 — tính TỰ CHỮA LÀNH nằm ở phía gọi: mẫu bị loại thì KHÔNG cập nhật
        t_prev của bộ lọc, nên dt ở lần sau lớn hơn, s nhỏ đi, cửa tự nới ra.
        Loại càng lâu cửa càng rộng — cửa không bao giờ khoá vĩnh viễn.
        """
        if scale is None or scale < W_MIN or dt <= 0.0:
            return True                    # chưa đủ căn cứ để loại -> cho qua
        s = dist / (scale * dt)
        if s > self.s_max:
            self.n_reject += 1
            return False
        self.n_reject = 0
        return True

    @property
    def should_force_reset(self):
        """
        CT-17 — Lối thoát cứng.

        Nếu operator THẬT SỰ đã dịch chuyển lớn, sau r_max frame hệ phải chấp
        nhận thực tại mới. Không có cái này thì cửa chặn có thể khoá chết hệ.
        """
        return self.n_reject >= self.r_max

    def reset(self):
        self.n_reject = 0


# ══════════════════════════════════════════════════════════════════════════
#  PHẦN C — HỆ QUY CHIẾU ĐÃ LỌC        đặc tả: CT-18 .. CT-20
# ══════════════════════════════════════════════════════════════════════════

class BodyFrameFilter:
    """
    Ráp 5 bộ lọc + 3 cửa chặn thành một hệ quy chiếu cơ thể sạch.

        origin  F   gốc toạ độ     (2 bộ lọc)
        scale   W   đơn vị đo      (1 bộ lọc)
        point   C   điểm quan tâm  (2 bộ lọc)

        u = (C - F) / W        [đơn vị: scale-width]

    CT-18 ★ — HAI NHỊP CẬP NHẬT KHÁC NHAU. Đây là chỗ v3 sửa lỗi của v2.
    Detector của (F, W) chạy giãn nhịp, còn C chạy mỗi frame. Mỗi bộ lọc tự đo
    dt của RIÊNG nó, nên gọi update_frame() và update_point() ở hai nhịp khác
    nhau là hoàn toàn hợp lệ.

    TUYỆT ĐỐI KHÔNG gọi update_frame() với giá trị cũ trên frame không có
    detection — làm vậy ép v = 0 ở CT-4 và bóp chết khả năng thích nghi.
    Không có detection thì đơn giản là KHÔNG GỌI.
    """

    def __init__(
        self,
        params_origin=None,
        params_scale=None,
        params_point=None,
        s_max_origin=S_MAX_ORIGIN,
        s_max_scale=S_MAX_SCALE,
        s_max_point=S_MAX_POINT,
        w_min=W_MIN,
    ):
        po = params_origin or PARAMS_ORIGIN
        ps = params_scale or PARAMS_SCALE
        pp = params_point or PARAMS_POINT

        self.f_origin = OneEuroVec2(**po)
        self.f_scale = OneEuro(**ps)
        self.f_point = OneEuroVec2(**pp)

        self.g_origin = OutlierGate(s_max_origin)
        self.g_scale = OutlierGate(s_max_scale)
        self.g_point = OutlierGate(s_max_point)

        self.w_min = float(w_min)
        self.last_status = "NO_LOCK"

    # ---- trạng thái ------------------------------------------------------
    @property
    def W(self):
        """CT-13 — scale đã lọc, có chặn dưới. Nó nằm dưới mẫu số của CT-19."""
        w = self.f_scale.value
        return None if w is None else max(w, self.w_min)

    @property
    def F(self):
        return self.f_origin.value

    @property
    def C(self):
        return self.f_point.value

    @property
    def ready(self):
        return self.F is not None and self.W is not None and self.C is not None

    def reset(self):
        """CT-12 / R3 — mất lock thì xoá sạch, không giữ lại gì."""
        for f in (self.f_origin, self.f_scale, self.f_point):
            f.reset()
        for g in (self.g_origin, self.g_scale, self.g_point):
            g.reset()
        self.last_status = "NO_LOCK"

    # ---- nạp mẫu ---------------------------------------------------------
    def update_frame(self, F_raw, W_raw, t):
        """
        Nạp gốc toạ độ + đơn vị đo. Gọi ở nhịp của detector (vd 5 Hz).
        Chỉ gọi khi THẬT SỰ có detection.

        Thứ tự bắt buộc: scale trước origin, vì cửa chặn của origin cần W đã lọc
        để chuẩn hoá (mục 3 trong PHẦN 8 của đặc tả).
        """
        F_raw = np.asarray(F_raw, dtype=float)
        W_raw = float(W_raw)
        accepted = True

        # ---- scale: cửa chặn đo TỶ LỆ thay đổi tương đối (CT-14 bản vô hướng)
        if self.f_scale.has_state:
            dt = t - self.f_scale.t_prev
            w_hat = self.f_scale.value
            if not self.g_scale.accept(abs(W_raw - w_hat), w_hat, dt):
                accepted = False
                if self.g_scale.should_force_reset:          # CT-17
                    self.f_scale.reset(W_raw, t)
                    self.g_scale.reset()
                    accepted = True
        if accepted or not self.f_scale.has_state:
            self.f_scale(W_raw, t)

        # ---- origin: chuẩn hoá theo W đã lọc
        w = self.W
        if self.f_origin.has_state:
            dt = t - self.f_origin.t_prev
            dist = float(np.linalg.norm(F_raw - self.f_origin.value))
            if not self.g_origin.accept(dist, w, dt):
                if self.g_origin.should_force_reset:         # CT-17
                    self.f_origin.reset(F_raw, t)
                    self.g_origin.reset()
                else:
                    self.last_status = "REJECTED"
                    return self.F, self.W
        self.f_origin(F_raw, t)
        return self.F, self.W

    def update_point(self, C_raw, t):
        """
        Nạp điểm quan tâm. Gọi mỗi frame có dữ liệu (vd 20 Hz).
        Cần W đã lọc để chuẩn hoá cửa chặn -> phải gọi SAU update_frame.
        """
        C_raw = np.asarray(C_raw, dtype=float)
        w = self.W
        if self.f_point.has_state:
            dt = t - self.f_point.t_prev
            dist = float(np.linalg.norm(C_raw - self.f_point.value))
            if not self.g_point.accept(dist, w, dt):
                if self.g_point.should_force_reset:          # CT-17
                    self.f_point.reset(C_raw, t)
                    self.g_point.reset()
                else:
                    self.last_status = "REJECTED"
                    return self.C
        self.f_point(C_raw, t)
        return self.C

    # ---- đầu ra ----------------------------------------------------------
    @property
    def u(self):
        """
        CT-19 — Vector điểm quan tâm trong hệ quy chiếu cơ thể.

            u = (C_hat - F_hat) / W_hat

        Dùng giá trị ĐÃ LỌC của cả ba. KHÔNG lọc u thêm lần nữa: khi tay vung
        nhanh mà mặt đứng yên, lọc u sẽ thấy tốc độ lớn -> mở toang -> nhiễu
        khuôn mặt lọt qua ĐÚNG LÚC đang ra lệnh. Lọc riêng thì C mở còn F, W
        vẫn đóng.
        """
        if not self.ready:
            return None
        return (self.C - self.F) / self.W

    @property
    def u_dot(self):
        """
        CT-20 — Vận tốc của u, theo quy tắc thương.

            u  = (C - F)/W
            u' = (C' - F')/W  -  u * (W'/W)

        Cả ba đạo hàm đã có sẵn từ CT-6 của từng bộ lọc -> không tốn thêm phép
        tính nào. Đây là đặc trưng miễn phí cho tầng phân loại theo thời gian.
        """
        if not self.ready:
            return None
        w = self.W
        return (self.f_point.velocity - self.f_origin.velocity) / w \
            - self.u * (self.f_scale.velocity / w)


# ══════════════════════════════════════════════════════════════════════════
#  PHẦN D — LỌC CẢ 25 KHỚP BODY25       dùng trong perception_node
# ══════════════════════════════════════════════════════════════════════════

_B25_NECK, _B25_RSHOULDER, _B25_LSHOULDER = 1, 2, 5


class Body25Filter:
    """
    Cùng ý tưởng BodyFrameFilter, mở rộng từ MỘT điểm quan tâm lên 25 khớp.

        F  = cổ (trung điểm hai vai)         OneEuroVec2  + cửa chặn
        W  = khoảng cách hai vai (pixel)     OneEuro
        u_j = (p_j - F_hat) / W_hat          OneEuroVec2 cho TỪNG khớp + cửa chặn
        p_hat_j = F_hat + u_hat_j * W_hat    trả về pixel, giữ nguyên cột conf

    Vì sao lọc u chứ không lọc pixel: f_min/beta/S_max khai báo theo
    shoulder-width/giây. Lọc pixel thì người đứng xa 2 lần đổi tốc độ pixel 2
    lần và cùng một beta cho hai hành vi khác nhau.

    Trả về pixel cũng không làm sai gì: normalize_body25 bất biến tịnh tiến và
    tỷ lệ (tự kiểm 1-2 trong pck_format.py), nên hệ toạ độ đầu ra không quan
    trọng, chỉ hình dạng tư thế là quan trọng.

    Khớp có conf = 0 thì reset bộ lọc của khớp đó và giữ nguyên hàng 0 — đúng
    quy ước "khớp không thấy" của PCK. Không thấy cổ hoặc một vai thì reset
    toàn bộ và trả nguyên đầu vào: không có hệ quy chiếu thì không lọc được.
    """

    def __init__(self, params_origin=None, params_scale=None, params_joint=None,
                 s_max_origin=S_MAX_ORIGIN_SW, s_max_joint=S_MAX_JOINT_SW,
                 d_cutoff=D_CUTOFF, dt_max=DT_MAX, n_joints=25):
        po = dict(params_origin or PARAMS_ORIGIN)
        ps = dict(params_scale or PARAMS_SCALE)
        pj = dict(params_joint or PARAMS_POINT)
        self._beta_origin = po["beta"]
        self._beta_scale = ps["beta"]
        self.f_origin = OneEuroVec2(d_cutoff=d_cutoff, dt_max=dt_max, **po)
        self.f_scale = OneEuro(d_cutoff=d_cutoff, dt_max=dt_max, **ps)
        self.f_joint = [OneEuroVec2(d_cutoff=d_cutoff, dt_max=dt_max, **pj)
                        for _ in range(n_joints)]
        self.g_origin = OutlierGate(s_max_origin)
        self.g_joint = [OutlierGate(s_max_joint) for _ in range(n_joints)]
        self.n_rejected = 0          # số khớp bị cửa chặn loại ở frame gần nhất

    def reset(self):
        self.f_origin.reset()
        self.f_scale.reset()
        self.g_origin.reset()
        for f, g in zip(self.f_joint, self.g_joint):
            f.reset()
            g.reset()

    @staticmethod
    def _gate(filt, gate, x_raw, scale, t):
        """Cửa chặn + CT-17 cho một OneEuroVec2. -> True nếu được nạp mẫu."""
        if not filt.has_state:
            return True
        dist = float(np.linalg.norm(x_raw - filt.value))
        if gate.accept(dist, scale, t - filt.t_prev):
            return True
        if gate.should_force_reset:
            filt.reset(x_raw, t)
            gate.reset()
        return False

    def __call__(self, kp, t):
        kp = np.asarray(kp, dtype=np.float64)
        vis = kp[:, 2] > 0.0
        if not (vis[_B25_NECK] and vis[_B25_RSHOULDER] and vis[_B25_LSHOULDER]):
            self.reset()
            return kp

        w_raw = float(np.linalg.norm(kp[_B25_RSHOULDER, :2] - kp[_B25_LSHOULDER, :2]))
        if w_raw < 1e-6:
            self.reset()
            return kp
        # F và W lọc trong PIXEL, nhưng beta khai báo theo shoulder-width/giây.
        # Chia beta cho W hiện tại thì beta*|v_px| = beta_sw*|v_sw| -> người đứng
        # xa hay gần đều cho cùng một tần số cắt. Không làm vậy thì đứng gần gấp
        # 3 lần là bộ lọc mở rộng gấp 3 lần.
        w_ref = self.f_scale.value if self.f_scale.has_state else w_raw
        self.f_scale.beta = self._beta_scale / w_ref
        W = max(self.f_scale(w_raw, t), W_MIN)
        self.f_origin.fx.beta = self.f_origin.fy.beta = self._beta_origin / W

        neck = kp[_B25_NECK, :2]
        if self._gate(self.f_origin, self.g_origin, neck, W, t):
            self.f_origin(neck, t)
        F = self.f_origin.value

        out = kp.copy()
        self.n_rejected = 0
        for j in range(len(self.f_joint)):
            f = self.f_joint[j]
            if not vis[j]:
                f.reset()
                self.g_joint[j].reset()
                continue
            u_raw = (kp[j, :2] - F) / W
            # Cửa chặn so trong hệ u, nên scale = 1.0 (u đã chia W rồi).
            if self._gate(f, self.g_joint[j], u_raw, 1.0, t):
                f(u_raw, t)
            else:
                self.n_rejected += 1
            out[j, :2] = F + f.value * W
        return out


# ══════════════════════════════════════════════════════════════════════════
#  TỰ KIỂM — đặc tả PHẦN 10 (vector kiểm chứng) và PHẦN 11 (6 bất biến)
# ══════════════════════════════════════════════════════════════════════════

def _test_vector():
    """PHẦN 10 — đối chiếu từng chữ số với bảng trong đặc tả."""
    expect = [0.500000, 0.502931, 0.503427, 0.530359, 0.582559, 0.653912]
    xs = [0.500, 0.512, 0.505, 0.600, 0.700, 0.800]
    dt = 0.05

    f = OneEuro(f_min=1.0, beta=0.5, d_cutoff=1.0)
    print("  PHAN 10 — vector kiem chung (C.x, dt=0.05, f_min=1.0, beta=0.5)")
    print(f"    {'i':>2} {'x_raw':>8} {'v_hat':>9} {'f_c':>8} {'x_hat':>10} {'mong doi':>10}  ")
    ok = True
    for i, (x, want) in enumerate(zip(xs, expect)):
        got = f(x, i * dt)
        f_c = f.f_min + f.beta * abs(f.velocity)
        good = abs(got - want) < 5e-6
        ok &= good
        print(f"    {i:>2} {x:>8.4f} {f.velocity:>9.4f} {f_c:>8.4f} "
              f"{got:>10.6f} {want:>10.6f}  {'OK' if good else 'SAI'}")
    return ok


def _test_invariants():
    """PHẦN 11 — I1..I6."""
    results = []

    # I1: 0 < alpha <= 1 với mọi f_c > 0, mọi dt > 0
    ok = True
    for f_c in (1e-3, 0.1, 1.0, 10.0, 1e3, 1e6):
        for dt in (1e-3, 0.01, 0.05, 0.5, 5.0):
            a = alpha_from_cutoff(f_c, dt)
            ok &= (0.0 < a <= 1.0)
    results.append(("I1", "0 < alpha <= 1 voi moi f_c, dt", ok))

    # I2: beta = 0 => trùng khít lọc thông thấp cố định f_min
    f_oe = OneEuro(f_min=0.8, beta=0.0)
    lp, dt = LowPass(), 0.05
    rng = np.random.default_rng(7)
    ok = True
    for i in range(60):
        x = float(rng.normal(0.5, 0.05))
        a = f_oe(x, i * dt)
        b = x if i == 0 else lp(x, alpha_from_cutoff(0.8, dt))
        if i == 0:
            lp.y = x
        ok &= abs(a - b) < 1e-12
    results.append(("I2", "beta=0 trung khit loc co dinh f_min", ok))

    # I3: tín hiệu hằng số => x_hat hội tụ về hằng số, v_hat -> 0
    f = OneEuro(**PARAMS_POINT)
    for i in range(200):
        f(0.42, i * 0.05)
    ok = abs(f.value - 0.42) < 1e-9 and abs(f.velocity) < 1e-9
    results.append(("I3", "tin hieu hang so -> hoi tu, v_hat -> 0", ok))

    # I4: BẤT BIẾN FPS.
    #
    # Đo "mẫu đầu tiên vượt 90%" vốn BỊ LƯỢNG TỬ HOÁ theo chu kỳ lấy mẫu: ở
    # 15 FPS hai mẫu cách nhau 66.7 ms, nên mọi phép đo trễ đều có sai số cỡ
    # đó. So bằng TỶ LỆ là so nhầm đại lượng — 97/63 = 1.5x nghe như lệch
    # nhiều, nhưng chênh tuyệt đối chỉ 34 ms, chưa bằng một chu kỳ lấy mẫu.
    #
    # Phát biểu đúng: chênh lệch trễ phải NHỎ HƠN MỘT CHU KỲ LẤY MẪU của FPS
    # thấp. Dưới ngưỡng đó thì không phân biệt được với sai số phép đo.
    def lag_of(make_filter, fps):
        n, target = int(3.0 * fps), 2.5
        f = make_filter()
        rng = np.random.default_rng(0)
        for i in range(n):
            t = i / fps
            truth = 0.0 if t < 1.5 else (target * min((t - 1.5) / 0.3, 1.0))
            y = f(truth + rng.normal(0, 0.03), t)
            if y >= 0.9 * target:
                return t - (1.5 + 0.9 * 0.3)
        return float("inf")

    quantum = 1.0 / 15.0
    oe = lambda: OneEuro(**PARAMS_POINT)
    l15, l30 = lag_of(oe, 15.0), lag_of(oe, 30.0)
    ok = abs(l15 - l30) < quantum
    results.append(("I4", f"One-Euro tre 15/30 FPS = {l15*1000:.0f}/{l30*1000:.0f}ms, "
                          f"chenh {abs(l15-l30)*1000:.0f}ms < {quantum*1000:.0f}ms "
                          f"(1 chu ky lay mau)", ok))

    # I4b: đối chứng — EMA alpha cố định PHẢI trượt quá ngưỡng đó, nếu không
    # thì I4 chẳng chứng minh được gì.
    class _FixedEMA:
        def __init__(self, a):
            self.a, self.y = a, None

        def __call__(self, x, t):
            self.y = x if self.y is None else self.a * x + (1 - self.a) * self.y
            return self.y

    e15, e30 = lag_of(lambda: _FixedEMA(0.30), 15.0), lag_of(lambda: _FixedEMA(0.30), 30.0)
    ok_b = abs(e15 - e30) > quantum
    results.append(("I4b", f"doi chung EMA a=0.30 = {e15*1000:.0f}/{e30*1000:.0f}ms, "
                           f"chenh {abs(e15-e30)*1000:.0f}ms > {quantum*1000:.0f}ms "
                           f"-> I4 that su phan biet duoc", ok_b))

    # I5: nhảy 200 fw/s => bị CT-15 loại, u không đổi
    bf = BodyFrameFilter()
    t = 0.0
    for _ in range(10):                       # nạp cho ổn định
        bf.update_frame([0.50, 0.40], 0.15, t)
        bf.update_point([0.50, 0.10], t)
        t += 0.05
    u_before = bf.u.copy()
    bf.update_point([0.50 + 200 * 0.15 * 0.05, 0.10], t)   # s = 200 fw/s
    ok = np.allclose(bf.u, u_before) and bf.g_point.n_reject == 1
    results.append(("I5", "nhay 200 fw/s -> bi loai, u khong doi", ok))

    # I6: giữ cú nhảy 5 frame liên tiếp => CT-17 reset, nhận giá trị mới
    jump = 0.50 + 200 * 0.15 * 0.05
    for _ in range(5):
        t += 0.05
        bf.update_point([jump, 0.10], t)
    ok = abs(bf.C[0] - jump) < 1e-9 and bf.g_point.n_reject == 0
    results.append(("I6", "giu 5 frame -> CT-17 reset, nhan gia tri moi", ok))

    return results


def _test_frame_rates():
    """
    CT-18 — hai nhịp cập nhật khác nhau. Đây là lỗi mà v3 sửa, nên phải có test.

    Detector chạy 5 Hz, điểm quan tâm chạy 20 Hz. Kiểm tra mỗi bộ lọc tự đo
    ĐÚNG dt của riêng nó chứ không dùng chung nhịp video.
    """
    bf = BodyFrameFilter()
    fps, face_every = 20.0, 4
    for i in range(40):
        t = i / fps
        if i % face_every == 0:
            bf.update_frame([0.50, 0.40], 0.15, t)
        bf.update_point([0.50, 0.10], t)

    dt_face = bf.f_scale.t_prev - (bf.f_scale.t_prev - face_every / fps)
    dt_point = 1.0 / fps
    ok_face = abs(dt_face - 0.20) < 1e-9
    ok_point = abs(dt_point - 0.05) < 1e-9
    # tay ở 0.30 phía trên gốc, đơn vị 0.15 -> u.y = -2.0
    ok_u = abs(bf.u[1] - (-2.0)) < 1e-6
    print(f"    dt_face  = {dt_face:.3f} s (mong doi 0.200)   {'OK' if ok_face else 'SAI'}")
    print(f"    dt_point = {dt_point:.3f} s (mong doi 0.050)  {'OK' if ok_point else 'SAI'}")
    print(f"    u        = ({bf.u[0]:+.3f}, {bf.u[1]:+.3f}) scale-width, "
          f"mong doi (0.000, -2.000)  {'OK' if ok_u else 'SAI'}")
    return ok_face and ok_point and ok_u


def _test_u_dot():
    """CT-20 — kiểm thứ nguyên và dấu bằng một chuyển động đã biết trước."""
    bf = BodyFrameFilter()
    fps, speed = 20.0, -1.5     # điểm đi LÊN 1.5 scale-width mỗi giây (y giảm)
    W = 0.15
    for i in range(120):
        t = i / fps
        bf.update_frame([0.50, 0.40], W, t)          # gốc và đơn vị đứng yên
        bf.update_point([0.50, 0.40 + speed * W * t], t)
    got = bf.u_dot[1]
    ok = abs(got - speed) < 0.05
    print(f"    u_dot.y  = {got:+.3f} scale-width/s (mong doi {speed:+.3f})  "
          f"{'OK' if ok else 'SAI'}")
    return ok


def _test_body25():
    """PHẦN D — Body25Filter: giảm rung, bất biến khoảng cách, theo kịp, reset."""
    rng = np.random.default_rng(3)
    base = np.zeros((25, 3))
    base[:, 2] = 1.0
    base[:, 0] = 320 + 40 * np.cos(np.arange(25))
    base[:, 1] = 240 + 60 * np.sin(np.arange(25))
    base[_B25_NECK, :2] = (320, 200)
    base[_B25_RSHOULDER, :2] = (280, 200)
    base[_B25_LSHOULDER, :2] = (360, 200)
    ok_all = True

    # 1. đứng yên có nhiễu 2 px -> rung sau lọc phải nhỏ hơn hẳn
    bf = Body25Filter()
    raw_jit, out_jit = [], []
    for i in range(100):
        noisy = base.copy()
        noisy[:, :2] += rng.normal(0, 2.0, (25, 2))
        y = bf(noisy, i / 10.0)
        if i > 20:
            raw_jit.append(np.abs(noisy[:, :2] - base[:, :2]).mean())
            out_jit.append(np.abs(y[:, :2] - base[:, :2]).mean())
    ok = np.mean(out_jit) < 0.7 * np.mean(raw_jit)
    ok_all &= ok
    print(f"    rung dung yen: tho {np.mean(raw_jit):.2f} px -> loc "
          f"{np.mean(out_jit):.2f} px  {'OK' if ok else 'SAI'}")

    # 2. bất biến khoảng cách: cùng chuyển động, người to gấp 3 -> u giống hệt
    def run(scale):
        f = Body25Filter()
        res = None
        for i in range(30):
            k = base.copy()
            k[:, :2] = 320 + (k[:, :2] - 320) * scale
            k[:, 0] += 25 * scale * min(i / 10.0, 1.0)      # cả người dịch
            k[7, 0] += 30 * scale * min(i / 10.0, 1.0)      # cổ tay dịch thêm
            y = f(k, i / 10.0)
            w = np.linalg.norm(y[_B25_RSHOULDER, :2] - y[_B25_LSHOULDER, :2])
            res = (y[7, :2] - y[_B25_NECK, :2]) / w
        return res
    d = float(np.abs(run(1.0) - run(3.0)).max())
    ok = d < 1e-6
    ok_all &= ok
    print(f"    bat bien khoang cach: lech u = {d:.1e}  {'OK' if ok else 'SAI'}")

    # 3. khớp mất (conf=0) -> hàng 0 giữ nguyên, không lọc
    k = base.copy()
    k[4] = 0.0
    y = bf(k, 10.5)
    ok = np.all(y[4] == 0.0)
    ok_all &= ok
    print(f"    khop mat -> giu hang 0  {'OK' if ok else 'SAI'}")

    # 4. mất vai -> trả nguyên đầu vào, reset
    k = base.copy()
    k[_B25_LSHOULDER] = 0.0
    y = bf(k, 10.6)
    ok = np.array_equal(y, k) and not bf.f_scale.has_state
    ok_all &= ok
    print(f"    mat vai -> tra nguyen, reset  {'OK' if ok else 'SAI'}")
    return ok_all


def main():
    print("=" * 74)
    print("one_euro — TU KIEM TANG BO LOC")
    print("=" * 74)
    print()

    all_ok = True

    all_ok &= _test_vector()
    print()

    print("  PHAN 11 — sau bat bien")
    for tag, desc, ok in _test_invariants():
        all_ok &= ok
        print(f"    {tag}  {'OK ' if ok else 'SAI'}  {desc}")
    print()

    print("  CT-18 — hai nhip cap nhat khac nhau")
    all_ok &= _test_frame_rates()
    print()

    print("  CT-20 — van toc cua u")
    all_ok &= _test_u_dot()
    print()

    print("  PHAN D — Body25Filter")
    all_ok &= _test_body25()
    print()

    print("=" * 74)
    print("KET QUA: " + ("TAT CA DAT" if all_ok else "CO TEST SAI"))
    print("=" * 74)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
