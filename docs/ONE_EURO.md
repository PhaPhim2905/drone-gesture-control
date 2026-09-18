# ĐẶC TẢ CÔNG THỨC — BỘ LỌC ONE-EURO

> **Trạng thái:** **đã cài đặt** trong
> `ros2_ws/src/gesture_perception/gesture_perception/one_euro.py` (lớp `Body25Filter`,
> đang chạy trong `perception_node`). File này là đặc tả gốc: mỗi hàm trong code ghi
> rõ nó là công thức CT-mấy để dò ngược về đây.
> **Mục tiêu ban đầu:** đọc hiểu và duyệt được công thức TRƯỚC khi có dòng code nào.
>
> Mục "Vì sao làm việc này" bên dưới nói về `FACE_EMA_ALPHA` / `OperatorLock` của
> bản v2 (đã xoá khỏi nhánh chính). Giữ lại vì đó là **lý do** bộ lọc này tồn tại;
> lập luận về `alpha` gắn với số mẫu thay vì với thời gian vẫn đúng nguyên.

---

## Vì sao làm việc này

Ba lý do, xếp theo mức nghiêm trọng.

### 1. Có một lỗi thật trong code hiện tại

```python
FACE_EVERY_N  = 4       # face detector chỉ chạy 1/4 số frame
FACE_EMA_ALPHA = 0.3    # nhưng alpha vẫn để 0.3
```

EMA **âm thầm giả định các mẫu cách đều nhau**. Face detector giãn nhịp xuống
5 Hz nhưng `alpha` vẫn giữ nguyên, nên hằng số thời gian thực tế dài gấp 4 lần
ý định ban đầu:

| | tau | tương đương cutoff |
|---|---|---|
| nếu chạy mỗi frame (20 Hz) | 0.140 s | 1.14 Hz — *ý định* |
| thực tế ở 5 Hz (EVERY_N=4) | 0.561 s | **0.28 Hz** — *đang chạy* |

Không ai cố tình làm vậy. Đây là hệ quả âm thầm của việc `alpha` gắn với **số
mẫu** chứ không gắn với **thời gian**.

### 2. FPS trên Pi không ổn định

CPU nóng thì throttle, FPS tụt. Với `alpha` cố định, bộ lọc **tự đổi tính cách
theo nhiệt độ máy**. Đo được (cùng một bộ tham số):

| bộ lọc | 15 FPS | 30 FPS | chênh |
|---|---|---|---|
| EMA a=0.30 | 363 ms | 130 ms | 2.8x |
| One-Euro | 97 ms | 63 ms | 1.5x |

### 3. Độ trễ là vấn đề AN TOÀN

Ngân sách trễ toàn tuyến (khâu bộ lọc là số đo thật, còn lại là ước tính):

| khâu | EMA a=0.3 | One-Euro |
|---|---|---|
| camera: phơi sáng + truyền USB | 35 ms | 35 ms |
| MediaPipe Hands Lite | 16 ms | 16 ms |
| **bộ lọc** | **230 ms** | **80 ms** |
| CommandState hold 0.15 s | 150 ms | 150 ms |
| MAVLink + ArduPilot đáp ứng | 75 ms | 75 ms |
| **TỔNG** | **506 ms** | **356 ms** |
| quãng đường trôi @ 1.0 m/s | 0.51 m | 0.36 m |

Bộ lọc chiếm 45% ngân sách trễ. Anh kéo tay về trung tính, drone vẫn bay thêm
nửa mét — One-Euro cắt 15 cm trong đó.

### Kết quả đo (dữ liệu tổng hợp, `tools/algo_demo.py euro`, công cụ thử nghiệm cũ)

FPS=20, nhiễu sigma=0.03 face-width, vung tay 0 → 2.5 trong 0.3 s:

| bộ lọc | rung lúc đứng yên | trễ khi vung |
|---|---|---|
| không lọc | 0.0263 | 30 ms |
| EMA a=0.30 | 0.0133 | 230 ms |
| EMA a=0.10 | 0.0046 | 980 ms |
| **One-Euro** | **0.0105** | **80 ms** |

One-Euro **thắng EMA a=0.30 ở CẢ HAI cột cùng lúc**. Nó không phải một điểm
khác trên cùng đường đánh đổi — nó là một đường khác.

---

## PHẦN 0 — Ký hiệu

| Ký hiệu | Nghĩa | Đơn vị |
|---|---|---|
| `t` | thời điểm frame hiện tại | giây |
| `dt` | thời gian kể từ mẫu **được chấp nhận** gần nhất | giây |
| `z` | một tín hiệu vô hướng bất kỳ đang lọc | tuỳ tín hiệu |
| `z_raw` | giá trị đo từ MediaPipe (chưa lọc) | — |
| `z_hat` | giá trị **sau lọc** — thứ hệ thống dùng | — |
| `zd` | vận tốc thô của `z` | đơn vị/s |
| `zd_hat` | vận tốc **sau lọc** | đơn vị/s |
| `f_c` | tần số cắt hiện hành (đổi mỗi frame) | Hz |
| `f_min` | sàn tần số cắt — tham số chỉnh tay | Hz |
| `beta` | độ dốc thích nghi — tham số chỉnh tay | Hz·s/đơn vị |
| `f_d` | tần số cắt **cố định** cho bộ lọc vận tốc | Hz |
| `alpha` | hệ số EMA, **tính lại mỗi frame** | 0..1 |
| `tau` | hằng số thời gian | giây |
| `F` | tâm khuôn mặt `(F.x, F.y)` | toạ độ ảnh 0..1 |
| `W` | bề rộng khuôn mặt | toạ độ ảnh 0..1 |
| `C` | tâm neo bàn tay `(C.x, C.y)` | toạ độ ảnh 0..1 |
| `u` | vector tay trong hệ cơ thể | face-width |

---

## PHẦN 1 — Vào và ra mỗi frame

```
ĐẦU VÀO
  t                       thời điểm (time.perf_counter)
  C_raw = (C.x, C.y)      mỗi frame, nếu thấy tay
  F_raw = (F.x, F.y)      chỉ trên frame chạy face detector
  W_raw                   chỉ trên frame chạy face detector

ĐẦU RA
  u_hat  = (u.x, u.y)     vector tay đã lọc      [face-width]
  ud_hat = (ux', uy')     vận tốc tay đã lọc     [face-width/s]  <- bonus cho A4
  trạng thái: OK | REJECTED | RESET | NO_LOCK
```

**Số bộ lọc cần: 5 bộ One-Euro vô hướng** — `F.x`, `F.y`, `W`, `C.x`, `C.y`.
Mỗi bộ chứa 2 bộ lọc thông thấp (tín hiệu + vận tốc) → tổng 10 trạng thái nội bộ.

---

## PHẦN 2 — Hai công thức nền

### CT-1 — Đổi tần số cắt sang hệ số EMA

```
    tau   = 1 / (2*pi*f_c)

    alpha = 1 / (1 + tau/dt)
```

Gộp một dòng:

```
    alpha = (2*pi*f_c*dt) / (1 + 2*pi*f_c*dt)
```

**Kiểm tra biên:**

- `f_c -> vô cùng` ⟹ `alpha -> 1` (không lọc gì)
- `f_c -> 0` ⟹ `alpha -> 0` (đóng băng)
- `dt` lớn ⟹ `alpha` lớn (bù cho việc lâu không cập nhật)

> ⚠️ Đây là công thức quan trọng nhất của cả đặc tả. `dt` phải là **thời gian
> thật đo được**, không được thay bằng hằng số `1/FPS`.

### CT-2 — Cập nhật lọc thông thấp bậc 1

```
    z_hat[i] = alpha * z[i] + (1 - alpha) * z_hat[i-1]
```

---

## PHẦN 3 — One-Euro cho MỘT tín hiệu vô hướng

Chạy tuần tự CT-3 → CT-10 cho mỗi tín hiệu, mỗi khi có mẫu mới.

### CT-3 — Bước thời gian

```
    dt = t - t_prev
```

Kèm hai ràng buộc (chi tiết ở PHẦN 4):

```
    nếu dt <= 0       ->  dt = 1e-3          (bảo vệ chia 0)
    nếu dt > dt_max   ->  RESET, không lọc   (dt_max = 0.5 s)
```

### CT-4 — Vận tốc thô

```
    zd[i] = (z[i] - z_prev) / dt
```

### CT-5 — Hệ số alpha cho bộ lọc vận tốc

```
    tau_d   = 1 / (2*pi*f_d)
    alpha_d = 1 / (1 + tau_d/dt)
```

`f_d` **cố định**, không thích nghi. Mặc định `f_d = 1.0 Hz`.

### CT-6 — Vận tốc đã lọc

```
    zd_hat[i] = alpha_d * zd[i] + (1 - alpha_d) * zd_hat[i-1]
```

> **Vì sao bắt buộc phải có CT-5, CT-6:** vận tốc thô là hiệu hai mẫu nhiễu nên
> nhiễu bị khuếch đại:
>
> ```
> sigma_zd = sigma * sqrt(2) / dt
>          = 0.03 * 1.414 / 0.05
>          = 0.85 đơn vị/s
> ```
>
> Nhân với `beta = 0.5` ra **0.42 Hz cutoff giả tạo** — so với `f_min = 1.0 Hz`
> thì nhiễu đóng góp 30% độ mở bộ lọc. Bỏ bước này thì bộ lọc **tự vô hiệu hoá
> đúng lúc tay đứng yên**, tức đúng lúc ta cần nó nhất.

### CT-7 — Tần số cắt thích nghi ★

```
    f_c = f_min + beta * abs(zd_hat[i])
```

**Đây là linh hồn của thuật toán.** Mọi thứ còn lại là lọc thông thấp thường.

```
    tay đứng yên    ->  |zd| ~ 0  ->  f_c nhỏ  ->  alpha nhỏ  ->  lọc MẠNH -> hết rung
    tay vung nhanh  ->  |zd| lớn  ->  f_c lớn  ->  alpha -> 1 ->  lọc NHẸ  -> hết trễ
```

### CT-8 — Hệ số alpha cho bộ lọc tín hiệu

```
    tau   = 1 / (2*pi*f_c)
    alpha = 1 / (1 + tau/dt)
```

### CT-9 — Đầu ra

```
    z_hat[i] = alpha * z[i] + (1 - alpha) * z_hat[i-1]
```

### CT-10 — Cập nhật trạng thái (làm SAU CÙNG, chỉ khi mẫu được chấp nhận)

```
    z_prev = z[i]
    t_prev = t
```

---

## PHẦN 4 — Khởi tạo và điều kiện reset

### CT-11 — Mẫu đầu tiên

```
    z_hat[0]  = z[0]        (nhận thẳng, không lọc)
    zd_hat[0] = 0
    z_prev    = z[0]
    t_prev    = t[0]
```

### CT-12 — Ba điều kiện RESET

```
    R1.  chưa có trạng thái (z_hat == null)
    R2.  dt > dt_max = 0.5 s          <- mất tín hiệu quá lâu, trạng thái cũ vô nghĩa
    R3.  operator lock bị nhả (mất mặt > FACE_LOST_SEC)

    Khi RESET  ->  áp lại CT-11, KHÔNG chạy CT-3..CT-10
```

### CT-13 — Chặn dưới cho W

```
    W_hat = max(W_hat, W_min)         W_min = 0.02
```

Bắt buộc, vì `W_hat` nằm dưới mẫu số ở CT-19.

---

## PHẦN 5 — Cửa chặn điểm ngoại lai

> Đây là phần **One-Euro KHÔNG có sẵn** — và là điểm yếu quan trọng nhất phải
> biết về thuật toán này. Nếu bộ nhận diện nhảy sang người khác, One-Euro thấy
> "tốc độ cực lớn" → CT-7 mở toang → **cú nhảy đi qua nguyên vẹn**. One-Euro là
> bộ *làm mượt*, không phải bộ *kiểm định*.
>
> Cửa này chạy **TRƯỚC** CT-3, trên giá trị thô, so với trạng thái đã lọc.

### CT-14 — Tốc độ biểu kiến, chuẩn hoá theo face-width

Cho `F` và `C` (vector 2 chiều):

```
    s = norm(z_raw - z_hat_prev) / (W_hat * dt)         [face-width/s]
```

Cho `W` (vô hướng, đo theo tỷ lệ tương đối):

```
    s = abs(W_raw - W_hat_prev) / (W_hat_prev * dt)     [1/s]
```

Chia cho `W_hat` để bất biến khoảng cách; chia cho `dt` để bất biến FPS. Cùng
tinh thần với toàn hệ.

### CT-15 — Luật loại bỏ

```
    nếu s > S_max:                     LOẠI mẫu này
                                       KHÔNG chạy CT-3..CT-10
                                       KHÔNG cập nhật t_prev, z_prev
                                       giữ nguyên z_hat cũ
                                       n_reject = n_reject + 1

    ngược lại:                         CHẤP NHẬN
                                       n_reject = 0
                                       chạy CT-3..CT-10
```

### CT-16 — Tính tự chữa lành

Vì mẫu bị loại **không** cập nhật `t_prev`, nên `dt` ở frame sau lớn hơn, làm
`s` ở CT-14 nhỏ đi:

```
    s tỷ lệ nghịch với dt   ->  loại càng lâu, cửa càng nới ra
```

Cửa tự mở dần thay vì khoá vĩnh viễn. Đây là tính chất **có chủ ý**, không phải
tác dụng phụ.

### CT-17 — Lối thoát cứng

```
    nếu n_reject >= R_max = 5:
        RESET (CT-11) với chính z_raw
        n_reject = 0
```

Bảo đảm hệ không bao giờ kẹt: nếu operator **thật sự** đã dịch chuyển lớn, sau
5 frame hệ chấp nhận thực tại mới.

### Ngưỡng đề xuất `S_max`

| Tín hiệu | `S_max` | Căn cứ |
|---|---|---|
| `F` | 10 face-width/s | drone yaw 45°/s với FOV 60° làm cả khung trôi ~5 fw/s — phải để rộng |
| `W` | 1.5 /s | người đi bộ 2 m/s ở cự ly 5 m làm W đổi ~40%/s |
| `C` | 25 face-width/s | vung tay nhanh ≈ 3 fw trong 0.25 s = 12 fw/s, đỉnh ~20 |

Nhảy sang **người khác** tạo `s` cỡ 100–400 fw/s — cách ngưỡng cả bậc độ lớn,
phân tách rất an toàn.

> ⚠️ Ba số này suy từ hình học, **chưa hiệu chuẩn bằng dữ liệu thật**. Cần đo lại.

---

## PHẦN 6 — Áp cho năm tín hiệu

| tín hiệu | `f_min` | `beta` | `S_max` | nhịp cập nhật | ghi chú |
|---|---|---|---|---|---|
| `W` | 0.5 | 0.05 | 1.5 | 5 Hz | lọc mạnh nhất — **nằm dưới mẫu số** |
| `F.x`, `F.y` | 0.8 | 0.20 | 10.0 | 5 Hz | mặt di chuyển chậm |
| `C.x`, `C.y` | 1.0 | 0.50 | 25.0 | 20 Hz | mở nhanh nhất — phải theo kịp cú vung |

`f_d = 1.0 Hz` cho cả năm.

### CT-18 — Nhịp cập nhật khác nhau ★

```
    F, W   chỉ cập nhật khi frame_idx mod FACE_EVERY_N == 0
             ->  dt_face  ~  4/FPS  ~  0.20 s

    C      cập nhật MỖI frame có tay
             ->  dt_hand  ~  1/FPS  ~  0.05 s
```

> **Đây chính là lỗi mà v3 sửa.** `dt` phải đo từ lần cập nhật **của chính tín
> hiệu đó**, không phải từ frame video.
>
> Trên frame **không** chạy face detector: **không** gọi bộ lọc `F`, `W`. Tuyệt
> đối không nạp lại giá trị cũ — làm vậy sẽ ép `zd = 0` ở CT-4 và bóp chết khả
> năng thích nghi.

`alpha` sinh ra từ bộ tham số trên:

| tín hiệu | `f_min` | `beta` | `dt` | alpha lúc đứng yên | alpha khi \|zd\|=10/s |
|---|---|---|---|---|---|
| `W` | 0.50 | 0.05 | 0.20 | 0.386 | 0.557 |
| `F.x`, `F.y` | 0.80 | 0.20 | 0.20 | 0.501 | 0.779 |
| `C.x`, `C.y` | 1.00 | 0.50 | 0.05 | **0.239** | **0.653** |

Đọc hàng cuối: bộ lọc bàn tay đi từ `alpha = 0.239` (mượt) lên `alpha = 0.653`
(gần như không lọc) **một cách tự động**, chỉ dựa vào tốc độ. Đó là toàn bộ giá
trị của thuật toán, gói trong một hàng số.

---

## PHẦN 7 — Tính `u` và `u'`

### CT-19 — Vector tay trong hệ cơ thể

```
    u_hat = (C_hat - F_hat) / W_hat
```

Dùng **giá trị đã lọc** của cả ba. **Không lọc `u_hat` thêm lần nữa.**

> **Vì sao lọc ba đầu vào chứ không lọc `u`:** khi tay vung nhanh mà mặt đứng
> yên, lọc `u` sẽ thấy tốc độ lớn → mở toang → **nhiễu khuôn mặt lọt qua đúng
> lúc đang ra lệnh**. Lọc riêng thì `C` mở còn `F`, `W` vẫn đóng.
>
> Thêm nữa: `W` nằm dưới mẫu số nên nhiễu ở đó khuếch đại lên toàn hệ, phải lọc
> mạnh nhất. Ba tín hiệu ba bản chất, không thể dùng chung một bộ tham số.

### CT-20 — Vận tốc của `u` (quy tắc thương)

```
    u  = (C - F) / W

    u' = (C' - F')/W  -  u * (W'/W)
```

Cả ba đạo hàm `C'`, `F'`, `W'` **đã có sẵn** từ CT-6 của từng bộ lọc — không tốn
thêm phép tính nào.

Kiểm tra thứ nguyên: `(C'-F')/W` → `[1/s]`; `u*(W'/W)` → `[fw]*[1/s]`.
Khớp: `[face-width/s]`. ✓

Đây là **đặc trưng miễn phí cho A4** — TCN sẽ nhận `(u.x, u.y, u'.x, u'.y, ...)`
thay vì chỉ 2 số.

---

## PHẦN 8 — Thứ tự thực thi trong một frame

```
 1.  t = perf_counter()

 2.  nếu KHÔNG có operator lock:
         reset cả 5 bộ lọc  (CT-11 / R3)
         trả NO_LOCK, dừng

 3.  NHÁNH MẶT — chỉ khi frame_idx mod FACE_EVERY_N == 0 và có detection:
     3a.  cửa chặn cho W       (CT-14, CT-15)   -> nếu loại, bỏ qua 3b-3c
     3b.  cửa chặn cho F       (CT-14, CT-15)
     3c.  One-Euro cho W       (CT-3 .. CT-10)
     3d.  One-Euro cho F.x     (CT-3 .. CT-10)
     3e.  One-Euro cho F.y     (CT-3 .. CT-10)
     3f.  chặn dưới W          (CT-13)
     3g.  kiểm tra CT-17       (n_reject >= 5 -> reset)

     Nếu KHÔNG chạy face detector frame này: bỏ qua toàn bộ mục 3.
     W_hat, F_hat giữ nguyên. KHÔNG nạp lại giá trị cũ vào bộ lọc.

 4.  NHÁNH TAY — mỗi frame có tay:
     4a.  cửa chặn cho C       (CT-14, CT-15)   <- dùng W_hat mới nhất
     4b.  One-Euro cho C.x     (CT-3 .. CT-10)
     4c.  One-Euro cho C.y     (CT-3 .. CT-10)
     4d.  kiểm tra CT-17

     Nếu mất tay: KHÔNG reset bộ lọc C ngay. Chờ CT-12/R2 (dt > 0.5s) tự xử lý.

 5.  u_hat  = CT-19
 6.  ud_hat = CT-20

 7.  Trả (u_hat, ud_hat, trạng thái) cho tầng phân loại — không đổi gì phía dưới.
```

**Ràng buộc thứ tự bắt buộc:** mục 3 phải chạy trước mục 4, vì CT-14 của `C`
cần `W_hat` để chuẩn hoá.

---

## PHẦN 9 — Quy trình chỉnh tham số

Với **từng** tín hiệu, làm đúng thứ tự này:

```
Bước 1.  Đặt beta = 0.
         -> bộ lọc thoái hoá thành lọc cố định, f_min bị cô lập.
         Giữ tay/mặt ĐỨNG YÊN. Giảm f_min cho tới khi u_hat hết rung.
         Giảm quá tay thì thấy trễ ngay cả khi nhúc nhích nhẹ -> tăng lại chút.

Bước 2.  Giữ nguyên f_min vừa tìm được.
         Vung tay NHANH. Tăng beta cho tới khi hết cảm giác trễ.
         Tăng quá tay thì rung quay lại lúc đứng yên -> giảm lại.

Bước 3.  f_d để nguyên 1.0 Hz. Chỉ động vào nếu bước 2 không hội tụ.
```

Làm ngược thứ tự thì hai tham số nhiễu vào nhau và không hội tụ.

---

## PHẦN 10 — Vector kiểm chứng

Dùng để đối chiếu khi cài đặt. Tín hiệu `C.x`, `dt = 0.05 s`, `f_min = 1.0`,
`beta = 0.5`, `f_d = 1.0`.

```
alpha_d = alpha(f_d=1.0, dt=0.05) = 0.239057    (hằng số vì f_d cố định, dt đều)

 i    x_raw     zd thô    zd_hat      f_c     alpha      x_hat
 0   0.5000        --     0.0000       --    1.0000   0.500000    <- CT-11
 1   0.5120    0.2400     0.0574   1.0287    0.2442   0.502931
 2   0.5050   -0.1400     0.0102   1.0051    0.2400   0.503427
 3   0.6000    1.9000     0.4620   1.2310    0.2789   0.530359
 4   0.7000    2.0000     0.8296   1.4148    0.3077   0.582559
 5   0.8000    2.0000     1.1094   1.5547    0.3281   0.653912
```

Đọc cột `alpha`: nhiễu nhỏ ở i=1,2 chỉ đẩy `alpha` từ 0.239 lên 0.244 rồi tụt
về 0.240 — **bộ lọc không bị nhiễu đánh lừa**. Từ i=3 khi tín hiệu thật sự chạy,
`alpha` leo đều 0.279 → 0.308 → 0.328. Đúng hành vi mong muốn.

---

## PHẦN 11 — Bất biến phải kiểm

Sáu điều kiện dùng làm test sau khi cài:

```
 I1.  0 < alpha <= 1  luôn đúng, mọi f_c > 0, mọi dt > 0
 I2.  beta = 0  =>  One-Euro trùng khít lọc thông thấp cố định f_min
 I3.  Tín hiệu hằng số (không nhiễu)  =>  z_hat hội tụ về hằng số đó, zd_hat -> 0
 I4.  Cùng bộ tham số, chạy 15 FPS và 30 FPS
        =>  |trễ_15 - trễ_30|  <  1/15 s   (một chu kỳ lấy mẫu của FPS thấp)
 I4b. Đối chứng: EMA alpha cố định PHẢI trượt quá ngưỡng đó
        (nếu không thì I4 không chứng minh được gì)
 I5.  Nhảy 200 face-width/s  =>  bị CT-15 loại, z_hat không đổi
 I6.  Nhảy 200 fw/s giữ liên tục 5 frame  =>  CT-17 reset, z_hat nhận giá trị mới
```

> **Ghi chú về I4 — bản đầu của đặc tả này phát biểu sai.** Nó viết "độ trễ
> chênh < 1.5×", nhưng phép đo "mẫu đầu tiên vượt 90%" **bị lượng tử hoá theo
> chu kỳ lấy mẫu**: ở 15 FPS hai mẫu cách nhau 66.7 ms, nên mọi phép đo trễ đều
> mang sai số cỡ đó. So bằng **tỷ lệ** là so nhầm đại lượng.
>
> Số đo thật: One-Euro 97/63 ms — tỷ lệ 1.54× nghe như lệch nhiều, nhưng chênh
> tuyệt đối chỉ **33 ms, chưa bằng một chu kỳ lấy mẫu**, tức không phân biệt
> được với sai số phép đo. Trong khi EMA a=0.30 cho 363/130 ms, chênh **233 ms
> — gấp 3.5 lần chu kỳ lấy mẫu**, là khác biệt hành vi thật.
>
> Phát biểu theo chênh lệch tuyệt đối so với chu kỳ lấy mẫu vừa đúng vật lý,
> vừa phân biệt được One-Euro với EMA. Đã kiểm trong
> `ros2_ws/src/gesture_perception/gesture_perception/one_euro.py`.

---

## PHẦN 12 — Cái đặc tả này KHÔNG làm

Nói rõ để không kỳ vọng nhầm:

- **Không** dự đoán trước — vẫn còn ~80 ms trễ, chỉ ít hơn 230 ms hiện tại
- **Không** lấp khoảng trống khi mất tay — đó là việc của `CommandState` và watchdog
- **Không** đổi bất kỳ ngưỡng nào ở tầng phân loại (`UP_ENTER`, `SIDE_ENTER`, ...)
  — `u_hat` cùng đơn vị với `u` cũ, chỉ sạch hơn
- **Không** động tới ⑨–⑭ (hysteresis, `CommandState`, failsafe, MAVLink)

---

## Tóm tắt: 20 công thức

```
NỀN        CT-1  alpha từ f_c        CT-2  cập nhật EMA
ONE-EURO   CT-3  dt                  CT-4  zd thô           CT-5  alpha_d
           CT-6  zd_hat              CT-7  f_c thích nghi ★  CT-8  alpha
           CT-9  z_hat               CT-10 cập nhật trạng thái
KHỞI TẠO   CT-11 mẫu đầu             CT-12 ba điều kiện reset
           CT-13 chặn dưới W
CỬA CHẶN   CT-14 tốc độ biểu kiến    CT-15 luật loại        CT-16 tự chữa lành
           CT-17 lối thoát cứng
ÁP DỤNG    CT-18 nhịp cập nhật ★     CT-19 u_hat            CT-20 ud_hat
```

Hai công thức đánh dấu ★ mang toàn bộ giá trị: **CT-7** là thuật toán, **CT-18**
là chỗ sửa lỗi thật trong code hiện tại.

---

## Nguồn

Casiez, G., Roussel, N., Vogel, D. (2012).
*"1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive
Systems."* Proceedings of CHI 2012.

Tên "1€" nhại dòng thuật toán "$1 Recognizer" — nghĩa là "rẻ và đơn giản". Đây
là lựa chọn mặc định trong HCI, VR/AR hand tracking và motion capture. MediaPipe
cũng dùng One-Euro trong bộ làm mượt landmark nội bộ của nó.

---

## Ba điểm cần duyệt trước khi viết code

1. **Ba ngưỡng `S_max`** (PHẦN 5) — suy từ hình học, chưa có dữ liệu thật
2. **`dt_max = 0.5 s`** (CT-12/R2) — ngưỡng coi trạng thái cũ là vô nghĩa
3. **Quyết định không lọc `u_hat`** (CT-19) — lọc 3 đầu vào thay vì 1 đầu ra
