# RUNBOOK v3 — thu dữ liệu, huấn luyện, kiểm tra

Trình tự chạy cho hệ nhận diện **toàn thân** (v3). Mở file này ra là chạy được
ngay, không cần nhớ gì.

---

## 0. Quy tắc phải nhớ — HAI môi trường Python

Dự án cố ý dùng hai môi trường tách rời nhau:

| Môi trường | Có gì | Dùng cho |
|:---|:---|:---|
| `python` (hệ thống 3.10) | opencv, mediapipe, numpy, ai-edge-litert | thu dữ liệu, dựng tập, chạy realtime, bay |
| `.venv-train` | tensorflow 2.19 | **chỉ** huấn luyện |

Hai bên **không chia sẻ package nào**. Chúng chỉ chia sẻ đúng hai thứ: định dạng
vector 50 chiều của `ros2_ws/src/gesture_perception/gesture_perception/pck_format.py`, và file `.tflite`. Đó là lý do
TensorFlow không bao giờ kéo đổ được numpy/mediapipe của môi trường chạy thật —
và cũng là lý do trên Pi không bao giờ phải cài TensorFlow.

**Chỉ bước huấn luyện dùng `.venv-train`. Mọi bước khác dùng `python`.**

Chạy nhầm thì chương trình tự báo và in đúng lệnh cần gõ. Kiểm tra nhanh:

```powershell
python -c "import sys; print(sys.executable)"
# phải ra ...\Programs\Python\Python310\python.exe
```

Nếu ra đường dẫn có `.venv-train` thì gõ `deactivate`, hoặc mở terminal mới
(`.vscode/settings.json` đã tắt auto-activate của VS Code).

---

## 1. Xác định camera — chạy một lần

```powershell
python -m gesture_perception.doctor --probe   (chay trong ros2_ws\src\gesture_perception)
python -m gesture_perception.doctor --probe   (chay trong ros2_ws\src\gesture_perception)

```

Che tay trước ống kính cam USB, cái nào tối là nó. `q` để thoát. Nhớ số index,
mọi lệnh sau đều dùng. Dưới đây giả sử là **1**.

> Trên Windows, DirectShow và Media Foundation **đánh số camera khác nhau**.
> `--backend auto` (mặc định) tự thử lần lượt. Nếu vẫn không mở được, thử
> `--backend msmf`.

---

## 2. Đặt camera và đứng cho đúng

| | |
|:---|:---|
| **Khoảng cách** | 2.0 – 2.5 m |
| **Chiều cao ống kính** | ngang ngực đến ngang vai (1.2 – 1.4 m) |
| **Hướng** | nằm ngang, hoặc chếch xuống một chút |

Đừng để cam trên bàn cạnh laptop. Ống kính thấp nhìn hếch lên + người đứng gần
= tay giơ lên ra ngoài khung. Đã đo ở phiên thu hỏng đầu tiên: **67% số mẫu
không có cổ tay trong đó**, mà cổ tay chính là thứ định nghĩa cử chỉ.

Cơ sở con số: FOV ngang ~60°, dọc ~46°. Dang tay chữ T rộng ~1.7 m cộng lề cần
2.0 m ⇒ tối thiểu 1.73 m. Từ hông tới đầu ngón tay khi giơ lên ~1.15 m cộng lề
cần 1.4 m ⇒ tối thiểu 1.65 m. Lấy 2.0 – 2.5 m là dư cho cả hai chiều.

### Kiểm tra trước khi bấm SPACE

Vào vị trí, làm **hai tư thế cực đoan nhất** rồi nhìn bảng bên phải:

1. **Giơ hai tay thẳng lên quá đầu** → `co tay T OK   co tay P OK` phải **xanh, không nhấp nháy**
2. **Dang tay chữ T** → vẫn xanh
3. `|max|` ≤ 1.0 (chữ xanh)

Lọt hai tư thế đó thì bảy tư thế còn lại chắc chắn lọt. Hiện băng đỏ
**"CO TAY RA NGOAI KHUNG"** thì lùi thêm — lúc đó máy **không ghi mẫu nào**,
không sợ dính dữ liệu bẩn.

---

## 3. Thu dữ liệu

```powershell
python training\1_record.py --cam 1
```

Thêm `--save-video` nếu muốn lưu cả video thô (xem mục 7).

### Danh mục mười lớp — giai đoạn này dùng **sáu**

Sáu lớp đánh dấu ✅ là **roster v1**, bộ mà `train_pck.py` cam kết nhận. Thiếu
một trong sáu thì train **bị chặn**. Bốn lớp hướng bay để giai đoạn sau: chúng
nằm trên cùng một cung, chỉ cách nhau 45°, cần dữ liệu dày hơn nhiều.

| | Lớp | Tư thế | Lệnh |
|:---:|:---|:---|:---|
| ✅ | `ASSUME_GUIDANCE` | hai tay giơ **thẳng** lên, dang rộng bằng vai (chữ V) | mở quyền điều khiển |
| ✅ | `HOVER` | hai tay dang ngang **duỗi hết biên** (chữ T) | giữ chỗ |
| ✅ | `TAKEOFF` | dang ngang, **gập khuỷu 90°** — cẳng tay dựng đứng (chữ U) | cất cánh |
| | `MOVE_UP` | hai tay dang ngang, chếch **lên** ~45° | lên cao |
| | `MOVE_DOWN` | hai tay dang ngang, chếch **xuống** ~45° | hạ thấp |
| | `MOVE_LEFT` | tay **trái** dang ngang, tay phải buông xuôi | sang trái |
| | `MOVE_RIGHT` | tay **phải** dang ngang, tay trái buông xuôi | sang phải |
| ✅ | `LAND` | hai tay bắt chéo **trước bụng** | hạ cánh |
| ✅ | `STOP` | hai tay bắt chéo **trên đầu** | dừng khẩn |
| ✅ | `NEGATIVE` | mọi thứ khác | — |

Phím chọn lớp chạy theo **thứ tự hiện trên màn hình**, nên khi truyền `--classes`
thì phím 1..6 bám đúng thứ tự mình gõ, không phải thứ tự trong bảng này.

**Ba cặp dễ nhầm** — mỗi cặp chỉ khác nhau ở đúng một đại lượng:

| Cặp | Giống nhau | **Khác nhau ở** |
|:---|:---|:---|
| `TAKEOFF` ↔ `HOVER` | bắp tay đều ngang vai | **cẳng tay**: dựng thẳng / nằm ngang |
| `TAKEOFF` ↔ `ASSUME_GUIDANCE` | cổ tay đều cao hơn vai | **góc khuỷu**: gập 90° / duỗi thẳng |
| `LAND` ↔ `STOP` | đều bắt chéo | **độ cao**: dưới bụng / trên đầu |

### Phím

| Phím | Việc |
|:---:|:---|
| `1`–`9`, `0` | chọn lớp (đang thu thì tự kết thúc take) |
| `SPACE` | bắt đầu / kết thúc một **take** |
| `u` | xoá take vừa thu (chỉ trong phiên này) |
| `q` / `ESC` | thoát |

### Thu thử ít lớp trước

Không nhất thiết phải thu cả bộ mới chạy được vòng. Dùng `--classes` để lấy một
tập con — phím chọn lớp tự đánh số lại theo đúng thứ tự gõ:

```powershell
python training\1_record.py --cam 1 --classes TAKEOFF,LAND --save-video
```

> ⚠️ Model 2 lớp **luôn** trả về một trong hai lớp đó, kể cả khi không có ai
> trước camera hoặc người đang làm tư thế hoàn toàn khác. Softmax không có chỗ
> để nói "không giống lớp nào". Đó là mô hình để **kiểm tra đường ống**, không
> phải để điều khiển. Muốn nó biết im lặng thì phải có lớp `NEGATIVE`.

---

## 3b. Quy trình thu — số liệu cụ thể

### Bao nhiêu take mỗi lớp

**Tối thiểu 4, nhắm 8.** Con số rơi ra từ cách đánh giá.

`train_pck.py` chạy **kiểm chứng chéo K-fold theo take** (mặc định K = 4). Mỗi
lớp chia take của chính nó vòng tròn vào 4 rổ; fold thứ *i* lấy rổ *i* làm test,
rổ kế làm val, phần còn lại train. Mỗi take được làm test **đúng một lần**, nên
gộp 4 fold lại là có dự đoán cho **toàn bộ** tập mà không mẫu nào tự chấm mình.

| Số take/lớp | | |
|:---:|:---|:---|
| < 4 | **bị CHẶN** | không đủ rổ để chia, train dừng lại |
| 4 | chạy được | mỗi rổ 1 take — kết quả dao động mạnh |
| 6 | ổn | |
| **8** | **khuyến nghị** | mỗi rổ 2 take, độ lệch giữa các fold mới đáng tin |

So với cách chia một lần của bản trước, K-fold dùng **mọi** take để đánh giá chứ
không chỉ 1–2 take, nên con số ổn định hơn hẳn ở cỡ dữ liệu này. Đổi lại nó train
K lần — với tập vài nghìn mẫu thì mất thêm chừng một phút.

Quan trọng hơn số take: **mỗi take phải là một điều kiện KHÁC**. `train_pck.py`
đo và báo `TRUNG DIEU KIEN` nếu hai take của cùng một lớp gần nhau hơn cả độ
rộng nội bộ của chính chúng.

⚠️ Đứng xa hay gần **không** tạo ra điều kiện khác: phép chuẩn hoá chia cho độ
dài chi nên đã khử sạch khoảng cách. Thứ thật sự tạo khác biệt là **xoay người**
(±20–30°), **lệch khung** (đứng trái/phải khung hình), đổi nền, đổi đèn, đổi áo.

### Một take diễn ra thế nào

Anh đứng cách máy 2–4 m nên **không có tay nào rảnh để bấm SPACE**. Vì vậy một
take chạy tự động từ đầu đến cuối:

```
SPACE  →  ĐẾM NGƯỢC 8 s  →  GHI 15 s  →  TỰ DỪNG
          (đi ra, vào tư thế)   (đứng yên trong tư thế)
```

Chỉnh bằng `--delay` và `--duration`. Bíp báo:

| Tiếng | Nghĩa |
|:---|:---|
| 3 tiếng ngắn | còn 3, 2, 1 giây — **vào tư thế ngay** |
| 1 tiếng cao dài | **bắt đầu ghi** — từ giây này giữ nguyên |
| 2 tiếng thấp | **xong** — được thả tay |

Ở 4 m thì tai nghe được, còn mắt thì không đọc nổi chữ trên màn hình và cũng
không thể vừa giữ tư thế vừa cúi nhìn.

**Trong 15 giây ghi:**

| Giây | Làm gì |
|:---:|:---|
| 0 – 3 | giữ yên đúng tư thế chuẩn |
| 3 – 6 | nhích hai cổ tay lên/xuống quanh vị trí chuẩn, biên độ ~5 – 10 cm |
| 6 – 9 | xoay người qua lại ±10° quanh góc của take |
| 9 – 12 | bước tới rồi lùi ~30 cm |
| 12 – 15 | về tư thế chuẩn, giữ yên |

Tay **không rời tư thế** trong suốt 15 giây đó. Chỉ sau tiếng bíp đôi mới thả.

**Đừng đứng bất động.** Đã đo: phiên thu tốt cho `std = 0.043`. Dưới `0.008` thì
kiến trúc `hybrid` (có BatchNorm) sập xuống mức đoán bừa — số đo trong docstring
`build_hybrid` của `training/4_train.py`.

Nhưng **đừng đi ra khỏi tư thế**. Khoảng giữa hai cử chỉ (tay đang đưa từ chỗ
này sang chỗ kia) **không** thuộc lớp nào — nó thuộc `NEGATIVE`, và phải thu
riêng có chủ đích.

### Ma trận 8 take

Góc dương = xoay người sang **phải của anh** (vai phải lùi ra sau).

| Take | Khoảng cách | Góc xoay người | Vị trí trong khung | Camera |
|:---:|:---:|:---:|:---|:---|
| 1 | 2.0 m | 0° | giữa | ngang ngực |
| 2 | 3.0 m | 0° | giữa | ngang ngực |
| 3 | 2.0 m | **+25°** | giữa | ngang ngực |
| 4 | 2.0 m | **−25°** | giữa | ngang ngực |
| 5 | 2.5 m | 0° | **lệch hẳn sang trái khung** | ngang ngực |
| 6 | 2.5 m | 0° | **lệch hẳn sang phải khung** | ngang ngực |
| 7 | 3.0 m | ±15° đảo qua lại | giữa | **cao hơn đầu, chếch xuống ~15°** |
| 8 | 4.0 m | tự do 0 … ±30° | vừa làm vừa dịch chuyển | ngang ngực |

**Giới hạn góc: ±30°.** Quá mức đó, một cánh tay che cánh tay kia trong ảnh
chiếu, MediaPipe hạ `visibility` và mẫu bị cửa chặn cổ tay loại. Người đứng
nghiêng hơn 45° so với drone thì **không đọc được cử chỉ** — trường hợp đó thuộc
`NEGATIVE`, không thuộc lớp cử chỉ nào.

**Take 7 quan trọng nhất** và cũng dễ bỏ qua nhất: đó là góc nhìn thật của drone
— nhìn từ trên xuống. Kê camera lên nóc tủ, lên thang, hoặc lên chồng sách cao
hơn đầu.

**Take 8** cố tình lộn xộn: vừa làm vừa đi, góc đổi liên tục. Nếu take này rơi
vào tập test thì điểm số sẽ thấp hơn hẳn các take khác — và đó mới là con số gần
thực tế bay nhất.

### Nếu phòng không đủ 4 m

Đổi thành 2.0 / 2.5 / 3.0 m, giữ nguyên phần góc. Khoảng cách quan trọng vì hai
lý do: xa thì khớp nhiễu hơn (ít pixel hơn), và xa thì thấy cả chân còn gần thì
chân ra ngoài khung. **Không** phải vì tỷ lệ — `normalize_body25` đã khử tỷ lệ
rồi. Ba mức cách nhau 0.5 m là đủ tạo ra hai hiệu ứng đó.

### Lớp `NEGATIVE` — thu ngược lại

8 take, mỗi take một nội dung **khác hẳn**, không phải một tư thế giữ nguyên:

| Take | Nội dung |
|:---:|:---|
| 1 | đứng yên buông tay, đổi tư thế đứng vài lần |
| 2 | đi qua đi lại trong khung |
| 3 | gãi đầu, vuốt tóc, chỉnh áo |
| 4 | cầm điện thoại, nghe điện thoại |
| 5 | khoanh tay, chống nạnh, ngồi xổm, cúi nhặt đồ |
| 6 | xoay lưng, đứng nghiêng 60 – 90° |
| 7 | **một tay giơ nửa chừng** — giữ ở đủ mọi độ cao |
| 8 | **đưa tay chậm từ cử chỉ này sang cử chỉ khác, lặp đi lặp lại** |

Take 7 và 8 là hai take giá trị nhất trong cả bộ dữ liệu. Chúng dạy mạng nhận ra
**khoảng giữa** — trạng thái tồn tại trong mọi lần đổi lệnh. Không có chúng, đưa
tay từ `HOVER` sang `STOP` sẽ đi xuyên qua vùng của `MOVE_UP`, và drone leo lên
một nhịp trước khi dừng khẩn cấp.

Lớp này **không** bị cửa chặn cổ tay — cứ để tay khuất tự nhiên.

### Tổng thời gian

| | 2 lớp (thử) | 10 lớp (đầy đủ) |
|:---|:---:|:---:|
| số take | 16 | 80 |
| thu thuần | 2,7 phút | 13 phút |
| kể cả di chuyển, đặt lại camera | ~10 phút | ~45 phút |

Có thể thoát rồi chạy lại nhiều lần. Mỗi lần ghi thêm một file
`data/gestures/session_*.jsonl`, bảng đếm trên màn hình cộng dồn tất cả phiên cũ.
Chia làm nhiều buổi còn **tốt hơn** — khác ánh sáng, khác áo, khác nền.

---

## 4. Dựng tập

```powershell
python training\3_build_dataset.py
```

Đọc mọi `.jsonl` trong `data/gestures/` → ghi đè `data/dataset_v3.npz`.

**Đọc bảng nó in ra trước khi đi tiếp.** Ba điều kiện phải đạt:

| Cột | Yêu cầu |
|:---|:---|
| số dòng | đủ **9 lớp** |
| `take` | **≥ 4** mỗi lớp — **8 là mốc nên nhắm** (xem mục 3b) |
| `thieu tay` | gần 0 |

Chưa đạt thì quay lại mục 3 thu bù. Đừng train vội — dữ liệu hỏng thì con số
đẹp cũng vô nghĩa.

---

## 5. Huấn luyện

### 5.1 Kiểm dữ liệu TRƯỚC — chạy bằng python chính

```powershell
python training\4_train.py --npz data\dataset_v3.npz --check
```

Không cần TensorFlow. Nó chấm dữ liệu rồi thoát, chia làm hai mức:

| | Nghĩa là |
|:---|:---|
| `x` **CHAN** | train xong sẽ ra model không dùng được, hoặc một con số không đo được gì. **Train dừng lại.** |
| `!` **LUU Y** | dùng được nhưng yếu ở một chỗ cụ thể, và chỗ đó được nói tên |

Các lỗi CHAN: thiếu lớp trong bộ sáu, thiếu `NEGATIVE`, lớp dưới 4 take, lớp
dưới 150 mẫu, hai cử chỉ chồng lấn quá 2%.

Muốn train bất chấp thì thêm `--force` — nhưng khi đó file model mang hậu tố
**`_UNSAFE`**, và `live_demo.py` hiện một băng cảnh báo khi nạp nó. Tên file tự
khai ra, không dựa vào trí nhớ ai.

### 5.2 Train — bước duy nhất đổi python

```powershell
.venv-train\Scripts\python.exe training\4_train.py --npz data\dataset_v3.npz
```

Xuất ra `models/v3_pck_body25.{keras,tflite,labels.json}`.

`--arch` có ba lựa chọn:

| | Ghi chú |
|:---|:---|
| `pck` | **mặc định, dùng cái này trước.** Không có BatchNorm nên không sập khi tập còn nhỏ |
| `hybrid` | có BatchNorm, bền hơn với nhiễu và **chặn đầu vào lạ tốt hơn** — đổi sang khi đã đủ nhiều take |
| `light` | bản nhỏ, chỉ để đối chứng độ bền |

### 5.3 Đọc kết quả — bốn khối, theo thứ tự quan trọng

**1. `accuracy theo frame  xx.xx% +/- y.yy`**
Trung bình và độ lệch giữa các fold. Độ lệch > 6% nghĩa là con số trung bình
**chưa ổn định** — thường vì các take không đủ khác nhau.

**2. Bảng ngưỡng `tau`**
Model được phép nói "không biết". Ngưỡng chọn sao cho đạt **đồng thời** hai mục
tiêu: độ chính xác của *lệnh đã phát* ≥ 99% (`--muc-tieu`), và chặn được ≥ 90%
đầu vào lạ (`--muc-tieu-la`). Cột `do phu` là cái giá phải trả — người dùng cảm
nhận nó thành "phải giữ tay lâu hơn".

**3. `LENH SAI MOI PHUT` ← con số quyết định**
Mô phỏng đúng luật của tầng quyết định: phải có `--k-vote` frame **liên tiếp**
cùng nhãn và cùng vượt ngưỡng thì mới phát lệnh. Đây mới là thứ đo được rủi ro
thật, vì accuracy theo frame và số lệnh sai **không tỷ lệ với nhau**: nhầm rải
rác từng frame thì bộ đếm không bao giờ đầy; nhầm thành chùm thì một lệnh sai đi
thẳng ra drone.

**4. `CAP DE NHAM`**
Ba cặp cử chỉ chỉ khác nhau ở một đại lượng. Ma trận chung dễ che mất chúng — 5
lần nhầm trên 1200 mẫu trông như bụi — nhưng chính chúng là chỗ hệ gãy khi bay.

Ba con số `tau`, nhiệt độ `T`, `k_vote` được ghi vào `labels.json`.
`live_demo.py` **đọc** chúng, không tự đặt ngưỡng.

---

## 6. Xem nó nhận diện

```powershell
python training\5_export_to_ros.py   # roi tren Pi: ros2 launch gesture_bringup camera_only.launch.py
```

Tự lấy model `v3_*` mới nhất trong `models/`.

Ba con số trên màn hình:

| | Đọc thế nào |
|:---|:---|
| `pose ms` | thời gian MediaPipe — toàn bộ chi phí thật của hệ |
| `clf ms` | thời gian bộ phân loại, đo được ~0.03 ms = 0.2% cột trên |
| `conf` | **không phải xác suất đúng**, xem cảnh báo dưới |

> Đã đo trên model 20 lớp: 25 điểm ngẫu nhiên cho entropy **0.183**, tư thế thật
> cho **0.553**. Mạng **tự tin hơn khi nhận rác**. Softmax luôn cộng lại bằng 1
> nên không có chỗ để nói "không giống lớp nào". Vì vậy không được dùng ngưỡng
> confidence hay entropy làm cửa chặn — phải dạy nó lớp `NEGATIVE` bằng dữ liệu
> thật.

### Quay vòng

Thấy lớp nào hay nhầm → quay lại **mục 3** thu thêm take cho đúng lớp đó → **4**
→ **5** → **6**. Không phải thu lại từ đầu, dữ liệu cũ vẫn nằm nguyên.

Đổi kiến trúc thì chỉ chạy lại mục 5 với `--arch` khác, không đụng dữ liệu.

---

## 7. Video — lưu ở đâu, dùng thế nào

Mặc định **không lưu video**, chỉ lưu 25 toạ độ khớp ra `.jsonl`. Bật bằng cờ:

```powershell
python training\1_record.py --cam 1 --save-video
```

Mỗi take thành một file:

```
data/video/HOVER__20260907_160840_3.mp4
           └─nhãn─┘  └────mã take────┘
```

Nhãn nằm **trước dấu `__`**. Đổi tên file là đổi nhãn, không có file cấu hình
nào phải sửa theo.

Trích đặc trưng từ video:

```powershell
python training\3b_build_from_video.py
python training\3_build_dataset.py
```

`build_from_video.py` chạy MediaPipe trên từng video → ghi `.jsonl` vào
`data/gestures/`, y như vừa thu bằng camera. Từ đó đi tiếp bình thường.

### Vì sao đáng lưu video

`.jsonl` đã là dữ liệu thô ở **tầng khớp**, đủ để đổi phép chuẩn hoá mà không
phải thu lại. Nhưng ba thứ nó không cứu được, và cả ba đều sẽ xảy ra:

1. **Đổi `POSE_MODEL_COMPLEXITY`** từ 0 (Lite) sang 1 → khớp sẽ khác. Không có
   video thì phải mời người ra đứng thu lại toàn bộ.
2. **Đổi `LM_MIN_VISIBILITY`**, hoặc đổi hẳn sang bộ ước lượng tư thế khác.
3. **Dữ liệu ngoài sân**, ở cự ly bay thật 5 – 10 m, quay bằng điện thoại hoặc
   bằng chính camera trên drone. Không có đường nào khác để đưa loại dữ liệu đó
   vào tập huấn luyện.

Cái giá: ~1–2 MB mỗi take 10 giây, cả bộ 50–100 MB. `data/video/` nằm trong
`.gitignore`.

### Dùng video quay sẵn từ nguồn khác

```powershell
python training\3b_build_from_video.py --dir D:\quay_ngoai_san --flip
```

Đặt tên file theo đúng quy ước `NHAN__ma_take.mp4`.

> ⚠️ **Soi gương.** `record_dataset.py` lật ảnh **trước** khi ghi, nên video nó
> lưu ra đã lật rồi — mặc định `build_from_video.py` **không** lật thêm. Video
> từ nguồn khác thì tuỳ: camera trước điện thoại thường đã tự soi gương, camera
> sau thì không. Xem lại video: người trong đó giơ tay phải mà anh thấy ở phía
> trái màn hình thì **không** cần `--flip`; ngược lại thì cần. **Sai chỗ này là
> `MOVE_LEFT` và `MOVE_RIGHT` hoán đổi cho nhau, và drone bay ngược hướng ra
> lệnh.**

---

## 8. Bảng tra nhanh

```powershell
# xac dinh camera
python -m gesture_perception.doctor --probe   (chay trong ros2_ws\src\gesture_perception)


# thu du lieu  (them --save-video neu muon giu video)
python training\1_record.py --cam 1 --classes TAKEOFF,LAND,HOVER,STOP,ASSUME_GUIDANCE,NEGATIVE

# tu cham tu the vua thu
python training\2_check_poses.py

# video -> jsonl   (chi khi co dung --save-video, hoac co video quay san)
python training\3b_build_from_video.py

# jsonl -> npz
python training\3_build_dataset.py

# kiem du lieu TRUOC, bang python CHINH (khong can TensorFlow)
python training\4_train.py --npz data\dataset_v3.npz --check

# huan luyen   <- BUOC DUY NHAT dung .venv-train
.venv-train\Scripts\python.exe training\4_train.py --npz data\dataset_v3.npz

# xem realtime
python training\5_export_to_ros.py   # roi tren Pi: ros2 launch gesture_bringup camera_only.launch.py
```

---

## 9. Các file không nằm trong vòng này

| File | Việc |
|:---|:---|
| `ros2_ws/src/gesture_perception/test/test_perception_core.py` | kiểm chuỗi `One-Euro → pck_format → tflite`, không cần camera |
| `ros2_ws/src/gesture_perception/gesture_perception/one_euro.py` | tầng lọc One-Euro, **đã nối** vào perception_node (Body25Filter) |
| `ros2_ws/src/gesture_perception/gesture_perception/pck_format.py` | cầu nối MediaPipe ↔ BODY25, chạy thẳng file để tự kiểm |

---

## 10. Sự cố thường gặp

| Triệu chứng | Nguyên nhân | Xử lý |
|:---|:---|:---|
| `No module named cv2` | terminal đang bật `.venv-train` | `deactivate`, hoặc mở terminal mới |
| `No module named tensorflow` | train bằng python chính | dùng `.venv-train\Scripts\python.exe` |
| không mở được camera | sai index, sai backend, **hoặc cam USB tuột dây** | `python -m gesture_perception.doctor --probe   (chay trong ros2_ws\src\gesture_perception)` — nó in cả danh sách Windows; dòng `KHONG CO MAT` nghĩa là thiết bị đó đang không kết nối, cắm lại dây / đổi cổng USB |
| băng đỏ `CO TAY RA NGOAI KHUNG` | đứng quá gần / cam quá thấp | lùi ra, kê cam cao lên |
| `CHI CO 1 LOP` | chưa thu đủ bộ sáu | thu tiếp rồi `build_dataset.py` lại |
| `THIEU LOP: ...` | tập không đủ bộ sáu của roster v1 | thu các lớp còn thiếu |
| `KHONG CO LOP NEGATIVE` | chưa thu `NEGATIVE` | bắt buộc phải có — xem mục 3b |
| `x <lop>: n take, can >= 4` | lớp đó chưa đủ take cho K-fold | thu thêm take, mỗi take một điều kiện |
| `TRUNG DIEU KIEN` | hai take của cùng lớp thu ở cùng chỗ | thu lại một trong hai ở **góc xoay** khác |
| `CHUA CHAN DU dau vao la` | `NEGATIVE` chưa bao được vùng giáp ranh | thu thêm `NEGATIVE` ở các **tư thế trung gian** |
| model tên `*_UNSAFE` | đã train bằng `--force` | chỉ để kiểm chuỗi chạy, **không đem ra bay** |
| `dataset_v3.npz van con, dung tu du lieu DA BI XOA` | xoá `.jsonl` nhưng `.npz` cũ còn | `del data\dataset_v3.npz` |
