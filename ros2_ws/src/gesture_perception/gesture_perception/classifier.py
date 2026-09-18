"""
CLASSIFIER — .tflite + .labels.json -> nhãn cử chỉ cho MỘT frame.

Không import ROS, không import MediaPipe: chạy được trên laptop Windows để test.

HỢP ĐỒNG với training/4_train.py: file .labels.json chứa
    labels         thứ tự lớp đầu ra của model
    temperature    T, hiệu chuẩn trên dự đoán out-of-fold
    threshold      tau, chọn sao cho lệnh đã phát đạt độ chính xác mục tiêu
File này ĐỌC hai số đó, không tự đặt. Mỗi lần train lại chúng được hiệu chuẩn
lại trên chính tập dữ liệu đó, và tầng chạy phải đi theo.

k_vote trong labels.json KHÔNG dùng ở đây nữa. Luật "k frame liên tiếp" phụ
thuộc FPS (8 frame ở 8 fps = 1 s, ở 15 fps = 0.53 s). command_node thay bằng
thời gian giữ tính theo GIÂY, cùng ý nghĩa mà không đổi theo máy.
"""

import json
from pathlib import Path

import numpy as np

from .settings import UNKNOWN_GESTURE


def load_interpreter(model_path):
    """
    ai-edge-litert là đường chính (cả laptop lẫn Pi). Hai đường dự phòng chỉ để
    máy nào lỡ cài kiểu cũ vẫn chạy được — không cài TensorFlow lên Pi bao giờ.
    """
    errors = []
    for mod, attr in (("ai_edge_litert.interpreter", "Interpreter"),
                      ("tflite_runtime.interpreter", "Interpreter"),
                      ("tensorflow.lite", "Interpreter")):
        try:
            m = __import__(mod, fromlist=[attr])
            itp = getattr(m, attr)(model_path=str(model_path))
            itp.allocate_tensors()
            return itp, mod
        except ImportError as e:
            errors.append(f"{mod}: {e}")
    raise ImportError("Khong co thu vien doc .tflite nao:\n  " + "\n  ".join(errors)
                      + "\n  Cai: pip install -r requirements.txt")


class GestureClassifier:
    def __init__(self, model_path, tau_override=None):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Khong thay model: {model_path}")
        labels_path = model_path.with_suffix(".labels.json")
        self.meta = json.loads(labels_path.read_text(encoding="utf8"))
        self.labels = list(self.meta["labels"])
        self.name = model_path.stem

        self.itp, self.backend = load_interpreter(model_path)
        self._inp = self.itp.get_input_details()[0]["index"]
        self._out = self.itp.get_output_details()[0]["index"]

        self.temperature = float(self.meta.get("temperature") or 1.0)
        tau = self.meta.get("threshold")
        self.tau = float(tau) if tau is not None else None
        self.set_tau(tau_override)

    def set_tau(self, tau_override):
        """None = giữ ngưỡng của model. <= 0 = tắt cửa chặn (chỉ để đo)."""
        if tau_override is None:
            return
        self.tau = float(tau_override) if tau_override > 0 else None

    @property
    def unsafe(self):
        return self.meta.get("an_toan") is False or self.name.endswith("_UNSAFE")

    def probs(self, kp_norm):
        """(25, 2) đã chuẩn hoá PCK -> xác suất ĐÃ hiệu chuẩn nhiệt độ."""
        x = np.asarray(kp_norm, dtype=np.float32).reshape(1, 25, 2)
        self.itp.set_tensor(self._inp, x)
        self.itp.invoke()
        p = self.itp.get_tensor(self._out)[0].astype(np.float64)
        # p^(1/T) rồi chuẩn hoá lại ĐÚNG BẰNG chia logit cho T. Phải làm trước
        # khi so ngưỡng, vì tau được chọn trên xác suất đã hiệu chuẩn.
        if abs(self.temperature - 1.0) > 1e-9:
            q = np.power(np.clip(p, 1e-12, 1.0), 1.0 / self.temperature)
            p = q / q.sum()
        return p

    def predict(self, kp_norm):
        """-> (nhãn, độ tin cậy, xác suất). Dưới tau thì nhãn = AMBIGUOUS."""
        p = self.probs(kp_norm)
        k = int(p.argmax())
        conf = float(p[k])
        label = self.labels[k]
        if self.tau is not None and conf < self.tau:
            label = UNKNOWN_GESTURE
        return label, conf, p

    def top(self, p, n=3):
        order = np.argsort(p)[::-1][:n]
        return [self.labels[i] for i in order], [float(p[i]) for i in order]
