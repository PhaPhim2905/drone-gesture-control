#!/usr/bin/env python3
"""
Kiểm chuỗi NHẬN DIỆN không cần camera, không cần ROS.

    python ros2_ws/src/gesture_perception/test/test_perception_core.py

Chuỗi phải đi trọn:  BODY25 giả -> One-Euro -> pck_format -> .tflite -> nhãn.
Thay cho tools/test_tflite.py.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gesture_perception import pck_format as pf  # noqa: E402
from gesture_perception import one_euro  # noqa: E402
from gesture_perception.classifier import GestureClassifier  # noqa: E402
from gesture_perception.perception_core import GesturePerception  # noqa: E402
from gesture_perception.settings import NO_OPERATOR, UNKNOWN_GESTURE  # noqa: E402

MODEL = HERE.parents[1] / "gesture_bringup" / "models" / "v3_pck_body25.tflite"


def fake_pose(arms="t", s=200.0):
    return pf._fake_body25(s, arms)


def test_pck_selftest():
    assert pf._selftest() == 0


def test_one_euro_selftest():
    assert one_euro.main() == 0


def test_model_loads_with_contract():
    clf = GestureClassifier(MODEL)
    assert clf.labels, "labels rong"
    assert clf.tau is not None and 0.5 < clf.tau < 1.0
    assert clf.temperature > 0
    assert not clf.unsafe


def test_probs_sum_to_one_and_fast():
    import time
    clf = GestureClassifier(MODEL)
    n = pf.normalize_body25(fake_pose("up"))
    p = clf.probs(n)
    assert abs(p.sum() - 1.0) < 1e-5 and p.shape == (len(clf.labels),)
    t0 = time.perf_counter()
    for _ in range(200):
        clf.probs(n)
    assert (time.perf_counter() - t0) / 200 < 0.005, "suy luan > 5 ms/frame"


def test_tau_gate():
    clf = GestureClassifier(MODEL, tau_override=0.999999)
    label, conf, _ = clf.predict(np.random.default_rng(0).uniform(-0.7, 0.7, (25, 2)))
    assert label == UNKNOWN_GESTURE
    clf.set_tau(0)       # tắt cửa chặn
    label, _, _ = clf.predict(pf.normalize_body25(fake_pose("t")))
    assert label in clf.labels


def test_core_chain_without_mediapipe():
    core = GesturePerception(MODEL, one_euro=True)
    r = None
    for i in range(20):
        r = core.classify_body25(fake_pose("up"), i * 0.1)
    assert r.kp_norm is not None and r.kp_norm.shape == (25, 2)
    assert r.label in core.clf.labels + [UNKNOWN_GESTURE]
    assert len(r.top_labels) == 3


def test_one_euro_same_label_as_raw_on_static_pose():
    """Tư thế tĩnh không nhiễu: bật hay tắt lọc phải ra CÙNG nhãn."""
    for arms in ("down", "t", "up"):
        a = GesturePerception(MODEL, one_euro=True, tau_override=0)
        b = GesturePerception(MODEL, one_euro=False, tau_override=0)
        for i in range(15):
            ra = a.classify_body25(fake_pose(arms), i * 0.1)
            rb = b.classify_body25(fake_pose(arms), i * 0.1)
        assert ra.label == rb.label, f"{arms}: {ra.label} != {rb.label}"
        assert np.abs(ra.kp_norm - rb.kp_norm).max() < 1e-6


def test_mode_gate_only_guided():
    from gesture_perception.mode_gate import ModeGate
    g = ModeGate("GUIDED", state_timeout=2.0)
    assert g.check(0.0)[0] is False, "chua co state thi KHONG nhan dien"
    g.on_state(True, "LOITER", 1.0)
    ok, why = g.check(1.1)
    assert not ok and "GUIDED" in why
    g.on_state(False, "GUIDED", 2.0)
    assert g.check(2.1)[0] is False, "FCU chua ket noi"
    g.on_state(True, "GUIDED", 3.0)
    assert g.check(3.1) == (True, "")
    assert g.check(5.5)[0] is False, "mat /mavros/state > 2 s phai dong cong"
    assert ModeGate("").check(0.0) == (True, ""), "require_mode rong = luon nhan dien"


def test_latest_frame_drops_stale_frames():
    """Xử lý chậm hơn camera: luôn lấy frame MỚI NHẤT, frame cũ bị bỏ, không dồn trễ."""
    import threading
    import time
    from gesture_perception.frame_source import LatestFrame

    class FakeCap:
        def __init__(self):
            self.n = 0

        def read(self):
            time.sleep(0.005)               # camera 200 fps
            self.n += 1
            return True, np.full((4, 4, 3), self.n % 256, np.uint8)

        def release(self):
            pass

    src = LatestFrame(cap=FakeCap())
    try:
        ages, seqs = [], []
        for _ in range(8):
            item = src.get(timeout=1.0)
            assert item is not None
            frame, t_cap, _stamp, seq, name = item
            ages.append(time.monotonic() - t_cap)
            seqs.append(seq)
            time.sleep(0.05)                # xử lý 20 fps
        assert name == "camera"
        assert all(b > a for a, b in zip(seqs, seqs[1:])), "khong duoc lay lai frame cu"
        assert src.dropped > 20, f"phai bo frame cu, dropped={src.dropped}"
        assert max(ages) < 0.03, f"tuoi frame lay ra phai nho, max={max(ages)*1000:.0f} ms"
    finally:
        src.stop()
    assert not threading.active_count() > 50


def test_empty_body_is_no_operator():
    core = GesturePerception(MODEL)
    r = core.classify_body25(np.zeros((25, 3)), 0.0)
    assert r.label == NO_OPERATOR and r.kp_norm is None


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [ok]   {name}")
        except AssertionError as e:
            fails += 1
            print(f"  [FAIL] {name}: {e}")
    print("=" * 60)
    print("CHUOI DAY DU CHAY THONG" if not fails else f"THAT BAI {fails}/{len(tests)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
