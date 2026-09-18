#!/usr/bin/env bash
# THỬ CẢ CHUỖI, TỰ ĐỘNG, KHÔNG CẦN ĐỨNG TRƯỚC CAMERA:
#   SITL (instance riêng) + MAVROS + perception_node (video take) + command_node
#
#   ./sim/chain_test/run.sh
#   VIDEO_DIR=/duong/dan/data/video ./sim/chain_test/run.sh
#
# Kịch bản (pilot giả bằng pymavlink, cổng SERIAL1 của SITL):
#   A  STABILIZE rồi LOITER   -> perception PHẢI IM LẶNG (cổng GUIDED)
#   B  pilot gạt GUIDED + arm -> video NEGATIVE, ASSUME_GUIDANCE, TAKEOFF, HOVER, LAND
#                                phát lặp theo nhịp thật
#   Đạt khi: không cử chỉ nào lúc chưa GUIDED, MO QUYEN, cất cánh > 1.5 m,
#   LAND bằng cử chỉ, perception ngừng khi sang LAND.
#
# Không đụng SITL/Gazebo anh đang mở: instance -I6 (TCP 5820/5822), ROS_DOMAIN_ID 77.
# Chạy sau MỖI lần train / đổi tau / sửa perception hay command.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WIN_VIDEO="/mnt/d/CT_UAV/3. Project/3.1 Drone_AI_Gesture/4.Cauhinh/Drone_test_voice_whisper/data/video"
VIDEO_DIR="${VIDEO_DIR:-$([[ -d "$REPO/data/video" ]] && echo "$REPO/data/video" || echo "$WIN_VIDEO")}"
pick() { ls "$VIDEO_DIR"/"$1"__*.mp4 2>/dev/null | head -1; }
VIDS=""
for g in NEGATIVE ASSUME_GUIDANCE TAKEOFF HOVER LAND; do
  f=$(pick $g); [[ -z "$f" ]] && { echo "Thieu video $g trong $VIDEO_DIR (quay bang 1_record.py --save-video)"; exit 1; }
  VIDS="${VIDS:+$VIDS,}$f"
done

set +u
source "$REPO/scripts/env.sh" > /dev/null
set -u
export ROS_DOMAIN_ID=77
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
A="${ARDUPILOT_DIR:-$HOME/ardupilot}"; DP=$A/Tools/autotest/default_params
D=$HOME/sitl_run/chain_test; rm -rf "$D"; mkdir -p "$D"; cd "$D"
printf 'DISARM_DELAY,0\n' > test.parm      # video TAKEOFF có thể tới sau >10 s

cleanup() {
  kill ${PM:-} 2>/dev/null; kill -INT ${PL:-} 2>/dev/null; sleep 3
  kill ${PL:-} 2>/dev/null; kill ${PS:-} 2>/dev/null; wait 2>/dev/null
}
trap cleanup EXIT

echo ">>> SITL -I6"
"$A/build/sitl/bin/arducopter" -w -I6 --model quad --defaults "$DP/copter.parm,$D/test.parm" \
  --home -35.363261,149.165230,584,353 > sitl.out 2>&1 &
PS=$!
sleep 3
echo ">>> ROS: mavros + perception (video) + command"
ros2 launch gesture_bringup sitl.launch.py fcu_url:=tcp://127.0.0.1:5820 \
  video_path:="$VIDS" flip:=false > launch.log 2>&1 &
PL=$!
sleep 6
python3 -u "$HERE/monitor.py" 330 > monitor.log 2>&1 &
PM=$!
echo ">>> pilot gia (toi da ~4 phut). Log: $D"
timeout 340 "$HOME/venv-ardupilot/bin/python" -u "$HERE/pilot.py" tcp:127.0.0.1:5822 230 > pilot.log 2>&1
sleep 3

L() { sed -E 's/^\[[a-z_]+-[0-9]+\] //; s/\[INFO\] \[[0-9.]+\] //' "$1"; }
echo; echo "=== su kien"
L launch.log | grep -E "MediaPipe san sang|BAT DAU NHAN DIEN|NGUNG NHAN DIEN|MO QUYEN|CAT CANH|takeoff|HA CANH|set_mode|KHOA|EXTENDED"
echo; echo "=== do tre (lan do cuoi)"
L launch.log | grep -E "nhan dien: cam .*GUIDED 100%" | tail -1
L launch.log | grep -E "tre chup frame" | tail -1

pass=0; fail=0
check() { if eval "$2"; then echo "  DAT   $1"; pass=$((pass+1)); else echo "  TRUOT $1"; fail=$((fail+1)); fi; }
before=$(awk '/FCU mode LOITER -> GUIDED/{exit} /GESTURE [1-9][0-9]* msg/{n++} END{print n+0}' monitor.log)
peak=$(grep -oE "peak=[0-9.]+" pilot.log | tail -1 | cut -d= -f2)
echo; echo "=== ket qua"
check "chua GUIDED: 0 cu chi duoc phat"            "[[ ${before:-1} -eq 0 ]]"
check "GUIDED: bat dau nhan dien"                   "grep -q 'BAT DAU NHAN DIEN' launch.log"
check "ASSUME_GUIDANCE mo khoa"                     "grep -q 'MO QUYEN DIEU KHIEN' launch.log"
check "TAKEOFF bang cu chi, len > 1.5 m (${peak:-0} m)" "awk -v p=${peak:-0} 'BEGIN{exit !(p>1.5)}'"
check "LAND bang cu chi"                            "grep -q 'set_mode LAND (LAND): FCU nhan' launch.log"
check "sang LAND: ngung nhan dien"                  "grep -q 'NGUNG NHAN DIEN: dang LAND' launch.log"
check "ha canh xong, disarm"                        "grep -q 'DA CAT CANH VA HA CANH XONG' pilot.log"
echo "  => $pass dat, $fail truot"
exit $fail
