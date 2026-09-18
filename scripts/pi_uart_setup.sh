#!/usr/bin/env bash
# Bật UART trên GPIO của Pi 5 để nói chuyện với Pixhawk 6X qua cổng TELEM2.
# Chạy MỘT LẦN trên Pi, rồi khởi động lại:
#
#   ./scripts/pi_uart_setup.sh
#   sudo reboot
#
# DÂY (chỉ 3 sợi, KHÔNG nối chân 5V):
#   Pixhawk TELEM2 (JST-GH 6 chân)        Pi 5 (header 40 chân)
#     chân 2  TX  ───────────────────►  chân 10  GPIO15 RXD
#     chân 3  RX  ◄───────────────────  chân 8   GPIO14 TXD
#     chân 6  GND ────────────────────  chân 6   GND
#   Chân 1 (5V), 4 (CTS), 5 (RTS) để trống. Mức 3.3 V hai bên khớp nhau.
#
# THAM SỐ PIXHAWK 6X cần có (anh tự đặt trong Mission Planner, script KHÔNG ghi):
#   SERIAL2_PROTOCOL = 2     MAVLink2
#   SERIAL2_BAUD     = 921   921600
#   BRD_SER2_RTSCTS  = 0     không dùng CTS/RTS vì chỉ nối 3 dây
#
# Sau reboot kiểm tra:
#   python3 pixhawk_tools/diagnostic.py --port /dev/ttyAMA0 --baud 921600
set -euo pipefail

CFG=/boot/firmware/config.txt
CMDLINE=/boot/firmware/cmdline.txt
DEV=/dev/ttyAMA0

if ! grep -q "Raspberry Pi 5" /proc/device-tree/model 2>/dev/null; then
  echo "Canh bao: khong phai Pi 5 ($(tr -d '\0' < /proc/device-tree/model 2>/dev/null)). Ten cong co the khac."
fi

# 1. Pi 5: GPIO14/15 là UART0 -> /dev/ttyAMA0. Cổng debug 3 chân riêng là ttyAMA10,
#    nên bật uart0 không đụng tới console debug.
if grep -qE '^\s*dtparam=uart0=on' "$CFG"; then
  echo ">>> 1. $CFG da co dtparam=uart0=on"
else
  sudo cp "$CFG" "$CFG.bak.$(date +%Y%m%d%H%M%S)"
  printf '\n[all]\n# Pixhawk TELEM2 (scripts/pi_uart_setup.sh)\ndtparam=uart0=on\n' | sudo tee -a "$CFG" > /dev/null
  echo ">>> 1. da them dtparam=uart0=on vao $CFG (co ban sao .bak)"
  NEED_REBOOT=1
fi

# 2. Không cho Linux mở console đăng nhập trên cổng này (sẽ tranh byte với MAVLink)
if grep -qE 'console=(ttyAMA0|serial0)' "$CMDLINE"; then
  if [[ "$(readlink -f /dev/serial0 2>/dev/null)" == "$DEV" ]] || grep -q 'console=ttyAMA0' "$CMDLINE"; then
    sudo cp "$CMDLINE" "$CMDLINE.bak.$(date +%Y%m%d%H%M%S)"
    sudo sed -i -E 's/console=(ttyAMA0|serial0),[0-9]+ ?//g' "$CMDLINE"
    echo ">>> 2. da bo console tren $DEV khoi $CMDLINE (co ban sao .bak)"
    NEED_REBOOT=1
  else
    echo ">>> 2. console=serial0 dang tro cong debug, khong dung toi $DEV - giu nguyen"
  fi
else
  echo ">>> 2. $CMDLINE khong co console tren $DEV"
fi
sudo systemctl disable --now serial-getty@ttyAMA0.service 2>/dev/null || true

# 3. Quyền mở cổng serial
if id -nG "$USER" | grep -qw dialout; then
  echo ">>> 3. $USER da trong nhom dialout"
else
  sudo usermod -aG dialout "$USER"
  echo ">>> 3. da them $USER vao dialout"
  NEED_REBOOT=1
fi

if [[ "${NEED_REBOOT:-0}" == 1 ]]; then
  echo
  echo "XONG. Khoi dong lai: sudo reboot"
elif [[ -e "$DEV" ]]; then
  echo
  echo "XONG, $DEV da co. Kiem tra: python3 pixhawk_tools/diagnostic.py --port $DEV --baud 921600"
else
  echo
  echo "Chua thay $DEV du da cau hinh. Khoi dong lai: sudo reboot"
fi
