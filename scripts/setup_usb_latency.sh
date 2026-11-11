#!/usr/bin/env bash
# Make the U2D2 (FTDI) serial adapter fast: latency timer 16 ms -> 1 ms, permanently (udev rule).
# Without this every Dynamixel read takes ~16 ms and motion control gets choppy.
set -e
RULE='ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"'
echo "$RULE" | sudo tee /etc/udev/rules.d/99-u2d2-latency.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
for f in /sys/bus/usb-serial/devices/ttyUSB*/latency_timer; do
  [ -e "$f" ] && echo 1 | sudo tee "$f" >/dev/null && echo "$f = $(cat $f)"
done
sudo usermod -aG dialout "$USER" && echo "Added $USER to dialout (log out and in again once)."
