#!/usr/bin/env bash
# Keyboard teleoperation:  driver + vision (for the assist keys) in the background, teleop here.
#   ./launch/run_teleop.sh [--mock] [--no-vision]
set -e
MOCK=false; VISION=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mock) MOCK=true ;;
    --no-vision) VISION=false ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac; shift
done
WS="${PEG_WS:-$HOME/peg_ws}"
set +u
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
if [ "$VISION" = true ]; then
  ros2 launch peg_sorting bringup.launch.py mock:=$MOCK &
else
  ros2 launch peg_sorting driver.launch.py mock:=$MOCK &
fi
BRINGUP=$!
cleanup() { kill -INT $BRINGUP 2>/dev/null || true; wait $BRINGUP 2>/dev/null || true; }
trap cleanup EXIT
sleep 4
ros2 run peg_sorting teleop_keyboard
