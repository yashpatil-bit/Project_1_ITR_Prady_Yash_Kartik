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
# workspace: $PEG_WS, else the repo itself (colcon build run inside the repo), else ~/peg_ws
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "$PEG_WS" ]; then WS="$PEG_WS"
elif [ -f "$REPO/install/setup.bash" ]; then WS="$REPO"
else WS="$HOME/peg_ws"; fi
set +u
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
if [ "$VISION" = true ]; then
  ros2 launch peg_sorting bringup.launch.py mock:=$MOCK &
else
  ros2 launch peg_sorting driver.launch.py mock:=$MOCK &
fi
BRINGUP=$!
# (background jobs of a script ignore SIGINT, so stop the launch with SIGTERM)
cleanup() {
  kill -TERM $BRINGUP 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 $BRINGUP 2>/dev/null || break; sleep 0.5; done
  kill -KILL $BRINGUP 2>/dev/null || true
  pkill -f "peg_sorting/lib/peg_sorting/(driver|vision)" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
sleep 4
ros2 run peg_sorting teleop_keyboard
