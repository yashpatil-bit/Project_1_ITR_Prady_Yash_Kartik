#!/usr/bin/env bash
# Full autonomous run:  driver + vision in the background, task manager in this terminal.
#   ./launch/run_autonomy.sh            real robot
#   ./launch/run_autonomy.sh --mock     simulated robot + synthetic camera
#   ./launch/run_autonomy.sh --mock --scene sim_scene_aligned.yaml
#   ./launch/run_autonomy.sh --no-confirm --speed 1.3
# In this terminal:  SPACE = EMERGENCY STOP,  q = abort.
set -e
MOCK=false; CONFIRM=true; SPEED=0.0; SHOW=true; SCENE=sim_scene.yaml
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mock) MOCK=true ;;
    --no-confirm) CONFIRM=false ;;
    --speed) SPEED="$2"; shift ;;
    --no-window) SHOW=false ;;
    --scene) SCENE="$2"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac; shift
done
WS="${PEG_WS:-$HOME/peg_ws}"
set +u
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
if ! ros2 pkg prefix peg_sorting >/dev/null 2>&1; then
  echo "peg_sorting not found. Build the workspace first (see README) or set PEG_WS."; exit 1
fi
ros2 launch peg_sorting bringup.launch.py mock:=$MOCK show_window:=$SHOW sim_scene:=$SCENE &
BRINGUP=$!
cleanup() { kill -INT $BRINGUP 2>/dev/null || true; wait $BRINGUP 2>/dev/null || true; }
trap cleanup EXIT
sleep 4
ros2 run peg_sorting task_manager --ros-args -p confirm:=$CONFIRM -p speed_scale:=$SPEED
