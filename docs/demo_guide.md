# Public demo guide

## The day before
- [ ] `git pull`, `colcon build --symlink-install`, `python3 -m pytest -q peg_sorting/test`
- [ ] Mock dry run: `./launch/run_autonomy.sh --mock --scene sim_scene_aligned.yaml --no-confirm`
- [ ] Charge the laptop; bring a USB-3 cable for the RealSense and the U2D2 cable.

## On site (about 15 minutes)
1. Plug in: 12 V to the power hub, U2D2 USB, RealSense to a **USB-3** port.
2. Terminal 1: `ros2 launch peg_sorting bringup.launch.py` (driver + vision + live window).
3. Terminal 2: `ros2 run peg_sorting teleop_keyboard`. `b` (observe pose); check the arm is out of the camera image.
4. If the camera or robot was moved: stop the bringup, run `ros2 run peg_sorting calibrate_aruco`, restart the bringup,
   then `ros2 run peg_sorting check_calibration`.
5. Lighting different from the lab? `hsv_tuner` (vision stopped) for any colour that is not detected.
6. Board: `ros2 run peg_sorting suggest_board_pose` and, if allowed, place the board as suggested; watch the yaw numbers
   in the vision window go to about 0.
7. Terminal 3: `ros2 run peg_sorting estop` and keep a finger on SPACE.

## Autonomous run (50 pts)
```bash
./launch/run_autonomy.sh            # or: ros2 run peg_sorting task_manager (if the bringup is already running)
```
- Read the scene report aloud: 4 pegs, 4 sockets, shapes, predicted yaw errors.
- Before ENTER, fix every `!!` / `WARNING` line in the scene report, then restart the task manager:
  - **Peg spacing.** Seen from the robot, each peg needs about 3 cm of free space on both sides,
    because the open jaws come down there. Keep side-by-side pegs at least 8 cm apart centre to
    centre, or stagger them at least 4 cm front/back. At 5.8 cm the jaw pushed the yellow peg over
    in the lab.
  - **Square yaw.** The square must show a yaw error of 4 degrees or less. Nudge the board
    (turn and slide it) while watching the numbers next to the sockets in the vision window, or
    run `ros2 run peg_sorting suggest_board_pose`.
- ENTER starts the timer. Hands off; SPACE if anything looks wrong.
- At the end the summary prints pegs placed, time and failures. Show `python3 scripts/evaluate_timing.py --last 1`.

## Teleop run (30 pts / fallback)
```bash
./launch/run_teleop.sh
```
ENTER (timer) → `v` → for each peg: `1`..`4`, `f` down, `c`, `r` up, `5`..`8`, `f` slowly, `o`, `r`, `y`. ESC writes the log.

## If something goes wrong
| Symptom | Quick action |
|---|---|
| Peg missed by a few mm, every time | `check_calibration` (2 min) |
| A colour not detected | `hsv_tuner` for that colour |
| Peg rests on the board instead of going in | yaw: rotate the board (watch the numbers in the window) |
| Arm hits a peg during transport | raise `task.yaml → transport.obstacle_margin` |
| Anything unsafe | SPACE, then switch off the 12 V if needed |
