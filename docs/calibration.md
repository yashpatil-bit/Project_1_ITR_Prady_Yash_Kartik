# Calibration notes

Everything is expressed in the **robot base frame**: origin on the table directly below the joint‑1
axis (the centre of the rotating base), **x** forward (arm direction at joint1 = 0), **y** to the robot's left,
**z** up. Run the steps in this order the first time; afterwards only redo what changed.

| # | What | Tool | Writes | Redo when |
|---|---|---|---|---|
| 1 | Motor IDs / baud | `scan_dynamixel` | (edit `robot.yaml`) | never |
| 2 | Joint directions, gripper ticks | `teleop_keyboard` | `robot.yaml` | fingers changed |
| 3 | Tool length | ruler | `robot.yaml → kinematics.tool_length` | fingers changed |
| 4 | Observe pose | teleop key `P` | `robot.yaml → poses.observe` | camera moved |
| 5 | Camera intrinsics | factory (+ realsense-viewer on-chip calib.) | nothing | depth looks wavy |
| 6 | Camera → base | `calibrate_aruco`, then `calibrate_handeye` | `camera_calib.yaml` | camera or robot moved |
| 7 | Fine offset | `check_calibration` | `camera_calib.yaml → xyz_correction_m` | after 6 |
| 8 | Colours | `hsv_tuner` | `colors.yaml` | lighting changed |
| 9 | Table / board | ruler | `vision.yaml → table_z, board.thickness` | never |

Build with `colcon build --symlink-install`: then the tools write into the files in the repository,
and `git diff` shows what changed (commit the calibrated values!).

## 1–2. Motors, joint directions, gripper

* `ros2 run peg_sorting scan_dynamixel` → expect IDs 11–15 @ 1 000 000 baud.
* Start the driver, then teleop. `w` must move the gripper **away** from the robot, `a` to its **left**,
  `r` **up**. A joint moving the wrong way → flip its entry in `dynamixel.joint_signs`. If the arm's
  "zero" pose (all joints 0: arm straight up, then forward) looks bent, set `joint_offsets_rad`.
* Gripper: `gripper.open_ticks` / `closed_ticks` are raw Dynamixel positions (0–4095, 2048 = centre).
  To measure them: stop the driver, open *Dynamixel Wizard 2.0*, torque **off** motor 15, move the fingers by hand
  to fully open and to fully closed (fingertips just touching), and read *Present Position*. Use values about 20 ticks
  inside those ends. If the gripper is closing in the wrong direction, swap the two numbers.
* `gripper.current_limit` (×2.69 mA) sets the grip force (current-based position mode). 300 ≈ 0.8 A is firm but safe
  for 3-D printed pegs.
* `gripper.grasp_detect_fraction`: after closing, if the gripper is still more than this fraction open (0..1) it holds
  something. Close on nothing and on a peg with teleop `c`; it prints the final position. Pick a value in between.

## 3. Tool length (custom fingers)

`kinematics.tool_length` = distance along the gripper from the **joint‑4 axis** (the last pitch motor's shaft)
to the **point between the fingers where a peg is held** (about 1–1.5 cm above the fingertip ends). Stock gripper:
0.126 m. Measure with a ruler. A 5 mm error here gives a 5 mm grasp height error and, with a tilted gripper, a horizontal
error. The touch-point calibration (step 6B) absorbs any remaining horizontal error.

## 5. Intrinsics

The RealSense stores its factory intrinsics; the code reads them at start-up (`pyrealsense2`, colour stream, depth
aligned to colour). Nothing to enter. To check depth quality: `realsense-viewer` → 3-D view → the table must be flat.
If not, run *On-Chip Calibration* and *Tare* (More menu) with the camera looking at the flat plate.

## 6A. Extrinsics with the ArUco marker

1. Leave the marker flat on the plate where the camera sees it and the arm (in its observe pose) doesn't cover it.
2. Measure the printed black square side → `aruco.marker_length_m` (only used by the plain-PnP fallback; the
   default depth-plane method does not need it, but enter it anyway).
3. Measure the **marker centre** from the base origin: `x` forward, `y` left (negative = right). Tip: mark the floor
   point below joint 1 by jogging the gripper straight down at x = 0.15, y = 0 and measuring from there.
4. Run `ros2 run peg_sorting calibrate_aruco` (vision node stopped). Open the check image it prints.
   The **red arrow** is the marker x-axis: set `aruco.marker_in_base.yaw_deg` to its direction
   (0 = forward, 90 = left, 180 = towards the robot, -90 = right). Run again.
5. In the check image the **robot axes** (red x, green y, blue z) must start under joint 1, and the yellow workspace outline
   must lie on the plate. The tool prints the table height seen through the new calibration; it must be about 0 mm.

## 6B. Extrinsics from touch points (recommended after 6A)

`ros2 run peg_sorting calibrate_handeye` (driver + vision running). Use 5–6 spots: near and far, left and right,
including points near the board. At each spot:
ENTER (camera measures the peg) → jog over the peg, descend so the fingers sit where they would grasp → `c` → ENTER.
Residual RMS should be < 3 mm. It writes the transform and resets `xyz_correction_m`.

## 6C. Workspace calibration (recommended - the robot measures itself against the camera)

`ros2 run peg_sorting calibrate_workspace` (driver + vision running). Remove the board and all pegs but the
green one. The arm opens at a hand-over spot; stand the green peg between the fingers and press ENTER. The robot
then places the peg at ~8 spots and the camera measures each one. A 2-D affine map "camera -> robot" is fitted and
saved as `planar_correction`; it absorbs arm model errors (on our arm the real reach was ~8 % shorter than the
standard Manipulator X model) as well as camera errors. Residual RMS must be < 6 mm or nothing is saved.
Restart the vision node afterwards, then put the board back.

## 7. Fine offset

`ros2 run peg_sorting check_calibration`: the arm hovers 1 cm above each detected peg; jog to the true centre if needed
and press ENTER. The mean offset goes to `xyz_correction_m`.

## 8. Colours

`ros2 run peg_sorting hsv_tuner`: the window shows only the pixels inside the current range. Click a peg to
print its HSV value, widen the range around it until the whole peg is lit and nothing else, `n` for the next
colour, `s` to save. Do the **board** too (wood + the dark burnt edges of the holes), because the socket detector
uses it. Lock exposure and white balance in `vision.yaml → camera` if the lab lighting changes.
