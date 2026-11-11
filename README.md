# Tri-Wizard Trial 1 — Peg Sorting with the Manipulator X

Vision-guided pick-and-place of four pegs (triangle, square, hexagon, circle) into
their matching sockets with a **ROBOTIS OpenMANIPULATOR-X (Manipulator X)** and an
**Intel RealSense RGB-D camera**, on **ROS 2 Humble**.

* **Full autonomy**: colour + depth perception → shape classification → socket detection →
  constant-pitch top-grasp planning with obstacle avoidance → guarded insertion → verification and retries.
* **Keyboard teleoperation**: Cartesian jogging, vision-assisted "go above peg / socket" keys and a timing log.
* **Safety**: one-key emergency stop (SPACE) in every tool, plus a dedicated `estop` terminal.
  Guarded insertion stops when the arm is blocked.
* **Evaluation**: every run is logged (CSV + JSON). `scripts/evaluate_timing.py` reports success rate and timing.
* **Mock mode**: the whole stack runs on a laptop with a simulated arm and a rendered RGB-D camera,
  so you can test everything before going to the lab.

Team: Prady, Yash, Kartik

---

## Contents

1. [Hardware](#1-hardware)
2. [Repository layout](#2-repository-layout)
3. [Setup (OS, versions, dependencies)](#3-setup)
4. [Try it without hardware (mock mode)](#4-try-it-without-hardware-mock-mode)
5. [First bring-up on the real robot](#5-first-bring-up-on-the-real-robot)
6. [Calibration](#6-calibration) (camera intrinsics, camera→robot extrinsics, colours)
7. [Running: autonomous and teleop](#7-running)
8. [Orientation: important limit of a 4-DOF arm](#8-orientation-important-limit-of-a-4-dof-arm)
9. [Safety and emergency stop](#9-safety-and-emergency-stop)
10. [Evaluation: success rate and timing](#10-evaluation-success-rate-and-timing)
11. [How it works](#11-how-it-works)
12. [Parameter files](#12-parameter-files)
13. [Optional: YOLOv8 shape detection](#13-optional-yolov8-shape-detection)
14. [Troubleshooting](#14-troubleshooting)
15. [Tests](#15-tests)

---

## 1. Hardware

| Item | Details |
|---|---|
| Robot | ROBOTIS OpenMANIPULATOR-X: 4 × Dynamixel XM430-W350-T joints (IDs 11–14) + gripper motor (ID 15), 1 Mbps, protocol 2.0. The gripper has custom long fingers. |
| Motor interface | U2D2 USB adapter on a U2D2 Power Hub Board, 12 V supply ("SAUMYA" brick). Shows up as `/dev/ttyUSB0`. |
| Camera | One Intel RealSense depth camera on the aluminium post to the robot's side, looking down at the work area at an angle. The code supports any D4xx (D415/D435/D455). |
| Scene | Metal plate in front of the robot. MDF socket board with triangle, square, hexagon and circle holes. ArUco marker on the plate, used for calibration. |
| Pegs | ~10 cm tall, 4–6 cm across. **yellow = triangle, blue = square, orange = hexagon, green = circle**. |

Frames: **robot base frame** = origin on the table surface directly **below the joint‑1 (base rotation) axis**,
**x forward** (where the arm points with joint1 = 0), **y to the robot's left**, **z up**. Every
position in configs and logs is in metres in this frame.

## 2. Repository layout

This follows the recommended structure, turned into two ROS 2 packages:

```
├── README.md                          ← you are here
├── requirements.txt / requirements-yolo.txt
├── launch/
│   ├── run_autonomy.sh                ← full autonomous run (one command)
│   └── run_teleop.sh                  ← keyboard teleop (one command)
├── scripts/
│   ├── evaluate_timing.py             ← success-rate / timing report over logged runs
│   └── setup_usb_latency.sh           ← makes the U2D2 fast (1 ms latency) + dialout group
├── docs/
│   ├── demo_guide.md                  ← step-by-step for the public demo
│   └── calibration.md                 ← detailed calibration notes
├── docker/Dockerfile                  ← ROS 2 Humble image (mock runs / CI)
├── peg_sorting_interfaces/            ← ROS 2 msgs/srv: Peg, Socket, GetScene
└── peg_sorting/                       ← main ROS 2 package (ament_python)
    ├── config/
    │   ├── colors.yaml                ← HSV thresholds per peg colour + board
    │   ├── camera_calib.yaml          ← camera→robot extrinsics + ArUco marker info
    │   ├── vision.yaml                ← camera settings, workspace, detection params
    │   ├── robot.yaml                 ← Dynamixel IDs, gripper ticks, kinematics, poses, speed limits
    │   ├── task.yaml                  ← grasp / place / transport / retry parameters
    │   └── sim_scene*.yaml            ← synthetic scenes for mock mode and tests
    ├── launch/bringup.launch.py, driver.launch.py
    ├── peg_sorting/
    │   ├── vision/      color_segmentation.py, shape_detection.py, peg_detection.py,
    │   │                socket_detection.py, scene.py, geometry.py, camera.py, aruco.py, yolo_classifier.py
    │   ├── planning/    kinematics.py (FK/IK), trajectory.py, path_planner.py,
    │   │                pick_place_planner.py, board_placement.py
    │   ├── control/     dynamixel_bus.py (hardware), robot_interface.py (ROS client)
    │   ├── nodes/       driver_node.py, vision_node.py, task_manager.py, teleop_keyboard.py, estop.py
    │   ├── tools/       scan_dynamixel, hsv_tuner, calibrate_aruco, calibrate_handeye,
    │   │                check_calibration, suggest_board_pose, capture_dataset, train_yolo, evaluate_timing
    │   ├── sim/         synthetic_scene.py (ray-cast RGB-D renderer for mock mode / tests)
    │   └── utils/       config, transforms, keyboard, run_logger
    └── test/            pytest unit + end-to-end vision tests
```

## 3. Setup

### Tested versions

| Component | Version |
|---|---|
| OS | **Ubuntu 22.04 LTS** (x86-64) |
| ROS 2 | **Humble Hawksbill** |
| Python | 3.10 (Ubuntu 22.04 system Python) |
| numpy | 1.21.x (system; must stay < 2 for ROS Humble) |
| OpenCV | opencv-contrib-python 4.10.0.84 (pip) |
| pyrealsense2 | ≥ 2.54 (pip) |
| dynamixel-sdk | ≥ 3.7.31 (pip, or apt `ros-humble-dynamixel-sdk`) |
| control_msgs | apt `ros-humble-control-msgs` |

### Step by step

```bash
# 1) ROS 2 Humble (skip if installed) - https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
sudo apt update && sudo apt install -y ros-humble-ros-base python3-colcon-common-extensions python3-pip
```

```bash
# 2) ROS packages used by this project
sudo apt install -y ros-humble-control-msgs ros-humble-dynamixel-sdk python3-serial
```

```bash
# 3) Workspace + this repository
mkdir -p ~/peg_ws/src && cd ~/peg_ws/src
git clone https://github.com/yashpatil-bit/Project_1_ITR_Prady_Yash_Kartik.git
```

```bash
# 4) Python dependencies
pip3 install -r ~/peg_ws/src/Project_1_ITR_Prady_Yash_Kartik/requirements.txt
```

```bash
# 5) RealSense USB permissions (udev rules), then unplug/replug the camera
sudo curl -sSL -o /etc/udev/rules.d/99-realsense-libusb.rules https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

```bash
# 6) U2D2: dialout group + 1 ms USB latency (log out and back in once afterwards)
~/peg_ws/src/Project_1_ITR_Prady_Yash_Kartik/scripts/setup_usb_latency.sh
```

```bash
# 7) Build. --symlink-install matters: calibration tools then write straight into the repo's config files.
cd ~/peg_ws && source /opt/ros/humble/setup.bash && colcon build --symlink-install
```

```bash
# 8) In every new terminal (or add both lines to ~/.bashrc)
source /opt/ros/humble/setup.bash && source ~/peg_ws/install/setup.bash
```

The launch scripts assume the workspace is at `~/peg_ws`; for another location, set `export PEG_WS=/path/to/ws`.

## 4. Try it without hardware (mock mode)

The mock mode swaps the motors for a simulated arm and the RealSense for a ray-cast renderer of
the scene in `config/sim_scene.yaml` (same table, board, ArUco marker and pegs as the lab). Pegs
attach to the simulated gripper and drop into a socket only if they are centred **and aligned**.

```bash
cd ~/peg_ws/src/Project_1_ITR_Prady_Yash_Kartik
./launch/run_autonomy.sh --mock                                   # board placed like the lab photo
```

```bash
./launch/run_autonomy.sh --mock --scene sim_scene_aligned.yaml    # board where suggest_board_pose puts it
```

```bash
./launch/run_teleop.sh --mock                                     # practise teleop
```

Results in mock mode (ROS 2 Humble): with the board as in the lab photo, the planner warns that the square and
hexagon will arrive 13–17° rotated; they end up resting on the board and the run reports **2/4**. With the
board where `suggest_board_pose` puts it (and the triangle turned as the scene report says): **4/4 in about 50 s,
0 failures**. Section 8 explains why.

## 5. First bring-up on the real robot

1. **Power**: 12 V on, U2D2 USB to the laptop. Find the motors:
   ```bash
   ros2 run peg_sorting scan_dynamixel
   ```
   Expected: IDs 11, 12, 13, 14, 15 at 1000000 baud. If different, edit `config/robot.yaml → dynamixel`.
2. **Driver**. The arm holds its current pose when torque comes on; it doesn't jump.
   ```bash
   ros2 launch peg_sorting driver.launch.py
   ```
3. **Teleop** (second terminal). Press `i` to print joints and tool pose, then `h` for home.
   ```bash
   ros2 run peg_sorting teleop_keyboard
   ```
   Check the directions: `w` moves the gripper forward, `a` left, `r` up. If a joint turns the wrong way, flip its sign in
   `robot.yaml → dynamixel.joint_signs`.
4. **Gripper ticks** (custom fingers!). Open/close with `o` / `c` and watch them. If the fingers do not fully open or
   close, or they strain, adjust `robot.yaml → gripper.open_ticks / closed_ticks` (raw motor positions; `i`
   prints the gripper value 0..1). To read raw ticks, stop the driver, turn off torque in the Dynamixel Wizard,
   move the fingers by hand and read *Present Position*. See `docs/calibration.md`.
5. **Tool length** (custom fingers!). `robot.yaml → kinematics.tool_length` is the distance from the joint‑4 axis
   (the last pitch motor) to the **point between the fingertips where the peg is held**, roughly 1.5 cm above the
   fingertip ends. The stock gripper is 0.126 m. Measure yours with a ruler; this number sets how accurately you grasp.
6. **Observe pose**. Jog the arm somewhere it is **out of the camera's view** of the plate and board (e.g. raised and
   turned away from the camera), then press `P` in teleop to save it as `poses.observe`.

## 6. Calibration

The PDF asks for calibration notes for both RGB-D cameras. **This setup uses one RealSense** (one camera
is enough; the code has a single `vision` node and nothing depends on a second view).

### 6.1 Intrinsics (per camera)

RealSense cameras are factory calibrated. `pyrealsense2` reads the colour intrinsics from the device every
time, and depth is aligned to colour in software, so **there is nothing to enter**. If depth looks noisy or
wavy (a flat table is not flat in `realsense-viewer`'s 3-D view), run *On-Chip Calibration* and
*Tare Calibration* in `realsense-viewer` (More → On-Chip Calibration). Settings in `config/vision.yaml → camera`:
848×480 @ 30 fps, *High Accuracy* preset, spatial + temporal filters. Fix `exposure` / `white_balance` if
the lab lighting makes colours flicker.

### 6.2 Extrinsics: camera → robot base (required)

Pick one of the methods; B and C refine A.

**A. ArUco marker (5 minutes).** The marker already on the plate is used.
1. Measure the black square's side → `camera_calib.yaml → aruco.marker_length_m`.
2. Measure the marker **centre** in the robot base frame: x forward from the joint‑1 axis, y to the left → `aruco.marker_in_base.x/y`.
   `yaw_deg` is the direction of the marker's x-axis, counter-clockwise from the robot's forward direction. It is a
   multiple of 90° if the marker is square to the robot. The tool draws that axis as a **red arrow** on its check image,
   so run it once, look at the arrow, then set `yaw_deg` (0 = arrow points forward, 90 = arrow points to the robot's left).
3. Stop the vision node (it owns the camera) and run:
   ```bash
   ros2 run peg_sorting calibrate_aruco
   ```
   The dictionary and ID are detected automatically. The marker pose is computed from the colour corners **and**
   a RANSAC fit of the table plane in the depth image, averaged over 30 frames. In simulation this is
   about 2 mm / 0.4° accurate, versus about 10 mm / 1.4° for plain `solvePnP`. The tool reports the table height
   seen through the new calibration (should be about 0 mm) and saves a check image with the robot axes and
   workspace drawn on it.

**B. Touch points (most accurate, about 10 minutes).** Driver and vision running:
```bash
ros2 run peg_sorting calibrate_handeye
```
Put one peg at 4–6 spots spread over the workspace. At each spot the camera measures it, then you jog the gripper onto it and
close it. A rigid transform (Kabsch) is fitted and the residuals are printed. Because it uses the same tool point as grasping, errors in tool
length and marker measurement cancel out.

**C. Check and fine-tune.**
```bash
ros2 run peg_sorting check_calibration
```
The arm hovers 1 cm above every detected peg. If it is consistently off, jog it to the centre and press ENTER. The
average offset is saved as `camera_calib.yaml → xyz_correction_m`.

### 6.3 Colour thresholds (parameter file `config/colors.yaml`)

```bash
ros2 run peg_sorting hsv_tuner      # vision node stopped; n = next colour, s = save, click = print HSV
```
Tune each peg colour (yellow, blue, orange, green) and the board (MDF + the burnt edges). Pixels that are
too low to be a peg are ignored automatically (depth), so the orange-ish burnt edges of the holes cannot be
confused with the orange peg.

## 7. Running

### 7.1 Autonomous (50 pts)

```bash
cd ~/peg_ws/src/Project_1_ITR_Prady_Yash_Kartik
./launch/run_autonomy.sh                       # add --speed 1.3 to go faster, --no-confirm to skip the ENTER prompt
```
What happens:
1. The arm goes to the observe pose. Pegs and sockets are detected (averaged over 5 frames) and a **scene report** is printed:
   each peg's colour, shape, position, its socket and the predicted yaw error.
2. Press **ENTER** to start the timed run.
3. For each peg (nearest first): plan → open → approach above the peg → **vertical descent** → settle → close
   (**grasp check**: did the gripper stop on something?) → lift → transport around the other pegs → **slow guarded insertion**
   (if it is blocked, back off and retry with ±4 mm offsets) → settle → release → retreat.
4. Back to the observe pose: **verify** the peg is in its socket (x-y within 25 mm and *not* sitting one board thickness
   too high). If not, retry (max 2 retries; a known orientation problem is not retried).
5. The summary and log path are printed.

Keys in that terminal: **SPACE = emergency stop**, `q` = abort after the current step.

### 7.2 Teleoperation (30 pts)

```bash
./launch/run_teleop.sh            # --no-vision to run without the camera
```

| Key | Action | Key | Action |
|---|---|---|---|
| `w` `s` / ↑ ↓ | forward / back (x) | `o` / `c` | open / close gripper |
| `a` `d` / ← → | left / right (y) | `h` / `b` | home / observe pose |
| `r` `f` | up / down (z) | `P` | save current joints as observe pose |
| `t` `g` | tool pitch steeper / flatter | `+` `-` | step size 2–40 mm |
| `q` `e` | rotate base | `i` | print joints + tool pose |
| `v` | detect and list pegs + sockets | `1`–`4` | go above peg n (pre-grasp) |
| `5`–`8` | go above the socket for peg n‑4 (insertion height, same pitch) | ENTER | start the run timer |
| `y` / `n` | log last peg placed / failed | **SPACE** | **EMERGENCY STOP** (`x` releases) |
| ESC | quit (writes the teleop log) | | |

Typical teleop cycle: `v` → `1` (above peg) → `f`… (down) → `c` → `r`… (up) → `5` (above its socket) → `f`… (slowly in) → `o` → `r` → `y`.
**Keep the pitch constant (don't press t/g) while carrying a peg**, otherwise it is no longer vertical.

## 8. Orientation: important limit of a 4-DOF arm

The Manipulator X has **no wrist roll**. All three pitch joints are parallel, so the gripper's yaw
is always the base angle `joint1 = atan2(y, x)`. The jaws always close **tangentially**
(perpendicular to the line from the robot base). Consequences:

* A parallel-jaw **squeeze** turns a square or hexagon peg until a flat face lies against each jaw.
* The peg turns with joint 1 while it is carried.
* So **the yaw at which a peg arrives at its socket is fixed by the socket's bearing from the robot.** No
  trajectory, re-grasp or retry can change it, because every top-grasp move keeps `peg yaw − bearing` constant.

A square or hexagon socket only accepts the peg if the board is oriented right for that socket's bearing. With the board
straight across in front of the robot (as in the lab photo), the square and hexagon arrive about 13–17° rotated
(the circle never matters).

What the software does about it:
* `task_manager` prints the **predicted yaw error** for every peg before the run. The live vision window (`show_window`)
  shows it next to each socket too.
* `ros2 run peg_sorting suggest_board_pose` searches for **a reachable board pose where every socket lines up**
  (for the sim board: move it 45 mm and turn it 10° → all errors < 0.5°). If you are allowed to place the board, put it
  there and fine-tune while watching the numbers in the vision window.
* The **triangle** can settle in one of two orientations, 60° apart, depending on how the peg is standing before the
  grasp. When the other one would line up, the scene report says **"turn this peg by N° before starting"**. Do that while
  setting up the pegs.
* Insertion is slow and guarded, with small offset retries, and verification detects a peg left resting on the
  board. A placement that failed because of a known yaw error is not retried.
* `task.yaml → alignment.grasp_model`: `squeeze` (default; hard fingers turn the peg) or `hold` (soft fingers grip the
  peg without turning it). With `hold`, the scene report tells you by how much to **rotate each peg** before
  starting. **Test which model your gripper follows**: put the square peg down 20° rotated, close the gripper on it and
  see whether it turns.

## 9. Safety and emergency stop

* **SPACE** in the autonomy terminal, in teleop, or in a separate stop terminal sends `/emergency_stop = true`:
  ```bash
  ros2 run peg_sorting estop
  ```
  The driver aborts the motion immediately and **holds the current position** (torque stays on, so the arm doesn't fall).
  New motion goals are rejected until you release it: `r` in the `estop` tool, `x` in teleop, or:
  ```bash
  ros2 topic pub --once /emergency_stop std_msgs/msg/Bool "data: false"
  ```
* **Hard stop**: switch off the 12 V supply (the arm will sag; keep a hand near it).
* Joint limits are checked in IK, in the planner and again in the driver. Trajectories outside them are rejected.
* Speeds: `robot.yaml → limits` (1.5 rad/s, 0.2 m/s by default). Insertion runs at 4 cm/s and is **guarded**: if the
  tracking error goes above 0.2 rad (the arm is pushing against something), it stops.
* The driver starts by holding the current pose (no jump at torque-on). Ctrl‑C in the task manager sends an emergency stop.
* Before the first run at full speed, run once with `--speed 0.5`, keeping a hand on the stop key.

## 10. Evaluation: success rate and timing

Every autonomous run and every teleop session writes `~/peg_sorting_logs/run_<id>.csv` (one row per attempt:
peg, success, verification result, failure reason, time to grasp / place / end, predicted yaw error, pitch, route)
and `run_<id>.json` (total time, pegs placed, failures).

```bash
python3 scripts/evaluate_timing.py                 # all runs
```

```bash
python3 scripts/evaluate_timing.py --mode autonomy --last 5
```
It prints the peg success rate, completed and zero-failure runs, mean/best/worst run time, per-peg times and
failure reasons. Raise `task.yaml → execution.speed_scale` (or `--speed`) step by step for the speed bonus.

## 11. How it works

```
 RealSense ──► vision node ───────────────┐  /vision/get_scene (GetScene.srv)
 (RGB + aligned depth)                    │   pegs: colour, shape, top-centre xyz, height, yaw
                                          │   sockets: shape, centre xyz, yaw;  board pose
                                          ▼
                                   task_manager ──plan──► PickPlacePlanner (IK, trajectories, path)
                                          │
              FollowJointTrajectory / GripperCommand actions, /emergency_stop
                                          ▼
                     driver node ──► Dynamixel SDK ──► U2D2 ──► Manipulator X
```

**Vision** (`peg_sorting/vision`)
1. *Colour segmentation*: HSV thresholds + morphology (`colors.yaml`).
2. *Depth*: each pixel is deprojected and transformed into the base frame, giving a height map. Peg pixels must be
   3–20 cm above the table and inside the workspace, which removes the board's burnt edges, the ruler, cables and the marker.
3. *Pose*: the peg's 3-D points within 12 mm of its highest point are its **top face**. Their x-y footprint is rasterised
   top-down (metric, so there is no perspective distortion from the oblique camera). Its centroid is the peg position, and the top is the z.
4. *Shape*: scale-invariant features of that outline: area ÷ enclosing-circle area (circle 1.0, hexagon 0.83,
   square 0.64, triangle 0.41) and the 3rd/4th/6th Fourier harmonics of radius-vs-angle, with nearest-prototype
   classification. The harmonic phase gives the polygon's **yaw**. 99.7% correct on noisy synthetic samples, about 1° yaw error.
   If confidence is low, the colour → shape prior is used (and logged as such).
5. *Sockets*: the image is warped onto the board plane (homography from the calibration) into a 1 mm/px
   **bird's-eye view**. The board is the MDF-coloured region, and its holes are classified with the same shape classifier.
   Tall things (pegs, arm) are masked so they cannot create fake holes. Socket poses are cached for the run
   (holes disappear once filled).
6. *Averaging*: N frames, median positions, majority vote on shapes.

**Planning** (`peg_sorting/planning`)
* *FK/IK*: closed-form for the 4-DOF arm (`kinematics.py`). Verified against the ROBOTIS zero pose, and FK∘IK round-trips
  to 1e‑6 m.
* *Grasp*: top grasp, **pre-grasp offset → vertical descent along the board normal → settle → close**.
* *Constant pitch*: one tool pitch, as close to straight down as reachable, is used for the whole pick-and-place, so the
  peg stays vertical while carried and inserted (far sockets need a tilted gripper; the peg is still vertical).
* *Path*: transport at the lowest height that clears the board, around the other pegs (circular obstacles), choosing
  between direct, inner-radius and outer-radius routes in polar coordinates (`path_planner.py`).
* *Trajectories*: minimum-jerk time scaling, sampled at 50 Hz through the IK in cylindrical coordinates
  (straight vertical approaches, arcs around the base), velocity-limited per joint (`trajectory.py`).

**Control** (`peg_sorting/control`, `nodes/driver_node.py`)
* Our own ROS 2 driver on the Dynamixel SDK. No dependency on a particular ROBOTIS package version; standard
  `control_msgs` actions under the same names as the ROBOTIS ros2_control setup.
* Joints in position mode, streamed at 100 Hz. The gripper is in current-based position mode, so it grips with limited force and **stall = object held**.
* Path-tolerance monitoring for guarded moves; preemption; emergency-stop hold.

## 12. Parameter files

| File | What to edit |
|---|---|
| `config/colors.yaml` | HSV ranges per peg colour and for the board; colour → shape prior |
| `config/camera_calib.yaml` | written by the calibration tools; marker size + position for ArUco calibration; `xyz_correction_m` |
| `config/vision.yaml` | camera resolution / exposure, **workspace bounds**, board thickness, detection thresholds, `shape_backend` |
| `config/robot.yaml` | port, IDs, joint signs, **gripper ticks**, **tool_length**, joint limits, **observe/home poses**, speed limits |
| `config/task.yaml` | grasp depth, approach clearance, insertion speeds/heights, transport clearance, retries, `speed_scale`, `grasp_model`, log dir |

## 13. Optional: YOLOv8 shape detection

The supplementary repos use YOLO. Here the geometric classifier is the default (no training, and it gives the yaw). A YOLO
path is included if you want it:

```bash
pip3 install -r requirements-yolo.txt
```

```bash
ros2 run peg_sorting capture_dataset --out ~/peg_dataset     # SPACE saves image + auto labels; fix labels in a labelling tool
```

```bash
ros2 run peg_sorting train_yolo --data ~/peg_dataset/data.yaml --epochs 60
```
Then set `vision.yaml → shape_backend: yolo` and `yolo.model_path`. YOLO gives the shape label for each colour blob;
position and yaw still come from depth.

## 14. Troubleshooting

| Problem | Fix |
|---|---|
| `cannot open /dev/ttyUSB0` | `sudo usermod -aG dialout $USER`, log out/in; check `ls /dev/ttyUSB*` |
| `no answer from Dynamixel IDs` | 12 V on? Run `ros2 run peg_sorting scan_dynamixel`; set IDs/baud in `robot.yaml` |
| Jerky motion / slow control loop | run `scripts/setup_usb_latency.sh` (USB latency 16 ms → 1 ms) |
| `No device connected` (RealSense) | udev rules (setup step 5), use a USB **3** port/cable, close `realsense-viewer` |
| Positions off by a constant | `check_calibration` → `xyz_correction_m` |
| Peg not detected | `hsv_tuner`; check `vision.yaml → workspace`; is the arm in the observe pose blocking the view? |
| Socket missing | a peg standing in front of it (from the camera's view)? Move it; check `colors.yaml → board` |
| Gripper closes but "grasp failed" | adjust `gripper.closed_ticks` / `grasp_detect_fraction`; lower `grasp_depth` |
| Arm stops at insertion (blocked) | yaw misaligned (section 8) or calibration off by > 3 mm |
| `GOAL_TOLERANCE_VIOLATED` warnings | gravity sag of joint 2; harmless below about 0.05 rad, or raise `driver.goal_tolerance_rad` |

## 15. Tests

```bash
cd ~/peg_ws/src/Project_1_ITR_Prady_Yash_Kartik/peg_sorting && python3 -m pytest -q test
```
These cover the kinematics (FK/IK round trip, straight vertical approach, velocity limits), the shape classifier on noisy
samples, end-to-end vision on the rendered scene (positions within 4 mm, yaw within 4°, all sockets found), ArUco
calibration accuracy (< 5 mm, < 1°), the planner (constant pitch, insertion depth, gripper sequence),
obstacle avoidance and the orientation model.
