"""Low-level Dynamixel (XM430-W350-T, protocol 2.0) access through the U2D2.

Only this file talks to the serial port.  ``MockBus`` has the same interface
and simulates the motors, so everything above it can run without hardware.
"""
import math
import threading
import time

TICKS_PER_REV = 4096
CENTER = 2048

# XM430 control table (protocol 2.0)
ADDR_OPERATING_MODE = 11
ADDR_DRIVE_MODE = 10
ADDR_TORQUE_ENABLE = 64
ADDR_POSITION_P_GAIN = 84
ADDR_GOAL_CURRENT = 102
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_MOVING = 122
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_HARDWARE_ERROR = 70

MODE_POSITION = 3
MODE_CURRENT_BASED_POSITION = 5


def rad_to_tick(rad, sign=1, offset=0.0):
    return int(round(CENTER + sign * (rad + offset) * TICKS_PER_REV / (2 * math.pi)))


def tick_to_rad(tick, sign=1, offset=0.0):
    return sign * (tick - CENTER) * 2 * math.pi / TICKS_PER_REV - offset


def _signed32(v):
    return v - (1 << 32) if v >= (1 << 31) else v


def _signed16(v):
    return v - (1 << 16) if v >= (1 << 15) else v


class DynamixelError(RuntimeError):
    pass


class DynamixelBus:
    def __init__(self, port, baudrate, joint_ids, gripper_id, protocol=2.0):
        from dynamixel_sdk import (COMM_SUCCESS, GroupSyncRead, GroupSyncWrite,
                                   PacketHandler, PortHandler)
        self.COMM_SUCCESS = COMM_SUCCESS
        self.joint_ids = list(joint_ids)
        self.gripper_id = gripper_id
        self.all_ids = self.joint_ids + ([gripper_id] if gripper_id is not None else [])
        self.lock = threading.RLock()
        self.port = PortHandler(port)
        self.pk = PacketHandler(protocol)
        if not self.port.openPort():
            raise DynamixelError('cannot open %s (is the U2D2 plugged in? are you in the dialout group?)' % port)
        if not self.port.setBaudRate(baudrate):
            raise DynamixelError('cannot set baudrate %d' % baudrate)
        missing = [i for i in self.all_ids if not self.ping(i)]
        if missing:
            raise DynamixelError('no answer from Dynamixel IDs %s on %s @ %d baud. Is the 12 V supply on? '
                                 'Run `ros2 run peg_sorting scan_dynamixel`.' % (missing, port, baudrate))
        self._sw_goal = GroupSyncWrite(self.port, self.pk, ADDR_GOAL_POSITION, 4)
        self._sr_pos = GroupSyncRead(self.port, self.pk, ADDR_PRESENT_POSITION, 4)
        for i in self.all_ids:
            self._sr_pos.addParam(i)
        self._sr_cur = GroupSyncRead(self.port, self.pk, ADDR_PRESENT_CURRENT, 2)
        if gripper_id is not None:
            self._sr_cur.addParam(gripper_id)

    # -------------------------------------------------------------- basics
    def ping(self, dxl_id):
        with self.lock:
            _, res, _ = self.pk.ping(self.port, dxl_id)
        return res == self.COMM_SUCCESS

    def _w1(self, i, addr, v):
        with self.lock:
            res, err = self.pk.write1ByteTxRx(self.port, i, addr, int(v))
        if res != self.COMM_SUCCESS:
            raise DynamixelError('write id %d addr %d failed: %s' % (i, addr, self.pk.getTxRxResult(res)))

    def _w2(self, i, addr, v):
        with self.lock:
            res, err = self.pk.write2ByteTxRx(self.port, i, addr, int(v) & 0xFFFF)
        if res != self.COMM_SUCCESS:
            raise DynamixelError('write id %d addr %d failed' % (i, addr))

    def _w4(self, i, addr, v):
        with self.lock:
            res, err = self.pk.write4ByteTxRx(self.port, i, addr, int(v) & 0xFFFFFFFF)
        if res != self.COMM_SUCCESS:
            raise DynamixelError('write id %d addr %d failed' % (i, addr))

    def _r1(self, i, addr):
        with self.lock:
            v, res, err = self.pk.read1ByteTxRx(self.port, i, addr)
        return v if res == self.COMM_SUCCESS else None

    # -------------------------------------------------------------- setup
    def configure(self, gripper_current_limit=300, gripper_profile_velocity=0,
                  joint_profile_velocity=0, joint_profile_acceleration=0):
        """Torque off, set operating modes, hold the current position, torque on."""
        for i in self.all_ids:
            self._w1(i, ADDR_TORQUE_ENABLE, 0)
        for i in self.joint_ids:
            self._w1(i, ADDR_OPERATING_MODE, MODE_POSITION)
            self._w1(i, ADDR_DRIVE_MODE, 0)
            self._w4(i, ADDR_PROFILE_ACCELERATION, joint_profile_acceleration)
            self._w4(i, ADDR_PROFILE_VELOCITY, joint_profile_velocity)
        if self.gripper_id is not None:
            g = self.gripper_id
            self._w1(g, ADDR_OPERATING_MODE, MODE_CURRENT_BASED_POSITION)
            self._w1(g, ADDR_DRIVE_MODE, 0)
            self._w2(g, ADDR_GOAL_CURRENT, gripper_current_limit)
            self._w4(g, ADDR_PROFILE_VELOCITY, gripper_profile_velocity)
        ticks = self.read_positions()
        self.write_goal_ticks(dict(zip(self.all_ids, ticks)))

    def set_torque(self, on, ids=None):
        for i in (ids or self.all_ids):
            self._w1(i, ADDR_TORQUE_ENABLE, 1 if on else 0)

    def hardware_errors(self):
        out = {}
        for i in self.all_ids:
            v = self._r1(i, ADDR_HARDWARE_ERROR)
            if v:
                out[i] = v
        return out

    # -------------------------------------------------------------- I/O
    def read_positions(self):
        """Present position ticks for all_ids (joints..., gripper)."""
        with self.lock:
            res = self._sr_pos.txRxPacket()
            if res != self.COMM_SUCCESS:
                raise DynamixelError('sync read failed: %s' % self.pk.getTxRxResult(res))
            return [_signed32(self._sr_pos.getData(i, ADDR_PRESENT_POSITION, 4)) for i in self.all_ids]

    def read_gripper_current(self):
        if self.gripper_id is None:
            return 0
        with self.lock:
            res = self._sr_cur.txRxPacket()
            if res != self.COMM_SUCCESS:
                return 0
            return _signed16(self._sr_cur.getData(self.gripper_id, ADDR_PRESENT_CURRENT, 2))

    def write_goal_ticks(self, id_to_tick):
        with self.lock:
            self._sw_goal.clearParam()
            for i, t in id_to_tick.items():
                t = int(t) & 0xFFFFFFFF
                self._sw_goal.addParam(i, [t & 0xFF, (t >> 8) & 0xFF, (t >> 16) & 0xFF, (t >> 24) & 0xFF])
            res = self._sw_goal.txPacket()
        if res != self.COMM_SUCCESS:
            raise DynamixelError('sync write failed')

    def close(self):
        try:
            self.port.closePort()
        except Exception:  # noqa: BLE001
            pass


class MockBus:
    """Simulated motors: each joint follows its goal with a first-order lag;
    the gripper stops when it reaches an object (if a 'stall_at' tick is set)."""

    def __init__(self, port, baudrate, joint_ids, gripper_id, protocol=2.0, initial_ticks=None):
        self.joint_ids = list(joint_ids)
        self.gripper_id = gripper_id
        self.all_ids = self.joint_ids + ([gripper_id] if gripper_id is not None else [])
        self.lock = threading.RLock()
        self.pos = dict(zip(self.all_ids, initial_ticks or [CENTER] * len(self.all_ids)))
        self.goal = dict(self.pos)
        self.torque = True
        self.tau = 0.03
        self.gripper_stall_tick = None   # set by the simulator when a peg is between the fingers
        self.gripper_closed_tick = 0     # set by the driver (direction of closing)
        self._t = time.time()

    def configure(self, **kwargs):
        pass

    def set_torque(self, on, ids=None):
        self.torque = on

    def ping(self, i):
        return True

    def hardware_errors(self):
        return {}

    def _step(self):
        now = time.time()
        dt = now - self._t
        self._t = now
        if not self.torque:
            return
        a = 1.0 - math.exp(-dt / self.tau) if dt > 0 else 0.0
        gripper_rate = 2500.0 * dt  # ticks per call (max speed)
        for i in self.all_ids:
            if i == self.gripper_id:
                g = self.goal[i]
                st = self.gripper_stall_tick
                if st is not None:
                    g = max(g, st) if self.gripper_closed_tick < st else min(g, st)
                d = g - self.pos[i]
                self.pos[i] += max(-gripper_rate, min(gripper_rate, d))
            else:
                self.pos[i] += a * (self.goal[i] - self.pos[i])

    def read_positions(self):
        with self.lock:
            self._step()
            return [int(round(self.pos[i])) for i in self.all_ids]

    def read_gripper_current(self):
        if self.gripper_stall_tick is not None and abs(self.pos[self.gripper_id] - self.gripper_stall_tick) < 3:
            return 250
        return 20

    def write_goal_ticks(self, id_to_tick):
        with self.lock:
            self._step()
            for i, t in id_to_tick.items():
                self.goal[i] = float(t)

    def close(self):
        pass
