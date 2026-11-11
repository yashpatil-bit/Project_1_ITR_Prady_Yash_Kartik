#!/usr/bin/env python3
"""Read the status of every Manipulator X motor (stop the driver first - it owns the port).

    python3 scripts/dxl_status.py              show status
    python3 scripts/dxl_status.py --reboot     also reboot motors that report a hardware error

Hardware error bits: 0x01 input voltage, 0x04 overheating, 0x08 motor encoder,
0x10 electrical shock, 0x20 OVERLOAD (most common: joint forced / arm resting on a stop).
A motor with a hardware error turns its torque off and ignores commands until rebooted
(or until the 12 V power is switched off and on).
"""
import argparse
import math
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

ERR = {0x01: 'input voltage', 0x04: 'overheating', 0x08: 'encoder', 0x10: 'electrical shock', 0x20: 'OVERLOAD'}
MODES = {0: 'current', 1: 'velocity', 3: 'position', 4: 'extended position', 5: 'current-based position', 16: 'PWM'}


def s32(v):
    return v - (1 << 32) if v >= (1 << 31) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='/dev/ttyUSB0')
    ap.add_argument('--baud', type=int, default=1000000)
    ap.add_argument('--ids', default='11,12,13,14,15')
    ap.add_argument('--reboot', action='store_true')
    a = ap.parse_args()
    ph, pk = PortHandler(a.port), PacketHandler(2.0)
    if not ph.openPort() or not ph.setBaudRate(a.baud):
        print('cannot open %s (driver still running? permissions?)' % a.port)
        return 1
    print('%-4s %-6s %-23s %-26s %8s %8s %6s %6s %13s' % ('ID', 'torque', 'mode', 'hardware error', 'present', 'goal',
                                                          'temp', 'volt', 'limits'))
    bad = []
    for i in [int(x) for x in a.ids.split(',')]:
        def r(addr, n):
            fn = {1: pk.read1ByteTxRx, 2: pk.read2ByteTxRx, 4: pk.read4ByteTxRx}[n]
            v, res, _ = fn(ph, i, addr)
            return v if res == COMM_SUCCESS else None
        if r(64, 1) is None:
            print('%-4d no answer' % i)
            continue
        err = r(70, 1) or 0
        pos, goal = s32(r(132, 4)), s32(r(116, 4))
        errs = ', '.join(t for b, t in ERR.items() if err & b) or 'none'
        if err:
            bad.append(i)
        print('%-4d %-6s %-23s %-26s %8d %8d %5dC %5.1fV %6d-%-6d  (%.2f rad)' % (
            i, 'ON' if r(64, 1) else 'off', MODES.get(r(11, 1), r(11, 1)), errs, pos, goal, r(146, 1),
            (r(144, 2) or 0) / 10.0, r(52, 4), r(48, 4), (pos - 2048) * 2 * math.pi / 4096))
    if bad and a.reboot:
        print('\nWARNING: rebooting releases the torque - SUPPORT THE ARM WITH YOUR HAND now.')
        time.sleep(3)
        for i in bad:
            res, _ = pk.reboot(ph, i)
            print('reboot ID %d: %s' % (i, 'ok' if res == COMM_SUCCESS else pk.getTxRxResult(res)))
        time.sleep(1)
    elif bad:
        print('\nMotors %s report a hardware error -> run again with --reboot (support the arm!), '
              'or switch the 12 V off and on.' % bad)
    ph.closePort()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
