"""Scan the U2D2 for Dynamixel motors (no ROS needed).

    ros2 run peg_sorting scan_dynamixel [--port /dev/ttyUSB0] [--max-id 30]
"""
import argparse
import glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='')
    ap.add_argument('--max-id', type=int, default=30)
    ap.add_argument('--bauds', default='1000000,57600,115200,2000000,3000000,4000000')
    a = ap.parse_args()
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
    ports = [a.port] if a.port else sorted(glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*'))
    if not ports:
        print('No /dev/ttyUSB* found. Is the U2D2 plugged in?  (ls -l /dev/ttyUSB*)')
        return
    pk = PacketHandler(2.0)
    for port in ports:
        ph = PortHandler(port)
        if not ph.openPort():
            print('%s: cannot open (add yourself to the dialout group: sudo usermod -aG dialout $USER, then re-login)' % port)
            continue
        for baud in [int(b) for b in a.bauds.split(',')]:
            ph.setBaudRate(baud)
            found = []
            for i in range(0, a.max_id + 1):
                model, res, _ = pk.ping(ph, i)
                if res == COMM_SUCCESS:
                    found.append((i, model))
            if found:
                print('%s @ %d baud: %s' % (port, baud, ', '.join('ID %d (model %d)' % f for f in found)))
        ph.closePort()
    print('Manipulator X default: IDs 11,12,13,14 (joints) + 15 (gripper) @ 1000000 baud, XM430-W350 = model 1020')


if __name__ == '__main__':
    main()
