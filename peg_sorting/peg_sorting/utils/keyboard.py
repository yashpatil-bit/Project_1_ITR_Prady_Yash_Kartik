"""Non-blocking single-key terminal input (Linux)."""
import os
import select
import sys
import termios
import tty


class KeyReader:
    """with KeyReader() as kr:  key = kr.get(timeout=0.1)  -> str or None.
    Arrow keys are returned as 'UP', 'DOWN', 'LEFT', 'RIGHT'."""

    def __init__(self):
        self.fd = None
        self.old = None

    def __enter__(self):
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.fd is not None and self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    @property
    def available(self):
        return self.fd is not None

    def get(self, timeout=0.1):
        if self.fd is None:
            return None
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return None
        ch = os.read(self.fd, 1).decode(errors='ignore')
        if ch == '\x1b':
            r, _, _ = select.select([self.fd], [], [], 0.01)
            if r:
                seq = os.read(self.fd, 2).decode(errors='ignore')
                return {'[A': 'UP', '[B': 'DOWN', '[C': 'RIGHT', '[D': 'LEFT'}.get(seq, 'ESC')
            return 'ESC'
        return ch
