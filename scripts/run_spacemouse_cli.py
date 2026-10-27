#!/usr/bin/env python3
"""SpaceMouse-only CLI; shares session UI/setup, uses spacemouse_runner exclusively."""
from __future__ import annotations

import sys

from run_teleop_cli import main

if __name__ == '__main__':
    if any(arg == '--controller' or arg.startswith('--controller=') for arg in sys.argv[1:]):
        raise SystemExit('This entry point always uses SpaceMouse; omit --controller.')
    sys.argv[1:1] = ['--controller', 'spacemouse']
    main()
