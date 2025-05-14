from chart_data import ChartData

from config import Config

import signal

# Source of truth about configuration. All new options should be put there if possible
CFG = Config()

CHART_DATA = ChartData(CFG.v_interval)

QUITTING = False

def sig_handler(_i, _whatever):
    global QUITTING, CFG
    if QUITTING:
        print("Quitting now!")
        sys.exit(1)
    print(f"Cleaning up... (Send SIGINT or SIGTERM again to quit now)")
    QUITTING = True

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)
