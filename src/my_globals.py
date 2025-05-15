from chart_data import ChartData

from config import Config

import signal
import sys

# Source of truth about configuration. All new options should be put there if possible
CFG = Config()

CHART_DATA = ChartData(CFG.v_interval, CFG.master_process)

def sig_handler(_i, _whatever):
    global CFG
    if CFG.quitting:
        print("Quitting now!")
        sys.exit(1)
    print(f"Cleaning up... (Send SIGINT or SIGTERM again to quit now)")
    CFG.quitting = True

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)
