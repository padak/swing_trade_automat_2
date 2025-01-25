import signal
import time

def signal_handler(sig, frame):
    print('\nCTRL+C detected in test script. Exiting.')
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
print("Signal handler registered. Press CTRL+C to test.")

while True:
    time.sleep(1) 