import win32gui
import traci
import time
import subprocess

# Start SUMO via subprocess to see its window name
proc = subprocess.Popen([
    "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe",
    "-c", "C:/Users/ravur_48/OneDrive/Desktop/traffic_project/traffic.sumocfg"
])
time.sleep(2)

def enum_cb(hwnd, results):
    if win32gui.IsWindowVisible(hwnd):
        name = win32gui.GetWindowText(hwnd)
        if "SUMO" in name.upper() or "ECLIPSE" in name.upper():
            results.append((hwnd, name))

results = []
win32gui.EnumWindows(enum_cb, results)
print(results)
proc.terminate()
