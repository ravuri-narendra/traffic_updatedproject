import os
import sys
import time
import cv2
import numpy as np
import mss
import win32gui
from ultralytics import YOLO

# -------------------- SUMO SETUP --------------------
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    raise Exception("Please set SUMO_HOME environment variable")

import traci

TL_ID = "center"
MIN_GREEN = 5
MAX_GREEN = 60
YELLOW = 3
EMERGENCY_HOLD = 10
DENSITY_FACTOR = 2.0

# -------------------- VISION AGENT --------------------
class VisionAgent:
    def __init__(self):
        print("[INFO] Loading YOLO model...", flush=True)
        self.model = YOLO('yolov8n.pt')
        print("[INFO] YOLO model loaded.", flush=True)
        self.confidence = 0.25

        self.emergency_lower = np.array([90, 80, 50]) # Broader Blue range
        self.emergency_upper = np.array([140, 255, 255])
        self.emergency_threshold = 20

        self.sct = mss.mss()
        self.window_rect = None

        self.rois = {"N": None, "S": None, "E": None, "W": None}

    def initialize_window(self):
        time.sleep(2)  # Give time for the GUI to open
        hwnds = []
        def enum_cb(hwnd, arg):
            if win32gui.IsWindowVisible(hwnd):
                name = win32gui.GetWindowText(hwnd)
                if name:
                    if "SUMO 1." in name and "Visual Studio Code" not in name:
                        hwnds.append(hwnd)
                        print(f"[DEBUG] Found potential SUMO window: {name}", flush=True)

        win32gui.EnumWindows(enum_cb, None)
        if not hwnds:
            # Fallback for headless or if window not found
            print("[WARNING] SUMO-GUI window not found. Capture might fail.", flush=True)
            self.window_rect = {"left": 0, "top": 0, "width": 800, "height": 600}
            return
        
        hwnd = hwnds[0]
        rect = win32gui.GetWindowRect(hwnd)
        self.window_rect = {
            "left": rect[0],
            "top": rect[1] + 50,
            "width": rect[2] - rect[0],
            "height": rect[3] - rect[1] - 50
        }

    def capture_screen(self):
        try:
            img = np.array(self.sct.grab(self.window_rect))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"[ERROR] Capture failed: {e}", flush=True)
            return np.zeros((600, 800, 3), dtype=np.uint8)

    def setup_rois(self, frame):
        h, w = frame.shape[:2]
        self.rois["N"] = np.array([[int(0.45*w), int(0.1*h)], [int(0.55*w), int(0.1*h)], [int(0.55*w), int(0.4*h)], [int(0.45*w), int(0.4*h)]], dtype=np.int32)
        self.rois["S"] = np.array([[int(0.45*w), int(0.6*h)], [int(0.55*w), int(0.6*h)], [int(0.55*w), int(0.9*h)], [int(0.45*w), int(0.9*h)]], dtype=np.int32)
        self.rois["E"] = np.array([[int(0.6*w), int(0.45*h)], [int(0.9*w), int(0.45*h)], [int(0.9*w), int(0.55*h)], [int(0.6*w), int(0.55*h)]], dtype=np.int32)
        self.rois["W"] = np.array([[int(0.1*w), int(0.45*h)], [int(0.4*w), int(0.45*h)], [int(0.4*w), int(0.55*h)], [int(0.1*w), int(0.55*h)]], dtype=np.int32)

    def detect_vehicles(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        detections = []
        annotated = frame.copy()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]
            if label not in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']: continue
            centroid = ((x1+x2)//2, (y1+y2)//2)
            detections.append({'bbox': (x1, y1, x2, y2), 'centroid': centroid, 'class': label})
            cv2.rectangle(annotated, (x1,y1),(x2,y2),(0,255,0),2)
        return detections, annotated

    def map_to_lanes(self, detections, frame):
        densities = {"N": 0, "S": 0, "E": 0, "W": 0}
        emergency = []
        for det in detections:
            cx, cy = det['centroid']
            for lane, pts in self.rois.items():
                if cv2.pointPolygonTest(pts, (cx,cy), False) >= 0:
                    densities[lane] += 2.5 if det['class'] in ['truck','bus'] else 1
                    break

        for lane, pts in self.rois.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8); cv2.fillPoly(mask, [pts], 255)
            roi_pixels = cv2.bitwise_and(frame, frame, mask=mask)
            hsv = cv2.cvtColor(roi_pixels, cv2.COLOR_BGR2HSV)
            l_green = np.array([35, 40, 40]); u_green = np.array([85, 255, 255])
            l_road = np.array([0, 0, 0]); u_road = np.array([180, 60, 200])
            non_road = cv2.bitwise_not(cv2.bitwise_or(cv2.inRange(hsv, l_green, u_green), cv2.inRange(hsv, l_road, u_road)))
            h_density = cv2.countNonZero(cv2.bitwise_and(non_road, mask)) / 100.0
            densities[lane] = max(densities[lane], h_density)

            e_mask = cv2.inRange(hsv, self.emergency_lower, self.emergency_upper)
            if cv2.countNonZero(e_mask) >= self.emergency_threshold: emergency.append(lane)
        return densities, emergency

    def draw_rois(self, frame):
        colors = {"N": (255,0,0), "S": (0,255,0), "E": (0,0,255), "W": (255,255,0)}
        for lane, pts in self.rois.items(): cv2.polylines(frame, [pts], True, colors[lane], 2)
        return frame

# -------------------- DECISION AGENT --------------------
class DecisionAgent:
    def __init__(self):
        self.phase_timer = 0
        self.current_phase = None
        self.lanes = ["N", "E", "S", "W"]
        self.current_index = 0
        self.wait_time = {lane: 0 for lane in self.lanes}
        self.max_wait = 50

    def compute_time(self, density):
        return min(MAX_GREEN, MIN_GREEN + density * DENSITY_FACTOR)

    def decide(self, densities, emergency):
        for lane in self.lanes: self.wait_time[lane] += 1
        if emergency:
            chosen = max(set(emergency), key=emergency.count)
            if chosen == self.current_phase and self.phase_timer > 0:
                return None, 0, "Emergency Already Served"
            self.wait_time[chosen] = 0
            return chosen, EMERGENCY_HOLD, "Emergency"

        for lane in self.lanes:
            if self.wait_time[lane] > self.max_wait and densities[lane] > 0:
                self.wait_time[lane] = 0
                return lane, self.compute_time(densities[lane]), "Starvation Fix"

        for _ in range(4):
            self.current_index = (self.current_index + 1) % 4
            candidate = self.lanes[self.current_index]
            if densities[candidate] > 0:
                self.wait_time[candidate] = 0
                return candidate, self.compute_time(densities[candidate]), "Round-Robin"
        return None, 0, "No Traffic"

# -------------------- MAIN --------------------
def run():
    print("[INFO] Starting SUMO-GUI...", flush=True)
    try:
        traci.start([
            "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe",
            "-c", "C:/Users/ravur_48/OneDrive/Desktop/traffic_project/traffic.sumocfg",
            "--start"
        ])
    except Exception as e:
        print(f"[ERROR] Could not start SUMO: {e}", flush=True)
        return

    vision = VisionAgent()
    decision = DecisionAgent()
    traci.trafficlight.setPhase(TL_ID, 0)
    vision.initialize_window()
    
    frame = vision.capture_screen()
    vision.setup_rois(frame)
    
    last_detections = []
    step = 0

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            frame = vision.capture_screen()
            if step % 5 == 0:
                detections, annotated = vision.detect_vehicles(frame)
                last_detections = detections
            else:
                detections = last_detections
                annotated = frame.copy()

            densities, emergency = vision.map_to_lanes(detections, frame)
            
            should_decide = (decision.phase_timer <= 0) or (len(emergency) > 0)
            reason = "Running Cycle"
            
            if should_decide:
                phase, green, d_reason = decision.decide(densities, emergency)
                if phase:
                    phase_map = {"N": 0, "E": 2, "S": 4, "W": 6}
                    traci.trafficlight.setPhase(TL_ID, phase_map[phase])
                    decision.phase_timer = int(green)
                    decision.current_phase = phase
                    reason = d_reason

            # Update Dashboard State
            current_phase_idx = traci.trafficlight.getPhase(TL_ID)
            # 0=N, 1=N_Y, 2=E, 3=E_Y, 4=S, 5=S_Y, 6=W, 7=W_Y
            ui_phases = {l: "red" for l in ["N", "S", "E", "W"]}
            if current_phase_idx == 0: ui_phases["N"] = "green"
            elif current_phase_idx == 1: ui_phases["N"] = "yellow"
            elif current_phase_idx == 2: ui_phases["E"] = "green"
            elif current_phase_idx == 3: ui_phases["E"] = "yellow"
            elif current_phase_idx == 4: ui_phases["S"] = "green"
            elif current_phase_idx == 5: ui_phases["S"] = "yellow"
            elif current_phase_idx == 6: ui_phases["W"] = "green"
            elif current_phase_idx == 7: ui_phases["W"] = "yellow"

            # Log current state to console
            print(f"Step {step} | Phases: {ui_phases} | Densities: {densities} | Reason: {reason}", flush=True)

            annotated = vision.draw_rois(annotated)
            cv2.imshow("Vision Analytics", annotated)
            if cv2.waitKey(1) == ord('q'): break

            traci.simulationStep()
            time.sleep(0.01)

            if decision.phase_timer > 0:
                decision.phase_timer -= 0.5
            else:
                if decision.current_phase is not None:
                    phase_map = {"N": 0, "E": 2, "S": 4, "W": 6}
                    yellow_phase = phase_map[decision.current_phase] + 1
                    traci.trafficlight.setPhase(TL_ID, yellow_phase)
                    for _ in range(YELLOW):
                        traci.simulationStep()
                        time.sleep(0.05)
                    decision.current_phase = None
            step += 1
            
    except traci.exceptions.FatalTraCIError:
        print("\n[INFO] SUMO closed by user.", flush=True)
    except Exception as e:
        print(f"\n[ERROR] Simulation Loop Error: {e}", flush=True)
    finally:
        cv2.destroyAllWindows()
        try: traci.close()
        except: pass
        print("\n[INFO] Simulation ended.")

if __name__ == "__main__":
    run()