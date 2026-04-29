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
try:
    import serial
except ImportError:
    print("[WARNING] 'pyserial' not installed. Run: pip install pyserial")
    serial = None

# Arduino Config
ARDUINO_PORT = "COM7"  # Change to your Arduino's COM port
BAUD_RATE = 115200
arduino = None

try:
    if serial:
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
        print(f"[INFO] Arduino connected on {ARDUINO_PORT}")
except Exception as e:
    print(f"[WARNING] Arduino not detected on {ARDUINO_PORT}: {e}")

TL_ID = "center"
MIN_GREEN = 5
MAX_GREEN = 60
YELLOW = 3
ALL_RED = 2
EMERGENCY_HOLD = 60        # Max green time (seconds) for EV; released earlier if cleared
EV_COOLDOWN    = 60       # Seconds before a served EV lane can be re-queued
DENSITY_FACTOR = 2.0
DENSITY_THRESHOLD = 0.5
SIM_SPEED_DELAY = 1.0  # <-- CHANGE THIS TO 0.1 to make the simulation run 10x faster!

# 1. LANE & PHASE MAPPING
LANE_MAP = {
    "N": "N2C",
    "S": "S2C",
    "E": "E2C",
    "W": "W2C"
}

PHASE_MAP = {
    "N": 0,
    "E": 3,
    "S": 6,
    "W": 9
}


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
        # Extended ROIs closer to the center (intersection line) to avoid a blind spot
        self.rois["N"] = np.array([[int(0.45*w), int(0.1*h)], [int(0.55*w), int(0.1*h)], [int(0.55*w), int(0.48*h)], [int(0.45*w), int(0.48*h)]], dtype=np.int32)
        self.rois["S"] = np.array([[int(0.45*w), int(0.52*h)], [int(0.55*w), int(0.52*h)], [int(0.55*w), int(0.9*h)], [int(0.45*w), int(0.9*h)]], dtype=np.int32)
        self.rois["E"] = np.array([[int(0.52*w), int(0.45*h)], [int(0.9*w), int(0.45*h)], [int(0.9*w), int(0.55*h)], [int(0.52*w), int(0.55*h)]], dtype=np.int32)
        self.rois["W"] = np.array([[int(0.1*w), int(0.45*h)], [int(0.48*w), int(0.45*h)], [int(0.48*w), int(0.55*h)], [int(0.1*w), int(0.55*h)]], dtype=np.int32)

    def detect_vehicles(self, frame):
        start_time = time.time()
        # BIG PERFORMANCE FIX: Tell YOLO to resize the image to 320x320 internally before processing. 
        # This takes 4x less CPU power and is much faster for a basic traffic simulation!
        results = self.model(frame, conf=self.confidence, imgsz=320, verbose=False)[0]
        inf_time = (time.time() - start_time) * 1000
        if inf_time > 300:
            print(f"[WARNING] 🐌 CPU is slow! YOLO took {inf_time:.0f}ms", flush=True)

        detections = []
        annotated = frame.copy()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]
            if label not in ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']: continue
            centroid = ((x1+x2)//2, (y1+y2)//2)
            detections.append({'bbox': (x1, y1, x2, y2), 'centroid': centroid, 'class': label})
            
            # --- Visual Distinction for Real-World Variety ---
            box_color = (0, 255, 0) # Green for standard vehicles
            if label == 'person': box_color = (0, 165, 255) # Orange for pedestrians
            elif label in ['bicycle', 'motorcycle']: box_color = (255, 255, 0) # Cyan for 2-wheelers
            elif label in ['truck', 'bus']: box_color = (0, 0, 255) # Red for heavy vehicles
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
        return detections, annotated

    def map_to_lanes(self, detections, frame):
        densities = {"N": 0, "S": 0, "E": 0, "W": 0}
        emergency = []
        for det in detections:
            cx, cy = det['centroid']
            for lane, pts in self.rois.items():
                if cv2.pointPolygonTest(pts, (cx,cy), False) >= 0:
                    if det['class'] in ['truck', 'bus']:
                        densities[lane] += 2.5
                    elif det['class'] in ['motorcycle', 'bicycle']:
                        densities[lane] += 0.5
                    elif det['class'] == 'person':
                        densities[lane] += 0.3  # People have a minor density footprint but still add to wait time
                    else:
                        densities[lane] += 1.0  # Standard Cars
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
            if cv2.countNonZero(e_mask) >= self.emergency_threshold: 
                emergency.append(lane)
        return densities, emergency

    def draw_rois(self, frame):
        colors = {"N": (255,0,0), "S": (0,255,0), "E": (0,0,255), "W": (255,255,0)}
        for lane, pts in self.rois.items(): cv2.polylines(frame, [pts], True, colors[lane], 2)
        return frame

# -------------------- DECISION AGENT --------------------
class DecisionAgent:
    def __init__(self):
        self.phase_timer    = 0
        self.current_phase  = None
        self.lanes          = ["N", "E", "S", "W"]
        self.current_index  = 0
        self.total_emergencies = 0
        self.decision_reason   = "System Booting"

        # --- Multi-EV state ---
        # ev_queue  : ordered list of lanes waiting for EV green (FIFO by first-detection time)
        # ev_first_seen : {lane: timestamp} — when each EV was first detected this cycle
        # ev_cooldown   : {lane: timestamp} — when each lane was last served as EV
        #                  A lane cannot re-enter ev_queue until EV_COOLDOWN seconds pass.
        self.ev_queue      = []
        self.ev_first_seen = {}   # lane → time.time() when first detected
        self.ev_cooldown   = {}   # lane → time.time() when last served
        self.resume_index  = None # index to restore normal sequence after all EVs are done
        self.ev_pending    = False # True while EVs are queued but current phase still running
        self.is_serving_ev = False # True while actively holding green for an EV

    def compute_time(self, density):
        if density <= DENSITY_THRESHOLD:
            return 0
        return min(MAX_GREEN, MIN_GREEN + (density * DENSITY_FACTOR))

    def notify_ev_detected(self, ev_lanes):
        """Called every vision cycle with currently detected EV lanes.

        Rules:
          1. A lane already in ev_queue is NOT re-added (dedup).
          2. A lane that was recently served (within EV_COOLDOWN seconds)
             is NOT re-added — prevents endless re-queueing of a slow ambulance.
          3. When the first NEW lane is queued, resume_index is recorded so
             the normal round-robin can continue correctly after all EVs are served.
          4. ev_queue stays sorted by first-detection time so the ambulance
             that has been waiting longest is ALWAYS served first.
        """
        now = time.time()
        new_lanes_added = []

        for lane in ev_lanes:
            # Skip if already queued
            if lane in self.ev_queue:
                continue
            # Skip if recently served (cooldown active)
            if lane in self.ev_cooldown and (now - self.ev_cooldown[lane]) < EV_COOLDOWN:
                remaining_cd = EV_COOLDOWN - (now - self.ev_cooldown[lane])
                print(f"[EV] Lane {lane} cooldown active — {remaining_cd:.0f}s before re-queue.",
                      flush=True)
                continue
            # Record first-seen time (only if not already tracked)
            if lane not in self.ev_first_seen:
                self.ev_first_seen[lane] = now

            self.ev_queue.append(lane)
            new_lanes_added.append(lane)

        if not new_lanes_added:
            return

        # Sort ev_queue by first-detection time → earliest detected goes first
        self.ev_queue.sort(key=lambda l: self.ev_first_seen.get(l, now))

        # Save the resume point once (when the very first EV is queued this cycle)
        if self.resume_index is None and self.current_phase is not None:
            current_pos = self.lanes.index(self.current_phase)
            self.resume_index = (current_pos + 1) % 4

        self.ev_pending = True

        for lane in new_lanes_added:
            position = self.ev_queue.index(lane) + 1  # 1-indexed position in queue
            print(
                f"[EV] 🚨 New EV in lane {lane} (queue position #{position}/{len(self.ev_queue)}). "
                f"Current lane ({self.current_phase}) finishes first. "
                f"Full EV queue: {self.ev_queue}. "
                f"Normal sequence resumes from: {self.lanes[self.resume_index] if self.resume_index is not None else '?'}",
                flush=True
            )

    def decide(self, densities, current_lane):
        """
        Priority order (each only triggers when phase_timer == 0):
          1. EV queue  → serve EVs in first-detected order, 20s each
          2. Resume    → jump back to saved normal-sequence index after all EVs done
          3. Stay-Green → keep current lane if no one else is waiting
          4. Round-Robin → advance to next lane with traffic
        """
        # 1. Emergency Priority — fires only when the current green has expired
        if self.ev_queue:
            self.is_serving_ev = True
            chosen_lane = self.ev_queue.pop(0)          # First-detected EV goes first
            self.ev_cooldown[chosen_lane] = time.time() # Start cooldown for this lane
            if chosen_lane in self.ev_first_seen:       # Clean up tracking dict
                del self.ev_first_seen[chosen_lane]
            # ev_pending stays True until the queue is fully empty
            if not self.ev_queue:
                self.ev_pending = False
            return chosen_lane, EMERGENCY_HOLD, f"🚨 EV Priority (Dynamic) — {len(self.ev_queue)} more in queue"

        self.is_serving_ev = False

        # 2. Resume normal sequence after all EVs have been served
        if self.resume_index is not None:
            self.current_index = self.resume_index
            self.resume_index  = None

        # 3. Stay-Green logic (no competition)
        if current_lane and densities.get(current_lane, 0) > DENSITY_THRESHOLD:
            others_waiting = any(v > DENSITY_THRESHOLD for k, v in densities.items() if k != current_lane)
            if not others_waiting:
                return current_lane, self.compute_time(densities[current_lane]), "Stay Green (No competition)"

        # 4. Standard Round-Robin
        start_search = (self.current_index + 1) % 4
        for i in range(4):
            idx  = (start_search + i) % 4
            lane = self.lanes[idx]
            if densities[lane] > DENSITY_THRESHOLD:
                self.current_index = idx
                return lane, self.compute_time(densities[lane]), "Normal Round-Robin"

        return None, 0, "No traffic detected"

# -------------------- UTILS --------------------
def validate_sumo_setup():
    print("\n=== LANE MAPPING VALIDATION ===")
    all_edges = traci.edge.getIDList()
    all_lanes = traci.lane.getIDList()
    
    for direction, entity_id in LANE_MAP.items():
        exists = (entity_id in all_edges) or (entity_id in all_lanes)
        status = "OK" if exists else "NOT FOUND ⚠️"
        print(f"{direction} \u2192 {entity_id}: {status}")
        if not exists:
            # Auto-detect fallback
            print(f"  [INFO] Searching for best match for {direction}...")
            # Simple heuristic: look for edges ending or starting with direction letter
            matches = [e for e in all_edges if direction in e.upper()]
            if matches: print(f"  [TIP] Did you mean one of these? {matches}")

    print("\n=== PHASE VALIDATION ===")
    try:
        logic = traci.trafficlight.getAllProgramLogics(TL_ID)[0]
        phases = logic.phases
        print(f"Traffic Light '{TL_ID}' has {len(phases)} phases.")
        for direction, phase_idx in PHASE_MAP.items():
            if phase_idx < len(phases):
                state = phases[phase_idx].state
                print(f"  Phase {phase_idx} ({direction}): {state}")
            else:
                print(f"  [ERROR] Phase index {phase_idx} for {direction} out of bounds!")
    except Exception as e:
        print(f"  [ERROR] Could not validate TL phases: {e}")
    print("==============================\n")
# -------------------- SERIAL SEND FUNCTION --------------------
def send_to_arduino(direction, color):
    global arduino
    if arduino:
        try:
            color_map = {
                "GREEN": "G",
                "YELLOW": "Y",
                "RED": "R"
            }
            msg = f"{direction}{color_map.get(color, 'R')}\n"
            arduino.write(msg.encode())
            arduino.flush()
            print(f"[ARDUINO] Sent: {msg.strip()}")
        except Exception as e:
            print(f"[ERROR] Arduino send failed: {e}")
# -------------------- MAIN --------------------
def run():
    print("[INFO] Starting SUMO-GUI...", flush=True)
    try:
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui.exe')
        traci.start([
            sumo_binary, "-c", "traffic.sumocfg", "--start"
        ])
    except Exception as e:
        print(f"[ERROR] Could not start SUMO: {e}", flush=True)
        return

    # Perform Validation
    validate_sumo_setup()

    vision = VisionAgent()
    decision = DecisionAgent()
    
    traci.trafficlight.setPhase(TL_ID, 0)
    vision.initialize_window()
    
    cv2.namedWindow("Vision Analysis", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Vision Analysis", 10, 600) 
    
    frame = vision.capture_screen()
    vision.setup_rois(frame)
    
    step = 0
    densities = {d: 0.0 for d in LANE_MAP.keys()}
    v_densities = {d: 0.0 for d in LANE_MAP.keys()} # Vision only
    s_densities = {d: 0.0 for d in LANE_MAP.keys()} # SUMO only
    current_emergency = []
    last_print_time = time.time()

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            # 1. Vision + SUMO Hybrid Processing (Every 5 steps)
            if step % 5 == 0:
                frame = vision.capture_screen()
                detections, annotated = vision.detect_vehicles(frame)
                v_densities, current_emergency = vision.map_to_lanes(detections, frame)

                # --- HYBRID PROCESSING (Vision + SUMO Fallback) ---
                for direction, entity_id in LANE_MAP.items():
                    # 1. Get raw vision density and apply Smoothing (Memory)
                    raw_v = v_densities[direction]
                    vision_val = max(raw_v, densities[direction] * 0.7)

                    # 2. Get SUMO Ground Truth
                    try:
                        sumo_count = traci.edge.getLastStepVehicleNumber(entity_id) if entity_id in traci.edge.getIDList() else traci.lane.getLastStepVehicleNumber(entity_id)
                        s_densities[direction] = float(sumo_count)
                    except:
                        s_densities[direction] = 0.0
                    
                    # 3. Vision-Priority Hybrid Decision (90% Vision / 10% SUMO)
                    # This gives 'Priority to Vision' so the AI leads the project,
                    # while SUMO acts as a 10% safety anchor.
                    densities[direction] = (vision_val * 0.9) + (s_densities[direction] * 0.1)
                
                # EV queue is now managed via notify_ev_detected() below (after vision block)
                
                annotated = vision.draw_rois(annotated)
                cv2.imshow("Vision Analysis", annotated)

            # --- 1.5. EV QUEUE UPDATE (No Preemption) ---
            # When an EV is detected the current phase is allowed to finish.
            # notify_ev_detected() queues the lane and saves where to resume.
            if current_emergency:
                decision.notify_ev_detected(current_emergency)


            # 2. State Machine for Traffic Signal
            if decision.phase_timer <= 0:
                next_lane, green_time, reason = decision.decide(densities, decision.current_phase)
                
                if next_lane and green_time > 0:
                    # --- TRANSITION START ---
                    if decision.current_phase is not None and next_lane != decision.current_phase:
                        # 1. Switch to Yellow (Phase index + 1)
                        traci.trafficlight.setPhase(TL_ID, PHASE_MAP[decision.current_phase] + 1)
                        send_to_arduino(decision.current_phase, "YELLOW")
                        for _ in range(YELLOW):
                            traci.simulationStep()
                            time.sleep(SIM_SPEED_DELAY)
                            cv2.waitKey(1)  # <-- IMPORTANT: Keeps the OpenCV window from freezing
                            step += 1
                        
                        # 2. Switch to All-Red (Phase index + 2)
                        traci.trafficlight.setPhase(TL_ID, PHASE_MAP[decision.current_phase] + 2)
                        send_to_arduino(decision.current_phase, "RED") 
                        for _ in range(ALL_RED):
                            traci.simulationStep()
                            time.sleep(SIM_SPEED_DELAY)
                            cv2.waitKey(1)  # <-- IMPORTANT: Keeps the OpenCV window from freezing
                            step += 1
                        
                    # 3. Switch to Green (Target Lane)
                    traci.trafficlight.setPhase(TL_ID, PHASE_MAP[next_lane])
                    send_to_arduino(next_lane, "GREEN")
                    
                    decision.current_phase = next_lane
                    decision.phase_timer = int(green_time)
                    decision.decision_reason = reason
                    if "Emergency" in reason: decision.total_emergencies += 1
                    # --- TRANSITION END ---
                else:
                    if decision.current_phase is not None:
                        traci.trafficlight.setPhase(TL_ID, PHASE_MAP[decision.current_phase] + 1)
                        send_to_arduino(decision.current_phase, "YELLOW")
                        for _ in range(YELLOW): 
                            traci.simulationStep()
                            time.sleep(SIM_SPEED_DELAY)
                            cv2.waitKey(1)  # <-- IMPORTANT: Keeps the OpenCV window from freezing
                            step += 1
                        decision.current_phase = None
                    traci.trafficlight.setRedYellowGreenState(TL_ID, "rrrrrrrrrrrrrrrr")
                    decision.decision_reason = "Idle (All Clear)"

            
            # 4. Step Simulation
            traci.simulationStep()
            time.sleep(SIM_SPEED_DELAY) # Full sync: 1 simulation second = SIM_SPEED_DELAY real seconds

            # 5. Debug Console Output (Strictly every 10 real-world seconds)
            if time.time() - last_print_time >= 10.0:
                last_print_time = time.time()
                os.system('cls' if os.name == 'nt' else 'clear')
                print("==============================")
                print(f"[STEP {step}] - UPDATED EVERY 10 SECONDS")
                print("\nHYBRID DENSITY (Vision | SUMO):")
                for d in LANE_MAP.keys():
                    print(f"{d}: {v_densities[d]:.1f} | {s_densities[d]:.1f} \u2192 Total: {densities[d]:.1f}")

                print("\nSIGNAL:")
                print(f"Active Lane : {decision.current_phase or 'ALL RED'}")
                print(f"Green Time  : {max(0, int(decision.phase_timer))} sec remaining")
                print(f"Reason      : {decision.decision_reason}")
                print("\nEMERGENCY:")
                is_ev = "YES" if decision.ev_queue or current_emergency else "NO"
                print(f"Detected    : {is_ev}")
                if decision.ev_queue:
                    for pos, ev_lane in enumerate(decision.ev_queue, 1):
                        wait_since = decision.ev_first_seen.get(ev_lane)
                        wait_str   = f"{time.time()-wait_since:.0f}s ago" if wait_since else "unknown"
                        marker = "▶ NEXT" if pos == 1 else f"  #{pos} "
                        print(f"  {marker}  Lane {ev_lane}  (detected {wait_str})")
                    if decision.resume_index is not None:
                        print(f"Resume After: {decision.lanes[decision.resume_index]} (normal round-robin)")
                    if decision.ev_pending:
                        print(f"EV Status   : ⏳ Serving lane {decision.current_phase} — {max(0,int(decision.phase_timer))}s left, then EV queue begins")
                elif current_emergency:
                    print(f"EV Lane     : {current_emergency[0]} (detected — queueing next cycle)")
                else:
                    print(f"EV Lane     : NONE")
                if arduino: print(f"Arduino     : Connected on {ARDUINO_PORT}")
                else: print("Arduino     : NOT CONNECTED")
                print("==============================")

            if cv2.waitKey(1) == ord('q'): break

            # --- DYNAMIC EV CLEARING ---
            # If we are serving an EV and it is no longer detected in the lane, 
            # we end the priority phase early to resume normal traffic flow.
            if decision.is_serving_ev and decision.phase_timer > 2:
                if decision.current_phase not in current_emergency:
                    print(f"[EV] Lane {decision.current_phase} cleared. Resuming normal process...", flush=True)
                    decision.phase_timer = 0
                    decision.is_serving_ev = False

            if decision.phase_timer > 0:
                decision.phase_timer -= 1  # Standard decrement
            
            step += 1
            
    except Exception as e:
        print(f"\n[ERROR] Simulation Error: {e}", flush=True)
    finally:
        cv2.destroyAllWindows()
        try: traci.close()
        except: pass
        print("\n[INFO] Simulation ended.")

if __name__ == "__main__":
    run()
