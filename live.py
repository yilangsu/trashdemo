import os
# CRITICAL: Force gpiozero to use pigpio for glitch-free hardware PWM
os.environ['GPIOZERO_PIN_FACTORY'] = 'pigpio'

import io
import json
import socketserver
from http import server
from threading import Condition, Lock, Thread
from time import monotonic, sleep
from urllib.parse import parse_qs, urlparse
from PIL import Image
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import pigpio
from classify import classify_pil, classify_pil_gemini

try:
    import board
    import busio
    import adafruit_vl53l1x
except ImportError:  # pragma: no cover - optional dependency for auto-trigger
    board = None
    busio = None
    adafruit_vl53l1x = None

SERVO_PIN = 18
PORT = 8000
HOME_US = 1500
RECYCLE_US = 1850
TRASH_US = 1150
AUTO_TRIGGER = True
PRESENT_THRESHOLD_PCT = 20.0
ABSENT_THRESHOLD_PCT = 8.0
PRESENT_CONFIRMATIONS = 2
ABSENT_CONFIRMATIONS = 6
BASELINE_SAMPLES = 8
BASELINE_TIMEOUT_SECONDS = 8.0
SENSOR_POLL_SECONDS = 0.05

pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("Could not connect to pigpiod. Start it with: sudo pigpiod")

classify_lock = Lock()
auto_armed = True
present_streak = 0
absent_streak = 0
status_text = "AUTO not armed"
sensor_state = "idle"
sensor_distance_mm = None
sensor_delta_mm = None
sensor_delta_pct = None
sort_active = False
baseline_distance_mm = None
tof_sensor = None

# Which classifier the SORT button / auto-trigger uses.
# False = on-device WasteNet (default, offline). True = Gemini + Philadelphia rules.
use_gemini = False
gemini_error = None


def _set_engine(gemini_on):
    global use_gemini
    use_gemini = bool(gemini_on)
    return "gemini" if use_gemini else "local"


def _run_classifier(image):
    """Pick the active engine. Returns (label, score, is_recycle, reason, engine).
    If Gemini is on but fails, fall back to the on-device model so a demo never
    hard-stops."""
    global gemini_error
    if use_gemini:
        try:
            label, score, is_recycle, reason = classify_pil_gemini(image)
            gemini_error = None
            return label, score, is_recycle, reason, "gemini"
        except Exception as exc:  # network down, missing key/package, etc.
            gemini_error = str(exc)
            print(f"Gemini classify failed, using on-device model instead: {exc}")
    label, score, is_recycle = classify_pil(image)
    return label, score, is_recycle, "", "local"


def _read_sensor_distance_mm():
    if tof_sensor is None:
        return None
    if tof_sensor.data_ready:
        distance_mm = tof_sensor.distance
        tof_sensor.clear_interrupt()
        return distance_mm
    return None


def _wait_for_sensor_distance_mm(timeout_seconds=0.25):
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        distance_mm = _read_sensor_distance_mm()
        if distance_mm is not None:
            return distance_mm
        sleep(SENSOR_POLL_SECONDS)
    return None


def _arm_auto_from_sensor():
    global baseline_distance_mm, auto_armed, present_streak, absent_streak, status_text
    global sensor_state, sensor_distance_mm, sensor_delta_mm, sensor_delta_pct
    if tof_sensor is None:
        status_text = "AUTO unavailable: VL53L1X is not connected"
        sensor_state = "unavailable"
        print(status_text)
        return False

    status_text = "AUTO arming from ToF sensor..."
    sensor_state = "arming"
    sleep(0.5)
    samples = []
    attempts = 0
    max_attempts = max(BASELINE_SAMPLES, int(BASELINE_TIMEOUT_SECONDS / SENSOR_POLL_SECONDS))

    # Collect a short empty-platform baseline before arming.
    while len(samples) < BASELINE_SAMPLES and attempts < max_attempts:
        distance_mm = _wait_for_sensor_distance_mm(0.25)
        if distance_mm is not None:
            samples.append(distance_mm)
        sleep(SENSOR_POLL_SECONDS)
        attempts += 1

    if not samples:
        status_text = "AUTO arming failed: no ToF readings"
        sensor_state = "arming_failed"
        print(status_text)
        return False

    baseline_distance_mm = float(sum(samples) / len(samples))
    auto_armed = True
    present_streak = 0
    absent_streak = 0
    sensor_state = "armed"
    sensor_distance_mm = None
    sensor_delta_mm = None
    sensor_delta_pct = None
    status_text = f"AUTO armed: baseline {baseline_distance_mm:.1f} mm"
    print(f"Auto-trigger armed from baseline {baseline_distance_mm:.1f} mm")
    return True


def _classify_frame(frame, source):
    with classify_lock:
        image = Image.open(io.BytesIO(frame))
        label, score, is_recycle, reason, engine = _run_classifier(image)

    tail = f" | {reason}" if reason else ""
    print(f"{source} SORT [{engine}] -> {label} ({score:.0%}) "
          f"{'RECYCLE' if is_recycle else 'TRASH'}{tail}")
    return label, score, is_recycle, reason, engine


def _sort_frame(frame, source):
    global sort_active, status_text
    sort_active = True
    status_text = f"{source} sorting..."
    try:
        label, score, is_recycle, reason, engine = _classify_frame(frame, source)
        Thread(target=move_servo, args=(is_recycle,), daemon=True).start()
        return label, score, is_recycle, reason, engine
    finally:
        sort_active = False


def _auto_monitor():
    global auto_armed, present_streak, absent_streak, baseline_distance_mm, status_text
    global sensor_state, sensor_distance_mm, sensor_delta_mm, sensor_delta_pct
    if not AUTO_TRIGGER:
        return
    if tof_sensor is None:
        print("Auto-trigger disabled: VL53L1X is not available")
        sensor_state = "unavailable"
        return

    print("Auto-trigger monitor running")
    while True:
        if classify_lock.locked() or sort_active:
            sleep(SENSOR_POLL_SECONDS)
            continue

        distance_mm = _read_sensor_distance_mm()
        if distance_mm is None or baseline_distance_mm is None:
            if baseline_distance_mm is None:
                sensor_state = "waiting_for_baseline"
            else:
                sensor_state = "waiting_for_reading"
            sleep(SENSOR_POLL_SECONDS)
            continue

        sensor_distance_mm = float(distance_mm)
        delta_mm = baseline_distance_mm - float(distance_mm)
        delta_pct = (delta_mm / baseline_distance_mm * 100.0) if baseline_distance_mm > 0 else 0.0
        sensor_delta_mm = delta_mm
        sensor_delta_pct = delta_pct
        if auto_armed and delta_pct >= PRESENT_THRESHOLD_PCT:
            sensor_state = "present"
            present_streak += 1
            absent_streak = 0
            status_text = f"AUTO detecting: {delta_pct:.0f}% change ({delta_mm:.1f} mm)"
        elif delta_pct <= ABSENT_THRESHOLD_PCT:
            sensor_state = "absent"
            absent_streak += 1
            present_streak = 0
            status_text = f"AUTO waiting: {distance_mm:.1f} mm"
        else:
            sensor_state = "uncertain"
            present_streak = 0
            absent_streak = 0

        if auto_armed and present_streak >= PRESENT_CONFIRMATIONS:
            auto_armed = False
            present_streak = 0
            sensor_state = "fired"
            status_text = "AUTO fired"
            with output.condition:
                frame = output.frame
            if frame is not None:
                Thread(target=_sort_frame, args=(frame, "AUTO"), daemon=True).start()
        elif not auto_armed and absent_streak >= ABSENT_CONFIRMATIONS:
            auto_armed = True
            sensor_state = "rearmed"
            status_text = f"AUTO re-armed at {distance_mm:.1f} mm"
            present_streak = 0
            absent_streak = 0
            print("Auto-trigger re-armed")
        sleep(SENSOR_POLL_SECONDS)

def move_servo(is_recycle):
    """Move to recycle/trash PWM, then return to center."""
    pulsewidth = RECYCLE_US if is_recycle else TRASH_US
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)
    sleep(2.0)
    pi.set_servo_pulsewidth(SERVO_PIN, HOME_US)

# --- the web page ---
PAGE = """<!doctype html>
<html><head><title>Trash vs Recycle</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:sans-serif;text-align:center;background:#111;color:#eee;margin:0;padding:16px}
img{max-width:100%;border-radius:12px;border:3px solid #333}
button{font-size:22px;padding:16px 28px;margin:12px;border:0;border-radius:12px;
background:#2e7d32;color:#fff;cursor:pointer}
button:active{background:#1b5e20}
#result{font-size:34px;font-weight:bold;min-height:50px;margin-top:8px}
.recycle{color:#4caf50}.trash{color:#ff5252}.thinking{color:#ffb300}
.status{font-size:18px;color:#bbb;min-height:24px;margin-top:4px}
.engine{margin:10px auto 0;max-width:520px}
.switch{display:inline-flex;align-items:center;gap:10px;font-size:18px;cursor:pointer;user-select:none}
.switch input{width:22px;height:22px}
.enginenote{font-size:15px;color:#9ad;margin-top:6px}
.reason{font-size:19px;color:#cfd8dc;min-height:26px;margin-top:4px;font-style:italic}
.sensor{margin:16px auto 0;max-width:520px;padding:12px 14px;border:1px solid #333;border-radius:12px;background:#171717;text-align:left;font-size:16px;line-height:1.5}
.sensor strong{color:#fff}
</style></head>
<body>
<h1>Trash vs Recycle</h1>
<img src="/stream.mjpg">
<div><button onclick="sort()">SORT</button></div>
<div><button onclick="armAuto()">ARM AUTO</button></div>
<div class="engine">
<label class="switch"><input type="checkbox" id="geminiToggle" onchange="setEngine()"> <span>🤖 Use Gemini 3 + Philadelphia rules</span></label>
<div id="engineNote" class="enginenote">📟 On-device model (WasteNet)</div>
</div>
<div id="result">Hold an object in view, then hit SORT</div>
<div id="reason" class="reason"></div>
<div id="status">AUTO not armed</div>
<div class="sensor">
    <div><strong>State:</strong> <span id="sensorState">idle</span></div>
    <div><strong>Baseline:</strong> <span id="baselineValue">-</span></div>
    <div><strong>Distance:</strong> <span id="distanceValue">-</span></div>
    <div><strong>Delta:</strong> <span id="deltaValue">-</span></div>
    <div><strong>Delta %:</strong> <span id="deltaPctValue">-</span></div>
    <div><strong>Armed:</strong> <span id="armedValue">yes</span></div>
    <div><strong>Present streak:</strong> <span id="presentValue">0</span></div>
    <div><strong>Absent streak:</strong> <span id="absentValue">0</span></div>
</div>
<script>
function sort(){
var r=document.getElementById('result');
var reason=document.getElementById('reason');
r.className='thinking';
r.textContent='📸 SNAP! thinking...';
reason.textContent='';
fetch('/sort').then(x=>x.json()).then(d=>{
r.className = d.is_recycle ? 'recycle' : 'trash';
r.textContent = (d.is_recycle?'♻️ RECYCLE':'🗑️ TRASH') + ' — '+d.label+' ('+Math.round(d.score*100)+'%)';
reason.textContent = d.reason ? ('“'+d.reason+'”  ·  '+(d.engine==='gemini'?'Gemini · Philly rules':'on-device')) : '';
}).catch(e=>{r.className='trash';r.textContent='error: '+e;});
}
function setEngine(){
var on=document.getElementById('geminiToggle').checked;
var note=document.getElementById('engineNote');
note.textContent = on ? 'switching to Gemini…' : 'switching to on-device model…';
fetch('/engine?use=' + (on?'gemini':'local')).then(x=>x.json()).then(d=>{
note.textContent = d.engine==='gemini' ? '☁️ Gemini 3 + Philadelphia rules (cloud)' : '📟 On-device model (WasteNet)';
}).catch(e=>{note.textContent='error: '+e;});
}
function armAuto(){
var s=document.getElementById('status');
s.textContent='arming from current empty view...';
fetch('/arm').then(x=>x.json()).then(d=>{s.textContent=d.status;}).catch(e=>{s.textContent='error: '+e;});
}
function refreshStatus(){
fetch('/status').then(x=>x.json()).then(d=>{
document.getElementById('status').textContent=d.status;
if(d.engine){
var note=document.getElementById('engineNote');
if(d.engine==='gemini'){
note.textContent = d.gemini_error ? ('⚠️ Gemini failed, using on-device — '+d.gemini_error) : '☁️ Gemini 3 + Philadelphia rules (cloud)';
}else{
note.textContent = '📟 On-device model (WasteNet)';
}
}
document.getElementById('sensorState').textContent=d.sensor_state ?? 'idle';
document.getElementById('baselineValue').textContent=d.baseline_distance_mm == null ? '-' : Math.round(d.baseline_distance_mm) + ' mm';
document.getElementById('distanceValue').textContent=d.sensor_distance_mm == null ? '-' : Math.round(d.sensor_distance_mm) + ' mm';
document.getElementById('deltaValue').textContent=d.sensor_delta_mm == null ? '-' : d.sensor_delta_mm.toFixed(1) + ' mm';
document.getElementById('deltaPctValue').textContent=d.sensor_delta_pct == null ? '-' : d.sensor_delta_pct.toFixed(0) + '%';
document.getElementById('armedValue').textContent=d.auto_armed ? 'yes' : 'no';
document.getElementById('presentValue').textContent=d.present_streak;
document.getElementById('absentValue').textContent=d.absent_streak;
}).catch(()=>{});
}
setInterval(refreshStatus, 500);
</script>
</body></html>"""

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            # buf is a memoryview into the encoder's DMA buffer, which gets
            # recycled for the next frame as soon as this call returns -- copy
            # it now or later readers of self.frame get stale/torn data.
            self.frame = bytes(buf)
            self.condition.notify_all()

output = StreamingOutput()

class Handler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", 0)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception:
                pass  # browser disconnected
        elif self.path == "/sort":
            with output.condition:
                if output.frame is None:
                    output.condition.wait(timeout=1.0)
                frame = output.frame
            if frame is None:
                image = picam2.capture_image("main")
                with classify_lock:
                    label, score, is_recycle, reason, engine = _run_classifier(image)
            else:
                label, score, is_recycle, reason, engine = _classify_frame(frame, "MANUAL")
            data = json.dumps({
                "label": label,
                "score": score,
                "is_recycle": is_recycle,
                "reason": reason,
                "engine": engine,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)
            Thread(target=move_servo, args=(is_recycle,), daemon=True).start()
        elif self.path.startswith("/engine"):
            params = parse_qs(urlparse(self.path).query)
            engine = _set_engine(params.get("use", ["local"])[0] == "gemini")
            data = json.dumps({"engine": engine, "gemini_error": gemini_error}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/arm":
            if tof_sensor is None:
                status = "VL53L1X not available"
            else:
                _arm_auto_from_sensor()
                status = status_text
            data = json.dumps({"status": status}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/status":
            data = json.dumps({
                "status": status_text,
                "sensor_state": sensor_state,
                "baseline_distance_mm": baseline_distance_mm,
                "sensor_distance_mm": sensor_distance_mm,
                "sensor_delta_mm": sensor_delta_mm,
                "sensor_delta_pct": sensor_delta_pct,
                "auto_armed": auto_armed,
                "present_streak": present_streak,
                "absent_streak": absent_streak,
                "engine": "gemini" if use_gemini else "local",
                "gemini_error": gemini_error,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass  # quiet

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

if __name__ == "__main__":
    print("Warming up model (takes a few seconds)...")
    classify_pil(Image.new("RGB", (224, 224)))

    if board is not None and busio is not None and adafruit_vl53l1x is not None:
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            tof_sensor = adafruit_vl53l1x.VL53L1X(i2c)
            tof_sensor.distance_mode = 2
            tof_sensor.timing_budget = 100
            tof_sensor.start_ranging()
            print("VL53L1X ready for auto-trigger")
            Thread(target=_arm_auto_from_sensor, daemon=True).start()
        except OSError as exc:
            tof_sensor = None
            print(f"VL53L1X unavailable, auto-trigger will be disabled: {exc}")
    else:
        print("VL53L1X libraries missing, auto-trigger will be unavailable")

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.start_recording(MJPEGEncoder(), FileOutput(output))

    # Let auto white balance converge on the (usually empty) bin under our
    # fixed lighting, then freeze it. Auto white balance re-guessing per shot
    # is what pushes reflective/metallic items toward a warm, cardboard-ish
    # color cast.
    sleep(1.0)
    awb_gains = picam2.capture_metadata().get("ColourGains", (1.0, 1.0))
    picam2.set_controls({"AwbEnable": False, "ColourGains": awb_gains})
    print(f"White balance locked at gains {awb_gains}")

    Thread(target=_auto_monitor, daemon=True).start()
    print(f"\n LIVE! open this on your Mac browser: http://10.103.210.108:{PORT}\n")
    try:
        StreamingServer(("", PORT), Handler).serve_forever()
    finally:
        if tof_sensor is not None:
            tof_sensor.stop_ranging()
        picam2.stop_recording()
