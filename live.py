import os
# CRITICAL: Force gpiozero to use pigpio for glitch-free hardware PWM
os.environ['GPIOZERO_PIN_FACTORY'] = 'pigpio'

import io
import json
import socketserver
import subprocess
from http import server
from threading import Condition, Lock, Thread
from time import monotonic, sleep
from urllib.parse import parse_qs, urlparse
from PIL import Image
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import pigpio
from classify import classify_pil, classify_pil_gemini, EWASTE

try:
    import board
    import busio
    import adafruit_vl53l1x
except ImportError:  # pragma: no cover - optional dependency for auto-trigger
    board = None
    busio = None
    adafruit_vl53l1x = None

SERVO_PIN = 18
# Warning LED for rejected e-waste (battery / electronics). Wired to a Pi GPIO
# pin exactly like the servo -- LED + resistor from this pin to ground. Change
# this if you wired the LED to a different pin.
# GPIO16 = physical pin 36, on the bottom row of the 40-pin header.
LED_PIN = 16
LED_BLINK_INTERVAL = 0.25  # seconds on / off while the e-waste alert is active
# Manual-sort push switch: one leg to GPIO23, the other to ground. Internal
# pull-up keeps the pin high until pressed, so a press reads as a falling edge.
# GPIO23 = physical pin 16.
BUTTON_PIN = 23
BUTTON_DEBOUNCE_US = 250000  # 250ms glitch filter to ignore switch bounce
PORT = 8000
HOME_US = 1500
RECYCLE_US = 2200
TRASH_US = 800
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

# e-waste warning LED: drive it low (off) to start.
pi.set_mode(LED_PIN, pigpio.OUTPUT)
pi.write(LED_PIN, 0)

# Manual-sort push switch.
pi.set_mode(BUTTON_PIN, pigpio.INPUT)
pi.set_pull_up_down(BUTTON_PIN, pigpio.PUD_UP)
pi.set_glitch_filter(BUTTON_PIN, BUTTON_DEBOUNCE_US)

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

# e-waste rejection: when the classifier spots a battery/electronics we refuse
# to sort it (servo stays home) and blink the warning LED until it's cleared.
ewaste_alert = False
ewaste_item = None

# Last classification result, whatever triggered it (page SORT button, GPIO
# button, or auto-trigger) -- /status exposes this so the page can show it
# even when the sort didn't come from the page's own fetch('/sort') call.
last_result_source = None
last_result_label = None
last_result_score = None
last_result_is_recycle = None
last_result_reason = None
last_result_engine = None
last_result_is_ewaste = None


def _record_result(source, label, score, is_recycle, reason, engine, is_ewaste):
    global last_result_source, last_result_label, last_result_score
    global last_result_is_recycle, last_result_reason, last_result_engine, last_result_is_ewaste
    last_result_source = source
    last_result_label = label
    last_result_score = score
    last_result_is_recycle = is_recycle
    last_result_reason = reason
    last_result_engine = engine
    last_result_is_ewaste = is_ewaste

# Which classifier the SORT button / auto-trigger uses.
# Always Gemini 3 + Philadelphia rules. On-device WasteNet stays available only as
# an automatic fail-safe (see _run_classifier) when Gemini errors out -- it is
# deliberately NOT a user-selectable mode.
use_gemini = True
gemini_error = None


def _set_engine(gemini_on):
    global use_gemini
    use_gemini = bool(gemini_on)
    return "gemini" if use_gemini else "local"


def _run_classifier(image):
    """Pick the active engine. Returns
    (label, score, is_recycle, reason, engine, is_ewaste).
    If Gemini is on but fails, fall back to the on-device model so a demo never
    hard-stops."""
    global gemini_error
    if use_gemini:
        try:
            label, score, is_recycle, reason, is_ewaste = classify_pil_gemini(image)
            gemini_error = None
            return label, score, is_recycle, reason, "gemini", is_ewaste
        except Exception as exc:  # network down, missing key/package, etc.
            gemini_error = str(exc)
            print(f"Gemini classify failed, using on-device model instead: {exc}")
    label, score, is_recycle = classify_pil(image)
    # the on-device WasteNet model has a "battery" class -- treat it as e-waste.
    is_ewaste = label in EWASTE
    if is_ewaste:
        is_recycle = False
    return label, score, is_recycle, "", "local", is_ewaste


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
        label, score, is_recycle, reason, engine, is_ewaste = _run_classifier(image)

    if is_ewaste:
        bucket = "E-WASTE (rejected)"
    else:
        bucket = "RECYCLE" if is_recycle else "TRASH"
    tail = f" | {reason}" if reason else ""
    print(f"{source} SORT [{engine}] -> {label} ({score:.0%}) {bucket}{tail}")
    _record_result(source, label, score, is_recycle, reason, engine, is_ewaste)
    return label, score, is_recycle, reason, engine, is_ewaste


def _sort_frame(frame, source):
    global sort_active, status_text
    sort_active = True
    status_text = f"{source} sorting..."
    try:
        label, score, is_recycle, reason, engine, is_ewaste = _classify_frame(frame, source)
        _act_on_result(label, is_recycle, is_ewaste)
        return label, score, is_recycle, reason, engine, is_ewaste
    finally:
        sort_active = False


def _button_sort_trigger():
    if classify_lock.locked() or sort_active:
        return  # already mid-sort, ignore this press
    with output.condition:
        if output.frame is None:
            output.condition.wait(timeout=1.0)
        frame = output.frame
    if frame is not None:
        _sort_frame(frame, "BUTTON")
    else:
        image = picam2.capture_image("main")
        with classify_lock:
            label, score, is_recycle, reason, engine, is_ewaste = _run_classifier(image)
        _record_result("BUTTON", label, score, is_recycle, reason, engine, is_ewaste)
        _act_on_result(label, is_recycle, is_ewaste)


def _on_button_edge(gpio, level, tick):
    if level == 0:  # falling edge: pull-up wiring means pressed == low
        Thread(target=_button_sort_trigger, daemon=True).start()


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
            # platform is empty again -> whatever was there (incl. rejected
            # e-waste) has been removed, so stop the warning LED.
            _clear_ewaste_alert()
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


def _led_blinker():
    """Background thread: blink the warning LED while an e-waste alert is active,
    keep it off otherwise. Runs for the life of the process."""
    led_on = False
    while True:
        if ewaste_alert:
            led_on = not led_on
            pi.write(LED_PIN, 1 if led_on else 0)
            sleep(LED_BLINK_INTERVAL)
        else:
            if led_on:
                led_on = False
                pi.write(LED_PIN, 0)
            sleep(0.05)


def _raise_ewaste_alert(item):
    """Flag rejected e-waste: start the blinking LED and hold the servo at home
    (we refuse to sort a battery into either bin)."""
    global ewaste_alert, ewaste_item, status_text
    ewaste_alert = True
    ewaste_item = item
    status_text = f"E-WASTE REJECTED: {item} -- remove it, take to e-waste drop-off"
    pi.set_servo_pulsewidth(SERVO_PIN, HOME_US)  # stay centered, don't sort
    print(f"E-WASTE DETECTED: {item} -> LED blinking, servo held at HOME")


def _clear_ewaste_alert():
    """Stop the warning LED (item removed / replaced with something sortable)."""
    global ewaste_alert, ewaste_item
    if ewaste_alert:
        print("E-waste alert cleared")
    ewaste_alert = False
    ewaste_item = None


def _act_on_result(label, is_recycle, is_ewaste):
    """Decide what the hardware does with a classification: blink-and-hold for
    e-waste, otherwise clear any prior alert and drive the sorting servo."""
    if is_ewaste:
        _raise_ewaste_alert(label)
    else:
        _clear_ewaste_alert()
        Thread(target=move_servo, args=(is_recycle,), daemon=True).start()

# --- the web page ---
DEBUG_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Trash vs Recycle — DEBUG</title>
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
.enginenote{display:inline-block;font-size:16px;font-weight:bold;margin-top:8px;padding:7px 16px;border-radius:999px;background:#263238;color:#cfd8dc}
.enginenote.local{background:#1b5e20;color:#fff}
.enginenote.gemini{background:#4527a0;color:#fff}
.enginenote.err{background:#b71c1c;color:#fff;font-weight:normal;font-size:14px}
.reason{font-size:19px;color:#cfd8dc;min-height:26px;margin-top:4px;font-style:italic}
.sensor{margin:16px auto 0;max-width:520px;padding:12px 14px;border:1px solid #333;border-radius:12px;background:#171717;text-align:left;font-size:16px;line-height:1.5}
.sensor strong{color:#fff}
.ewaste{display:none;margin:14px auto 0;max-width:520px;padding:16px;border-radius:12px;background:#b71c1c;color:#fff;font-size:23px;font-weight:bold;animation:ewasteblink 0.5s steps(1,end) infinite}
.ewaste.show{display:block}
.ewaste small{display:block;font-size:15px;font-weight:normal;margin-top:6px}
.clearbtn{background:#455a64}
.clearbtn:active{background:#263238}
@keyframes ewasteblink{50%{background:#3b0000}}
.layout{display:flex;gap:28px;align-items:flex-start;justify-content:center;flex-wrap:wrap;text-align:left;max-width:1100px;margin:0 auto}
.col-cam{flex:1 1 460px;max-width:640px;text-align:center}
.col-status{flex:1 1 380px;max-width:520px}
.col-status .engine{margin:10px 0 0}
.col-status .sensor{margin:16px 0 0}
.col-status .ewaste{margin:14px 0 0}
</style></head>
<body>
<h1>Trash vs Recycle — DEBUG <a href="/" style="font-size:14px;color:#7aa">(open demo screen)</a></h1>
<div class="layout">
<div class="col-cam">
<img src="/stream.mjpg">
<div><button onclick="sort()">SORT</button></div>
<div><button onclick="armAuto()">ARM AUTO</button></div>
</div>
<div class="col-status">
<div id="result">Hold an object in view, then hit SORT</div>
<div id="reason" class="reason"></div>
<div id="status">AUTO not armed</div>
<div id="ewaste" class="ewaste">⛔ E-WASTE DETECTED — DO NOT RECYCLE
<small id="ewasteItem"></small>
<small>Remove it and take it to an e-waste / hazardous-waste drop-off. The warning LED is blinking.</small>
<div><button class="clearbtn" onclick="clearAlert()">CLEAR ALERT</button></div>
</div>
<div class="engine">
<div id="engineNote" class="enginenote gemini">☁️ ACTIVE: Gemini 3 + Philadelphia rules (auto-falls back to on-device if it errors)</div>
</div>
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
</div>
</div>
<script>
var manualSortInFlight = false;
function sort(){
manualSortInFlight = true;
var r=document.getElementById('result');
var reason=document.getElementById('reason');
r.className='thinking';
r.textContent='📸 SNAP! thinking...';
reason.textContent='';
fetch('/sort').then(x=>x.json()).then(d=>{
if(d.is_ewaste){
r.className='trash';
r.textContent='⛔ E-WASTE — '+d.label+' (rejected, not sorted)';
showEwaste(true, d.label);
}else{
r.className = d.is_recycle ? 'recycle' : 'trash';
r.textContent = (d.is_recycle?'♻️ RECYCLE':'🗑️ TRASH') + ' — '+d.label+' ('+Math.round(d.score*100)+'%)';
showEwaste(false);
}
var tag = d.engine==='gemini' ? 'Gemini · Philly rules' : '⚠️ Gemini failed → on-device fallback';
reason.textContent = d.reason ? ('“'+d.reason+'”  ·  '+tag) : tag;
}).catch(e=>{r.className='trash';r.textContent='error: '+e;}).then(()=>{manualSortInFlight=false;});
}
function showEwaste(on, item){
var b=document.getElementById('ewaste');
if(on){b.classList.add('show');document.getElementById('ewasteItem').textContent = item ? ('Detected: '+item) : '';}
else{b.classList.remove('show');}
}
function clearAlert(){
fetch('/clear').then(x=>x.json()).then(d=>{showEwaste(false);}).catch(e=>{});
}
function applyEngineBadge(engine, err){
var note=document.getElementById('engineNote');
if(engine==='gemini' && !err){
note.className='enginenote gemini';
note.textContent='☁️ ACTIVE: Gemini + Philadelphia rules';
}else if(engine==='gemini' && err){
note.className='enginenote err';
note.textContent='⚠️ Gemini failed → running on-device. '+err;
}else{
note.className='enginenote local';
note.textContent='📟 ACTIVE: On-device ensemble (WasteNet + ONNX)';
}
}
function armAuto(){
var s=document.getElementById('status');
s.textContent='arming from current empty view...';
fetch('/arm').then(x=>x.json()).then(d=>{s.textContent=d.status;}).catch(e=>{s.textContent='error: '+e;});
}
function refreshStatus(){
fetch('/status').then(x=>x.json()).then(d=>{
document.getElementById('status').textContent=d.status;
if(d.engine){ applyEngineBadge(d.engine, d.gemini_error); }
document.getElementById('sensorState').textContent=d.sensor_state ?? 'idle';
document.getElementById('baselineValue').textContent=d.baseline_distance_mm == null ? '-' : Math.round(d.baseline_distance_mm) + ' mm';
document.getElementById('distanceValue').textContent=d.sensor_distance_mm == null ? '-' : Math.round(d.sensor_distance_mm) + ' mm';
document.getElementById('deltaValue').textContent=d.sensor_delta_mm == null ? '-' : d.sensor_delta_mm.toFixed(1) + ' mm';
document.getElementById('deltaPctValue').textContent=d.sensor_delta_pct == null ? '-' : d.sensor_delta_pct.toFixed(0) + '%';
document.getElementById('armedValue').textContent=d.auto_armed ? 'yes' : 'no';
document.getElementById('presentValue').textContent=d.present_streak;
document.getElementById('absentValue').textContent=d.absent_streak;
showEwaste(d.ewaste_alert, d.ewaste_item);
if(!manualSortInFlight){
var r=document.getElementById('result');
var reasonEl=document.getElementById('reason');
if(d.sort_active){
r.className='thinking';
r.textContent='📸 SNAP! thinking...';
reasonEl.textContent='';
}else if(d.last_result_label){
if(d.last_result_is_ewaste){
r.className='trash';
r.textContent='⛔ E-WASTE — '+d.last_result_label+' (rejected, not sorted)';
}else{
r.className = d.last_result_is_recycle ? 'recycle' : 'trash';
r.textContent = (d.last_result_is_recycle?'♻️ RECYCLE':'🗑️ TRASH') + ' — '+d.last_result_label+' ('+Math.round(d.last_result_score*100)+'%)';
}
var tag = d.last_result_engine==='gemini' ? 'Gemini · Philly rules' : '⚠️ Gemini failed → on-device fallback';
var srcTag = d.last_result_source ? (' · via '+d.last_result_source) : '';
reasonEl.textContent = (d.last_result_reason ? ('“'+d.last_result_reason+'”  ·  '+tag) : tag) + srcTag;
}
}
}).catch(()=>{});
}
setInterval(refreshStatus, 500);
</script>
</body></html>"""

# --- the demo screen shown full-screen on the Pi's HDMI display ---
# Deliberately minimal: live camera + one big verdict card + a full-screen
# e-waste warning. No SORT / ARM / engine controls -- classification is driven
# by the physical GPIO23 button (and the ToF auto-trigger when wired). Every
# bit of state comes from polling /status, so this page carries no logic of its
# own beyond rendering last_result_* and ewaste_alert.
DEMO_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Smart Waste Sorter</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:radial-gradient(120% 120% at 50% -8%,#13291c 0%,#0a1610 45%,#040704 100%);
color:#ffffff;display:flex;flex-direction:column;height:100vh;padding:2.2vmin}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:2vmin;padding:0 0.4vmin 1.4vmin}
.brand-logo{height:clamp(34px,7vmin,88px);width:auto;display:block}
.rules{display:inline-flex;align-items:center;gap:1vmin;font-size:clamp(12px,1.9vmin,24px);font-weight:800;
color:#fff;background:rgba(150,190,156,.16);border:0.3vmin solid rgba(150,190,156,.55);
padding:0.7vmin 1.9vmin;border-radius:999px;letter-spacing:.4px;white-space:nowrap}
.rules .dot{width:1.3vmin;height:1.3vmin;min-width:8px;min-height:8px;border-radius:50%;background:#96be9c;box-shadow:0 0 1.6vmin #96be9c}
.grid{flex:1;display:flex;gap:2.4vmin;min-height:0}
.camera{flex:1.35;min-width:0;position:relative;border-radius:2.2vmin;overflow:hidden;
background:#000;border:0.35vmin solid rgba(150,190,156,.4);box-shadow:0 1.4vmin 4vmin rgba(0,0,0,.55)}
.camera img{width:100%;height:100%;object-fit:cover;display:block}
.camera .tag{position:absolute;top:1.4vmin;left:1.4vmin;background:rgba(4,7,4,.6);display:inline-flex;align-items:center;gap:0.9vmin;
backdrop-filter:blur(6px);padding:0.7vmin 1.6vmin;border-radius:999px;font-size:clamp(11px,1.5vmin,18px);font-weight:600;color:#e2f2e6}
.camera .tag .rec{width:1.2vmin;height:1.2vmin;min-width:7px;min-height:7px;border-radius:50%;background:#ff5b5b;box-shadow:0 0 1.2vmin #ff5b5b}
.card{flex:1;min-width:0;border-radius:2.2vmin;padding:3vmin;display:flex;flex-direction:column;
justify-content:center;gap:1.8vmin;border:0.35vmin solid rgba(255,255,255,.08);
background:linear-gradient(160deg,rgba(255,255,255,.06),rgba(255,255,255,.015));
box-shadow:0 1.4vmin 4vmin rgba(0,0,0,.5);transition:background .35s,border-color .35s}
.kicker{font-size:clamp(13px,2vmin,24px);font-weight:800;letter-spacing:3px;text-transform:uppercase;color:#96be9c}
.verdict{font-size:clamp(34px,7.4vmin,96px);font-weight:900;line-height:1.02;letter-spacing:-1px;color:#fff}
.item{font-size:clamp(20px,3.6vmin,46px);font-weight:700;color:#fff;text-transform:capitalize}
.meter-label{font-size:clamp(12px,1.7vmin,20px);font-weight:700;color:#cfe6d6}
.reason{font-size:clamp(15px,2.4vmin,30px);line-height:1.35;color:#fff;font-style:italic}
/* verdict theming -- green=recycle, graphite=landfill, red=e-waste */
.card.idle .kicker{color:#96be9c}.card.idle .verdict{color:#eef7f0}
.card.thinking{border-color:rgba(150,190,156,.45);background:linear-gradient(160deg,rgba(150,190,156,.16),rgba(150,190,156,.03))}
.card.thinking .kicker,.card.thinking .verdict{color:#acdaae}
.card.recycle{border-color:rgba(150,190,156,.75);background:linear-gradient(160deg,rgba(150,190,156,.26),rgba(150,190,156,.04))}
.card.recycle .kicker,.card.recycle .verdict{color:#a9e6b6}
.card.trash{border-color:rgba(255,255,255,.24);background:linear-gradient(160deg,rgba(255,255,255,.1),rgba(255,255,255,.02))}
.card.trash .kicker{color:#c2ccc5}.card.trash .verdict{color:#f2f6f3}
.dots::after{content:'';animation:dots 1.4s steps(4,end) infinite}
@keyframes dots{0%{content:''}25%{content:'.'}50%{content:'..'}75%{content:'...'}}
/* full-screen e-waste takeover */
.ewaste{position:fixed;inset:0;z-index:50;display:none;flex-direction:column;align-items:center;justify-content:center;
text-align:center;padding:5vmin;background:#7f1010;color:#fff;animation:ewblink 0.7s steps(1,end) infinite}
.ewaste.show{display:flex}
@keyframes ewblink{50%{background:#3a0606}}
.ewaste .ic{position:relative;width:26vmin;height:22vmin}
.ewaste .ic::before{content:"";position:absolute;left:50%;top:0;transform:translateX(-50%);width:0;height:0;
border-left:13vmin solid transparent;border-right:13vmin solid transparent;border-bottom:22vmin solid #fff}
.ewaste .ic::after{content:"!";position:absolute;left:0;right:0;bottom:1.5vmin;text-align:center;color:#7f1010;font-weight:900;font-size:12vmin;line-height:1}
.ewaste .t{font-size:clamp(34px,8vmin,120px);font-weight:900;letter-spacing:1px;margin-top:1vmin}
.ewaste .s{font-size:clamp(22px,4.6vmin,64px);font-weight:800;color:#ffd7d7;letter-spacing:4px;margin-top:0.4vmin}
.ewaste .it{font-size:clamp(18px,3.4vmin,44px);font-weight:700;margin-top:2.6vmin;text-transform:capitalize}
.ewaste .n{font-size:clamp(14px,2.3vmin,28px);color:#ffe1e1;margin-top:1.4vmin;max-width:24em}
@media (orientation:portrait){.grid{flex-direction:column}.camera{flex:1.1}}
</style></head>
<body>
<div class="topbar">
  <img class="brand-logo" src="data:image/png;base64,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" alt="SYFT">
  <div class="rules"><span class="dot"></span><span id="rulesLabel">Philadelphia rules</span></div>
</div>
<div class="grid">
  <div class="camera"><img src="/stream.mjpg" alt="live camera"><div class="tag"><span class="rec"></span>Live camera</div></div>
  <div class="card idle" id="card">
    <div class="kicker" id="kicker">Ready</div>
    <div class="verdict" id="verdict">Place an item &amp; press the button</div>
    <div class="item" id="item"></div>
    <div class="meter-label" id="meterLabel"></div>
    <div class="reason" id="reason"></div>
  </div>
</div>
<div class="ewaste" id="ewaste">
  <div class="ic"></div>
  <div class="t">E-WASTE DETECTED</div>
  <div class="s">DO NOT RECYCLE</div>
  <div class="it" id="ewItem"></div>
  <div class="n">Remove it and take it to an e-waste / hazardous-waste drop-off. The warning light is blinking.</div>
</div>
<script>
var card=document.getElementById('card');
var kicker=document.getElementById('kicker');
var verdict=document.getElementById('verdict');
var itemEl=document.getElementById('item');
var meterLabel=document.getElementById('meterLabel');
var reason=document.getElementById('reason');
var ewaste=document.getElementById('ewaste');
var ewItem=document.getElementById('ewItem');
var rulesLabel=document.getElementById('rulesLabel');
function setCard(cls){card.className='card '+cls;}
function refresh(){
fetch('/status').then(x=>x.json()).then(d=>{
  // top-right badge names the ruleset in effect right now (falls back if Gemini errors)
  rulesLabel.textContent = (d.last_result_engine==='local') ? 'Standard rules' : 'Philadelphia rules';
  // e-waste warning takes over the whole screen
  if(d.ewaste_alert){
    ewaste.classList.add('show');
    ewItem.textContent = d.ewaste_item ? (d.ewaste_item) : '';
    return;
  }
  ewaste.classList.remove('show');
  // mid-classification
  if(d.sort_active){
    setCard('thinking');
    kicker.textContent='Analyzing';
    verdict.innerHTML='Thinking<span class="dots"></span>';
    itemEl.textContent='';meterLabel.textContent='';reason.textContent='';
    return;
  }
  // a result to show
  if(d.last_result_label){
    var rec=d.last_result_is_recycle;
    setCard(rec?'recycle':'trash');
    kicker.textContent = rec ? 'Recyclable' : 'Landfill';
    verdict.textContent = rec ? 'RECYCLE' : 'TRASH';
    itemEl.textContent = d.last_result_label;
    meterLabel.textContent = Math.round((d.last_result_score||0)*100)+'% confident';
    reason.textContent = d.last_result_reason ? ('“'+d.last_result_reason+'”') : '';
    return;
  }
  // idle
  setCard('idle');
  kicker.textContent='Ready';
  verdict.textContent='Place an item & press the button';
  itemEl.textContent='';meterLabel.textContent='';reason.textContent='';
}).catch(()=>{});
}
setInterval(refresh,500);refresh();
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
        if self.path == "/" or self.path == "/debug":
            # "/" is the polished demo screen shown on the Pi's HDMI display;
            # "/debug" is the full control panel for a developer's laptop.
            body = (DEBUG_PAGE if self.path == "/debug" else DEMO_PAGE).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
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
                    label, score, is_recycle, reason, engine, is_ewaste = _run_classifier(image)
                _record_result("MANUAL", label, score, is_recycle, reason, engine, is_ewaste)
            else:
                label, score, is_recycle, reason, engine, is_ewaste = _classify_frame(frame, "MANUAL")
            data = json.dumps({
                "label": label,
                "score": score,
                "is_recycle": is_recycle,
                "is_ewaste": is_ewaste,
                "reason": reason,
                "engine": engine,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)
            _act_on_result(label, is_recycle, is_ewaste)
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
        elif self.path == "/clear":
            _clear_ewaste_alert()
            data = json.dumps({"ewaste_alert": ewaste_alert, "status": status_text}).encode()
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
                "ewaste_alert": ewaste_alert,
                "ewaste_item": ewaste_item,
                "sort_active": sort_active,
                "last_result_source": last_result_source,
                "last_result_label": last_result_label,
                "last_result_score": last_result_score,
                "last_result_is_recycle": last_result_is_recycle,
                "last_result_reason": last_result_reason,
                "last_result_engine": last_result_engine,
                "last_result_is_ewaste": last_result_is_ewaste,
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

def _launch_kiosk_browser():
    """Show the live preview full-screen on the Pi's own HDMI display."""
    sleep(2.0)  # give the HTTP server a moment to start listening
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")
    url = f"http://localhost:{PORT}/"
    try:
        # Chromium runs as a singleton: relaunching it while a kiosk window is
        # already open just focuses that window instead of reloading it, so
        # code changes never show up on screen. Kill any running instance
        # first so we always get a fresh page load.
        subprocess.run(["pkill", "-x", "chromium"], env=env)
        sleep(0.5)
        subprocess.Popen([
            "chromium", "--kiosk", "--noerrdialogs", "--disable-infobars",
            "--no-first-run", "--disable-session-crashed-bubble",
            "--disable-features=TranslateUI", "--password-store=basic",
            "--force-device-scale-factor=0.75",
            url,
        ], env=env)
        print(f"Launched kiosk browser on the HDMI display -> {url}")
    except FileNotFoundError:
        print("chromium not found; skipping kiosk browser launch on HDMI display")

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
        # Any sensor problem -- unplugged, miswired, wrong I2C address -- shows
        # up here (OSError from the bus, or ValueError "No I2C device at
        # address" when the probe finds nothing). Swallow them all so a wiring
        # issue just disables auto-trigger instead of crashing the whole app;
        # everything downstream already handles tof_sensor being None.
        except Exception as exc:
            tof_sensor = None
            print(f"VL53L1X unavailable, auto-trigger disabled (sensor ignored): {exc}")
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
    Thread(target=_led_blinker, daemon=True).start()
    Thread(target=_launch_kiosk_browser, daemon=True).start()
    button_cb = pi.callback(BUTTON_PIN, pigpio.FALLING_EDGE, _on_button_edge)
    print("Manual-sort button ready on GPIO23 (physical pin 16)")
    print(f"\n LIVE! open this on your Mac browser: http://10.103.210.108:{PORT}\n")
    try:
        StreamingServer(("", PORT), Handler).serve_forever()
    finally:
        button_cb.cancel()
        pi.write(LED_PIN, 0)  # make sure the warning LED is off on exit
        if tof_sensor is not None:
            tof_sensor.stop_ranging()
        picam2.stop_recording()
