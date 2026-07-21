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
# False = on-device WasteNet (default, offline). True = Gemini + Philadelphia rules.
use_gemini = False
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
<label class="switch"><input type="checkbox" id="geminiToggle" onchange="setEngine()"> <span>🤖 Use Gemini 3 + Philadelphia rules</span></label>
<div id="engineNote" class="enginenote local">📟 ACTIVE: On-device ensemble (WasteNet + ONNX)</div>
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
var wantGemini=document.getElementById('geminiToggle').checked;
var tag = d.engine==='gemini' ? 'Gemini · Philly rules' : (wantGemini ? '⚠️ Gemini failed → on-device' : 'on-device ensemble');
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
function setEngine(){
var on=document.getElementById('geminiToggle').checked;
var note=document.getElementById('engineNote');
note.className='enginenote';
note.textContent = on ? 'switching to Gemini…' : 'switching to on-device model…';
fetch('/engine?use=' + (on?'gemini':'local')).then(x=>x.json()).then(d=>{
applyEngineBadge(d.engine, d.gemini_error);
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
var wantGemini=document.getElementById('geminiToggle').checked;
var tag = d.last_result_engine==='gemini' ? 'Gemini · Philly rules' : (wantGemini ? '⚠️ Gemini failed → on-device' : 'on-device ensemble');
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
background:radial-gradient(120% 120% at 50% -10%,#16263a 0%,#0b1622 45%,#05090f 100%);
color:#e8eef5;display:flex;flex-direction:column;height:100vh;padding:2.2vmin}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:0 0.6vmin 1.6vmin}
.brand{display:flex;align-items:center;gap:1.2vmin;font-weight:800;font-size:clamp(20px,3.2vmin,40px);letter-spacing:0.5px}
.brand .logo{font-size:clamp(24px,3.8vmin,46px)}
.brand small{display:block;font-weight:500;font-size:clamp(10px,1.4vmin,16px);color:#8aa0b6;letter-spacing:2px;text-transform:uppercase}
.live{display:flex;align-items:center;gap:1vmin;font-size:clamp(12px,1.7vmin,20px);font-weight:700;color:#9fb3c7;letter-spacing:1.5px}
.live .dot{width:1.4vmin;height:1.4vmin;min-width:9px;min-height:9px;border-radius:50%;background:#ef4444;box-shadow:0 0 0 0 rgba(239,68,68,.6);animation:pulse 1.6s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(239,68,68,.6)}70%{box-shadow:0 0 0 2.2vmin rgba(239,68,68,0)}100%{box-shadow:0 0 0 0 rgba(239,68,68,0)}}
.grid{flex:1;display:flex;gap:2.4vmin;min-height:0}
.camera{flex:1.35;min-width:0;position:relative;border-radius:2.2vmin;overflow:hidden;
background:#000;border:0.35vmin solid #223345;box-shadow:0 1.4vmin 4vmin rgba(0,0,0,.55)}
.camera img{width:100%;height:100%;object-fit:cover;display:block}
.camera .tag{position:absolute;top:1.4vmin;left:1.4vmin;background:rgba(5,9,15,.62);
backdrop-filter:blur(6px);padding:0.7vmin 1.6vmin;border-radius:999px;font-size:clamp(11px,1.5vmin,18px);font-weight:600;color:#cdd9e6}
.card{flex:1;min-width:0;border-radius:2.2vmin;padding:3vmin;display:flex;flex-direction:column;
justify-content:center;gap:1.8vmin;border:0.35vmin solid rgba(255,255,255,.06);
background:linear-gradient(160deg,rgba(255,255,255,.05),rgba(255,255,255,.015));
box-shadow:0 1.4vmin 4vmin rgba(0,0,0,.5);transition:background .35s,border-color .35s}
.kicker{font-size:clamp(13px,2vmin,24px);font-weight:800;letter-spacing:3px;text-transform:uppercase;color:#8aa0b6}
.verdict{font-size:clamp(34px,7.4vmin,96px);font-weight:900;line-height:1.02;letter-spacing:-1px}
.item{font-size:clamp(20px,3.6vmin,46px);font-weight:700;color:#f2f6fb;text-transform:capitalize}
.meter{height:2vmin;min-height:12px;border-radius:999px;background:rgba(255,255,255,.09);overflow:hidden;position:relative;display:none}
.meter-fill{height:100%;width:0;border-radius:999px;transition:width .5s ease,background .35s}
.meter-label{font-size:clamp(12px,1.7vmin,20px);font-weight:700;color:#aebccb;margin-top:-0.4vmin}
.reason{font-size:clamp(15px,2.3vmin,30px);line-height:1.35;color:#d5e0ec;font-style:italic}
.reason .by{display:block;font-style:normal;font-weight:700;font-size:clamp(11px,1.5vmin,18px);
letter-spacing:1.5px;text-transform:uppercase;color:#7f93a8;margin-top:1vmin}
/* verdict theming */
.card.idle .kicker{color:#8aa0b6}.card.idle .verdict{color:#c3d0dd}
.card.thinking{border-color:rgba(245,158,11,.4);background:linear-gradient(160deg,rgba(245,158,11,.14),rgba(245,158,11,.03))}
.card.thinking .kicker,.card.thinking .verdict{color:#fbbf24}
.card.recycle{border-color:rgba(34,197,94,.5);background:linear-gradient(160deg,rgba(34,197,94,.18),rgba(34,197,94,.03))}
.card.recycle .kicker,.card.recycle .verdict{color:#4ade80}
.card.recycle .meter-fill{background:#22c55e}
.card.trash{border-color:rgba(239,68,68,.5);background:linear-gradient(160deg,rgba(239,68,68,.16),rgba(239,68,68,.03))}
.card.trash .kicker,.card.trash .verdict{color:#f87171}
.card.trash .meter-fill{background:#ef4444}
.dots::after{content:'';animation:dots 1.4s steps(4,end) infinite}
@keyframes dots{0%{content:''}25%{content:'.'}50%{content:'..'}75%{content:'...'}}
/* full-screen e-waste takeover */
.ewaste{position:fixed;inset:0;z-index:50;display:none;flex-direction:column;align-items:center;justify-content:center;
text-align:center;padding:5vmin;background:#7f1010;color:#fff;animation:ewblink 0.7s steps(1,end) infinite}
.ewaste.show{display:flex}
@keyframes ewblink{50%{background:#3a0606}}
.ewaste .ic{font-size:clamp(60px,16vmin,220px);line-height:1}
.ewaste .t{font-size:clamp(34px,8vmin,120px);font-weight:900;letter-spacing:1px;margin-top:1vmin}
.ewaste .s{font-size:clamp(22px,4.6vmin,64px);font-weight:800;color:#ffd7d7;letter-spacing:4px;margin-top:0.4vmin}
.ewaste .it{font-size:clamp(18px,3.4vmin,44px);font-weight:700;margin-top:2.6vmin;text-transform:capitalize}
.ewaste .n{font-size:clamp(14px,2.3vmin,28px);color:#ffe1e1;margin-top:1.4vmin;max-width:24em}
@media (orientation:portrait){.grid{flex-direction:column}.camera{flex:1.1}}
</style></head>
<body>
<div class="topbar">
  <div class="brand"><span class="logo">♻️</span><div>Smart Waste Sorter<small>AI recycling assistant</small></div></div>
  <div class="live"><span class="dot"></span>LIVE</div>
</div>
<div class="grid">
  <div class="camera"><img src="/stream.mjpg" alt="live camera"><div class="tag">📷 Live camera</div></div>
  <div class="card idle" id="card">
    <div class="kicker" id="kicker">Ready</div>
    <div class="verdict" id="verdict">Place an item &amp; press the button</div>
    <div class="item" id="item"></div>
    <div class="meter" id="meter"><div class="meter-fill" id="meterFill"></div></div>
    <div class="meter-label" id="meterLabel"></div>
    <div class="reason" id="reason"></div>
  </div>
</div>
<div class="ewaste" id="ewaste">
  <div class="ic">⛔</div>
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
var meter=document.getElementById('meter');
var meterFill=document.getElementById('meterFill');
var meterLabel=document.getElementById('meterLabel');
var reason=document.getElementById('reason');
var ewaste=document.getElementById('ewaste');
var ewItem=document.getElementById('ewItem');
function setCard(cls){card.className='card '+cls;}
function refresh(){
fetch('/status').then(x=>x.json()).then(d=>{
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
    itemEl.textContent='';meter.style.display='none';meterLabel.textContent='';reason.textContent='';
    return;
  }
  // a result to show
  if(d.last_result_label){
    var rec=d.last_result_is_recycle;
    setCard(rec?'recycle':'trash');
    kicker.textContent = rec ? 'Recyclable' : 'Landfill';
    verdict.textContent = rec ? '♻️ RECYCLE' : '🗑️ TRASH';
    itemEl.textContent = d.last_result_label;
    var pct=Math.round((d.last_result_score||0)*100);
    meter.style.display='block';
    meterFill.style.width=pct+'%';
    meterLabel.textContent=pct+'% confident';
    var by = d.last_result_engine==='gemini' ? 'Gemini · Philadelphia rules' : 'On-device model';
    reason.innerHTML = (d.last_result_reason ? ('“'+d.last_result_reason+'”') : '') + '<span class="by">— '+by+'</span>';
    return;
  }
  // idle
  setCard('idle');
  kicker.textContent='Ready';
  verdict.textContent='Place an item & press the button';
  itemEl.textContent='';meter.style.display='none';meterLabel.textContent='';reason.textContent='';
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
