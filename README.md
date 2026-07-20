# Trash vs Recycling Demo

AI decides if an object is **recycling** or **trash**, then a servo on a Raspberry Pi
moves to sort it. Model runs fully offline on the Pi.

## Files
- `classify.py` — the AI. Downloads and caches the WasteNet TFLite model into
  `./model/` on first run, then loads it from disk (no internet needed after that).
- `demo.py` — runs on the Pi: Pi Camera + servo.
- `model/` — cached model files. Includes an older ONNX model (`model.onnx`) that
  `classify.py` falls back to only if the WasteNet TFLite runtime isn't available.

## The model
[`KrisnaSantosa15/wastenet-garbage-classifier`](https://github.com/KrisnaSantosa15/wastenet-garbage-classifier)
— a TensorFlow Lite model with 12 classes: battery, biological, brown-glass,
cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass.
brown-glass/cardboard/green-glass/metal/paper/plastic/white-glass count as RECYCLE;
everything else (battery, biological, clothes, shoes, trash) is TRASH.

Requires the `ai-edge-litert` package (or `tflite-runtime`/`tensorflow`) for
inference — see install steps below.

**First run needs internet** to download `model.tflite` and `labels.txt` from
GitHub into `./model/`. Run `python3 classify.py test.jpg` once while online so
the files are cached before going offline for the actual demo.

## Faster inference (fallback path only)
The WasteNet TFLite model above is the primary, fast path and needs none of this.
The notes below only apply if you're on the older PyTorch/ONNX fallback (used when
no TFLite runtime is installed): `classify.py` caches the processor/model once and
uses CPU int8 dynamic quantization when available, which is faster than the generic
Hugging Face pipeline on a Raspberry Pi.

If you need to disable quantization for any reason, set:
```
TRASHDEMO_QUANTIZE=0
```

To measure latency before and after the ONNX path, run:
```
python3 bench_classify.py test.jpg --runs 20
```

To export the local model to ONNX for a larger CPU speedup, run:
```
python3 export_onnx.py
```
If `model/model.onnx` exists, `classify.py` will use it automatically.
Then rerun the benchmark to compare the numbers.

## The confidence threshold (important)
At the top of `classify.py`:
```
CONFIDENCE = 0.60
```
Correct guesses come back ~95% confident; bad guesses come back low. So if the model
is under 60% sure, we call it TRASH. This is your live tuning knob:
- Trash items sneaking through as recycle? Raise it (0.70, 0.75).
- Real recyclables wrongly called trash? Lower it (0.50).

---

## HOW TO MAKE PHOTOS THE MODEL ACTUALLY GETS RIGHT

WasteNet was trained on the Kaggle "Garbage Classification" dataset (15,000+ images,
12 classes). Its README doesn't document the photo style (background/lighting/framing),
so treat these as general best practices rather than dataset-matched advice — test your
actual props and trust `classify.py`'s output over assumptions:

- **Plain, uncluttered background.** A white or solid-color sheet/wall behind the
  object reduces noise. Avoid busy backgrounds.
- **ONE object per photo.** No clutter, no hands in frame, nothing else on the table.
- **Object centered and CLOSE** — it should fill most of the frame.
- **Even, bright lighting.** No harsh shadows. Indoor room light or a lamp is fine.
- **Consistent angle**, object flat/upright and clearly recognizable.

## Demo-day tips
- **Pre-test every prop tonight.** Run each real demo object through `classify.py` and
  keep the ones that classify correctly. This is standard practice, not cheating —
  you're matching your props to what the model handles well.
- Hold objects close to the camera, one at a time, over a white background.
- Do the `pip install` on the Pi BEFORE the demo so nothing downloads live.

## Running it
On your laptop (testing):
```
python3 classify.py "photo.jpg"
```

On the Pi (real demo):
```
sudo apt update && sudo apt install -y python3-picamera2
python3 -m venv --system-site-packages ~/trashdemo-env
source ~/trashdemo-env/bin/activate
pip install ai-edge-litert pillow numpy gpiozero
python3 classify.py test.jpg  # one-time: downloads & caches the WasteNet model
python demo.py
```
After the one-time `pip install` and first run, the Pi runs 100% offline.

## Servo wiring
| Servo wire | Connect to |
|---|---|
| Signal (orange/yellow) | Pi GPIO17 |
| Power (red) | External 5V (battery/power bank) — NOT the Pi's 5V |
| Ground (brown/black) | External power GND **and** Pi GND together |

The external ground and Pi ground MUST be connected together, or the servo just twitches.
Servo turning the wrong way? Swap `.max()` and `.min()` in `demo.py`.
