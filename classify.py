"""
Shared AI logic. No hardware here so it runs on your laptop OR the Pi.
Default model: WasteNet from KrisnaSantosa15/wastenet-garbage-classifier.
Test it today: put a photo of trash/a bottle next to this file and run:
    pip install pillow numpy requests
    python classify.py test.jpg
"""
import io
import json
import os
from urllib.request import urlretrieve

from PIL import Image

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional dependency
    tf = None

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment
    torch = None

try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except ImportError:  # pragma: no cover - depends on local environment
    try:
        from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
    except ImportError:
        TFLiteInterpreter = None

try:
    from transformers import AutoModelForImageClassification
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForImageClassification = None

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional dependency
    ort = None

# which classes count as recycling; anything else -> trash
RECYCLE = {
    "brown-glass",
    "cardboard",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "white-glass",
}

# e-waste: batteries and electronics must NEVER go in curbside recycling or the
# regular trash -- they get flagged for separate hazardous-waste disposal and
# trip the warning LED. These are the on-device WasteNet labels that count as
# e-waste; the Gemini path flags e-waste directly via its is_ewaste field.
EWASTE = {"battery"}

# if the model isn't at least this confident, call it TRASH.
# (correct guesses come back ~95%+, bad guesses come back low -> this catches them)
CONFIDENCE = 0.60

# maps WasteNet's 12 classes down to the ONNX model's 6, so the two can be
# compared for agreement.
_WASTENET_TO_COARSE = {
    "brown-glass": "glass",
    "green-glass": "glass",
    "white-glass": "glass",
    "battery": "trash",
    "biological": "trash",
    "clothes": "trash",
    "shoes": "trash",
}

# WasteNet model and labels from the upstream GitHub repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL = os.path.join(_HERE, "model")
WASTENET_BASE_URL = "https://raw.githubusercontent.com/KrisnaSantosa15/wastenet-garbage-classifier/main"
WASTENET_TFLITE_URL = f"{WASTENET_BASE_URL}/tflite/model.tflite"
WASTENET_LABELS_URL = f"{WASTENET_BASE_URL}/tflite/labels.txt"
WASTENET_TFLITE_MODEL = os.path.join(_LOCAL, "wastenet_model.tflite")
WASTENET_LABELS_PATH = os.path.join(_LOCAL, "wastenet_labels.txt")

MODEL = _LOCAL if os.path.isdir(_LOCAL) else "yangy50/garbage-classification"
ONNX_MODEL = os.path.join(_LOCAL, "model.onnx")
CONFIG_PATH = os.path.join(_LOCAL, "config.json")
PREPROCESSOR_PATH = os.path.join(_LOCAL, "preprocessor_config.json")

_processor = None
_model = None
_session = None
_tflite_interpreter = None
_tflite_input_details = None
_tflite_output_details = None
_wastenet_labels = None
_id2label = None
_image_size = 224
_image_mean = (0.5, 0.5, 0.5)
_image_std = (0.5, 0.5, 0.5)


def _download_file(url, path):
    if os.path.isfile(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {url} -> {path}...")
    urlretrieve(url, path)


def _load_wastenet_labels():
    global _wastenet_labels
    if _wastenet_labels is not None:
        return _wastenet_labels

    if not os.path.isfile(WASTENET_LABELS_PATH):
        _download_file(WASTENET_LABELS_URL, WASTENET_LABELS_PATH)

    with open(WASTENET_LABELS_PATH, "r", encoding="utf-8") as handle:
        _wastenet_labels = [line.strip() for line in handle if line.strip()]
    return _wastenet_labels


def _preprocess_wastenet_image(image):
    resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    if np is None:
        raise RuntimeError("numpy is required for WasteNet inference")
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def _get_wastenet_interpreter():
    global _tflite_interpreter, _tflite_input_details, _tflite_output_details
    if _tflite_interpreter is not None:
        return _tflite_interpreter

    interpreter_cls = TFLiteInterpreter
    if interpreter_cls is None and tf is not None:
        interpreter_cls = tf.lite.Interpreter

    if interpreter_cls is None:
        return None

    _download_file(WASTENET_TFLITE_URL, WASTENET_TFLITE_MODEL)
    _download_file(WASTENET_LABELS_URL, WASTENET_LABELS_PATH)

    print("Loading WasteNet TFLite model from local disk...")
    _tflite_interpreter = interpreter_cls(model_path=WASTENET_TFLITE_MODEL)
    _tflite_interpreter.allocate_tensors()
    _tflite_input_details = _tflite_interpreter.get_input_details()[0]
    _tflite_output_details = _tflite_interpreter.get_output_details()[0]
    return _tflite_interpreter


def _load_preprocessor_config():
    global _image_size, _image_mean, _image_std
    if not os.path.isfile(PREPROCESSOR_PATH):
        return

    with open(PREPROCESSOR_PATH, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    size = config.get("size", 224)
    if isinstance(size, dict):
        size = size.get("shortest_edge", 224)
    _image_size = int(size)
    _image_mean = tuple(config.get("image_mean", [0.5, 0.5, 0.5]))
    _image_std = tuple(config.get("image_std", [0.5, 0.5, 0.5]))


def _load_id2label():
    global _id2label
    if _id2label is not None:
        return _id2label

    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        _id2label = {int(key): value for key, value in config["id2label"].items()}
    return _id2label


def _preprocess_image(image):
    resized = image.convert("RGB").resize((_image_size, _image_size), Image.Resampling.BILINEAR)
    pixels = list(resized.getdata())
    values = [component / 255.0 for pixel in pixels for component in pixel]
    values = [
        (value - _image_mean[index % 3]) / _image_std[index % 3]
        for index, value in enumerate(values)
    ]

    if torch is not None:
        tensor = torch.tensor(values, dtype=torch.float32).reshape(1, _image_size, _image_size, 3)
        return tensor.permute(0, 3, 1, 2)

    if np is None:
        raise RuntimeError("numpy is required when torch is unavailable")
    return np.asarray(values, dtype=np.float32).reshape(1, _image_size, _image_size, 3).transpose(0, 3, 1, 2)


def _maybe_quantize(model):
    """Use int8 linear layers on CPU when available."""
    if torch is None:
        return model
    if os.getenv("TRASHDEMO_QUANTIZE", "1") == "0":
        return model
    if torch.cuda.is_available():
        return model

    try:
        return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as exc:  # pragma: no cover - quantization can vary by build
        print(f"Quantization unavailable, using float model: {exc}")
        return model


def _get_runtime():
    global _processor, _model, _session
    _load_preprocessor_config()
    if os.path.isfile(ONNX_MODEL) and ort is not None and np is not None:
        if _session is None:
            print("Loading ONNX model from local disk...")
            _session = ort.InferenceSession(
                ONNX_MODEL,
                providers=["CPUExecutionProvider"],
            )
            _load_id2label()
        return None, _session

    if _model is None:
        if AutoModelForImageClassification is None or torch is None:
            return None, None
        print("Loading model from local disk...")
        model = AutoModelForImageClassification.from_pretrained(MODEL)
        model.eval()
        _model = _maybe_quantize(model)
    return None, _model


def _classify_wastenet(image):
    """Returns (label, score) from the WasteNet TFLite model, or None if unavailable."""
    interpreter = _get_wastenet_interpreter()
    if interpreter is None:
        return None
    if np is None:
        raise RuntimeError("numpy is required for WasteNet inference")

    inputs = _preprocess_wastenet_image(image)
    input_details = _tflite_input_details
    output_details = _tflite_output_details
    input_data = np.asarray(inputs, dtype=input_details["dtype"])

    if np.issubdtype(input_details["dtype"], np.integer):
        scale, zero_point = input_details.get("quantization", (0.0, 0))
        if scale and scale > 0:
            input_data = input_data / scale + zero_point
        input_data = np.clip(np.rint(input_data), np.iinfo(input_details["dtype"]).min, np.iinfo(input_details["dtype"]).max).astype(input_details["dtype"])
    else:
        input_data = input_data.astype(np.float32)

    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    if output.min() < 0.0 or output.max() > 1.0 + 1e-3:
        probabilities = np.exp(output - np.max(output))
        probabilities = probabilities / probabilities.sum()
    else:
        probabilities = output

    class_id = int(np.argmax(probabilities))
    score = float(probabilities[class_id])
    label = _load_wastenet_labels()[class_id].lower()
    return label, score


def _classify_onnx_or_torch(image):
    """Returns (label, score) from the ONNX/torch ViT model, or None if unavailable."""
    _, runtime = _get_runtime()
    if runtime is None:
        return None

    inputs = _preprocess_image(image)

    if _session is not None:
        if np is None:
            raise RuntimeError("numpy is required for ONNX inference")
        logits = _session.run(None, {"pixel_values": np.asarray(inputs)})[0]
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities = probabilities / probabilities.sum(axis=-1, keepdims=True)
        class_id = int(probabilities.argmax(axis=-1)[0])
        score = float(probabilities[0, class_id])
        label_map = _load_id2label()
        if label_map is None:
            raise RuntimeError("model/config.json is required for ONNX label mapping")
        label = label_map[class_id].lower()
    else:
        with torch.inference_mode():
            outputs = runtime(pixel_values=inputs)
            probabilities = outputs.logits.softmax(dim=-1)[0]
            class_id = int(probabilities.argmax().item())
            score = float(probabilities[class_id].item())
        label = runtime.config.id2label[class_id].lower()
    return label, score


def classify_pil(image):
    """Classify a PIL image using both models as an ensemble. Returns (label, score, is_recycle)."""
    wastenet_result = _classify_wastenet(image)
    onnx_result = _classify_onnx_or_torch(image)

    if wastenet_result is None and onnx_result is None:
        raise RuntimeError("No classifier backend is available")

    if wastenet_result is None:
        label, score = onnx_result
        agree = True
    elif onnx_result is None:
        label, score = wastenet_result
        agree = True
    else:
        w_label, w_score = wastenet_result
        o_label, o_score = onnx_result
        agree = _WASTENET_TO_COARSE.get(w_label, w_label) == o_label
        label, score = (w_label, w_score) if w_score >= o_score else (o_label, o_score)
        print(f"ensemble: wastenet={w_label} ({w_score:.0%})  onnx={o_label} ({o_score:.0%})  agree={agree}")

    # only recycle if both models agree, it's a recyclable class, AND the
    # winning model is confident. Disagreement is treated as an edge case and
    # defaults to TRASH rather than risking contaminating recycling.
    is_recycle = agree and (label in RECYCLE) and (score >= CONFIDENCE)
    return label, score, is_recycle


def classify(image_path):
    """Returns (label, score, is_recycle)."""
    return classify_pil(Image.open(image_path))


# ---------------------------------------------------------------------------
# Cloud classifier: Google Gemini + Philadelphia-specific recycling rules.
# Optional path, enabled from the web UI toggle in live.py. The on-device
# WasteNet model stays the default; this only runs when explicitly turned on.
#   Requires:  pip install google-genai   and   GEMINI_API_KEY in the env.
# ---------------------------------------------------------------------------

def _load_dotenv():
    """Minimal .env loader (no extra dependency). Reads KEY=value lines from a
    .env file next to this script into the environment. A real shell env var
    (e.g. an exported GEMINI_API_KEY) always wins over the file."""
    env_path = os.path.join(_HERE, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# gemini-2.5-flash is broadly available and cheap. To use a different one (e.g. a
# Gemini 3 flash variant), list what your key supports and set GEMINI_MODEL in
# .env -- no code change needed. See the README / list-models command.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Location-specific rules straight from the City of Philadelphia's curbside
# single-stream program (phila.gov/programs/recycling-program/what-to-recycle).
# This is what makes the decision "Philadelphia-specific" rather than generic.
PHILLY_SYSTEM_PROMPT = """You are the sorting brain for an automated trash/recycling \
bin located in PHILADELPHIA, Pennsylvania. Follow the City of Philadelphia's official \
curbside single-stream recycling rules EXACTLY -- not generic or national guidance.

You are shown ONE photo of ONE object being dropped into the bin. Decide whether it \
belongs in RECYCLE or TRASH under Philadelphia's rules, then answer.

PHILADELPHIA ACCEPTS (RECYCLE) -- only if empty, rinsed, and dry:
- Paper: newspaper, magazines, catalogs, junk mail, envelopes, writing/scrap paper, \
paper bags, paper cups, phone books, paperback books, greeting cards, non-metallic \
wrapping paper.
- Cardboard (flattened, free of grease and food): corrugated shipping boxes, CLEAN \
pizza boxes, paper towel/toilet rolls, cardboard egg cartons, dry food boxes.
- Cartons: milk, juice, wine, and soup cartons.
- Glass: all bottles and jars (lids/caps on).
- Metal: aluminum/steel/tin cans, empty paint cans, empty aerosol cans, aluminum \
baking trays, jar lids and bottle caps on empty containers.
- Plastics #1, #2, and #5 ONLY: bottles, jugs, cups, tubs, food/beverage and takeout \
containers, detergent/shampoo bottles, pump/spray bottles (lids/caps on).

PHILADELPHIA DOES NOT ACCEPT (TRASH):
- Plastic bags and any plastic film or wrap.
- Plastics #3, #4, #6, #7 -- including Styrofoam/polystyrene foam and packing peanuts.
- Food-soiled or greasy paper/cardboard (greasy pizza boxes, used paper plates, \
napkins, tissues, paper towels).
- Needles and syringes.
- Clothing, hangers, textiles, shoes.
- Pots, pans, ceramics, drinking glasses, mirrors, wood.
- Food, liquid, or any organic waste.

E-WASTE -- SPECIAL HANDLING (set is_ewaste=true):
- Batteries of ANY kind (AA/AAA/9V, button/coin cells, lithium, car, power banks), \
phones, tablets, laptops, chargers, power adapters, power cords, cables, \
headphones/earbuds, light bulbs, and any small electronic device or ANYTHING that \
contains a battery, plug, or circuit board.
- E-waste is NEVER curbside-recyclable and is NOT regular trash. If the object is \
e-waste, set is_ewaste=true AND is_recycle=false. It must go to a hazardous-waste / \
e-waste drop-off. Say so briefly in the reason (e.g. "Battery -- e-waste, take to \
drop-off, never curbside").

DECISION RULES:
- Call it RECYCLE only if it clearly matches an accepted Philadelphia category AND \
looks empty/clean enough not to contaminate the batch.
- Contaminated recyclables (food residue, liquid still inside, grease) are TRASH.
- When genuinely unsure, choose TRASH -- one dirty item can spoil a whole batch \
("when in doubt, throw it out").
- If you cannot identify the object, choose TRASH with low confidence.

Keep the reason short (max ~15 words), plain enough to read off a screen, and cite \
the Philly rule (e.g. "Empty #1 plastic bottle -- Philly takes #1/#2/#5")."""

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai  # lazy import: only needed when Gemini is enabled
        _gemini_client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
    return _gemini_client


def classify_pil_gemini(image):
    """Classify a PIL image with Gemini under Philadelphia rules.
    Returns (label, score, is_recycle, reason, is_ewaste)."""
    from google.genai import types
    from pydantic import BaseModel

    class Verdict(BaseModel):
        item: str
        is_recycle: bool
        is_ewaste: bool
        reason: str
        confidence: float

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)

    client = _get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg"),
            "Recycle or trash under Philadelphia curbside rules? Explain briefly.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=PHILLY_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=Verdict,
        ),
    )
    verdict = response.parsed
    is_ewaste = bool(verdict.is_ewaste)
    # e-waste is never recyclable -- enforce it here so a stray true can't leak
    # a battery into the recycling stream.
    is_recycle = bool(verdict.is_recycle) and not is_ewaste
    return verdict.item, float(verdict.confidence), is_recycle, verdict.reason, is_ewaste


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    label, score, is_recycle = classify(path)
    bucket = "RECYCLE" if is_recycle else "TRASH"
    print(f"Saw: {label} ({score:.0%})  ->  {bucket}")
