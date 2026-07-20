"""
Shared AI logic. No hardware here so it runs on your laptop OR the Pi.
Default model: WasteNet from KrisnaSantosa15/wastenet-garbage-classifier.
Test it today: put a photo of trash/a bottle next to this file and run:
    pip install pillow numpy requests
    python classify.py test.jpg
"""
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

# if the model isn't at least this confident, call it TRASH.
# (correct guesses come back ~95%+, bad guesses come back low -> this catches them)
CONFIDENCE = 0.60

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
            raise RuntimeError(
                "PyTorch and transformers are required for inference. "
                "Install them with: pip install torch transformers pillow"
            )
        print("Loading model from local disk...")
        model = AutoModelForImageClassification.from_pretrained(MODEL)
        model.eval()
        _model = _maybe_quantize(model)
    return None, _model


def classify_pil(image):
    """Classify a PIL image directly. Returns (label, score, is_recycle)."""
    interpreter = _get_wastenet_interpreter()
    if interpreter is not None:
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
    else:
        _, runtime = _get_runtime()
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

    # only recycle if it's a recyclable class AND the model is confident
    is_recycle = (label in RECYCLE) and (score >= CONFIDENCE)
    return label, score, is_recycle


def classify(image_path):
    """Returns (label, score, is_recycle)."""
    return classify_pil(Image.open(image_path))


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    label, score, is_recycle = classify(path)
    bucket = "RECYCLE" if is_recycle else "TRASH"
    print(f"Saw: {label} ({score:.0%})  ->  {bucket}")
