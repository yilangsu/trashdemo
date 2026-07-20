"""Export the local model to ONNX for faster CPU inference.

Usage:
    python3 export_onnx.py

This writes model/model.onnx next to the existing Hugging Face weights.
"""

import os

try:
    import torch
    from transformers import AutoModelForImageClassification
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "torch and transformers are required to export ONNX. "
        "Install them with: pip install torch transformers"
    ) from exc


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")


def main():
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
    model.eval()

    sample = torch.randn(1, 3, 224, 224)
    output_names = ["logits"]

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        ONNX_PATH,
        input_names=["pixel_values"],
        output_names=output_names,
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )

    print(f"Wrote {ONNX_PATH}")


if __name__ == "__main__":
    main()