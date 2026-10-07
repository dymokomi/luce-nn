#!/usr/bin/env python3
"""Write build/models/<model>.tensors: a photo prepared as Depth Anything expects it
(the long side 518, both sides multiples of 14, ImageNet normalization) and
onnxruntime's depth for it, for tests/model_check.lucb. The model and the photo are
the caller's (not in the repository):

    build/venv/bin/python tests/make_model_fixture.py MODEL.onnx PHOTO.jpg
"""
from pathlib import Path
import sys, time
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tensors_file import write_tensors

model, photo = Path(sys.argv[1]), Path(sys.argv[2])
image = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
w, h = image.size
scale = 518 / max(w, h)
size = (max(14, round(w * scale / 14) * 14), max(14, round(h * scale / 14) * 14))
rgb = np.asarray(image.resize(size, Image.BICUBIC), np.float32) / 255
x = ((rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
name = session.get_inputs()[0].name
session.run(None, {name: x})
started = time.perf_counter()
depth = session.run(None, {name: x})[0]
print(f"onnxruntime: {(time.perf_counter() - started) * 1000:.0f} ms for {size[0]}x{size[1]}")
out = Path(__file__).resolve().parents[1] / "build/models"
out.mkdir(parents=True, exist_ok=True)
write_tensors(out / f"{model.stem}.tensors", {name: x}, {session.get_outputs()[0].name: depth})
print(out / f"{model.stem}.tensors")
