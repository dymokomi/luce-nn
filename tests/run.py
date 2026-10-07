#!/usr/bin/env python3
"""luce-nn's gate.

1. The module's own tests (the wire format, names, broadcasting, gemm, resize taps),
   native and C.
2. tests/op_check.lucb holds every operator case of tests/fixtures (made by
   make_fixtures.py from onnxruntime) to onnxruntime's outputs, native and C.
3. When build/models holds a model and its fixture (make_model_fixture.py), the whole
   model is run and held to onnxruntime too. Models are not in the repository.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--base", type=Path, default=Path(shutil.which("luce-base") or ROOT.parent / "luce-base/build/luce-base"))
parser.add_argument("--model", type=Path, help="an ONNX model whose build/models/<stem>.tensors fixture exists")
args = parser.parse_args()
base = args.base.resolve()
env = dict(os.environ, LUCE_BASE=str(base))
MODES = [["--native"], ["--backend=c"]]


def run(command):
    subprocess.run([str(part) for part in command], check=True, env=env, cwd=ROOT, timeout=1800)


for flags in MODES:
    run([base, "test", ROOT / "src/nn", *flags])
with tempfile.TemporaryDirectory() as tmp:
    for flags in MODES:
        tool = Path(tmp) / f"op_check{flags[0].replace('-', '_').replace('=', '_')}"
        run([base, "build", ROOT / "tests/op_check.lucb", *flags, "-o", tool])
        run([tool, ROOT / "tests/fixtures"])
        if args.model:
            fixture = ROOT / "build/models" / f"{args.model.stem}.tensors"
            run([tool, "--model", args.model.resolve(), fixture])
print("PASS luce-nn: unit tests and every operator case against onnxruntime, native and C"
      + (f", and {args.model.name} whole" if args.model else ""))
