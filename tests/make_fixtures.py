#!/usr/bin/env python3
"""Write tests/fixtures: one small ONNX model per operator case, its inputs, and
onnxruntime's outputs for them, for tests/ops to hold luce-nn to.

    python3 -m venv build/venv && build/venv/bin/pip install onnx onnxruntime numpy
    build/venv/bin/python tests/make_fixtures.py

Each case is <name>.onnx and <name>.tensors (the inputs, then the expected outputs).
tensors_file.py describes the .tensors format. cases.txt lists the case names.
"""
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tensors_file import write_tensors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests/fixtures"
rng = np.random.default_rng(2026)
cases = []

FLOAT = TensorProto.FLOAT
INT64 = TensorProto.INT64


def values(*shape, low=-2.0, high=2.0):
    return rng.uniform(low, high, shape).astype(np.float32)


def case(name, nodes, inputs, initializers=None, outputs=("y",), opset=14):
    """A graph of `nodes` over float32 `inputs` (name -> array) and constant
    `initializers` (name -> array); outputs typed by running it."""
    initializers = initializers or {}
    graph_inputs = [helper.make_tensor_value_info(k, FLOAT, v.shape) for k, v in inputs.items()]
    graph_outputs = [helper.make_tensor_value_info(o, FLOAT, None) for o in outputs]
    graph = helper.make_graph(nodes, name, graph_inputs, graph_outputs,
                              [numpy_helper.from_array(v, k) for k, v in initializers.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    results = session.run(list(outputs), inputs)
    (OUT / f"{name}.onnx").write_bytes(model.SerializeToString())
    write_tensors(OUT / f"{name}.tensors", inputs, dict(zip(outputs, results)))
    cases.append(name)


def node(op, ins, outs=("y",), **attrs):
    return helper.make_node(op, list(ins), list(outs), **attrs)


OUT.mkdir(parents=True, exist_ok=True)

# Arithmetic with broadcasting.
for op in ["Add", "Sub", "Mul", "Div"]:
    case(f"{op.lower()}_same", [node(op, ["a", "b"])], {"a": values(2, 3, 5), "b": values(2, 3, 5, low=0.5)})
    case(f"{op.lower()}_bias", [node(op, ["a", "b"])], {"a": values(4, 7), "b": values(7, low=0.5)})
    case(f"{op.lower()}_scalar_left", [node(op, ["a", "b"])], {"a": values(1, low=0.5), "b": values(3, 4, low=0.5)})
    case(f"{op.lower()}_both_ways", [node(op, ["a", "b"])], {"a": values(2, 1, 4, low=0.5), "b": values(3, 1, low=0.5)})
case("pow_square", [node("Pow", ["a", "two"])], {"a": values(3, 5)}, {"two": np.array(2.0, np.float32)})
case("pow_half", [node("Pow", ["a", "e"])], {"a": values(3, 5, low=0.1)}, {"e": np.array(0.5, np.float32)})
for op in ["Sqrt", "Relu", "Erf", "Sigmoid", "Tanh", "Exp", "Neg"]:
    low = 0.01 if op == "Sqrt" else -3.0
    case(f"{op.lower()}", [node(op, ["a"])], {"a": values(4, 9, low=low, high=3.0)})
case("equal_where", [node("Equal", ["a", "k"], ["m"]), node("Where", ["m", "b", "a"])],
     {"a": np.array([[1, 2, 3], [4, 2, 6]], np.float32), "b": values(2, 3)}, {"k": np.array(2.0, np.float32)})

# Matrix products.
case("matmul_2d", [node("MatMul", ["a", "b"])], {"a": values(37, 53), "b": values(53, 29)})
case("matmul_large", [node("MatMul", ["a", "b"])], {"a": values(130, 384), "b": values(384, 300)})
case("matmul_batched", [node("MatMul", ["a", "b"])], {"a": values(2, 3, 9, 16), "b": values(2, 3, 16, 7)})
case("matmul_weights", [node("MatMul", ["a", "w"])], {"a": values(2, 11, 24)}, {"w": values(24, 10)})
case("matmul_broadcast", [node("MatMul", ["a", "b"])], {"a": values(4, 1, 5, 6), "b": values(3, 6, 2)})
case("matmul_vector", [node("MatMul", ["a", "b"])], {"a": values(6), "b": values(2, 6, 3)})

# Reductions.
case("reduce_last", [node("ReduceMean", ["a"], axes=[-1])], {"a": values(3, 4, 17)})
case("reduce_middle", [node("ReduceMean", ["a"], axes=[1], keepdims=0)], {"a": values(3, 4, 5)})
case("reduce_two", [node("ReduceMean", ["a"], axes=[0, 2])], {"a": values(3, 4, 5)})
case("reduce_input_axes", [node("ReduceMean", ["a", "axes"])], {"a": values(2, 6, 3)},
     {"axes": np.array([-1], np.int64)}, opset=18)
case("softmax_last", [node("Softmax", ["a"], axis=-1)], {"a": values(2, 3, 11, high=6.0)})
case("softmax_middle", [node("Softmax", ["a"], axis=1)], {"a": values(2, 5, 3)})

# Layout, and the int64 shape arithmetic exporters write.
case("shape_reshape", [node("Shape", ["a"], ["s"]), node("Gather", ["s", "i"], ["d"], axis=0),
                       node("Mul", ["d", "two"], ["d2"]), node("Unsqueeze", ["d2", "ax"], ["d3"]),
                       node("Concat", ["d3", "minus"], ["t"], axis=0), node("Reshape", ["a", "t"])],
     {"a": values(4, 3, 6)}, {"i": np.array(0, np.int64), "two": np.array(2, np.int64), "ax": np.array([0], np.int64),
                             "minus": np.array([-1], np.int64)})
case("reshape_zero", [node("Reshape", ["a", "t"])], {"a": values(2, 3, 4)}, {"t": np.array([0, -1], np.int64)})
case("transpose_heads", [node("Transpose", ["a"], perm=[0, 2, 1, 3])], {"a": values(2, 5, 3, 4)})
case("transpose_default", [node("Transpose", ["a"])], {"a": values(2, 3, 4)})
case("concat_middle", [node("Concat", ["a", "b", "c"], axis=1)], {"a": values(2, 1, 3), "b": values(2, 4, 3), "c": values(2, 2, 3)})
case("concat_last", [node("Concat", ["a", "b"], axis=-1)], {"a": values(3, 2), "b": values(3, 5)})
case("gather_rows", [node("Gather", ["a", "i"], axis=0)], {"a": values(6, 4)}, {"i": np.array([5, 0, -1, 2], np.int64)})
case("gather_scalar_middle", [node("Gather", ["a", "i"], axis=1)], {"a": values(2, 5, 3)}, {"i": np.array(3, np.int64)})
case("slice_steps", [node("Slice", ["a", "st", "en", "ax", "sp"])], {"a": values(5, 9, 4)},
     {"st": np.array([1, -1], np.int64), "en": np.array([9, -10], np.int64), "ax": np.array([1, 2], np.int64),
      "sp": np.array([2, -1], np.int64)})
case("slice_token", [node("Slice", ["a", "st", "en", "ax"])], {"a": values(1, 7, 6)},
     {"st": np.array([1], np.int64), "en": np.array([9223372036854775807], np.int64), "ax": np.array([1], np.int64)})
case("unsqueeze_squeeze", [node("Unsqueeze", ["a", "ax"], ["u"]), node("Squeeze", ["u", "sx"])],
     {"a": values(3, 4)}, {"ax": np.array([0, 3], np.int64), "sx": np.array([0], np.int64)})
case("expand", [node("Expand", ["a", "s"])], {"a": values(3, 1)}, {"s": np.array([2, 3, 4], np.int64)})

# Convolutions.
case("conv_3x3", [node("Conv", ["x", "w", "b"], pads=[1, 1, 1, 1])], {"x": values(1, 5, 9, 11)},
     {"w": values(4, 5, 3, 3), "b": values(4)})
case("conv_stride", [node("Conv", ["x", "w", "b"], pads=[1, 1, 1, 1], strides=[2, 2])], {"x": values(2, 3, 10, 9)},
     {"w": values(6, 3, 3, 3), "b": values(6)})
case("conv_1x1", [node("Conv", ["x", "w"])], {"x": values(1, 8, 6, 7)}, {"w": values(5, 8, 1, 1)})
case("conv_patches", [node("Conv", ["x", "w", "b"], kernel_shape=[14, 14], strides=[14, 14])], {"x": values(1, 3, 28, 42)},
     {"w": values(16, 3, 14, 14), "b": values(16)})
case("conv_group_dilated", [node("Conv", ["x", "w"], group=2, dilations=[2, 2], pads=[2, 2, 2, 2])], {"x": values(1, 4, 8, 8)},
     {"w": values(6, 2, 3, 3)})
case("conv_transpose_4", [node("ConvTranspose", ["x", "w", "b"], kernel_shape=[4, 4], strides=[4, 4])], {"x": values(1, 6, 3, 5)},
     {"w": values(6, 4, 4, 4), "b": values(4)})
case("conv_transpose_padded", [node("ConvTranspose", ["x", "w"], strides=[2, 2], pads=[1, 1, 1, 1], output_padding=[1, 1])],
     {"x": values(1, 3, 4, 5)}, {"w": values(3, 2, 3, 3)})

# Resizes, as the depth head and the position embedding use them, and others.
empty = np.array([], np.float32)
case("resize_linear_corners", [node("Resize", ["x", "roi", "scales"], mode="linear", coordinate_transformation_mode="align_corners")],
     {"x": values(1, 2, 5, 7)}, {"roi": empty, "scales": np.array([1, 1, 2, 2], np.float32)})
case("resize_linear_sizes", [node("Resize", ["x", "roi", "", "sizes"], mode="linear", coordinate_transformation_mode="align_corners")],
     {"x": values(1, 2, 5, 7)}, {"roi": empty, "sizes": np.array([1, 2, 13, 9], np.int64)})
case("resize_cubic", [node("Resize", ["x", "roi", "", "sizes"], mode="cubic", coordinate_transformation_mode="half_pixel")],
     {"x": values(1, 3, 37, 37)}, {"roi": empty, "sizes": np.array([1, 3, 25, 37], np.int64)})
case("resize_cubic_up", [node("Resize", ["x", "roi", "", "sizes"], mode="cubic", coordinate_transformation_mode="half_pixel")],
     {"x": values(1, 2, 6, 5)}, {"roi": empty, "sizes": np.array([1, 2, 15, 11], np.int64)})
case("resize_linear_half", [node("Resize", ["x", "roi", "scales"], mode="linear")],
     {"x": values(1, 1, 4, 6)}, {"roi": empty, "scales": np.array([1, 1, 2.5, 1.5], np.float32)})
case("resize_nearest", [node("Resize", ["x", "roi", "scales"], mode="nearest", coordinate_transformation_mode="asymmetric",
                             nearest_mode="floor")],
     {"x": values(1, 2, 3, 4)}, {"roi": empty, "scales": np.array([1, 1, 2, 3], np.float32)})

(OUT / "cases.txt").write_text("\n".join(cases) + "\n")
print(f"{len(cases)} cases (onnx {onnx.__version__}, onnxruntime {ort.__version__})")
