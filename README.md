# luce-nn

Neural network inference for Luce Base. luce-nn loads ONNX models, the format PyTorch
and Hugging Face export, and runs their graphs on every processor. Results are checked
against onnxruntime.

```luce
from luce_nn import nn

var model = try nn.Model.open("depth_anything_v2_small.onnx")
defer model.close()
var dims: i64[4] = [1, 3, 518, 350]
try model.set_input("pixel_values", dims, pixels)       # float32, normalized
try model.run()
let depth = model.output("predicted_depth") else return  # const nn.Tensor*
```

## What it runs

A model is the ONNX file's graph, run node by node in the file's order. Weights are
loaded once. Each intermediate tensor is freed after the last node that reads it, so a
run holds little more than the weights. Tensors are float32, int64 or bool, row-major,
with up to eight dimensions.

The operators are the opset-13-and-later ones that vision transformers and
convolutional networks use:

- **Arithmetic:** Add, Sub, Mul, Div, Pow and Equal (numpy broadcasting), Where.
- **Unary:** Sqrt, Relu, Erf, Sigmoid, Tanh, Exp, Neg.
- **Products and reductions:** MatMul (batched, broadcast), ReduceMean, Softmax.
- **Shape and layout:** Shape, Reshape, Transpose, Concat, Gather, Slice, Unsqueeze,
  Squeeze, Expand, Identity, Constant.
- **Image operators:** Conv and ConvTranspose (2D, with groups, strides, pads and
  dilations), Resize (nearest, linear and cubic, with half_pixel, pytorch_half_pixel,
  align_corners and asymmetric coordinates).

A model that uses anything else fails to load, and the error names the gap. Weights
kept in external files are not read yet.

Depth Anything V2 Small runs in about 1.1 s on an M-series Mac, on CPU. That is
onnxruntime's whole-model result within 1e-4. Most of the time goes to the matrix
products: `gemm` uses a 4-row register-blocked kernel. `tests/bench/vectorizer.lucb`
measures the loops whose speed depends on Luce Base's SIMD vectorizer, which is being
improved for them. A GPU backend through luce-gpu compute has not been started.

## Tests

`luc test` runs the module's tests and `tests/ops`, which runs every case of
`tests/fixtures`. Each case is a small ONNX model whose expected output comes from
onnxruntime; there are 63 cases covering every operator and its broadcasting,
padding and resize variants. `tests/make_fixtures.py` regenerates them.

To check a whole model too, make its fixture from a photo with
`tests/make_model_fixture.py MODEL.onnx PHOTO.jpg`. That leaves the model and its
fixture in `build/models/`, where `tests/ops` runs them on every `luc test`. Models
are not kept in the repository.

## License

MIT or Apache-2.0, at your choice. Models carry their own licenses: Depth Anything V2
Small and Depth Anything 3 Small, Base and Mono-Large are Apache-2.0, while their
larger siblings are non-commercial.
