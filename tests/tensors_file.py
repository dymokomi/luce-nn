"""The .tensors format the tests share: little-endian "LNNT", a u32 count, then per
tensor a u32 name length and the name, a u32 role (0 input, 1 expected output), a u32
element type (1 float32, 7 int64, 9 bool), a u32 rank, the dims as i64, and the data
(bool as one byte each)."""
import struct
import numpy as np


def write_tensors(path, inputs, outputs):
    data = bytearray(b"LNNT")
    data += struct.pack("<I", len(inputs) + len(outputs))
    for role, group in ((0, inputs), (1, outputs)):
        for name, array in group.items():
            array = np.ascontiguousarray(array)
            kind = {np.float32: 1, np.int64: 7, np.bool_: 9}[array.dtype.type]
            encoded = name.encode()
            data += struct.pack("<I", len(encoded)) + encoded
            data += struct.pack("<III", role, kind, array.ndim)
            data += struct.pack(f"<{array.ndim}q", *array.shape)
            data += (array.astype(np.uint8) if kind == 9 else array).tobytes()
    path.write_bytes(bytes(data))
