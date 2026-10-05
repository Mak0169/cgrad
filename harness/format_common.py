"""harness/format_common.py — shared loader / writer for the cgrad dump file format.

Text format:
    line 1     : shape as comma-separated ints (e.g. "2,3"), or the word "scalar"
                 for a 0-d tensor
    lines 2..N : one value per line, row-major

Reference dumps are named <stem>.txt; the C++ engine writes <stem>_candidate.txt
next to them in the same case directory.
"""

from pathlib import Path

import numpy as np


SCALAR = "scalar"
EXT = ".txt"
CANDIDATE_SUFFIX = "_candidate" + EXT


def load_tensor(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        raise ValueError(f"{path}: empty dump file")
    header, values = lines[0], lines[1:]

    if header == SCALAR:
        if len(values) != 1:
            raise ValueError(f"{path}: scalar dump must have exactly 1 value, got {len(values)}")
        return np.array(float(values[0]), dtype=np.float64)

    shape = tuple(int(x) for x in header.split(","))
    expected = int(np.prod(shape))
    if len(values) != expected:
        raise ValueError(f"{path}: shape {shape} needs {expected} values, got {len(values)}")
    return np.array([float(x) for x in values], dtype=np.float64).reshape(shape)


def _fmt(v):
    # Plain Python repr: NumPy 2 scalars repr as "np.float64(...)", which float()/strtod reject.
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    return repr(float(v))


def write_tensor(arr, path):
    arr = np.asarray(arr)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 0:
        path.write_text(SCALAR + "\n" + _fmt(arr.item()) + "\n")
        return
    lines = [",".join(str(d) for d in arr.shape)]
    lines.extend(_fmt(v) for v in np.ascontiguousarray(arr).ravel())
    path.write_text("\n".join(lines) + "\n")
