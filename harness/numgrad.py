"""harness/numgrad.py — finite-difference check of the C++ engine's gradients.

Independent of PyTorch: each case's forward is re-implemented in plain NumPy
(float64), and the loss  L = sum(forward(inputs) * upstream)  is differentiated
with central differences. The result is compared against the C++ engine's
grad_<i>_candidate.txt dumps.

    python harness/numgrad.py --op add                      # check the C++ grads
    python harness/numgrad.py --op add --against reference  # self-test on PyTorch grads
"""

import argparse
import os
import sys

import numpy as np

from format_common import load_tensor, EXT, CANDIDATE_SUFFIX


def _softmax_cross_entropy(logits, labels):
    labels = labels.astype(np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -log_probs[np.arange(len(labels)), labels].mean()


# NumPy forwards; must match the torch forwards in ref_torch.py.
FORWARDS = {
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    "sum": lambda a: a.sum(),
    "matmul": lambda a, b: a @ b,
    "relu": lambda a: np.maximum(a, 0.0),
    "softmax_cross_entropy": _softmax_cross_entropy,
    "broadcast_add": lambda a, b: a + b,
    "broadcast_size1": lambda a, b: a + b,
    "broadcast_scalar": lambda a, b: a + b,
    "grad_through_broadcast": lambda a, b: a * b,
}


def finite_diff(loss, inputs, i, eps=1e-6):
    """Central-difference gradient of loss(*inputs) w.r.t. inputs[i]."""
    x = inputs[i]
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = loss(*inputs)
        x[idx] = orig - eps
        f_minus = loss(*inputs)
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def run_case(name, dump_dir, against="candidate", rtol=1e-3, atol=1e-5, eps=1e-6):
    print(f"Finite-difference check -- {name} (against {against} grads)")
    base = os.path.join(dump_dir, name)
    if not os.path.isdir(base):
        print(f"  No dumps found at {base} (run ref_torch.py first)")
        return False

    n_inputs = 0
    while os.path.isfile(os.path.join(base, f"input_{n_inputs}{EXT}")):
        n_inputs += 1
    inputs = [load_tensor(os.path.join(base, f"input_{i}{EXT}")) for i in range(n_inputs)]
    upstream = load_tensor(os.path.join(base, f"upstream{EXT}"))
    fwd = FORWARDS[name]

    def loss(*xs):
        return float(np.sum(fwd(*xs) * upstream))

    ok = True
    checked = 0
    for i in range(n_inputs):
        # Only differentiable inputs have a reference grad (e.g. not CE labels).
        if not os.path.isfile(os.path.join(base, f"grad_{i}{EXT}")):
            continue
        suffix = CANDIDATE_SUFFIX if against == "candidate" else EXT
        grad_path = os.path.join(base, f"grad_{i}{suffix}")
        if not os.path.isfile(grad_path):
            print(f"  grad_{i}: missing {os.path.basename(grad_path)} -> FAIL")
            ok = False
            continue

        analytic = load_tensor(grad_path)
        numeric = finite_diff(loss, inputs, i, eps=eps)
        checked += 1

        if analytic.shape != numeric.shape:
            print(f"  grad_{i}: shape mismatch analytic={analytic.shape} numeric={numeric.shape} -> FAIL")
            ok = False
            continue

        diff = np.abs(analytic - numeric)
        max_diff = float(np.nanmax(diff)) if diff.size else 0.0
        if np.allclose(analytic, numeric, rtol=rtol, atol=atol):
            print(f"  grad_{i}: PASS (max abs diff = {max_diff:.2e})")
        else:
            diff = np.where(np.isnan(diff), np.inf, diff)
            idx = tuple(int(j) for j in np.unravel_index(int(np.argmax(diff)), diff.shape))
            print(f"  grad_{i}: FAIL, max diff={float(diff[idx]):.3e} at {idx}, "
                  f"analytic={float(analytic[idx])!r} numeric={float(numeric[idx])!r}")
            ok = False

    if checked == 0:
        print(f"  No gradients checked for '{name}'")
        ok = False

    print(f"  Case '{name}': {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", "--case", dest="op", required=True,
                        help="Case name (as in ref_torch.py), or 'all'")
    parser.add_argument("--dump-dir", default="harness/data")
    parser.add_argument("--against", choices=["candidate", "reference"], default="candidate",
                        help="candidate = C++ engine grads (default); reference = PyTorch grads (self-test)")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    if args.op == "all":
        names = list(FORWARDS)
    elif args.op in FORWARDS:
        names = [args.op]
    else:
        print(f"Unknown op: {args.op}. Known: {', '.join(FORWARDS)}", file=sys.stderr)
        sys.exit(1)

    results = [run_case(n, args.dump_dir, args.against, args.rtol, args.atol, args.eps) for n in names]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
