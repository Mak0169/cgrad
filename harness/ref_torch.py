"""harness/ref_torch.py — PyTorch float64 reference dumps.

For each case, writes into <output-dir>/<case>/:
    input_<i>.txt     forward inputs (the C++ engine must read these, not regenerate them)
    upstream.txt      gradient to seed backward with: C++ runs out.backward(upstream)
    output_0.txt      forward output
    grad_<i>.txt      gradient of input_<i>, only for differentiable inputs

A random (non-ones) upstream gradient is used so backward bugs can't hide behind
an all-ones seed from a plain .sum().
"""

import argparse
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F

from format_common import write_tensor, EXT

DT = torch.float64


def randn(*shape):
    # np.asarray: randn() with no shape returns a Python float, not a 0-d array.
    return np.asarray(np.random.randn(*shape), dtype=np.float64)


def away_from_zero(x, margin=1e-2):
    # Keep ReLU inputs off the kink so finite differences are well defined.
    return np.where(np.abs(x) < margin, np.sign(x + 1e-12) * margin, x)


# Each case returns (inputs, differentiable flags, torch forward taking tensors).
def case_add():
    return [randn(3, 4), randn(3, 4)], [True, True], lambda a, b: a + b


def case_mul():
    return [randn(3, 4), randn(3, 4)], [True, True], lambda a, b: a * b


def case_sum():
    return [randn(2, 3, 4)], [True], lambda a: a.sum()


def case_matmul():
    return [randn(2, 3), randn(3, 4)], [True, True], lambda a, b: a @ b


def case_relu():
    return [away_from_zero(randn(3, 4))], [True], F.relu


def case_softmax_cross_entropy():
    logits = randn(4, 3)
    labels = np.array([0, 1, 2, 0], dtype=np.int64)
    return [logits, labels], [True, False], lambda x, y: F.cross_entropy(x, y)


# Broadcasting: every case has a gradient that must be summed over a broadcast dim.
def case_broadcast_add():
    # rank difference: [3] + [2,3]
    return [randn(3), randn(2, 3)], [True, True], lambda a, b: a + b


def case_broadcast_size1():
    # size-1 dim: [2,1] + [2,3]
    return [randn(2, 1), randn(2, 3)], [True, True], lambda a, b: a + b


def case_broadcast_scalar():
    # scalar + [2,3]
    return [np.array(2.5), randn(2, 3)], [True, True], lambda a, b: a + b


def case_grad_through_broadcast():
    # both operands broadcast, with mul so the grads depend on the other operand:
    # [2,1,3] * [4,1] -> [2,4,3]
    return [randn(2, 1, 3), randn(4, 1)], [True, True], lambda a, b: a * b


CASES = {
    "add": case_add,
    "mul": case_mul,
    "sum": case_sum,
    "matmul": case_matmul,
    "relu": case_relu,
    "softmax_cross_entropy": case_softmax_cross_entropy,
    "broadcast_add": case_broadcast_add,
    "broadcast_size1": case_broadcast_size1,
    "broadcast_scalar": case_broadcast_scalar,
    "grad_through_broadcast": case_grad_through_broadcast,
}


def run_case(name, output_dir):
    inputs_np, diff, fn = CASES[name]()
    tensors = [
        torch.tensor(x, dtype=DT, requires_grad=True) if d else torch.tensor(x)
        for x, d in zip(inputs_np, diff)
    ]
    out = fn(*tensors)
    upstream = randn(*out.shape)
    out.backward(torch.tensor(upstream, dtype=DT))

    base = os.path.join(output_dir, name)
    os.makedirs(base, exist_ok=True)
    for f in os.listdir(base):
        if f.endswith(EXT) and "_candidate" not in f:
            os.remove(os.path.join(base, f))  # drop stale reference dumps

    for i, x in enumerate(inputs_np):
        write_tensor(x, os.path.join(base, f"input_{i}{EXT}"))
    write_tensor(upstream, os.path.join(base, f"upstream{EXT}"))
    write_tensor(out.detach().numpy(), os.path.join(base, f"output_0{EXT}"))
    for i, t in enumerate(tensors):
        if diff[i]:
            write_tensor(t.grad.numpy(), os.path.join(base, f"grad_{i}{EXT}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", help="Specific case to run, or 'all'")
    parser.add_argument("--output-dir", default="harness/data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.case == "all":
        names = list(CASES)
    elif args.case in CASES:
        names = [args.case]
    else:
        print(f"Unknown case: {args.case}. Known: {', '.join(CASES)}", file=sys.stderr)
        sys.exit(1)

    for name in names:
        np.random.seed(args.seed)  # per-case seed: regenerating one case doesn't shift the others
        run_case(name, args.output_dir)
    print(f"Wrote PyTorch reference dumps for {', '.join(names)} to {args.output_dir}")


if __name__ == "__main__":
    main()
