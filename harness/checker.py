import argparse
import sys
import os

import numpy as np

from format_common import load_tensor, EXT, CANDIDATE_SUFFIX

# Only these dumps must have a C++ candidate; inputs/upstream are consumed, not produced.
REQUIRED_PREFIXES = ("output", "grad")


def run_check(case, dump_dir, rtol=1e-4, atol=1e-5):
    print(f"Checking case: {case}")
    base = os.path.join(dump_dir, case)
    if not os.path.isdir(base):
        print(f"  No dumps found at {base}")
        return False

    ref_names = sorted(
        f for f in os.listdir(base)
        if f.endswith(EXT) and not f.endswith(CANDIDATE_SUFFIX)
    )

    ok = True
    compared = 0
    for ref_name in ref_names:
        stem = ref_name[:-len(EXT)]
        if not stem.startswith(REQUIRED_PREFIXES):
            continue
        can_path = os.path.join(base, stem + CANDIDATE_SUFFIX)
        if not os.path.isfile(can_path):
            print(f"  {stem}: missing candidate dump -> FAIL")
            ok = False
            continue

        ref = load_tensor(os.path.join(base, ref_name))
        can = load_tensor(can_path)
        compared += 1

        if ref.shape != can.shape:
            print(f"  {stem}: shape mismatch ref={ref.shape} can={can.shape} -> FAIL")
            ok = False
            continue

        # NaN anywhere is a failure, even if both sides agree on it.
        match = bool(np.allclose(can, ref, rtol=rtol, atol=atol, equal_nan=False))
        if match:
            print(f"  {stem}: PASS")
        else:
            diff = np.abs(ref - can)
            diff = np.where(np.isnan(diff), np.inf, diff)
            idx = tuple(int(j) for j in np.unravel_index(int(np.argmax(diff)), diff.shape))
            print(f"  {stem}: FAIL, max diff={float(diff[idx]):.3e} at {idx}, "
                  f"ref={float(ref[idx])!r} can={float(can[idx])!r}")
            ok = False

    if compared == 0:
        print(f"  No tensors compared for '{case}'")
        ok = False

    print(f"  Case '{case}': {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--dump-dir", default="harness/data")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    result = run_check(args.case, args.dump_dir, args.rtol, args.atol)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
