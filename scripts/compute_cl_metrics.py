"""Compute AA, F and LA (Eqs. (9)-(11) of the PIECED paper) from accuracy matrices.

Each CSV file holds one seed's N x N accuracy matrix without a header:
    row n    = task being evaluated (1..N)
    column m = training stage, i.e. after training up to task m (1..N)
    entry    = a_{n,m}, linear-evaluation accuracy (%) on the test samples of task n
Cells with m < n are not used and may be left empty.

Usage:
    python compute_cl_metrics.py seed0.csv seed1.csv seed2.csv seed3.csv seed4.csv

Because AA, F and LA are all computed from the same matrix, every seed satisfies
LA - AA = (N - 1) / N * F exactly; the script checks this as a sanity test.
"""
import csv
import statistics
import sys


def load_matrix(path):
    with open(path, newline="") as f:
        rows = [[float(v) if v.strip() else float("nan") for v in row] for row in csv.reader(f) if row]
    n = len(rows)
    if any(len(r) != n for r in rows):
        raise ValueError(f"{path}: expected a square N x N matrix, got row lengths {[len(r) for r in rows]}")
    return rows


def metrics(a):
    n = len(a)
    aa = sum(a[i][n - 1] for i in range(n)) / n                      # Eq. (9)
    f = sum(a[i][i] - a[i][n - 1] for i in range(n - 1)) / (n - 1)    # Eq. (10)
    la = sum(a[i][i] for i in range(n)) / n                           # Eq. (11)
    assert abs((la - aa) - (n - 1) / n * f) < 1e-9, "identity violated: check the matrix"
    return aa, f, la


def main(paths):
    per_seed = []
    for p in paths:
        aa, f, la = metrics(load_matrix(p))
        per_seed.append((aa, f, la))
        print(f"{p}: AA={aa:.2f}  F={f:.2f}  LA={la:.2f}")
    if len(per_seed) > 1:
        cols = list(zip(*per_seed))
        summary = [f"{name}={statistics.mean(c):.1f} ({statistics.stdev(c):.1f})"
                   for name, c in zip(("AA", "F", "LA"), cols)]
        print(f"mean (std) over {len(per_seed)} seeds: " + "  ".join(summary))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])