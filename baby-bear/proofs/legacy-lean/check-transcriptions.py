#!/usr/bin/env python3
"""check-transcriptions.py -- diff the hand-written Lean interface against the Rust.

`extraction/p3_baby_bear/*.lean` holds hand transcriptions of upstream bodies
that hax does not extract (see TCB.md layer 3, "Given real, faithful bodies").
Nothing in the build checks that those bodies say what the Rust says: a `lake
build` stays green for any body that is total and kernel-reducible, correct or
not.  That is not a hypothetical -- `p3_mds.util.first_row_to_first_col` shipped
through first review as the identity, discharging all six of its `of_isOk`
obligations while making the six `MATRIX_CIRC_MDS_*_COL` constants hold each
circulant matrix's first *row* where the Rust holds its first *column*.

This script closes that gap for every transcription that evaluates to data.  It
reads the Rust, computes what each constant must be, `#eval`s the Lean, and
diffs.  It is NOT run by CI and NOT run by `build-proofs.sh`; `SYNC.md` step 3
tells you to run it by hand after a re-extraction.

Two strengths of check, reported per row, because they are not worth the same:

  [rust]    the expected value is read straight out of the Rust source.  This
            detects drift: change the Rust and the check fails.
  [formula] the script re-implements a Rust formula (e.g. `to_monty`).  This
            detects Lean-side drift and catches a disagreement between two
            independent transcriptions -- but both could be wrong the same way,
            so it is strictly weaker than [rust].

Usage:  ./baby-bear/proofs/legacy-lean/check-transcriptions.py
Exit:   0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent            # .../legacy-lean
REPO_ROOT = PKG_DIR.parents[2]                       # .../Plonky3

MASK32 = (1 << 32) - 1


def read(rel: str) -> str:
    path = REPO_ROOT / rel
    if not path.is_file():
        sys.exit(f"missing Rust source: {path}")
    return path.read_text()


def grab(src: str, pattern: str, what: str) -> str:
    m = re.search(pattern, src, re.S)
    if not m:
        sys.exit(f"could not parse {what} -- the Rust moved; update this script")
    return m.group(1)


# --- parse the Rust -----------------------------------------------------------

baby_bear_rs = read("baby-bear/src/baby_bear.rs")
data_traits_rs = read("monty-31/src/data_traits.rs")
mds_rs = read("baby-bear/src/mds.rs")

PRIME = int(grab(baby_bear_rs, r"const PRIME:\s*u32\s*=\s*(0x[0-9A-Fa-f]+|\d+)", "PRIME"), 0)
MONTY_BITS = int(grab(baby_bear_rs, r"const MONTY_BITS:\s*u32\s*=\s*(0x[0-9A-Fa-f]+|\d+)", "MONTY_BITS"), 0)
MONTY_MU = int(grab(baby_bear_rs, r"const MONTY_MU:\s*u32\s*=\s*(0x[0-9A-Fa-f]+|\d+)", "MONTY_MU"), 0)
TWO_ADICITY = int(grab(baby_bear_rs, r"const TWO_ADICITY:\s*usize\s*=\s*(\d+)", "TWO_ADICITY"), 0)
BARRETT_N = int(grab(data_traits_rs, r"const N:\s*usize\s*=\s*(\d+)", "BarrettParameters::N"), 0)

# `first_row_to_first_col(&[...])` for each width, hex or decimal.
MDS_ROWS: dict[int, list[int]] = {}
for m in re.finditer(
    r"MATRIX_CIRC_MDS_(\d+)_COL:\s*\[i64;\s*\d+\]\s*=\s*first_row_to_first_col\(&\[(.*?)\]\s*\)\s*;",
    mds_rs,
    re.S,
):
    n = int(m.group(1))
    row = [int(x, 0) for x in re.findall(r"0x[0-9A-Fa-f]+|-?\d+", m.group(2))]
    if len(row) != n:
        sys.exit(f"MATRIX_CIRC_MDS_{n}_COL: parsed {len(row)} literals, expected {n}")
    MDS_ROWS[n] = row
if len(MDS_ROWS) != 6:
    sys.exit(f"expected 6 MDS tables, parsed {len(MDS_ROWS)} -- the Rust moved")


# --- the Rust formulas, re-implemented ----------------------------------------

def to_monty(x: int) -> int:
    """monty-31/src/utils.rs:7 -- (((x as u64) << MONTY_BITS) % PRIME as u64) as u32"""
    return (((x & MASK32) << MONTY_BITS) % PRIME) & MASK32


def first_row_to_first_col(row: list[int]) -> list[int]:
    """mds/src/util.rs:52 -- col[0] = row[0]; col[i] = row[N - i]"""
    return row[:1] + row[1:][::-1]


MONTY_MASK = ((1 << MONTY_BITS) - 1) & MASK32          # data_traits.rs:23
ODD_FACTOR = PRIME >> TWO_ADICITY                      # data_traits.rs:93
PRIME_I128 = PRIME                                     # data_traits.rs:54
PSEUDO_INV = (1 << (2 * BARRETT_N)) // PRIME_I128      # data_traits.rs:55
BARRETT_MASK = ~((1 << 10) - 1)                        # data_traits.rs:56

TO_MONTY_SAMPLES = [0, 1, 31, 12345, PRIME - 1, MASK32]


# --- the checks ---------------------------------------------------------------
# (tag, strength, lean expression yielding an Int or a list, expected value)

checks: list[tuple[str, str, str, object]] = [
    ("MontyParameters.PRIME", "rust",
     "(MontyParameters.PRIME BabyBearParameters).toNat", PRIME),
    ("MontyParameters.MONTY_BITS", "rust",
     "(MontyParameters.MONTY_BITS BabyBearParameters).toNat", MONTY_BITS),
    ("MontyParameters.MONTY_MU", "rust",
     "(MontyParameters.MONTY_MU BabyBearParameters).toNat", MONTY_MU),
    ("MontyParameters.MONTY_MASK", "formula",
     "(MontyParameters.MONTY_MASK BabyBearParameters).toNat", MONTY_MASK),
    ("TwoAdicData.TWO_ADICITY", "rust",
     "(TwoAdicData.TWO_ADICITY BabyBearParameters).toNat", TWO_ADICITY),
    ("TwoAdicData.ODD_FACTOR", "formula",
     "(TwoAdicData.ODD_FACTOR BabyBearParameters).toInt", ODD_FACTOR),
    ("BarrettParameters.N", "rust",
     "(BarrettParameters.N BabyBearParameters).toNat", BARRETT_N),
    ("BarrettParameters.PRIME_I128", "formula",
     "(BarrettParameters.PRIME_I128 BabyBearParameters).toInt", PRIME_I128),
    ("BarrettParameters.PSEUDO_INV", "formula",
     "(BarrettParameters.PSEUDO_INV BabyBearParameters).toInt", PSEUDO_INV),
    ("BarrettParameters.MASK", "formula",
     "(BarrettParameters.MASK BabyBearParameters).toInt", BARRETT_MASK),
]

for x in TO_MONTY_SAMPLES:
    checks.append((f"utils.to_monty {x}", "formula",
                   f"(utils.to_monty BabyBearParameters ({x} : u32)).toNat", to_monty(x)))
    checks.append((f"Impl.new {x}", "formula",
                   f"newValue ({x} : u32)", to_monty(x)))

for n in sorted(MDS_ROWS):
    checks.append((f"MATRIX_CIRC_MDS_{n}_COL", "rust",
                   f"colList (Impl.MATRIX_CIRC_MDS_{n}_COL_hoisted)",
                   first_row_to_first_col(MDS_ROWS[n])))


# --- emit, run, diff ----------------------------------------------------------

PRELUDE = """import p3_baby_bear
open p3_baby_bear.baby_bear p3_baby_bear.mds
open p3_monty_31.data_traits p3_monty_31.monty_31

/-- `Impl.new x` is monadic; project the Montgomery word, or -1 on fail/div. -/
private def newValue (x : u32) : Int :=
  match Impl.new BabyBearParameters x with
  | some (.ok v) => (v.value.toNat : Int)
  | _ => -1

private def colList {n : usize} (a : RustArray i64 n) : List Int :=
  a.toVec.toArray.toList.map (fun w => w.toInt)

private def emit (tag : String) (v : Int) : IO Unit :=
  IO.println (tag ++ "|" ++ toString v)

private def emitL (tag : String) (v : List Int) : IO Unit :=
  IO.println (tag ++ "|" ++ String.intercalate "," (v.map toString))

def probe : IO Unit := do
"""


def lean_source() -> str:
    body = []
    for i, (_tag, _strength, expr, expected) in enumerate(checks):
        fn = "emitL" if isinstance(expected, list) else "emit"
        body.append(f'  {fn} "c{i}" ({expr})')
    return PRELUDE + "\n".join(body) + "\n\n#eval probe\n"


def main() -> int:
    # Build first, ALWAYS.  `lake env lean` resolves `import p3_baby_bear` to
    # the compiled olean, not to the source: without this the probe silently
    # measures the last successful build and reports green against an edit it
    # never saw.  That is the exact failure this script exists to catch.
    build = subprocess.run(["lake", "build"], cwd=PKG_DIR, capture_output=True, text=True)
    if build.returncode != 0:
        sys.stderr.write(build.stdout + build.stderr)
        return sys.exit("`lake build` failed -- fix the build before checking transcriptions")

    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "Probe.lean"
        probe.write_text(lean_source())
        proc = subprocess.run(
            ["lake", "env", "lean", str(probe)],
            cwd=PKG_DIR, capture_output=True, text=True,
        )

    got: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "|" in line and line.startswith("c"):
            k, _, v = line.partition("|")
            got[k] = v.strip()

    if not got:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return sys.exit("no probe output -- does `lake build` succeed?")

    width = max(len(t) for t, _, _, _ in checks)
    failures = 0
    for i, (tag, strength, _expr, expected) in enumerate(checks):
        want = ",".join(str(x) for x in expected) if isinstance(expected, list) else str(expected)
        have = got.get(f"c{i}")
        ok = have == want
        failures += not ok
        print(f"  [{strength:>7}] {tag:<{width}}  {'ok' if ok else 'FAIL'}")
        if not ok:
            print(f"              expected: {want}")
            print(f"              from Lean: {have}")

    n_rust = sum(1 for _, s, _, _ in checks if s == "rust")
    print()
    print(f"{len(checks)} checks ({n_rust} against Rust data, "
          f"{len(checks) - n_rust} against re-implemented formulas), {failures} failed.")
    if failures:
        print("\nA failure means the Lean interface and the Rust disagree. Fix the")
        print("Lean body, not this script, unless the Rust genuinely moved.")
        return 1
    print("\nNot covered (no evaluable data): p3_field.dup.Dup.dup and")
    print("hax_ext AsRef.as_ref -- both remain faithful-by-inspection only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
