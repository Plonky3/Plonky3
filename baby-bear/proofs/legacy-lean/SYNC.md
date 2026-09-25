> **This document is AI-generated and agent-maintained.**

# Re-extraction runbook

Follow this when `baby-bear/src` (or a dependency's `src`) changes, or when the
hax / Lean pins move. It is self-contained: there is no repo-root orchestrator.

## Does an upstream change need re-extraction?

Re-extract if a change touched `baby-bear/src/**` or `baby-bear/Cargo.toml`.
Also re-extract if `monty-31/src`, `field/src`, `mds/src`, `poseidon1/src`,
`poseidon2/src` or `symmetric/src` changed a **signature the axioms mirror**
(anything in the layer-3 table of [`TCB.md`](TCB.md)) — a change to a dependency
*body* is invisible here, because bodies are assumed, not extracted.

Detection is deliberately conservative; re-extraction is idempotent, so running
it unnecessarily is harmless.

## Step 1 — re-extract

```bash
./baby-bear/proofs/legacy-lean/build-proofs.sh
```

Prerequisites: `cargo-hax` on `PATH`, a matching `hax-engine` (the driver aborts
on a version mismatch — override with `HAX_ENGINE_BINARY`), the pinned nightly
(`rustup toolchain install nightly-2025-11-08`) with the extraction target
(`rustup target add thumbv7em-none-eabi --toolchain nightly-2025-11-08`), and
`elan`/`lake`.

**`cargo hax` may print errors and still write a usable file.** Do not abort on
hax's exit status as long as `extraction/p3_baby_bear.lean` was overwritten. Judge the
result from `lake build`.

If the script ends green, go to step 3.

## Step 2 — reconcile drift

Only `extraction/p3_baby_bear.lean` is machine-owned; everything else —
`extraction/p3_baby_bear/`, `spec/`, the patches — is hand-written, and hax
never touches it.

1. Read any `.rej` files and the `lake build` errors.
2. Hand-edit `extraction/p3_baby_bear.lean` until `lake build` is green, following the
   conventions: mark every edit `-- PATCHED`, and when replacing a tactic keep
   the original on the next line as a comment (`-- (by rfl)`).
3. **Preserve the previous patch's intent, but do not preserve it needlessly.**
   If a hunk is no longer required — hax improved, or an obligation now
   discharges by `rfl` — *drop it*. A shrinking patch is the goal.
4. Prefer fixing an axiom over adding a `sorry`. Filling in a stub member
   shrinks the TCB; a new `sorry` grows it. If a `by rfl` now fails where it
   used to pass, understand why before admitting it.
5. Patches are hand-owned, not regenerated. Fix the individual patch that broke
   — its `.rej` shows exactly which context moved — and keep its header's
   `Cost:` and `Drop when:` fields honest. To add a genuinely new patch:
   ```bash
   ./baby-bear/proofs/legacy-lean/patches/new-patch.sh post-extraction 070-slug
   ./baby-bear/proofs/legacy-lean/patches/check-patches.sh
   ```
   `new-patch.sh` captures only the delta beyond the patches that already apply,
   so patches stay one-change-each. Note `build-proofs.sh` applies with **zero
   fuzz**, so drift fails loudly rather than landing somewhere plausible.

### Keeping the admitted set minimal

**Prefer fixing the interface over patching the proof.** hax's
`RustM.of_isOk _ (by rfl)` obligations fail only when the value flows through an
admitted stub. Giving that stub a real body discharges the obligation and
removes a patch hunk — this is how the generated file got to zero `sorry`.

If a `by rfl` starts failing after a re-extraction, in order:

1. Find which stub gates it (walk back to the enclosing `RustM.of_isOk` and see
   which interface symbol the block calls).
2. Give that stub a faithful body from upstream. (There is no longer an
   automated check for stub fidelity — `SanityCheck.lean` was removed — so read
   the Rust and the Lean side by side.)
3. Only if the value genuinely cannot be computed — e.g. it routes through a
   loop, like `SAMPLING_BITS_M` — consider `native_decide`, and record the
   `Lean.ofReduceBool` cost in `TCB.md`.
4. **Never admit with `sorry`.** This directory is `sorry`-free and builds with
   zero warnings; keep it that way. If something genuinely must be assumed,
   declare it as a named `opaque` constant in the interface and add a row to
   `TCB.md` layer 3. `sorry` elaborates to `sorryAx`, which can inhabit `False`;
   `opaque` asserts only that an inhabitant exists. Use `opaque`, not `axiom` —
   `axiom` breaks the code generator, which this build needs.

   **`opaque` is not available at an `of_isOk`-gated site.** If the generated
   file wraps the symbol in `RustM.of_isOk _ (by rfl)`, making it `opaque`
   turns the obligation from provable into *unprovable*: `RustM α` is
   `Option (Except Error α)`, so an opaque inhabitant could be `div` or `fail`,
   and `isOk` does not reduce through an opaque head. `native_decide` does not
   rescue it either — the evaluator cannot unfold an opaque. At such a site the
   only options are a total body (step 2) or patching the generated file, and
   the body is always the better trust position. Check before reaching for
   `opaque`: `grep -B20 '<symbol>' extraction/p3_baby_bear.lean | grep of_isOk`.

Watch for one non-obvious trap: use `Vector.ofFn`, not `Vector.map`, when a
stub builds an array. `Array.map`'s `size` does not reduce definitionally, which
silently breaks the length assertions the extraction makes about constant
tables.

To recompute the failing set from scratch:

```bash
# 1. restore every site
perl -0777 -i -pe 's/^([ \t]*)\(by (?:sorry|native_decide)\)[^\n]*\n(?:[ \t]*--[^\n]*\n)*?[ \t]*-- \(by rfl\)$/$1(by rfl)/gm' extraction/p3_baby_bear.lean
# 2. list the ones that genuinely fail
lake build 2>&1 | grep 'Tactic `rfl` failed' \
  | grep -oE 'p3_baby_bear\.lean:[0-9]+' | cut -d: -f2 | sort -n -u
# 3. fix the gating stub, or re-admit only those (bottom-up), then rebuild
```

At the pins recorded in `TCB.md` this yields exactly one failing site
(`SAMPLING_BITS_M`).

## Step 3 — refresh `TCB.md`

Re-derive every figure; do not leave a stale number:

```bash
cd baby-bear/proofs/legacy-lean
wc -l extraction/p3_baby_bear.lean extraction/p3_baby_bear/*.lean \
      spec/p3_baby_bear_proofs/*.lean

# patches: count files and sum hunks across the set (there is no single patch)
ls patches/post-extraction/[0-9]*.patch | wc -l
cat patches/post-extraction/[0-9]*.patch | grep -c '^@@'
cat patches/post-extraction/[0-9]*.patch | wc -l

# `opaque` DECLARATIONS, not prose mentions -- a bare `grep -c opaque`
# over-counts, because the interface's own comments discuss the keyword.
grep -hcE '^[[:space:]]*(@\[[a-z_, ]*\][[:space:]]*)?opaque[[:space:]]' \
    extraction/p3_baby_bear/*.lean | awk '{s+=$1} END {print s}'

grep -c 'sorry'  extraction/p3_baby_bear.lean       # must be 0
grep -c '^theorem' spec/p3_baby_bear_proofs/constants.lean

# `of_isOk` obligations: the total, then how many are discharged by `rfl`.
# Count `(by rfl)` ANYWHERE on the line, not just `^\s*(by rfl)$` -- one
# obligation is written inline and an anchored pattern silently undercounts.
# `rfl` + `native_decide` must sum to the `of_isOk` total.
grep -c 'RustM.of_isOk' extraction/p3_baby_bear.lean
grep -c '^[^-]*(by rfl)' extraction/p3_baby_bear.lean       # excludes the commented site
grep -c '(by native_decide)' extraction/p3_baby_bear.lean

grep -c 'by sorry) -- PATCHED' extraction/p3_baby_bear.lean  # must be 0

# Warnings. The two `Lean.Grind.USize64.{nat,int}Cast` lints come from the
# PINNED Hax proof library under .lake/packages and are not ours to fix, so
# filter them; what remains must be empty.
rm -rf .lake/build && lake exe cache get && lake build 2>&1 \
  | grep '^warning' | grep -v 'Hax/rust_primitives/USize64'

./patches/check-patches.sh
python3 -c "import json;print([(p['name'],p.get('rev')) for p in json.load(open('lake-manifest.json'))['packages']])"
```

Then walk the `TCB.md` audit checklist and confirm every box. Note any
`opaque` added or removed and update the layer-3 and layer-4 tables. A clean
build must stay at **0 warnings**: that is what makes an accidentally
reintroduced `sorry` visible.

## Maintenance: bumping the hax or Lean pin

Keep this on its own commit, separate from any upstream sync.

1. **`lean-toolchain` is not simply Hax's tag.** Two pins compete for it: the
   Hax `proof-libs/legacy-lean` library declares `v4.29.0-rc1`, while CompPoly
   and mathlib (layer 6) require `v4.30.0`. One Lake build gets one toolchain,
   so the root is pinned to **the higher of the two** — currently `v4.30.0` —
   and the Hax library is built under it. That works today; if a future bump
   breaks the Hax library, the fallbacks are to pin CompPoly to a release
   matching Hax's toolchain, or to keep the two in separate packages.
2. `lake update` (refreshes `lake-manifest.json`), then re-pin the `rev` in
   `lakefile.toml` to the same SHA. **Never leave the lakefile floating on
   `main`** — see `TCB.md` layer 2. Follow with `lake exe cache get`: a pin move
   invalidates the mathlib cache, and without it the next build compiles mathlib
   from source.
3. Re-run `build-proofs.sh` and reconcile. Fix broken patches rather than
   admitting anything.
4. Confirm the spec was actually re-elaborated, not replayed — `lake build`
   should report `Built p3_baby_bear_proofs.constants`. See Troubleshooting.

Do **not** run `lake update` during a routine re-extraction.

## What "good" looks like

- `build-proofs.sh` runs green through step 6/6; no errors, no `.rej` or
  `.orig` files.
- `grep -c sorry extraction/p3_baby_bear.lean` is **0** — every assumption lives
  in `extraction/p3_baby_bear/`, not in the generated file.
- `git status --porcelain baby-bear/src monty-31/src` is empty — the extraction
  modifies no Rust source.
- Re-running `build-proofs.sh` leaves `extraction/p3_baby_bear.lean`
  byte-identical, and `patches/check-patches.sh` exits 0.
- The coverage probes still pass:
  ```bash
  cd baby-bear/proofs/legacy-lean
  grep -c 2013265921 extraction/p3_baby_bear.lean   # PRIME  = 0x78000001, >= 1
  grep -c 2281701377 extraction/p3_baby_bear.lean   # MONTY_MU = 0x88000001, >= 1
  grep -cE '^structure BabyBearParameters' extraction/p3_baby_bear.lean  # 1
  grep -ciE 'neon|avx2|avx512' extraction/p3_baby_bear.lean              # 0
  ```
  The last two matter most: an extraction can be `sorry`-free and still be
  vacuous. An earlier survey of this crate reported "zero `sorry`" for output
  that never defined `BabyBearParameters` at all and was full of leaked NEON
  instances. Check that the field data is *defined*, not merely mentioned.

## Troubleshooting

### `failed to read file '.../X.olean', incompatible header`

A dependency's build tree holds oleans produced by a different Lean than the one
`lean-toolchain` selects. This happened once after `.lake/` was moved on disk:
`lake build` kept succeeding because our own oleans were still cached, so the
failure stayed hidden until a source change forced re-elaboration.

`lake exe cache get` will **not** overwrite olean files that already exist, so
re-running it does not help. Move the offending package's build tree aside and
re-fetch:

```bash
cd baby-bear/proofs/legacy-lean
mv .lake/packages/<pkg>/.lake/build /tmp/<pkg>-build-backup
lake exe cache get
lake build
```

Restore the backup if the result is worse. Two sanity signals that the spec is
genuinely being checked rather than replayed:

- `lake build` reports `Built p3_baby_bear_proofs.constants`, not `Replayed`.
- Lake is content-hashed, so `touch` does **not** force a rebuild. Make a real
  content change (add a comment) if you need to prove the spec re-elaborates.
