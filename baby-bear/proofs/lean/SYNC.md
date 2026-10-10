> **This document is AI-generated and agent-maintained.**

# Re-extraction runbook

Follow this when `baby-bear/src` (or a dependency's `src`) changes, or when the
hax / Lean pins move. Steps 1–4 are the whole sync: re-extract, reconcile, re-measure,
and bring the documentation in line with the result.

## Current pins

Checked for a newer hax release on **2026-09-30**: 0.4.1 (released
2026-09-23) is the latest. `extract.sh` fails if this table does not state the
pins it uses, and prints a note when a newer hax release exists.

| Pin | Value | Set in |
|---|---|---|
| hax (`cargo-hax`) | `0.4.1` (tag `cargo-hax-v0.4.1` = `8d6a41802f32918fe7ebbaad0dfc86937ba3011d`) | `extract.sh`: `HAX_VERSION`, `HAX_COMMIT` |
| charon | `nightly-2026.09.02` (the hax 0.4.1 default) | `extract.sh`: `EXPECT_CHARON` |
| charon's Rust toolchain | `nightly-2026-08-18` (+ `thumbv7em-none-eabi`) | `extract.sh`: `CHARON_TOOLCHAIN` |
| aeneas | `build-183e4f0` (the hax 0.4.1 default) | `extract.sh`: `EXPECT_AENEAS`; `lakefile.toml`: aeneas `rev` |
| hax-lean (`CoreModels`) | `v0.3.27` | `lakefile.toml`: hax `rev` |
| Lean | `leanprover/lean4:v4.31.0` | `lean-toolchain` |
| CompPoly | `v4.31.0` (same mathlib as aeneas) | `lakefile.toml`: CompPoly `rev` |

How to install each tool is in the Requirements table of
[`../README.md`](../README.md), which repeats these versions; step 0 checks it
too.

## Last sync

The proofs were last synced against `main` at
**`24c27941fc2b9a498d6bfdac99838183bac16662`** (2026-09-30): `extract.sh` was
green, with 6715 of 6769 upstream tests passing with and without the
pre-patches.

This is how current the proofs are. They are not re-checked when the Rust
changes, so they can fall behind `main`, and a Rust change never waits on them.
The commits since this one that could affect them are:

```bash
git log --oneline <last sync>..origin/main -- baby-bear ':!baby-bear/proofs' \
    monty-31/src mds/src field/src poseidon1/src poseidon2/src \
    symmetric/src matrix/src Cargo.toml Cargo.lock
```

That is the crate, the dependencies it extracts or hand-mirrors, the sources
the pre-extraction patches touch, and the manifests (see the next section). An
empty list means the proofs are current, however many other commits have
landed. The list is coarse by design: it names every commit to those
directories, including ones to items that are not extracted. When a patch
starts touching a new directory, add it here.

The recorded commit is upstream `main`'s: the commit merged in before the
sync's green run, or, once this tree is on `main`, `main`'s head when the sync
ran. Step 4 below updates it.

## Does an upstream change need re-extraction?

The generated Lean is not committed, so every `extract.sh` run re-extracts from
the Rust as it is. The question is only whether an upstream change can break
the build. It can if it touched `baby-bear/src/**`, `baby-bear/Cargo.toml`, or
any source a pre-extraction patch applies to (`field/src`, `symmetric/src`,
`matrix/src`, the workspace `Cargo.toml`). A change to an item that is
actually extracted from `monty-31/src` or `mds/src` (`DEPS` in `extract.sh`:
the parameter traits, `MontyField31::new`, its `Clone` impl,
`utils::to_monty`, `first_row_to_first_col`) shows up in
`P3Monty31/Extraction/` or `P3Mds/Extraction/`. A change to something that is
hand-mirrored does not: `MontyField31::new_array`, `new_2d_array` and the
`no_packing` Poseidon layers in `P3BabyBear/Assumptions/P3Monty31.lean`, and the
p3-field / p3-poseidon1 / p3-poseidon2 items in `P3Monty31/Assumptions/P3Field.lean`
and `P3BabyBear/Assumptions/P3Poseidon{1,2}.lean` (see `TCB.md`, layer 3). Nothing detects that
drift automatically: read the diffs of the commits **Last sync**'s command lists.

## Step 1 — re-extract

```bash
./baby-bear/proofs/lean/extract.sh
```

The script checks its tools against the pins above (step 0) and stops on the
first failure. Steps, and what a failure at each one means:

| Step | Fails when | Where to look |
|---|---|---|
| 0 tools | a prerequisite is missing, `cargo-hax` is not the pinned release, or a pin disagrees (`hax.toml` above the crate; `lakefile.toml`, `lean-toolchain` or the table above out of step) | the message names the pin |
| 1 conventions | a patch header is malformed or a patch does not apply | `patches/check-patches.sh` output |
| 2 upstream tests | a test diverges under the pre-patches: the script writes a `# Divergence:` entry into the responsible patch's header, and fails while any entry's `class` is `TODO` | that patch's header; `patches/README.md` |
| 3 cfg'd check | a pre-patch hid something that is still reachable | rustc's error names the use |
| 4 extraction | charon or aeneas reports an error | the log; see the troubleshooting table below |
| 6 stubs | a template in `<Lib>/Extraction/*External_Template.lean` declares names the matching `<Lib>/Assumptions/*External.lean` does not, or the reverse; or that file is hax's unfilled seed | the message lists the names; step 2 below |
| 8 `lake build` | the Lean does not elaborate | step 2 below |

## Step 2 — reconcile drift

Every `<Lib>/Extraction/` is rewritten by the next run: never fix anything
there except by a post-extraction patch. Everything else (each `Assumptions/`,
`P3BabyBear/Verification/`, the patches) is hand-written.

1. Read any `.rej` files and the `lake build` errors.
2. An `Unknown identifier p3_…` is a dependency item the output newly refers
   to. Prefer adding a `--start-from` root to `DEPS` in `extract.sh`; if
   aeneas cannot translate it, hand-write it in the `Assumptions/` of the
   library whose extraction refers to it, with a docstring citing the upstream
   source.
3. A step-6 failure means aeneas now expects different external declarations.
   Update the named `<Lib>/Assumptions/` file, starting from the template, and
   keep every body `opaque`. If hax has just created one (a new external item
   in a library that had none), fill in its holes the same way.
4. Hand-edit the generated files until `lake build` is green, marking every
   edit `-- PATCHED`, then capture it before the next `extract.sh` run:
   ```bash
   ./baby-bear/proofs/lean/patches/new-patch.sh post-extraction 040-slug
   ./baby-bear/proofs/lean/patches/check-patches.sh
   ```
   Prefer fixing an `Assumptions/` file over a patch. Never `sorry`.
5. **Do not preserve a patch needlessly.** If a hunk no longer applies because
   aeneas improved, delete it.

## Step 3 — re-measure

```bash
cd baby-bear/proofs/lean
cat P3BabyBear/Extraction/{Types,Funs}.lean | wc -l                        # generated, p3-baby-bear
cat P3{Monty31,Mds}/Extraction/{Types,Funs}.lean | wc -l                    # generated, dependencies
cat P3*/Assumptions/*.lean | wc -l                                          # hand-written, trusted
grep -hcE '^(noncomputable )?opaque ' P3*/Assumptions/*.lean | paste -sd+ - | bc   # opaque declarations
grep -hE '^axiom ' P3*/Assumptions/*.lean P3BabyBear/Verification/*.lean | wc -l  # must be 0
grep -rnE '(^|[^`])sorry([^`]|$)' P3*/Extraction P3*/Assumptions P3BabyBear/Verification \
    --include='*.lean' --exclude='*_Template.lean' | wc -l                  # must be 0 (`sorry` in prose is skipped)
for d in pre-extraction post-extraction; do
  echo "$d: $(ls patches/$d/*.patch | wc -l) files, $(cat patches/$d/*.patch | grep -c '^@@') hunks"
done
lake build 2>&1 | grep -cE '^(warning|error)'                               # must be 0
```

and the axiom footprint of every theorem, which must be within
`[propext, Classical.choice, Quot.sound]`:

```bash
{ echo 'import P3BabyBear'; echo 'open p3_baby_bear'
  grep -hE '^theorem ' P3BabyBear/Verification/Proofs.lean | awk '{print "#print axioms", $2}'
} > /tmp/ax.lean
lake env lean /tmp/ax.lean
```

## Step 4 — update the documentation

A sync is not finished while any document describes the tree as it was. Read
the diff of the sync (Rust, patches, `Assumptions/`, `Verification/`,
`extract.sh`) against this table, and update every document it names:

| When the sync changed… | Update |
|---|---|
| anything (every sync) | **Last sync** above: the `main` commit the green run was on, the date, and the test counts |
| any count from step 3 (lines, `opaque`s, hunks, tests run, theorems) | the measured-totals table at the top of `TCB.md`; the test counts in `TCB.md` layer 4; the "Currently" row in `patches/README.md` |
| a patch: added, removed, or its hunks or targets changed | its own header (`Target`, `Hunks`, `Cost`, `Drop when`); the patch tables in `TCB.md` layer 4; this file's list of sources the pre-patches touch (above) |
| a hand-written `Assumptions/` file, or `DEPS` in `extract.sh` | `TCB.md` layer 3 (both tables, and the opaque list); "Where the dependency code lives" in `README.md`; the hand-mirrored list in "Does an upstream change need re-extraction?" above |
| a workaround for an extractor bug, added or dropped | `TCB.md` layer 5; the troubleshooting table below |
| a theorem: added, removed, renamed or restated | `TCB.md`, "The theorems are not in the TCB" and "What is *not* established"; "What is proved" in `../README.md` |
| a file added, moved or removed in the package | the layout tree and import chain in `README.md` |
| the shape of the generated code (names, file sizes, a quoted example) | the naming table and "Where a Rust item ends up" in `README.md` |
| a pin | **Current pins** above; the Requirements table in `../README.md`; `TCB.md` layers 1, 2 and 6 |
| how long a run takes | "Build" in `../README.md` |

Every Lean name a document quotes must still resolve: `#check` each changed one
against `import P3BabyBear`. Describe the tree as it is, not how it changed.

## Troubleshooting the extraction

| Symptom | Cause |
|---|---|
| NEON or AVX modules in the output | `--targets` did not reach charon. `-C --target` does not (TCB layer 5, bug 1) |
| `Fields missing: clone_from` / `ne` | charon's multi-target mode dropped the default-method roots; extend post-patch 010 |
| `GATs cannot work with the --lift-associated-types option` | a new `-> impl Trait` trait method is reachable; gate it as pre-patch 030 does |
| `P3BabyBear.lean does not import P3BabyBear.Verification.ProofObligations` | expected: the root imports it through `P3BabyBear.Verification`, and hax only checks for the direct import (`README.md`, Layout) |
| `Type error after transformations` (2, in `core`'s slice-iterator macros) | expected, in the scoped p3-monty-31 run (`TCB.md`, layer 5, bug 7) |
| `Mutually recursive trait declarations … will not type-check` | informational while those traits stay hand-written in `P3Monty31/Assumptions/P3Field.lean` |
| `E0514` / `can't find crate for core` from charon | stale `rustc-build-sysroot` cache: `rm -rf ~/Library/Caches/org.rust-lang.miri` |

## Maintenance: bumping hax

Find the latest release:

```bash
git ls-remote --tags https://github.com/cryspen/hax 'refs/tags/cargo-hax-v*'   # or: cargo search cargo-hax
```

A new `cargo-hax` resolves a new charon, aeneas, Lean toolchain and hax-lean.

1. `cargo install --locked cargo-hax@<new> --force` and
   `(cd baby-bear && cargo hax tools install)`, then
   `cargo hax tools show` (run in `baby-bear/`) for the versions it resolves,
   and `git ls-remote https://github.com/cryspen/hax refs/tags/cargo-hax-v<new>^{}`
   for the tag's commit.
2. In `extract.sh`, update `HAX_VERSION`, `HAX_COMMIT`, `EXPECT_CHARON`,
   `EXPECT_AENEAS` and `CHARON_TOOLCHAIN` (the `channel` in
   `~/.cache/hax/tools/charon/<version>/rust-toolchain`).
3. Put the aeneas and hax-lean `rev`s into `lakefile.toml` and the Lean version
   into `lean-toolchain`. Choose the CompPoly tag whose mathlib matches
   aeneas's, then `lake update`.
4. Update the **Current pins** table above, including the date checked, and
   the Requirements table in `../README.md`.
5. Run step 1. Step 0 fails until every pin agrees. Re-check each "Drop when"
   in the patch headers: a bump is the usual moment a patch becomes deletable.
6. Finish with step 4.

If no newer release exists, update only the date in **Current pins**.
