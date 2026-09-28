> **This document is AI-generated and agent-maintained.**

# Re-extraction runbook

Follow this when `baby-bear/src` (or a dependency's `src`) changes, or when the
hax / Lean pins move.

## Current pins

Checked for a newer hax release on **2026-09-28**: 0.4.1 (released
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

## Does an upstream change need re-extraction?

The generated Lean is not committed, so every `extract.sh` run re-extracts from
the Rust as it is. The question is only whether an upstream change can break
the build. It can if it touched `baby-bear/src/**`, `baby-bear/Cargo.toml`, or
any source a pre-extraction patch applies to (`field/src`, `symmetric/src`,
`matrix/src`, the workspace `Cargo.toml`). A change to an item that is
actually extracted from `monty-31/src` or `mds/src` (`DEPS` in `extract.sh`:
the parameter traits, `MontyField31::new`, its `Clone` impl,
`utils::to_monty`, `first_row_to_first_col`) shows up in
`generated/p3-monty-31/` or `generated/p3-mds/`. A change to something that is
hand-mirrored does not: `MontyField31::new_array`, `new_2d_array` and the
`no_packing` Poseidon layers in `assumptions/Interface/P3Monty31Missing.lean`,
and the p3-field / p3-poseidon1 / p3-poseidon2 items in the other
`assumptions/Interface/` files (see `TCB.md`, layer 3). Nothing detects that
drift automatically: read the diff.

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
| 6 stubs | a template in `generated/*/*/Extraction/*External_Template.lean` declares names the matching file in `assumptions/stubs/*/Assumptions/` does not, or the reverse | the message lists the names; step 2 below |
| 8 `lake build` | the Lean does not elaborate | step 2 below |

## Step 2 — reconcile drift

Everything in `generated/` is rewritten by the next run: never fix anything
there except by a post-extraction patch. Everything else (`assumptions/`,
`spec/`, the patches) is hand-written.

1. Read any `.rej` files and the `lake build` errors.
2. An `Unknown identifier p3_…` is a dependency item the output newly refers
   to. Prefer adding a `--start-from` root to `DEPS` in `extract.sh`; if
   aeneas cannot translate it, add it to the matching `assumptions/Interface/`
   file with a docstring citing the upstream source.
3. A step-6 failure means aeneas now expects different external declarations.
   Update the named `assumptions/stubs/<Lib>/Assumptions/` file, starting from the
   template, and keep every body `opaque`.
4. Hand-edit the generated files until `lake build` is green, marking every
   edit `-- PATCHED`, then capture it before the next `extract.sh` run:
   ```bash
   ./baby-bear/proofs/lean/patches/new-patch.sh post-extraction 040-slug
   ./baby-bear/proofs/lean/patches/check-patches.sh
   ```
   Prefer fixing `assumptions/` over a patch. Never `sorry`.
5. **Do not preserve a patch needlessly.** If a hunk no longer applies because
   aeneas improved, delete it.

## Step 3 — refresh `TCB.md`

```bash
cd baby-bear/proofs/lean
cat generated/p3-baby-bear/P3BabyBear/Extraction/{Types,Funs}.lean | wc -l    # generated, p3-baby-bear
cat generated/p3-{monty-31,mds}/*/Extraction/{Types,Funs}.lean | wc -l        # generated, dependencies
find assumptions -name '*.lean' | xargs cat | wc -l                         # hand-written, trusted
grep -rhcE '^(noncomputable )?opaque ' assumptions | paste -sd+ - | bc       # opaque declarations
grep -rhE '^axiom ' assumptions spec | wc -l                                # must be 0
grep -rn 'sorry' generated/p3-*/*/Extraction assumptions spec \
    --include='*.lean' --exclude='*_Template.lean' | wc -l                  # must be 0
for d in pre-extraction post-extraction; do
  echo "$d: $(ls patches/$d/*.patch | wc -l) files, $(cat patches/$d/*.patch | grep -c '^@@') hunks"
done
lake build 2>&1 | grep -cE '^(warning|error)'                               # must be 0
```

and the axiom footprint of every theorem, which must be exactly
`[propext, Classical.choice, Quot.sound]`:

```bash
{ echo 'import P3BabyBearProofs'; echo 'open P3BabyBearProofs'
  grep -hE '^theorem ' spec/P3BabyBearProofs/*.lean | awk '{print "#print axioms", $2}'
} > /tmp/ax.lean
lake env lean /tmp/ax.lean
```

## Troubleshooting the extraction

| Symptom | Cause |
|---|---|
| NEON or AVX modules in the output | `--targets` did not reach charon. `-C --target` does not (TCB layer 5, bug 1) |
| `Fields missing: clone_from` / `ne` | charon's multi-target mode dropped the default-method roots; extend post-patch 010 |
| `GATs cannot work with the --lift-associated-types option` | a new `-> impl Trait` trait method is reachable; gate it as pre-patch 030 does |
| `Mutually recursive trait declarations … will not type-check` | informational while those traits stay hand-written in `assumptions/Interface/` |
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

If no newer release exists, update only the date in **Current pins**.
