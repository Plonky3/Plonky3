# Patches

There are two kinds of patches.

| | `pre-extraction/` | `post-extraction/` |
|---|---|---|
| Patches | Rust source (any crate in the workspace) | generated Lean (each `<Lib>/Extraction/`) |
| Applied | **before** `cargo hax` runs | **after** `cargo hax` runs |
| Reverted | yes — unconditionally, on exit | no; the next run regenerates `Extraction/` and re-applies it |
| Checked in | the `.patch` files | the `.patch` files (never `Extraction/`) |
| Trust cost | **changes the artifact under verification** | changes only the Lean encoding of it |
| Checked by | `test-pre-patches.py` + a `cargo check` of the extracted variant | `lake build` |
| Currently | 3 patches, `010-`..`030-`. The hidden items are behind `cfg(hax_backend_lean)`; the cfg declaration and some redundant bounds are unconditional | 3 patches, `010-`..`030-` |

A **pre-extraction** patch says *charon/aeneas could not cope with this Rust,
so we modified the Rust*. That is strictly worse: the thing verified is no
longer exactly the thing that ships. Two rules keep that cost down:

1. **Gate on `cfg(hax_backend_lean)`.** hax's Lean backend compiles with that
   cfg (`extract.sh` sets it for every crate); normal builds never do. A
   patch that only changes what exists under the cfg leaves the normal build,
   and therefore every upstream test, exactly as shipped. What remains
   unconditional must be provably inert. The current patches' only
   unconditional changes are redundant (implied) trait bounds and a cfg name
   declaration.
2. **Let rustc check the claim.** A patch hides items; it does not rewrite
   bodies. `extract.sh` compiles p3-baby-bear under the cfg (the variant
   charon sees), so anything still reachable that used a hidden item fails to
   compile.

A **post-extraction** patch says *aeneas translated this badly and we fixed
the translation*. The Rust that ships is still exactly what was extracted, so
the patch is a claim about the encoding. A reviewer checks it by reading the
diff against `.pristine/`, the unpatched output of the last run.

## Upstream tests (`test-pre-patches.py`)

Step 2 of `extract.sh`, modelled on
[zk-sdk-verification#52](https://github.com/QED-it/zk-sdk-verification/pull/52).
It runs `cargo test --workspace --no-fail-fast` on two copies of the tree, one
as shipped and one with `pre-extraction/` applied, and compares them per test.
It also compares the source of every `#[test]` fn, so a patch cannot pass by
weakening a test. A **divergence** is a test that passes as shipped and fails
or disappears with the patches, a patched tree that does not build its tests,
or an edited test. Each divergence is attributed to the first patch that
introduces it, and **the script writes it into that patch's header**, in a
block it maintains just above the diff:

```
# Tested:     test-pre-patches.py: 6688 of 6742 upstream tests pass as
#             shipped, 6687 with the pre-patches; 1 divergence from this patch.
# Divergence: p3_mds[unittests src/lib.rs]::util::tests::first_row_to_first_col_odd_length
#   observed: passes as shipped; FAILED with the patches: assertion `left == right` failed
#   class:    TODO
#   was:      TODO
#   now:      TODO
```

`Tested:` and `observed:` belong to the script and are rewritten every run.
`class`, `was` and `now` (and optionally `witness`, `approved`) belong to the
patch author, and are kept across runs: fill them in by editing the header.
An entry that stops diverging is removed. While any entry says `TODO` the run
fails, so a divergence is never accepted without a person having judged it.
Editing a header does not invalidate the cached test result, because the cache
key hashes only the diff part of each patch. Classifying and re-running is
instant.

| Class | Meaning |
| --- | --- |
| `totality` | Panic / `Result` / `Option` channel changed |
| `observable` | A different value on some input |
| `signature` | An upstream test no longer compiles against a changed item |
| `test-edited` | The patch changes or removes an upstream test |
| `harness` | Layout only (line numbers, file contents); no semantic change |

`--check` writes nothing and instead fails if a header is out of date, which is
the mode for CI. Tests that already fail as shipped are reported but not
attributed. The goal is **no divergences at all**, and today there are none. Results are cached under
`/tmp/hax-extract/pretest/`, keyed on the tree's content outside
`baby-bear/proofs/`, each patch's diff, the script and `rustc --version`.

## Use

```bash
../extract.sh                # everything: tools, tests, both phases, lake build
./check-patches.sh           # conventions + dry-run; step 1 of the above
./test-pre-patches.py        # step 2 on its own
```

Patches apply in filename order (`010-`, `020-`, …), all-or-nothing, with
`patch -p1 -F0` (no fuzz). Pre-extraction paths are from the **repo root**;
post-extraction paths are from **this package** (e.g.
`P3BabyBear/Extraction/Funs.lean`). Do not apply by hand unless
debugging a reject.

After any run, `git status --porcelain` must show no Rust changes. The revert
is `patch -R` of exactly what was applied — never `git checkout`.

## Update

Patches are not regenerated. `./new-patch.sh` never overwrites.

| Situation | Action |
|-----------|--------|
| New deviation | Edit the live file(s) (for post-extraction, under `<Lib>/Extraction/`), then `./new-patch.sh <phase> 040-short-slug` (number must sort last) before the next `extract.sh` run. Fill the TODO header. |
| A patch no longer applies | Edit **that** `.patch` until `./check-patches.sh` is green. `.rej` files show what moved. |
| A patch is obsolete | Delete it. Shrinking the set is the goal. |

In generated Lean: mark every edit `-- PATCHED`; keep a replaced call as a
comment beside it. Prefer fixing an `Assumptions/` file over adding a
patch. Never `sorry`. In Rust: mark every edit `PATCHED` in a comment.

Header fields: `Patch`, `Phase`, `Target`, `Hunks`, `Cost` (required);
`Upstream`, `Drop when` when they apply. `Tested` and `Divergence` are written
by `test-pre-patches.py`. One concern per file;
`NNN-lowercase-slug.patch`, no spaces.
