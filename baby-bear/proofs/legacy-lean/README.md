# Hax `legacy-lean` backend

Lake package for the hax `legacy-lean` extraction of `p3-baby-bear`. This
directory is the package root.

```
legacy-lean/
  extraction/                      generated Lean + the interface it needs
  spec/                            theorems about the extraction
  patches/
    pre-extraction/                Rust diffs, before hax, reverted after
    post-extraction/               Lean diffs, after hax
    check-patches.sh               conventions + dry-run
    new-patch.sh                   author a new patch
  build-proofs.sh                  extract → patch → `lake build`
  lakefile.toml
  lean-toolchain
  lake-manifest.json
  TCB.md                           trusted codebase
  SYNC.md                          re-extraction runbook
```

Two libraries, split by `srcDir` so hax's output directory holds nothing hax did
not write (plus the axioms required to elaborate it):

| Library | `srcDir` | What belongs there |
|---------|----------|-------------------|
| `p3_baby_bear` | `extraction/` | Generated file and its axiomatized dependency interface |
| `p3_baby_bear_proofs` | `spec/` | Hand-written theorems. hax never writes here |

Depends on **Hax** (pinned revision — must match the engine that produced the
extraction) and **CompPoly** (the specification the theorems import).

## `extraction/`

hax's default output path, `<crate>/proofs/<backend>/extraction/`, not
configurable. `build-proofs.sh` overwrites only `p3_baby_bear.lean`.

| Path | Role |
|------|------|
| `p3_baby_bear.lean` | Machine-owned. Raw hax output + post-extraction patches |
| `p3_baby_bear/p3_*.lean` | Hand-written stand-ins for upstream crates (`p3-field`, `p3-monty-31`, `p3-mds`, `p3-poseidon1/2`, `p3-symmetric`) |
| `p3_baby_bear/hax_ext.lean` | Gaps in the Hax proof library. Candidates to upstream and delete |
| `p3_baby_bear/dependencies.lean` | Import aggregator for the files above. Injection target of patch `010` |

## `spec/`

| Path | Role |
|------|------|
| `p3_baby_bear_proofs.lean` | Library root |
| `p3_baby_bear_proofs/` | Theorems relating the extraction to CompPoly |

## `patches/`

Two phases. How to author, update, and bootstrap:
[`patches/README.md`](patches/README.md). Do not catalogue patches here — each
file's header is the rationale; `check-patches.sh` enforces that.

| Path | Applied to | When | Reverted |
|------|------------|------|----------|
| `pre-extraction/` | Rust source, paths from the **repo root** | before `cargo hax` | yes, on exit |
| `post-extraction/` | generated Lean, paths from **this package** | after `cargo hax` | no |

`pre-extraction/` is empty: extraction uses `thumbv7em-none-eabi`, which has no
SIMD, so the backends in `baby-bear/src` and monty-31's packing modules are not
in the crate graph. No `--cfg hax` source patch is required.

`post-extraction/pristine.snapshot.lean` is the last unmodified hax output. It
is rewritten every run and is not committed.

## Scripts

| Script | Does |
|--------|------|
| `build-proofs.sh` | check patches → pre-patch → `cargo hax into legacy-lean` → revert → snapshot → post-patch → `lake build` |
| `patches/check-patches.sh` | Header conventions + dry-run apply. Step 0 of `build-proofs.sh` |
| `patches/new-patch.sh` | Capture a new delta against (pristine + existing patches). Never overwrites |

## Build

```bash
cd baby-bear/proofs/legacy-lean
lake exe cache get            # CompPoly pulls mathlib; skip this and the first build takes hours
lake build

# regenerate from Rust (cargo-hax, pinned nightly, elan/lake):
./build-proofs.sh
```

`build-proofs.sh` runs `cache get` for you. When `baby-bear/src` (or a mirrored
dependency signature) changes, follow [`SYNC.md`](SYNC.md).
