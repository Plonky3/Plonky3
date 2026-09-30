# p3-baby-bear proofs

This directory houses the formal verification of this crate, in Lean 4.
[`lean/`](lean/) is a Lake package: it extracts `p3-baby-bear` to Lean with
hax's `lean` backend (charon + aeneas, hax 0.4.1) and proves theorems about the
result.

The extraction is scalar-only (`thumbv7em-none-eabi`). AVX2, AVX-512 and NEON
are compiled in only when those `target_feature`s are enabled
(`baby-bear/src/lib.rs`, `monty-31/src/lib.rs`); they are out of scope. What is
modelled is `p3-monty-31`'s `no_packing` path, which is what those cfgs select
when the features are off. A default x86-64 target does not enable AVX2, so
that path is also what an unconfigured x86-64 build runs.

What is proved:

- BabyBear's constants (`PRIME`, `TWO_ADICITY`, `MONTY_BITS`, `MONTY_MU`) match
  CompPoly's specification of the field; the modulus is prime, `MONTY_MU` is
  the Montgomery inverse, `gcd(7, p - 1) = 1` (so `x ↦ x^7` is a permutation
  of the field), and `TWO_ADICITY` is the largest power of two dividing
  `p - 1`.
- `MontyField31::new` never panics on BabyBear and returns `x · 2^32 mod p`.
  `new_array` and `new_2d_array` never panic either, but those two are the
  hand-written transcriptions in
  `lean/P3BabyBear/Assumptions/P3Monty31.lean`: aeneas drops the Rust
  functions, so the theorems are about that transcription.
- Nine of the eleven Poseidon round-constant length assertions hold
  (`poseidon{1,2}.const_check_N`). The other two are extracted as `poseidon1._`
  (`BABYBEAR_POSEIDON1_RC_16`) and `poseidon2._`
  (`BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL`) and are not proved.

Nothing is proved about field arithmetic (add, mul, Montgomery reduction,
inversion) or about the Poseidon1/2 and MDS permutations.

## Requirements

Install these once. `lean/extract.sh` checks each one and installs nothing; a
missing tool stops it with the command to run.

| Tool | Version | Install |
|---|---|---|
| Rust | current `stable`, via `rustup`: upstream CI builds with the latest stable, and step 2 builds upstream's tests with yours | <https://rustup.rs>; `rustup update stable` |
| `cargo-hax` | **0.4.1**, on `PATH` (or set `HAX_BIN`) | `cargo install --locked cargo-hax@0.4.1` |
| charon and aeneas | the versions hax 0.4.1 pins (`charon nightly-2026.09.02`, `aeneas build-183e4f0`) | `(cd baby-bear && cargo hax tools install)` (checksum-verified, into `~/.cache/hax/`) |
| charon's Rust toolchain | `nightly-2026-08-18` with `rustc-dev`, `llvm-tools`, `rust-src` and target `thumbv7em-none-eabi` | `rustup toolchain install nightly-2026-08-18 --profile minimal --component rustc-dev,llvm-tools,rust-src --target thumbv7em-none-eabi` |
| Lean | `leanprover/lean4:v4.31.0` (`lean/lean-toolchain`) | [elan](https://github.com/leanprover/elan); it fetches this version on first use |
| other | `python3`, `rsync`, `patch`, `git` | system packages |

The pins, and how to move them to a newer hax, are in
[`lean/SYNC.md`](lean/SYNC.md).

## Build

```bash
./baby-bear/proofs/lean/extract.sh              # everything
./baby-bear/proofs/lean/extract.sh --tools-only # check the requirements only
```

The generated Lean is not committed, so this is how to build. A run takes about
30 seconds, most of it the three extractions. The exception is when the Rust or
a pre-extraction patch has changed: step 2 then re-runs the workspace test
suite on the tree with and without the patches, which takes about 10 minutes;
the result is cached on the content of the Rust tree. After a run, `lake build`
in `lean/` rebuilds the proofs alone.

The proofs are not rebuilt when the Rust changes. [`lean/SYNC.md`](lean/SYNC.md)
records the `main` commit they were last synced against, and the command that
lists the commits since then that touch this crate's sources.

See [`lean/README.md`](lean/README.md) for the layout and how to navigate the
extracted code, and [`lean/TCB.md`](lean/TCB.md) for what is trusted.
