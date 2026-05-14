# p3-rpo-xhash

Permutations for the **RPO** (Rescue Prime Optimized) and **xHash** hash families
over four Plonky3 fields:

- BabyBear (p = 15·2²⁷ + 1, width 24)
- KoalaBear (p = 2³¹ − 2²⁴ + 1, width 24)
- Mersenne31 (p = 2³¹ − 1, width 24)
- Goldilocks (p = 2⁶⁴ − 2³² + 1, width 12)

## Running tests

```bash
cargo test -p p3-rpo-xhash
```

The interesting tests:
- `rpo::*::pow_inv*_known_answer` — Sage-derived ground truth for each inverse S-box
- `rpo::*::pow*_roundtrip` — `x^d` then `x^{1/d}` returns `x`
- `rpo::goldilocks::miden_compat::matches_miden_crypto_rpo256` — RPO-Goldilocks
  primitives match miden-crypto's `Rpo256` on three known inputs (state =
  `[0..12]`, `[42; 12]`, `[(i+1)·10⁹+7]`).

## Running benchmarks

```bash
# All variants (full sampling, ~3 minutes):
cargo bench -p p3-rpo-xhash --bench hashes

# Single field, quick mode (~10 seconds each):
cargo bench -p p3-rpo-xhash --bench hashes babybear -- --quick
```

Benches use `criterion`. Output goes to `target/criterion/`.

## Numbers on Apple M4 (single permutation, scalar paths on aarch64 NEON)

Full-sampling cargo bench (variance is sub-1%):

```
rpo/permute24/babybear      1.93 µs
rpo/permute24/koalabear     1.88 µs
rpo/permute24/m31_cir       6.43 µs    (paper's truncated 32×32 circulant, naive O(N²))
rpo/permute24/m31_bb_mds    1.53 µs    (BB column over M31 via Karatsuba+Barrett)
rpo/permute12/goldilocks    2.63 µs

xhash/permute24/babybear    1.12 µs
xhash/permute24/koalabear   1.06 µs
xhash/permute24/m31_cir     3.21 µs
xhash/permute24/m31_bb_mds  0.86 µs
xhash/permute12/goldilocks  1.36 µs
```

The M31 `_cir` variant uses the paper's spec MDS and is included for reference. The
`_bb_mds` variant uses BabyBear's MDS column lifted to M31 and routed through
Plonky3's Karatsuba+Barrett convolution; it is **~4× faster** but has not been
proven MDS over M31. The MDS check is computationally heavy but should be feasible on a workstation.

## Comparison vs Plonky3 Poseidon1 (same machine, scalar paths)

```
Field        Width   RPO      xHash    Poseidon1
BabyBear     24      1.93     1.12     1.91
KoalaBear    24      1.88     1.06     1.71
Mersenne31   24      1.53*    0.86*    — (no w24)
Goldilocks   12      2.63     1.36     0.92

* using BB-column MDS over M31
```

xHash beats Poseidon1 scalar at width 24 for all three 31-bit fields. RPO is
within 13% of Poseidon1 on BB/KB. Goldilocks Poseidon1 is faster (different
trade-off: Poseidon1 has cheap partial rounds, RPO/xHash pays the inverse
S-box every round).

## What's where

```
src/
  rpo/        RPO permutation generic (RpoHash) + per-field instantiations
  xhash/      xHash permutation generic (XHash) + per-field instantiations
  pow_map/    Extension-field S-box (z ↦ z^d over F_{p^k}) per field
  ext_arith/  F_{p^2} and F_{p^3} arithmetic primitives
  fft_mds/    Field-independent FFT MDS at widths 8 and 12 (used by Goldilocks)
  mds_goldilocks.rs  Goldilocks MDS via hi/lo-split FFT
  mds_m31_bb.rs      M31 MDS using BB column via Karatsuba+Barrett
  reduce/     Field-specific modular reduction helpers
```

Public type aliases:

| Field      | RPO type            | xHash type           |
|------------|---------------------|----------------------|
| BabyBear   | `RpoBabyBear`       | `XHashBabyBear`      |
| KoalaBear  | `RpoKoalaBear`      | `XHashKoalaBear`     |
| Mersenne31 | `RpoM31Cir`, `RpoM31BBMds` | `XHashM31Cir`, `XHashM31BBMds` |
| Goldilocks | `RpoGoldilocks`     | `XHashGoldilocks`    |

Factory functions take an `impl rand::Rng` to generate round constants.

### A note on xHash-BabyBear's extension degree

BabyBear is the one field where xHash's extension S-box uses **F_{p²}**
(α² = 11) rather than the F_{p³} used for the other three fields. The reason
is the smallest valid exponent: `gcd(d, p^k − 1) = 1` is required for `x^d`
to be a permutation. For BabyBear,

- in F_{p²}: `d = 7` works (the same exponent as the base field).
- in F_{p³}: `d = 5` and `d = 7` both fail; the smallest valid prime is
  `d = 11`, which is materially more expensive (3 squarings + 2 multiplications
  vs. 2 + 2 for `d = 7`).

For KoalaBear, Mersenne31, and Goldilocks, F_{p³} admits a small-degree
exponent (3, 5, 7 respectively), so they all use F_{p³}.

**Caveat**: this deviates from the canonical xHash construction (which uses
F_{p³}). The xHash-BabyBear instance — including its round count — would
need a fresh cryptanalysis (algebraic / Gröbner-basis attacks at minimum)
before relying on it for any production setting.

## SIMD on aarch64

The forward and inverse base-field S-boxes and the RC addition use
`<F as Field>::Packing` from Plonky3 — 4-wide NEON for BabyBear / KoalaBear /
Mersenne31 (via `vqdmulhq_s32`-based Montgomery), 2-wide for Goldilocks.
Goldilocks doesn't have a widening 64-bit multiply in NEON, so its packed
multiplication is implemented via interleaved scalar `mul`+`umulh` in inline
asm and ends up being slower than the auto-scheduled scalar path on Apple
Silicon — we therefore do **not** use the packed path for Goldilocks S-boxes.
