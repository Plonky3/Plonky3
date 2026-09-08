# Plonky3 architecture

Plonky3 is a collection of small crates rather than one fixed proof system. Applications select fields, transforms, commitments, challengers, and a STARK layer, then connect them through traits from the foundational crates.

## Crate map

| Area | Crates | Responsibility |
| --- | --- | --- |
| Interfaces and utilities | `p3-air`, `p3-field`, `p3-matrix`, `p3-commit`, `p3-symmetric`, `p3-util`, `p3-maybe-rayon` | AIR, field, matrix, commitment, hash/permutation, low-level utility, and sequential/parallel interfaces |
| Prime fields | `p3-baby-bear`, `p3-koala-bear`, `p3-monty-31`, `p3-mersenne-31`, `p3-goldilocks`, `p3-bn254` | Scalar, extension, and packed field implementations |
| Binary fields | `p3-binary-field` | Binary tower fields, GHASH representation, packed arithmetic, and binary challenger |
| Transforms and codes | `p3-dft`, `p3-binary-dft`, `p3-multilinear-util`, `p3-sumcheck`, `p3-zk-codes` | Multiplicative FFTs, additive NTTs, multilinear operations, sumcheck, and ZK Reed–Solomon encoding |
| Commitments | `p3-merkle-tree`, `p3-fri`, `p3-stir`, `p3-whir`, `p3-binary-pcs`, `p3-circle` | Matrix commitments and polynomial opening/proximity protocols |
| Transcripts and analysis | `p3-challenger`, `p3-security` | Typed Fiat–Shamir challengers and composable soundness accounting |
| STARK systems | `p3-uni-stark`, `p3-batch-stark`, `p3-multi-stark`, `p3-lookup` | Single-AIR, batched, and multilinear STARK proving plus LogUp lookup arguments |
| Permutations and hashes | `p3-mds`, `p3-poseidon1`, `p3-poseidon2`, `p3-rescue`, `p3-monolith`, `p3-blake3`, `p3-keccak`, `p3-sha256` | Algebraic permutations, conventional hashes, and shared MDS layers |
| AIR implementations | `p3-poseidon1-air`, `p3-poseidon2-air`, `p3-monolith-air`, `p3-blake3-air`, `p3-keccak-air`, `p3-sha256-air` | AIR constraints and trace generation for the corresponding primitive |
| Development | `p3-field-testing`, `p3-examples` | Reusable field test helpers and runnable proof examples |

The common univariate path is:

```text
AIR + trace
    -> p3-uni-stark or p3-batch-stark
    -> p3-commit::Pcs implementation
    -> evaluation transform + MMCS/Merkle hashing
    -> Fiat-Shamir challenger
    -> proof / typed verification result
```

`p3-multi-stark` instead works with multilinear polynomials, sumcheck, and a `MultilinearPcs`. `p3-circle` supplies circle-domain FFT, FRI, and PCS machinery for Mersenne-31; the generic univariate and batched STARK crates consume that PCS.

## Backend capabilities

| Backend | Polynomial/domain model | Hiding PCS supplied | STARK integration |
| --- | --- | --- | --- |
| `p3-fri::TwoAdicFriPcs` | Univariate polynomials over two-adic multiplicative cosets | No; `p3-fri::HidingFriPcs` is the hiding variant | `p3-uni-stark` and `p3-batch-stark`; their ZK paths follow the backend's `ZK` capability |
| `p3-stir::TwoAdicStirPcs` | Univariate Reed–Solomon proximity testing on two-adic domains | No hiding variant is exposed | Usable through the generic univariate PCS interface |
| `p3-circle::CirclePcs` | Univariate evaluations on circle-group domains with circle FRI | No; the backend's `ZK` capability is false | Consumed by `p3-uni-stark` and `p3-batch-stark` |
| `p3-whir` | Multilinear polynomials and constrained Reed–Solomon codes | Yes: `HidingWhirPcs` provides honest-verifier ZK at the PCS layer | Implements `MultilinearPcs`; applications must compose and analyze the complete protocol |
| `p3-binary-pcs::BinaryPcs` | Multilinear polynomials over `BinaryField128` with an additive-domain code | No; query symbols and the final codeword are revealed | Used by the binary multilinear STARK example in an explicitly non-ZK composition |

A hiding PCS is one part of a zero-knowledge proof system. It does not by itself establish that every trace commitment, auxiliary argument, transcript message, or application-level statement is zero knowledge. The univariate STARK implementations contain explicit paths selected by the backend's `ZK` capability and tests with hiding FRI; other compositions need their own complete protocol analysis.

## Portability and features

The workspace libraries are `no_std`. The embedded check selects library targets from Cargo metadata and excludes the host-oriented `p3-examples` and `p3-field-testing` packages explicitly. Host-only bins and examples do not affect that build because every selected package is built with `--lib`.

The `parallel` feature switches participating crates from the sequential `p3-maybe-rayon` implementation to Rayon. SIMD implementations use Rust target features rather than Cargo features; see the root README for the supported CPU flags. Test-helper feature names are crate-specific: `p3-commit` exposes `test-utils`, while `p3-sumcheck` exposes `test-util`. Both are off by default.
