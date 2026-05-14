//! Extension-field S-box permutations: z ↦ z^d over F_{p^k}.
//!
//! Each sub-module defines a `Permutation<[F; WIDTH]>` type that applies the
//! S-box `z ↦ z^d` over an extension field F_{p^k}, treating the WIDTH-element
//! base-field state as WIDTH/k tuples of k coefficients each (stride-k layout).
//!
//! These types perform ONLY the S-box. The MDS layer happens separately.
//!
//! The batched S-box separates addition-chain stages (sqr, sqr, mul, …) across
//! all S-boxes for instruction-level parallelism.
//!
//! Variants:
//!   BabyBear   `PowMap24`  x^7 over F_{p^2}  (width 24, 12 pairs)
//!   KoalaBear  `PowMap24`  x^3 over F_{p^3}  (width 24, 8 triples)
//!   M31        `PowMap24`  x^5 over F_{p^3}  (width 24, 8 triples)
//!   Goldilocks `PowMap12`  x^7 over F_{p^3}  (width 12, 4 triples)

pub mod babybear;
pub mod goldilocks;
pub mod koalabear;
pub mod m31;
