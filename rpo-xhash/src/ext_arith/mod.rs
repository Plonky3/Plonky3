//! Extension field arithmetic for `pow_map` extension S-boxes.
//!
//! Each submodule provides field-specific `mul` / `sqr` functions operating
//! on raw limb arrays (or directly on field types where it's cheaper). No
//! generics, no branches — each field gets specialised code with its
//! irreducible polynomial and reduction baked in.
//!
//! Variants currently kept (in lockstep with the `pow_map::*` submodules):
//!
//!   `fp2::babybear`         BabyBear   F_p² with α² = 11   (xHash-BabyBear)
//!   `fp3::m31`              Mersenne31 F_p³ with α³ = 5    (xHash-M31)
//!   `fp3::koalabear_field`  KoalaBear  F_p³ with α³+α+4=0  (xHash-KoalaBear)
//!   `fp3::goldilocks`       Goldilocks F_p³ with α³ = 2    (xHash-Goldilocks)

pub mod fp2;
pub mod fp3;
