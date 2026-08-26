//! Shared fixtures for `ring_switch` tests.

use p3_binary_field::{BinaryField8, BinaryField128, TowerLevel};
use p3_multilinear_util::poly::Poly;

pub(crate) type F = BinaryField8;
pub(crate) type EF = BinaryField128;

/// A base-field polynomial with deterministic, seed-dependent evaluations.
pub(crate) fn base_poly(log_n: usize, seed: u64) -> Poly<F> {
    Poly::new(
        (0..1usize << log_n)
            .map(|i| {
                F::from_repr(
                    seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .wrapping_add(i as u64) as u8,
                )
            })
            .collect(),
    )
}
