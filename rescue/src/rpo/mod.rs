//! Rescue-Prime Optimized (RPO).
//!
//! - Goldilocks (width 12): [eprint 2022/1577](https://eprint.iacr.org/2022/1577).
//! - Mersenne-31 (width 24, with concluding linear layer):
//!   [eprint 2024/1635](https://eprint.iacr.org/2024/1635).

use alloc::vec::Vec;

use p3_field::{PermutationMonomial, PrimeField};

use crate::util::shake256_hash;

mod goldilocks;
mod mersenne_31;

pub(crate) use goldilocks::apply_inv_sbox_x7;
pub use goldilocks::*;
pub use mersenne_31::*;

/// The Rescue-Prime Optimized permutation.
///
/// Each round is two halves:
///   `MDS, +ARK1, x^ALPHA, MDS, +ARK2, x^(1/ALPHA)`.
///
/// When `final_linear_layer` is `true`, an additional `MDS + add constants`
/// step is appended after the last round. This is the "CLS" (concluding
/// linear step) introduced for RPO-M31 in
/// [eprint 2024/1635](https://eprint.iacr.org/2024/1635). The original
/// Goldilocks RPO ([eprint 2022/1577](https://eprint.iacr.org/2022/1577))
/// does **not** include this step, and should be constructed via
/// [`Rpo::new`] instead.
#[derive(Clone, Debug)]
pub(crate) struct Rpo<F, Mds, const WIDTH: usize, const ALPHA: u64> {
    pub(crate) num_rounds: usize,
    pub(crate) mds: Mds,
    pub(crate) round_constants: Vec<F>,
    pub(crate) final_linear_layer: bool,
}

impl<F, Mds, const WIDTH: usize, const ALPHA: u64> Rpo<F, Mds, WIDTH, ALPHA>
where
    F: PrimeField + PermutationMonomial<ALPHA>,
{
    /// `round_constants` must have length `2 * WIDTH * num_rounds`.
    pub fn new(num_rounds: usize, round_constants: Vec<F>, mds: Mds) -> Self {
        const {
            assert!(WIDTH > 0);
            assert!(ALPHA > 1);
        }
        assert_eq!(round_constants.len(), 2 * WIDTH * num_rounds);
        Self {
            num_rounds,
            mds,
            round_constants,
            final_linear_layer: false,
        }
    }

    /// `round_constants` must have length `2 * WIDTH * num_rounds + WIDTH`;
    /// the final block is the CLS layer.
    pub fn new_with_final_linear_layer(
        num_rounds: usize,
        round_constants: Vec<F>,
        mds: Mds,
    ) -> Self {
        const {
            assert!(WIDTH > 0);
            assert!(ALPHA > 1);
        }
        assert_eq!(round_constants.len(), 2 * WIDTH * num_rounds + WIDTH);
        Self {
            num_rounds,
            mds,
            round_constants,
            final_linear_layer: true,
        }
    }

    /// Derive round constants from SHAKE-256.
    ///
    /// Each constant consumes `bytes_per_constant` SHAKE bytes (interpreted
    /// little-endian and reduced mod `p`); per the RPO papers this should be
    /// `ceil(log2(p) / 8) + 1`. Bounded to 16 so the accumulator fits in a
    /// `u128` before reduction.
    pub fn shake_round_constants(
        seed: &[u8],
        num_rounds: usize,
        bytes_per_constant: usize,
        include_final_layer: bool,
    ) -> Vec<F> {
        assert!((1..=16).contains(&bytes_per_constant));
        let num_constants = 2 * WIDTH * num_rounds + if include_final_layer { WIDTH } else { 0 };
        let byte_string = shake256_hash(seed, bytes_per_constant * num_constants);

        byte_string
            .chunks(bytes_per_constant)
            .map(|chunk| {
                let integer = chunk
                    .iter()
                    .rev()
                    .fold(0u128, |acc, &byte| (acc << 8) | byte as u128);
                F::from_u128(integer)
            })
            .collect()
    }
}
