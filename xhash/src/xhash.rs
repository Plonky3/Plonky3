//! Generic XHash permutation.
//!
//! A XHash instance is parameterised over a base field `F`, an MDS permutation
//! `Mds: MdsPermutation<F, WIDTH>`, and the field-specific S-box callbacks.
//! The S-boxes are not exposed as a trait; per-field instantiations call the
//! permutation routine directly, passing closures.

use alloc::vec::Vec;

use p3_field::PrimeField;

use crate::util::shake256_hash;

/// Per-round constant layout.
///
/// For each of the `num_rounds` rounds we have three width-sized blocks of
/// constants:
///   - `RC_F[i]` — added before the forward S-box,
///   - `RC_B[i]` — added before the backward S-box,
///   - `RC_E[i]` — added before the extension-field S-box.
///
/// Followed by a single width-sized `RC_L` block for the final linear layer.
///
/// Total length: `(3 * num_rounds + 1) * WIDTH`.
#[derive(Clone, Debug)]
pub(crate) struct XHash<F, Mds, const WIDTH: usize, const ALPHA: u64> {
    pub(crate) num_rounds: usize,
    pub(crate) mds: Mds,
    pub(crate) round_constants: Vec<F>,
}

impl<F, Mds, const WIDTH: usize, const ALPHA: u64> XHash<F, Mds, WIDTH, ALPHA>
where
    F: PrimeField,
{
    /// `round_constants` must have length `(3 * num_rounds + 1) * WIDTH`.
    pub fn new(num_rounds: usize, round_constants: Vec<F>, mds: Mds) -> Self {
        const {
            assert!(WIDTH > 0);
            assert!(ALPHA > 1);
        }
        assert_eq!(round_constants.len(), (3 * num_rounds + 1) * WIDTH);
        Self {
            num_rounds,
            mds,
            round_constants,
        }
    }

    /// Derive round constants from SHAKE-256.
    ///
    /// Mirrors the construction used in
    /// [`p3_rescue::Rpo::shake_round_constants`]: each constant consumes
    /// `bytes_per_constant` SHAKE bytes (interpreted little-endian, reduced
    /// mod `p`). The caller must pass `bytes_per_constant = ceil(log2(p)/8) + 1`,
    /// and the value is bounded to 16 so the accumulator fits in a `u128`.
    pub fn shake_round_constants(
        seed: &[u8],
        num_rounds: usize,
        bytes_per_constant: usize,
    ) -> Vec<F> {
        assert!((1..=16).contains(&bytes_per_constant));
        let num_constants = (3 * num_rounds + 1) * WIDTH;
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
