//! Concrete (MDS) layer for Monolith-64.

use p3_goldilocks::{Goldilocks, SmallConvolveGoldilocks};
use p3_mds::MdsPermutation;
use p3_mds::karatsuba_convolution::Convolve;
use p3_mds::util::first_row_to_first_col;
use p3_symmetric::Permutation;

/// MDS matrix implementation for the Monolith-64 Concrete layer.
#[derive(Clone, Debug, Default)]
pub struct MonolithMdsMatrixGoldilocks;

/// First row of the 8x8 circulant MDS matrix for Monolith-64 compression mode.
///
/// From Section 4.5 of the paper:
/// M = circ(23, 8, 13, 10, 7, 6, 21, 8)
///
/// Row sum is 96, well within the 2^51 "small RHS" bound required by
/// `SmallConvolveGoldilocks`.
const MATRIX_CIRC_MDS_8_GOLDILOCKS_MONOLITH: [i64; 8] = [23, 8, 13, 10, 7, 6, 21, 8];

/// First row of the 12x12 circulant MDS matrix for Monolith-64 sponge mode.
///
/// From Section 4.5 of the paper:
/// M = circ(7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8)
///
/// Row sum is 160, well within the 2^51 "small RHS" bound required by
/// `SmallConvolveGoldilocks`.
const MATRIX_CIRC_MDS_12_GOLDILOCKS_MONOLITH: [i64; 12] =
    [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8];

impl Permutation<[Goldilocks; 8]> for MonolithMdsMatrixGoldilocks {
    fn permute(&self, input: [Goldilocks; 8]) -> [Goldilocks; 8] {
        const COL: [i64; 8] = first_row_to_first_col(&MATRIX_CIRC_MDS_8_GOLDILOCKS_MONOLITH);
        SmallConvolveGoldilocks::apply(input, COL, SmallConvolveGoldilocks::conv8)
    }
}
impl MdsPermutation<Goldilocks, 8> for MonolithMdsMatrixGoldilocks {}

impl Permutation<[Goldilocks; 12]> for MonolithMdsMatrixGoldilocks {
    fn permute(&self, input: [Goldilocks; 12]) -> [Goldilocks; 12] {
        const COL: [i64; 12] = first_row_to_first_col(&MATRIX_CIRC_MDS_12_GOLDILOCKS_MONOLITH);
        SmallConvolveGoldilocks::apply(input, COL, SmallConvolveGoldilocks::conv12)
    }
}
impl MdsPermutation<Goldilocks, 12> for MonolithMdsMatrixGoldilocks {}

#[cfg(test)]
mod tests {
    use core::array;

    use p3_mds::util::apply_circulant;

    use super::*;

    #[test]
    fn width_8_matches_naive_circulant() {
        let input: [Goldilocks; 8] = array::from_fn(|i| Goldilocks::new(i as u64 + 1));

        let fast = MonolithMdsMatrixGoldilocks.permute(input);
        let naive = apply_circulant(
            &MATRIX_CIRC_MDS_8_GOLDILOCKS_MONOLITH.map(|x| x as u64),
            &input,
        );

        assert_eq!(fast, naive);
    }

    #[test]
    fn width_12_matches_naive_circulant() {
        let input: [Goldilocks; 12] = array::from_fn(|i| Goldilocks::new(i as u64 + 1));

        let fast = MonolithMdsMatrixGoldilocks.permute(input);
        let naive = apply_circulant(
            &MATRIX_CIRC_MDS_12_GOLDILOCKS_MONOLITH.map(|x| x as u64),
            &input,
        );

        assert_eq!(fast, naive);
    }
}
