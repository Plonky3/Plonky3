//! Concrete (MDS) layer for Monolith-64.

use p3_goldilocks::Goldilocks;
use p3_mds::MdsPermutation;
use p3_mds::util::apply_circulant;
use p3_symmetric::Permutation;

/// MDS matrix implementation for the Monolith-64 Concrete layer.
#[derive(Clone, Debug, Default)]
pub struct MonolithMdsMatrixGoldilocks;

/// First row of the 8x8 circulant MDS matrix for Monolith-64 compression mode.
///
/// From Section 4.5 of the paper:
/// M = circ(23, 8, 13, 10, 7, 6, 21, 8)
const MATRIX_CIRC_MDS_8_GOLDILOCKS_MONOLITH: [u64; 8] = [23, 8, 13, 10, 7, 6, 21, 8];

/// First row of the 12x12 circulant MDS matrix for Monolith-64 sponge mode.
///
/// From Section 4.5 of the paper:
/// M = circ(7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8)
const MATRIX_CIRC_MDS_12_GOLDILOCKS_MONOLITH: [u64; 12] =
    [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8];

impl Permutation<[Goldilocks; 8]> for MonolithMdsMatrixGoldilocks {
    fn permute(&self, input: [Goldilocks; 8]) -> [Goldilocks; 8] {
        apply_circulant(&MATRIX_CIRC_MDS_8_GOLDILOCKS_MONOLITH, &input)
    }
}
impl MdsPermutation<Goldilocks, 8> for MonolithMdsMatrixGoldilocks {}

impl Permutation<[Goldilocks; 12]> for MonolithMdsMatrixGoldilocks {
    fn permute(&self, input: [Goldilocks; 12]) -> [Goldilocks; 12] {
        apply_circulant(&MATRIX_CIRC_MDS_12_GOLDILOCKS_MONOLITH, &input)
    }
}
impl MdsPermutation<Goldilocks, 12> for MonolithMdsMatrixGoldilocks {}
