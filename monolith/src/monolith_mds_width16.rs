//! Monolith-31's default MDS permutation.
//! With significant inspiration from https://extgit.iaik.tugraz.at/krypto/zkfriendlyhashzoo/

use p3_mds::MdsPermutation;
use p3_symmetric::Permutation;

use crate::monolith_width16::reduce64;

#[derive(Clone)]
pub struct MonolithMdsMatrixM31Width16;

const MATRIX_CIRC_MDS_16_M31_MONOLITH: [u64; 16] = [
    61402, 17845, 26798, 59689, 12021, 40901, 41351, 27521, 56951, 12034, 53865, 43244, 7454,
    33823, 28750, 1108,
];

fn dot_product<const N: usize>(u: &[u64; N], v: &[u64; N]) -> u64 {
    u.iter().zip(v).map(|(x, y)| x * y).sum()
}

fn apply_circulant_width16<const N: usize>(circ_matrix: &[u64; N], input: [u64; N]) -> [u64; N] {
    let mut matrix = *circ_matrix;

    let mut output = [0; N];
    for out_i in output.iter_mut().take(N - 1) {
        *out_i = dot_product(&matrix, &input);
        matrix.rotate_right(1);
    }
    output[N - 1] = dot_product(&matrix, &input);
    output
}

impl Permutation<[u64; 16]> for MonolithMdsMatrixM31Width16 {
    fn permute(&self, input: [u64; 16]) -> [u64; 16] {
        let matrix: [u64; 16] = MATRIX_CIRC_MDS_16_M31_MONOLITH[..]
            .try_into()
            .unwrap();
        let mut output = apply_circulant_width16(&matrix, input);

        for el in output.iter_mut() {
            reduce64(el);
        }

        output
    }

    fn permute_mut(&self, input: &mut [u64; 16]) {
        *input = self.permute(*input);
    }
}

impl MdsPermutation<u64, 16> for MonolithMdsMatrixM31Width16 {}
