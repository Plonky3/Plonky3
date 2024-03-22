use p3_poseidon2::{DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{KoalaBear, monty_reduce};

// Diffusion matrices for Koalabear16.
//
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17]
// [0, 1, 2, 4, 8, 16, 32, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
// Thuse can be verified by the following sage code (Changing vector to the desired vector):
// 
// field = GF(2^31 - 2^24 + 1);
// vector = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17];
// const_mat = matrix(field, 16, lambda i, j: 1);
// diag_mat  = diagonal_matrix(field, vector);
// assert (const_mat + diag_mat).characteristic_polynomial().is_irreducible()
// 
// In order to use these to their fullest potential we need to slightly reimage what the matrix looks like.
// Note that if (1 + D(v)) is a valid matrix then so is r(1 + D(v)) for any constant scalar r. Hence we should operate
// such that (1 + D(v)) is the monty form of the matrix. This allows for delayed reduction tricks.

/// As we are multiplying by powers of 2 (or 0), we save the corresponding shifts.
/// The first entry of this constant should never be accessed.
const MATRIX_DIAG_16_KOALABEAR_SHIFTS: [i32; 16] = [
    i32::MIN, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15
];

fn matmul_internal_shift<const WIDTH: usize>(
    state: &mut [KoalaBear; WIDTH],
    mat_internal_diag_shifts: [i32; WIDTH],
) {
    let sum: u64 = state.iter().map(|x| x.value as u64).sum();
    state[0] = KoalaBear{ value:monty_reduce(sum) };
    for i in 1..WIDTH {
        let result = ((state[i].value as u64) << mat_internal_diag_shifts[i]) + sum.clone();
        state[i] = KoalaBear{ value:monty_reduce(result) };
    }
}


#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixKoalabearScalar;

impl Permutation<[KoalaBear; 16]> for DiffusionMatrixKoalabearScalar {
    fn permute_mut(&self, state: &mut [KoalaBear; 16]) {
        matmul_internal_shift::<16>(state, MATRIX_DIAG_16_KOALABEAR_SHIFTS);
    }
}

impl DiffusionPermutation<KoalaBear, 16> for DiffusionMatrixKoalabearScalar {}


#[cfg(test)]
mod tests {
    use rand::Rng;
    use p3_mds::util::{dot_product};

    use super::*;

    // Test that matmul_internal_shift is correctly computing the matrix multiplication.
    #[test]
    fn matmul() {
        type F = KoalaBear;

        const MAT_DIAG: [u32; 16] = [0, 1, 2, 4, 8, 16, 32, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
        let mat_diag_koalabear = MAT_DIAG.map(|x| KoalaBear { value: x });
        let ones = [1; 16].map(|x| KoalaBear { value: x });


        for i in 1..16 {
            assert!(MAT_DIAG[i] == 1 << MATRIX_DIAG_16_KOALABEAR_SHIFTS[i]);
        }

        let mut rng = rand::thread_rng();
        let input = rng.gen::<[F; 16]>();
        let mut output = input.clone();

        matmul_internal_shift::<16>(&mut output, MATRIX_DIAG_16_KOALABEAR_SHIFTS);

        for i in 0..16 {
            let mut vec = ones;
            vec[i] += mat_diag_koalabear[i];
            let expected = dot_product(input, vec);
            assert_eq!(output[i], expected);
        }
    }
}
