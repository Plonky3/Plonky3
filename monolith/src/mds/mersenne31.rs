//! Concrete (MDS) layer for Monolith-31.

use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_mds::MdsPermutation;
use p3_mds::karatsuba_convolution::Convolve;
use p3_mds::util::{dot_product, first_row_to_first_col};
use p3_mersenne_31::Mersenne31;
use p3_symmetric::Permutation;
use sha3::digest::{ExtendableOutput, Update};
use sha3::{Shake128, Shake128Reader};

use crate::util::get_random_u32;

/// MDS matrix implementation for the Monolith-31 Concrete layer.
#[derive(Clone, Debug)]
pub struct MonolithMdsMatrixMersenne31<const NUM_ROUNDS: usize>;

/// Precomputed first row of the 16x16 circulant MDS matrix for Mersenne31.
///
/// Taken from Section 4.5 of the Monolith paper.
const MATRIX_CIRC_MDS_16_MERSENNE31_MONOLITH_ROW: [i64; 16] = [
    61402, 17845, 26798, 59689, 12021, 40901, 41351, 27521, 56951, 12034, 53865, 43244, 7454,
    33823, 28750, 1108,
];

/// Convolution engine for the Monolith MDS matrix over Mersenne31.
///
/// The matrix entries sum to ~524757 < 2^20 << 2^24, satisfying the
/// "small RHS" requirement for safe i64 accumulation with N=16.
struct MonolithConvolveMersenne31;

impl Convolve<Mersenne31, i64, i64> for MonolithConvolveMersenne31 {
    const T_ZERO: i64 = 0;
    const U_ZERO: i64 = 0;

    #[inline(always)]
    fn halve(val: i64) -> i64 {
        val >> 1
    }

    /// Lift a Mersenne31 element to i64 for convolution.
    #[inline(always)]
    fn read(input: Mersenne31) -> i64 {
        input.as_canonical_u32() as i64
    }

    /// For N=16: |x| < N * 2^31 = 2^35, |y| < 2^17 (max entry ~61402),
    /// so each product < 2^52. Sum of 16 products < 2^56, fits in i64.
    #[inline(always)]
    fn parity_dot<const N: usize>(u: [i64; N], v: [i64; N]) -> i64 {
        dot_product(u, v)
    }

    #[inline(always)]
    fn reduce(z: i64) -> Mersenne31 {
        debug_assert!(z >= 0);
        Mersenne31::from_u64(z as u64)
    }
}

impl<const WIDTH: usize, const NUM_ROUNDS: usize> Permutation<[Mersenne31; WIDTH]>
    for MonolithMdsMatrixMersenne31<NUM_ROUNDS>
{
    fn permute(&self, input: [Mersenne31; WIDTH]) -> [Mersenne31; WIDTH] {
        if WIDTH == 16 {
            const COL: [i64; 16] =
                first_row_to_first_col(&MATRIX_CIRC_MDS_16_MERSENNE31_MONOLITH_ROW);
            // Safety: WIDTH == 16 so the cast is valid.
            let input_16: [Mersenne31; 16] = input[..].try_into().unwrap();
            let out_16 = MonolithConvolveMersenne31::apply(
                input_16,
                COL,
                MonolithConvolveMersenne31::conv16,
            );
            out_16[..].try_into().unwrap()
        } else {
            // For non-standard widths, derive a Cauchy MDS matrix from SHAKE-128.
            let mut shake = Shake128::default();
            shake.update(b"Monolith");
            shake.update(&[WIDTH as u8, NUM_ROUNDS as u8]);
            shake.update(&Mersenne31::ORDER_U32.to_le_bytes());
            // The [16, 15] encodes the bit parameters for the Cauchy construction.
            shake.update(&[16, 15]);
            shake.update(b"MDS");
            let mut shake_finalized = shake.finalize_xof();
            apply_cauchy_mds_matrix(&mut shake_finalized, input)
        }
    }
}

impl<const WIDTH: usize, const NUM_ROUNDS: usize> MdsPermutation<Mersenne31, WIDTH>
    for MonolithMdsMatrixMersenne31<NUM_ROUNDS>
{
}

/// Multiply a state vector by a Cauchy MDS matrix derived from SHAKE-128.
///
/// A Cauchy matrix has entries 1 / (x_i + y_j) where x and y are vectors
/// with distinct x-components. The vectors are derived from the SHAKE
/// stream with careful masking to ensure no overflow when computing
/// x_i + y_j (which must stay below p).
fn apply_cauchy_mds_matrix<F: PrimeField32, const WIDTH: usize>(
    shake: &mut Shake128Reader,
    to_multiply: [F; WIDTH],
) -> [F; WIDTH] {
    let mut output: [F; WIDTH] = [F::ZERO; WIDTH];

    // Compute masks that ensure x_i + y_j < p.
    // - x_mask keeps the value well below p (by ~8 bits).
    // - y_mask keeps the value below p/4 so x + y < p.
    let bits = F::bits();
    let x_mask = (1 << (bits - 9)) - 1;
    let y_mask = ((1 << bits) - 1) >> 2;

    // Sample y values with distinct x-components (x = y & x_mask).
    let y = get_random_y_i::<WIDTH>(shake, x_mask, y_mask);
    let mut x = y;
    x.iter_mut().for_each(|x_i| *x_i &= x_mask);

    // Compute output[i] = sum_j (1/(x_i + y_j)) * input[j].
    for (i, x_i) in x.iter().enumerate() {
        for (j, y_j) in y.iter().enumerate() {
            let val = unsafe {
                // Safety: x_i < x_mask < p/256 and y_j < y_mask < p/4,
                // so x_i + y_j < p and from_canonical_unchecked is valid.
                F::from_canonical_unchecked(x_i + y_j).inverse()
            };
            output[i] += val * to_multiply[j];
        }
    }

    output
}

/// Sample WIDTH random y-values from SHAKE such that their x-components
/// (obtained by masking with x_mask) are all distinct.
fn get_random_y_i<const WIDTH: usize>(
    shake: &mut Shake128Reader,
    x_mask: u32,
    y_mask: u32,
) -> [u32; WIDTH] {
    let mut res = [0; WIDTH];

    for i in 0..WIDTH {
        // Rejection loop: keep sampling until x_i is distinct from all previous.
        let mut y_i = get_random_u32(shake) & y_mask;
        let mut x_i = y_i & x_mask;
        while res.iter().take(i).any(|r| r & x_mask == x_i) {
            y_i = get_random_u32(shake) & y_mask;
            x_i = y_i & x_mask;
        }
        res[i] = y_i;
    }

    res
}
