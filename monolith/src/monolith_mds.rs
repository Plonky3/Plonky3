//! Monolith-31's default MDS permutation.
//! With significant inspiration from https://extgit.iaik.tugraz.at/krypto/zkfriendlyhashzoo/

use p3_field::PrimeField32;
use p3_mds::util::apply_circulant;
use p3_mds::MdsPermutation;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::{Shake128, Shake128Reader};

#[derive(Clone)]
pub struct MonolithMdsMatrixMersenne31<const NUM_ROUNDS: usize>;

const MATRIX_CIRC_MDS_16_MERSENNE31_MONOLITH: [u64; 16] = [
    61402, 17845, 26798, 59689, 12021, 40901, 41351, 27521, 56951, 12034, 53865, 43244, 7454,
    33823, 28750, 1108,
];

impl<const WIDTH: usize, const NUM_ROUNDS: usize> CryptographicPermutation<[Mersenne31; WIDTH]>
    for MonolithMdsMatrixMersenne31<NUM_ROUNDS>
{
    fn permute(&self, input: [Mersenne31; WIDTH]) -> [Mersenne31; WIDTH] {
        if WIDTH == 16 {
            let matrix: [u64; WIDTH] = MATRIX_CIRC_MDS_16_MERSENNE31_MONOLITH[..]
                .try_into()
                .unwrap();
            apply_circulant(&matrix, input)
        } else {
            let mut shake = Shake128::default();
            shake.update("Monolith".as_bytes());
            shake.update(&[WIDTH as u8, NUM_ROUNDS as u8]);
            shake.update(&Mersenne31::ORDER_U32.to_le_bytes());
            shake.update(&[16, 15]);
            shake.update("MDS".as_bytes());
            let mut shake_finalized = shake.finalize_xof();
            apply_cauchy_mds_matrix(&mut shake_finalized, input)
        }
    }
}

impl<const WIDTH: usize, const NUM_ROUNDS: usize> ArrayPermutation<Mersenne31, WIDTH>
    for MonolithMdsMatrixMersenne31<NUM_ROUNDS>
{
}
impl<const WIDTH: usize, const NUM_ROUNDS: usize> MdsPermutation<Mersenne31, WIDTH>
    for MonolithMdsMatrixMersenne31<NUM_ROUNDS>
{
}

fn apply_cauchy_mds_matrix<F: PrimeField32, const WIDTH: usize>(
    shake: &mut Shake128Reader,
    to_multiply: [F; WIDTH],
) -> [F; WIDTH] {
    let mut output: [F; WIDTH] = [F::ZERO; WIDTH];

    let mut p = F::ORDER_U32;
    let mut tmp = 0;
    while p != 0 {
        tmp += 1;
        p >>= 1;
    }
    let x_mask = (1 << (tmp - 7 - 2)) - 1;
    let y_mask = ((1 << tmp) - 1) >> 2;

    let y = get_random_y_i::<WIDTH>(shake, x_mask, y_mask);
    let mut x = y.to_owned();
    x.iter_mut().for_each(|x_i| *x_i &= x_mask);

    for (i, x_i) in x.iter().enumerate() {
        for (j, yj) in y.iter().enumerate() {
            output[i] += F::from_canonical_u32(x_i + yj).inverse() * to_multiply[j];
        }
    }

    output
}

fn get_random_y_i<const WIDTH: usize>(
    shake: &mut Shake128Reader,
    x_mask: u32,
    y_mask: u32,
) -> [u32; WIDTH] {
    let mut res = [0; WIDTH];
    for i in 0..WIDTH {
        let mut valid = false;
        while !valid {
            let mut rand = [0u8; 4];
            shake.read(&mut rand);

            let y_i = u32::from_le_bytes(rand) & y_mask;

            // check distinct x_i
            let x_i = y_i & x_mask;
            valid = true;
            for r in res.iter().take(i) {
                if r & x_mask == x_i {
                    valid = false;
                    break;
                }
            }
            if valid {
                res[i] = y_i;
            }
        }
    }

    res
}
