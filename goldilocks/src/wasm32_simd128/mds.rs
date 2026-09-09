use core::arch::wasm32::{
    i64x2_add, i64x2_mul, i64x2_neg, i64x2_shl, i64x2_shr, i64x2_splat, i64x2_sub, u64x2_shr,
    u64x2_splat, v128, v128_and, v128_or,
};
use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use p3_mds::MdsPermutation;
use p3_mds::karatsuba_convolution::Convolve;
use p3_mds::util::{apply_circulant, first_row_to_first_col};
use p3_symmetric::Permutation;

use crate::wasm32_simd128::packing::PackedGoldilocksWasmSimd128;
use crate::{
    MATRIX_CIRC_MDS_8_SML_ROW, MATRIX_CIRC_MDS_12_SML_ROW, MATRIX_CIRC_MDS_16_SML_ROW,
    MATRIX_CIRC_MDS_24_GOLDILOCKS, MdsMatrixGoldilocks,
};

/// Two signed integer lanes used for exact, unreduced limb convolutions.
#[derive(Clone, Copy)]
struct SignedLimb(v128);

impl Add for SignedLimb {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(i64x2_add(self.0, rhs.0))
    }
}

impl AddAssign for SignedLimb {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for SignedLimb {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(i64x2_sub(self.0, rhs.0))
    }
}

impl SubAssign for SignedLimb {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for SignedLimb {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self(i64x2_neg(self.0))
    }
}

/// Integer Karatsuba convolution for the fixed width-8/12/16 MDS coefficients.
///
/// Inputs are 32-bit limbs, N <= 16, and every coefficient is below 128.
/// Before each base-case dot product, an operand is a signed sum of at most N
/// original operands. Bounding a dot product by N terms and every subsequent
/// reconstruction level by a factor of 3 gives the conservative bound
/// `N^3 * 3^4 * 2^32 * 128 < 2^58` on all intermediates. Thus signed 64-bit
/// arithmetic is exact, including the arithmetic shifts in CRT reconstruction.
struct LimbConvolve;

impl Convolve<SignedLimb, SignedLimb, i64> for LimbConvolve {
    // SAFETY: both representations contain exactly 128 bits, and all-zero bits
    // represent zero in each signed integer lane.
    const T_ZERO: SignedLimb =
        SignedLimb(unsafe { core::mem::transmute::<[i64; 2], v128>([0; 2]) });
    const U_ZERO: i64 = 0;

    #[inline(always)]
    fn halve(val: SignedLimb) -> SignedLimb {
        SignedLimb(i64x2_shr(val.0, 1))
    }

    #[inline(always)]
    fn read(input: SignedLimb) -> SignedLimb {
        input
    }

    #[inline(always)]
    fn parity_dot<const N: usize>(lhs: [SignedLimb; N], rhs: [i64; N]) -> SignedLimb {
        let mut sum = Self::T_ZERO.0;
        for i in 0..N {
            sum = i64x2_add(sum, i64x2_mul(lhs[i].0, i64x2_splat(rhs[i])));
        }
        SignedLimb(sum)
    }

    #[inline(always)]
    fn reduce(z: SignedLimb) -> SignedLimb {
        z
    }
}

#[inline(always)]
fn apply_small_mds<const N: usize>(
    input: [PackedGoldilocksWasmSimd128; N],
    col: [i64; N],
    conv: impl Fn([SignedLimb; N], [i64; N], &mut [SignedLimb]),
) -> [PackedGoldilocksWasmSimd128; N] {
    let mask = u64x2_splat(u32::MAX as u64);
    let low = input.map(|x| SignedLimb(v128_and(x.to_vector(), mask)));
    let high = input.map(|x| SignedLimb(u64x2_shr(x.to_vector(), 32)));
    let mut low_out = [LimbConvolve::T_ZERO; N];
    let mut high_out = [LimbConvolve::T_ZERO; N];
    conv(low, col, &mut low_out);
    conv(high, col, &mut high_out);

    core::array::from_fn(|i| {
        // All coefficients are non-negative and the largest row sum is 371.
        // Both limb outputs are therefore in [0, 371 * (2^32 - 1)], even for
        // non-canonical full-u64 input representatives. Recombine the limbs
        // into the exact output below 2^73 without overflowing either lane.
        let middle = i64x2_add(high_out[i].0, u64x2_shr(low_out[i].0, 32));
        let lo = v128_or(v128_and(low_out[i].0, mask), i64x2_shl(middle, 32));
        let hi = u64x2_shr(middle, 32);

        // Fold 2^64 = 2^32 - 1 (mod P). Here hi <= 370, so the correction
        // is below 2^41 and is canonical, as required by add_canonical.
        let correction = i64x2_sub(i64x2_shl(hi, 32), hi);
        PackedGoldilocksWasmSimd128::from_vector(lo)
            .add_canonical(PackedGoldilocksWasmSimd128::from_vector(correction))
    })
}

impl Permutation<[PackedGoldilocksWasmSimd128; 8]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedGoldilocksWasmSimd128; 8]) -> [PackedGoldilocksWasmSimd128; 8] {
        const COL: [i64; 8] = first_row_to_first_col(&MATRIX_CIRC_MDS_8_SML_ROW);
        apply_small_mds(input, COL, LimbConvolve::conv8)
    }
}

impl MdsPermutation<PackedGoldilocksWasmSimd128, 8> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksWasmSimd128; 12]> for MdsMatrixGoldilocks {
    fn permute(
        &self,
        input: [PackedGoldilocksWasmSimd128; 12],
    ) -> [PackedGoldilocksWasmSimd128; 12] {
        const COL: [i64; 12] = first_row_to_first_col(&MATRIX_CIRC_MDS_12_SML_ROW);
        apply_small_mds(input, COL, LimbConvolve::conv12)
    }
}

impl MdsPermutation<PackedGoldilocksWasmSimd128, 12> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksWasmSimd128; 16]> for MdsMatrixGoldilocks {
    fn permute(
        &self,
        input: [PackedGoldilocksWasmSimd128; 16],
    ) -> [PackedGoldilocksWasmSimd128; 16] {
        const COL: [i64; 16] = first_row_to_first_col(&MATRIX_CIRC_MDS_16_SML_ROW);
        apply_small_mds(input, COL, LimbConvolve::conv16)
    }
}

impl MdsPermutation<PackedGoldilocksWasmSimd128, 16> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksWasmSimd128; 24]> for MdsMatrixGoldilocks {
    fn permute(
        &self,
        input: [PackedGoldilocksWasmSimd128; 24],
    ) -> [PackedGoldilocksWasmSimd128; 24] {
        apply_circulant(&MATRIX_CIRC_MDS_24_GOLDILOCKS, &input)
    }
}

impl MdsPermutation<PackedGoldilocksWasmSimd128, 24> for MdsMatrixGoldilocks {}

#[cfg(test)]
mod tests {
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::{
        Goldilocks, MATRIX_CIRC_MDS_8_SML_ROW, MATRIX_CIRC_MDS_12_SML_ROW,
        MATRIX_CIRC_MDS_16_SML_ROW, MdsMatrixGoldilocks, PackedGoldilocksWasmSimd128,
    };

    fn check<const WIDTH: usize>(lanes: [[Goldilocks; WIDTH]; 2])
    where
        MdsMatrixGoldilocks:
            Permutation<[Goldilocks; WIDTH]> + Permutation<[PackedGoldilocksWasmSimd128; WIDTH]>,
    {
        let packed =
            core::array::from_fn(|i| PackedGoldilocksWasmSimd128([lanes[0][i], lanes[1][i]]));
        let actual = MdsMatrixGoldilocks.permute(packed);
        for (lane, input) in lanes.into_iter().enumerate() {
            let expected = MdsMatrixGoldilocks.permute(input);
            assert_eq!(actual.map(|x| x.0[lane]), expected);

            // Use direct integer matrix multiplication, independently of both the
            // packed and scalar Karatsuba implementations. Width 24's full-size
            // coefficients do not fit in a single unreduced u128 dot product.
            let row: &[i64] = match WIDTH {
                8 => &MATRIX_CIRC_MDS_8_SML_ROW,
                12 => &MATRIX_CIRC_MDS_12_SML_ROW,
                16 => &MATRIX_CIRC_MDS_16_SML_ROW,
                _ => continue,
            };
            for i in 0..WIDTH {
                let sum: u128 = (0..WIDTH)
                    .map(|j| input[j].value as u128 * row[(WIDTH + j - i) % WIDTH] as u128)
                    .sum();
                assert_eq!(
                    actual[i].0[lane],
                    Goldilocks::new((sum % crate::P as u128) as u64)
                );
            }
        }
    }

    macro_rules! test_wasm_mds {
        ($name:ident, $width:literal) => {
            #[test]
            fn $name() {
                let edges = [
                    0,
                    1,
                    (1 << 32) - 1,
                    1 << 32,
                    1 << 63,
                    crate::P - 1,
                    crate::P,
                    crate::P + 1,
                    u64::MAX,
                ];
                for offset in 0..edges.len() {
                    check::<$width>(core::array::from_fn(|lane| {
                        core::array::from_fn(|i| {
                            Goldilocks::new(edges[(offset + i + lane * 3) % edges.len()])
                        })
                    }));
                    check::<$width>([[Goldilocks::new(edges[offset]); $width]; 2]);
                    for position in 0..$width {
                        check::<$width>(core::array::from_fn(|lane| {
                            core::array::from_fn(|i| {
                                Goldilocks::new(if i == (position + lane) % $width {
                                    edges[offset]
                                } else {
                                    0
                                })
                            })
                        }));
                    }
                }
                let mut rng = SmallRng::seed_from_u64(1);
                for _ in 0..64 {
                    check::<$width>(core::array::from_fn(|_| {
                        core::array::from_fn(|_| Goldilocks::new(rng.random()))
                    }));
                }
            }
        };
    }

    fn check_poseidon1<const WIDTH: usize>(
        perm: impl Permutation<[Goldilocks; WIDTH]> + Permutation<[PackedGoldilocksWasmSimd128; WIDTH]>,
    ) {
        let mut rng = SmallRng::seed_from_u64(0x905E1);
        for _ in 0..16 {
            let lanes: [[Goldilocks; WIDTH]; 2] =
                core::array::from_fn(|_| core::array::from_fn(|_| Goldilocks::new(rng.random())));
            let packed =
                core::array::from_fn(|i| PackedGoldilocksWasmSimd128([lanes[0][i], lanes[1][i]]));
            let actual = perm.permute(packed);
            for (lane, input) in lanes.into_iter().enumerate() {
                assert_eq!(actual.map(|x| x.0[lane]), perm.permute(input));
            }
        }
    }

    #[test]
    fn packed_poseidon1_matches_scalar() {
        check_poseidon1(crate::poseidon1::default_goldilocks_poseidon1_8());
        check_poseidon1(crate::poseidon1::default_goldilocks_poseidon1_12());
    }

    test_wasm_mds!(test_wasm_mds_width_8, 8);
    test_wasm_mds!(test_wasm_mds_width_12, 12);
    test_wasm_mds!(test_wasm_mds_width_16, 16);
    test_wasm_mds!(test_wasm_mds_width_24, 24);
}
