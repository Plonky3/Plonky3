//! Width-24 MDS for Mersenne31 using BabyBear's `MATRIX_CIRC_MDS_24` row.
//!
//! Plonky3's `MdsMatrixMersenne31` doesn't ship a width-24 variant. As a
//! comparison point against the paper's `RpoCirMds24`, we apply BabyBear's
//! 24×24 circulant column (verified MDS over BabyBear and KoalaBear) over
//! Mersenne31, using Plonky3's Karatsuba + Barrett-style convolution.
//!
//! WARNING: MDS-ness of this column over GF(2³¹−1) has NOT been verified.
//! Use this only for performance comparison, NOT in production.

use p3_field::integers::QuotientMap;
use p3_field::PrimeField32;
use p3_mds::karatsuba_convolution::Convolve;
use p3_mds::util::first_row_to_first_col;
use p3_mds::MdsPermutation;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::Permutation;

/// First row of BabyBear's 24×24 circulant MDS, copied from `baby-bear/src/mds.rs`.
const BB_MATRIX_CIRC_MDS_24_ROW: [i64; 24] = [
    0x2D0AAAAB, 0x64850517, 0x17F5551D, 0x04ECBEB5, 0x6D91A8D5, 0x60703026, 0x18D6F3CA, 0x729601A7,
    0x77CDA9E2, 0x3C0F5038, 0x26D52A61, 0x0360405D, 0x68FC71C8, 0x2495A71D, 0x5D57AFC2, 0x1689DD98,
    0x3C2C3DBE, 0x0C23DC41, 0x0524C7F2, 0x6BE4DF69, 0x0A6E572C, 0x5C7790FA, 0x17E118F6, 0x0878A07F,
];

/// Re-implementation of Plonky3's private `LargeConvolveMersenne31` for use outside
/// the `p3_mersenne_31` crate. Matches the upstream `parity_dot` / `reduce` exactly.
struct LargeConvolveMersenne31;

impl Convolve<Mersenne31, i64, i64> for LargeConvolveMersenne31 {
    const T_ZERO: i64 = 0;
    const U_ZERO: i64 = 0;

    #[inline(always)]
    fn halve(val: i64) -> i64 {
        val >> 1
    }

    #[inline(always)]
    fn read(input: Mersenne31) -> i64 {
        input.as_canonical_u32() as i64
    }

    #[inline]
    fn parity_dot<const N: usize>(u: [i64; N], v: [i64; N]) -> i64 {
        // Bound: N^2 * 2^62 for N <= 64 fits in i128.
        let mut dp = 0i128;
        for i in 0..N {
            dp += u[i] as i128 * v[i] as i128;
        }

        const LOWMASK: i128 = (1 << 42) - 1;
        const HIGHMASK: i128 = !LOWMASK;

        let low_bits = (dp & LOWMASK) as i64;
        let high_bits = ((dp & HIGHMASK) >> 31) as i64;

        low_bits + high_bits
    }

    #[inline]
    fn reduce(z: i64) -> Mersenne31 {
        debug_assert!(z > -(1i64 << 49));
        debug_assert!(z < (1i64 << 49));

        const MASK: i64 = (1 << 31) - 1;
        // SAFETY: 0 <= z & MASK < 2^31 = P + 1, and Mersenne31 accepts [0, P].
        let low_bits = unsafe { Mersenne31::from_canonical_unchecked((z & MASK) as u32) };

        let high_bits = ((z >> 31) & MASK) as i32;
        let sign_bits = (z >> 62) as i32;

        let high = unsafe { Mersenne31::from_canonical_unchecked((high_bits + sign_bits) as u32) };
        low_bits + high
    }
}

/// Width-24 MDS over Mersenne31 driven by BabyBear's MDS row.
#[derive(Clone, Debug, Default)]
pub struct Mds24M31BBCol;

impl Permutation<[Mersenne31; 24]> for Mds24M31BBCol {
    fn permute(&self, input: [Mersenne31; 24]) -> [Mersenne31; 24] {
        const COL: [i64; 24] = first_row_to_first_col(&BB_MATRIX_CIRC_MDS_24_ROW);
        LargeConvolveMersenne31::apply(
            input,
            COL,
            <LargeConvolveMersenne31 as Convolve<Mersenne31, i64, i64>>::conv24,
        )
    }

    fn permute_mut(&self, input: &mut [Mersenne31; 24]) {
        *input = self.permute(*input);
    }
}

impl MdsPermutation<Mersenne31, 24> for Mds24M31BBCol {}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn changes_state() {
        let mut rng = StdRng::seed_from_u64(13);
        let input: [Mersenne31; 24] =
            core::array::from_fn(|_| Mersenne31::new(rng.random::<u32>() % Mersenne31::ORDER_U32));
        let output = Mds24M31BBCol.permute(input);
        assert_ne!(output, input);
    }

    #[test]
    fn deterministic() {
        let input: [Mersenne31; 24] =
            core::array::from_fn(|i| Mersenne31::new((i as u32 + 1) * 31));
        let o1 = Mds24M31BBCol.permute(input);
        let o2 = Mds24M31BBCol.permute(input);
        assert_eq!(o1, o2);
    }
}
