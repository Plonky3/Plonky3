//! MDS permutation for Goldilocks on aarch64.

use core::arch::aarch64::*;
use core::mem::transmute;

use p3_mds::MdsPermutation;
use p3_symmetric::Permutation;

use super::packing::PackedGoldilocksNeon;
use super::utils::{pack_lanes, unpack_lanes};
use crate::{Goldilocks, MdsMatrixGoldilocks};

// ---------------------------------------------------------------------------
// Packed MdsMatrixGoldilocks (delegates to scalar Karatsuba per lane)
// ---------------------------------------------------------------------------

/// Apply the scalar MDS to each lane of a packed NEON state independently.
#[inline]
fn mds_packed<const WIDTH: usize>(
    mds: &MdsMatrixGoldilocks,
    input: &mut [PackedGoldilocksNeon; WIDTH],
) where
    MdsMatrixGoldilocks: Permutation<[Goldilocks; WIDTH]>,
{
    let (mut lane0, mut lane1) = unpack_lanes(input);
    unsafe {
        mds.permute_mut(&mut *(&mut lane0 as *mut [u64; WIDTH] as *mut [Goldilocks; WIDTH]));
        mds.permute_mut(&mut *(&mut lane1 as *mut [u64; WIDTH] as *mut [Goldilocks; WIDTH]));
    }
    pack_lanes(input, &lane0, &lane1);
}

impl Permutation<[PackedGoldilocksNeon; 8]> for MdsMatrixGoldilocks {
    fn permute_mut(&self, input: &mut [PackedGoldilocksNeon; 8]) {
        mds_packed(self, input);
    }
}

impl MdsPermutation<PackedGoldilocksNeon, 8> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksNeon; 12]> for MdsMatrixGoldilocks {
    fn permute_mut(&self, input: &mut [PackedGoldilocksNeon; 12]) {
        mds_packed(self, input);
    }
}

impl MdsPermutation<PackedGoldilocksNeon, 12> for MdsMatrixGoldilocks {}

// ---------------------------------------------------------------------------
// NEON-accelerated circulant MDS (16-bit chunk multiply-accumulate)
// ---------------------------------------------------------------------------

/// Goldilocks identity: `2^64 ≡ 2^32 − 1 (mod P)`.
const EPSILON_U32: u32 = 0xffffffff;

/// Reduce two accumulated 4×32-bit chunk vectors back to Goldilocks field
/// elements. Each `uint32x4_t` holds four 32-bit accumulators representing
/// the four 16-bit chunks of a Goldilocks element:
///
/// ```text
///     elem = c[0] + c[1]·2¹⁶ + c[2]·2³² + c[3]·2⁴⁸
/// ```
///
/// Returns two Goldilocks values packed in a `uint64x2_t`.
///
/// Ported from plonky2.
#[inline(always)]
unsafe fn mds_reduce([cumul_a, cumul_b]: [uint32x4_t; 2]) -> uint64x2_t {
    unsafe {
        let mut lo = vreinterpretq_u64_u32(vuzp1q_u32(cumul_a, cumul_b));
        let mut hi = vreinterpretq_u64_u32(vuzp2q_u32(cumul_a, cumul_b));

        hi = vsraq_n_u64::<16>(hi, lo);
        lo = vsliq_n_u64::<16>(lo, hi);

        let top = {
            let hi_u8 = vreinterpretq_u8_u64(hi);
            let top_idx =
                transmute::<[u8; 8], uint8x8_t>([0x06, 0x07, 0xff, 0xff, 0x0e, 0x0f, 0xff, 0xff]);
            let top_u8 = vqtbl1_u8(hi_u8, top_idx);
            vreinterpret_u32_u8(top_u8)
        };

        let adj_lo = vmlal_n_u32(lo, top, EPSILON_U32);
        let wraparound_mask = vcgtq_u64(lo, adj_lo);
        vsraq_n_u64::<32>(adj_lo, wraparound_mask)
    }
}

/// NEON-accelerated width-8 circulant MDS.
///
/// Circulant first row: `[7, 1, 3, 8, 8, 3, 4, 9]`
/// (matches `MATRIX_CIRC_MDS_8_SML_ROW`).
#[inline(always)]
pub unsafe fn mds_neon_w8(state: &[u64; 8]) -> [u64; 8] {
    unsafe {
        const ROW: [u32; 8] = [7, 1, 3, 8, 8, 3, 4, 9];

        const M: [[u32; 8]; 8] = {
            let mut m = [[0u32; 8]; 8];
            let mut i = 0;
            while i < 8 {
                let mut j = 0;
                while j < 8 {
                    m[i][j] = ROW[(j + 8 - i) % 8];
                    j += 1;
                }
                i += 1;
            }
            m
        };

        let c: [uint32x4_t; 8] = core::array::from_fn(|i| vmovl_u16(vcreate_u16(state[i])));

        let mut res = [0u64; 8];

        let mut pair = 0;
        while pair < 4 {
            let i0 = 2 * pair;
            let i1 = i0 + 1;

            let mut a0 = vdupq_n_u32(0);
            let mut a1 = vdupq_n_u32(0);

            let mut j = 0;
            while j < 8 {
                a0 = vmlaq_n_u32(a0, c[j], M[i0][j]);
                a1 = vmlaq_n_u32(a1, c[j], M[i1][j]);
                j += 1;
            }

            let r = mds_reduce([a0, a1]);
            res[i0] = vgetq_lane_u64::<0>(r);
            res[i1] = vgetq_lane_u64::<1>(r);
            pair += 1;
        }

        res
    }
}

/// NEON-accelerated width-12 circulant MDS.
///
/// Circulant first row: `[1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10]`
/// (matches `MATRIX_CIRC_MDS_12_SML_ROW`).
#[inline(always)]
pub unsafe fn mds_neon_w12(state: &[u64; 12]) -> [u64; 12] {
    unsafe {
        const ROW: [u32; 12] = [1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10];

        const M: [[u32; 12]; 12] = {
            let mut m = [[0u32; 12]; 12];
            let mut i = 0;
            while i < 12 {
                let mut j = 0;
                while j < 12 {
                    m[i][j] = ROW[(j + 12 - i) % 12];
                    j += 1;
                }
                i += 1;
            }
            m
        };

        let c: [uint32x4_t; 12] = core::array::from_fn(|i| vmovl_u16(vcreate_u16(state[i])));

        let mut res = [0u64; 12];

        let mut pair = 0;
        while pair < 6 {
            let i0 = 2 * pair;
            let i1 = i0 + 1;

            let mut a0 = vdupq_n_u32(0);
            let mut a1 = vdupq_n_u32(0);

            let mut j = 0;
            while j < 12 {
                a0 = vmlaq_n_u32(a0, c[j], M[i0][j]);
                a1 = vmlaq_n_u32(a1, c[j], M[i1][j]);
                j += 1;
            }

            let r = mds_reduce([a0, a1]);
            res[i0] = vgetq_lane_u64::<0>(r);
            res[i1] = vgetq_lane_u64::<1>(r);
            pair += 1;
        }

        res
    }
}

/// NEON-accelerated MDS wrapper for use with the generic Poseidon1.
///
/// Zero-sized type that implements `Permutation<[Goldilocks; 8]>` and
/// `Permutation<[Goldilocks; 12]>` using the NEON chunk technique. Plugs
/// into `Poseidon1ExternalLayerGeneric` to accelerate full-round MDS while
/// keeping LLVM-optimized partial rounds from the generic Poseidon1.
#[derive(Clone, Debug, Default)]
pub struct MdsNeonGoldilocks;

impl Permutation<[Goldilocks; 8]> for MdsNeonGoldilocks {
    fn permute_mut(&self, state: &mut [Goldilocks; 8]) {
        let raw = unsafe { &*(state as *const [Goldilocks; 8] as *const [u64; 8]) };
        let result = unsafe { mds_neon_w8(raw) };
        *unsafe { &mut *(state as *mut [Goldilocks; 8] as *mut [u64; 8]) } = result;
    }
}

impl Permutation<[Goldilocks; 12]> for MdsNeonGoldilocks {
    fn permute_mut(&self, state: &mut [Goldilocks; 12]) {
        let raw = unsafe { &*(state as *const [Goldilocks; 12] as *const [u64; 12]) };
        let result = unsafe { mds_neon_w12(raw) };
        *unsafe { &mut *(state as *mut [Goldilocks; 12] as *mut [u64; 12]) } = result;
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeField64;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::{Goldilocks, MdsMatrixGoldilocks, PackedGoldilocksNeon};

    type F = Goldilocks;

    // -- Packed MdsMatrixGoldilocks tests --

    macro_rules! test_neon_mds {
        ($name:ident, $width:literal) => {
            #[test]
            fn $name() {
                let mut rng = SmallRng::seed_from_u64(1);
                let mds = MdsMatrixGoldilocks;

                let input: [Goldilocks; $width] = rng.random();
                let expected = mds.permute(input);

                let packed_input = input.map(Into::<PackedGoldilocksNeon>::into);
                let packed_output = mds.permute(packed_input);

                let neon_output = packed_output.map(|x| x.0[0]);
                assert_eq!(neon_output, expected);
            }
        };
    }

    test_neon_mds!(test_neon_mds_width_8, 8);
    test_neon_mds!(test_neon_mds_width_12, 12);

    // -- NEON MDS correctness tests --

    #[test]
    fn test_mds_neon_w8_matches_karatsuba() {
        let mds = MdsMatrixGoldilocks;
        let mut rng = SmallRng::seed_from_u64(42);

        for _ in 0..100 {
            let input: [F; 8] = rng.random();
            let expected = mds.permute(input);

            let raw: [u64; 8] = input.map(|x| x.as_canonical_u64());
            let result = unsafe { super::mds_neon_w8(&raw) };

            for i in 0..8 {
                assert_eq!(
                    F::new(result[i]).as_canonical_u64(),
                    expected[i].as_canonical_u64(),
                    "NEON MDS w8 mismatch at index {i}"
                );
            }
        }
    }

    #[test]
    fn test_mds_neon_w12_matches_karatsuba() {
        let mds = MdsMatrixGoldilocks;
        let mut rng = SmallRng::seed_from_u64(43);

        for _ in 0..100 {
            let input: [F; 12] = rng.random();
            let expected = mds.permute(input);

            let raw: [u64; 12] = input.map(|x| x.as_canonical_u64());
            let result = unsafe { super::mds_neon_w12(&raw) };

            for i in 0..12 {
                assert_eq!(
                    F::new(result[i]).as_canonical_u64(),
                    expected[i].as_canonical_u64(),
                    "NEON MDS w12 mismatch at index {i}"
                );
            }
        }
    }

    #[test]
    fn test_mds_neon_boundary_w8() {
        let mds = MdsMatrixGoldilocks;
        let p_minus_1 = F::ORDER_U64 - 1;

        for &val in &[0u64, 1, p_minus_1] {
            let input: [F; 8] = [F::new(val); 8];
            let expected = mds.permute(input);

            let raw = [val; 8];
            let result = unsafe { super::mds_neon_w8(&raw) };

            for i in 0..8 {
                assert_eq!(
                    F::new(result[i]).as_canonical_u64(),
                    expected[i].as_canonical_u64(),
                    "NEON MDS w8 boundary mismatch at index {i} for value {val}"
                );
            }
        }
    }

    #[test]
    fn test_mds_neon_boundary_w12() {
        let mds = MdsMatrixGoldilocks;
        let p_minus_1 = F::ORDER_U64 - 1;

        for &val in &[0u64, 1, p_minus_1] {
            let input: [F; 12] = [F::new(val); 12];
            let expected = mds.permute(input);

            let raw = [val; 12];
            let result = unsafe { super::mds_neon_w12(&raw) };

            for i in 0..12 {
                assert_eq!(
                    F::new(result[i]).as_canonical_u64(),
                    expected[i].as_canonical_u64(),
                    "NEON MDS w12 boundary mismatch at index {i} for value {val}"
                );
            }
        }
    }
}
