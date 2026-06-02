use alloc::vec::Vec;

use p3_field::integers::QuotientMap;
use p3_field::{InjectiveMonomial, PrimeCharacteristicRing, PrimeField32};
use p3_mds::MdsPermutation;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{CryptographicPermutation, Permutation};

use super::Rpo;
use crate::util::{exp_acc, square_n};

pub const RPO_M31_ALPHA: u64 = 5;
pub const RPO_M31_WIDTH: usize = 24;
pub const RPO_M31_CAPACITY: usize = 8;
pub const RPO_M31_NUM_ROUNDS: usize = 7;

// The SHAKE-256-derived round constants — taken from the
// [AbdelStark/rpo-xhash-m31](https://github.com/AbdelStark/rpo-xhash-m31)
// reference implementation.
const RPO_M31_SEED: &str = "RPO‑M31:p=2147483647,m=24,c=8,n=7";

/// `ceil(log2(p) / 8) + 1` with `p = 2^31 - 1`.
const BYTES_PER_CONSTANT: usize = 5;

/// First row of the 32x32 circulant whose top-left 24x24 sub-block is the
/// RPO-M31 MDS matrix. Source: eprint 2024/1635, Appendix A.3.
const MDS_FIRST_ROW_32: [u32; 32] = [
    185_870_542,
    2_144_994_796,
    1_696_461_115,
    215_190_769,
    930_115_258,
    766_567_118,
    2_003_379_079,
    1_770_558_586,
    1_779_722_644,
    434_368_282,
    289_154_277,
    1_979_813_463,
    1_436_360_233,
    1_342_944_808,
    63_026_005,
    903_393_155,
    1_512_525_948,
    105_409_451,
    1_072_974_295,
    979_558_870,
    436_105_640,
    2_126_764_826,
    1_981_550_821,
    636_196_459,
    645_360_517,
    412_540_024,
    1_649_351_985,
    1_485_803_845,
    53_244_687,
    719_457_988,
    270_924_307,
    82_564_914,
];

/// RPO-M31 MDS: the 24x24 top-left sub-block of the 32x32 circulant defined
/// by `MDS_FIRST_ROW_32`.
///
/// Not itself circulant, so applied as a dense matrix-vector product. Inputs
/// and coefficients fit in `u32`; their products in `u64`; the row sum of 24
/// products in `u128`. Reduction back to M31 happens once per row.
#[derive(Clone, Copy, Debug, Default)]
pub struct MdsMatrixRpoMersenne31;

impl Permutation<[Mersenne31; RPO_M31_WIDTH]> for MdsMatrixRpoMersenne31 {
    fn permute_mut(&self, state: &mut [Mersenne31; RPO_M31_WIDTH]) {
        let lifted: [u64; RPO_M31_WIDTH] =
            core::array::from_fn(|i| state[i].as_canonical_u32() as u64);

        for row in 0..RPO_M31_WIDTH {
            // Each (coeff, input) is < 2^31 so the product fits in u64; the
            // sum of 24 such products is < 24 * 2^62 < 2^67, so we accumulate
            // in u128 and reduce once at the end.
            let mut acc: u128 = 0;
            for col in 0..RPO_M31_WIDTH {
                let coeff = MDS_FIRST_ROW_32[(col + 32 - row) % 32] as u64;
                acc += (coeff * lifted[col]) as u128;
            }
            // SAFETY: `acc` < 2^67.
            state[row] = unsafe { reduce_u128_to_m31(acc) };
        }
    }
}

impl MdsPermutation<Mersenne31, RPO_M31_WIDTH> for MdsMatrixRpoMersenne31 {}

/// Reduce a value in `[0, 2^67)` modulo `p = 2^31 - 1` using the identity
/// `2^31 ≡ 1 (mod p)`.
///
/// The bound `v < 2^67` is the caller's responsibility; for `MdsMatrixRpoMersenne31`
/// it follows from `24 * (2^31 - 1)^2 < 2^67`.
#[inline]
unsafe fn reduce_u128_to_m31(v: u128) -> Mersenne31 {
    const M: u64 = (1u64 << 31) - 1;
    // Three 31-bit chunks: top one is at most 5 bits, so the sum is < 2^32.
    let lo = (v as u64) & M;
    let mid = ((v >> 31) as u64) & M;
    let hi = (v >> 62) as u64;
    let s = lo + mid + hi;
    // One more `2^31 ≡ 1` reduction, then conditional subtract.
    let s = (s & M) + (s >> 31);
    let canonical = s.min(s.wrapping_sub(M)) as u32;
    // SAFETY: the conditional subtract above leaves `canonical < p = 2^31 - 1`.
    unsafe { <Mersenne31 as QuotientMap<u32>>::from_canonical_unchecked(canonical) }
}

/// RPO over Mersenne-31 at width 24 with concluding linear layer
/// (eprint 2024/1635).
#[derive(Clone, Debug)]
pub struct RpoMersenne31 {
    inner: Rpo<Mersenne31, MdsMatrixRpoMersenne31, RPO_M31_WIDTH, RPO_M31_ALPHA>,
}

impl RpoMersenne31 {
    pub fn from_standard_constants() -> Self {
        let rcs: Vec<Mersenne31> = Rpo::<
            Mersenne31,
            MdsMatrixRpoMersenne31,
            RPO_M31_WIDTH,
            RPO_M31_ALPHA,
        >::shake_round_constants(
            RPO_M31_SEED.as_bytes(),
            RPO_M31_NUM_ROUNDS,
            BYTES_PER_CONSTANT,
            true,
        );
        Self {
            inner: Rpo::new_with_final_linear_layer(
                RPO_M31_NUM_ROUNDS,
                rcs,
                MdsMatrixRpoMersenne31,
            ),
        }
    }
}

impl Permutation<[Mersenne31; RPO_M31_WIDTH]> for RpoMersenne31 {
    fn permute_mut(&self, state: &mut [Mersenne31; RPO_M31_WIDTH]) {
        let rcs = &self.inner.round_constants;
        for round in 0..self.inner.num_rounds {
            self.inner.mds.permute_mut(state);
            for (s, rc) in state.iter_mut().zip(&rcs[2 * round * RPO_M31_WIDTH..]) {
                *s += *rc;
            }
            state.iter_mut().for_each(|s| *s = s.injective_exp_n());

            self.inner.mds.permute_mut(state);
            for (s, rc) in state
                .iter_mut()
                .zip(&rcs[(2 * round + 1) * RPO_M31_WIDTH..])
            {
                *s += *rc;
            }
            apply_inv_sbox_x5(state);
        }

        // Concluding linear step: one extra MDS + ARK after the rounds.
        self.inner.mds.permute_mut(state);
        for (s, rc) in state
            .iter_mut()
            .zip(&rcs[2 * self.inner.num_rounds * RPO_M31_WIDTH..])
        {
            *s += *rc;
        }
    }
}

impl CryptographicPermutation<[Mersenne31; RPO_M31_WIDTH]> for RpoMersenne31 {}

/// Width-parallel inverse S-box `x -> x^(1/5)` over `[Mersenne31; 24]`.
///
/// Computes `x^1717986917` via the same addition chain as
/// [`p3_field::exponentiation::exp_1717986917`], but applied across all 24
/// lanes step-by-step. Each step issues 24 independent multiplications,
/// exposing 24-way ILP to the CPU.
#[inline]
fn apply_inv_sbox_x5(state: &mut [Mersenne31; RPO_M31_WIDTH]) {
    // Binary expansion of 1717986917 (30 squares + 7 mults):
    //   1100110011001100110011001100101
    let p1 = *state;

    let mut p10 = p1;
    p10.iter_mut().for_each(|t| *t = t.square());

    let mut p11 = p10;
    p11.iter_mut().zip(p1).for_each(|(t, x)| *t *= x);

    let mut p101 = p10;
    p101.iter_mut().zip(p11).for_each(|(t, x)| *t *= x);

    let p110011 = exp_acc::<_, RPO_M31_WIDTH, 4>(p11, p11);
    let p11001100000000 = square_n::<_, RPO_M31_WIDTH, 8>(p110011);

    let mut p11001100110011 = p11001100000000;
    p11001100110011
        .iter_mut()
        .zip(p110011)
        .for_each(|(t, x)| *t *= x);

    let p1100110011001100110011 = exp_acc::<_, RPO_M31_WIDTH, 8>(p11001100000000, p11001100110011);
    let p11001100110011001100110011 = exp_acc::<_, RPO_M31_WIDTH, 4>(p1100110011001100110011, p11);
    let p1100110011001100110011001100000 =
        square_n::<_, RPO_M31_WIDTH, 5>(p11001100110011001100110011);

    state
        .iter_mut()
        .zip(p1100110011001100110011001100000)
        .zip(p101)
        .for_each(|((s, a), b)| *s = a * b);
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;

    use super::*;

    #[test]
    fn rpo_mersenne31_test_vector() {
        let rpo = RpoMersenne31::from_standard_constants();

        let input: [Mersenne31; RPO_M31_WIDTH] =
            core::array::from_fn(|i| Mersenne31::from_u32(i as u32));
        let expected: [Mersenne31; RPO_M31_WIDTH] = [
            1_990_425_063,
            95_513_650,
            1_492_148_912,
            1_455_268_556,
            347_571_427,
            1_892_690_094,
            34_080_484,
            1_175_631_837,
            1_348_619_901,
            1_096_114_017,
            310_913_313,
            1_912_324_205,
            609_442_899,
            2_112_777_835,
            1_331_189_849,
            507_241_525,
            1_800_223_977,
            568_712_449,
            2_123_164_950,
            86_025_361,
            1_585_828_474,
            1_334_334_486,
            188_486_534,
            1_147_991_035,
        ]
        .map(Mersenne31::from_u32);

        assert_eq!(rpo.permute(input), expected);
    }

    #[test]
    fn mds_rpo_mersenne31_first_column_matches_first_row_constant() {
        // MDS * e_0 must equal the first column of the underlying 32x32
        // circulant, restricted to the first 24 rows.
        let mut state: [Mersenne31; RPO_M31_WIDTH] = [Mersenne31::ZERO; RPO_M31_WIDTH];
        state[0] = Mersenne31::ONE;
        MdsMatrixRpoMersenne31.permute_mut(&mut state);
        for i in 0..RPO_M31_WIDTH {
            let expected = Mersenne31::from_u32(MDS_FIRST_ROW_32[(32 - i) % 32]);
            assert_eq!(state[i], expected, "mismatch at row {i}");
        }
    }
}
