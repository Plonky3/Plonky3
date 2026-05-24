use alloc::vec::Vec;

use p3_field::extension::BinomialExtensionField;
use p3_field::integers::QuotientMap;
use p3_field::{BasedVectorSpace, InjectiveMonomial, PrimeCharacteristicRing, PrimeField32};
use p3_mds::MdsPermutation;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{CryptographicPermutation, Permutation};

use super::xhash::XHash;
use crate::util::{exp_acc, square_n};

pub const XHASH_M31_ALPHA: u64 = 5;
pub const XHASH_M31_WIDTH: usize = 24;
pub const XHASH_M31_CAPACITY: usize = 8;
pub const XHASH_M31_NUM_ROUNDS: usize = 3;

const XHASH_M31_SEED: &str = "XHash-M31:p=2147483647,m=24,c=8,n=3";

/// `ceil(log2(p) / 8) + 1` with `p = 2^31 - 1`.
const BYTES_PER_CONSTANT: usize = 5;

/// First row of the 32x32 circulant from eprint 2024/1635, Appendix A.3.
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
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
];

/// XHash-M31 MDS: the 24x24 top-left sub-block of the 32x32 circulant defined
/// by [`MDS_FIRST_ROW_32`]. Same matrix used by RPO-M31.
#[derive(Clone, Copy, Default)]
pub struct MdsMatrixXHashMersenne31;

impl Permutation<[Mersenne31; XHASH_M31_WIDTH]> for MdsMatrixXHashMersenne31 {
    fn permute_mut(&self, state: &mut [Mersenne31; XHASH_M31_WIDTH]) {
        let lifted: [u64; XHASH_M31_WIDTH] =
            core::array::from_fn(|i| state[i].as_canonical_u32() as u64);

        for row in 0..XHASH_M31_WIDTH {
            // Each (coeff, input) is < 2^31 so the product fits in u64;
            // the sum of 24 such products is < 24 * 2^62 < 2^67, so we
            // accumulate in u128 and reduce once at the end.
            let mut acc: u128 = 0;
            for col in 0..XHASH_M31_WIDTH {
                let coeff = MDS_FIRST_ROW_32[(col + 32 - row) % 32] as u64;
                acc += (coeff * lifted[col]) as u128;
            }
            // SAFETY: `acc < 2^67`.
            state[row] = unsafe { reduce_u128_to_m31(acc) };
        }
    }
}

impl MdsPermutation<Mersenne31, XHASH_M31_WIDTH> for MdsMatrixXHashMersenne31 {}

/// Reduce a value in `[0, 2^67)` modulo `p = 2^31 - 1`.
///
/// The bound `v < 2^67` is the caller's responsibility; for
/// `MdsMatrixXHashMersenne31` it follows from `24 * (2^31 - 1)^2 < 2^67`.
#[inline]
unsafe fn reduce_u128_to_m31(v: u128) -> Mersenne31 {
    const M: u64 = (1u64 << 31) - 1;
    let lo = (v as u64) & M;
    let mid = ((v >> 31) as u64) & M;
    let hi = (v >> 62) as u64;
    let s = lo + mid + hi;
    let s = (s & M) + (s >> 31);
    let canonical = s.min(s.wrapping_sub(M)) as u32;
    // SAFETY: `canonical < p`.
    unsafe { <Mersenne31 as QuotientMap<u32>>::from_canonical_unchecked(canonical) }
}

/// Cubic extension `Mersenne31[X] / (X^3 - 5)`, native P3 type for M31.
type ExtMersenne31 = BinomialExtensionField<Mersenne31, 3>;

/// XHash over Mersenne-31 at width 24 with concluding linear layer
/// (eprint 2024/1635).
#[derive(Clone)]
pub struct XHashMersenne31 {
    inner: XHash<Mersenne31, MdsMatrixXHashMersenne31, XHASH_M31_WIDTH, XHASH_M31_ALPHA>,
}

impl XHashMersenne31 {
    pub fn from_standard_constants() -> Self {
        let rcs: Vec<Mersenne31> = XHash::<
            Mersenne31,
            MdsMatrixXHashMersenne31,
            XHASH_M31_WIDTH,
            XHASH_M31_ALPHA,
        >::shake_round_constants(
            XHASH_M31_SEED.as_bytes(),
            XHASH_M31_NUM_ROUNDS,
            BYTES_PER_CONSTANT,
        );
        Self {
            inner: XHash::new(XHASH_M31_NUM_ROUNDS, rcs, MdsMatrixXHashMersenne31),
        }
    }
}

impl Permutation<[Mersenne31; XHASH_M31_WIDTH]> for XHashMersenne31 {
    fn permute_mut(&self, state: &mut [Mersenne31; XHASH_M31_WIDTH]) {
        let rcs = &self.inner.round_constants;
        for round in 0..self.inner.num_rounds {
            // F sub-round
            self.inner.mds.permute_mut(state);
            for (s, rc) in state.iter_mut().zip(&rcs[3 * round * XHASH_M31_WIDTH..]) {
                *s += *rc;
            }
            state.iter_mut().for_each(|s| *s = s.injective_exp_n());

            // B sub-round
            self.inner.mds.permute_mut(state);
            for (s, rc) in state
                .iter_mut()
                .zip(&rcs[(3 * round + 1) * XHASH_M31_WIDTH..])
            {
                *s += *rc;
            }
            apply_inv_sbox_x5(state);

            // E sub-round (no MDS)
            for (s, rc) in state
                .iter_mut()
                .zip(&rcs[(3 * round + 2) * XHASH_M31_WIDTH..])
            {
                *s += *rc;
            }
            apply_ext_pow5(state);
        }

        // Concluding linear step.
        self.inner.mds.permute_mut(state);
        for (s, rc) in state
            .iter_mut()
            .zip(&rcs[3 * self.inner.num_rounds * XHASH_M31_WIDTH..])
        {
            *s += *rc;
        }
    }
}

impl CryptographicPermutation<[Mersenne31; XHASH_M31_WIDTH]> for XHashMersenne31 {}

/// Apply `x → x^5` over `F_p[X] / (X^3 - 5)` to each of the eight 3-element
/// triples of the state.
#[inline]
fn apply_ext_pow5(state: &mut [Mersenne31; XHASH_M31_WIDTH]) {
    for chunk in state.chunks_mut(3) {
        let triple = [chunk[0], chunk[1], chunk[2]];
        let raised = ExtMersenne31::new(triple).exp_const_u64::<5>();
        let coeffs = raised.as_basis_coefficients_slice();
        chunk[0] = coeffs[0];
        chunk[1] = coeffs[1];
        chunk[2] = coeffs[2];
    }
}

/// Width-parallel inverse S-box `x → x^(1/5)` over `[Mersenne31; 24]`.
///
/// Computes `x^1717986917` via the same addition chain as
/// [`p3_field::exponentiation::exp_1717986917`].
#[inline]
fn apply_inv_sbox_x5(state: &mut [Mersenne31; XHASH_M31_WIDTH]) {
    // Binary expansion of 1717986917 (30 squares + 7 mults):
    //   1100110011001100110011001100101
    let p1 = *state;

    let mut p10 = p1;
    p10.iter_mut().for_each(|t| *t = t.square());

    let mut p11 = p10;
    p11.iter_mut().zip(p1).for_each(|(t, x)| *t *= x);

    let mut p101 = p10;
    p101.iter_mut().zip(p11).for_each(|(t, x)| *t *= x);

    let p110011 = exp_acc::<_, XHASH_M31_WIDTH, 4>(p11, p11);
    let p11001100000000 = square_n(p110011, 8);

    let mut p11001100110011 = p11001100000000;
    p11001100110011
        .iter_mut()
        .zip(p110011)
        .for_each(|(t, x)| *t *= x);

    let p1100110011001100110011 =
        exp_acc::<_, XHASH_M31_WIDTH, 8>(p11001100000000, p11001100110011);
    let p11001100110011001100110011 =
        exp_acc::<_, XHASH_M31_WIDTH, 4>(p1100110011001100110011, p11);
    let p1100110011001100110011001100000 = square_n(p11001100110011001100110011, 5);

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
    fn xhash_mersenne31_is_deterministic_and_nontrivial() {
        let xhash = XHashMersenne31::from_standard_constants();
        let input: [Mersenne31; XHASH_M31_WIDTH] =
            core::array::from_fn(|i| Mersenne31::from_u32(i as u32));
        let a = xhash.permute(input);
        let b = xhash.permute(input);
        assert_eq!(a, b);
        assert_ne!(a, input);
    }

    /// Permutation regression vector for input `[0, 1, …, 23]`.
    ///
    /// **Derived from this implementation.**
    #[test]
    fn xhash_mersenne31_test_vector() {
        let xhash = XHashMersenne31::from_standard_constants();
        let input: [Mersenne31; XHASH_M31_WIDTH] =
            core::array::from_fn(|i| Mersenne31::from_u32(i as u32));
        let expected: [Mersenne31; XHASH_M31_WIDTH] = [
            691_639_269,
            667_975_737,
            1_340_979_208,
            695_559_657,
            1_102_163_247,
            1_420_187_086,
            1_987_519_437,
            126_712_269,
            744_259_186,
            1_431_215_180,
            6_399_862,
            1_547_541_411,
            1_128_442_818,
            781_221_948,
            41_841_471,
            82_840_779,
            397_027_057,
            12_418_849,
            1_508_573_376,
            604_130_137,
            524_643_095,
            1_944_381_296,
            366_196_219,
            268_654_718,
        ]
        .map(Mersenne31::from_u32);
        assert_eq!(xhash.permute(input), expected);
    }
}
