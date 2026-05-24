use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, MdsMatrixBabyBear, fp2_pow7};
use p3_field::InjectiveMonomial;
use p3_symmetric::{CryptographicPermutation, Permutation};

use super::xhash::XHash;
use crate::util::{exp_acc, square_n};

pub const XHASH_BB_ALPHA: u64 = 7;
pub const XHASH_BB_WIDTH: usize = 24;
pub const XHASH_BB_CAPACITY: usize = 8;
pub const XHASH_BB_NUM_ROUNDS: usize = 3;

const XHASH_BB_SEED: &str = "XHash-BB:p=2013265921,m=24,c=8,n=3";

/// `ceil(log2(p) / 8) + 1` with `p = 15 * 2^27 + 1`.
const BYTES_PER_CONSTANT: usize = 5;

/// XHash over BabyBear at width 24 with concluding linear layer.
///
/// Uses [`MdsMatrixBabyBear`] (24x24 circulant) and SHAKE-derived round constants
/// from a fixed seed.
/// The parameter choice (width 24, capacity 8, 3 rounds, base S-box `x^7 / x^{1/7}`,
/// extension S-box `x^7` over `F_p[α] / (α^2 - 11)`) mirrors the
/// [XHash-M31](https://eprint.iacr.org/2024/1635) layout, with the smallest
/// extension that admits `x^7` over BabyBear.
#[derive(Clone)]
pub struct XHashBabyBear {
    inner: XHash<BabyBear, MdsMatrixBabyBear, XHASH_BB_WIDTH, XHASH_BB_ALPHA>,
}

impl XHashBabyBear {
    pub fn from_standard_constants() -> Self {
        let rcs: Vec<BabyBear> =
            XHash::<BabyBear, MdsMatrixBabyBear, XHASH_BB_WIDTH, XHASH_BB_ALPHA>::shake_round_constants(
                XHASH_BB_SEED.as_bytes(),
                XHASH_BB_NUM_ROUNDS,
                BYTES_PER_CONSTANT,
            );
        Self {
            inner: XHash::new(XHASH_BB_NUM_ROUNDS, rcs, MdsMatrixBabyBear::default()),
        }
    }
}

impl Permutation<[BabyBear; XHASH_BB_WIDTH]> for XHashBabyBear {
    fn permute_mut(&self, state: &mut [BabyBear; XHASH_BB_WIDTH]) {
        let rcs = &self.inner.round_constants;
        for round in 0..self.inner.num_rounds {
            // F sub-round
            self.inner.mds.permute_mut(state);
            for (s, rc) in state.iter_mut().zip(&rcs[3 * round * XHASH_BB_WIDTH..]) {
                *s += *rc;
            }
            state.iter_mut().for_each(|s| *s = s.injective_exp_n());

            // B sub-round
            self.inner.mds.permute_mut(state);
            for (s, rc) in state
                .iter_mut()
                .zip(&rcs[(3 * round + 1) * XHASH_BB_WIDTH..])
            {
                *s += *rc;
            }
            apply_inv_sbox_x7(state);

            // E sub-round (no MDS): twelve F_{p^2} pairs raised to the 7th power.
            for (s, rc) in state
                .iter_mut()
                .zip(&rcs[(3 * round + 2) * XHASH_BB_WIDTH..])
            {
                *s += *rc;
            }
            apply_ext_pow7(state);
        }

        self.inner.mds.permute_mut(state);
        for (s, rc) in state
            .iter_mut()
            .zip(&rcs[3 * self.inner.num_rounds * XHASH_BB_WIDTH..])
        {
            *s += *rc;
        }
    }
}

impl CryptographicPermutation<[BabyBear; XHASH_BB_WIDTH]> for XHashBabyBear {}

/// Apply `x → x^7` over `F_p[α] / (α^2 - 11)` to each of the twelve pairs.
#[inline]
fn apply_ext_pow7(state: &mut [BabyBear; XHASH_BB_WIDTH]) {
    for chunk in state.chunks_mut(2) {
        let pair = [chunk[0], chunk[1]];
        let raised = fp2_pow7(&pair);
        chunk[0] = raised[0];
        chunk[1] = raised[1];
    }
}

/// Width-parallel inverse S-box `x → x^(1/7)` over `[BabyBear; 24]`.
///
/// Computes `x^1725656503` via the same addition chain as
/// [`p3_field::exponentiation::exp_1725656503`].
#[inline]
fn apply_inv_sbox_x7(state: &mut [BabyBear; XHASH_BB_WIDTH]) {
    // Binary expansion of 1725656503 (29 squares + 8 mults):
    //   1100110110110110110110110110111
    let p1 = *state;

    let p10 = square_n(p1, 1);

    let mut p11 = p10;
    p11.iter_mut().zip(p1).for_each(|(t, x)| *t *= x);

    let p110 = square_n(p11, 1);

    let mut p111 = p110;
    p111.iter_mut().zip(p1).for_each(|(t, x)| *t *= x);

    let p11000 = square_n(p110, 2);

    let mut p11011 = p11000;
    p11011.iter_mut().zip(p11).for_each(|(t, x)| *t *= x);

    let p11000000 = square_n(p11000, 3);

    let mut p11011011 = p11000000;
    p11011011.iter_mut().zip(p11011).for_each(|(t, x)| *t *= x);

    let mut p110011011 = p11011011;
    p110011011
        .iter_mut()
        .zip(p11000000)
        .for_each(|(t, x)| *t *= x);

    let p110011011011011011 = exp_acc::<_, XHASH_BB_WIDTH, 9>(p110011011, p11011011);
    let p110011011011011011011011011 =
        exp_acc::<_, XHASH_BB_WIDTH, 9>(p110011011011011011, p11011011);
    let p1100110110110110110110110110000 = square_n(p110011011011011011011011011, 4);

    state
        .iter_mut()
        .zip(p1100110110110110110110110110000)
        .zip(p111)
        .for_each(|((s, a), b)| *s = a * b);
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;

    use super::*;

    #[test]
    fn xhash_babybear_is_deterministic_and_nontrivial() {
        let xhash = XHashBabyBear::from_standard_constants();
        let input: [BabyBear; XHASH_BB_WIDTH] =
            core::array::from_fn(|i| BabyBear::from_u32(i as u32));
        let a = xhash.permute(input);
        let b = xhash.permute(input);
        assert_eq!(a, b);
        assert_ne!(a, input);
    }

    /// Permutation regression vector for input `[0, 1, …, 23]`.
    ///
    /// **Derived from this implementation.**
    #[test]
    fn xhash_babybear_test_vector() {
        let xhash = XHashBabyBear::from_standard_constants();
        let input: [BabyBear; XHASH_BB_WIDTH] =
            core::array::from_fn(|i| BabyBear::from_u32(i as u32));
        let expected: [BabyBear; XHASH_BB_WIDTH] = [
            203_046_743,
            334_878_070,
            1_072_742_678,
            442_108_873,
            187_118_059,
            1_481_616_589,
            1_978_250_593,
            597_938_921,
            719_613_617,
            1_037_191_809,
            1_840_838_112,
            350_800_105,
            1_092_409_497,
            738_199_758,
            541_294_062,
            761_138_510,
            1_728_623_348,
            513_441_700,
            2_000_810_333,
            81_879_219,
            66_174_463,
            1_855_801_502,
            725_575_542,
            315_913_414,
        ]
        .map(BabyBear::from_u32);
        assert_eq!(xhash.permute(input), expected);
    }
}
