use alloc::vec::Vec;

use p3_field::InjectiveMonomial;
use p3_koala_bear::{KoalaBear, MdsMatrixKoalaBear, fp3_cube};
use p3_symmetric::{CryptographicPermutation, Permutation};

use super::xhash::XHash;
use crate::util::{exp_acc, square_n};

pub const XHASH_KB_ALPHA: u64 = 3;
pub const XHASH_KB_WIDTH: usize = 24;
pub const XHASH_KB_CAPACITY: usize = 8;
pub const XHASH_KB_NUM_ROUNDS: usize = 3;

const XHASH_KB_SEED: &str = "XHash-KB:p=2130706433,m=24,c=8,n=3";

/// `ceil(log2(p) / 8) + 1` with `p = 2^31 - 2^24 + 1`.
const BYTES_PER_CONSTANT: usize = 5;

/// XHash over KoalaBear at width 24 with concluding linear layer.
///
/// Uses [`MdsMatrixKoalaBear`] (24x24 circulant) and SHAKE-derived round constants
/// from a fixed seed.
/// The parameter choice (width 24, capacity 8, 3 rounds, base S-box `x^3 / x^{1/3}`,
/// extension S-box `x^3` over `F_p[α] / (α^3 + α + 4)`) mirrors the
/// [XHash-M31](https://eprint.iacr.org/2024/1635) layout with the field's native
/// S-box exponent (`gcd(3, p_KB - 1) = 1`, while every binomial `X^3 - W` is
/// reducible — see [`p3_koala_bear::fp3_mul`]).
#[derive(Clone)]
pub struct XHashKoalaBear {
    inner: XHash<KoalaBear, MdsMatrixKoalaBear, XHASH_KB_WIDTH, XHASH_KB_ALPHA>,
}

impl XHashKoalaBear {
    pub fn from_standard_constants() -> Self {
        let rcs: Vec<KoalaBear> = XHash::<
            KoalaBear,
            MdsMatrixKoalaBear,
            XHASH_KB_WIDTH,
            XHASH_KB_ALPHA,
        >::shake_round_constants(
            XHASH_KB_SEED.as_bytes(),
            XHASH_KB_NUM_ROUNDS,
            BYTES_PER_CONSTANT,
        );
        Self {
            inner: XHash::new(XHASH_KB_NUM_ROUNDS, rcs, MdsMatrixKoalaBear::default()),
        }
    }
}

impl Permutation<[KoalaBear; XHASH_KB_WIDTH]> for XHashKoalaBear {
    fn permute_mut(&self, state: &mut [KoalaBear; XHASH_KB_WIDTH]) {
        let rcs = &self.inner.round_constants;
        for round in 0..self.inner.num_rounds {
            // F sub-round
            self.inner.mds.permute_mut(state);
            for (s, rc) in state.iter_mut().zip(&rcs[3 * round * XHASH_KB_WIDTH..]) {
                *s += *rc;
            }
            state.iter_mut().for_each(|s| *s = s.injective_exp_n());

            // B sub-round
            self.inner.mds.permute_mut(state);
            for (s, rc) in state
                .iter_mut()
                .zip(&rcs[(3 * round + 1) * XHASH_KB_WIDTH..])
            {
                *s += *rc;
            }
            apply_inv_sbox_x3(state);

            // E sub-round (no MDS): eight F_{p^3} triples cubed.
            for (s, rc) in state
                .iter_mut()
                .zip(&rcs[(3 * round + 2) * XHASH_KB_WIDTH..])
            {
                *s += *rc;
            }
            apply_ext_cube(state);
        }

        self.inner.mds.permute_mut(state);
        for (s, rc) in state
            .iter_mut()
            .zip(&rcs[3 * self.inner.num_rounds * XHASH_KB_WIDTH..])
        {
            *s += *rc;
        }
    }
}

impl CryptographicPermutation<[KoalaBear; XHASH_KB_WIDTH]> for XHashKoalaBear {}

/// Apply `x → x^3` over `F_p[α] / (α^3 + α + 4)` to each of the eight triples.
#[inline]
fn apply_ext_cube(state: &mut [KoalaBear; XHASH_KB_WIDTH]) {
    for chunk in state.chunks_mut(3) {
        let triple = [chunk[0], chunk[1], chunk[2]];
        let cubed = fp3_cube(&triple);
        chunk[0] = cubed[0];
        chunk[1] = cubed[1];
        chunk[2] = cubed[2];
    }
}

/// Width-parallel inverse S-box `x → x^(1/3)` over `[KoalaBear; 24]`.
///
/// Computes `x^1420470955` via the same addition chain as
/// [`p3_field::exponentiation::exp_1420470955`].
#[inline]
fn apply_inv_sbox_x3(state: &mut [KoalaBear; XHASH_KB_WIDTH]) {
    // Binary expansion of 1420470955 (29 squares + 7 mults):
    //   1010100101010101010101010101011
    let p1 = *state;

    let p100 = square_n(p1, 2);

    let mut p101 = p100;
    p101.iter_mut().zip(p1).for_each(|(t, x)| *t *= x);

    let p10000 = square_n(p100, 2);

    let mut p10101 = p10000;
    p10101.iter_mut().zip(p101).for_each(|(t, x)| *t *= x);

    let p10101000000 = square_n(p10101, 6);

    let mut p10101010101 = p10101000000;
    p10101010101
        .iter_mut()
        .zip(p10101)
        .for_each(|(t, x)| *t *= x);

    let mut p101010010101 = p10101000000;
    p101010010101
        .iter_mut()
        .zip(p10101010101)
        .for_each(|(t, x)| *t *= x);

    let p101010010101010101010101 = exp_acc::<_, XHASH_KB_WIDTH, 12>(p101010010101, p10101010101);
    let p101010010101010101010101010101 =
        exp_acc::<_, XHASH_KB_WIDTH, 6>(p101010010101010101010101, p10101);
    let p1010100101010101010101010101010 = square_n(p101010010101010101010101010101, 1);

    state
        .iter_mut()
        .zip(p1010100101010101010101010101010)
        .zip(p1)
        .for_each(|((s, a), b)| *s = a * b);
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;

    use super::*;

    #[test]
    fn xhash_koalabear_is_deterministic_and_nontrivial() {
        let xhash = XHashKoalaBear::from_standard_constants();
        let input: [KoalaBear; XHASH_KB_WIDTH] =
            core::array::from_fn(|i| KoalaBear::from_u32(i as u32));
        let a = xhash.permute(input);
        let b = xhash.permute(input);
        assert_eq!(a, b);
        assert_ne!(a, input);
    }

    /// Permutation regression vector for input `[0, 1, …, 23]`.
    ///
    /// **Derived from this implementation.**
    #[test]
    fn xhash_koalabear_test_vector() {
        let xhash = XHashKoalaBear::from_standard_constants();
        let input: [KoalaBear; XHASH_KB_WIDTH] =
            core::array::from_fn(|i| KoalaBear::from_u32(i as u32));
        let expected: [KoalaBear; XHASH_KB_WIDTH] = [
            1_026_990_277,
            261_521_938,
            725_810_584,
            1_942_022_136,
            1_141_513_359,
            218_444_592,
            1_570_121_930,
            1_548_384_314,
            403_842_140,
            1_369_109_572,
            355_329_395,
            607_501_589,
            1_442_570_348,
            777_723_246,
            706_919_842,
            1_913_408_113,
            1_803_569_394,
            1_182_218_242,
            1_133_697_787,
            582_787_368,
            649_933_908,
            999_474_581,
            509_643_209,
            1_802_590_390,
        ]
        .map(KoalaBear::from_u32);
        assert_eq!(xhash.permute(input), expected);
    }
}
