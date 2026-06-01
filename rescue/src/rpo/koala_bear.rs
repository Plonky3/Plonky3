use alloc::vec::Vec;

use p3_field::InjectiveMonomial;
use p3_koala_bear::{KoalaBear, MdsMatrixKoalaBear};
use p3_symmetric::{CryptographicPermutation, Permutation};

use super::Rpo;
use crate::util::{exp_acc, square_n};

pub const RPO_KB_ALPHA: u64 = 3;
pub const RPO_KB_WIDTH: usize = 24;
pub const RPO_KB_CAPACITY: usize = 8;
pub const RPO_KB_NUM_ROUNDS: usize = 7;

/// Plonky3-internal seed (no published RPO-KoalaBear reference exists). Round
/// constants derived from this string will not interop with any external
/// implementation; the format mirrors the M31 seed shape (see the 31-bit
/// RPO paper, eprint 2024/1635).
const RPO_KB_SEED: &str = "RPO-KB:p=2130706433,m=24,c=8,n=7";

/// `ceil(log2(p) / 8) + 1` with `p = 2^31 - 2^24 + 1`.
const BYTES_PER_CONSTANT: usize = 5;

/// RPO over KoalaBear at width 24 with concluding linear layer.
///
/// Uses Plonky3's native [`MdsMatrixKoalaBear`] (24x24 circulant) and SHAKE-derived
/// round constants from a fixed seed. No published KoalaBear RPO instance exists; the
/// parameter choice (width 24, capacity 8, 7 rounds, x^3 / x^{1/3} S-boxes) mirrors
/// the [RPO-M31](https://eprint.iacr.org/2024/1635) layout for a comparable small
/// field, using the field's native S-box exponent.
#[derive(Clone)]
pub struct RpoKoalaBear {
    inner: Rpo<KoalaBear, MdsMatrixKoalaBear, RPO_KB_WIDTH, RPO_KB_ALPHA>,
}

impl RpoKoalaBear {
    pub fn from_standard_constants() -> Self {
        let rcs: Vec<KoalaBear> =
            Rpo::<KoalaBear, MdsMatrixKoalaBear, RPO_KB_WIDTH, RPO_KB_ALPHA>::shake_round_constants(
                RPO_KB_SEED.as_bytes(),
                RPO_KB_NUM_ROUNDS,
                BYTES_PER_CONSTANT,
                true,
            );
        Self {
            inner: Rpo::new_with_final_linear_layer(
                RPO_KB_NUM_ROUNDS,
                rcs,
                MdsMatrixKoalaBear::default(),
            ),
        }
    }
}

impl Permutation<[KoalaBear; RPO_KB_WIDTH]> for RpoKoalaBear {
    fn permute_mut(&self, state: &mut [KoalaBear; RPO_KB_WIDTH]) {
        let rcs = &self.inner.round_constants;
        for round in 0..self.inner.num_rounds {
            self.inner.mds.permute_mut(state);
            for (s, rc) in state.iter_mut().zip(&rcs[2 * round * RPO_KB_WIDTH..]) {
                *s += *rc;
            }
            state.iter_mut().for_each(|s| *s = s.injective_exp_n());

            self.inner.mds.permute_mut(state);
            for (s, rc) in state.iter_mut().zip(&rcs[(2 * round + 1) * RPO_KB_WIDTH..]) {
                *s += *rc;
            }
            apply_inv_sbox_x3(state);
        }

        // Concluding linear step: one extra MDS + ARK after the rounds.
        self.inner.mds.permute_mut(state);
        for (s, rc) in state
            .iter_mut()
            .zip(&rcs[2 * self.inner.num_rounds * RPO_KB_WIDTH..])
        {
            *s += *rc;
        }
    }
}

impl CryptographicPermutation<[KoalaBear; RPO_KB_WIDTH]> for RpoKoalaBear {}

/// Width-parallel inverse S-box `x -> x^(1/3)` over `[KoalaBear; 24]`.
///
/// Computes `x^1420470955` via the same addition chain as
/// [`p3_field::exponentiation::exp_1420470955`], but applied across all 24
/// lanes step-by-step. Each step issues 24 independent multiplications,
/// exposing 24-way ILP to the CPU.
#[inline]
fn apply_inv_sbox_x3(state: &mut [KoalaBear; RPO_KB_WIDTH]) {
    // Binary expansion of 1420470955 (29 squares + 7 mults):
    //   1010100101010101010101010101011
    let p1 = *state;

    let p100 = square_n::<_, RPO_KB_WIDTH, 2>(p1);

    let mut p101 = p100;
    p101.iter_mut().zip(p1).for_each(|(t, x)| *t *= x);

    let p10000 = square_n::<_, RPO_KB_WIDTH, 2>(p100);

    let mut p10101 = p10000;
    p10101.iter_mut().zip(p101).for_each(|(t, x)| *t *= x);

    let p10101000000 = square_n::<_, RPO_KB_WIDTH, 6>(p10101);

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

    let p101010010101010101010101 = exp_acc::<_, RPO_KB_WIDTH, 12>(p101010010101, p10101010101);
    let p101010010101010101010101010101 =
        exp_acc::<_, RPO_KB_WIDTH, 6>(p101010010101010101010101, p10101);
    let p1010100101010101010101010101010 =
        square_n::<_, RPO_KB_WIDTH, 1>(p101010010101010101010101010101);

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
    use proptest::prelude::*;

    use super::*;

    /// Permutation regression vector for input `[0, 1, …, 23]`.
    ///
    /// **Derived from this implementation.** No published reference exists; this
    /// vector pins the current output to catch silent regressions.
    #[test]
    fn rpo_koalabear_test_vector() {
        let rpo = RpoKoalaBear::from_standard_constants();

        let input: [KoalaBear; RPO_KB_WIDTH] =
            core::array::from_fn(|i| KoalaBear::from_u32(i as u32));
        let expected: [KoalaBear; RPO_KB_WIDTH] = [
            1_407_896_285,
            1_062_342_513,
            1_822_322_044,
            864_303_338,
            510_786_778,
            1_251_794_877,
            757_025_745,
            547_176_545,
            675_022_842,
            1_465_657_099,
            1_078_871_545,
            1_998_725_156,
            1_434_537_809,
            1_874_670_136,
            1_612_157_256,
            891_444_931,
            965_969_718,
            275_772_368,
            221_105_388,
            770_314_268,
            912_214_035,
            63_895_892,
            1_738_877_181,
            1_622_784_127,
        ]
        .map(KoalaBear::from_u32);

        assert_eq!(rpo.permute(input), expected);
    }

    #[test]
    fn inv_sbox_x3_round_trips_through_x3() {
        // For all 24 lanes, ((x^3))^(1/3) == x.
        let mut state: [KoalaBear; RPO_KB_WIDTH] =
            core::array::from_fn(|i| KoalaBear::from_u32((i as u32 + 1) * 37));
        let original = state;
        state.iter_mut().for_each(|s| *s = s.injective_exp_n());
        apply_inv_sbox_x3(&mut state);
        assert_eq!(state, original);
    }

    proptest! {
        #[test]
        fn proptest_inv_sbox_x3_round_trips(seeds in prop::array::uniform24(any::<u32>())) {
            let mut state: [KoalaBear; RPO_KB_WIDTH] =
                core::array::from_fn(|i| KoalaBear::from_u32(seeds[i]));
            let original = state;
            state.iter_mut().for_each(|s| *s = s.injective_exp_n());
            apply_inv_sbox_x3(&mut state);
            prop_assert_eq!(state, original);
        }
    }
}
