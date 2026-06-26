use alloc::vec::Vec;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use p3_baby_bear::PackedBabyBearNeon;
use p3_baby_bear::{BabyBear, MdsMatrixBabyBear};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use p3_field::PackedValue;
use p3_field::{InjectiveMonomial, PrimeCharacteristicRing};
use p3_symmetric::{CryptographicPermutation, Permutation};

use super::Rpo;
use crate::util::{exp_acc, square_n};

pub const RPO_BB_ALPHA: u64 = 7;
pub const RPO_BB_WIDTH: usize = 24;
pub const RPO_BB_CAPACITY: usize = 8;
pub const RPO_BB_NUM_ROUNDS: usize = 7;

/// Plonky3-internal seed (no published RPO-BabyBear reference exists). Round
/// constants derived from this string will not interop with any external
/// implementation; the format mirrors the M31 seed shape (see the
/// 31-bit RPO paper, eprint 2024/1635).
const RPO_BB_SEED: &str = "RPO-BB:p=2013265921,m=24,c=8,n=7";

/// `ceil(log2(p) / 8) + 1` with `p = 15 * 2^27 + 1`.
const BYTES_PER_CONSTANT: usize = 5;

/// RPO over BabyBear at width 24 with concluding linear layer.
///
/// Uses Plonky3's native [`MdsMatrixBabyBear`] (24x24 circulant) and SHAKE-derived
/// round constants from a fixed seed. No published BabyBear RPO instance exists; the
/// parameter choice (width 24, capacity 8, 7 rounds, x^7 / x^{1/7} S-boxes) mirrors
/// the [RPO-M31](https://eprint.iacr.org/2024/1635) layout for a comparable small
/// field.
#[derive(Clone)]
pub struct RpoBabyBear {
    inner: Rpo<BabyBear, MdsMatrixBabyBear, RPO_BB_WIDTH, RPO_BB_ALPHA>,
}

impl RpoBabyBear {
    pub fn from_standard_constants() -> Self {
        let rcs: Vec<BabyBear> =
            Rpo::<BabyBear, MdsMatrixBabyBear, RPO_BB_WIDTH, RPO_BB_ALPHA>::shake_round_constants(
                RPO_BB_SEED.as_bytes(),
                RPO_BB_NUM_ROUNDS,
                BYTES_PER_CONSTANT,
                true,
            );
        Self {
            inner: Rpo::new_with_final_linear_layer(
                RPO_BB_NUM_ROUNDS,
                rcs,
                MdsMatrixBabyBear::default(),
            ),
        }
    }
}

impl Permutation<[BabyBear; RPO_BB_WIDTH]> for RpoBabyBear {
    fn permute_mut(&self, state: &mut [BabyBear; RPO_BB_WIDTH]) {
        let rcs = &self.inner.round_constants;
        for round in 0..self.inner.num_rounds {
            self.inner.mds.permute_mut(state);
            for (s, rc) in state.iter_mut().zip(&rcs[2 * round * RPO_BB_WIDTH..]) {
                *s += *rc;
            }
            state.iter_mut().for_each(|s| *s = s.injective_exp_n());

            self.inner.mds.permute_mut(state);
            for (s, rc) in state.iter_mut().zip(&rcs[(2 * round + 1) * RPO_BB_WIDTH..]) {
                *s += *rc;
            }
            apply_inv_sbox_x7(state);
        }

        // Concluding linear step: one extra MDS + ARK after the rounds.
        self.inner.mds.permute_mut(state);
        for (s, rc) in state
            .iter_mut()
            .zip(&rcs[2 * self.inner.num_rounds * RPO_BB_WIDTH..])
        {
            *s += *rc;
        }
    }
}

impl CryptographicPermutation<[BabyBear; RPO_BB_WIDTH]> for RpoBabyBear {}

/// Inverse S-box `x -> x^(1/7)` over `[BabyBear; 24]`.
///
/// On aarch64 the 24-element state reinterprets as six packed NEON vectors so
/// the addition chain runs four lanes per multiply; on other targets it runs
/// lane-by-lane.
#[inline]
fn apply_inv_sbox_x7(state: &mut [BabyBear; RPO_BB_WIDTH]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // `RPO_BB_WIDTH` is a multiple of the NEON packing width (4), so the
        // whole state reinterprets as packed vectors with no remainder.
        let packed: &mut [PackedBabyBearNeon; RPO_BB_WIDTH / 4] =
            PackedBabyBearNeon::pack_slice_mut(state)
                .try_into()
                .unwrap();
        inv_sbox_x7(packed);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    inv_sbox_x7(state);
}

/// Width-parallel inverse S-box `x -> x^(1/7)` over `[R; N]`.
///
/// Computes `x^1725656503` via the same addition chain as
/// [`p3_field::exponentiation::exp_1725656503`], applied across all `N`
/// lanes step-by-step. Each step issues `N` independent multiplications,
/// exposing `N`-way ILP to the CPU.
#[inline]
fn inv_sbox_x7<R: PrimeCharacteristicRing + Copy, const N: usize>(state: &mut [R; N]) {
    // Binary expansion of 1725656503 (29 squares + 8 mults):
    //   1100110110110110110110110110111
    let p1 = *state;

    let p10 = square_n::<_, N, 1>(p1);

    let mut p11 = p10;
    p11.iter_mut().zip(p1).for_each(|(t, x)| *t *= x);

    let p110 = square_n::<_, N, 1>(p11);

    let mut p111 = p110;
    p111.iter_mut().zip(p1).for_each(|(t, x)| *t *= x);

    let p11000 = square_n::<_, N, 2>(p110);

    let mut p11011 = p11000;
    p11011.iter_mut().zip(p11).for_each(|(t, x)| *t *= x);

    let p11000000 = square_n::<_, N, 3>(p11000);

    let mut p11011011 = p11000000;
    p11011011.iter_mut().zip(p11011).for_each(|(t, x)| *t *= x);

    let mut p110011011 = p11011011;
    p110011011
        .iter_mut()
        .zip(p11000000)
        .for_each(|(t, x)| *t *= x);

    let p110011011011011011 = exp_acc::<_, N, 9>(p110011011, p11011011);
    let p110011011011011011011011011 = exp_acc::<_, N, 9>(p110011011011011011, p11011011);
    let p1100110110110110110110110110000 = square_n::<_, N, 4>(p110011011011011011011011011);

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
    use proptest::prelude::*;

    use super::*;

    /// Permutation regression vector for input `[0, 1, …, 23]`.
    ///
    /// **Derived from this implementation.** No published reference exists; this
    /// vector pins the current output to catch silent regressions.
    #[test]
    fn rpo_babybear_test_vector() {
        let rpo = RpoBabyBear::from_standard_constants();

        let input: [BabyBear; RPO_BB_WIDTH] =
            core::array::from_fn(|i| BabyBear::from_u32(i as u32));
        let expected: [BabyBear; RPO_BB_WIDTH] = [
            580_889_464,
            812_545_993,
            1_435_256_485,
            1_944_190_928,
            468_280_959,
            1_574_957_037,
            614_259_202,
            1_971_827_593,
            1_157_818_138,
            41_725_352,
            1_454_051_006,
            1_975_269_624,
            1_052_998_898,
            1_151_877_439,
            1_238_988_248,
            973_164_623,
            1_378_588_581,
            1_290_093_470,
            599_149_080,
            819_216_820,
            1_015_689_941,
            1_980_884_825,
            607_611_746,
            918_354_105,
        ]
        .map(BabyBear::from_u32);

        assert_eq!(rpo.permute(input), expected);
    }

    #[test]
    fn inv_sbox_x7_round_trips_through_x7() {
        // For all 24 lanes, ((x^7))^(1/7) == x.
        let mut state: [BabyBear; RPO_BB_WIDTH] =
            core::array::from_fn(|i| BabyBear::from_u32((i as u32 + 1) * 37));
        let original = state;
        state.iter_mut().for_each(|s| *s = s.injective_exp_n());
        apply_inv_sbox_x7(&mut state);
        assert_eq!(state, original);
    }

    proptest! {
        #[test]
        fn proptest_inv_sbox_x7_round_trips(seeds in prop::array::uniform24(any::<u32>())) {
            let mut state: [BabyBear; RPO_BB_WIDTH] =
                core::array::from_fn(|i| BabyBear::from_u32(seeds[i]));
            let original = state;
            state.iter_mut().for_each(|s| *s = s.injective_exp_n());
            apply_inv_sbox_x7(&mut state);
            prop_assert_eq!(state, original);
        }
    }
}
