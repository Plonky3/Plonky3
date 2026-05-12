//! Rescue-Prime eXtension (RPX), [eprint 2023/1045](https://eprint.iacr.org/2023/1045).
//!
//! Width-12 instance over Goldilocks. Reuses the MDS matrix and SHAKE-derived
//! ARK constants of the RPO-Goldilocks instance.

use alloc::format;
use alloc::vec::Vec;

use p3_field::extension::CubicTrinomialExtensionField;
use p3_field::{BasedVectorSpace, InjectiveMonomial, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_symmetric::{CryptographicPermutation, Permutation};

use crate::rpo::{
    MdsMatrixRpoGoldilocks, RPO_GOLDILOCKS_ALPHA, RPO_GOLDILOCKS_NUM_ROUNDS, RPO_GOLDILOCKS_WIDTH,
    Rpo, apply_inv_sbox_x7,
};

/// State width of the RPX-Goldilocks permutation (matches RPO).
pub const RPX_GOLDILOCKS_WIDTH: usize = RPO_GOLDILOCKS_WIDTH;

/// Number of RPX-Goldilocks rounds; the schedule below references rounds 0..7.
pub const RPX_GOLDILOCKS_NUM_ROUNDS: usize = RPO_GOLDILOCKS_NUM_ROUNDS;

const ALPHA: u64 = RPO_GOLDILOCKS_ALPHA;
const W: usize = RPX_GOLDILOCKS_WIDTH;

/// `ceil(log2(p) / 8) + 1` with `p = 2^64 - 2^32 + 1`.
const BYTES_PER_CONSTANT: usize = 9;

/// Cubic extension `Goldilocks[X] / (X^3 - X - 1)` used by the (E) round.
type ExtGoldilocks = CubicTrinomialExtensionField<Goldilocks>;

/// RPX over Goldilocks at width 12, per eprint 2023/1045.
///
/// Each call to [`Permutation::permute_mut`] applies the schedule
/// `(FB)(E)(FB)(E)(FB)(E)(M)`:
///
/// - **(FB)** rounds (indices 0, 2, 4): one RPO half — `MDS → +ARK1 → x^7 →
///   MDS → +ARK2 → x^(1/7)`.
/// - **(E)** rounds (indices 1, 3, 5): `+ARK1` followed by raising each of the
///   four 3-element triples of the state to the 7th power in the cubic
///   extension `Goldilocks[X] / (X^3 - X - 1)`.
/// - **(M)** round (index 6): `MDS → +ARK1`.
///
/// The MDS, S-boxes, and round constants are shared with
/// [`crate::rpo::RpoGoldilocks`]; only the per-round structure differs.
#[derive(Clone, Debug)]
pub struct RpxGoldilocks {
    inner: Rpo<Goldilocks, MdsMatrixRpoGoldilocks, W, ALPHA>,
}

impl RpxGoldilocks {
    /// Standard parameter choice from the paper is
    /// `(capacity = 4, security = 128)`.
    pub fn from_standard_constants(capacity: usize, security_level: usize) -> Self {
        // Same SHAKE seed as RPO-Goldilocks: the same ARK table is shared
        // between the two permutations.
        let seed = format!(
            "RPO({},{},{},{})",
            Goldilocks::ORDER_U64,
            W,
            capacity,
            security_level,
        );
        let rcs: Vec<Goldilocks> =
            Rpo::<Goldilocks, MdsMatrixRpoGoldilocks, W, ALPHA>::shake_round_constants(
                seed.as_bytes(),
                RPX_GOLDILOCKS_NUM_ROUNDS,
                BYTES_PER_CONSTANT,
                false,
            );
        Self {
            inner: Rpo::new(RPX_GOLDILOCKS_NUM_ROUNDS, rcs, MdsMatrixRpoGoldilocks),
        }
    }

    /// (FB) round: `MDS → +ARK1 → x^7 → MDS → +ARK2 → x^(1/7)`.
    #[inline]
    fn apply_fb_round(&self, state: &mut [Goldilocks; W], round: usize) {
        let rcs = &self.inner.round_constants;
        self.inner.mds.permute_mut(state);
        for (s, rc) in state.iter_mut().zip(&rcs[2 * round * W..]) {
            *s += *rc;
        }
        state.iter_mut().for_each(|s| *s = s.injective_exp_n());

        self.inner.mds.permute_mut(state);
        for (s, rc) in state.iter_mut().zip(&rcs[(2 * round + 1) * W..]) {
            *s += *rc;
        }
        apply_inv_sbox_x7(state);
    }

    /// (E) round: `+ARK1` and then `x^7` in the cubic extension applied to
    /// each of the four 3-element triples of the state.
    ///
    /// `exp_const_u64::<7>` on the cubic extension uses
    /// `x → x² → x³ → x⁴ → x⁷` (2 squares + 2 mults), matching miden's
    /// hand-rolled `power7` operation count.
    #[inline]
    fn apply_ext_round(&self, state: &mut [Goldilocks; W], round: usize) {
        let rcs = &self.inner.round_constants;
        for (s, rc) in state.iter_mut().zip(&rcs[2 * round * W..]) {
            *s += *rc;
        }
        let [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11] = *state;
        let e0 = ExtGoldilocks::new([s0, s1, s2]).exp_const_u64::<7>();
        let e1 = ExtGoldilocks::new([s3, s4, s5]).exp_const_u64::<7>();
        let e2 = ExtGoldilocks::new([s6, s7, s8]).exp_const_u64::<7>();
        let e3 = ExtGoldilocks::new([s9, s10, s11]).exp_const_u64::<7>();
        let c0 = e0.as_basis_coefficients_slice();
        let c1 = e1.as_basis_coefficients_slice();
        let c2 = e2.as_basis_coefficients_slice();
        let c3 = e3.as_basis_coefficients_slice();
        *state = [
            c0[0], c0[1], c0[2], c1[0], c1[1], c1[2], c2[0], c2[1], c2[2], c3[0], c3[1], c3[2],
        ];
    }

    /// (M) round: `MDS → +ARK1`.
    #[inline]
    fn apply_final_round(&self, state: &mut [Goldilocks; W], round: usize) {
        let rcs = &self.inner.round_constants;
        self.inner.mds.permute_mut(state);
        for (s, rc) in state.iter_mut().zip(&rcs[2 * round * W..]) {
            *s += *rc;
        }
    }
}

impl Permutation<[Goldilocks; W]> for RpxGoldilocks {
    fn permute_mut(&self, state: &mut [Goldilocks; W]) {
        self.apply_fb_round(state, 0);
        self.apply_ext_round(state, 1);
        self.apply_fb_round(state, 2);
        self.apply_ext_round(state, 3);
        self.apply_fb_round(state, 4);
        self.apply_ext_round(state, 5);
        self.apply_final_round(state, 6);
    }
}

impl CryptographicPermutation<[Goldilocks; W]> for RpxGoldilocks {}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;

    use super::*;

    #[test]
    fn rpx_goldilocks_is_deterministic_and_nontrivial() {
        let rpx = RpxGoldilocks::from_standard_constants(4, 128);
        let input: [Goldilocks; W] = core::array::from_fn(|i| Goldilocks::from_u64(i as u64));
        let a = rpx.permute(input);
        let b = rpx.permute(input);
        assert_eq!(a, b);
        assert_ne!(a, input);
    }

    /// Permutation regression vector for input `[0, 1, …, 11]`.
    ///
    /// **Derived from this implementation.** miden-crypto does not publish a
    /// permutation-only RPX test vector — its tests exercise the full
    /// `hash_elements` pipeline — so this vector pins the current output to
    /// catch silent regressions and should be cross-validated against
    /// miden's `RpxPermutation256::apply_permutation` before being relied on
    /// as a spec witness.
    #[test]
    fn rpx_goldilocks_test_vector() {
        let rpx = RpxGoldilocks::from_standard_constants(4, 128);
        let input: [Goldilocks; W] = core::array::from_fn(|i| Goldilocks::from_u64(i as u64));
        let expected: [Goldilocks; W] = [
            3614697924784493998,
            4917065433670799835,
            12893407190838344317,
            16769932886818781879,
            17010299523770013195,
            9826755761378503206,
            1872785960340665977,
            7783788981462778586,
            45778307605882514,
            7437259891664617628,
            17010253034795346176,
            6863075881906649113,
        ]
        .map(Goldilocks::from_u64);
        assert_eq!(rpx.permute(input), expected);
    }
}
