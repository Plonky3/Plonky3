use alloc::format;
use alloc::vec::Vec;

use p3_field::extension::CubicTrinomialExtensionField;
use p3_field::{BasedVectorSpace, InjectiveMonomial, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, SmallConvolveGoldilocks};
use p3_mds::MdsPermutation;
use p3_mds::karatsuba_convolution::Convolve;
use p3_mds::util::first_row_to_first_col;
use p3_symmetric::{CryptographicPermutation, Permutation};

use super::xhash::XHash;
use crate::util::exp_acc;

pub const XHASH_GOLDILOCKS_ALPHA: u64 = 7;
pub const XHASH_GOLDILOCKS_WIDTH: usize = 12;
pub const XHASH_GOLDILOCKS_NUM_ROUNDS: usize = 3;

/// `ceil(log2(p) / 8) + 1` with `p = 2^64 - 2^32 + 1`.
const BYTES_PER_CONSTANT: usize = 9;

/// First row of the width-12 circulant MDS from eprint 2022/1577. Reused by
/// the XHash-Goldilocks construction of [eprint 2023/1045](https://eprint.iacr.org/2023/1045).
const MDS_12_FIRST_ROW: [i64; XHASH_GOLDILOCKS_WIDTH] = [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8];

/// MDS matrix for the XHash-Goldilocks instance, identical to the
/// RPO-Goldilocks matrix specified in eprint 2022/1577.
#[derive(Clone, Copy, Default)]
pub struct MdsMatrixXHashGoldilocks;

impl Permutation<[Goldilocks; XHASH_GOLDILOCKS_WIDTH]> for MdsMatrixXHashGoldilocks {
    fn permute(
        &self,
        input: [Goldilocks; XHASH_GOLDILOCKS_WIDTH],
    ) -> [Goldilocks; XHASH_GOLDILOCKS_WIDTH] {
        const COL: [i64; XHASH_GOLDILOCKS_WIDTH] = first_row_to_first_col(&MDS_12_FIRST_ROW);
        SmallConvolveGoldilocks::apply(input, COL, SmallConvolveGoldilocks::conv12)
    }

    fn permute_mut(&self, input: &mut [Goldilocks; XHASH_GOLDILOCKS_WIDTH]) {
        *input = self.permute(*input);
    }
}

impl MdsPermutation<Goldilocks, XHASH_GOLDILOCKS_WIDTH> for MdsMatrixXHashGoldilocks {}

/// Cubic extension `Goldilocks[X] / (X^3 - X - 1)` used by the extension S-box,
/// matching the RPX construction.
type ExtGoldilocks = CubicTrinomialExtensionField<Goldilocks>;

/// XHash over Goldilocks at width 12 ("RPX" of eprint 2023/1045).
#[derive(Clone)]
pub struct XHashGoldilocks {
    inner:
        XHash<Goldilocks, MdsMatrixXHashGoldilocks, XHASH_GOLDILOCKS_WIDTH, XHASH_GOLDILOCKS_ALPHA>,
}

impl XHashGoldilocks {
    /// Standard parameter choice from the paper is `(capacity = 4, security = 128)`.
    pub fn from_standard_constants(capacity: usize, security_level: usize) -> Self {
        let seed = format!(
            "XHash({},{},{},{})",
            Goldilocks::ORDER_U64,
            XHASH_GOLDILOCKS_WIDTH,
            capacity,
            security_level,
        );
        let rcs: Vec<Goldilocks> = XHash::<
            Goldilocks,
            MdsMatrixXHashGoldilocks,
            XHASH_GOLDILOCKS_WIDTH,
            XHASH_GOLDILOCKS_ALPHA,
        >::shake_round_constants(
            seed.as_bytes(),
            XHASH_GOLDILOCKS_NUM_ROUNDS,
            BYTES_PER_CONSTANT,
        );
        Self {
            inner: XHash::new(XHASH_GOLDILOCKS_NUM_ROUNDS, rcs, MdsMatrixXHashGoldilocks),
        }
    }
}

impl Permutation<[Goldilocks; XHASH_GOLDILOCKS_WIDTH]> for XHashGoldilocks {
    fn permute_mut(&self, state: &mut [Goldilocks; XHASH_GOLDILOCKS_WIDTH]) {
        let rcs = &self.inner.round_constants;
        for round in 0..self.inner.num_rounds {
            // F sub-round: MDS → +ARK → x^7
            self.inner.mds.permute_mut(state);
            for (s, rc) in state
                .iter_mut()
                .zip(&rcs[3 * round * XHASH_GOLDILOCKS_WIDTH..])
            {
                *s += *rc;
            }
            state.iter_mut().for_each(|s| *s = s.injective_exp_n());

            // B sub-round: MDS → +ARK → x^(1/7)
            self.inner.mds.permute_mut(state);
            for (s, rc) in state
                .iter_mut()
                .zip(&rcs[(3 * round + 1) * XHASH_GOLDILOCKS_WIDTH..])
            {
                *s += *rc;
            }
            apply_inv_sbox_x7(state);

            // E sub-round: +ARK → x^7 in cubic extension (no MDS).
            for (s, rc) in state
                .iter_mut()
                .zip(&rcs[(3 * round + 2) * XHASH_GOLDILOCKS_WIDTH..])
            {
                *s += *rc;
            }
            apply_ext_pow7(state);
        }

        // Concluding linear step: MDS → +ARK.
        self.inner.mds.permute_mut(state);
        for (s, rc) in state
            .iter_mut()
            .zip(&rcs[3 * self.inner.num_rounds * XHASH_GOLDILOCKS_WIDTH..])
        {
            *s += *rc;
        }
    }
}

impl CryptographicPermutation<[Goldilocks; XHASH_GOLDILOCKS_WIDTH]> for XHashGoldilocks {}

/// `(s0, s1, s2, …, s9, s10, s11) → s = [E(s0..2)^7, E(s3..5)^7, E(s6..8)^7, E(s9..11)^7]`
/// where `E(a, b, c) = a + bX + cX^2 ∈ Goldilocks[X] / (X^3 - X - 1)`.
#[inline]
fn apply_ext_pow7(state: &mut [Goldilocks; XHASH_GOLDILOCKS_WIDTH]) {
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

/// Width-parallel inverse S-box `x → x^(1/7)` over `[Goldilocks; 12]`.
///
/// Computes `x^10540996611094048183` via the same addition chain as
/// [`p3_field::exponentiation::exp_10540996611094048183`].
#[inline]
fn apply_inv_sbox_x7(state: &mut [Goldilocks; XHASH_GOLDILOCKS_WIDTH]) {
    // Binary expansion of 10540996611094048183 (63 squares + 8 mults):
    //   1001001001001001001001001001000110110110110110110110110110110111
    let mut t1 = *state;
    t1.iter_mut().for_each(|t| *t = t.square());
    let mut t2 = t1;
    t2.iter_mut().for_each(|t| *t = t.square());
    let t3 = exp_acc::<_, XHASH_GOLDILOCKS_WIDTH, 3>(t2, t2);
    let t4 = exp_acc::<_, XHASH_GOLDILOCKS_WIDTH, 6>(t3, t3);
    let t5 = exp_acc::<_, XHASH_GOLDILOCKS_WIDTH, 12>(t4, t4);
    let t6 = exp_acc::<_, XHASH_GOLDILOCKS_WIDTH, 6>(t5, t3);
    let t7 = exp_acc::<_, XHASH_GOLDILOCKS_WIDTH, 31>(t6, t6);

    for (i, s) in state.iter_mut().enumerate() {
        let a = (t7[i].square() * t6[i]).square().square();
        let b = t1[i] * t2[i] * *s;
        *s = a * b;
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;

    use super::*;

    #[test]
    fn xhash_goldilocks_is_deterministic_and_nontrivial() {
        let xhash = XHashGoldilocks::from_standard_constants(4, 128);
        let input: [Goldilocks; XHASH_GOLDILOCKS_WIDTH] =
            core::array::from_fn(|i| Goldilocks::from_u64(i as u64));
        let a = xhash.permute(input);
        let b = xhash.permute(input);
        assert_eq!(a, b);
        assert_ne!(a, input);
    }

    /// Permutation regression vector for input `[0, 1, …, 11]`.
    ///
    /// **Derived from this implementation.** No published RPX vector exists
    /// at the permutation level; this pins the current output to catch
    /// silent regressions.
    #[test]
    fn xhash_goldilocks_test_vector() {
        let xhash = XHashGoldilocks::from_standard_constants(4, 128);
        let input: [Goldilocks; XHASH_GOLDILOCKS_WIDTH] =
            core::array::from_fn(|i| Goldilocks::from_u64(i as u64));
        let expected: [Goldilocks; XHASH_GOLDILOCKS_WIDTH] = [
            5_390_430_308_793_032_119,
            6_524_326_288_766_165_104,
            431_956_421_529_044_799,
            1_339_068_497_270_367_993,
            11_896_154_916_681_774_533,
            14_507_257_167_887_892_029,
            747_836_442_406_064_804,
            15_975_802_802_377_619_008,
            12_976_470_140_382_469_367,
            7_029_730_082_753_525_827,
            203_513_988_405_600_405,
            6_540_786_288_571_718_884,
        ]
        .map(Goldilocks::from_u64);
        assert_eq!(xhash.permute(input), expected);
    }
}
