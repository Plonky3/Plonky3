use alloc::format;
use alloc::vec::Vec;

use p3_field::{InjectiveMonomial, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, SmallConvolveGoldilocks};
use p3_mds::MdsPermutation;
use p3_mds::karatsuba_convolution::Convolve;
use p3_mds::util::first_row_to_first_col;
use p3_symmetric::{CryptographicPermutation, Permutation};

use super::Rpo;
use crate::util::exp_acc;

pub const RPO_GOLDILOCKS_ALPHA: u64 = 7;
pub const RPO_GOLDILOCKS_WIDTH: usize = 12;
pub const RPO_GOLDILOCKS_CAPACITY: usize = 4;
pub const RPO_GOLDILOCKS_NUM_ROUNDS: usize = 7;
pub const RPO_GOLDILOCKS_SECURITY_LEVEL: usize = 128;

/// `ceil(log2(p) / 8) + 1` with `p = 2^64 - 2^32 + 1`.
const BYTES_PER_CONSTANT: usize = 9;

/// First row of the width-12 circulant MDS from eprint 2022/1577.
const MDS_12_FIRST_ROW: [i64; 12] = [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8];

/// MDS matrix specified by the RPO paper for the Goldilocks instance. This
/// is a different matrix from `p3_goldilocks::MdsMatrixGoldilocks`.
///
/// The first row sums to well below `2^51`, so we apply the matrix as a
/// length-12 cyclic convolution via [`SmallConvolveGoldilocks`].
// TODO: Consider adding support for FFT-based MDS.
#[derive(Clone, Copy, Debug, Default)]
pub struct MdsMatrixRpoGoldilocks;

impl Permutation<[Goldilocks; RPO_GOLDILOCKS_WIDTH]> for MdsMatrixRpoGoldilocks {
    fn permute(
        &self,
        input: [Goldilocks; RPO_GOLDILOCKS_WIDTH],
    ) -> [Goldilocks; RPO_GOLDILOCKS_WIDTH] {
        const COL: [i64; RPO_GOLDILOCKS_WIDTH] = first_row_to_first_col(&MDS_12_FIRST_ROW);
        SmallConvolveGoldilocks::apply(input, COL, SmallConvolveGoldilocks::conv12)
    }

    fn permute_mut(&self, input: &mut [Goldilocks; RPO_GOLDILOCKS_WIDTH]) {
        *input = self.permute(*input);
    }
}

impl MdsPermutation<Goldilocks, RPO_GOLDILOCKS_WIDTH> for MdsMatrixRpoGoldilocks {}

/// RPO over Goldilocks at width 12 (eprint 2022/1577).
#[derive(Clone, Debug)]
pub struct RpoGoldilocks {
    inner: Rpo<Goldilocks, MdsMatrixRpoGoldilocks, RPO_GOLDILOCKS_WIDTH, RPO_GOLDILOCKS_ALPHA>,
}

impl RpoGoldilocks {
    /// Uses the paper's `(capacity = 4, security = 128)` parameter choice.
    pub fn from_standard_constants() -> Self {
        // Seed format `"RPO(p,m,c,λ)"` follows the original RPO paper
        // (eprint 2022/1577 §2.2). This matches Miden-crypto's RPO over
        // Goldilocks, so the test vector below is interoperable. The 31-bit
        // RPO variants use a different seed shape because the later paper
        // (eprint 2024/1635) and its reference implementation use the
        // `"RPO‑<field>:p=…,m=…,c=…,n=…"` form.
        let seed = format!(
            "RPO({},{},{},{})",
            Goldilocks::ORDER_U64,
            RPO_GOLDILOCKS_WIDTH,
            RPO_GOLDILOCKS_CAPACITY,
            RPO_GOLDILOCKS_SECURITY_LEVEL,
        );
        let rcs: Vec<Goldilocks> = Rpo::<
            Goldilocks,
            MdsMatrixRpoGoldilocks,
            RPO_GOLDILOCKS_WIDTH,
            RPO_GOLDILOCKS_ALPHA,
        >::shake_round_constants(
            seed.as_bytes(),
            RPO_GOLDILOCKS_NUM_ROUNDS,
            BYTES_PER_CONSTANT,
            false,
        );
        Self {
            inner: Rpo::new(RPO_GOLDILOCKS_NUM_ROUNDS, rcs, MdsMatrixRpoGoldilocks),
        }
    }
}

impl Permutation<[Goldilocks; RPO_GOLDILOCKS_WIDTH]> for RpoGoldilocks {
    fn permute_mut(&self, state: &mut [Goldilocks; RPO_GOLDILOCKS_WIDTH]) {
        let rcs = &self.inner.round_constants;
        for round in 0..self.inner.num_rounds {
            self.inner.mds.permute_mut(state);
            for (s, rc) in state
                .iter_mut()
                .zip(&rcs[2 * round * RPO_GOLDILOCKS_WIDTH..])
            {
                *s += *rc;
            }
            state.iter_mut().for_each(|s| *s = s.injective_exp_n());

            self.inner.mds.permute_mut(state);
            for (s, rc) in state
                .iter_mut()
                .zip(&rcs[(2 * round + 1) * RPO_GOLDILOCKS_WIDTH..])
            {
                *s += *rc;
            }
            apply_inv_sbox_x7(state);
        }
    }
}

impl CryptographicPermutation<[Goldilocks; RPO_GOLDILOCKS_WIDTH]> for RpoGoldilocks {}

/// Width-parallel inverse S-box `x -> x^(1/7)` over `[Goldilocks; 12]`.
///
/// Computes `x^10540996611094048183` via an addition chain reaching the same
/// exponent as [`p3_field::exponentiation::exp_10540996611094048183`], applied
/// across all 12 lanes step-by-step.
#[inline]
fn apply_inv_sbox_x7(state: &mut [Goldilocks; RPO_GOLDILOCKS_WIDTH]) {
    // Binary expansion of 10540996611094048183 (63 squares + 9 mults):
    //   1001001001001001001001001001000110110110110110110110110110110111
    let mut t1 = *state;
    t1.iter_mut().for_each(|t| *t = t.square());
    let mut t2 = t1;
    t2.iter_mut().for_each(|t| *t = t.square());
    let t3 = exp_acc::<_, RPO_GOLDILOCKS_WIDTH, 3>(t2, t2);
    let t4 = exp_acc::<_, RPO_GOLDILOCKS_WIDTH, 6>(t3, t3);
    let t5 = exp_acc::<_, RPO_GOLDILOCKS_WIDTH, 12>(t4, t4);
    let t6 = exp_acc::<_, RPO_GOLDILOCKS_WIDTH, 6>(t5, t3);
    let t7 = exp_acc::<_, RPO_GOLDILOCKS_WIDTH, 31>(t6, t6);

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
    use proptest::prelude::*;

    use super::*;

    // Test vector taken from Miden-crypto:
    // https://github.com/0xMiden/crypto/blob/next/miden-crypto/src/hash/algebraic_sponge/rescue/rpo/mod.rs .
    #[test]
    fn rpo_goldilocks_width12_test_vector() {
        let rpo = RpoGoldilocks::from_standard_constants();

        let input: [Goldilocks; RPO_GOLDILOCKS_WIDTH] =
            core::array::from_fn(|i| Goldilocks::from_u64(i as u64));
        let expected: [Goldilocks; RPO_GOLDILOCKS_WIDTH] = [
            15056646954853821376,
            594518210294093573,
            10395398226526937664,
            3903707756219396109,
            7670128982698747483,
            4249514323476682720,
            16506822133651532340,
            10593868791806571942,
            9413309068803954142,
            15946782832277734471,
            7904287043744270535,
            16548919317472389167,
        ]
        .map(Goldilocks::from_u64);

        assert_eq!(rpo.permute(input), expected);
    }

    #[test]
    fn mds_rpo_goldilocks_width_12_matches_circulant() {
        // MDS * e_0 must equal the first column of the circulant.
        let mut state: [Goldilocks; RPO_GOLDILOCKS_WIDTH] =
            [Goldilocks::ZERO; RPO_GOLDILOCKS_WIDTH];
        state[0] = Goldilocks::ONE;
        MdsMatrixRpoGoldilocks.permute_mut(&mut state);
        let expected: [Goldilocks; RPO_GOLDILOCKS_WIDTH] = core::array::from_fn(|i| {
            Goldilocks::from_u64(
                MDS_12_FIRST_ROW[(RPO_GOLDILOCKS_WIDTH - i) % RPO_GOLDILOCKS_WIDTH] as u64,
            )
        });
        assert_eq!(state, expected);
    }

    fn mds_naive(input: [Goldilocks; RPO_GOLDILOCKS_WIDTH]) -> [Goldilocks; RPO_GOLDILOCKS_WIDTH] {
        let coeffs: [Goldilocks; RPO_GOLDILOCKS_WIDTH] =
            core::array::from_fn(|i| Goldilocks::from_u64(MDS_12_FIRST_ROW[i] as u64));
        core::array::from_fn(|r| {
            (0..RPO_GOLDILOCKS_WIDTH)
                .map(|c| coeffs[(c + RPO_GOLDILOCKS_WIDTH - r) % RPO_GOLDILOCKS_WIDTH] * input[c])
                .sum()
        })
    }

    proptest! {
        #[test]
        fn proptest_mds_matches_naive_matrix_product(
            seeds in prop::array::uniform12(any::<u64>())
        ) {
            let input: [Goldilocks; RPO_GOLDILOCKS_WIDTH] =
                core::array::from_fn(|i| Goldilocks::from_u64(seeds[i]));
            prop_assert_eq!(MdsMatrixRpoGoldilocks.permute(input), mds_naive(input));
        }
    }
}
