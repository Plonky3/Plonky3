use alloc::format;
use alloc::vec::Vec;

use p3_field::{InjectiveMonomial, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_mds::MdsPermutation;
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

/// MDS matrix specified by the RPO paper for the Goldilocks instance. This
/// is a different matrix from `p3_goldilocks::MdsMatrixGoldilocks`.
///
/// Applied via a frequency-domain decomposition: each state element splits
/// into two 32-bit limbs, both limbs are multiplied by the circulant in the
/// frequency domain using a 12-point real FFT (decomposed as 3 × 4-point real
/// FFTs followed by three Hadamard products with the precomputed eigenvalues),
/// and the limbs are recombined into a single Goldilocks element.
#[derive(Clone, Copy, Debug, Default)]
pub struct MdsMatrixRpoGoldilocks;

impl Permutation<[Goldilocks; RPO_GOLDILOCKS_WIDTH]> for MdsMatrixRpoGoldilocks {
    fn permute_mut(&self, input: &mut [Goldilocks; RPO_GOLDILOCKS_WIDTH]) {
        apply_mds_freq(input);
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

/// Multiply `state` by the width-12 RPO circulant MDS in the frequency domain.
///
/// Splits each Goldilocks element into a high and a low 32-bit limb, runs the
/// limb-parallel FFT-domain multiplication, and recombines the limbs back into
/// `state` via the Goldilocks-specific `2^64 ≡ 2^32 − 1 (mod p)` reduction.
/// The result is non-canonical but represents the correct field element.
#[inline(always)]
fn apply_mds_freq(state: &mut [Goldilocks; RPO_GOLDILOCKS_WIDTH]) {
    // Linearity of the MDS lets us run the FFT-domain multiplication on the
    // low and high 32-bit limbs independently and recombine after, which keeps
    // every intermediate value inside `i64`.
    let mut state_l = [0u64; RPO_GOLDILOCKS_WIDTH];
    let mut state_h = [0u64; RPO_GOLDILOCKS_WIDTH];
    for r in 0..RPO_GOLDILOCKS_WIDTH {
        let s = state[r].as_canonical_u64();
        state_h[r] = s >> 32;
        state_l[r] = s & 0xFFFF_FFFF;
    }

    let state_h = freq::mds_multiply_freq(state_h);
    let state_l = freq::mds_multiply_freq(state_l);

    for r in 0..RPO_GOLDILOCKS_WIDTH {
        // Recombine and reduce modulo `p = 2^64 - 2^32 + 1`.
        //
        // The combined value `s` fits in u96, with `s = (state_h << 32) + state_l`.
        // Using `2^64 ≡ 2^32 - 1 (mod p)`, the high u32 of `s` (above 2^64) contributes
        // `s_hi * (2^32 - 1) = (s_hi << 32) - s_hi` back into the low 64 bits.
        let s = state_l[r] as u128 + ((state_h[r] as u128) << 32);
        let s_hi = (s >> 64) as u64;
        let s_lo = s as u64;
        let z = (s_hi << 32) - s_hi;
        let (res, over) = s_lo.overflowing_add(z);

        // On overflow we lost one full `2^64`, worth `2^32 - 1` mod p — add it back.
        // `0u32.wrapping_sub(over as u32) as u64` is `0` when `!over` and `2^32 - 1`
        // when `over`, producing the correction without a branch.
        state[r] = Goldilocks::new(res.wrapping_add(0u32.wrapping_sub(over as u32) as u64));
    }
}

/// Frequency-domain helpers for the width-12 RPO Goldilocks MDS multiplication.
///
/// The circulant matrix is multiplied via a 12-point real FFT decomposed as
/// 3 × 4-point real FFTs followed by three Hadamard products with precomputed
/// eigenvalues. The 4-point inverse FFTs are "squashed" with the eigenvalue
/// multiplications and twiddle factors into `block1` / `block2` / `block3`.
/// Divisions by 2 in the inverse FFT path are absorbed into the eigenvalues,
/// leaving only integer additions / subtractions / small-constant
/// multiplications in `i64`.
mod freq {
    /// Real-FFT eigenvalues for the y0 block (scaled by `1/4`).
    const MDS_FREQ_BLOCK_ONE: [i64; 3] = [16, 8, 16];
    /// Complex-FFT eigenvalues for the y1 block (scaled by `1/2`).
    const MDS_FREQ_BLOCK_TWO: [(i64, i64); 3] = [(-1, 2), (-1, 1), (4, 8)];
    /// Real-FFT eigenvalues for the y2 block (scaled by `1/4`).
    const MDS_FREQ_BLOCK_THREE: [i64; 3] = [-8, 1, 1];

    /// Apply the circulant MDS to `state` in the frequency domain.
    ///
    /// Input limbs must lie in `[0, 2^32)`; outputs are non-negative `i64`
    /// values bitwise-reinterpreted as `u64`.
    #[inline(always)]
    pub(super) const fn mds_multiply_freq(state: [u64; 12]) -> [u64; 12] {
        let [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11] = state;

        // 3 × 4-point real FFT, deinterleaved into the three blocks.
        let (u0, u1, u2) = fft4_real([s0, s3, s6, s9]);
        let (u4, u5, u6) = fft4_real([s1, s4, s7, s10]);
        let (u8, u9, u10) = fft4_real([s2, s5, s8, s11]);

        // Hadamard products with the precomputed eigenvalues. The fourth block
        // is the complex conjugate of `block2` under real-FFT symmetry, so it
        // is implicit and not computed.
        let [v0, v4, v8] = block1([u0, u4, u8], MDS_FREQ_BLOCK_ONE);
        let [v1, v5, v9] = block2([u1, u5, u9], MDS_FREQ_BLOCK_TWO);
        let [v2, v6, v10] = block3([u2, u6, u10], MDS_FREQ_BLOCK_THREE);

        // 3 × 4-point inverse real FFT and re-interleave.
        let [s0, s3, s6, s9] = ifft4_real((v0, v1, v2));
        let [s1, s4, s7, s10] = ifft4_real((v4, v5, v6));
        let [s2, s5, s8, s11] = ifft4_real((v8, v9, v10));

        [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]
    }

    /// Real 2-point FFT: `(x0 + x1, x0 - x1)`.
    #[inline(always)]
    const fn fft2_real(x: [u64; 2]) -> [i64; 2] {
        [x[0] as i64 + x[1] as i64, x[0] as i64 - x[1] as i64]
    }

    /// Real 2-point inverse FFT without the `1/2` scale; the scale is folded
    /// into the eigenvalue constants.
    #[inline(always)]
    const fn ifft2_real(y: [i64; 2]) -> [u64; 2] {
        [(y[0] + y[1]) as u64, (y[0] - y[1]) as u64]
    }

    /// Real 4-point FFT: returns `(y0, y1, y2)` with `y1` complex.
    /// `y3` is omitted by Hermitian symmetry (`y3 = conj(y1)`).
    #[inline(always)]
    const fn fft4_real(x: [u64; 4]) -> (i64, (i64, i64), i64) {
        let [z0, z2] = fft2_real([x[0], x[2]]);
        let [z1, z3] = fft2_real([x[1], x[3]]);
        let y0 = z0 + z1;
        let y1 = (z2, -z3);
        let y2 = z0 - z1;
        (y0, y1, y2)
    }

    /// Real 4-point inverse FFT. Skips both `1/2` divisions; the eigenvalues
    /// are pre-scaled so the final result is correct.
    #[inline(always)]
    const fn ifft4_real(y: (i64, (i64, i64), i64)) -> [u64; 4] {
        let z0 = y.0 + y.2;
        let z1 = y.0 - y.2;
        let z2 = y.1.0;
        let z3 = -y.1.1;

        let [x0, x2] = ifft2_real([z0, z2]);
        let [x1, x3] = ifft2_real([z1, z3]);

        [x0, x1, x2, x3]
    }

    /// Hadamard product for the y0 (real DC) block; cyclic convolution of length 3.
    #[inline(always)]
    const fn block1(x: [i64; 3], y: [i64; 3]) -> [i64; 3] {
        let [x0, x1, x2] = x;
        let [y0, y1, y2] = y;
        let z0 = x0 * y0 + x1 * y2 + x2 * y1;
        let z1 = x0 * y1 + x1 * y0 + x2 * y2;
        let z2 = x0 * y2 + x1 * y1 + x2 * y0;
        [z0, z1, z2]
    }

    /// Hadamard product for the y1 (complex) block; convolution of length 3
    /// with complex entries, Karatsuba-style.
    #[inline(always)]
    const fn block2(x: [(i64, i64); 3], y: [(i64, i64); 3]) -> [(i64, i64); 3] {
        let [(x0r, x0i), (x1r, x1i), (x2r, x2i)] = x;
        let [(y0r, y0i), (y1r, y1i), (y2r, y2i)] = y;
        let x0s = x0r + x0i;
        let x1s = x1r + x1i;
        let x2s = x2r + x2i;
        let y0s = y0r + y0i;
        let y1s = y1r + y1i;
        let y2s = y2r + y2i;

        // z0 = x0·y0 − i·x1·y2 − i·x2·y1, via Karatsuba complex multiplication.
        let m0 = (x0r * y0r, x0i * y0i);
        let m1 = (x1r * y2r, x1i * y2i);
        let m2 = (x2r * y1r, x2i * y1i);
        let z0r = (m0.0 - m0.1) + (x1s * y2s - m1.0 - m1.1) + (x2s * y1s - m2.0 - m2.1);
        let z0i = (x0s * y0s - m0.0 - m0.1) + (-m1.0 + m1.1) + (-m2.0 + m2.1);
        let z0 = (z0r, z0i);

        // z1 = x0·y1 + x1·y0 − i·x2·y2.
        let m0 = (x0r * y1r, x0i * y1i);
        let m1 = (x1r * y0r, x1i * y0i);
        let m2 = (x2r * y2r, x2i * y2i);
        let z1r = (m0.0 - m0.1) + (m1.0 - m1.1) + (x2s * y2s - m2.0 - m2.1);
        let z1i = (x0s * y1s - m0.0 - m0.1) + (x1s * y0s - m1.0 - m1.1) + (-m2.0 + m2.1);
        let z1 = (z1r, z1i);

        // z2 = x0·y2 + x1·y1 + x2·y0.
        let m0 = (x0r * y2r, x0i * y2i);
        let m1 = (x1r * y1r, x1i * y1i);
        let m2 = (x2r * y0r, x2i * y0i);
        let z2r = (m0.0 - m0.1) + (m1.0 - m1.1) + (m2.0 - m2.1);
        let z2i = (x0s * y2s - m0.0 - m0.1) + (x1s * y1s - m1.0 - m1.1) + (x2s * y0s - m2.0 - m2.1);
        let z2 = (z2r, z2i);

        [z0, z1, z2]
    }

    /// Hadamard product for the y2 (real Nyquist) block; negacyclic convolution of length 3.
    #[inline(always)]
    const fn block3(x: [i64; 3], y: [i64; 3]) -> [i64; 3] {
        let [x0, x1, x2] = x;
        let [y0, y1, y2] = y;
        let z0 = x0 * y0 - x1 * y2 - x2 * y1;
        let z1 = x0 * y1 + x1 * y0 - x2 * y2;
        let z2 = x0 * y2 + x1 * y1 + x2 * y0;
        [z0, z1, z2]
    }
}

/// Width-parallel inverse S-box `x -> x^(1/7)` over `[Goldilocks; 12]`.
///
/// Computes `x^10540996611094048183` via an addition chain reaching the same
/// exponent as [`p3_field::exponentiation::exp_10540996611094048183`], applied
/// across all 12 lanes step-by-step.
#[inline(always)]
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

    /// First row of the width-12 circulant MDS from eprint 2022/1577.
    const MDS_12_FIRST_ROW: [i64; 12] = [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8];

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
