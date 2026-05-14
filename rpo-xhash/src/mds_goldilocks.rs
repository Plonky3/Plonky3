//! Base-field MDS types for Goldilocks (p = 2^64 - 2^32 + 1).
//!
//! Uses the hi/lo split strategy: split 64-bit elements into two 32-bit halves,
//! run the field-independent FFT MDS on each half independently, then recombine.

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use p3_mds::MdsPermutation;
use p3_symmetric::Permutation;

use crate::fft_mds::mds12_multiply_freq;
use crate::reduce::goldilocks::combine_halves;

#[inline(always)]
fn apply_fft_mds_gl<const N: usize>(
    state: &mut [Goldilocks; N],
    freq_fn: fn([u64; N]) -> [u64; N],
) {
    let mut lo = [0u64; N];
    let mut hi = [0u64; N];
    for i in 0..N {
        let s = state[i].as_canonical_u64();
        hi[i] = s >> 32;
        lo[i] = (s as u32) as u64;
    }
    let hi = freq_fn(hi);
    let lo = freq_fn(lo);
    for i in 0..N {
        state[i] = Goldilocks::new(combine_halves(lo[i], hi[i]));
    }
}

/// 12×12 base-field FFT MDS for Goldilocks. Uses the same circulant row as
/// miden-crypto's `Rpo256` / `Rpx256` (`[7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8]`).
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct MdsBase12;

impl Permutation<[Goldilocks; 12]> for MdsBase12 {
    #[inline(always)]
    fn permute_mut(&self, state: &mut [Goldilocks; 12]) {
        apply_fft_mds_gl(state, mds12_multiply_freq);
    }
}
impl MdsPermutation<Goldilocks, 12> for MdsBase12 {}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;

    // First row of miden-crypto's RPO/RPX MDS matrix.
    const MDS12_ROW: [Goldilocks; 12] = [
        Goldilocks::new(7),
        Goldilocks::new(23),
        Goldilocks::new(8),
        Goldilocks::new(26),
        Goldilocks::new(13),
        Goldilocks::new(10),
        Goldilocks::new(9),
        Goldilocks::new(7),
        Goldilocks::new(6),
        Goldilocks::new(22),
        Goldilocks::new(21),
        Goldilocks::new(8),
    ];

    #[inline(always)]
    fn apply_circulant_naive<const N: usize>(
        state: &mut [Goldilocks; N],
        first_row: [Goldilocks; N],
    ) {
        let input = *state;
        let mut out = [Goldilocks::ZERO; N];
        for (row, out_cell) in out.iter_mut().enumerate().take(N) {
            let mut acc = Goldilocks::ZERO;
            for (col, input_col) in input.iter().enumerate().take(N) {
                let idx = (col + N - row) % N;
                acc += first_row[idx] * *input_col;
            }
            *out_cell = acc;
        }
        *state = out;
    }

    #[test]
    fn mds12_gl_changes_state() {
        let mut state: [Goldilocks; 12] =
            core::array::from_fn(|i| Goldilocks::new((i as u64 + 1) * 1000000007));
        let original = state;
        MdsBase12.permute_mut(&mut state);
        assert_ne!(state, original);
    }

    #[test]
    fn mds12_matches_naive_circulant() {
        let state: [Goldilocks; 12] =
            core::array::from_fn(|i| Goldilocks::new((i as u64 + 1) * 1000000007));
        let mut fast = state;
        let mut slow = state;
        MdsBase12.permute_mut(&mut fast);
        apply_circulant_naive(&mut slow, MDS12_ROW);
        assert_eq!(fast, slow);
    }
}
