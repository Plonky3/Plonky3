use alloc::vec;
use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use shake::Shake256;
use shake::digest::{ExtendableOutput, Update, XofReader};

/// Compute the SHAKE-256 XOF output of a seed.
pub(crate) fn shake256_hash(seed_bytes: &[u8], num_bytes: usize) -> Vec<u8> {
    let mut hasher = Shake256::default();
    hasher.update(seed_bytes);
    let mut reader = hasher.finalize_xof();
    let mut result = vec![0u8; num_bytes];
    reader.read(&mut result);
    result
}

/// Width-parallel `state.map(|x| x^(2^M))`.
///
/// Squares each lane `M` times. Each step squares all `N` lanes before
/// advancing, exposing `N`-way ILP within a step so the CPU can hide
/// per-multiplication latency.
#[inline]
pub(crate) fn square_n<R, const N: usize>(mut state: [R; N], m: usize) -> [R; N]
where
    R: PrimeCharacteristicRing + Copy,
{
    for _ in 0..m {
        state.iter_mut().for_each(|x| *x = x.square());
    }
    state
}

/// Width-parallel `base.map(|x| x^(2^M)) * tail` (lane-wise).
///
/// Building block for evaluating addition chains over `[R; N]` while
/// preserving lane parallelism.
#[inline]
pub(crate) fn exp_acc<R, const N: usize, const M: usize>(base: [R; N], tail: [R; N]) -> [R; N]
where
    R: PrimeCharacteristicRing + Copy,
{
    let mut r = square_n(base, M);
    r.iter_mut().zip(tail).for_each(|(x, t)| *x *= t);
    r
}
