//! The arithmetic a Flock verifier runs over.

use alloc::vec::Vec;

use p3_binary_field::{fold_bits, fold_rows};
use p3_sumcheck::ring_switch::bits::BitTensorArith;
use p3_verifier_arith::{Lift, Native};

use crate::{Challenge, Val};

/// Values and operations a Flock verifier computes with, natively or inside a circuit.
pub trait FlockArith: BitTensorArith + Lift<Val = Val, Challenge = Challenge> {
    /// `out[i] = sum_o weights[o] * bit i of instance o`, every instance `2^k_log` bits, zero from `useful_bits` on.
    fn fold_bits(
        &mut self,
        bits: &[u64],
        k_log: usize,
        _useful_bits: usize,
        weights: &[Self::EF],
    ) -> Vec<Self::EF> {
        fold_bits(bits, k_log, weights)
    }
}

impl FlockArith for Native<Val, Challenge> {
    fn fold_bits(
        &mut self,
        bits: &[u64],
        k_log: usize,
        useful_bits: usize,
        weights: &[Challenge],
    ) -> Vec<Challenge> {
        fold_rows(bits, k_log, useful_bits, weights)
    }
}
