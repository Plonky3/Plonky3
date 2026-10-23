//! The arithmetic a Flock verifier runs over.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{fold_bits, fold_rows};
use p3_challenger::CanObserve;
use p3_field::{Dup, PrimeCharacteristicRing};
use p3_sumcheck::ring_switch::bits::BitTensorArith;
use p3_verifier_arith::{Arith, Lift, Native};

use crate::circuit::Circuit;
use crate::ports::{PortsError, PublicPorts};
use crate::skip::{K_SKIP, SKIP_SIZE, lagrange_weights};
use crate::{Batch, Challenge, Val};

/// Values and operations a Flock verifier computes with, natively or inside a circuit.
pub trait FlockArith: BitTensorArith + Lift<Val = Val, Challenge = Challenge> {
    /// Bind a batch's statement into the transcript, where the prover does.
    ///
    /// The default makes the whole seed a constant of the arithmetic.
    ///
    /// # Errors
    ///
    /// The public words do not fit the batch.
    fn bind_statement<C: CanObserve<Val>>(
        &mut self,
        batch: &Batch<'_>,
        public: &PublicPorts,
        challenger: &mut C,
    ) -> Result<(), PortsError> {
        self.seeding(|| public.bind(batch, challenger))
    }

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

    /// `sum_(r, c) rows(r) (A + alpha B)[r][c] cols(c)`, the circuit's matrices under the lincheck's weights.
    ///
    /// Row `64 i + u` weighs `L_u(skip) eq(inner, i)` and column `64 j + v` weighs `eq(point, j) slices[v]`.
    ///
    /// The default walks the circuit once, linear in its gates; each weight is a high table entry times a low one.
    fn circuit_form(
        &mut self,
        circuit: &Circuit,
        skip: &Self::EF,
        inner: &[Self::EF],
        point: &[Self::EF],
        slices: &[Self::EF],
        alpha: &Self::EF,
    ) -> Self::EF {
        let lagrange = lagrange_weights(skip, SKIP_SIZE);
        let (row_hi, row_lo, row_bits) = split_weights(self, inner, &lagrange);
        let (col_hi, col_lo, col_bits) = split_weights(self, point, slices);
        let (row_mask, col_mask) = ((1 << row_bits) - 1, (1 << col_bits) - 1);
        // Rows sum per high entry, which scales each sum once.
        let mut sums = vec![(Self::EF::ZERO, Self::EF::ZERO); row_hi.len()];
        circuit.walk_rows_by(
            |s| col_hi[s >> col_bits].dup() * col_lo[s & col_mask].dup(),
            |row, a, b| {
                let low = &row_lo[row & row_mask];
                let (sa, sb) = &mut sums[row >> row_bits];
                *sa += low.dup() * a;
                *sb += low.dup() * b;
            },
        );
        sums.into_iter()
            .zip(row_hi)
            .map(|((sa, sb), hi)| hi * (sa + alpha.dup() * sb))
            .sum()
    }
}

/// The equality weights of `point`, then 64 low entries, as a high table, a low table with the 64 folded in, and the low table's index bits.
fn split_weights<A: Arith + ?Sized>(
    arith: &mut A,
    point: &[A::EF],
    low: &[A::EF],
) -> (Vec<A::EF>, Vec<A::EF>, usize) {
    let lo_bits = point.len() / 2;
    let (hi, lo) = point.split_at(point.len() - lo_bits);
    let lo_table = arith
        .eq_table(lo)
        .into_iter()
        .flat_map(|e| low.iter().map(move |l| e.dup() * l.dup()))
        .collect();
    (arith.eq_table(hi), lo_table, lo_bits + K_SKIP)
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
