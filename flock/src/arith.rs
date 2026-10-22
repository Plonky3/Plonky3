//! The arithmetic a Flock verifier runs over.

use alloc::vec::Vec;

use p3_binary_field::{fold_bits, fold_rows};
use p3_challenger::CanObserve;
use p3_field::{Dup, PrimeCharacteristicRing};
use p3_sumcheck::ring_switch::bits::BitTensorArith;
use p3_verifier_arith::{Lift, Native};

use crate::circuit::Circuit;
use crate::ports::{PortsError, PublicPorts};
use crate::skip::{SKIP_SIZE, lagrange_weights};
use crate::{Batch, Challenge, Val};

/// Slots of an instance up to which the lincheck's weight tables are whole.
const SPLIT_LOG: usize = 24;

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
    /// The default walks the circuit once, linear in its gates.
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
        let mut form = Self::EF::ZERO;
        if circuit.k_log() <= SPLIT_LOG {
            let rows: Vec<Self::EF> = self
                .eq_table(inner)
                .into_iter()
                .flat_map(|e| lagrange.iter().map(move |l| e.dup() * l.dup()))
                .collect();
            let columns: Vec<Self::EF> = self
                .eq_table(point)
                .into_iter()
                .flat_map(|e| slices.iter().map(move |v| e.dup() * v.dup()))
                .collect();
            circuit.walk_rows(&columns, |row, a, b| {
                form += rows[row].dup() * (a + alpha.dup() * b);
            });
            return form;
        }
        // Past `2^SPLIT_LOG` slots, each weight table splits into two halves whose product is the entry.
        let half = |arith: &mut Self, point: &[Self::EF]| {
            let lo_bits = point.len() / 2;
            let (hi, lo) = point.split_at(point.len() - lo_bits);
            (arith.eq_table(hi), arith.eq_table(lo), lo_bits)
        };
        let (row_hi, row_lo, row_bits) = half(self, inner);
        let (col_hi, col_lo, col_bits) = half(self, point);
        let at = |hi: &[Self::EF], lo: &[Self::EF], bits: usize, i: usize| {
            hi[i >> bits].dup() * lo[i & ((1 << bits) - 1)].dup()
        };
        circuit.walk_rows_by(
            |s| at(&col_hi, &col_lo, col_bits, s / SKIP_SIZE) * slices[s % SKIP_SIZE].dup(),
            |row, a, b| {
                let weight = at(&row_hi, &row_lo, row_bits, row / SKIP_SIZE)
                    * lagrange[row % SKIP_SIZE].dup();
                form += weight * (a + alpha.dup() * b);
            },
        );
        form
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
