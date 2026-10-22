//! [`WireArith`]: the Flock verifier's arithmetic on the wires of a Flock circuit.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{Poly64, Poly192};
use p3_challenger::CanObserve;
use p3_field::{Dup, Field, PrimeCharacteristicRing};
use p3_multilinear_util::point::Point;
use p3_sumcheck::ring_switch::bits::BitTensorArith;
use p3_verifier_arith::{Arith, Lift};
use spin::Mutex;

use super::deferred::{Matrix, PendingForm, form_rows};
use super::gates::{BinaryElement, Byte, Cost, Gates, Sample};
use super::statement::WireStatement;
use super::value::{Bin, Gf64, Gf192, Lowering, Shared};
use crate::circuit::{Circuit, Wire};
use crate::ports::{PortsError, PublicPorts};
use crate::{Batch, Challenge, FlockArith, Val};

/// A bit of a circuit, with its native value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bit {
    /// The wire.
    pub wire: Wire,
    /// The native value.
    pub value: bool,
}

/// The Flock verifier's arithmetic whose values are wires of a Flock circuit.
///
/// Running a verifier on it builds the circuit that checks the same proof, every proof value an advice wire.
///
/// The lincheck's matrix evaluation is deferred: the circuit form becomes a pending claim the caller reduces.
#[derive(Debug)]
pub struct WireArith {
    /// The circuit under construction.
    pub(crate) gates: Shared,
    /// The circuit forms the lincheck deferred, in order.
    pub(crate) forms: Vec<PendingForm<Gf192>>,
    /// The last point a row evaluation used, with its equality table.
    eq_cache: Option<(Vec<Gf192>, Vec<Gf192>)>,
    /// The statement of the proof being verified, when its words are wires, with its native table.
    pub(crate) statement: Option<(WireStatement, Vec<u64>)>,
    /// Compiled matrices of the circuits verified so far.
    matrices: Vec<Arc<Matrix>>,
    /// The digest of the last statement of wires bound, on wires.
    pub(crate) statement_digest: Option<[Byte; 32]>,
}

impl Default for WireArith {
    fn default() -> Self {
        Self::new()
    }
}

impl WireArith {
    /// An empty circuit.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gates: Arc::new(Mutex::new(Gates::new())),
            forms: Vec::new(),
            eq_cache: None,
            statement: None,
            matrices: Vec::new(),
            statement_digest: None,
        }
    }

    /// Attribute the gates from here on to `component`, returning the previous one.
    pub fn component(&mut self, component: &'static str) -> &'static str {
        self.gates.lock().component(component)
    }

    /// The cost of every component so far.
    #[must_use]
    pub fn costs(&self) -> Vec<Cost> {
        let mut gates = self.gates.lock();
        let current = gates.current.component;
        gates.component(current);
        gates.costs.clone()
    }

    /// The same arithmetic, with circuits' matrices already compiled.
    #[must_use]
    pub fn with_matrices(mut self, matrices: &[Arc<Matrix>]) -> Self {
        self.matrices.extend(matrices.iter().cloned());
        self
    }

    /// A circuit's compiled matrices, compiled at most once.
    pub fn matrix(&mut self, circuit: &Circuit) -> Arc<Matrix> {
        if let Some(m) = self
            .matrices
            .iter()
            .find(|m| m.digest() == circuit.digest())
        {
            return Arc::clone(m);
        }
        let m = Arc::new(Matrix::new(circuit));
        self.matrices.push(Arc::clone(&m));
        m
    }

    /// The circuit forms deferred so far, oldest first, taken out.
    pub fn take_forms(&mut self) -> Vec<PendingForm<Gf192>> {
        core::mem::take(&mut self.forms)
    }

    /// Run `f` on the circuit under construction.
    pub(crate) fn with_gates<R>(&self, f: impl FnOnce(&mut Gates) -> R) -> R {
        f(&mut self.gates.lock())
    }

    /// Advice carrying `value`.
    pub(crate) fn advice<E: Lowering>(&self, value: E) -> Bin<E> {
        let wires = self.with_gates(|g| g.advice((0..E::BITS).map(|i| value.bit(i))));
        Bin::wired(&self.gates, wires, value)
    }

    /// Advice bits carrying `values`.
    pub(crate) fn advice_bits(&self, values: &[bool]) -> Vec<Bit> {
        let wires = self.with_gates(|g| g.advice(values.iter().copied()));
        wires
            .into_iter()
            .zip(values)
            .map(|(wire, &value)| Bit { wire, value })
            .collect()
    }

    /// Reveal a value: its bits become public outputs of the circuit.
    pub fn reveal<E: Lowering>(&mut self, value: &Bin<E>) {
        let value = value.dup().settled();
        let native = value.value();
        self.with_gates(|g| {
            for (i, wire) in value.wires(g).into_iter().enumerate() {
                g.reveal(wire, native.bit(i));
            }
        });
    }

    /// Require two values to be equal, every bit of their sum asserted zero.
    ///
    /// Returns whether they are equal natively.
    pub fn assert_equal<E: Lowering>(&mut self, a: Bin<E>, b: Bin<E>) -> bool {
        let diff = (a + b).settled();
        let zero = diff.value() == E::ZERO;
        self.with_gates(|g| {
            for wire in diff.wires(g) {
                g.assert_zero(wire);
            }
        });
        zero
    }

    /// The bits of a value, with their native values.
    pub(crate) fn bits<E: Lowering>(&self, value: &Bin<E>) -> Vec<Bit> {
        let value = value.dup().settled();
        let native = value.value();
        let wires = self.with_gates(|g| value.wires(g));
        wires
            .into_iter()
            .enumerate()
            .map(|(i, wire)| Bit {
                wire,
                value: native.bit(i),
            })
            .collect()
    }

    /// The value with the given bits.
    pub(crate) fn value_of_bits<E: Lowering>(&self, bits: &[Bit]) -> Bin<E> {
        assert_eq!(bits.len(), E::BITS, "one bit per coordinate");
        let value = E::from_bits(|i| bits[i].value);
        let constant = bits
            .iter()
            .all(|b| b.wire.is_zero() || b.wire == self.with_gates(|g| g.constant(true)));
        if constant {
            Bin::from(value)
        } else {
            Bin::wired(&self.gates, bits.iter().map(|b| b.wire).collect(), value)
        }
    }

    /// The bytes of a value, little-endian.
    pub(crate) fn bytes<E: Lowering>(&self, value: &Bin<E>) -> Vec<Byte> {
        pack_bytes(&self.bits(value))
    }

    /// `bit * value`.
    pub(crate) fn scale<E: Lowering>(&self, bit: &Bit, value: Bin<E>) -> Bin<E> {
        if bit.wire.is_zero() {
            return Bin::ZERO;
        }
        if value.as_const().is_some_and(|c| *c == E::ZERO) {
            return value;
        }
        value.scaled(bit.wire, bit.value, &self.gates)
    }

    /// The transpose of a bit matrix whose rows are values: column `v` holds bit `v` of every row.
    pub(crate) fn transpose<R: Lowering, C: Lowering>(&self, rows: &[Bin<R>]) -> Vec<Bin<C>> {
        assert_eq!(rows.len(), C::BITS, "one row per bit of a column");
        let bits: Vec<Vec<Bit>> = rows.iter().map(|row| self.bits(row)).collect();
        (0..R::BITS)
            .map(|v| {
                let column: Vec<Bit> = bits.iter().map(|row| row[v]).collect();
                self.value_of_bits(&column)
            })
            .collect()
    }
}

/// Bits as bytes, eight to a byte, low bit first.
pub(crate) fn pack_bytes(bits: &[Bit]) -> Vec<Byte> {
    bits.as_chunks::<8>()
        .0
        .iter()
        .map(|byte| Byte {
            wires: core::array::from_fn(|i| byte[i].wire),
            value: byte
                .iter()
                .enumerate()
                .fold(0, |acc, (i, b)| acc | (u8::from(b.value) << i)),
        })
        .collect()
}

impl Arith for WireArith {
    type Val = Val;
    type Challenge = Challenge;
    type F = Gf64;
    type EF = Gf192;
    type Bit = Bit;
    type Index = Vec<Bit>;

    fn bit(&self, bit: &Bit) -> Gf64 {
        let mut wires = vec![Wire::ZERO; 64];
        wires[0] = bit.wire;
        if bit.wire.is_zero() {
            return Gf64::ZERO;
        }
        Bin::wired(&self.gates, wires, Poly64::from_bool(bit.value))
    }

    fn index_bits(&mut self, index: &Vec<Bit>, count: usize) -> Vec<Bit> {
        // Bits above the sampled width are zero.
        let mut bits: Vec<Bit> = index.iter().take(count).copied().collect();
        bits.resize(
            count,
            Bit {
                wire: Wire::ZERO,
                value: false,
            },
        );
        bits
    }

    fn index_shr(&self, index: &Vec<Bit>, shift: usize) -> Vec<Bit> {
        index.iter().skip(shift).copied().collect()
    }

    fn constant_index(&mut self, value: usize) -> Vec<Bit> {
        let one = self.with_gates(|g| g.constant(true));
        (0..usize::BITS - value.leading_zeros())
            .map(|i| {
                let set = (value >> i) & 1 == 1;
                Bit {
                    wire: if set { one } else { Wire::ZERO },
                    value: set,
                }
            })
            .collect()
    }

    fn index_with_high(&mut self, low: &Vec<Bit>, low_bits: usize, high: usize) -> Vec<Bit> {
        let mut bits = self.index_bits(low, low_bits);
        bits.extend(self.constant_index(high));
        bits
    }

    fn select(&mut self, bit: &Bit, then: Gf192, otherwise: Gf192) -> Gf192 {
        let diff = then + otherwise.dup();
        otherwise + self.scale(bit, diff)
    }

    fn select_base(&mut self, bit: &Bit, then: Gf64, otherwise: Gf64) -> Gf64 {
        let diff = then + otherwise.dup();
        otherwise + self.scale(bit, diff)
    }

    fn eval_multilinear(&mut self, evals: &[Gf64], point: &Point<Gf192>) -> Gf192 {
        let previous = self.component("leaf folds");
        // Every query of a round evaluates at the same point, so its equality table is shared.
        let coordinates = point.as_slice();
        let cached = self.eq_cache.as_ref().is_some_and(|(p, _)| {
            p.len() == coordinates.len() && p.iter().zip(coordinates).all(|(a, b)| same(a, b))
        });
        if !cached {
            let table = self.eq_table(coordinates);
            self.eq_cache = Some((coordinates.to_vec(), table));
        }
        let (_, table) = self.eq_cache.as_ref().expect("cached above");
        let value = table
            .iter()
            .zip(evals)
            .map(|(e, v)| e.dup() * v.dup())
            .sum::<Gf192>()
            .settled();
        self.component(previous);
        value
    }

    fn batch_inverse<E>(
        &mut self,
        values: &[Gf192],
        zero: impl FnOnce(usize) -> E,
    ) -> Result<Vec<Gf192>, E> {
        if let Some(position) = values.iter().position(|v| v.value() == Poly192::ZERO) {
            return Err(zero(position));
        }
        Ok(values
            .iter()
            .map(|v| {
                v.as_const().map_or_else(
                    || {
                        // The inverse is advice, held to `v inverse = 1`.
                        let inverse = self.advice(v.value().inverse());
                        let product = v.dup() * inverse.dup();
                        let inverted = self.assert_equal(product, Gf192::ONE);
                        debug_assert!(inverted, "a nonzero value has an inverse");
                        inverse
                    },
                    |c| Gf192::from(c.inverse()),
                )
            })
            .collect())
    }

    fn assert_zero<E>(&mut self, value: Gf192, err: impl FnOnce() -> E) -> Result<(), E> {
        if self.assert_equal(value, Gf192::ZERO) {
            Ok(())
        } else {
            Err(err())
        }
    }

    fn coefficients(&mut self, value: &Gf192) -> Vec<Gf64> {
        value.limbs().to_vec()
    }
}

/// Whether two values are the same wires or the same constant.
fn same(a: &Gf192, b: &Gf192) -> bool {
    match (a.as_const(), b.as_const()) {
        (Some(x), Some(y)) => x == y,
        (None, None) => a.ptr_eq(b),
        _ => false,
    }
}

impl Lift for WireArith {
    fn load(&mut self, value: Val) -> Gf64 {
        self.advice(value)
    }

    fn load_ext(&mut self, value: Challenge) -> Gf192 {
        self.advice(value)
    }

    fn value(&self, value: &Gf64) -> Val {
        value.value()
    }

    fn value_ext(&self, value: &Gf192) -> Challenge {
        value.value()
    }

    fn seeding<T>(&mut self, construct: impl FnOnce() -> T) -> T {
        self.gates.lock().seeding = true;
        let transcript = construct();
        self.gates.lock().seeding = false;
        transcript
    }

    fn will_observe(&mut self, units: Vec<Gf64>) {
        let bytes: Vec<Byte> = units.iter().flat_map(|u| self.bytes(u)).collect();
        self.gates.lock().pending.extend(bytes);
    }

    fn sampled_ext(&mut self, value: Challenge) -> Gf192 {
        let limbs: [Gf64; 3] =
            core::array::from_fn(|_| match self.gates.lock().sampled.pop_front() {
                Some(Sample::Value(wires, value)) => Bin::wired(&self.gates, wires, value),
                _ => panic!("a challenge-field sample was drawn"),
            });
        let sample = Gf192::from_limbs(&limbs);
        assert_eq!(
            sample.value(),
            value,
            "the mirror samples what the transcript samples"
        );
        sample
    }

    fn sampled_index(&mut self, value: usize, bits: usize) -> Vec<Bit> {
        let index: Vec<Bit> = match self.gates.lock().sampled.pop_front() {
            Some(Sample::Index(index)) => index
                .into_iter()
                .map(|(wire, value)| Bit { wire, value })
                .collect(),
            _ => panic!("a query index sample was drawn"),
        };
        let index: Vec<Bit> = index.into_iter().take(bits).collect();
        let recomposed: usize = index
            .iter()
            .enumerate()
            .filter(|(_, b)| b.value)
            .map(|(i, _)| 1usize << i)
            .sum();
        assert_eq!(
            recomposed, value,
            "the mirror samples what the transcript samples"
        );
        index
    }
}

impl BitTensorArith for WireArith {
    fn tensor_columns(&mut self, rows: &[Gf64]) -> Vec<Gf192> {
        // The coordinates are the raw bits, so a column is a rewiring of the rows.
        self.transpose(rows)
    }

    fn equality_rows(&mut self, a: &[Gf192], b: &[Gf192]) -> Vec<Gf192> {
        // The native algorithm: per factor, rows scaled by `1 + b`, plus the original's columns scaled by `a`.
        let previous = self.component("equality element");
        let mut rows = one_rows();
        for (a, b) in a.iter().zip(b) {
            let columns: Vec<Gf192> = self.transpose(&rows);
            let scaled: Vec<Gf192> = columns.into_iter().map(|c| c * a.dup()).collect();
            let back: Vec<Gf192> = self.transpose(&scaled);
            let factor = Gf192::ONE + b.dup();
            rows = rows
                .into_iter()
                .zip(back)
                .map(|(row, extra)| row * factor.dup() + extra)
                .collect();
        }
        self.component(previous);
        rows
    }
}

/// The rows of `1 ⊗ 1`: row `u` is one exactly where coordinate `u` of one is set.
fn one_rows() -> Vec<Gf192> {
    (0..192)
        .map(|u| {
            if Poly192::ONE.bit(u) {
                Gf192::ONE
            } else {
                Gf192::ZERO
            }
        })
        .collect()
}

impl FlockArith for WireArith {
    fn bind_statement<C: CanObserve<Val>>(
        &mut self,
        batch: &Batch<'_>,
        public: &PublicPorts,
        challenger: &mut C,
    ) -> Result<(), PortsError> {
        match self.statement.take() {
            Some((statement, values)) => {
                let bound = self.bind_wire_statement(&statement, batch, public, challenger);
                self.statement = Some((statement, values));
                bound
            }
            None => self.seeding(|| public.bind(batch, challenger)),
        }
    }

    fn fold_bits(
        &mut self,
        bits: &[u64],
        k_log: usize,
        useful_bits: usize,
        weights: &[Gf192],
    ) -> Vec<Gf192> {
        match self.statement.take() {
            Some((statement, values)) if values == bits => {
                let public_words = useful_bits / 64;
                let folded = self.fold_wire_bits(&statement, public_words, weights);
                self.statement = Some((statement, values));
                folded
            }
            other => {
                self.statement = other;
                p3_binary_field::fold_bits(bits, k_log, weights)
            }
        }
    }

    fn circuit_form(
        &mut self,
        circuit: &Circuit,
        skip: &Gf192,
        inner: &[Gf192],
        point: &[Gf192],
        slices: &[Gf192],
        alpha: &Gf192,
    ) -> Gf192 {
        // The row fold `h(v)` is advice; the form is `sum_v slices[v] h(v)`, the matrices left to a deferred claim.
        let previous = self.component("deferral");
        let native = |v: &[Gf192]| v.iter().map(Bin::value).collect::<Vec<_>>();
        let matrix = self.matrix(circuit);
        let h = form_rows(
            &matrix,
            skip.value(),
            &native(inner),
            &native(point),
            alpha.value(),
        );
        let h: Vec<Gf192> = h.into_iter().map(|v| self.advice(v)).collect();
        let form = slices.iter().zip(&h).map(|(s, h)| s.dup() * h.dup()).sum();
        self.forms.push(PendingForm {
            k_log: circuit.k_log(),
            skip: skip.dup(),
            inner: inner.to_vec(),
            point: point.to_vec(),
            alpha: alpha.dup(),
            h,
        });
        self.component(previous);
        form
    }
}
