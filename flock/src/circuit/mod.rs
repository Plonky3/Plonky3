//! Boolean circuits as gate lists: an R1CS over GF(2) whose matrices are never built.
//!
//! One instance's witness has `2^k_log` bits: input ports, output ports, the constant one, the products, zero padding.

mod builder;
mod list;
mod wire;

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Range;

pub use builder::Builder;
pub(crate) use list::GateList;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::CryptographicHasher;
pub use wire::{Wire, Word};

use crate::witness::Program;

/// One gate; a gate's index is its wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Gate {
    /// Committed free wire at a slot, with row `z[s] * 1 = z[s]`.
    Free(u32),
    /// Uncommitted sum of two wires: no row.
    Xor(u32, u32),
    /// Product committed at a slot, with row `x * y = z[s]`.
    And(u32, u32, u32),
    /// Affine wire committed at a slot, with row `x * 1 = z[s]`.
    Copy(u32, u32),
}

/// A slot or gate index as stored in a gate.
///
/// # Panics
///
/// - The index does not fit in 32 bits.
fn slot(index: usize) -> u32 {
    u32::try_from(index).expect("circuit index fits in 32 bits")
}

/// A port of a circuit, by its index among the inputs or among the outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Port {
    /// An input port.
    Input(usize),
    /// An output port.
    Output(usize),
}

/// A finished gate list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Circuit {
    /// Gates in order.
    gates: GateList,
    /// Slot of the constant one.
    const_pos: usize,
    /// Bits per instance, as a power of two.
    k_log: usize,
    /// Slots before the zero padding.
    useful_bits: usize,
    /// Words the input ports span.
    n_input_words: usize,
    /// Each input port's words.
    inputs: Vec<Range<usize>>,
    /// Each output port's words, after the inputs'.
    outputs: Vec<Range<usize>>,
    /// Keccak-256 of the encoding, the one binding of the circuit every transcript absorbs.
    digest: [u8; 32],
    /// The compiled walk every prover and verifier pass runs.
    program: Arc<Program>,
}

impl Circuit {
    /// Seal a gate list, hashing its encoding once, then compile it; with `keep`, the gates stay beside the walk.
    ///
    /// The constant sits right after the last port.
    fn new(
        mut gates: GateList,
        inputs: Vec<Range<usize>>,
        outputs: Vec<Range<usize>>,
        k_log: usize,
        useful_bits: usize,
        keep: bool,
    ) -> Self {
        let n_input_words = inputs.last().map_or(0, |r| r.end);
        let const_pos = const_slot(&inputs, &outputs);
        let digest = digest_of(&gates, &inputs, &outputs, k_log, useful_bits);
        let program = Program::new(&mut gates, !keep, const_pos, k_log, 64 * n_input_words);
        Self {
            gates,
            const_pos,
            k_log,
            useful_bits,
            n_input_words,
            inputs,
            outputs,
            digest,
            program: Arc::new(program),
        }
    }

    /// The circuit with its gate list dropped: every pass runs the compiled walk, so it proves and verifies alike.
    #[must_use]
    pub fn compile(mut self) -> Self {
        self.gates = GateList::default();
        self
    }

    /// The compiled walk.
    pub(crate) fn program(&self) -> Arc<Program> {
        Arc::clone(&self.program)
    }

    /// Bits per instance, as a power of two.
    #[must_use]
    pub const fn k_log(&self) -> usize {
        self.k_log
    }

    /// Slots that carry data; the rest are zero.
    #[must_use]
    pub const fn useful_bits(&self) -> usize {
        self.useful_bits
    }

    /// Slot of the constant one.
    #[must_use]
    pub const fn const_pos(&self) -> usize {
        self.const_pos
    }

    /// Words the input ports span.
    #[must_use]
    pub const fn n_input_words(&self) -> usize {
        self.n_input_words
    }

    /// A port's words in an instance, or none if the circuit has no such port.
    #[must_use]
    pub fn port(&self, port: Port) -> Option<Range<usize>> {
        match port {
            Port::Input(i) => self.inputs.get(i),
            Port::Output(i) => self.outputs.get(i),
        }
        .cloned()
    }

    /// Keccak-256 of the circuit's injective encoding.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// One instance's output port words, read from its packed `z`.
    ///
    /// # Panics
    ///
    /// - The instance is narrower than its ports.
    #[must_use]
    pub fn output_words<'a>(&self, z: &'a [u64]) -> &'a [u64] {
        // Output ports sit between the inputs and the constant.
        &z[self.n_input_words..self.const_pos / 64]
    }

    /// Committed products, the R1CS rows that are not ports or the constant.
    #[must_use]
    pub fn num_ands(&self) -> usize {
        self.program.ands()
    }

    /// Operations of the compiled walk: products, port and copy rows, and the XORs read more than once.
    #[must_use]
    pub fn num_ops(&self) -> usize {
        self.program.ops()
    }

    /// Terms of the compiled walk's linear forms: the work of one pass besides its products.
    #[must_use]
    pub fn num_terms(&self) -> usize {
        self.program.terms()
    }

    /// The gates in order, empty once [`Self::compile`] dropped them.
    #[cfg(test)]
    pub(crate) const fn gates(&self) -> &GateList {
        &self.gates
    }

    /// One instance's packed `z`, `A z` and `B z`, by one walk on bits.
    ///
    /// This is the reference every faster generator is tested against.
    ///
    /// # Panics
    ///
    /// - The input or a table has the wrong length.
    #[cfg(test)]
    pub(crate) fn witness_instance(
        &self,
        inputs: &[u64],
        z: &mut [u64],
        az: &mut [u64],
        bz: &mut [u64],
    ) {
        assert_eq!(inputs.len(), self.n_input_words, "input words");
        let words = (1 << self.k_log) / 64;
        for table in [&mut *z, &mut *az, &mut *bz] {
            assert_eq!(table.len(), words, "table words");
            table.fill(0);
        }
        let set =
            |buf: &mut [u64], s: u32, v: bool| buf[s as usize / 64] |= u64::from(v) << (s % 64);

        // Wire i is gate i's value.
        let mut wires: Vec<bool> = Vec::with_capacity(self.gates.len());
        for gate in self.gates.iter() {
            let v = match gate {
                Gate::Free(s) => {
                    let v = s as usize == self.const_pos
                        || (inputs[s as usize / 64] >> (s % 64)) & 1 == 1;
                    set(z, s, v);
                    set(az, s, v);
                    set(bz, s, true);
                    v
                }
                Gate::Xor(x, y) => wires[x as usize] ^ wires[y as usize],
                Gate::And(x, y, s) => {
                    let (x, y) = (wires[x as usize], wires[y as usize]);
                    set(z, s, x & y);
                    set(az, s, x);
                    set(bz, s, y);
                    x & y
                }
                Gate::Copy(x, s) => {
                    let v = wires[x as usize];
                    set(z, s, v);
                    set(az, s, v);
                    set(bz, s, true);
                    v
                }
            };
            wires.push(v);
        }
    }

    /// Every committed row's `(A_0 w, B_0 w)` entries over a characteristic-two field, by one forward walk.
    ///
    /// Rows no gate commits are zero on both sides and are not visited.
    ///
    /// # Panics
    ///
    /// - The vector has the wrong length.
    #[cfg(test)]
    pub(crate) fn walk_rows<V: PrimeCharacteristicRing>(
        &self,
        w: &[V],
        row: impl FnMut(usize, V, V),
    ) {
        assert_eq!(w.len(), 1 << self.k_log, "one entry per slot");
        self.walk_rows_by(|s| w[s].dup(), row);
    }

    /// Every committed row's `(A_0 w, B_0 w)` entries by one forward walk, the entry of slot `s` given by `w(s)`.
    pub(crate) fn walk_rows_by<V: PrimeCharacteristicRing>(
        &self,
        w: impl Fn(usize) -> V,
        row: impl FnMut(usize, V, V),
    ) {
        self.program.walk_rows_by(w, row);
    }

    /// Add `(a A_0 + b B_0)^T u` into the output, by one backward walk.
    ///
    /// # Panics
    ///
    /// - The vector or the output has the wrong length.
    #[cfg(test)]
    pub(crate) fn add_fold_transpose<V: Field>(&self, a: V, b: V, u: &[V], m: &mut [V]) {
        assert_eq!(u.len(), 1 << self.k_log, "one weight per row");
        self.add_fold_transpose_by(a, b, |s| u[s], m);
    }

    /// Add `(a A_0 + b B_0)^T u` into the output by one backward walk, the weight of row `s` given by `u(s)`.
    ///
    /// # Panics
    ///
    /// - The output has the wrong length.
    pub(crate) fn add_fold_transpose_by<V: Field>(
        &self,
        a: V,
        b: V,
        u: impl Fn(usize) -> V,
        m: &mut [V],
    ) {
        assert_eq!(m.len(), 1 << self.k_log, "one output per slot");
        self.program.fold_transpose_by(a, b, u, |s, v| m[s] += v);
    }
}

/// Slot of the constant one: right after the last port.
fn const_slot(inputs: &[Range<usize>], outputs: &[Range<usize>]) -> usize {
    let n_input_words = inputs.last().map_or(0, |r| r.end);
    64 * outputs.last().map_or(n_input_words, |r| r.end)
}

/// The digest of a sealed gate list with its ports, as [`Circuit::digest`] gives it.
pub(crate) fn digest_of(
    gates: &GateList,
    inputs: &[Range<usize>],
    outputs: &[Range<usize>],
    k_log: usize,
    useful_bits: usize,
) -> [u8; 32] {
    let n_input_words = inputs.last().map_or(0, |r| r.end);
    let shape = [
        k_log,
        const_slot(inputs, outputs),
        useful_bits,
        n_input_words,
        gates.len(),
    ];
    digest(gates, shape)
}

/// Keccak-256 of the shape, then of the digests of fixed chunks of gates.
///
/// The shape counts the gates and every gate has a fixed width, so the encoding is injective.
///
/// The chunks hash in parallel, so binding a large circuit costs little.
fn digest(gates: &GateList, shape: [usize; 5]) -> [u8; 32] {
    let chunks: Vec<[u8; 32]> = gates
        .runs()
        .into_par_iter()
        .map(|gates| {
            let mut bytes = Vec::with_capacity(13 * list::CHUNK);
            encode_gates(gates, &mut bytes);
            Keccak256Hash.hash_slice(&bytes)
        })
        .collect();
    let shape: Vec<u8> = shape
        .iter()
        .flat_map(|&v| (v as u64).to_be_bytes())
        .collect();
    Keccak256Hash.hash_iter_slices(
        core::iter::once(shape.as_slice()).chain(chunks.iter().map(|c| c.as_slice())),
    )
}

/// Append gates at a fixed width: a kind byte, then three big-endian words.
fn encode_gates(gates: impl Iterator<Item = Gate>, out: &mut Vec<u8>) {
    for gate in gates {
        let (kind, words) = match gate {
            Gate::Free(s) => (0, [s, 0, 0]),
            Gate::Xor(x, y) => (1, [x, y, 0]),
            Gate::And(x, y, s) => (2, [x, y, s]),
            Gate::Copy(x, s) => (3, [x, s, 0]),
        };
        out.push(kind);
        for w in words {
            out.extend_from_slice(&w.to_be_bytes());
        }
    }
}
