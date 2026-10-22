//! Boolean circuits as gate lists: an R1CS over GF(2) whose matrices are never built.
//!
//! One instance's witness has `2^k_log` bits: input ports, output ports, the constant one, the products, zero padding.

mod builder;
mod wire;

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

pub use builder::Builder;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::CryptographicHasher;
pub use wire::{Wire, Word};

use crate::witness::Program;

/// Gates a digest chunk encodes and hashes on its own.
const DIGEST_GATES: usize = 1 << 12;

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
    gates: Vec<Gate>,
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
    /// The compiled walk, once the gates are dropped.
    program: Option<Arc<Program>>,
}

impl Circuit {
    /// Seal a gate list, hashing its encoding once.
    ///
    /// The constant sits right after the last port.
    fn new(
        gates: Vec<Gate>,
        inputs: Vec<Range<usize>>,
        outputs: Vec<Range<usize>>,
        k_log: usize,
        useful_bits: usize,
    ) -> Self {
        let n_input_words = inputs.last().map_or(0, |r| r.end);
        let port_words = outputs.last().map_or(n_input_words, |r| r.end);
        let mut circuit = Self {
            gates,
            const_pos: 64 * port_words,
            k_log,
            useful_bits,
            n_input_words,
            inputs,
            outputs,
            digest: [0; 32],
            program: None,
        };
        circuit.digest = circuit.compute_digest();
        circuit
    }

    /// The circuit with its gates compiled into a register walk and dropped: it can be verified and evaluated, not proven.
    #[must_use]
    pub fn compile(mut self) -> Self {
        if self.program.is_none() {
            self.program = Some(Arc::new(Program::new(&self)));
            self.gates = Vec::new();
        }
        self
    }

    /// The compiled walk, if the gates were dropped for it.
    pub(crate) fn compiled(&self) -> Option<&Program> {
        self.program.as_deref()
    }

    /// The compiled walk, compiled now unless the circuit already is.
    pub(crate) fn program(&self) -> Arc<Program> {
        self.program
            .clone()
            .unwrap_or_else(|| Arc::new(Program::new(self)))
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
        self.gates
            .iter()
            .filter(|g| matches!(g, Gate::And(..)))
            .count()
    }

    /// The gates in order.
    pub(crate) fn gates(&self) -> &[Gate] {
        &self.gates
    }

    /// Keccak-256 of the shape, then of the digests of fixed chunks of gates.
    ///
    /// The shape counts the gates and every gate has a fixed width, so the encoding is injective.
    ///
    /// The chunks hash in parallel, so binding a large circuit costs little.
    fn compute_digest(&self) -> [u8; 32] {
        let chunks: Vec<[u8; 32]> = self
            .gates
            .par_chunks(DIGEST_GATES)
            .map(|gates| {
                let mut bytes = Vec::with_capacity(13 * gates.len());
                encode_gates(gates, &mut bytes);
                Keccak256Hash.hash_slice(&bytes)
            })
            .collect();
        let mut shape = Vec::with_capacity(40);
        for v in [
            self.k_log,
            self.const_pos,
            self.useful_bits,
            self.n_input_words,
            self.gates.len(),
        ] {
            shape.extend_from_slice(&(v as u64).to_be_bytes());
        }
        Keccak256Hash.hash_iter_slices(
            core::iter::once(shape.as_slice()).chain(chunks.iter().map(|c| c.as_slice())),
        )
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
        for &gate in &self.gates {
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
    pub(crate) fn walk_rows<V: PrimeCharacteristicRing>(
        &self,
        w: &[V],
        row: impl FnMut(usize, V, V),
    ) {
        assert_eq!(w.len(), 1 << self.k_log, "one entry per slot");
        self.walk_rows_by(|s| w[s].dup(), row);
    }

    /// As [`Self::walk_rows`], the entry of slot `s` given by `w(s)`.
    pub(crate) fn walk_rows_by<V: PrimeCharacteristicRing>(
        &self,
        w: impl Fn(usize) -> V,
        mut row: impl FnMut(usize, V, V),
    ) {
        if let Some(program) = &self.program {
            return program.walk_rows_by(w, row);
        }
        // B reads the constant on port and copy rows.
        let wc = &w(self.const_pos);

        // Each wire carries its column combination; a committed wire is its own column.
        let mut wires: Vec<V> = Vec::with_capacity(self.gates.len());
        for &gate in &self.gates {
            let v = match gate {
                Gate::Free(s) => {
                    let ws = w(s as usize);
                    row(s as usize, ws.dup(), wc.dup());
                    ws
                }
                Gate::Xor(x, y) => wires[x as usize].dup() + wires[y as usize].dup(),
                Gate::And(x, y, s) => {
                    let s = s as usize;
                    row(s, wires[x as usize].dup(), wires[y as usize].dup());
                    w(s)
                }
                Gate::Copy(x, s) => {
                    let s = s as usize;
                    row(s, wires[x as usize].dup(), wc.dup());
                    w(s)
                }
            };
            wires.push(v);
        }
    }

    /// Add `(a A_0 + b B_0)^T u` into the output, by one backward walk.
    ///
    /// # Panics
    ///
    /// - The vector or the output has the wrong length.
    pub(crate) fn add_fold_transpose<V: Field>(&self, a: V, b: V, u: &[V], m: &mut [V]) {
        assert_eq!(u.len(), 1 << self.k_log, "one weight per row");
        assert_eq!(m.len(), u.len(), "one output per slot");
        if self.gates.len() > LARGE_GATES {
            return self.add_fold_transpose_registers(a, b, u, m);
        }
        let c = self.const_pos;

        // Each wire's adjoint: the row weight its consumers handed back.
        let mut adj = V::zero_vec(self.gates.len());
        for (i, &gate) in self.gates.iter().enumerate().rev() {
            let g = adj[i];
            match gate {
                // A reads the slot, B the constant.
                Gate::Free(s) => {
                    let s = s as usize;
                    m[s] += g + a * u[s];
                    m[c] += b * u[s];
                }
                // An XOR passes its adjoint to both operands.
                Gate::Xor(x, y) => {
                    adj[x as usize] += g;
                    adj[y as usize] += g;
                }
                // A reads x, B reads y; the slot collects its consumers' adjoint.
                Gate::And(x, y, s) => {
                    let s = s as usize;
                    m[s] += g;
                    adj[x as usize] += a * u[s];
                    adj[y as usize] += b * u[s];
                }
                // A reads x, B the constant.
                Gate::Copy(x, s) => {
                    let s = s as usize;
                    m[s] += g;
                    adj[x as usize] += a * u[s];
                    m[c] += b * u[s];
                }
            }
        }
    }
}

impl Circuit {
    /// As [`Self::add_fold_transpose`], each live adjoint in a register rather than one per gate.
    ///
    /// A wire's adjoint lives from its last reader, met first going backward, to its own gate.
    pub(crate) fn add_fold_transpose_registers<V: Field>(&self, a: V, b: V, u: &[V], m: &mut [V]) {
        const NONE: u32 = u32::MAX;
        let c = self.const_pos;
        let mut reg: Vec<u32> = vec![NONE; self.gates.len()];
        let mut file: Vec<V> = Vec::new();
        let mut free: Vec<u32> = Vec::new();
        // Add into a wire's adjoint, giving it a register the first time.
        let add = |w: u32, v: V, reg: &mut [u32], file: &mut Vec<V>, free: &mut Vec<u32>| {
            let r = &mut reg[w as usize];
            if *r == NONE {
                *r = free.pop().unwrap_or_else(|| {
                    file.push(V::ZERO);
                    (file.len() - 1) as u32
                });
                file[*r as usize] = v;
            } else {
                file[*r as usize] += v;
            }
        };
        for (i, &gate) in self.gates.iter().enumerate().rev() {
            // The gate's own adjoint is complete: take it and release its register.
            let g = match reg[i] {
                NONE => V::ZERO,
                r => {
                    free.push(r);
                    file[r as usize]
                }
            };
            match gate {
                Gate::Free(s) => {
                    let s = s as usize;
                    m[s] += g + a * u[s];
                    m[c] += b * u[s];
                }
                Gate::Xor(x, y) => {
                    add(x, g, &mut reg, &mut file, &mut free);
                    add(y, g, &mut reg, &mut file, &mut free);
                }
                Gate::And(x, y, s) => {
                    let s = s as usize;
                    m[s] += g;
                    add(x, a * u[s], &mut reg, &mut file, &mut free);
                    add(y, b * u[s], &mut reg, &mut file, &mut free);
                }
                Gate::Copy(x, s) => {
                    let s = s as usize;
                    m[s] += g;
                    add(x, a * u[s], &mut reg, &mut file, &mut free);
                    m[c] += b * u[s];
                }
            }
        }
    }
}

/// Gates past which the backward walk keeps live adjoints in registers.
const LARGE_GATES: usize = 1 << 25;

/// Append gates at a fixed width: a kind byte, then three big-endian words.
fn encode_gates(gates: &[Gate], out: &mut Vec<u8>) {
    for &gate in gates {
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
