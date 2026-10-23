//! Aggregation trees: a node instance verifies one proof and carries one claim per matrix below it.
//!
//! A node proof is a batch of two node instances, so a tree of node proofs is binary and every level verifies proofs of the level below.
//!
//! A node instance outputs the digest of the leaf statements below it and, per slot, a claim on one circuit's zero-extended stacked matrix.
//!
//! Its statement is only the BLAKE3 commitment to that output and the fail bit: the parent takes the output as advice and checks the commitment.
//!
//! Claims on the same matrix fold into one (Thaler, "Proofs, Arguments, and Zero-Knowledge", Section 4.5.2), so a slot never holds more than one.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use p3_blake3::Blake3;
use p3_field::{Dup, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;
use p3_symmetric::CryptographicHasher;
use p3_verifier_arith::Arith;

use super::blake3::hash;
use super::gates::{BinaryElement, Byte};
use super::{
    Bit, Blake3Mirror, Gf192, Matrix, MatrixClaim, RecursionCircuit, RecursionError, WireArith,
    fold_claims_in, line_polynomial,
};
use crate::circuit::Wire;
use crate::{
    Batch, BatchError, Challenge, FlockChallenger, FlockConfig, FlockProof, ProvingError,
    PublicPorts, Witness,
};

/// The shape every node instance of a tree shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeLayout {
    /// Matrices an output carries claims on.
    pub slots: usize,
    /// Row and column coordinates of every carried claim; a smaller matrix is zero-extended.
    pub k_max: usize,
    /// Advice words of every node instance circuit, padded to one shape when set.
    pub advice_words: Option<usize>,
    /// Bits per node instance, padded to one shape when set.
    pub k_log: Option<usize>,
}

impl NodeLayout {
    /// Coordinates of a carried claim's point.
    const fn coordinates(&self) -> usize {
        1 + 2 * self.k_max
    }
}

/// A claim carried for one matrix: its circuit's digest and the claim on the zero-extended stacked matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    /// The circuit's digest.
    pub digest: [u8; 32],
    /// The claim, of `1 + 2 k_max` coordinates.
    pub claim: MatrixClaim,
}

/// A node instance's output: the digest of the leaf statements below it, and one claim per matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeOutput {
    /// BLAKE3 over the leaf statements' digests, pairwise up the tree.
    pub leaf_acc: [u8; 32],
    /// Claims by slot.
    pub slots: Vec<Option<Slot>>,
}

impl NodeOutput {
    /// The encoding the commitment hashes: the leaf digest, then per slot a presence byte, the digest, the point and the value.
    #[must_use]
    pub fn encode(&self, layout: &NodeLayout) -> Vec<u8> {
        let mut out = self.leaf_acc.to_vec();
        for slot in &self.slots {
            out.push(u8::from(slot.is_some()));
            let digest = slot.as_ref().map_or([0; 32], |s| s.digest);
            out.extend_from_slice(&digest);
            let zero = vec![Challenge::ZERO; layout.coordinates()];
            let (point, value) = slot
                .as_ref()
                .map_or((zero.as_slice(), Challenge::ZERO), |s| {
                    (s.claim.point.as_slice(), s.claim.value)
                });
            for c in point.iter().chain([&value]) {
                out.extend((0..24).map(|byte| {
                    (0..8).fold(0u8, |acc, b| acc | (u8::from(c.bit(8 * byte + b)) << b))
                }));
            }
        }
        out
    }

    /// The commitment a node statement reveals.
    #[must_use]
    pub fn commitment(&self, layout: &NodeLayout) -> [u8; 32] {
        Blake3.hash_slice(&self.encode(layout))
    }
}

/// A claim slot on wires.
#[derive(Clone, Debug)]
struct WireSlot {
    /// Whether the slot holds a claim.
    present: Bit,
    /// The circuit's digest.
    digest: [Byte; 32],
    /// The claim.
    claim: MatrixClaim<Gf192>,
}

/// A proof a node instance verifies.
///
/// Every proof's BLAKE3 transcript starts empty.
#[derive(Debug)]
pub enum Child<'a> {
    /// Any Flock proof over BLAKE3: its statement is advice, and its digest a leaf of the tree.
    Leaf {
        /// Its parameters.
        config: &'a FlockConfig,
        /// Its batch.
        batch: Batch<'a>,
        /// Its statement.
        public: &'a PublicPorts,
        /// The proof.
        proof: &'a FlockProof<Blake3>,
    },
    /// A node proof, with its instances' outputs.
    Node {
        /// Its parameters.
        config: &'a FlockConfig,
        /// The node proof.
        node: &'a NodeProof,
    },
}

/// A proof of two node instances, with what verifies it.
#[derive(Debug)]
pub struct NodeProof {
    /// The node instance circuit.
    pub circuit: RecursionCircuit,
    /// The statement: each instance's commitment and fail bit.
    pub public: PublicPorts,
    /// The proof.
    pub proof: FlockProof<Blake3>,
    /// Each instance's output.
    pub outputs: [NodeOutput; 2],
}

/// `true` when every bit of the two is equal, as a bit.
fn equal_bytes(arith: &WireArith, a: &[Byte], b: &[Byte]) -> Bit {
    arith.with_gates(|g| {
        let mut differ = Wire::ZERO;
        let mut value = true;
        for (x, y) in a.iter().zip(b) {
            value &= x.value == y.value;
            for (&wx, &wy) in x.wires.iter().zip(&y.wires) {
                let d = g.builder.xor(wx, wy);
                differ = g.builder.or(differ, d);
            }
        }
        let wire = g.builder.not(differ);
        Bit { wire, value }
    })
}

/// `x AND y`.
fn and(arith: &WireArith, x: Bit, y: Bit) -> Bit {
    let wire = arith.with_gates(|g| g.builder.and(x.wire, y.wire));
    Bit {
        wire,
        value: x.value && y.value,
    }
}

/// `x OR y`.
fn or(arith: &WireArith, x: Bit, y: Bit) -> Bit {
    let wire = arith.with_gates(|g| g.builder.or(x.wire, y.wire));
    Bit {
        wire,
        value: x.value || y.value,
    }
}

/// `NOT x`.
fn not(arith: &WireArith, x: Bit) -> Bit {
    let wire = arith.with_gates(|g| g.builder.not(x.wire));
    Bit {
        wire,
        value: !x.value,
    }
}

/// `then` where the bit is one and `otherwise` where it is zero, byte by byte.
fn select_bytes(
    arith: &WireArith,
    bit: Bit,
    then: &[Byte; 32],
    otherwise: &[Byte; 32],
) -> [Byte; 32] {
    arith.with_gates(|g| {
        core::array::from_fn(|i| Byte {
            wires: core::array::from_fn(|b| {
                g.builder
                    .mux(bit.wire, then[i].wires[b], otherwise[i].wires[b])
            }),
            value: if bit.value {
                then[i].value
            } else {
                otherwise[i].value
            },
        })
    })
}

/// `then` where the bit is one and `otherwise` where it is zero, coordinate by coordinate.
fn select_claim(
    arith: &mut WireArith,
    bit: Bit,
    then: &MatrixClaim<Gf192>,
    otherwise: &MatrixClaim<Gf192>,
) -> MatrixClaim<Gf192> {
    MatrixClaim {
        point: then
            .point
            .iter()
            .zip(&otherwise.point)
            .map(|(t, o)| arith.select(&bit, t.dup(), o.dup()))
            .collect(),
        value: arith.select(&bit, then.value.dup(), otherwise.value.dup()),
    }
}

/// The native claim behind wires.
fn native_claim(claim: &MatrixClaim<Gf192>) -> MatrixClaim {
    MatrixClaim {
        point: claim.point.iter().map(Gf192::value).collect(),
        value: claim.value.value(),
    }
}

/// The restriction the prover sends for a fold: the true one when both claims are on `matrix`, a line through the values otherwise.
///
/// A fold whose result the circuit discards only needs `q(0)` and `q(1)` right.
fn fold_message(
    matrix: Option<&Matrix>,
    first: &MatrixClaim,
    second: &MatrixClaim,
) -> Vec<Challenge> {
    let n = first.point.len();
    let mut q = match matrix {
        Some(m) if first != second => line_polynomial(m, first, second),
        _ if first == second => vec![first.value],
        _ => vec![first.value, first.value + second.value],
    };
    q.resize(n + 1, Challenge::ZERO);
    q
}

/// A claim on a smaller matrix, its point zero-extended to `k_max` row and column coordinates.
fn extend(claim: MatrixClaim<Gf192>, k: usize, k_max: usize) -> MatrixClaim<Gf192> {
    let pad = core::iter::repeat_n(Gf192::ZERO, k_max - k);
    let point = core::iter::once(claim.point[0].dup())
        .chain(pad.clone())
        .chain(claim.point[1..=k].iter().map(Dup::dup))
        .chain(pad)
        .chain(claim.point[1 + k..].iter().map(Dup::dup))
        .collect();
    MatrixClaim {
        point,
        value: claim.value,
    }
}

/// Advice bytes carrying `bytes`.
fn advice_bytes(arith: &WireArith, bytes: &[u8]) -> Vec<Byte> {
    let wires = arith.with_gates(|g| {
        g.advice(
            bytes
                .iter()
                .flat_map(|&b| (0..8).map(move |i| (b >> i) & 1 == 1)),
        )
    });
    bytes
        .iter()
        .enumerate()
        .map(|(i, &value)| Byte {
            wires: core::array::from_fn(|b| wires[8 * i + b]),
            value,
        })
        .collect()
}

/// A challenge-field value from 24 little-endian bytes.
fn value_of(arith: &WireArith, bytes: &[Byte]) -> Gf192 {
    let bits: Vec<Bit> = bytes
        .iter()
        .flat_map(|b| {
            (0..8).map(move |i| Bit {
                wire: b.wires[i],
                value: (b.value >> i) & 1 == 1,
            })
        })
        .collect();
    arith.value_of_bits(&bits)
}

/// The slots of an output given as advice bytes, and the leaf digest, parsed.
fn parse_output(
    arith: &WireArith,
    layout: &NodeLayout,
    bytes: &[Byte],
) -> ([Byte; 32], Vec<WireSlot>) {
    let leaf_acc: [Byte; 32] = bytes[..32].try_into().expect("a digest");
    let per_slot = 1 + 32 + 24 * (layout.coordinates() + 1);
    let slots = bytes[32..]
        .chunks_exact(per_slot)
        .map(|slot| {
            let present = Bit {
                wire: slot[0].wires[0],
                value: slot[0].value & 1 == 1,
            };
            let digest: [Byte; 32] = slot[1..33].try_into().expect("a digest");
            let mut values: Vec<Gf192> = slot[33..]
                .as_chunks::<24>()
                .0
                .iter()
                .map(|c| value_of(arith, c))
                .collect();
            let value = values.pop().expect("a value");
            WireSlot {
                present,
                digest,
                claim: MatrixClaim {
                    point: values,
                    value,
                },
            }
        })
        .collect();
    (leaf_acc, slots)
}

/// The native output behind wire slots.
fn native_output(leaf_acc: &[Byte; 32], slots: &[WireSlot]) -> NodeOutput {
    NodeOutput {
        leaf_acc: leaf_acc.map(|b| b.value),
        slots: slots
            .iter()
            .map(|s| {
                s.present.value.then(|| Slot {
                    digest: s.digest.map(|b| b.value),
                    claim: native_claim(&s.claim),
                })
            })
            .collect(),
    }
}

/// The encoding of wire slots, as [`NodeOutput::encode`] lays it out.
fn encode_output(arith: &WireArith, leaf_acc: &[Byte; 32], slots: &[WireSlot]) -> Vec<Byte> {
    let mut out = leaf_acc.to_vec();
    for slot in slots {
        let zero = arith.with_gates(|g| g.constant(false));
        out.push(Byte {
            wires: core::array::from_fn(|i| if i == 0 { slot.present.wire } else { zero }),
            value: u8::from(slot.present.value),
        });
        out.extend_from_slice(&slot.digest);
        for c in slot.claim.point.iter().chain([&slot.claim.value]) {
            out.extend(arith.bytes(c));
        }
    }
    out
}

/// Look up a compiled matrix by circuit digest.
fn find<'m>(matrices: &'m [Arc<Matrix>], digest: &[u8; 32]) -> Option<&'m Matrix> {
    matrices
        .iter()
        .find(|m| m.digest() == digest)
        .map(AsRef::as_ref)
}

impl WireArith {
    /// Fold two claims on wires, the prover's restriction computed from `matrix` when they share it.
    fn fold(
        &mut self,
        mirror: &mut Blake3Mirror,
        matrix: Option<&Matrix>,
        first: &MatrixClaim<Gf192>,
        second: &MatrixClaim<Gf192>,
    ) -> Result<MatrixClaim<Gf192>, RecursionError> {
        let q = fold_message(matrix, &native_claim(first), &native_claim(second));
        Ok(fold_claims_in(self, first, second, &q, mirror)?)
    }

    /// Merge two slots carried for the same position: their claims fold when both are present, their digests must agree.
    fn merge_slots(
        &mut self,
        mirror: &mut Blake3Mirror,
        matrices: &[Arc<Matrix>],
        a: &WireSlot,
        b: &WireSlot,
    ) -> Result<WireSlot, RecursionError> {
        let both = and(self, a.present, b.present);
        let present = or(self, a.present, b.present);
        let same = equal_bytes(self, &a.digest, &b.digest);
        let differ = not(self, same);
        let conflict = and(self, both, differ);
        self.with_gates(|g| g.assert_zero(conflict.wire));
        let digest = select_bytes(self, a.present, &a.digest, &b.digest);
        // An absent side stands in for itself, so a lone claim folds with a copy of itself.
        let first = select_claim(self, a.present, &a.claim, &b.claim);
        let second = select_claim(self, b.present, &b.claim, &a.claim);
        let matrix = (both.value && same.value)
            .then(|| find(matrices, &a.digest.map(|x| x.value)))
            .flatten();
        let folded = self.fold(mirror, matrix, &first, &second)?;
        // An absent slot holds zeros, so its encoding is canonical.
        let claim = MatrixClaim {
            point: folded
                .point
                .into_iter()
                .map(|c| self.scale(&present, c))
                .collect(),
            value: self.scale(&present, folded.value),
        };
        Ok(WireSlot {
            present,
            digest,
            claim,
        })
    }

    /// Put a claim on the matrix of `digest` into the slot holding that matrix, or the first empty one.
    fn insert_claim(
        &mut self,
        mirror: &mut Blake3Mirror,
        matrices: &[Arc<Matrix>],
        slots: &[WireSlot],
        digest: &[Byte; 32],
        claim: &MatrixClaim<Gf192>,
    ) -> Result<Vec<WireSlot>, RecursionError> {
        let matches: Vec<Bit> = slots
            .iter()
            .map(|s| {
                let same = equal_bytes(self, &s.digest, digest);
                and(self, s.present, same)
            })
            .collect();
        let zero = Bit {
            wire: Wire::ZERO,
            value: false,
        };
        let any = matches.iter().fold(zero, |acc, &m| or(self, acc, m));
        let none = not(self, any);
        let mut earlier_empty = zero;
        let mut placed = zero;
        let mut out = Vec::with_capacity(slots.len());
        for (slot, m) in slots.iter().zip(matches) {
            let empty = not(self, slot.present);
            let seen = not(self, earlier_empty);
            let first_empty = and(self, empty, seen);
            earlier_empty = or(self, earlier_empty, empty);
            let fresh = and(self, none, first_empty);
            let target = or(self, m, fresh);
            placed = or(self, placed, target);
            let first = select_claim(self, slot.present, &slot.claim, claim);
            let matrix = (m.value)
                .then(|| find(matrices, &digest.map(|x| x.value)))
                .flatten();
            let folded = self.fold(mirror, matrix, &first, claim)?;
            out.push(WireSlot {
                present: or(self, slot.present, target),
                digest: select_bytes(self, target, digest, &slot.digest),
                claim: select_claim(self, target, &folded, &slot.claim),
            });
        }
        // A claim with no slot left is an error the statement exposes.
        let unplaced = not(self, placed);
        self.with_gates(|g| g.assert_zero(unplaced.wire));
        Ok(out)
    }
}

impl NodeProof {
    /// The batch of the two instances.
    ///
    /// # Errors
    ///
    /// When the circuit is too small for a proof.
    pub const fn batch(&self) -> Result<Batch<'_>, BatchError> {
        Batch::new(&self.circuit.circuit, 1)
    }
}

/// The statement of two node instances: each one's revealed words.
fn node_public(
    circuit: &RecursionCircuit,
    batch: &Batch<'_>,
    revealed: [&[u64]; 2],
) -> PublicPorts {
    let port = circuit.revealed_port();
    let range = circuit.circuit.port(port).expect("a revealed port");
    PublicPorts::new(batch, &[port], |i, words| {
        for (w, &v) in words[range.clone()].iter_mut().zip(revealed[i]) {
            *w = v;
        }
    })
}

/// Build the node instance circuit verifying one child, with the advice its proof satisfies.
///
/// `matrices` holds the compiled matrices of every circuit a carried claim may name, for the folds.
///
/// # Errors
///
/// When the child is rejected.
#[tracing::instrument(skip_all, name = "node instance")]
pub fn node_instance(
    layout: &NodeLayout,
    child: &Child<'_>,
    matrices: &[Arc<Matrix>],
) -> Result<(RecursionCircuit, NodeOutput), RecursionError> {
    let mut arith = WireArith::new().with_matrices(matrices);
    let mut mirror = Blake3Mirror::new(&arith, Vec::new());
    let zero_slot = |arith: &WireArith| {
        let zero = arith.with_gates(|g| g.constant(false));
        WireSlot {
            present: Bit {
                wire: Wire::ZERO,
                value: false,
            },
            digest: core::array::from_fn(|_| Byte {
                wires: [zero; 8],
                value: 0,
            }),
            claim: MatrixClaim {
                point: vec![Gf192::ZERO; layout.coordinates()],
                value: Gf192::ZERO,
            },
        }
    };
    let (leaf_acc, slots, digest, claim, k) = match child {
        Child::Leaf {
            config,
            batch,
            public,
            proof,
        } => {
            let statement = arith.advice_statement(public);
            let claim =
                arith.verify_proof(config, batch, public, proof, &mut mirror, Some(statement))?;
            let leaf_acc = arith
                .statement_digest
                .take()
                .expect("a leaf statement has public words");
            let digest = {
                let d = *batch.circuit().digest();
                arith.with_gates(|g| {
                    core::array::from_fn(|i| Byte {
                        wires: core::array::from_fn(|b| g.constant((d[i] >> b) & 1 == 1)),
                        value: d[i],
                    })
                })
            };
            let slots = (0..layout.slots).map(|_| zero_slot(&arith)).collect();
            (leaf_acc, slots, digest, claim, batch.circuit().k_log())
        }
        Child::Node { config, node } => {
            let batch = node.batch().expect("a node proof has a batch");
            let circuit_digest = advice_bytes(&arith, batch.circuit().digest());
            let digest: [Byte; 32] = circuit_digest.try_into().expect("a digest");
            let mut statement = arith.advice_statement(&node.public);
            statement.circuit_digest = Some(digest);
            let bits = statement.bits.clone();
            let claim = arith.verify_proof(
                config,
                &batch,
                &node.public,
                &node.proof,
                &mut mirror,
                Some(statement),
            )?;
            arith.statement_digest = None;
            // Each instance's output is advice, held to the commitment its statement reveals.
            let per_instance = bits.len() / 2;
            let mut outputs = Vec::with_capacity(2);
            for (i, output) in node.outputs.iter().enumerate() {
                let words = &bits[i * per_instance..(i + 1) * per_instance];
                let bytes = advice_bytes(&arith, &output.encode(layout));
                let committed = arith.with_gates(|g| hash(&mut g.builder, &bytes));
                for (c, byte) in committed.iter().zip(words.as_chunks::<8>().0) {
                    for (b, bit) in byte.iter().enumerate() {
                        arith.with_gates(|g| {
                            let d = g.builder.xor(c.wires[b], bit.wire);
                            g.assert_zero(d);
                        });
                    }
                }
                // The fail bit follows the commitment.
                let fail = words[256];
                arith.with_gates(|g| g.assert_zero(fail.wire));
                outputs.push(parse_output(&arith, layout, &bytes));
            }
            let leaf_input: Vec<Byte> = outputs[0].0.iter().chain(&outputs[1].0).copied().collect();
            let leaf_acc = arith.with_gates(|g| hash(&mut g.builder, &leaf_input));
            let mut slots = Vec::with_capacity(layout.slots);
            arith.component("folds");
            for (a, b) in outputs[0].1.iter().zip(&outputs[1].1) {
                slots.push(arith.merge_slots(&mut mirror, matrices, a, b)?);
            }
            (leaf_acc, slots, digest, claim, batch.circuit().k_log())
        }
    };
    arith.component("folds");
    let child_matrix = arith.matrix(match child {
        Child::Leaf { batch, .. } => batch.circuit(),
        Child::Node { node, .. } => &node.circuit.circuit,
    });
    let mut all: Vec<Arc<Matrix>> = matrices.to_vec();
    all.push(child_matrix);
    let claim = extend(claim, k, layout.k_max);
    let slots = arith.insert_claim(&mut mirror, &all, &slots, &digest, &claim)?;
    arith.component("commitment");
    let output = native_output(&leaf_acc, &slots);
    let encoded = encode_output(&arith, &leaf_acc, &slots);
    let commitment = arith.with_gates(|g| hash(&mut g.builder, &encoded));
    debug_assert_eq!(commitment.map(|b| b.value), output.commitment(layout));
    arith.with_gates(|g| {
        for byte in &commitment {
            for (b, &wire) in byte.wires.iter().enumerate() {
                g.reveal(wire, (byte.value >> b) & 1 == 1);
            }
        }
    });
    drop(mirror);
    Ok((
        arith.finish_padded(layout.advice_words, layout.k_log),
        output,
    ))
}

/// A node's instance circuit with the advice of its two instances, ready to prove.
#[derive(Debug)]
pub struct NodeCircuit {
    /// The instance circuit, built for the first child.
    pub circuit: RecursionCircuit,
    /// The second instance's advice and revealed bits.
    pub second: (Vec<u64>, Vec<u64>),
    /// Each instance's output.
    pub outputs: [NodeOutput; 2],
}

/// Build a node: one instance circuit, two instances each verifying one child.
///
/// # Errors
///
/// When a child is rejected.
///
/// # Panics
///
/// When the two instance circuits differ, which would make the circuit depend on a child's values.
pub fn build_node(
    layout: &NodeLayout,
    children: [&Child<'_>; 2],
    matrices: &[Arc<Matrix>],
) -> Result<NodeCircuit, NodeError> {
    // The second instance first, keeping only its advice, so one circuit is alive at a time.
    let (second, out1) = node_instance(layout, children[1], matrices)?;
    let digest = *second.circuit.digest();
    let RecursionCircuit {
        circuit,
        advice,
        revealed,
        ..
    } = second;
    drop(circuit);
    let second_values = (advice, revealed);
    let (first, out0) = node_instance(layout, children[0], matrices)?;
    assert_eq!(
        first.circuit.digest(),
        &digest,
        "a node instance circuit does not depend on its child's values"
    );
    Ok(NodeCircuit {
        circuit: first,
        second: second_values,
        outputs: [out0, out1],
    })
}

impl NodeCircuit {
    /// Prove both instances in one batch.
    ///
    /// # Errors
    ///
    /// When the circuit is too small or the proof cannot be made.
    #[tracing::instrument(skip_all, name = "prove node")]
    pub fn prove(self, config: &FlockConfig) -> Result<NodeProof, NodeError> {
        let batch = Batch::new(&self.circuit.circuit, 1)?;
        let public = node_public(
            &self.circuit,
            &batch,
            [&self.circuit.revealed, &self.second.1],
        );
        let advice = [self.circuit.advice.as_slice(), self.second.0.as_slice()];
        let witness = Witness::generate(
            &self.circuit.circuit,
            1,
            &advice,
            &advice[0],
            |row, words| {
                for (w, &v) in words.iter_mut().zip(*row) {
                    *w = v;
                }
            },
        );
        let mut challenger = FlockChallenger::<Blake3>::from_hasher(Vec::new(), Blake3);
        let proof = crate::prove(config, &batch, &public, witness, &mut challenger)?;
        Ok(NodeProof {
            circuit: self.circuit,
            public,
            proof,
            outputs: self.outputs,
        })
    }
}

/// Build and prove a node: two instances, each verifying one child.
///
/// # Errors
///
/// When a child is rejected or the proof cannot be made.
pub fn prove_node(
    layout: &NodeLayout,
    config: &FlockConfig,
    children: [&Child<'_>; 2],
    matrices: &[Arc<Matrix>],
) -> Result<NodeProof, NodeError> {
    build_node(layout, children, matrices)?.prove(config)
}

/// Why a node cannot be made or is rejected.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum NodeError {
    /// A child is rejected.
    #[error(transparent)]
    Recursion(#[from] RecursionError),
    /// The node circuit is too small.
    #[error(transparent)]
    Batch(#[from] BatchError),
    /// The proof cannot be made.
    #[error(transparent)]
    Proving(#[from] ProvingError),
    /// The proof is rejected.
    #[error(transparent)]
    Verification(#[from] crate::VerificationError),
    /// An output does not open its commitment.
    #[error("an output does not open its commitment")]
    Commitment,
    /// A carried claim names no known circuit.
    #[error("a carried claim names no known circuit")]
    UnknownMatrix,
    /// A carried claim is false.
    #[error("a carried claim is false")]
    Claim,
    /// The leaf statements are not the ones the tree commits to.
    #[error("the leaf statements are not the ones the tree commits to")]
    Leaves,
}

/// The final native check of a tree: the root proof, then every carried claim, then the leaf digest.
///
/// `leaf_acc` is the expected digest of the leaf statements, `matrices` every circuit a claim may name.
///
/// # Errors
///
/// The first check that fails.
#[tracing::instrument(skip_all, name = "verify root")]
pub fn verify_root(
    layout: &NodeLayout,
    config: &FlockConfig,
    root: &NodeProof,
    matrices: &[Arc<Matrix>],
    leaf_acc: &[u8; 32],
) -> Result<(), NodeError> {
    let batch = root.batch()?;
    // The statement is rebuilt from the outputs: their commitments, and zero fail bits.
    let revealed: Vec<Vec<u64>> = root
        .outputs
        .iter()
        .map(|o| {
            let c = o.commitment(layout);
            let mut words: Vec<u64> = c
                .as_chunks::<8>()
                .0
                .iter()
                .map(|w| u64::from_le_bytes(*w))
                .collect();
            words.push(0);
            words
        })
        .collect();
    let public = node_public(&root.circuit, &batch, [&revealed[0], &revealed[1]]);
    if public != root.public {
        return Err(NodeError::Commitment);
    }
    // Every carried claim, grouped by matrix so that one walk evaluates a matrix's points together.
    let slots: Vec<&Slot> = root
        .outputs
        .iter()
        .flat_map(|o| o.slots.iter().flatten())
        .collect();
    let mut groups: Vec<(&Matrix, Vec<&Slot>)> = Vec::new();
    for slot in slots {
        let matrix = find(matrices, &slot.digest).ok_or(NodeError::UnknownMatrix)?;
        match groups
            .iter_mut()
            .find(|(m, _)| m.digest() == matrix.digest())
        {
            Some((_, group)) => group.push(slot),
            None => groups.push((matrix, vec![slot])),
        }
    }
    // The root proof and the matrix walks are independent, so they run at once.
    let (verified, claims) = join(
        || {
            let mut challenger = FlockChallenger::<Blake3>::from_hasher(Vec::new(), Blake3);
            crate::verify(config, &batch, &public, &root.proof, &mut challenger)
        },
        || {
            groups.par_iter().all(|(matrix, group)| {
                let points: Vec<Vec<Challenge>> =
                    group.iter().map(|s| s.claim.point.clone()).collect();
                matrix
                    .evaluate(&points)
                    .iter()
                    .zip(group)
                    .all(|(&v, s)| v == s.claim.value)
            })
        },
    );
    verified?;
    if !claims {
        return Err(NodeError::Claim);
    }
    let combined =
        Blake3.hash_slice(&[root.outputs[0].leaf_acc, root.outputs[1].leaf_acc].concat());
    if combined != *leaf_acc {
        return Err(NodeError::Leaves);
    }
    Ok(())
}

/// The digest a tree commits its leaf statements to: their statement digests, hashed pairwise up the tree.
///
/// Leaves are in tree order; their number is a power of two of at least two.
#[must_use]
pub fn leaf_digest(leaves: &[&PublicPorts]) -> [u8; 32] {
    let mut level: Vec<[u8; 32]> = leaves.iter().map(|p| *p.digest()).collect();
    while level.len() > 1 {
        level = level
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| Blake3.hash_slice(&[pair[0], pair[1]].concat()))
            .collect();
    }
    level[0]
}
