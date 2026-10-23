//! Flock proofs verified inside Flock circuits: binary recursion.
//!
//! [`WireArith`] runs the Flock verifier written once over [`FlockArith`](crate::FlockArith) on the wires of a Flock circuit.
//!
//! [`Blake3Mirror`] mirrors the BLAKE3 transcript and the Merkle openings of [`FlockMmcs`](crate::opening::FlockMmcs) are recomputed on wires.
//!
//! The lincheck's matrix evaluation is never done in the circuit: it becomes a [`MatrixClaim`] the final verifier checks natively.
//!
//! The module lives in the crate because the deferral walks the gate list and the circuit grows ports the public builder does not expose.

mod arith;
mod blake3;
mod challenger;
mod deferred;
mod gates;
mod keccak;
mod mmcs;
mod node;
mod statement;
mod value;

#[cfg(test)]
mod tests;

use alloc::vec::Vec;

use p3_blake3::Blake3;
use thiserror::Error;

pub use self::arith::{Bit, WireArith};
pub use self::challenger::Blake3Mirror;
pub use self::deferred::{
    DeferralError, FormProof, Matrix, MatrixClaim, PendingForm, evaluate_claim, fold_claims_in,
    line_polynomial, native_form, prove_form, reduce_form_in,
};
use self::gates::Gates;
pub use self::gates::{BinaryElement, Byte, Cost};
pub use self::mmcs::CapTarget;
pub use self::node::{
    Child, NodeCircuit, NodeError, NodeLayout, NodeOutput, NodeProof, Slot, build_node,
    leaf_digest, node_instance, prove_node, verify_root,
};
pub use self::statement::WireStatement;
pub use self::value::{Bin, Gf64, Gf192};
use crate::circuit::{Circuit, Port};
use crate::{
    Batch, BatchError, FlockChallenger, FlockConfig, FlockProof, ProvingError, PublicPorts,
    VerificationError, Witness,
};

/// A finished recursion circuit, with the advice that satisfies it and the revealed bits its statement pins.
#[derive(Debug)]
pub struct RecursionCircuit {
    /// The circuit: one instance, its inputs the advice and its outputs the revealed bits.
    pub circuit: Circuit,
    /// The advice bits, packed 64 per word.
    pub advice: Vec<u64>,
    /// The revealed bits the statement pins, packed 64 per word.
    pub revealed: Vec<u64>,
    /// Products and gates per component.
    pub costs: Vec<Cost>,
    /// Products of the circuit.
    pub ands: usize,
    /// Gates of the circuit, XORs included.
    pub gates: usize,
    /// Operations of its compiled walk.
    pub ops: usize,
    /// Terms of its compiled walk's linear forms.
    pub terms: usize,
}

impl WireArith {
    /// The finished circuit.
    ///
    /// Values still alive keep their wires, which no longer name a circuit.
    ///
    /// # Panics
    ///
    /// When a transcript value was never consumed.
    #[must_use]
    pub fn finish(self) -> RecursionCircuit {
        self.finish_padded(None, None)
    }

    /// As [`Self::finish`], the advice padded to `advice_words` and the instance to `2^k_log` bits.
    ///
    /// Circuits padded alike share their shape, so a verifier of one verifies the other.
    ///
    /// # Panics
    ///
    /// When the circuit is already past a target.
    #[must_use]
    #[tracing::instrument(skip_all, name = "finish circuit")]
    pub fn finish_padded(
        self,
        advice_words: Option<usize>,
        k_log: Option<usize>,
    ) -> RecursionCircuit {
        let costs = self.costs();
        let mut gates = core::mem::replace(&mut *self.gates.lock(), Gates::new());
        // The OR of every assertion, pinned to zero last.
        let fail = gates.fail;
        gates.reveal(fail, false);
        if let Some(words) = advice_words {
            let bits = gates.builder.advice_bits();
            assert!(bits <= 64 * words, "the advice fits its padding");
            let _ = gates.advice(core::iter::repeat_n(false, 64 * words - bits));
        }
        if let Some(k_log) = k_log {
            // Products start past the ports, which move up by the advice and revealed words.
            let ports = 64
                * (gates.builder.advice_bits().div_ceil(64)
                    + gates.builder.revealed_bits().div_ceil(64));
            let floor = (1usize << (k_log - 1)) + 1;
            gates.builder.reserve_slots(floor.saturating_sub(ports));
        }
        assert!(
            gates.pending.is_empty(),
            "every announced value is absorbed"
        );
        assert!(gates.sampled.is_empty(), "every sample is taken");
        let gate_count = gates.builder.num_gates();
        // Every pass runs the compiled walk, so the gate list goes as it compiles.
        let circuit = gates.builder.finish_compiled();
        if let Some(k_log) = k_log {
            assert_eq!(circuit.k_log(), k_log, "the circuit fits its padding");
        }
        RecursionCircuit {
            ands: circuit.num_ands(),
            gates: gate_count,
            ops: circuit.num_ops(),
            terms: circuit.num_terms(),
            circuit,
            advice: gates.advice,
            revealed: gates.revealed,
            costs,
        }
    }
}

/// Why a proof cannot be verified in a circuit.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RecursionError {
    /// The proof is rejected.
    #[error(transparent)]
    Verification(#[from] VerificationError),
    /// The deferral of its matrices is rejected.
    #[error(transparent)]
    Deferral(#[from] DeferralError),
}

impl WireArith {
    /// Verify a Flock proof over the mirrored BLAKE3 transcript, and defer its matrices to one claim.
    ///
    /// The transcript continues past the proof, so later reductions draw from it.
    ///
    /// Every proof value is advice; the statement is a constant of the circuit, or the given wires.
    ///
    /// # Errors
    ///
    /// When the proof or the deferral is rejected.
    #[tracing::instrument(skip_all, name = "verify in circuit")]
    pub fn verify_proof(
        &mut self,
        config: &FlockConfig,
        batch: &Batch<'_>,
        public: &PublicPorts,
        proof: &FlockProof<Blake3>,
        mirror: &mut Blake3Mirror,
        statement: Option<WireStatement>,
    ) -> Result<MatrixClaim<Gf192>, RecursionError> {
        self.component("field");
        self.statement = statement.map(|s| (s, public.values().to_vec()));
        let verified = crate::verify_in(self, config, batch, public, proof, mirror);
        self.statement = None;
        self.gates.lock().substitutions.clear();
        verified?;
        self.component("deferral");
        let form = self.forms.pop().expect("the lincheck defers its form");
        let native = native_form(self, &form);
        let matrix = self.matrix(batch.circuit());
        let form_proof = prove_form(&matrix, &native, &mut mirror.native().clone());
        let claim = reduce_form_in(self, &form, &form_proof, mirror)?;
        self.component("field");
        Ok(claim)
    }
}

impl WireArith {
    /// Every public word of a statement as fresh advice.
    #[must_use]
    pub fn advice_statement(&self, public: &PublicPorts) -> WireStatement {
        let len = public.words().len();
        let values: Vec<bool> = (0..1usize << public.log_instances())
            .flat_map(|i| {
                public.instance(i)[..len]
                    .iter()
                    .flat_map(|&w| (0..64).map(move |b| (w >> b) & 1 == 1))
                    .collect::<Vec<_>>()
            })
            .collect();
        let bits = if len == 0 {
            Vec::new()
        } else {
            self.advice_bits(&values)
        };
        WireStatement {
            bits,
            circuit_digest: None,
        }
    }
}

impl RecursionCircuit {
    /// The batch of the one instance.
    ///
    /// # Errors
    ///
    /// When the circuit is too small for a proof.
    pub const fn batch(&self) -> Result<Batch<'_>, BatchError> {
        Batch::new(&self.circuit, 0)
    }

    /// The statement: every revealed bit, pinned.
    #[must_use]
    pub fn public(&self, batch: &Batch<'_>, revealed: &[u64]) -> PublicPorts {
        let port = self.revealed_port();
        let range = self.circuit.port(port).expect("a revealed port");
        PublicPorts::new(batch, &[port], |_, words| {
            for (w, &v) in words[range.clone()].iter_mut().zip(revealed) {
                *w = v;
            }
        })
    }

    /// The revealed port: the last output port.
    #[must_use]
    pub fn revealed_port(&self) -> Port {
        let mut i = 0;
        while self.circuit.port(Port::Output(i + 1)).is_some() {
            i += 1;
        }
        Port::Output(i)
    }

    /// The witness of the one instance, from the advice.
    #[must_use]
    pub fn witness(&self, advice: &[u64]) -> Witness {
        Witness::generate(&self.circuit, 0, &[()], &(), |(), words| {
            for (w, &v) in words.iter_mut().zip(advice) {
                *w = v;
            }
        })
    }

    /// Prove the circuit on its advice, over a BLAKE3 transcript seeded with `initial`.
    ///
    /// # Errors
    ///
    /// When the advice violates the statement, or the proof cannot be made.
    pub fn prove(
        &self,
        config: &FlockConfig,
        initial: Vec<u8>,
    ) -> Result<FlockProof<Blake3>, ProvingError> {
        let batch = self.batch().expect("a recursion circuit has enough slots");
        let public = self.public(&batch, &self.revealed);
        let witness = self.witness(&self.advice);
        let mut challenger = FlockChallenger::<Blake3>::from_hasher(initial, Blake3);
        crate::prove(config, &batch, &public, witness, &mut challenger)
    }
}
