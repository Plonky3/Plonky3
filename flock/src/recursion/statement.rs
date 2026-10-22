//! Statements whose public words are wires: the verified proof's statement is advice of the circuit, not a constant.
//!
//! The circuit recomputes the statement's Keccak-256 digest and the transcript absorbs those wires where its seed holds the digest.

use alloc::vec::Vec;

use p3_challenger::CanObserve;
use p3_verifier_arith::Lift;

use super::arith::pack_bytes;
use super::gates::Byte;
use super::keccak::keccak256;
use super::{Bit, WireArith};
use crate::ports::{DIGEST_INSTANCES, PortsError, PublicPorts};
use crate::{Batch, Val};

/// The public words of a verified proof's statement, and its circuit's digest when that is advice too.
#[derive(Clone, Debug)]
pub struct WireStatement {
    /// Each instance's public words, 64 bits each, in the order of [`PublicPorts::words`], instance-major.
    pub bits: Vec<Bit>,
    /// The verified circuit's digest, when the circuit is not a constant.
    pub circuit_digest: Option<[Byte; 32]>,
}

impl WireArith {
    /// Bind a statement whose words are wires: its digest on wires, held in place of the constant in every seed that names it.
    pub(crate) fn bind_wire_statement<C: CanObserve<Val>>(
        &mut self,
        statement: &WireStatement,
        batch: &Batch<'_>,
        public: &PublicPorts,
        challenger: &mut C,
    ) -> Result<(), PortsError> {
        public.check(batch)?;
        let previous = self.component("statement");
        let gates_handle = alloc::sync::Arc::clone(&self.gates);
        let mut gates = gates_handle.lock();
        let len = public.words().len();
        if len > 0 {
            // The digest of the shape, then of each chunk of instances' words.
            let mut preimage: Vec<Byte> = public
                .shape_bytes()
                .into_iter()
                .map(|b| Byte {
                    wires: core::array::from_fn(|i| gates.constant((b >> i) & 1 == 1)),
                    value: b,
                })
                .collect();
            for chunk in statement.bits.chunks(64 * len * DIGEST_INSTANCES) {
                let chunk_bytes = pack_bytes(chunk);
                preimage.extend(keccak256(&mut gates.builder, &chunk_bytes));
            }
            let digest = keccak256(&mut gates.builder, &preimage);
            assert_eq!(
                digest.map(|b| b.value),
                *public.digest(),
                "the statement's digest"
            );
            gates.substitutions.push((*public.digest(), digest));
            self.statement_digest = Some(digest);
        }
        if let Some(digest) = &statement.circuit_digest {
            gates
                .substitutions
                .push((*batch.circuit().digest(), *digest));
        }
        gates.component(previous);
        drop(gates);
        self.seeding(|| public.bind(batch, challenger))
    }

    /// The public table folded over its instances, from the statement's wires.
    pub(crate) fn fold_wire_bits(
        &self,
        statement: &WireStatement,
        public_words: usize,
        weights: &[super::Gf192],
    ) -> Vec<super::Gf192> {
        let per_instance = 64 * public_words;
        (0..per_instance)
            .map(|i| {
                weights
                    .iter()
                    .enumerate()
                    .map(|(o, w)| self.scale(&statement.bits[o * per_instance + i], w.clone()))
                    .sum()
            })
            .collect()
    }
}
