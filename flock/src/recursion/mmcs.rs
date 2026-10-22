//! Merkle openings of [`FlockMmcs`] over BLAKE3, checked on wires.
//!
//! The native check rejects every malformed opening and restores each query's path; the circuit then recomputes every path, the index bits steering each level.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use p3_blake3::Blake3;
use p3_commit::{ExtensionMmcs, ExtensionMmcsError, Mmcs, MmcsArith};
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeError;
use p3_symmetric::MerkleCap;
use p3_verifier_arith::Arith;

use super::blake3::hash;
use super::gates::{Byte, Gates};
use super::value::{Gf64, Gf192};
use super::{Bit, WireArith};
use crate::opening::{Commitment, FlockMmcs};
use crate::{Challenge, Val};

/// A commitment on wires: the digest bytes of every cap entry.
pub type CapTarget = Vec<[Byte; 32]>;

/// `then` where the bit is one and `otherwise` where it is zero, byte by byte.
fn select(gates: &mut Gates, bit: Bit, then: &[Byte; 32], otherwise: &[Byte; 32]) -> [Byte; 32] {
    core::array::from_fn(|i| Byte {
        wires: core::array::from_fn(|b| {
            gates
                .builder
                .mux(bit.wire, then[i].wires[b], otherwise[i].wires[b])
        }),
        value: if bit.value {
            then[i].value
        } else {
            otherwise[i].value
        },
    })
}

/// `compress(left, right)`, the hash of the two digests back to back.
fn compress(gates: &mut Gates, left: &[Byte; 32], right: &[Byte; 32]) -> [Byte; 32] {
    let input: Vec<Byte> = left.iter().chain(right).copied().collect();
    hash(&mut gates.builder, &input)
}

impl MmcsArith<WireArith, Gf64> for FlockMmcs<Blake3> {
    type Commitment = Commitment;
    type MultiProof = <Self as Mmcs<Val>>::MultiProof;
    type Target = CapTarget;
    type Error = MerkleTreeError;

    fn load_commitment(&self, arith: &mut WireArith, commitment: &Commitment) -> CapTarget {
        commitment
            .roots()
            .iter()
            .map(|digest| arith.advice_digest(digest))
            .collect()
    }

    fn load_key_commitment(&self, arith: &mut WireArith, commitment: &Commitment) -> CapTarget {
        let gates = arith.gates.lock();
        commitment
            .roots()
            .iter()
            .map(|digest| {
                core::array::from_fn(|i| Byte {
                    wires: core::array::from_fn(|b| gates.constant((digest[i] >> b) & 1 == 1)),
                    value: digest[i],
                })
            })
            .collect()
    }

    fn will_observe(&self, arith: &mut WireArith, commitment: &CapTarget) {
        arith
            .gates
            .lock()
            .pending
            .extend(commitment.iter().flatten().copied());
    }

    fn verify_rows(
        &self,
        arith: &mut WireArith,
        commitment: &CapTarget,
        dimensions: &[Dimensions],
        indices: &[Vec<Bit>],
        rows: &[Vec<Cow<'_, [Gf64]>>],
        proof: &Self::MultiProof,
    ) -> Result<(), MerkleTreeError> {
        // The native check rejects every malformed opening and restores each query's path.
        let native_indices: Vec<usize> = indices
            .iter()
            .map(|bits| {
                bits.iter()
                    .enumerate()
                    .filter(|(_, b)| b.value)
                    .map(|(i, _)| 1usize << i)
                    .sum()
            })
            .collect();
        let native_rows: Vec<Vec<Vec<Val>>> = rows
            .iter()
            .map(|query| {
                query
                    .iter()
                    .map(|row| row.iter().map(Gf64::value).collect())
                    .collect()
            })
            .collect();
        let native_commitment =
            MerkleCap::new(commitment.iter().map(|d| d.map(|b| b.value)).collect());
        self.verify_multi_batch(
            &native_commitment,
            dimensions,
            &native_indices,
            &native_rows,
            proof,
        )?;
        let paths =
            self.restore_and_recompute_paths(dimensions, &native_indices, &native_rows, proof)?;
        let plan = self.opening_plan(dimensions)?;

        let previous = arith.component("merkle");
        for ((bits, query_rows), path) in indices.iter().zip(rows).zip(paths) {
            let siblings: Vec<[Byte; 32]> = path
                .siblings
                .iter()
                .map(|d| arith.advice_digest(d))
                .collect();
            let gather = |arith: &WireArith, matrices: &[usize]| -> Vec<Byte> {
                matrices
                    .iter()
                    .flat_map(|&m| query_rows[m].iter().flat_map(|v| arith.bytes(v)))
                    .collect()
            };
            let leaf = gather(arith, &plan.leaf);
            let mut gates = arith.gates.lock();
            let mut digest = hash(&mut gates.builder, &leaf);
            drop(gates);
            for ((depth, level), sibling) in plan.levels.iter().enumerate().zip(&siblings) {
                debug_assert_eq!(level.arity, 2, "a binary tree compresses pairs");
                let bit = bits.get(depth).copied().unwrap_or(Bit {
                    wire: crate::circuit::Wire::ZERO,
                    value: false,
                });
                let injected = (!level.inject.is_empty()).then(|| gather(arith, &level.inject));
                let mut gates = arith.gates.lock();
                // The index bit says whether the running digest is the right child.
                let left = select(&mut gates, bit, sibling, &digest);
                let right = select(&mut gates, bit, &digest, sibling);
                digest = compress(&mut gates, &left, &right);
                if let Some(injected) = injected {
                    let injected = hash(&mut gates.builder, &injected);
                    digest = compress(&mut gates, &digest, &injected);
                }
            }

            // The index bits above the walked levels name the cap entry.
            let cap_bits = bits.get(plan.levels.len()..).unwrap_or(&[]);
            let mut candidates = commitment.clone();
            {
                let mut gates = arith.gates.lock();
                for &bit in cap_bits {
                    if candidates.len() == 1 {
                        break;
                    }
                    candidates = candidates
                        .chunks(2)
                        .map(|pair| select(&mut gates, bit, &pair[1], &pair[0]))
                        .collect();
                }
                for (d, r) in digest.iter().zip(&candidates[0]) {
                    for (&x, &y) in d.wires.iter().zip(&r.wires) {
                        let diff = gates.builder.xor(x, y);
                        gates.assert_zero(diff);
                    }
                }
            }
            if digest
                .iter()
                .zip(&candidates[0])
                .any(|(d, r)| d.value != r.value)
            {
                return Err(MerkleTreeError::CapMismatch);
            }
        }
        arith.component(previous);
        Ok(())
    }
}

impl MmcsArith<WireArith, Gf64> for &FlockMmcs<Blake3> {
    type Commitment = Commitment;
    type MultiProof = <FlockMmcs<Blake3> as Mmcs<Val>>::MultiProof;
    type Target = CapTarget;
    type Error = MerkleTreeError;

    fn load_commitment(&self, arith: &mut WireArith, commitment: &Commitment) -> CapTarget {
        (*self).load_commitment(arith, commitment)
    }

    fn load_key_commitment(&self, arith: &mut WireArith, commitment: &Commitment) -> CapTarget {
        (*self).load_key_commitment(arith, commitment)
    }

    fn will_observe(&self, arith: &mut WireArith, commitment: &CapTarget) {
        (*self).will_observe(arith, commitment);
    }

    fn verify_rows(
        &self,
        arith: &mut WireArith,
        commitment: &CapTarget,
        dimensions: &[Dimensions],
        indices: &[Vec<Bit>],
        rows: &[Vec<Cow<'_, [Gf64]>>],
        proof: &Self::MultiProof,
    ) -> Result<(), MerkleTreeError> {
        (*self).verify_rows(arith, commitment, dimensions, indices, rows, proof)
    }
}

impl MmcsArith<WireArith, Gf192> for ExtensionMmcs<Val, Challenge, &FlockMmcs<Blake3>> {
    type Commitment = Commitment;
    type MultiProof = <FlockMmcs<Blake3> as Mmcs<Val>>::MultiProof;
    type Target = CapTarget;
    type Error = ExtensionMmcsError<MerkleTreeError>;

    fn load_commitment(&self, arith: &mut WireArith, commitment: &Commitment) -> CapTarget {
        self.inner().load_commitment(arith, commitment)
    }

    fn load_key_commitment(&self, arith: &mut WireArith, commitment: &Commitment) -> CapTarget {
        self.inner().load_key_commitment(arith, commitment)
    }

    fn will_observe(&self, arith: &mut WireArith, commitment: &CapTarget) {
        self.inner().will_observe(arith, commitment);
    }

    fn verify_rows(
        &self,
        arith: &mut WireArith,
        commitment: &CapTarget,
        dimensions: &[Dimensions],
        indices: &[Vec<Bit>],
        rows: &[Vec<Cow<'_, [Gf192]>>],
        proof: &Self::MultiProof,
    ) -> Result<(), Self::Error> {
        // The native checks fix the shape; an extension row is committed as its coefficients.
        let base_dimensions = Self::multi_opening_shape(dimensions, indices.len(), rows)?;
        let base_rows: Vec<Vec<Cow<'_, [Gf64]>>> = rows
            .iter()
            .map(|query| {
                query
                    .iter()
                    .map(|row| Cow::Owned(row.iter().flat_map(|v| arith.coefficients(v)).collect()))
                    .collect()
            })
            .collect();
        self.inner()
            .verify_rows(
                arith,
                commitment,
                &base_dimensions,
                indices,
                &base_rows,
                proof,
            )
            .map_err(ExtensionMmcsError::Inner)
    }
}

impl WireArith {
    /// A digest the prover sent, as advice bytes.
    pub(crate) fn advice_digest(&self, digest: &[u8; 32]) -> [Byte; 32] {
        let bits: Vec<bool> = digest
            .iter()
            .flat_map(|&b| (0..8).map(move |i| (b >> i) & 1 == 1))
            .collect();
        let wires = self.with_gates(|g| g.advice(bits));
        core::array::from_fn(|i| Byte {
            wires: core::array::from_fn(|b| wires[8 * i + b]),
            value: digest[i],
        })
    }
}
