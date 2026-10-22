//! Merkle openings checked inside a circuit.

use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;

use p3_commit::{ExtensionMmcs, ExtensionMmcsError, Mmcs, MmcsArith};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, PrimeField64};
use p3_matrix::Dimensions;
use p3_merkle_tree::{MerkleTreeError, MerkleTreeMmcs};
use p3_symmetric::{CryptographicPermutation, MerkleCap, PaddingFreeSponge, TruncatedPermutation};
use p3_verifier_arith::{Arith, Lift};
use serde::{Deserialize, Serialize};

use crate::chip::sponge;
use crate::{Base, CircuitArith, CircuitBuilder, Ext, PermutationChip};

/// A Merkle commitment inside a circuit: the digest of every cap entry.
pub type CapTarget<F> = Vec<Vec<Base<F>>>;

/// [`TruncatedPermutation`] of two digests on wires.
fn compress<F, EF, P>(
    builder: &mut CircuitBuilder<F, EF, P>,
    left: &[Base<F>],
    right: &[Base<F>],
) -> Vec<Base<F>>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
{
    let mut state = vec![Base::ZERO; P::WIDTH];
    state[..left.len()].clone_from_slice(left);
    state[left.len()..left.len() + right.len()].clone_from_slice(right);
    let mut output = builder.permute(&state);
    output.truncate(left.len());
    output
}

/// `then` where the bit is one and `otherwise` where it is zero, entry by entry.
fn select_all<F: Field>(bit: &Base<F>, then: &[Base<F>], otherwise: &[Base<F>]) -> Vec<Base<F>> {
    then.iter()
        .zip(otherwise)
        .map(|(t, o)| o.clone() + (t.clone() - o.clone()) * bit.clone())
        .collect()
}

/// The cap entry the bits name, least significant bit first.
fn select_root<F: Field>(cap: &CapTarget<F>, bits: &[Base<F>]) -> Vec<Base<F>> {
    // Each bit halves the candidates: even entries where it is zero, odd ones where it is one.
    let mut candidates: Vec<Vec<Base<F>>> = cap.clone();
    for bit in bits {
        if candidates.len() == 1 {
            break;
        }
        candidates = candidates
            .chunks(2)
            .map(|pair| select_all(bit, &pair[1], &pair[0]))
            .collect();
    }
    candidates.swap_remove(0)
}

impl<F, EF, P, Perm, const WIDTH: usize, const RATE: usize, const DIGEST: usize>
    MmcsArith<CircuitArith<F, EF, P>, Base<F>>
    for MerkleTreeMmcs<
        F::Packing,
        F::Packing,
        PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST>,
        TruncatedPermutation<Perm, 2, DIGEST, WIDTH>,
        2,
        DIGEST,
    >
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    Perm: CryptographicPermutation<[F; WIDTH]> + CryptographicPermutation<[F::Packing; WIDTH]>,
    [F; DIGEST]: Serialize + for<'de> Deserialize<'de>,
{
    type Commitment = MerkleCap<F, [F; DIGEST]>;
    type MultiProof = <Self as Mmcs<F>>::MultiProof;
    type Target = CapTarget<F>;
    type Error = MerkleTreeError;

    fn load_commitment(
        &self,
        arith: &mut CircuitArith<F, EF, P>,
        commitment: &MerkleCap<F, [F; DIGEST]>,
    ) -> CapTarget<F> {
        commitment
            .roots()
            .iter()
            .map(|digest| digest.iter().map(|&value| arith.load(value)).collect())
            .collect()
    }

    fn load_key_commitment(
        &self,
        arith: &mut CircuitArith<F, EF, P>,
        commitment: &MerkleCap<F, [F; DIGEST]>,
    ) -> CapTarget<F> {
        commitment
            .roots()
            .iter()
            .map(|digest| digest.iter().map(|&value| arith.load_key(value)).collect())
            .collect()
    }

    fn will_observe(&self, arith: &mut CircuitArith<F, EF, P>, commitment: &CapTarget<F>) {
        arith.will_observe(commitment.iter().flatten().cloned().collect());
    }

    fn verify_rows(
        &self,
        arith: &mut CircuitArith<F, EF, P>,
        commitment: &CapTarget<F>,
        dimensions: &[Dimensions],
        indices: &[Vec<Base<F>>],
        rows: &[Vec<Cow<'_, [Base<F>]>>],
        proof: &Self::MultiProof,
    ) -> Result<(), MerkleTreeError> {
        // The native check rejects every malformed opening and restores each query's path.
        let native_indices: Vec<usize> = indices
            .iter()
            .map(|bits| {
                bits.iter()
                    .enumerate()
                    .filter(|(_, bit)| arith.value(bit).is_one())
                    .map(|(i, _)| 1usize << i)
                    .sum()
            })
            .collect();
        let native_rows: Vec<Vec<Vec<F>>> = rows
            .iter()
            .map(|query| {
                query
                    .iter()
                    .map(|row| row.iter().map(|value| arith.value(value)).collect())
                    .collect()
            })
            .collect();
        let native_commitment = MerkleCap::new(
            commitment
                .iter()
                .map(|digest| core::array::from_fn(|i| arith.value(&digest[i])))
                .collect(),
        );
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

        let mut shared = arith.shared.lock();
        let builder = &mut shared.builder;
        for ((bits, query_rows), path) in indices.iter().zip(rows).zip(paths) {
            let gather = |matrices: &[usize]| -> Vec<Base<F>> {
                matrices
                    .iter()
                    .flat_map(|&m| query_rows[m].iter().cloned())
                    .collect()
            };

            // Walk the plan from the leaf, as the native verifier walks it.
            let mut digest = sponge(&gather(&plan.leaf), P::WIDTH, RATE, DIGEST, |state| {
                builder.permute(state)
            });
            for ((depth, level), sibling) in plan.levels.iter().enumerate().zip(&path.siblings) {
                // In a binary level, the index bit says whether the digest is the right child.
                debug_assert_eq!(level.arity, 2, "a binary tree compresses pairs");
                let sibling: Vec<Base<F>> = sibling.iter().map(|&v| builder.witness(v)).collect();
                let bit = bits.get(depth).cloned().unwrap_or(Base::ZERO);
                let left = select_all(&bit, &sibling, &digest);
                let right: Vec<Base<F>> = digest
                    .iter()
                    .zip(&sibling)
                    .zip(&left)
                    .map(|((d, s), l)| d.clone() + s.clone() - l.clone())
                    .collect();
                digest = compress(builder, &left, &right);

                // Hash in the rows the plan injects at the next height.
                if !level.inject.is_empty() {
                    let injected =
                        sponge(&gather(&level.inject), P::WIDTH, RATE, DIGEST, |state| {
                            builder.permute(state)
                        });
                    digest = compress(builder, &digest, &injected);
                }
            }

            // The index bits above the walked levels name the cap entry.
            let cap_bits = bits.get(plan.levels.len()..).unwrap_or(&[]);
            let root = select_root(commitment, cap_bits);
            for (d, r) in digest.iter().zip(&root) {
                builder
                    .assert_zero(&(d.clone() - r.clone()))
                    .map_err(|_| MerkleTreeError::CapMismatch)?;
            }
        }
        Ok(())
    }
}

impl<F, EF, P, Inner> MmcsArith<CircuitArith<F, EF, P>, Ext<F, EF>> for ExtensionMmcs<F, EF, Inner>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    Inner: Mmcs<F> + MmcsArith<CircuitArith<F, EF, P>, Base<F>, Error = <Inner as Mmcs<F>>::Error>,
{
    type Commitment = <Inner as MmcsArith<CircuitArith<F, EF, P>, Base<F>>>::Commitment;
    type MultiProof = <Inner as MmcsArith<CircuitArith<F, EF, P>, Base<F>>>::MultiProof;
    type Target = <Inner as MmcsArith<CircuitArith<F, EF, P>, Base<F>>>::Target;
    type Error = ExtensionMmcsError<<Inner as Mmcs<F>>::Error>;

    fn load_commitment(
        &self,
        arith: &mut CircuitArith<F, EF, P>,
        commitment: &Self::Commitment,
    ) -> Self::Target {
        self.inner().load_commitment(arith, commitment)
    }

    fn load_key_commitment(
        &self,
        arith: &mut CircuitArith<F, EF, P>,
        commitment: &Self::Commitment,
    ) -> Self::Target {
        self.inner().load_key_commitment(arith, commitment)
    }

    fn will_observe(&self, arith: &mut CircuitArith<F, EF, P>, commitment: &Self::Target) {
        self.inner().will_observe(arith, commitment);
    }

    fn verify_rows(
        &self,
        arith: &mut CircuitArith<F, EF, P>,
        commitment: &Self::Target,
        dimensions: &[Dimensions],
        indices: &[Vec<Base<F>>],
        rows: &[Vec<Cow<'_, [Ext<F, EF>]>>],
        proof: &Self::MultiProof,
    ) -> Result<(), Self::Error> {
        // The native checks fix the shape.
        //
        // An extension row is committed as its coefficients.
        let base_dimensions = Self::multi_opening_shape(dimensions, indices.len(), rows)?;
        let base_rows: Vec<Vec<Cow<'_, [Base<F>]>>> = rows
            .iter()
            .map(|query| {
                query
                    .iter()
                    .map(|row| {
                        Cow::Owned(
                            row.iter()
                                .flat_map(|value| arith.coefficients(value))
                                .collect(),
                        )
                    })
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
