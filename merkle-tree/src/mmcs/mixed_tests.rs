//! Commitment, opening and verification round trips through mixed-hash trees.
//!
//! The builder tests pin the digests of a mixed tree layer by layer.
//! These tests pin the other half: every verifier walks the same levels as the builder.
//! Honest openings verify, and an opening checked against the wrong switch level does not.

extern crate std;

use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_sha256::Sha256Compress;
use p3_symmetric::{
    CompressionFunctionFromHasher, MixedCompression, PaddingFreeSponge, SerializingCompression,
    TruncatedPermutation,
};
use rand::SeedableRng;
use rand::rngs::{SmallRng, StdRng};

use crate::{MerkleTreeHidingMmcs, MerkleTreeMmcs};

type F = BabyBear;
type Packing = <F as Field>::Packing;

type Perm16 = Poseidon2BabyBear<16>;
type Perm32 = Poseidon2BabyBear<32>;
type Hash = PaddingFreeSponge<Perm16, 16, 8, 8>;

type Lower2 = SerializingCompression<Sha256Compress>;
type Upper2 = TruncatedPermutation<Perm16, 2, 8, 16>;
type Mixed2 = MixedCompression<Lower2, Upper2>;
type Mmcs2 = MerkleTreeMmcs<Packing, Packing, Hash, Mixed2, 2, 8>;

type Lower4 = SerializingCompression<CompressionFunctionFromHasher<p3_blake2s::Blake2s256, 4, 32>>;
type Upper4 = TruncatedPermutation<Perm32, 4, 8, 32>;
type Mixed4 = MixedCompression<Lower4, Upper4>;
type Mmcs4 = MerkleTreeMmcs<Packing, Packing, Hash, Mixed4, 4, 8>;

/// Binary SHA-256 below `upper_from`, binary Poseidon2 from it on.
fn binary(upper_from: usize, cap_height: usize) -> Mmcs2 {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm16::new_from_rng_128(&mut rng);
    let c = Mixed2::new(
        SerializingCompression::new(Sha256Compress),
        Upper2::new(perm.clone()),
        upper_from,
    );
    Mmcs2::new(Hash::new(perm), c, cap_height)
}

/// Four-to-one BLAKE2s below `upper_from`, four-to-one Poseidon2 from it on.
fn quaternary(upper_from: usize, cap_height: usize) -> Mmcs4 {
    let mut rng = SmallRng::seed_from_u64(2);
    let perm16 = Perm16::new_from_rng_128(&mut rng);
    let perm32 = Perm32::new_from_rng_128(&mut rng);
    let c = Mixed4::new(
        SerializingCompression::new(CompressionFunctionFromHasher::new(p3_blake2s::Blake2s256)),
        Upper4::new(perm32),
        upper_from,
    );
    Mmcs4::new(Hash::new(perm16), c, cap_height)
}

/// Matrices on one ladder: two of the full height, then shorter ones injected higher up.
fn matrices(max_height: usize) -> Vec<RowMajorMatrix<F>> {
    let mut rng = SmallRng::seed_from_u64(max_height as u64);
    [
        (max_height, 3),
        (max_height, 1),
        (max_height.div_ceil(2), 5),
        (max_height.div_ceil(8), 2),
        (max_height.div_ceil(64), 4),
    ]
    .into_iter()
    .map(|(height, width)| RowMajorMatrix::rand(&mut rng, height, width))
    .collect()
}

/// Opens every `stride`-th row one at a time, and all of them as one pruned multi-opening, and checks both against `verifier`.
fn assert_round_trip<P, V>(prover: &P, verifier: &V, max_height: usize, stride: usize)
where
    P: Mmcs<F>,
    V: Mmcs<F, Commitment = P::Commitment, Proof = P::Proof, MultiProof = P::MultiProof>,
    V::Error: core::fmt::Debug,
{
    let mats = matrices(max_height);
    let dims: Vec<_> = mats.iter().map(Matrix::dimensions).collect();
    let (commit, data) = prover.commit(mats);

    let indices: Vec<usize> = (0..max_height).step_by(stride).collect();
    for &index in &indices {
        let opening = prover.open_batch(index, &data);
        let opening = BatchOpeningRef::<F, V>::new(&opening.opened_values, &opening.opening_proof);
        verifier
            .verify_batch(&commit, &dims, index, opening)
            .unwrap_or_else(|err| panic!("index {index}: {err:?}"));
    }

    let (opened, proof) = prover.open_multi_batch(&indices, &data);
    verifier
        .verify_multi_batch(&commit, &dims, &indices, &opened, &proof)
        .unwrap();
}

#[test]
fn binary_openings_verify_at_every_switch_level() {
    for upper_from in [0, 1, 2, 4, 7, usize::MAX] {
        for cap_height in [0, 2] {
            let mmcs = binary(upper_from, cap_height);
            assert_round_trip(&mmcs, &mmcs, 200, 7);
        }
    }
}

#[test]
fn quaternary_openings_verify_at_every_switch_level() {
    for upper_from in [0, 1, 2, 3, usize::MAX] {
        for cap_height in [0, 1] {
            let mmcs = quaternary(upper_from, cap_height);
            assert_round_trip(&mmcs, &mmcs, 300, 11);
        }
    }
}

#[test]
fn an_opening_is_bound_to_its_switch_level() {
    // A verifier configured with another switch level recomputes a different root.
    let mats = matrices(128);
    let dims: Vec<_> = mats.iter().map(Matrix::dimensions).collect();
    let prover = binary(3, 0);
    let (commit, data) = prover.commit(mats);
    let opening = prover.open_batch(5, &data);
    prover
        .verify_batch(&commit, &dims, 5, BatchOpeningRef::from(&opening))
        .unwrap();

    // Positive control: the prover's own configuration accepts the multi-opening.
    let (opened, proof) = prover.open_multi_batch(&[1, 5, 77], &data);
    prover
        .verify_multi_batch(&commit, &dims, &[1, 5, 77], &opened, &proof)
        .unwrap();

    for other in [0, 2, 4, usize::MAX] {
        let verifier = binary(other, 0);
        assert!(
            verifier
                .verify_batch(&commit, &dims, 5, BatchOpeningRef::from(&opening))
                .is_err(),
            "switch {other} accepted an opening made with switch 3"
        );

        assert!(
            verifier
                .verify_multi_batch(&commit, &dims, &[1, 5, 77], &opened, &proof)
                .is_err(),
            "switch {other} accepted a multi-opening made with switch 3"
        );
    }
}

#[test]
fn a_tampered_sibling_is_rejected_on_both_sides_of_the_switch() {
    // A path through a 2^7 tree has seven siblings: three SHA-256 levels, then four Poseidon2.
    let mmcs = binary(3, 0);
    let mats = matrices(128);
    let dims: Vec<_> = mats.iter().map(Matrix::dimensions).collect();
    let (commit, data) = mmcs.commit(mats);
    let opening = mmcs.open_batch(42, &data);
    for level in [0, 2, 3, 6] {
        let mut tampered = opening.clone();
        tampered.opening_proof[level][0] += F::ONE;
        assert!(
            mmcs.verify_batch(&commit, &dims, 42, BatchOpeningRef::from(&tampered))
                .is_err(),
            "tampered sibling at level {level} accepted"
        );
    }
}

#[test]
fn hiding_openings_verify() {
    // The hiding MMCS salts the leaves and delegates the tree to the plain one.
    let mut rng = SmallRng::seed_from_u64(3);
    let perm = Perm16::new_from_rng_128(&mut rng);
    let c = Mixed2::new(
        SerializingCompression::new(Sha256Compress),
        Upper2::new(perm.clone()),
        2,
    );
    let mmcs = MerkleTreeHidingMmcs::<Packing, Packing, _, _, _, 2, 8, 4>::new(
        Hash::new(perm),
        c,
        0,
        StdRng::seed_from_u64(4),
    );
    assert_round_trip(&mmcs, &mmcs, 64, 5);
}

#[test]
fn scalar_and_packed_verifiers_agree() {
    // The packed verifier groups lanes; the scalar one walks node by node.
    let mut rng = SmallRng::seed_from_u64(5);
    let perm = Perm16::new_from_rng_128(&mut rng);
    let c = Mixed2::new(
        SerializingCompression::new(Sha256Compress),
        Upper2::new(perm.clone()),
        3,
    );
    let packed = Mmcs2::new(Hash::new(perm.clone()), c.clone(), 1);
    let scalar = MerkleTreeMmcs::<F, F, _, _, 2, 8>::new(Hash::new(perm), c, 1);
    assert_round_trip(&packed, &scalar, 256, 9);
    assert_round_trip(&scalar, &packed, 256, 9);
}

#[test]
fn a_streamed_commitment_matches_the_committed_one() {
    // The streamed path hashes the leaves itself and builds only the layers above them.
    let matrix = RowMajorMatrix::<F>::rand(&mut SmallRng::seed_from_u64(6), 200, 3);
    let dims = [matrix.dimensions()];
    let width = matrix.width;
    for upper_from in [0, 2, 5, usize::MAX] {
        let mmcs = binary(upper_from, 1);
        let (commit, _) = mmcs.commit_matrix(matrix.clone());
        let (streamed, data) = mmcs.commit_matrix_streamed(matrix.dimensions(), |sink| {
            for (block, rows) in matrix.values.chunks(16 * width).enumerate() {
                sink(block * 16, rows);
            }
            matrix.clone()
        });
        assert!(streamed == commit, "upper_from {upper_from}");

        let opening = mmcs.open_batch(77, &data);
        mmcs.verify_batch(&streamed, &dims, 77, BatchOpeningRef::from(&opening))
            .unwrap();
    }
}

#[test]
fn restored_paths_match_single_openings() {
    // The pruned verifier rebuilds every full path, recomputing the shared nodes at their levels.
    let mats = matrices(256);
    let dims: Vec<_> = mats.iter().map(Matrix::dimensions).collect();
    let indices = [0, 9, 10, 130, 255];
    for upper_from in [0, 3, usize::MAX] {
        let mmcs = binary(upper_from, 0);
        let (_, data) = mmcs.commit(mats.clone());
        let opening = mmcs.open_batch_pruned(&indices, &data);
        let restored = mmcs
            .restore_and_recompute_paths(
                &dims,
                &indices,
                &opening.opened_values,
                &opening.pruned_proof,
            )
            .unwrap();
        for (&index, path) in indices.iter().zip(restored) {
            assert_eq!(
                path.siblings,
                mmcs.open_batch(index, &data).opening_proof,
                "upper_from {upper_from}, index {index}"
            );
        }
    }
}
