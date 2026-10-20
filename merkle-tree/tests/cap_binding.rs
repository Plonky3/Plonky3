use p3_commit::{BatchOpeningRef, Mmcs};
use p3_keccak::Keccak256Hash;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::CompressionFunctionFromHasher;

type ByteMmcs<const N: usize> = MerkleTreeMmcs<
    u8,
    u8,
    Keccak256Hash,
    CompressionFunctionFromHasher<Keccak256Hash, N, 32>,
    N,
    32,
>;

fn mmcs<const N: usize>(cap_height: usize) -> ByteMmcs<N> {
    ByteMmcs::new(
        Keccak256Hash,
        CompressionFunctionFromHasher::new(Keccak256Hash),
        cap_height,
    )
}

fn matrices(heights: &[usize]) -> Vec<RowMajorMatrix<u8>> {
    heights
        .iter()
        .enumerate()
        .map(|(i, &height)| RowMajorMatrix::new(vec![i as u8; height * 3], 3))
        .collect()
}

fn check_commitment_binding<const N: usize>() {
    for heights in [
        &[8, 1][..],
        &[64, 8, 1],
        &[53, 27, 7, 1],
        &[64, 8],
        &[53, 27, 7],
        &[64, 8, 8],
    ] {
        for cap_height in [0, 1, 2, usize::MAX] {
            let mmcs = mmcs::<N>(cap_height);
            let inputs = matrices(heights);
            let (commitment, _) = mmcs.commit(inputs.clone());
            for matrix in 0..inputs.len() {
                let mut changed = inputs.clone();
                changed[matrix].values[0] ^= 1;
                let (changed_commitment, _) = mmcs.commit(changed);
                assert_ne!(
                    commitment, changed_commitment,
                    "arity={N}, heights={heights:?}, cap={cap_height}, matrix={matrix}"
                );
            }
        }
    }
}

#[test]
fn cap_binding_commitment_depends_on_every_matrix() {
    check_commitment_binding::<2>();
    check_commitment_binding::<4>();
    check_commitment_binding::<8>();
}

fn check_opening_binding<const N: usize>(pruned: bool) {
    for heights in [
        &[8, 1][..],
        &[64, 8, 1],
        &[53, 27, 7, 1],
        &[64, 8],
        &[53, 27, 7],
        &[64, 8, 8],
    ] {
        for cap_height in [0, 1, 2, usize::MAX] {
            let mmcs = mmcs::<N>(cap_height);
            let inputs = matrices(heights);
            let dims: Vec<_> = inputs.iter().map(Matrix::dimensions).collect();
            let (commitment, data) = mmcs.commit(inputs);
            // Include an unordered query set, a duplicate, and both edge rows.
            let indices = [heights[0] - 1, 0, heights[0] / 2, heights[0] - 1];
            if !pruned {
                for &index in &indices {
                    let opening = mmcs.open_batch(index, &data);
                    mmcs.verify_batch(&commitment, &dims, index, (&opening).into())
                        .unwrap();
                    for matrix in 0..heights.len() {
                        let mut changed = opening.clone();
                        changed.opened_values[matrix][0] ^= 1;
                        assert!(
                            mmcs.verify_batch(&commitment, &dims, index, (&changed).into())
                                .is_err(),
                            "single: arity={N}, heights={heights:?}, cap={cap_height}, matrix={matrix}"
                        );
                    }
                }
                continue;
            }
            let (values, proof) = mmcs.open_multi_batch(&indices, &data);
            mmcs.verify_multi_batch(&commitment, &dims, &indices, &values, &proof)
                .unwrap();
            let paths = mmcs
                .restore_and_recompute_paths(&dims, &indices, &values, &proof)
                .unwrap();
            for (query, path) in paths.iter().enumerate() {
                let single = mmcs.open_batch(indices[query], &data);
                assert_eq!(path.siblings, single.opening_proof);
            }
            for matrix in 0..heights.len() {
                let mut changed = values.clone();
                // Preserve duplicate/group agreement: the cap must bind the row itself.
                for rows in &mut changed {
                    rows[matrix][0] ^= 1;
                }
                assert!(
                    mmcs.verify_multi_batch(&commitment, &dims, &indices, &changed, &proof)
                        .is_err(),
                    "multi: arity={N}, heights={heights:?}, cap={cap_height}, matrix={matrix}"
                );
                for (query, path) in paths.iter().enumerate() {
                    assert!(
                        mmcs.verify_batch(
                            &commitment,
                            &dims,
                            indices[query],
                            BatchOpeningRef::new(&changed[query], &path.siblings),
                        )
                        .is_err()
                    );
                }
            }
        }
    }
}

#[test]
fn cap_binding_rejects_tampered_single_openings() {
    check_opening_binding::<2>(false);
    check_opening_binding::<4>(false);
    check_opening_binding::<8>(false);
}

#[test]
fn cap_binding_rejects_tampered_pruned_openings() {
    check_opening_binding::<2>(true);
    check_opening_binding::<4>(true);
    check_opening_binding::<8>(true);
}

fn check_legacy_cap_rejected<const N: usize>() {
    let mmcs = mmcs::<N>(1);
    let inputs = matrices(&[8, 1]);
    let dims: Vec<_> = inputs.iter().map(Matrix::dimensions).collect();
    let (_, data) = mmcs.commit(inputs);
    // Extract the old cap below the final injection, bypassing MMCS cap selection.
    let legacy_cap = data.cap(1);
    let mut opening = mmcs.open_batch(0, &data);
    opening.opened_values[1][0] ^= 1;
    // A malformed legacy opening must not become valid after upgrading the verifier.
    assert!(
        mmcs.verify_batch(&legacy_cap, &dims, 0, (&opening).into())
            .is_err()
    );
    let (mut values, proof) = mmcs.open_multi_batch(&[0], &data);
    values[0][1][0] ^= 1;
    assert!(
        mmcs.verify_multi_batch(&legacy_cap, &dims, &[0], &values, &proof)
            .is_err()
    );
}

#[test]
fn cap_binding_rejects_legacy_unbound_caps() {
    check_legacy_cap_rejected::<2>();
    check_legacy_cap_rejected::<4>();
    check_legacy_cap_rejected::<8>();
}

#[test]
fn cap_binding_preserves_caps_that_already_include_every_matrix() {
    let inputs = matrices(&[64, 8]);
    let mmcs = mmcs::<2>(2);
    let (commitment, data) = mmcs.commit(inputs);
    assert_eq!(commitment, data.cap(2));
    assert_eq!(commitment.num_roots(), 4);
    assert_eq!(mmcs.open_batch(0, &data).opening_proof.len(), 4);
}
