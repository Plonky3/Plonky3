use alloc::vec;

use p3_binary_field::TowerLevel;
use p3_challenger::CanSample;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::*;
use crate::zk::challenger;

#[test]
fn opens_monomial_columns_and_replays_transcript() {
    let pcs = ReferenceZkPcs::new(ReferencePcsConfig::new(3, 8).unwrap());
    // f0=1+3X+5X²; f1=2+4X+6X², evaluated at zero.
    let messages = RowMajorMatrix::new((1..=6).map(Val::from_repr).collect(), 2);
    let mut rng = StdRng::seed_from_u64(42);
    let (root, data) = pcs.commit(messages, &mut rng).unwrap();
    let (mut prover, mut verifier) = (challenger(), challenger());
    let proof = pcs.open(data, Val::ZERO, &mut prover, &mut rng).unwrap();
    assert_eq!(
        proof.claimed_values,
        vec![Val::from_repr(1), Val::from_repr(2)]
    );
    assert_eq!(
        pcs.verify(&root, Val::ZERO, &proof, &mut verifier).unwrap(),
        proof.claimed_values
    );
    assert_eq!(
        CanSample::<Val>::sample(&mut prover),
        CanSample::<Val>::sample(&mut verifier)
    );
}

#[test]
fn rejects_empty_and_overflowing_geometry() {
    for (m, q) in [
        (0, 1),
        (1, 0),
        (usize::MAX, 1),
        (1, usize::MAX),
        (1 << 30, 1),
    ] {
        assert!(ReferencePcsConfig::new(m, q).is_err());
    }
}

fn fixture(seed: u64) -> (ReferenceZkPcs, ZkCommitment, ZkOpeningProof) {
    let pcs = ReferenceZkPcs::new(ReferencePcsConfig::new(2, 12).unwrap());
    let mut rng = StdRng::seed_from_u64(seed);
    let (commitment, data) = pcs
        .commit(
            RowMajorMatrix::new_col(vec![Val::ONE, Val::from_repr(7)]),
            &mut rng,
        )
        .unwrap();
    let proof = pcs
        .open(data, Val::from_repr(5), &mut challenger(), &mut rng)
        .unwrap();
    (pcs, commitment, proof)
}

#[test]
fn rejects_altered_claim_blend_roots_and_authenticated_rows() {
    let (pcs, commitment, proof) = fixture(7);
    for edit in 0..9 {
        let mut bad = proof.clone();
        let mut root = commitment.clone();
        match edit {
            0 => bad.claimed_values[0] += Val::ONE,
            1 => bad.blended_coefficients[0][0] += Val::ONE,
            2 => bad.blended_coefficients[0][pcs.config().coefficient_len() - 1] += Val::ONE,
            3 => root.root = p3_symmetric::MerkleCap::new(vec![[0; 32]]),
            4 => bad.mask_root = p3_symmetric::MerkleCap::new(vec![[0; 32]]),
            5 => bad.queries[0].h.values[0] += Val::ONE,
            6 => bad.queries[0].u.values[0] += Val::ONE,
            7 => bad.queries[0].h.salt += Val::ONE,
            8 => bad.sigma[0] += Val::ONE,
            _ => unreachable!(),
        }
        assert!(
            pcs.verify(&root, Val::from_repr(5), &bad, &mut challenger())
                .is_err(),
            "edit {edit}"
        );
    }
    assert!(
        pcs.verify(&commitment, Val::from_repr(6), &proof, &mut challenger())
            .is_err()
    );
}

#[test]
fn rejects_malformed_shapes_before_touching_transcript() {
    let (pcs, commitment, proof) = fixture(8);
    for edit in 0..16 {
        let mut bad = proof.clone();
        let mut root = commitment.clone();
        match edit {
            0 => bad.claimed_values.clear(),
            1 => bad.sigma.clear(),
            2 => bad.blended_coefficients.clear(),
            3 => {
                bad.blended_coefficients[0].pop();
            }
            4 => {
                bad.queries.pop();
            }
            5 => bad.queries[0].h.values.clear(),
            6 => bad.queries[0].u.values.push(Val::ZERO),
            7 => bad.queries[0].h.authentication_path.clear(),
            8 => root.num_polynomials = usize::MAX,
            9 => root.root = p3_symmetric::MerkleCap::new(vec![[0; 32]; 2]),
            10 => bad.mask_root = p3_symmetric::MerkleCap::new(vec![[0; 32]; 2]),
            11 => bad.queries.push(bad.queries[0].clone()),
            12 => bad.queries[0].h.authentication_path.push([0; 32]),
            13 => bad.queries[0].u.authentication_path.clear(),
            14 => bad.blended_coefficients[0].push(Val::ZERO),
            15 => bad.claimed_values.push(Val::ZERO),
            _ => unreachable!(),
        }
        let mut verifier = challenger();
        assert!(pcs.verify(&root, Val::ZERO, &bad, &mut verifier).is_err());
        assert_eq!(
            CanSample::<Val>::sample(&mut verifier),
            CanSample::<Val>::sample(&mut challenger()),
            "edit {edit}"
        );
    }
}

#[test]
fn rejects_config_mismatch_before_rng_or_transcript_mutation() {
    let (pcs, commitment, proof) = fixture(9);
    let other = ReferenceZkPcs::new(ReferencePcsConfig::new(3, 12).unwrap());
    let mut verifier = challenger();
    assert!(
        other
            .verify(&commitment, Val::ZERO, &proof, &mut verifier)
            .is_err()
    );
    assert_eq!(
        CanSample::<Val>::sample(&mut verifier),
        CanSample::<Val>::sample(&mut challenger())
    );
    let mut rng = StdRng::seed_from_u64(91);
    let (_, data) = pcs
        .commit(RowMajorMatrix::new_col(vec![Val::ONE; 2]), &mut rng)
        .unwrap();
    let mut expected_rng = StdRng::seed_from_u64(91);
    pcs.commit(
        RowMajorMatrix::new_col(vec![Val::ONE; 2]),
        &mut expected_rng,
    )
    .unwrap();
    let mut prover = challenger();
    assert!(other.open(data, Val::ZERO, &mut prover, &mut rng).is_err());
    assert_eq!(
        rand::Rng::next_u64(&mut rng),
        rand::Rng::next_u64(&mut expected_rng)
    );
    assert_eq!(
        CanSample::<Val>::sample(&mut prover),
        CanSample::<Val>::sample(&mut challenger())
    );
}

#[test]
fn invalid_message_shape_does_not_draw_private_randomness() {
    let pcs = ReferenceZkPcs::new(ReferencePcsConfig::new(2, 2).unwrap());
    let mut rng = StdRng::seed_from_u64(3);
    let mut expected = StdRng::seed_from_u64(3);
    let mut zero_width = RowMajorMatrix::new_col(vec![]);
    zero_width.width = 0;
    let mut ragged = RowMajorMatrix::new_col(vec![Val::ONE; 3]);
    ragged.width = 2;
    for matrix in [
        zero_width,
        ragged,
        RowMajorMatrix::new_col(vec![Val::ONE; 3]),
    ] {
        assert!(pcs.commit(matrix, &mut rng).is_err());
    }
    assert_eq!(
        rand::Rng::next_u64(&mut rng),
        rand::Rng::next_u64(&mut expected)
    );
}

#[test]
fn commitments_and_opening_randomness_are_fresh_and_serializable() {
    let (pcs, root, proof) = fixture(10);
    let (_, other_root, other_proof) = fixture(11);
    assert_ne!(root.root, other_root.root);
    assert_ne!(proof.mask_root, other_proof.mask_root);
    assert_ne!(proof.blended_coefficients, other_proof.blended_coefficients);
    let root: ZkCommitment = postcard::from_bytes(&postcard::to_allocvec(&root).unwrap()).unwrap();
    let proof: ZkOpeningProof =
        postcard::from_bytes(&postcard::to_allocvec(&proof).unwrap()).unwrap();
    pcs.verify(&root, Val::from_repr(5), &proof, &mut challenger())
        .unwrap();
}

#[test]
fn repeated_positions_use_the_same_authenticated_rows() {
    // q=12, n=64. Search deterministic fixtures until the transcript repeats an index.
    let mut found = false;
    for seed in 0..32 {
        let (pcs, root, proof) = fixture(seed);
        let mut replay = challenger();
        pcs.bind_opening(
            &root,
            Val::from_repr(5),
            &proof.claimed_values,
            &proof.mask_root,
            &proof.sigma,
            &mut replay,
        );
        let _: Val = sample_nonzero(&mut replay);
        observe_blend(&proof.blended_coefficients, &mut replay);
        let indices = pcs.sample_indices(&mut replay);
        for i in 0..indices.len() {
            for j in 0..i {
                if indices[i] == indices[j] {
                    found = true;
                    assert_eq!(proof.queries[i], proof.queries[j]);
                    pcs.verify(&root, Val::from_repr(5), &proof, &mut challenger())
                        .unwrap();
                    let mut bad = proof.clone();
                    bad.queries[i].h.values[0] += Val::ONE;
                    assert!(
                        pcs.verify(&root, Val::from_repr(5), &bad, &mut challenger())
                            .is_err()
                    );
                }
            }
        }
        if found {
            break;
        }
    }
    assert!(found);
}

#[test]
fn full_blend_and_repeated_query_joint_distribution_is_witness_independent() {
    // Exhaust the GF(4) subfield: M=2, kappa=1 is enough for one distinct query.
    // These two messages share lambda(f)=f0+z*f1=0 at z=2.
    // Enumerating independent u, rather than only the marginal query values, catches
    // masks that cease to hide after disclosing the entire blend.
    use alloc::collections::BTreeMap;

    use p3_binary_field::BinaryField8 as Small;
    let z = Small::from_repr(2);
    for gamma_repr in 1..4 {
        for x_repr in 1..4 {
            let gamma = Small::from_repr(gamma_repr);
            let x = Small::from_repr(x_repr);
            let mut distributions = Vec::new();
            for message in [[Small::ZERO, Small::ZERO], [z, Small::ONE]] {
                let mut counts = BTreeMap::<Vec<u8>, usize>::new();
                for r in 0..4 {
                    let h = [message[0], message[1], Small::from_repr(r)];
                    for packed_u in 0..64 {
                        let u = core::array::from_fn::<_, 3, _>(|j| {
                            Small::from_repr((packed_u >> (2 * j)) & 3)
                        });
                        let g = core::array::from_fn::<_, 3, _>(|j| gamma * h[j] + u[j]);
                        let hx = evaluate(&h, x);
                        let ux = evaluate(&u, x);
                        let mut view = vec![evaluate(&u[..2], z).to_repr()];
                        view.extend(g.map(|v| v.to_repr()));
                        // A duplicate query adds no independent equation.
                        view.extend([hx.to_repr(), ux.to_repr(), hx.to_repr(), ux.to_repr()]);
                        *counts.entry(view).or_default() += 1;
                    }
                }
                distributions.push(counts);
            }
            assert_eq!(
                distributions[0], distributions[1],
                "gamma {gamma_repr}, x {x_repr}"
            );
        }
    }
}

#[test]
fn tail_projection_has_full_rank_on_nonzero_affine_domain() {
    use p3_binary_field::BinaryField8 as Small;
    use p3_field::Field;
    let points: Vec<_> = (0..4)
        .map(|index| {
            let mut x = Small::cantor_basis(2);
            for bit in 0..2 {
                if index & (1 << bit) != 0 {
                    x += Small::cantor_basis(bit);
                }
            }
            x
        })
        .collect();
    assert!(points.iter().all(|&x| x != Small::ZERO));
    for m in [1, 2, 5] {
        // Every subset of <=q=3 rows has rank equal to its distinct row count;
        // kappa=q+1=4 independently sampled tail coefficients are available.
        for subset in 1usize..16 {
            let mut rows: Vec<Vec<Small>> = points
                .iter()
                .enumerate()
                .filter(|(j, _)| subset & (1 << j) != 0)
                .map(|(_, &x)| (0..4).map(|k| x.exp_u64((m + k) as u64)).collect())
                .collect();
            let expected = rows.len();
            let mut rank = 0;
            for col in 0..4 {
                if let Some(pivot) = (rank..rows.len()).find(|&i| rows[i][col] != Small::ZERO) {
                    rows.swap(rank, pivot);
                    let inverse = rows[rank][col].inverse();
                    for value in &mut rows[rank] {
                        *value *= inverse;
                    }
                    let pivot_row = rows[rank].clone();
                    for (i, row) in rows.iter_mut().enumerate() {
                        if i != rank {
                            let factor = row[col];
                            for (value, &pivot_value) in
                                row[col..].iter_mut().zip(&pivot_row[col..])
                            {
                                *value += factor * pivot_value;
                            }
                        }
                    }
                    rank += 1;
                }
            }
            assert_eq!(rank, expected);
        }
    }
}

#[test]
fn rejects_empty_caps_and_truncated_fixed_salts_during_deserialization() {
    // A zero-root cap cannot be constructed or deserialized into a Root.
    assert!(postcard::from_bytes::<Root>(&[0]).is_err());
    let (_, _, proof) = fixture(81);
    let bytes = postcard::to_allocvec(&proof).unwrap();
    // Prefix truncation sweeps through row values, the single fixed-size salt,
    // and authentication paths. No malformed partial opening reaches verify.
    for end in 0..bytes.len() {
        assert!(postcard::from_bytes::<ZkOpeningProof>(&bytes[..end]).is_err());
    }
}

#[test]
fn observes_outer_commitment_before_sampling_common_point() {
    let pcs = ReferenceZkPcs::new(ReferencePcsConfig::new(2, 6).unwrap());
    let mut rng = StdRng::seed_from_u64(82);
    let (root, data) = pcs
        .commit(
            RowMajorMatrix::new(
                vec![Val::ONE, Val::ZERO, Val::from_repr(3), Val::from_repr(7)],
                2,
            ),
            &mut rng,
        )
        .unwrap();
    let (mut prover, mut verifier) = (challenger(), challenger());
    pcs.observe_commitment(&root, &mut prover);
    pcs.observe_commitment(&root, &mut verifier);
    let z: Val = prover.sample();
    assert_eq!(z, CanSample::<Val>::sample(&mut verifier));
    let proof = pcs.open(data, z, &mut prover, &mut rng).unwrap();
    assert_eq!(
        proof.claimed_values,
        vec![Val::ONE + Val::from_repr(3) * z, Val::from_repr(7) * z]
    );
    pcs.verify(&root, z, &proof, &mut verifier).unwrap();
    assert_eq!(
        CanSample::<Val>::sample(&mut prover),
        CanSample::<Val>::sample(&mut verifier)
    );
}
