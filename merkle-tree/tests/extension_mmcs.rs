use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_commit::{ExtensionMmcs, ExtensionMmcsError, Mmcs};
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Dimensions;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
type Inner = MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, Hash, Compress, 2, 8>;
type Extension = ExtensionMmcs<F, EF, Inner>;

fn mmcs() -> Extension {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    Extension::new(Inner::new(Hash::new(perm.clone()), Compress::new(perm), 0))
}

#[test]
fn single_opening_rejects_overflowing_widths() {
    let mmcs = mmcs();
    let (commit, data) = mmcs.commit_matrix(RowMajorMatrix::new_col(vec![EF::ONE; 4]));
    let opening = mmcs.open_batch(1, &data);
    // The middle width aliases width 1 after multiplying by four in release mode.
    for width in [usize::MAX / 4 + 1, usize::MAX / 4 + 2, usize::MAX] {
        let dimensions = [Dimensions { width, height: 4 }];
        assert!(matches!(
            mmcs.verify_batch(&commit, &dimensions, 1, (&opening).into()),
            Err(ExtensionMmcsError::WidthOverflow { matrix: 0, width: got }) if got == width
        ));
    }
}

#[test]
fn multi_opening_rejects_overflowing_widths() {
    let mmcs = mmcs();
    let (commit, data) = mmcs.commit_matrix(RowMajorMatrix::new_col(vec![EF::ONE; 4]));
    let indices = [1, 3];
    let (values, proof) = mmcs.open_multi_batch(&indices, &data);
    for width in [usize::MAX / 4 + 1, usize::MAX / 4 + 2, usize::MAX] {
        let dimensions = [Dimensions { width, height: 4 }];
        assert!(matches!(
            mmcs.verify_multi_batch(&commit, &dimensions, &indices, &values, &proof),
            Err(ExtensionMmcsError::WidthOverflow { matrix: 0, width: got }) if got == width
        ));
    }
}

#[derive(PartialEq)]
struct UnreadableRow;

impl AsRef<[EF]> for UnreadableRow {
    fn as_ref(&self) -> &[EF] {
        panic!("malformed opening shape must be rejected before reading rows")
    }
}

#[test]
fn multi_opening_checks_shape_before_reading_rows() {
    let mmcs = mmcs();
    let (commit, data) = mmcs.commit_matrix(RowMajorMatrix::new_col(vec![EF::ONE; 4]));
    let (_, proof) = mmcs.open_multi_batch(&[1, 3], &data);
    let dimensions = [Dimensions {
        width: 1,
        height: 4,
    }];
    // Wrong query count and wrong matrix count must both be caught before flattening.
    for values in [vec![vec![UnreadableRow]], vec![vec![UnreadableRow], vec![]]] {
        assert!(
            mmcs.verify_multi_batch(&commit, &dimensions, &[1, 3], &values, &proof)
                .is_err()
        );
    }
}

#[test]
fn valid_openings_preserve_mixed_matrix_widths_and_heights() {
    let mmcs = mmcs();
    let dimensions = [
        Dimensions {
            width: 1,
            height: 4,
        },
        Dimensions {
            width: 2,
            height: 2,
        },
    ];
    let matrices = vec![
        RowMajorMatrix::new_col(vec![EF::ONE, EF::TWO, EF::ZERO, -EF::ONE]),
        RowMajorMatrix::new(vec![EF::TWO, EF::ONE, -EF::ONE, EF::ZERO], 2),
    ];
    let (commit, data) = mmcs.commit(matrices);
    let opening = mmcs.open_batch(3, &data);
    mmcs.verify_batch(&commit, &dimensions, 3, (&opening).into())
        .unwrap();
    let indices = [3, 0, 3];
    let (values, proof) = mmcs.open_multi_batch(&indices, &data);
    let borrowed: Vec<Vec<&[EF]>> = values
        .iter()
        .map(|rows| rows.iter().map(Vec::as_slice).collect())
        .collect();
    mmcs.verify_multi_batch(&commit, &dimensions, &indices, &borrowed, &proof)
        .unwrap();
}

#[test]
fn representable_widths_are_checked_in_extension_units() {
    let mmcs = mmcs();
    let (commit, data) = mmcs.commit_matrix(RowMajorMatrix::new_col(vec![EF::ONE; 4]));
    let opening = mmcs.open_batch(1, &data);
    let indices = [1, 3];
    let (values, proof) = mmcs.open_multi_batch(&indices, &data);
    for width in [0, 2, usize::MAX / 4] {
        let dimensions = [Dimensions { width, height: 4 }];
        let single = mmcs.verify_batch(&commit, &dimensions, 1, (&opening).into());
        let multi = mmcs.verify_multi_batch(&commit, &dimensions, &indices, &values, &proof);
        for result in [single, multi] {
            assert!(matches!(result, Err(ExtensionMmcsError::WrongWidth {
                matrix: 0, expected, got: 1
            }) if expected == width));
        }
    }
}

#[test]
fn rejects_missing_and_extra_queries_and_matrices() {
    let mmcs = mmcs();
    let dimensions = [Dimensions {
        width: 1,
        height: 4,
    }];
    let (commit, data) = mmcs.commit_matrix(RowMajorMatrix::new_col(vec![EF::ONE; 4]));
    let mut opening = mmcs.open_batch(1, &data);
    for rows in [vec![], vec![vec![EF::ONE], vec![EF::ONE]]] {
        opening.opened_values = rows;
        assert!(matches!(
            mmcs.verify_batch(&commit, &dimensions, 1, (&opening).into()),
            Err(ExtensionMmcsError::WrongBatchSize { expected: 1, .. })
        ));
    }
    let indices = [1, 3];
    let (values, proof) = mmcs.open_multi_batch(&indices, &data);
    for queries in [vec![], vec![values[0].clone()], vec![values[0].clone(); 3]] {
        assert!(matches!(
            mmcs.verify_multi_batch(&commit, &dimensions, &indices, &queries, &proof),
            Err(ExtensionMmcsError::WrongBatchSize { expected: 2, .. })
        ));
    }
    for rows in [vec![], vec![vec![EF::ONE], vec![EF::ONE]]] {
        let queries = vec![values[0].clone(), rows];
        assert!(matches!(
            mmcs.verify_multi_batch(&commit, &dimensions, &indices, &queries, &proof),
            Err(ExtensionMmcsError::WrongBatchSize { expected: 1, .. })
        ));
    }
}

#[test]
fn rejects_width_mismatch_in_later_matrix_and_query() {
    let mmcs = mmcs();
    let dimensions = [
        Dimensions {
            width: 1,
            height: 4,
        },
        Dimensions {
            width: 2,
            height: 4,
        },
    ];
    let (commit, data) = mmcs.commit(vec![
        RowMajorMatrix::new_col(vec![EF::ONE; 4]),
        RowMajorMatrix::new(vec![EF::TWO; 8], 2),
    ]);
    let mut opening = mmcs.open_batch(1, &data);
    opening.opened_values[1].pop();
    assert!(matches!(
        mmcs.verify_batch(&commit, &dimensions, 1, (&opening).into()),
        Err(ExtensionMmcsError::WrongWidth {
            matrix: 1,
            expected: 2,
            got: 1
        })
    ));
    let indices = [1, 3];
    let (mut values, proof) = mmcs.open_multi_batch(&indices, &data);
    values[1][1].push(EF::ONE);
    assert!(matches!(
        mmcs.verify_multi_batch(&commit, &dimensions, &indices, &values, &proof),
        Err(ExtensionMmcsError::WrongWidth {
            matrix: 1,
            expected: 2,
            got: 3
        })
    ));
}

#[test]
fn preserves_inner_verification_errors() {
    let mmcs = mmcs();
    let dimensions = [Dimensions {
        width: 1,
        height: 4,
    }];
    let (commit, data) = mmcs.commit_matrix(RowMajorMatrix::new_col(vec![EF::ONE; 4]));
    let mut opening = mmcs.open_batch(1, &data);
    opening.opened_values[0][0] = EF::TWO;
    assert!(matches!(
        mmcs.verify_batch(&commit, &dimensions, 1, (&opening).into()),
        Err(ExtensionMmcsError::Inner(_))
    ));
    let indices = [1, 3];
    let (mut values, proof) = mmcs.open_multi_batch(&indices, &data);
    values[1][0][0] = EF::TWO;
    assert!(matches!(
        mmcs.verify_multi_batch(&commit, &dimensions, &indices, &values, &proof),
        Err(ExtensionMmcsError::Inner(_))
    ));
}
