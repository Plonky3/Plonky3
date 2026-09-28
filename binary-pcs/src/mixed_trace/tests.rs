use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::BinaryField128;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::plan_stacked_layout;
use p3_sumcheck::{OpeningBatch, TableSpec};
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::BooleanTracePcs;
use crate::params::{BinaryPcsConfig, BinaryPcsParams};
use crate::test_util::{MyChallenger, MyMmcs, challenger, mmcs};

type EF = BinaryField128;
type Scheme = MixedTracePcs<EF, MyMmcs, MyMmcs>;

/// Variables the bit packing absorbs: one element holds `2^7` bits.
const ABSORBED: usize = 7;

/// A table of `bits` random bit columns, then `width - bits` random field columns.
fn mixed_table(seed: u64, log_height: usize, width: usize, bits: usize) -> Table<EF> {
    let rows = 1usize << log_height;
    let mut rng = SmallRng::seed_from_u64(seed);
    let cells = (0..width)
        .flat_map(|column| {
            (0..rows)
                .map(|_| {
                    if column < bits {
                        EF::from_bool(rng.random())
                    } else {
                        rng.random()
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    Table::new(RowMajorMatrix::new(cells, rows))
}

/// The Boolean trace commitment sized for the committed shapes.
fn boolean_pcs(committed: &[TableShape]) -> BooleanTracePcs<EF, MyMmcs, MyMmcs> {
    let params = BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: 40,
    };
    let (arity, _) = plan_stacked_layout(committed);
    let config = BinaryPcsConfig::try_new::<EF, EF>(arity - ABSORBED, params).unwrap();
    BooleanTracePcs::new(config, mmcs(), mmcs(), arity).unwrap()
}

/// A mixed commitment for these shapes and bit regions.
fn scheme(shapes: &[TableShape], bits: &[usize]) -> Scheme {
    let committed = committed_shapes::<EF>(shapes, bits);
    MixedTraceCommitment::new(boolean_pcs(&committed), bits.to_vec())
}

/// Every column at the current row, and the `next` columns one row ahead.
fn protocol(shapes: &[TableShape], next: &[usize]) -> OpeningProtocol {
    OpeningProtocol::new(
        shapes
            .iter()
            .map(|shape| {
                TableSpec::new(
                    *shape,
                    vec![OpeningBatch::new(
                        (0..shape.width()).collect(),
                        next.to_vec(),
                    )],
                )
            })
            .collect(),
    )
}

/// A column read one row ahead: row `z` reads row `z + 1`, the last row itself.
fn successor_reading(column: &[EF], point: &Point<EF>) -> EF {
    let rows = column.len();
    let eq = Poly::new_from_point(point.as_slice(), EF::ONE);
    (0..rows)
        .map(|row| eq.as_slice()[row] * column[(row + 1).min(rows - 1)])
        .sum()
}

/// Commit, open at the points, and verify, returning the opened values or the refusal.
fn round_trip(
    scheme: &Scheme,
    tables: Vec<Table<EF>>,
    protocol: &OpeningProtocol,
    points: &[Point<EF>],
) -> Result<Vec<OpeningEvals<EF>>, <Scheme as MultilinearPcs<EF, MyChallenger>>::Error> {
    let mut prover = challenger();
    let (commitment, data) = scheme.commit(tables, &mut prover)?;
    let proof = scheme.open_at(data, protocol, points, &mut prover)?;
    let mut verifier = challenger();
    scheme.observe_commitment(&commitment, &mut verifier);
    scheme.verify_at(&commitment, &proof, protocol, points, &mut verifier)
}

#[test]
fn a_cell_is_its_coordinates_over_the_basis() {
    // v = sum_k c_k * e_k, with c_k bit k of the little-endian bytes.
    let basis = coordinate_basis::<EF>();
    assert_eq!(basis.len(), 128);
    assert_eq!(basis[0], EF::ONE);
    let mut rng = SmallRng::seed_from_u64(0x3D1);
    for value in [EF::ZERO, EF::ONE, rng.random(), rng.random()] {
        let bytes = coordinate_bytes(core::slice::from_ref(&value));
        let rebuilt: EF = basis
            .iter()
            .enumerate()
            .filter(|&(k, _)| (bytes[k / 8] >> (k % 8)) & 1 == 1)
            .map(|(_, &element)| element)
            .sum();
        assert_eq!(rebuilt, value);
    }
}

#[test]
fn a_dense_column_widens_to_one_bit_column_per_coordinate() {
    // Three bits and two field elements commit 3 + 2 * 128 bit columns.
    let shapes = [TableShape::new(4, 5), TableShape::new(2, 1)];
    let committed = committed_shapes::<EF>(&shapes, &[3, 1]);
    assert_eq!(committed, [TableShape::new(4, 259), TableShape::new(2, 1)]);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    #[test]
    fn every_opened_value_is_the_source_column(
        seed in any::<u64>(),
        log_height in 1usize..9,
        dense in 1usize..3,
        bits in 0usize..4,
        with_next in any::<bool>(),
    ) {
        // One dense column keeps the stack above the one element the packing absorbs.
        let width = bits + dense;
        let shapes = [TableShape::new(log_height, width)];
        let table = mixed_table(seed, log_height, width, bits);
        let next: Vec<usize> = if with_next { (0..width).collect() } else { Vec::new() };
        let protocol = protocol(&shapes, &next);
        let mut rng = SmallRng::seed_from_u64(seed ^ 0x51);
        let points = vec![Point::<EF>::rand(&mut rng, log_height)];

        let evals = round_trip(&scheme(&shapes, &[bits]), vec![table.clone()], &protocol, &points)
            .unwrap();

        // Each value is the source column's own multilinear, bit or dense alike.
        for column in 0..width {
            let cells = table.poly(column).as_slice().to_vec();
            let direct = Poly::new(cells.clone()).eval_base(&points[0]);
            prop_assert_eq!(evals[0].current()[column], direct);
            if with_next {
                prop_assert_eq!(evals[0].next()[column], successor_reading(&cells, &points[0]));
            }
        }
    }
}

#[test]
fn an_all_bit_table_commits_exactly_as_the_boolean_trace() {
    // With no dense region the widening is the identity, so the proofs agree byte for byte.
    let shapes = [TableShape::new(8, 3), TableShape::new(5, 2)];
    let tables = vec![mixed_table(1, 8, 3, 3), mixed_table(2, 5, 2, 2)];
    let protocol = protocol(&shapes, &[]);
    let mut rng = SmallRng::seed_from_u64(3);
    let points = vec![
        Point::<EF>::rand(&mut rng, 8),
        Point::<EF>::rand(&mut rng, 5),
    ];

    let mixed = scheme(&shapes, &[3, 2]);
    let mut prover = challenger();
    let (_, data) = mixed.commit(tables.clone(), &mut prover).unwrap();
    let mixed_proof = mixed
        .open_at(data, &protocol, &points, &mut prover)
        .unwrap();

    let boolean = boolean_pcs(&shapes);
    let mut prover = challenger();
    let (_, data) = boolean.commit(tables, &mut prover).unwrap();
    let boolean_proof = boolean
        .open_at(data, &protocol, &points, &mut prover)
        .unwrap();

    assert_eq!(
        postcard::to_allocvec(&mixed_proof).unwrap(),
        postcard::to_allocvec(&boolean_proof).unwrap()
    );
}

#[test]
fn a_non_bit_cell_in_the_bit_region_is_refused() {
    // Column 1 is in the bit region, so a field element there addresses no bit.
    //
    // Two is zero in characteristic two, so the generator stands in for a non-bit.
    let shapes = [TableShape::new(3, 3)];
    let mut cells = mixed_table(4, 3, 3, 2)
        .iter_polys()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    cells[8 + 5] = EF::GENERATOR;
    let table = Table::new(RowMajorMatrix::new(cells, 8));
    let error = scheme(&shapes, &[2])
        .commit(vec![table], &mut challenger())
        .err()
        .unwrap();
    assert!(matches!(
        error,
        BooleanTraceCommitmentError::NonBooleanCell {
            table: 0,
            column: 1
        }
    ));
}

#[test]
fn a_bit_region_wider_than_its_table_is_refused() {
    let shapes = [TableShape::new(8, 2)];
    let committed = [TableShape::new(8, 2)];
    let scheme = MixedTraceCommitment::new(boolean_pcs(&committed), vec![3]);
    let error = scheme
        .commit(vec![mixed_table(5, 8, 2, 2)], &mut challenger())
        .err()
        .unwrap();
    assert!(matches!(
        error,
        BooleanTraceCommitmentError::BitRegionWidth {
            table: 0,
            bits: 3,
            width: 2
        }
    ));
    assert!(
        <Scheme as PrescribedPointPcs<EF, MyChallenger>>::prescribed_security(
            &scheme,
            &protocol(&shapes, &[])
        )
        .is_none()
    );
}

#[test]
fn a_forged_coordinate_value_is_rejected() {
    // The proof carries one value per committed column, so moving one breaks the opening.
    let shapes = [TableShape::new(6, 3)];
    let scheme = scheme(&shapes, &[1]);
    let protocol = protocol(&shapes, &[]);
    let mut rng = SmallRng::seed_from_u64(6);
    let points = vec![Point::<EF>::rand(&mut rng, 6)];

    let mut prover = challenger();
    let (commitment, data) = scheme
        .commit(vec![mixed_table(7, 6, 3, 1)], &mut prover)
        .unwrap();
    let mut proof = scheme
        .open_at(data, &protocol, &points, &mut prover)
        .unwrap();
    // Value 1 is the first coordinate of the first dense column.
    proof.values[1] += EF::ONE;

    let mut verifier = challenger();
    scheme.observe_commitment(&commitment, &mut verifier);
    assert!(
        scheme
            .verify_at(&commitment, &proof, &protocol, &points, &mut verifier)
            .is_err()
    );
}

#[test]
fn the_security_is_the_committed_protocols_own() {
    // The recombination draws nothing, so no term is added to what the commitment charges.
    let shapes = [TableShape::new(6, 4)];
    let bits = [2];
    let scheme = scheme(&shapes, &bits);
    let protocol = protocol(&shapes, &[0]);
    let mixed =
        <Scheme as PrescribedPointPcs<EF, MyChallenger>>::prescribed_security(&scheme, &protocol)
            .unwrap();
    let committed = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<EF, MyChallenger>>::prescribed_security(
        scheme.inner(),
        &scheme.expand_protocol(&protocol),
    )
    .unwrap();
    assert_eq!(mixed.error().bits(), committed.error().bits());
    assert_eq!(mixed.terms.len(), committed.terms.len());
}

#[test]
fn a_dense_column_costs_a_block_not_one_reduction_per_coordinate() {
    // One bit column and two dense ones commit 257 columns, read one row ahead in column 1.
    //
    // Widened to the whole table, the batch is one aligned block: one reduction in all.
    let shapes = [TableShape::new(6, 3)];
    let scheme = scheme(&shapes, &[1]);
    let protocol = protocol(&shapes, &[1]);
    let mut rng = SmallRng::seed_from_u64(8);
    let points = vec![Point::<EF>::rand(&mut rng, 6)];

    let mut prover = challenger();
    let (_, data) = scheme
        .commit(vec![mixed_table(9, 6, 3, 1)], &mut prover)
        .unwrap();
    let proof = scheme
        .open_at(data, &protocol, &points, &mut prover)
        .unwrap();
    assert_eq!(proof.opening.reduction.claims.len(), 1);
    // Both views of every committed column are opened.
    assert_eq!(proof.values.len(), 2 * 257);
}
