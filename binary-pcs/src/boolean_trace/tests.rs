use alloc::string::String;
use alloc::{format, vec};

use p3_binary_field::BinaryField128;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::ring_switch::bits::BitRingSwitchProofError;
use p3_sumcheck::{OpeningBatch, PrescribedPointPcs, TableSpec};
use p3_symmetric::CryptographicHasher;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::params::BinaryPcsParams;
use crate::test_util::{MyChallenger, MyMmcs, challenger, mmcs};

type EF = BinaryField128;

// The reduction error a rejection carries, so a test names the check that refused it.
fn reduction_error<E: core::fmt::Debug>(
    error: BooleanTraceError<EF, E>,
) -> BitRingSwitchProofError {
    match error {
        BooleanTraceCommitmentError::Boolean(BooleanPcsError::ReductionProof(inner)) => inner,
        other => panic!("{other:?}"),
    }
}

/// Columns every fixture table holds.
const FIXTURE_WIDTH: usize = 2;

/// A Boolean table of `2^log_height` rows and two columns.
fn table(seed: u64, log_height: usize) -> Table<EF> {
    let rows = 1usize << log_height;
    let mut rng = SmallRng::seed_from_u64(seed);
    let cells = (0..FIXTURE_WIDTH * rows)
        .map(|_| EF::from_bool(rng.random::<bool>()))
        .collect();
    Table::new(RowMajorMatrix::new(cells, rows))
}

/// A Boolean table with an explicitly chosen width.
fn table_with_width(seed: u64, log_height: usize, width: usize) -> Table<EF> {
    let rows = 1usize << log_height;
    let mut rng = SmallRng::seed_from_u64(seed);
    let cells = (0..width * rows)
        .map(|_| EF::from_bool(rng.random::<bool>()))
        .collect();
    Table::new(RowMajorMatrix::new(cells, rows))
}

fn packed_table(table: &Table<EF>) -> Table<EF> {
    let height = 1usize << table.num_variables();
    let words = (0..height.div_ceil(WORD_BITS))
        .flat_map(|block| {
            (0..table.num_polys()).map(move |column| {
                (0..WORD_BITS).fold(0u64, |word, lane| {
                    let row = block * WORD_BITS + lane;
                    if row < height && table.column(column).value(row) == EF::ONE {
                        word | (1u64 << lane)
                    } else {
                        word
                    }
                })
            })
        })
        .collect();
    Table::from_packed_bits(
        RowMajorMatrix::new(words, table.num_polys()),
        table.num_variables(),
    )
}

#[test]
fn packed_column_sums_match_a_per_bit_sum() {
    // Part of one word, one block, one partial task, and several tasks with a partial last one.
    let blocks_past_tasks = (3 * BLOCKS_PER_TASK + 1) * WORD_BITS;
    for (height, width, seed) in [
        (1, 1, 1),
        (4, 1, 2),
        (32, 5, 3),
        (64, 3, 4),
        (5 * WORD_BITS, 1, 5),
        (blocks_past_tasks, 7, 6),
    ] {
        let mut rng = SmallRng::seed_from_u64(seed);
        let used = if height < WORD_BITS {
            (1u64 << height) - 1
        } else {
            u64::MAX
        };
        let words = (0..height.div_ceil(WORD_BITS) * width)
            .map(|_| rng.random::<u64>() & used)
            .collect::<Vec<_>>();
        let row_weights = (0..height).map(|_| rng.random()).collect::<Vec<EF>>();

        let expected = (0..width)
            .map(|column| {
                (0..height)
                    .filter(|&row| {
                        (words[(row / WORD_BITS) * width + column] >> (row % WORD_BITS)) & 1 == 1
                    })
                    .map(|row| row_weights[row])
                    .sum::<EF>()
            })
            .collect::<Vec<_>>();
        let words = RowMajorMatrix::new(words, width);
        assert_eq!(
            packed_column_sums(&words, &row_weights),
            expected,
            "{height}x{width}"
        );
    }
}

#[test]
fn gathered_bits_place_every_cell_at_its_slot() {
    // Packed and dense, long and short, and packed tables spanning several column groups.
    let shapes = [
        TableShape::new(7, 2 * GATHER_GROUP + 3),
        TableShape::new(6, 5),
        TableShape::new(4, GATHER_GROUP + 1),
        TableShape::new(3, 3),
    ];
    let dense = shapes
        .iter()
        .enumerate()
        .map(|(seed, shape)| table_with_width(seed as u64, shape.num_variables(), shape.width()))
        .collect::<Vec<_>>();
    let tables = alloc::vec![
        packed_table(&dense[0]),
        dense[1].clone(),
        packed_table(&dense[2]),
        dense[3].clone(),
    ];
    let scheme = pcs(&shapes);

    let mut expected = alloc::vec![0u64; 1 << (scheme.num_variables() - 6)];
    for placement in scheme.placements(&shapes).unwrap() {
        let table = &dense[placement.idx()];
        let column_len = 1usize << table.num_variables();
        for (column, selector) in placement.selectors().iter().enumerate() {
            for row in 0..column_len {
                if table.column(column).value(row) == EF::ONE {
                    let index = selector.index() * column_len + row;
                    expected[index / WORD_BITS] |= 1 << (index % WORD_BITS);
                }
            }
        }
    }

    let gathered = scheme.gather_bits(&tables).unwrap();
    let gathered = gathered
        .iter()
        .map(|word| word.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(gathered, expected);
}

#[test]
fn a_gather_refuses_the_first_non_boolean_column_in_placement_order() {
    // The taller table is placed first, so its columns are reached before the short table's.
    let shapes = [TableShape::new(3, 2), TableShape::new(7, 4)];
    let scheme = pcs(&shapes);
    let refused = |bad: &[(usize, usize)]| {
        let tables = shapes
            .iter()
            .enumerate()
            .map(|(index, shape)| {
                let rows = 1 << shape.num_variables();
                let table = table_with_width(index as u64, shape.num_variables(), shape.width());
                let mut cells = table.iter_polys().flatten().copied().collect::<Vec<_>>();
                for &(_, column) in bad.iter().filter(|&&(table, _)| table == index) {
                    cells[column * rows + rows / 2] = EF::GENERATOR;
                }
                Table::new(RowMajorMatrix::new(cells, rows))
            })
            .collect::<Vec<_>>();
        match scheme.gather_bits(&tables) {
            Err(BooleanTraceCommitmentError::NonBooleanCell { table, column }) => (table, column),
            _ => panic!("a non-Boolean cell must be refused"),
        }
    };
    assert_eq!(refused(&[(0, 1), (1, 3)]), (1, 3));
    assert_eq!(refused(&[(1, 3), (1, 1)]), (1, 1));
    assert_eq!(refused(&[(0, 1), (0, 0)]), (0, 0));
}

/// A commitment over the batch these shapes describe.
fn pcs(shapes: &[TableShape]) -> BooleanTracePcs<EF, MyMmcs, MyMmcs> {
    let params = BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: 40,
    };
    let (arity, _) = plan_stacked_layout(shapes);
    // The packing absorbs seven variables, so the commitment holds the rest.
    let config = BinaryPcsConfig::try_new::<EF, EF>(arity - 7, params).unwrap();
    BooleanTracePcs::new(config, mmcs(), mmcs(), arity).unwrap()
}

/// Every column of every table, opened at one point per table.
fn protocol(shapes: &[TableShape]) -> OpeningProtocol {
    successor_protocol(shapes, &[])
}

/// Every column of every table at the current row, and the `next` columns one row ahead.
fn successor_protocol(shapes: &[TableShape], next: &[usize]) -> OpeningProtocol {
    OpeningProtocol::new(
        shapes
            .iter()
            .map(|shape| {
                TableSpec::new(
                    *shape,
                    alloc::vec![OpeningBatch::new(
                        (0..shape.width()).collect(),
                        next.to_vec()
                    )],
                )
            })
            .collect(),
    )
}

/// One table opened `num_batches` times, every column read at both rows each time.
fn both_views_protocol(shape: TableShape, num_batches: usize) -> OpeningProtocol {
    OpeningProtocol::new(vec![TableSpec::new(
        shape,
        (0..num_batches)
            .map(|_| OpeningBatch::new((0..shape.width()).collect(), (0..shape.width()).collect()))
            .collect(),
    )])
}

/// One complete-batch proof of both views whose value at `tampered` is moved by one.
///
/// The flow is the prover's own, so the moved value is bound before the column point is
/// drawn, exactly as an honest value would be.
fn forged_successor_proof(
    scheme: &BooleanTracePcs<EF, MyMmcs, MyMmcs>,
    data: BooleanTraceData<EF, MyMmcs>,
    shape: TableShape,
    points: &[Point<EF>],
    tampered: Option<usize>,
    challenger: &mut MyChallenger,
) -> BooleanTraceProof<EF, MyMmcs, MyMmcs> {
    type Scheme = BooleanTracePcs<EF, MyMmcs, MyMmcs>;
    let BooleanTraceCommitmentData { inner, tables } = data;
    let width = shape.width();
    let mut values = Vec::new();
    for point in points {
        let (current, successor) = Scheme::evaluate_views(&tables[0], point, true);
        values.extend(current);
        values.extend(successor);
    }
    if let Some(at) = tampered {
        values[at] += EF::ONE;
    }

    let batch_shape = ColumnBatchShape {
        table_variables: shape.num_variables(),
        width,
        num_batches: points.len(),
        next: true,
    };
    let mut transcript = ColumnBatchProverTranscript::new(challenger, batch_shape);
    let mut openings = Vec::with_capacity(points.len());
    for (point, batch_values) in points.iter().zip(values.chunks_exact(2 * width)) {
        let (current, successor) = batch_values.split_at(width);
        let column_point = transcript.batch(point, current, successor);
        let mut lifted_point = column_point;
        lifted_point.extend(point);
        openings.push(BitOpening {
            point: lifted_point,
            row_variables: shape.num_variables(),
            current: true,
            next: true,
        });
    }
    transcript.finish();

    let (_, opening) = scheme
        .inner
        .open_readings(inner, &openings, challenger)
        .unwrap();
    BooleanTraceCommitmentProof { values, opening }
}

/// A column read one row ahead: row `z` reads row `z + 1`, the last row itself.
fn successor_reading(column: &[EF], point: &Point<EF>) -> EF {
    let rows = column.len();
    let eq = Poly::new_from_point(point.as_slice(), EF::ONE);
    (0..rows)
        .map(|row| eq.as_slice()[row] * column[(row + 1).min(rows - 1)])
        .sum()
}

#[test]
fn an_opened_column_is_the_column_the_table_holds() {
    // Invariant: lifting a row point by a slot address reads that column and no other.
    //
    // Fixture state: three tables, so the slots are neither all one arity nor all aligned.
    //
    //     - table 0   2^10 rows, 2 columns
    //     - table 1   2^8  rows, 2 columns
    //     - table 2   2^4  rows, 2 columns
    //
    // The stack is log2_ceil(2048 + 512 + 32) = 12 variables wide.
    //
    // The third table is what puts a column shorter than one staging word in the batch.
    //
    //     - 2^10 and 2^8 rows  ->  whole words, written by the aligned run
    //     - 2^4 rows           ->  sixteen bits inside one word, set in place
    //
    // A wrong shift on that second path would place the column somewhere else.
    //
    // Every opened value is checked against the column's own multilinear.
    // That reference shares nothing with the packing or with the reduction.
    let shapes = [
        TableShape::new(10, FIXTURE_WIDTH),
        TableShape::new(8, FIXTURE_WIDTH),
        TableShape::new(4, FIXTURE_WIDTH),
    ];
    let tables = alloc::vec![table(0xB100, 10), table(0xB101, 8), table(0xB103, 4)];
    let scheme = pcs(&shapes);
    let protocol = protocol(&shapes);
    assert_eq!(scheme.num_variables(), 12);

    let mut rng = SmallRng::seed_from_u64(0xB102);
    let points = alloc::vec![
        Point::<EF>::rand(&mut rng, 10),
        Point::<EF>::rand(&mut rng, 8),
        Point::<EF>::rand(&mut rng, 4),
    ];

    let mut prover_chal = challenger();
    let (commitment, data) = scheme.commit(tables.clone(), &mut prover_chal).unwrap();
    let proof = scheme
        .open_at(data, &protocol, &points, &mut prover_chal)
        .unwrap();

    let mut verifier_chal = challenger();
    scheme.observe_commitment(&commitment, &mut verifier_chal);
    let evals = scheme
        .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
        .unwrap();

    // One batch per table, each carrying that table's whole width and no successor value.
    assert_eq!(evals.len(), shapes.len());
    for (index, batch) in evals.iter().enumerate() {
        assert!(batch.next().is_empty());
        for (column, &value) in batch.current().iter().enumerate() {
            let reference = Poly::new(tables[index].poly(column).as_slice().to_vec());
            assert_eq!(
                value,
                reference.eval_base(&points[index]),
                "{index}/{column}"
            );
        }
    }
}

#[test]
fn packed_table_round_trip_matches_dense_openings() {
    for (log_height, width, seed) in [(5, 8, 0xB10A), (10, FIXTURE_WIDTH, 0xB10C)] {
        let shapes = [TableShape::new(log_height, width)];
        let dense = table_with_width(seed, log_height, width);
        let packed = packed_table(&dense);
        let scheme = pcs(&shapes);
        let protocol = protocol(&shapes);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(seed + 1), log_height);

        let mut dense_challenger = challenger();
        let (dense_commitment, dense_data) =
            scheme.commit(vec![dense], &mut dense_challenger).unwrap();
        let dense_proof = scheme
            .open_at(
                dense_data,
                &protocol,
                core::slice::from_ref(&point),
                &mut dense_challenger,
            )
            .unwrap();
        let dense_after_open = p3_challenger::CanSample::<EF>::sample(&mut dense_challenger);
        let mut packed_challenger = challenger();
        let (packed_commitment, packed_data) =
            scheme.commit(vec![packed], &mut packed_challenger).unwrap();
        let packed_proof = scheme
            .open_at(
                packed_data,
                &protocol,
                core::slice::from_ref(&point),
                &mut packed_challenger,
            )
            .unwrap();
        let packed_after_open = p3_challenger::CanSample::<EF>::sample(&mut packed_challenger);

        assert_eq!(dense_commitment, packed_commitment);
        assert_eq!(
            postcard::to_allocvec(&dense_proof).unwrap(),
            postcard::to_allocvec(&packed_proof).unwrap()
        );
        assert_eq!(dense_after_open, packed_after_open);
        let mut verifier = challenger();
        scheme.observe_commitment(&packed_commitment, &mut verifier);
        scheme
            .verify_at(
                &packed_commitment,
                &packed_proof,
                &protocol,
                core::slice::from_ref(&point),
                &mut verifier,
            )
            .unwrap();
    }
}

#[test]
fn packed_mixed_heights_and_widths_match_dense_proof_bytes() {
    let shapes = [
        TableShape::new(6, 3),
        TableShape::new(4, 5),
        TableShape::new(7, 2),
    ];
    let dense = vec![
        table_with_width(0xB10C, 6, 3),
        table_with_width(0xB10D, 4, 5),
        table_with_width(0xB10E, 7, 2),
    ];
    let packed = dense.iter().map(packed_table).collect::<Vec<_>>();
    let scheme = pcs(&shapes);
    let protocol = protocol(&shapes);
    let mut rng = SmallRng::seed_from_u64(0xB10F);
    let points = shapes
        .iter()
        .map(|shape| Point::<EF>::rand(&mut rng, shape.num_variables()))
        .collect::<Vec<_>>();

    let mut dense_challenger = challenger();
    let (dense_commitment, dense_data) = scheme.commit(dense, &mut dense_challenger).unwrap();
    let dense_proof = scheme
        .open_at(dense_data, &protocol, &points, &mut dense_challenger)
        .unwrap();
    let dense_after_open = p3_challenger::CanSample::<EF>::sample(&mut dense_challenger);

    let mut packed_challenger = challenger();
    let (packed_commitment, packed_data) = scheme.commit(packed, &mut packed_challenger).unwrap();
    let packed_proof = scheme
        .open_at(packed_data, &protocol, &points, &mut packed_challenger)
        .unwrap();
    let packed_after_open = p3_challenger::CanSample::<EF>::sample(&mut packed_challenger);

    assert_eq!(dense_commitment, packed_commitment);
    assert_eq!(
        postcard::to_allocvec(&dense_proof).unwrap(),
        postcard::to_allocvec(&packed_proof).unwrap()
    );
    assert_eq!(dense_after_open, packed_after_open);
}

#[test]
fn the_sampled_path_draws_the_same_points_on_both_sides() {
    // Invariant: the sampled convention needs no point to cross the wire.
    // Each side binds the commitment itself.
    //
    //     commit  ->  binds the root, then the caller draws nothing of its own
    //     verify  ->  binds the root, then samples the points the prover sampled
    //
    // A caller that bound the root by hand would bind it twice and sample elsewhere.
    // Both sides therefore run on a fresh sponge, as every other scheme here expects.
    //
    // Fixture state: one table of 2^10 rows and two columns, so the stack has arity 11.
    let shapes = [TableShape::new(10, FIXTURE_WIDTH)];
    let scheme = pcs(&shapes);
    let protocol = protocol(&shapes);

    let mut prover_chal = challenger();
    let (commitment, data) = scheme
        .commit(alloc::vec![table(0xB200, 10)], &mut prover_chal)
        .unwrap();
    let proof = scheme
        .open(data, protocol.clone(), &mut prover_chal)
        .unwrap();

    let mut verifier_chal = challenger();
    scheme
        .verify(&commitment, &proof, &mut verifier_chal, protocol.clone())
        .unwrap();

    // Binding it a second time moves every later draw, so the two sides split.
    let mut double_bound = challenger();
    scheme.observe_commitment(&commitment, &mut double_bound);
    let refused = scheme
        .verify(&commitment, &proof, &mut double_bound, protocol)
        .unwrap_err();
    // The split transcript gives the replayed reduction a different column point.
    assert_eq!(
        reduction_error(refused),
        BitRingSwitchProofError::ClaimMismatch
    );
}

#[test]
fn a_disagreeing_shape_or_point_count_is_refused() {
    // Invariant: the shape agreement is checked before the transcript moves at all.
    let shapes = [TableShape::new(10, FIXTURE_WIDTH)];
    let scheme = pcs(&shapes);

    // Mutation: one column fewer changes the retained table shape.
    let narrow = [TableShape::new(10, 1)];
    let mut chal = challenger();
    let (_, data) = scheme
        .commit(alloc::vec![table(0xB300, 10)], &mut chal)
        .unwrap();
    let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB301), 10);
    let Err(error) = scheme.open_at(
        data,
        &protocol(&narrow),
        core::slice::from_ref(&point),
        &mut chal,
    ) else {
        panic!("a shape set stacking elsewhere describes another commitment")
    };
    let BooleanTraceCommitmentError::TableShapeMismatch {
        table: table_index,
        expected,
        actual,
    } = error
    else {
        panic!("the retained source shape must be checked before stacking")
    };
    assert_eq!(table_index, 0);
    assert_eq!(expected, TableShape::new(10, 1));
    assert_eq!(actual, TableShape::new(10, FIXTURE_WIDTH));

    // Mutation: no point at all, against a protocol scheduling one batch.
    let (_, data) = scheme
        .commit(alloc::vec![table(0xB300, 10)], &mut chal)
        .unwrap();
    let Err(error) = scheme.open_at(data, &protocol(&shapes), &[], &mut chal) else {
        panic!("a batch with no point is opened nowhere")
    };
    assert!(matches!(
        error,
        BooleanTraceCommitmentError::PointCount {
            expected: 1,
            actual: 0
        }
    ));
}

#[test]
fn a_proof_short_of_one_value_is_refused() {
    // Invariant: the value run must cover every opened column, one value each.
    //
    // Fixture state: one table of two columns, so an honest proof carries two values.
    //
    // Mutation: drop the last value, leaving one against two opened readings.
    let shapes = [TableShape::new(10, FIXTURE_WIDTH)];
    let scheme = pcs(&shapes);
    let protocol = protocol(&shapes);

    let mut prover_chal = challenger();
    let (commitment, data) = scheme
        .commit(alloc::vec![table(0xB400, 10)], &mut prover_chal)
        .unwrap();
    let points = alloc::vec![Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB401), 10)];
    let mut proof = scheme
        .open_at(data, &protocol, &points, &mut prover_chal)
        .unwrap();
    proof.values.pop();

    let mut verifier_chal = challenger();
    scheme.observe_commitment(&commitment, &mut verifier_chal);
    let Err(error) = scheme.verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
    else {
        panic!("a proof answering for one column cannot answer for two")
    };
    assert!(matches!(
        error,
        BooleanTraceCommitmentError::ValueCount {
            expected: 2,
            actual: 1
        }
    ));
}

#[test]
fn the_claim_plan_writes_every_value_position_exactly_once() {
    // Invariant: the value run is laid out by the claim plan alone, so no position is
    // answered twice and none is left at the fill the run starts from.
    //
    //     current [0, 1, 2], next []        the whole width at the current row
    //     current [2, 0, 1], next [1, 2]    both sides, out of table order
    //     current [0, 1],    next [1, 0]    both sides, each reordering the other
    //     current [],        next [2, 0]    a successor view read nowhere else
    //     current [3, 1],    next [1, 3, 0] a successor side wider than the current one
    for (width, current, next) in [
        (3, vec![0, 1, 2], vec![]),
        (3, vec![2, 0, 1], vec![1, 2]),
        (2, vec![0, 1], vec![1, 0]),
        (3, vec![], vec![2, 0]),
        (4, vec![3, 1], vec![1, 3, 0]),
    ] {
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(8, width),
            vec![OpeningBatch::new(current.clone(), next.clone())],
        )]);
        let len = value_count(&protocol);
        assert_eq!(len, current.len() + next.len());
        assert!(
            covers_every_value(&column_claims(&protocol), len),
            "{current:?} / {next:?}"
        );
    }

    // Several batches over several tables share one run, so the cursor has to advance
    // past every batch's own two sides before the next batch writes.
    let protocol = OpeningProtocol::new(vec![
        TableSpec::new(
            TableShape::new(8, 3),
            vec![
                OpeningBatch::new(vec![2, 0, 1], vec![1, 2]),
                OpeningBatch::new(vec![0], vec![2, 0]),
            ],
        ),
        TableSpec::new(
            TableShape::new(6, 2),
            vec![OpeningBatch::new(vec![1, 0], vec![0])],
        ),
    ]);
    let len = value_count(&protocol);
    assert_eq!(len, 5 + 3 + 3);
    assert!(covers_every_value(&column_claims(&protocol), len));

    // Both ways a plan can miss are refused, so the checks above are not vacuous.
    //
    //     one position written twice   a claim answering both views at one position
    //     one position never written   no claim at all against a run of one value
    let collided = ColumnClaim {
        table: 0,
        column: 0,
        opening: 0,
        current_at: Some(0),
        next_at: Some(0),
    };
    assert!(!covers_every_value(&[collided], 1));
    assert!(!covers_every_value(&[], 1));
}

#[test]
fn a_single_table_batch_uses_one_reduction_for_all_columns() {
    // Invariant: one complete current-row batch for one table is discharged by one
    // column-point ring switch, regardless of the table width.
    let shape = TableShape::new(8, 3);
    let scheme = pcs(&[shape]);
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        shape,
        vec![OpeningBatch::new(vec![0, 1, 2], Vec::new())],
    )]);
    let table = table_with_width(0xB500, 8, 3);
    let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB501), 8);

    let mut prover_chal = challenger();
    let (_, data) = scheme.commit(vec![table], &mut prover_chal).unwrap();
    let proof = scheme
        .open_at(data, &protocol, &[point], &mut prover_chal)
        .unwrap();

    assert_eq!(proof.values.len(), 3);
    assert_eq!(proof.opening.reductions.len(), 1);
}

#[test]
fn each_route_emits_the_reductions_its_plan_counts() {
    // Invariant: the route a protocol resolves to counts exactly the reductions its proof
    // carries, so the security charge and the opening read one resolution.
    //
    //     every column, current row              batched      1 reduction
    //     every column, both rows, two batches   batched      2 reductions
    //     current [2, 0]                         per column   2 reductions
    //     current [0, 1, 2], next [1]            per column   3 reductions
    //     two tables, every column               per column   3 + 2 reductions
    let shape = TableShape::new(8, 3);
    let two_tables = [shape, TableShape::new(6, 2)];
    let reordered = OpeningProtocol::new(vec![TableSpec::new(
        shape,
        vec![OpeningBatch::new(vec![2, 0], Vec::new())],
    )]);
    for (shapes, protocol, batched, reductions, seed) in [
        (vec![shape], protocol(&[shape]), true, 1, 0xB700),
        (vec![shape], both_views_protocol(shape, 2), true, 2, 0xB710),
        (vec![shape], reordered, false, 2, 0xB720),
        (
            vec![shape],
            successor_protocol(&[shape], &[1]),
            false,
            3,
            0xB730,
        ),
        (two_tables.to_vec(), protocol(&two_tables), false, 5, 0xB740),
    ] {
        let route = OpeningRoute::new(&protocol);
        assert_eq!(
            matches!(route, OpeningRoute::Batched(_)),
            batched,
            "{seed:#x}"
        );
        assert_eq!(route.num_reductions(), reductions, "{seed:#x}");

        let scheme = pcs(&shapes);
        let tables = shapes
            .iter()
            .enumerate()
            .map(|(index, shape)| {
                table_with_width(seed + index as u64, shape.num_variables(), shape.width())
            })
            .collect();
        let mut rng = SmallRng::seed_from_u64(seed + 0xF);
        let points = protocol
            .iter_openings()
            .map(|(table, _)| Point::<EF>::rand(&mut rng, shapes[table].num_variables()))
            .collect::<Vec<_>>();

        let mut prover_chal = challenger();
        let (_, data) = scheme.commit(tables, &mut prover_chal).unwrap();
        let proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();
        assert_eq!(
            proof.opening.reductions.len(),
            route.num_reductions(),
            "{seed:#x}"
        );
    }
}

#[test]
fn batched_values_are_bound_before_the_column_point() {
    // Changing a claimed column value changes the verifier's column point and therefore
    // cannot be repaired by reusing the original one-reduction proof.
    let shape = TableShape::new(8, 3);
    let scheme = pcs(&[shape]);
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        shape,
        vec![OpeningBatch::new(vec![0, 1, 2], Vec::new())],
    )]);
    let table = table_with_width(0xB502, 8, 3);
    let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB503), 8);

    let mut prover_chal = challenger();
    let (commitment, data) = scheme.commit(vec![table], &mut prover_chal).unwrap();
    let mut proof = scheme
        .open_at(
            data,
            &protocol,
            core::slice::from_ref(&point),
            &mut prover_chal,
        )
        .unwrap();
    proof.values[0] += EF::ONE;

    let mut verifier_chal = challenger();
    scheme.observe_commitment(&commitment, &mut verifier_chal);
    let refused = scheme
        .verify_at(&commitment, &proof, &protocol, &[point], &mut verifier_chal)
        .unwrap_err();
    assert_eq!(
        reduction_error(refused),
        BitRingSwitchProofError::ClaimMismatch
    );
}

#[test]
fn complete_batches_support_multiple_points_and_width_one() {
    // Width one has no column-point coordinates; two opening batches still use two
    // independent inner reductions and preserve the ordinary value return format.
    let shape = TableShape::new(8, 1);
    let scheme = pcs(&[shape]);
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        shape,
        vec![
            OpeningBatch::new(vec![0], Vec::new()),
            OpeningBatch::new(vec![0], Vec::new()),
        ],
    )]);
    let table = table_with_width(0xB504, 8, 1);
    let mut rng = SmallRng::seed_from_u64(0xB505);
    let points = vec![Point::<EF>::rand(&mut rng, 8), Point::rand(&mut rng, 8)];

    let mut prover_chal = challenger();
    let (commitment, data) = scheme
        .commit(vec![table.clone()], &mut prover_chal)
        .unwrap();
    let proof = scheme
        .open_at(data, &protocol, &points, &mut prover_chal)
        .unwrap();
    assert_eq!(proof.values.len(), 2);
    assert_eq!(proof.opening.reductions.len(), 2);

    let mut verifier_chal = challenger();
    scheme.observe_commitment(&commitment, &mut verifier_chal);
    let evals = scheme
        .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
        .unwrap();
    for (point, eval) in points.iter().zip(evals) {
        let reference = Poly::new(table.poly(0).as_slice().to_vec());
        assert_eq!(eval.current()[0], reference.eval_base(point));
    }
}

#[test]
fn optimized_security_charges_batches_and_column_coordinates() {
    let shape = TableShape::new(8, 3);
    let scheme = pcs(&[shape]);
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        shape,
        vec![
            OpeningBatch::new(vec![0, 1, 2], Vec::new()),
            OpeningBatch::new(vec![0, 1, 2], Vec::new()),
        ],
    )]);

    let security = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<
        EF,
        MyChallenger,
    >>::prescribed_security(&scheme, &protocol)
    .unwrap();
    let batching = security
        .terms
        .iter()
        .find(|term| term.label == "column-batching")
        .unwrap();
    assert!((batching.bits.bits() - 126.0).abs() < 1e-9);
    let ring_switch = security
        .terms
        .iter()
        .find(|term| term.label == "bit-ring-switch")
        .unwrap();
    assert!(ring_switch.bits.bits().is_finite());
}

#[test]
fn optimized_security_rejects_a_protocol_with_the_wrong_stacked_arity() {
    let committed_shape = TableShape::new(8, 3);
    let scheme = pcs(&[committed_shape]);
    let protocol_shape = TableShape::new(8, 2);
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        protocol_shape,
        vec![OpeningBatch::new(vec![0, 1], Vec::new())],
    )]);

    let security = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<
        EF,
        MyChallenger,
    >>::prescribed_security(&scheme, &protocol);
    assert!(security.is_none());
}

#[test]
fn non_power_two_complete_batches_round_trip_below_and_above_packing_width() {
    // Widths three and five exercise zero padding, while heights below and above 128
    // rows cover the short-column-in-word and whole-word packing layouts.
    for (width, log_height, num_batches, seed) in [
        (3, 6, 2, 0xB506),
        (3, 8, 2, 0xB507),
        (5, 6, 1, 0xB508),
        (5, 8, 1, 0xB509),
    ] {
        let shape = TableShape::new(log_height, width);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            (0..num_batches)
                .map(|_| OpeningBatch::new((0..width).collect(), Vec::new()))
                .collect(),
        )]);
        let table = table_with_width(seed, log_height, width);
        let mut rng = SmallRng::seed_from_u64(seed + 1);
        let points = (0..num_batches)
            .map(|_| Point::<EF>::rand(&mut rng, log_height))
            .collect::<Vec<_>>();

        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(vec![table.clone()], &mut prover_chal)
            .unwrap();
        let proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();
        assert_eq!(proof.opening.reductions.len(), num_batches);

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let evals = scheme
            .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
            .unwrap();
        for (batch, point) in points.iter().enumerate() {
            for column in 0..width {
                let reference = Poly::new(table.poly(column).as_slice().to_vec());
                assert_eq!(evals[batch].current()[column], reference.eval_base(point));
            }
        }
    }
}

#[test]
fn optimized_proofs_reject_each_column_tampering_reduction_tampering_and_point_reordering() {
    let shape = TableShape::new(8, 3);
    let scheme = pcs(&[shape]);
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        shape,
        vec![
            OpeningBatch::new(vec![0, 1, 2], Vec::new()),
            OpeningBatch::new(vec![0, 1, 2], Vec::new()),
        ],
    )]);
    let table = table_with_width(0xB50A, 8, 3);
    let mut rng = SmallRng::seed_from_u64(0xB50B);
    let points = vec![Point::<EF>::rand(&mut rng, 8), Point::rand(&mut rng, 8)];
    let mut prover_chal = challenger();
    let (commitment, data) = scheme.commit(vec![table], &mut prover_chal).unwrap();
    let proof = scheme
        .open_at(data, &protocol, &points, &mut prover_chal)
        .unwrap();

    // Untampered control.
    let mut verifier_chal = challenger();
    scheme.observe_commitment(&commitment, &mut verifier_chal);
    scheme
        .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
        .unwrap();

    for index in 0..proof.values.len() {
        let mut tampered = proof.clone();
        tampered.values[index] += EF::ONE;
        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let refused = scheme
            .verify_at(
                &commitment,
                &tampered,
                &protocol,
                &points,
                &mut verifier_chal,
            )
            .unwrap_err();
        // A value of the first batch is caught reading its claim off the element.
        // A value of the second is caught when its surviving claim fails to close.
        let expected = if index < 3 {
            BitRingSwitchProofError::ClaimMismatch
        } else {
            BitRingSwitchProofError::FinalCheck
        };
        assert_eq!(reduction_error(refused), expected, "value {index}");
    }

    let mut tampered = proof.clone();
    tampered.opening.reductions.pop();
    let mut verifier_chal = challenger();
    scheme.observe_commitment(&commitment, &mut verifier_chal);
    let refused = scheme
        .verify_at(
            &commitment,
            &tampered,
            &protocol,
            &points,
            &mut verifier_chal,
        )
        .unwrap_err();
    // A dropped reduction is a count disagreement, refused before the transcript moves.
    assert!(
        matches!(
            refused,
            BooleanTraceCommitmentError::Boolean(BooleanPcsError::ClaimCount {
                expected: 2,
                values: 2,
                reductions: 1
            })
        ),
        "{refused:?}"
    );

    let mut verifier_chal = challenger();
    scheme.observe_commitment(&commitment, &mut verifier_chal);
    let reordered = vec![points[1].clone(), points[0].clone()];
    let refused = scheme
        .verify_at(
            &commitment,
            &proof,
            &protocol,
            &reordered,
            &mut verifier_chal,
        )
        .unwrap_err();
    // Swapping the points reduces each claim at the other one's point.
    assert_eq!(
        reduction_error(refused),
        BitRingSwitchProofError::ClaimMismatch
    );
}

#[test]
fn optimized_shape_errors_leave_the_prover_transcript_untouched() {
    let shape = TableShape::new(8, 3);
    let scheme = pcs(&[shape]);
    // A batch naming the two views differently takes the per-column route, whose own
    // shape checks run before it binds anything either.
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        shape,
        vec![OpeningBatch::new(vec![0, 1, 2], vec![0])],
    )]);
    let mut prover_chal = challenger();
    let (_, data) = scheme
        .commit(vec![table_with_width(0xB50C, 8, 3)], &mut prover_chal)
        .unwrap();
    let mut expected = prover_chal.clone();
    let short_point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB50D), 7);
    let error = match scheme.open_at(data, &protocol, &[short_point], &mut prover_chal) {
        Ok(_) => panic!("a point with the wrong arity must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        BooleanTraceCommitmentError::PointArity { table: 0, .. }
    ));
    assert_eq!(
        p3_challenger::CanSample::<EF>::sample(&mut prover_chal),
        p3_challenger::CanSample::<EF>::sample(&mut expected),
    );

    let valid_protocol = OpeningProtocol::new(vec![TableSpec::new(
        shape,
        vec![OpeningBatch::new(vec![0, 1, 2], Vec::new())],
    )]);
    let mut prover_chal = challenger();
    let (_, data) = scheme
        .commit(vec![table_with_width(0xB510, 8, 3)], &mut prover_chal)
        .unwrap();
    let mut expected = prover_chal.clone();
    let bad_point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB511), 7);
    let error = match scheme.open_at(data, &valid_protocol, &[bad_point], &mut prover_chal) {
        Ok(_) => panic!("a point with the wrong arity must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        BooleanTraceCommitmentError::PointArity { table: 0, .. }
    ));
    assert_eq!(
        p3_challenger::CanSample::<EF>::sample(&mut prover_chal),
        p3_challenger::CanSample::<EF>::sample(&mut expected),
    );

    let narrow_shape = TableShape::new(8, 2);
    let narrow_protocol = OpeningProtocol::new(vec![TableSpec::new(
        narrow_shape,
        vec![OpeningBatch::new(vec![0, 1], Vec::new())],
    )]);
    let mut prover_chal = challenger();
    let (_, data) = scheme
        .commit(vec![table_with_width(0xB512, 8, 3)], &mut prover_chal)
        .unwrap();
    let mut expected = prover_chal.clone();
    let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB513), 8);
    let error = match scheme.open_at(data, &narrow_protocol, &[point], &mut prover_chal) {
        Ok(_) => panic!("a protocol with the wrong stacked arity must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        BooleanTraceCommitmentError::TableShapeMismatch { .. }
    ));
    assert_eq!(
        p3_challenger::CanSample::<EF>::sample(&mut prover_chal),
        p3_challenger::CanSample::<EF>::sample(&mut expected),
    );
}

#[test]
fn source_shape_errors_precede_sampling_for_open_and_open_at() {
    for (committed_shapes, protocol_shapes, expected) in [
        (
            vec![TableShape::new(8, 3)],
            vec![TableShape::new(8, 4)],
            "shape",
        ),
        (
            vec![TableShape::new(8, 3)],
            vec![TableShape::new(9, 2)],
            "shape",
        ),
        (
            vec![TableShape::new(8, 2), TableShape::new(8, 2)],
            vec![TableShape::new(8, 4)],
            "count",
        ),
    ] {
        let scheme = pcs(&committed_shapes);
        let tables = committed_shapes
            .iter()
            .enumerate()
            .map(|(index, shape)| {
                table_with_width(0xB520 + index as u64, shape.num_variables(), shape.width())
            })
            .collect();
        let sampled_protocol = protocol(&protocol_shapes);
        let mut sampled_challenger = challenger();
        let (_, data) = scheme.commit(tables, &mut sampled_challenger).unwrap();
        let mut expected_challenger = sampled_challenger.clone();
        let error = match scheme.open(data, sampled_protocol, &mut sampled_challenger) {
            Ok(_) => panic!("a source shape mismatch must be rejected before sampling"),
            Err(error) => error,
        };
        match expected {
            "shape" => assert!(matches!(
                error,
                BooleanTraceCommitmentError::TableShapeMismatch { .. }
            )),
            "count" => assert!(matches!(
                error,
                BooleanTraceCommitmentError::TableCountMismatch { .. }
            )),
            _ => unreachable!(),
        }
        assert_eq!(
            p3_challenger::CanSample::<EF>::sample(&mut sampled_challenger),
            p3_challenger::CanSample::<EF>::sample(&mut expected_challenger),
        );

        let protocol = protocol(&protocol_shapes);
        let points = protocol_shapes
            .iter()
            .map(|shape| {
                Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB530), shape.num_variables())
            })
            .collect::<Vec<_>>();
        let mut challenger = challenger();
        let (_, data) = scheme
            .commit(
                committed_shapes
                    .iter()
                    .enumerate()
                    .map(|(index, shape)| {
                        table_with_width(
                            0xB540 + index as u64,
                            shape.num_variables(),
                            shape.width(),
                        )
                    })
                    .collect(),
                &mut challenger,
            )
            .unwrap();
        let mut expected_challenger = challenger.clone();
        let error = match scheme.open_at(data, &protocol, &points, &mut challenger) {
            Ok(_) => panic!("a source shape mismatch must be rejected before opening"),
            Err(error) => error,
        };
        assert!(matches!(
            (expected, error),
            (
                "shape",
                BooleanTraceCommitmentError::TableShapeMismatch { .. }
            ) | (
                "count",
                BooleanTraceCommitmentError::TableCountMismatch { .. }
            )
        ));
        assert_eq!(
            p3_challenger::CanSample::<EF>::sample(&mut challenger),
            p3_challenger::CanSample::<EF>::sample(&mut expected_challenger),
        );
    }
}

#[test]
fn reordered_subset_batches_use_the_fallback_column_route() {
    let shape = TableShape::new(8, 3);
    let scheme = pcs(&[shape]);
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        shape,
        vec![OpeningBatch::new(vec![2, 0], Vec::new())],
    )]);
    let table = table_with_width(0xB50E, 8, 3);
    let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB50F), 8);

    let mut prover_chal = challenger();
    let (commitment, data) = scheme
        .commit(vec![table.clone()], &mut prover_chal)
        .unwrap();
    let proof = scheme
        .open_at(
            data,
            &protocol,
            core::slice::from_ref(&point),
            &mut prover_chal,
        )
        .unwrap();
    assert_eq!(proof.values.len(), 2);
    assert_eq!(proof.opening.reductions.len(), 2);

    let mut verifier_chal = challenger();
    scheme.observe_commitment(&commitment, &mut verifier_chal);
    let evals = scheme
        .verify_at(
            &commitment,
            &proof,
            &protocol,
            core::slice::from_ref(&point),
            &mut verifier_chal,
        )
        .unwrap();
    assert_eq!(
        evals[0].current(),
        &[
            Poly::new(table.poly(2).as_slice().to_vec()).eval_base(&point),
            Poly::new(table.poly(0).as_slice().to_vec()).eval_base(&point),
        ]
    );
}

#[test]
fn complete_successor_batches_round_trip_around_the_packing_width() {
    // Invariant: a batch claiming every column at both rows is one reduction, and each
    // claimed value is that column's own reading at the batch's point.
    //
    // Fixture state: table arities below, at and above the seven one element absorbs.
    //
    //     2^3 rows   the whole successor view sits inside one element
    //     2^7 rows   exactly one element
    //     2^9 rows   the view outruns one element, so the reduction sends carry and last
    //
    // Width one draws no column coordinate, and width three pads the column point.
    //
    // Every value is checked against a reference built from the source table alone.
    for (log_height, width, num_batches, seed) in [
        (3, 32, 2, 0xB600),
        (7, 8, 1, 0xB602),
        (9, 4, 3, 0xB604),
        (9, 1, 2, 0xB606),
        (8, 3, 2, 0xB608),
    ] {
        let shape = TableShape::new(log_height, width);
        let scheme = pcs(&[shape]);
        let protocol = both_views_protocol(shape, num_batches);
        let table = table_with_width(seed, log_height, width);
        let mut rng = SmallRng::seed_from_u64(seed + 1);
        let points = (0..num_batches)
            .map(|_| Point::<EF>::rand(&mut rng, log_height))
            .collect::<Vec<_>>();

        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(vec![table.clone()], &mut prover_chal)
            .unwrap();
        let proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();
        assert_eq!(proof.values.len(), 2 * width * num_batches);
        assert_eq!(proof.opening.reductions.len(), num_batches);

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let evals = scheme
            .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
            .unwrap();
        for (batch, point) in points.iter().enumerate() {
            for column in 0..width {
                let cells = table.poly(column);
                let reference = Poly::new(cells.as_slice().to_vec());
                assert_eq!(
                    evals[batch].current()[column],
                    reference.eval_base(point),
                    "{batch}/{column}"
                );
                assert_eq!(
                    evals[batch].next()[column],
                    successor_reading(cells.as_slice(), point),
                    "{batch}/{column} next"
                );
            }
        }
    }
}

#[test]
fn packed_successor_tables_match_dense_proof_bytes() {
    // Invariant: a table held as words and the same table held as cells claim the same
    // readings at both rows, so their proofs are one byte run and one sponge state.
    for (log_height, width, seed) in [(5, 8, 0xB60A), (10, FIXTURE_WIDTH, 0xB60C)] {
        let shape = TableShape::new(log_height, width);
        let dense = table_with_width(seed, log_height, width);
        let packed = packed_table(&dense);
        let scheme = pcs(&[shape]);
        let protocol = both_views_protocol(shape, 1);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(seed + 1), log_height);

        let mut dense_challenger = challenger();
        let (dense_commitment, dense_data) =
            scheme.commit(vec![dense], &mut dense_challenger).unwrap();
        let dense_proof = scheme
            .open_at(
                dense_data,
                &protocol,
                core::slice::from_ref(&point),
                &mut dense_challenger,
            )
            .unwrap();
        let dense_after_open = p3_challenger::CanSample::<EF>::sample(&mut dense_challenger);

        let mut packed_challenger = challenger();
        let (packed_commitment, packed_data) =
            scheme.commit(vec![packed], &mut packed_challenger).unwrap();
        let packed_proof = scheme
            .open_at(
                packed_data,
                &protocol,
                core::slice::from_ref(&point),
                &mut packed_challenger,
            )
            .unwrap();
        let packed_after_open = p3_challenger::CanSample::<EF>::sample(&mut packed_challenger);

        assert_eq!(dense_commitment, packed_commitment);
        assert_eq!(
            postcard::to_allocvec(&dense_proof).unwrap(),
            postcard::to_allocvec(&packed_proof).unwrap()
        );
        assert_eq!(dense_after_open, packed_after_open);

        let mut verifier = challenger();
        scheme.observe_commitment(&packed_commitment, &mut verifier);
        scheme
            .verify_at(
                &packed_commitment,
                &packed_proof,
                &protocol,
                core::slice::from_ref(&point),
                &mut verifier,
            )
            .unwrap();
    }
}

#[test]
fn partial_successor_views_use_the_fallback_column_route() {
    // Invariant: a batch naming the two views differently is answered column by column,
    // one claim per column either view names, a column in both sharing one claim.
    //
    //     current [0, 1, 2], next [1]   a subset read one row ahead
    //     current [0],       next [1]   a column read one row ahead and nowhere else
    //     current [0, 1],    next [1]   a column in both sets
    //     current [2, 0],    next [0]   columns out of their table order
    let shape = TableShape::new(8, 3);
    let scheme = pcs(&[shape]);
    let table = table_with_width(0xB60E, 8, 3);
    let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB60F), 8);

    for (current, next, claims) in [
        (vec![0, 1, 2], vec![1], 3),
        (vec![0], vec![1], 2),
        (vec![0, 1], vec![1], 2),
        (vec![2, 0], vec![0], 2),
    ] {
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![OpeningBatch::new(current.clone(), next.clone())],
        )]);

        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(vec![table.clone()], &mut prover_chal)
            .unwrap();
        let proof = scheme
            .open_at(
                data,
                &protocol,
                core::slice::from_ref(&point),
                &mut prover_chal,
            )
            .unwrap();
        assert_eq!(proof.values.len(), current.len() + next.len());
        assert_eq!(proof.opening.reductions.len(), claims);

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let evals = scheme
            .verify_at(
                &commitment,
                &proof,
                &protocol,
                core::slice::from_ref(&point),
                &mut verifier_chal,
            )
            .unwrap();

        for (&column, &value) in current.iter().zip(evals[0].current()) {
            let reference = Poly::new(table.poly(column).as_slice().to_vec());
            assert_eq!(value, reference.eval_base(&point), "current {column}");
        }
        for (&column, &value) in next.iter().zip(evals[0].next()) {
            let cells = table.poly(column);
            assert_eq!(
                value,
                successor_reading(cells.as_slice(), &point),
                "next {column}"
            );
        }
    }
}

#[test]
fn reordered_successor_views_keep_each_side_in_its_own_order() {
    // Invariant: the two sides of a batch are laid out independently, so a column named
    // by both lands at the position its own side gives it.
    //
    //     current [0, 1],    next [1, 0]   each side reverses the other
    //     current [2, 0, 1], next [1, 2]   three columns, neither side in table order
    //
    // Both shapes take the per-column route, one claim per column either side names.
    //
    // Every returned value is checked against a reference built from the table alone.
    let shape = TableShape::new(8, 3);
    let scheme = pcs(&[shape]);
    let table = table_with_width(0xB618, 8, 3);
    let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB619), 8);

    for (current, next, claims) in [(vec![0, 1], vec![1, 0], 2), (vec![2, 0, 1], vec![1, 2], 3)] {
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![OpeningBatch::new(current.clone(), next.clone())],
        )]);

        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(vec![table.clone()], &mut prover_chal)
            .unwrap();
        let proof = scheme
            .open_at(
                data,
                &protocol,
                core::slice::from_ref(&point),
                &mut prover_chal,
            )
            .unwrap();
        assert_eq!(proof.values.len(), current.len() + next.len());
        assert_eq!(proof.opening.reductions.len(), claims);

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let evals = scheme
            .verify_at(
                &commitment,
                &proof,
                &protocol,
                core::slice::from_ref(&point),
                &mut verifier_chal,
            )
            .unwrap();

        let current_reference = current
            .iter()
            .map(|&column| Poly::new(table.poly(column).as_slice().to_vec()).eval_base(&point))
            .collect::<Vec<_>>();
        let next_reference = next
            .iter()
            .map(|&column| successor_reading(table.poly(column).as_slice(), &point))
            .collect::<Vec<_>>();
        assert_eq!(evals[0].current(), current_reference, "current {current:?}");
        assert_eq!(evals[0].next(), next_reference, "next {next:?}");
    }
}

#[test]
fn a_second_table_is_read_one_row_ahead_at_its_own_point() {
    // Invariant: two tables in one commitment each read their own rows, so the successor
    // view of one never steps into the slot of the other.
    //
    // Fixture state: 2^10 and 2^8 rows of two columns each, stacking to arity 12.
    let shapes = [
        TableShape::new(10, FIXTURE_WIDTH),
        TableShape::new(8, FIXTURE_WIDTH),
    ];
    let tables = vec![table(0xB610, 10), table(0xB611, 8)];
    let scheme = pcs(&shapes);
    let protocol = successor_protocol(&shapes, &[1]);
    let mut rng = SmallRng::seed_from_u64(0xB612);
    let points = vec![
        Point::<EF>::rand(&mut rng, 10),
        Point::<EF>::rand(&mut rng, 8),
    ];

    let mut prover_chal = challenger();
    let (commitment, data) = scheme.commit(tables.clone(), &mut prover_chal).unwrap();
    let proof = scheme
        .open_at(data, &protocol, &points, &mut prover_chal)
        .unwrap();

    let mut verifier_chal = challenger();
    scheme.observe_commitment(&commitment, &mut verifier_chal);
    let evals = scheme
        .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
        .unwrap();

    for (index, batch) in evals.iter().enumerate() {
        for (column, &value) in batch.current().iter().enumerate() {
            let reference = Poly::new(tables[index].poly(column).as_slice().to_vec());
            assert_eq!(
                value,
                reference.eval_base(&points[index]),
                "{index}/{column}"
            );
        }
        let cells = tables[index].poly(1);
        assert_eq!(
            batch.next(),
            &[successor_reading(cells.as_slice(), &points[index])],
            "{index} next"
        );
    }
}

#[test]
fn a_tampered_successor_value_is_rejected_on_either_route() {
    // Invariant: every value of the run is bound before the challenges that combine it,
    // so moving any one of them, on either route, breaks the proof.
    let shape = TableShape::new(8, 3);
    let scheme = pcs(&[shape]);
    let table = table_with_width(0xB614, 8, 3);
    let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB615), 8);

    // The first protocol combines both views of every column, the second names them apart.
    for (per_column, protocol) in [
        (false, both_views_protocol(shape, 1)),
        (
            true,
            OpeningProtocol::new(vec![TableSpec::new(
                shape,
                vec![OpeningBatch::new(vec![0, 1, 2], vec![1])],
            )]),
        ),
    ] {
        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(vec![table.clone()], &mut prover_chal)
            .unwrap();
        let proof = scheme
            .open_at(
                data,
                &protocol,
                core::slice::from_ref(&point),
                &mut prover_chal,
            )
            .unwrap();

        // Untampered control.
        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        scheme
            .verify_at(
                &commitment,
                &proof,
                &protocol,
                core::slice::from_ref(&point),
                &mut verifier_chal,
            )
            .unwrap();

        for index in 0..proof.values.len() {
            let mut tampered = proof.clone();
            tampered.values[index] += EF::ONE;
            let mut verifier_chal = challenger();
            scheme.observe_commitment(&commitment, &mut verifier_chal);
            let refused = scheme
                .verify_at(
                    &commitment,
                    &tampered,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut verifier_chal,
                )
                .unwrap_err();
            // Combining feeds every value into one column point the current claim answers.
            // Named apart, the lone next value is the one the successor claim answers.
            let expected = if per_column && index >= shape.width() {
                BitRingSwitchProofError::SuccessorClaimMismatch
            } else {
                BitRingSwitchProofError::ClaimMismatch
            };
            assert_eq!(reduction_error(refused), expected, "value {index}");
        }
    }
}

#[test]
fn successor_security_charges_both_views_of_every_batch() {
    // Invariant: the two combined claims of a batch share one column point, so the
    // batching term charges each of them.
    //
    //     current only   two batches over two coordinates   ->  128 - log2(4) = 126
    //     both views     four claims over two coordinates   ->  128 - log2(8) = 125
    let shape = TableShape::new(8, 3);
    let scheme = pcs(&[shape]);
    let security = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<
        EF,
        MyChallenger,
    >>::prescribed_security(&scheme, &both_views_protocol(shape, 2))
    .expect("a successor view is answered, so it is priced");

    let bits = |label: &str| {
        security
            .terms
            .iter()
            .find(|term| term.label == label)
            .unwrap_or_else(|| panic!("missing {label}"))
            .bits
            .bits()
    };
    assert!((bits("column-batching") - 125.0).abs() < 1e-9);
    assert!(bits("bit-ring-switch").is_finite());

    // The per-column route prices a successor protocol too, with no batching term.
    let fallback = OpeningProtocol::new(vec![TableSpec::new(
        shape,
        vec![OpeningBatch::new(vec![0, 1, 2], vec![1])],
    )]);
    let security = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<
        EF,
        MyChallenger,
    >>::prescribed_security(&scheme, &fallback)
    .expect("a per-column successor protocol is priced");
    assert!(
        security
            .terms
            .iter()
            .all(|term| term.label != "column-batching")
    );
}

/// Keccak-256 hex digest of a value's postcard encoding.
fn pinned_digest<T: Serialize>(value: &T) -> String {
    let bytes = postcard::to_allocvec(value).expect("postcard serialization must not fail");
    Keccak256Hash
        .hash_iter(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The bits of the labelled term of a `prescribed_security` report, or a panic naming
/// the label that is missing.
fn term_bits(security: &PrescribedOpeningSecurity, label: &str) -> f64 {
    security
        .terms
        .iter()
        .find(|term| term.label == label)
        .unwrap_or_else(|| panic!("missing {label}"))
        .bits
        .bits()
}

/// Pins the opening proof bytes, the shared post-open/post-verify challenger state, and
/// the `prescribed_security` report, for all three opening routes: the batched route
/// reading both views, the batched route sampled through the transcript, and the
/// per-column route over several tables with a subset, a reordering and a partial
/// successor batch.
#[test]
fn opening_routes_and_their_security_are_pinned() {
    // (a) Batched, both views, prescribed points.
    {
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let protocol = both_views_protocol(shape, 2);
        let table = table_with_width(0xB900, 8, 3);
        let mut rng = SmallRng::seed_from_u64(0xB901);
        let points = vec![Point::<EF>::rand(&mut rng, 8), Point::rand(&mut rng, 8)];

        let mut prover_chal = challenger();
        let (commitment, data) = scheme.commit(vec![table], &mut prover_chal).unwrap();
        let proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();
        let after_open: EF = prover_chal.sample_algebra_element();

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        scheme
            .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
            .unwrap();
        let after_verify: EF = verifier_chal.sample_algebra_element();
        assert_eq!(after_open, after_verify);
        assert_eq!(
            pinned_digest(&(&proof, after_open)),
            "227c4eb378900f053ba51d57ba1668162309e0e8402522329aab065c82e48661"
        );

        let security = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<
            EF,
            MyChallenger,
        >>::prescribed_security(&scheme, &protocol)
        .unwrap();
        assert_eq!(security.terms.len(), 3);
        assert!((term_bits(&security, "column-batching") - 125.0).abs() < 1e-9);
        assert!((term_bits(&security, "binary-pcs-opening") - 121.95560588064154).abs() < 1e-9);
        assert!((term_bits(&security, "bit-ring-switch") - 123.09310940439148).abs() < 1e-9);
        assert!((security.log2_max_candidates - 0.0).abs() < 1e-9);
    }

    // (b) Batched, current only, sampled path.
    {
        let shapes = [TableShape::new(10, 2)];
        let scheme = pcs(&shapes);
        let protocol = protocol(&shapes);
        let table = table_with_width(0xB910, 10, 2);

        let mut prover_chal = challenger();
        let (commitment, data) = scheme.commit(vec![table], &mut prover_chal).unwrap();
        let proof = scheme
            .open(data, protocol.clone(), &mut prover_chal)
            .unwrap();
        let after_open: EF = prover_chal.sample_algebra_element();

        let mut verifier_chal = challenger();
        scheme
            .verify(&commitment, &proof, &mut verifier_chal, protocol.clone())
            .unwrap();
        let after_verify: EF = verifier_chal.sample_algebra_element();
        assert_eq!(after_open, after_verify);
        assert_eq!(
            pinned_digest(&(&proof, after_open)),
            "f53d7dbf97b73103bc07fc16aa2051e00e51b1edccabe4aea741275bcfb58394"
        );

        let security = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<
            EF,
            MyChallenger,
        >>::prescribed_security(&scheme, &protocol)
        .unwrap();
        assert_eq!(security.terms.len(), 3);
        assert!((term_bits(&security, "column-batching") - 128.0).abs() < 1e-9);
        assert!((term_bits(&security, "binary-pcs-opening") - 120.95560588064154).abs() < 1e-9);
        assert!((term_bits(&security, "bit-ring-switch") - 124.09310940439148).abs() < 1e-9);
        assert!((security.log2_max_candidates - 0.0).abs() < 1e-9);
    }

    // (c) Per-column: three tables, a reordered subset, a complete batch and a partial
    // successor batch.
    {
        let shapes = [
            TableShape::new(6, 3),
            TableShape::new(4, 5),
            TableShape::new(7, 2),
        ];
        let scheme = pcs(&shapes);
        let protocol = OpeningProtocol::new(vec![
            TableSpec::new(shapes[0], vec![OpeningBatch::new(vec![2, 0], vec![1])]),
            TableSpec::new(
                shapes[1],
                vec![OpeningBatch::new(
                    (0..shapes[1].width()).collect(),
                    Vec::new(),
                )],
            ),
            TableSpec::new(
                shapes[2],
                vec![
                    OpeningBatch::new(vec![1], vec![0, 1]),
                    OpeningBatch::new(Vec::new(), vec![1]),
                ],
            ),
        ]);
        let tables = vec![
            table_with_width(0xB920, 6, 3),
            table_with_width(0xB921, 4, 5),
            table_with_width(0xB922, 7, 2),
        ];
        let mut rng = SmallRng::seed_from_u64(0xB923);
        let points = vec![
            Point::<EF>::rand(&mut rng, 6),
            Point::<EF>::rand(&mut rng, 4),
            Point::<EF>::rand(&mut rng, 7),
            Point::<EF>::rand(&mut rng, 7),
        ];

        let mut prover_chal = challenger();
        let (commitment, data) = scheme.commit(tables, &mut prover_chal).unwrap();
        let proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();
        let after_open: EF = prover_chal.sample_algebra_element();

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        scheme
            .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
            .unwrap();
        let after_verify: EF = verifier_chal.sample_algebra_element();
        assert_eq!(after_open, after_verify);
        assert_eq!(
            pinned_digest(&(&proof, after_open)),
            "a8773e76db42b1d14631b6ee21baa3ba5eece6dd4423ab6e754d28ada5f9e07c"
        );

        let security = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<
            EF,
            MyChallenger,
        >>::prescribed_security(&scheme, &protocol)
        .unwrap();
        assert_eq!(security.terms.len(), 2);
        assert!((term_bits(&security, "binary-pcs-opening") - 121.77118130950412).abs() < 1e-9);
        assert!((term_bits(&security, "bit-ring-switch") - 120.8401286632216).abs() < 1e-9);
        assert!((security.log2_max_candidates - 0.0).abs() < 1e-9);
    }
}

#[test]
fn a_forged_next_value_is_rejected_at_every_position() {
    // Invariant: every next value is combined into the claim the reduction answers, so
    // a prover that binds a moved one and then opens honestly is still refused.
    //
    // Moving a value in a finished proof cannot show this. The column point moves with
    // it, so such a rejection would not say whether the combination reads that position
    // at all. This flow binds the moved value first, exactly as an honest run does.
    let shape = TableShape::new(8, 3);
    let scheme = pcs(&[shape]);
    let protocol = both_views_protocol(shape, 2);
    let mut rng = SmallRng::seed_from_u64(0xB617);
    let points = vec![Point::<EF>::rand(&mut rng, 8), Point::rand(&mut rng, 8)];

    let forge = |tampered: Option<usize>| {
        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(vec![table_with_width(0xB616, 8, 3)], &mut prover_chal)
            .unwrap();
        let proof =
            forged_successor_proof(&scheme, data, shape, &points, tampered, &mut prover_chal);
        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        scheme.verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
    };

    // Untampered control: this flow is the one the prover plays, so it verifies.
    forge(None).unwrap();

    // Every next value of every batch, which trails that batch's current ones.
    for batch in 0..points.len() {
        for column in 0..shape.width() {
            let at = batch * 2 * shape.width() + shape.width() + column;
            assert_eq!(
                reduction_error(forge(Some(at)).unwrap_err()),
                BitRingSwitchProofError::SuccessorClaimMismatch,
                "next value {at}"
            );
        }
    }
}
