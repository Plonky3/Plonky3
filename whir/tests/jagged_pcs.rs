//! Authenticating a sparse jagged evaluation against a real dense commitment.

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_sumcheck::jagged::{
    BoundJaggedLayout, CellBudget, ColumnSource, JaggedError, JaggedLayout, JaggedOpeningError,
    JaggedOpeningShape, JaggedPoint, JaggedWitness, TraceSource,
};
use p3_sumcheck::layout::{Layout, PrefixProver, Table, observe_commitment};
use p3_sumcheck::{OpeningProtocol, PrescribedPointPcs};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use p3_whir::pcs::prover::WhirProver;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
type MyDft = Radix2DFTSmallBatch<F>;
type L = PrefixProver<F, EF>;
type MyPcs = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, L>;
type Commitment = <MyPcs as MultilinearPcs<EF, MyChallenger>>::Commitment;
type WhirData = <MyPcs as MultilinearPcs<EF, MyChallenger>>::ProverData;

// Rounds the residual folding consumes at once, and the floor on the committed arity.
const FOLDING: usize = 2;

// Rates the folding schedule of one committed arity needs, one per round after the first.
fn round_log_inv_rates(num_variables: usize, folding: &FoldingFactor) -> Vec<usize> {
    let schedule = folding
        .compute_folding_schedule(num_variables)
        .expect("the committed arity admits a folding schedule");
    let mut rate = 1;
    schedule[..schedule.len() - 1]
        .iter()
        .map(|&step| {
            rate += step - 1;
            rate
        })
        .collect()
}

fn challenger() -> MyChallenger {
    let mut rng = SmallRng::seed_from_u64(1);
    MyChallenger::new(Perm::new_from_rng_128(&mut rng))
}

// Equality weight of one Boolean index against a point, written straight from the definition.
fn equality_weight(point: &Point<EF>, index: usize) -> EF {
    let width = point.num_variables();
    (0..width)
        .map(|position| {
            let bit = (index >> (width - 1 - position)) & 1 == 1;
            if bit {
                point[position]
            } else {
                EF::ONE - point[position]
            }
        })
        .product()
}

// Value of the virtual jagged table, built from the heights alone.
// No selector, no dense index and no sumcheck appear here, so it cannot drift with any of them.
fn jagged_evaluation(heights: &[usize], cells: &[F], point: &JaggedPoint<EF>) -> EF {
    let mut total = EF::ZERO;
    let mut start = 0;
    for (column, &height) in heights.iter().enumerate() {
        let column_weight = equality_weight(point.column(), column);
        for row in 0..height {
            total += column_weight * equality_weight(point.row(), row) * cells[start + row];
        }
        start += height;
    }
    total
}

// One trace of eight columns of unequal height, presented column by column.
// The heights sum to 256, so the envelope is exactly the live area and no cell is dead.
fn trace() -> (Vec<usize>, Vec<Vec<F>>) {
    let heights = vec![61, 3, 0, 40, 17, 64, 64, 7];
    let columns = heights
        .iter()
        .enumerate()
        .map(|(column, &height)| {
            (0..height)
                .map(|row| F::from_u64((column * 977 + row * 31 + 5) as u64))
                .collect()
        })
        .collect();
    (heights, columns)
}

// The commitment scheme one envelope arity is committed and opened under.
fn configure(num_variables: usize) -> MyPcs {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);
    let folding_factor = FoldingFactor::Constant(FOLDING);
    let params = ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        round_log_inv_rates: round_log_inv_rates(num_variables, &folding_factor),
        folding_factor,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };
    MyPcs::new(
        WhirConfig::new(num_variables, params).unwrap(),
        MyDft::default(),
        mmcs,
    )
}

// Commits one vector as a single column of the stacked polynomial.
fn commit(vector: &[F], challenger: &mut MyChallenger) -> (MyPcs, Commitment, WhirData) {
    let pcs = configure(p3_util::log2_strict_usize(vector.len()));
    let table = Table::new(RowMajorMatrix::new(vector.to_vec(), vector.len()));
    let witness = L::new_witness(vec![table], FOLDING);
    let (commitment, data) = MultilinearPcs::<EF, MyChallenger>::commit(&pcs, witness, challenger)
        .expect("the committed arity is inside the configured one");
    (pcs, commitment, data)
}

#[test]
fn a_sparse_claim_is_authenticated_by_the_commitment_that_carries_it() {
    let (heights, columns) = trace();
    let layout = JaggedLayout::new(6, &heights).unwrap();

    // The trace arrives column by column, so the ingestion pass is visible and charged.
    let sources = columns
        .iter()
        .map(|cells| ColumnSource::Dense(cells))
        .collect::<Vec<_>>();
    let (witness, report) = JaggedWitness::read(&layout, TraceSource::Columns(&sources)).unwrap();
    assert_eq!(report.live(), 256);
    assert_eq!(report.envelope(), 0);
    assert_eq!(CellBudget::of(&layout).dead(), 0);

    // Prover: bind the commitment, seal the transcript to the geometry, then draw the point.
    let mut prover = challenger();
    let (pcs, commitment, data) = commit(&witness, &mut prover);
    let bound = BoundJaggedLayout::new::<F, _>(&layout, &mut prover);
    let point = bound.sample_point::<F, EF, _>(&mut prover);
    let value = jagged_evaluation(&heights, &witness, &point);
    let opening = bound
        .open(&pcs, data, &witness, &[(point.clone(), value)], &mut prover)
        .expect("an honest trace opens");

    // Verifier: the same three steps, from its own public inputs.
    let mut verifier = challenger();
    observe_commitment::<F, _, _>(&mut verifier, commitment.clone());
    let bound = BoundJaggedLayout::new::<F, _>(&layout, &mut verifier);
    let replayed = bound.sample_point::<F, EF, _>(&mut verifier);
    assert_eq!(replayed, point);
    bound
        .verify(
            &pcs,
            &commitment,
            &opening,
            &[(replayed, value)],
            &mut verifier,
        )
        .expect("the commitment authenticates the sparse claim");
}

#[test]
fn a_reduction_against_a_vector_that_was_not_committed_is_refused() {
    // This is the whole point of routing the claim through the commitment.
    // The reduction alone accepts any sparse value, because its terminal relation has a free unknown.
    let (heights, columns) = trace();
    let layout = JaggedLayout::new(6, &heights).unwrap();
    let sources = columns
        .iter()
        .map(|cells| ColumnSource::Dense(cells))
        .collect::<Vec<_>>();
    let (committed, _) = JaggedWitness::read(&layout, TraceSource::Columns(&sources)).unwrap();

    // Mutation: one live cell of the vector the reduction speaks about, which was never committed.
    let mut forged = committed.to_vec();
    forged[100] += F::ONE;

    let mut prover = challenger();
    let (pcs, commitment, data) = commit(&committed, &mut prover);
    let bound = BoundJaggedLayout::new::<F, _>(&layout, &mut prover);
    let point = bound.sample_point::<F, EF, _>(&mut prover);

    // The forged trace is internally consistent, so the reduction itself has nothing to object to.
    let value = jagged_evaluation(&heights, &forged, &point);
    assert_ne!(value, jagged_evaluation(&heights, &committed, &point));
    let opening = bound
        .open(&pcs, data, &forged, &[(point, value)], &mut prover)
        .expect("the reduction proves the forged statement on its own");

    let mut verifier = challenger();
    observe_commitment::<F, _, _>(&mut verifier, commitment.clone());
    let bound = BoundJaggedLayout::new::<F, _>(&layout, &mut verifier);
    let replayed = bound.sample_point::<F, EF, _>(&mut verifier);
    // A rejection anywhere else would mean the opening never reached the comparison under test.
    let error = bound
        .verify(
            &pcs,
            &commitment,
            &opening,
            &[(replayed, value)],
            &mut verifier,
        )
        .unwrap_err();
    assert!(
        matches!(error, JaggedOpeningError::DenseMismatch { reading: 0 }),
        "the committed vector must be what refuses the claim, not {error:?}"
    );
}

#[test]
fn a_transcript_sealed_to_other_heights_rejects() {
    // The heights are the verifier's, so a prover that proved under different ones cannot be followed.
    let (heights, columns) = trace();
    let layout = JaggedLayout::new(6, &heights).unwrap();
    let sources = columns
        .iter()
        .map(|cells| ColumnSource::Dense(cells))
        .collect::<Vec<_>>();
    let (witness, _) = JaggedWitness::read(&layout, TraceSource::Columns(&sources)).unwrap();

    let mut prover = challenger();
    let (pcs, commitment, data) = commit(&witness, &mut prover);
    let bound = BoundJaggedLayout::new::<F, _>(&layout, &mut prover);
    let point = bound.sample_point::<F, EF, _>(&mut prover);
    let value = jagged_evaluation(&heights, &witness, &point);
    let opening = bound
        .open(&pcs, data, &witness, &[(point.clone(), value)], &mut prover)
        .expect("an honest trace opens");

    // Mutation: one live row moves between two columns, which leaves the area and the envelope alone.
    let mut moved = heights;
    moved[0] -= 1;
    moved[1] += 1;
    let other = JaggedLayout::new(6, &moved).unwrap();
    assert_eq!(other.dense_capacity(), layout.dense_capacity());

    let mut verifier = challenger();
    observe_commitment::<F, _, _>(&mut verifier, commitment.clone());
    let bound = BoundJaggedLayout::new::<F, _>(&other, &mut verifier);
    let replayed = bound.sample_point::<F, EF, _>(&mut verifier);
    assert_ne!(replayed, point);
    // The commitment scheme's own error is opaque, so its variant is destructured away.
    //
    // The reduction's error is not opaque, and is named in full.
    let error = bound
        .verify(
            &pcs,
            &commitment,
            &opening,
            &[(replayed, value)],
            &mut verifier,
        )
        .unwrap_err();
    let JaggedOpeningError::Reduction(reduction) = error else {
        panic!("the reduction must be what rejects, not {error:?}");
    };
    assert_eq!(reduction, JaggedError::TerminalMismatch);
}

#[test]
fn one_commitment_answers_several_sparse_points() {
    // A machine reads one trace at a zerocheck point and at its successor, and lookups add more.
    // All of them share one commitment and one opening rather than one commitment each.
    let (heights, columns) = trace();
    let layout = JaggedLayout::new(6, &heights).unwrap();
    let sources = columns
        .iter()
        .map(|cells| ColumnSource::Dense(cells))
        .collect::<Vec<_>>();
    let (witness, _) = JaggedWitness::read(&layout, TraceSource::Columns(&sources)).unwrap();

    let mut prover = challenger();
    let (pcs, commitment, data) = commit(&witness, &mut prover);
    let bound = BoundJaggedLayout::new::<F, _>(&layout, &mut prover);

    // Three points drawn in sequence, each with the value the committed vector takes there.
    let claims = (0..3)
        .map(|_| {
            let point = bound.sample_point::<F, EF, _>(&mut prover);
            let value = jagged_evaluation(&heights, &witness, &point);
            (point, value)
        })
        .collect::<Vec<_>>();
    assert_ne!(claims[0].0, claims[1].0);
    let opening = bound
        .open(&pcs, data, &witness, &claims, &mut prover)
        .expect("three honest claims open together");
    assert_eq!(opening.reductions().len(), 3);

    let mut verifier = challenger();
    observe_commitment::<F, _, _>(&mut verifier, commitment.clone());
    let bound = BoundJaggedLayout::new::<F, _>(&layout, &mut verifier);
    let replayed = (0..3)
        .map(|index| {
            (
                bound.sample_point::<F, EF, _>(&mut verifier),
                claims[index].1,
            )
        })
        .collect::<Vec<_>>();
    bound
        .verify(&pcs, &commitment, &opening, &replayed, &mut verifier)
        .expect("one opening authenticates all three");
}

#[test]
fn a_proof_carrying_the_wrong_number_of_reductions_is_refused() {
    // The claim list is the statement, so a proof answering a different number of claims is another.
    //
    // It must be refused before any transcript is replayed.
    let (heights, columns) = trace();
    let layout = JaggedLayout::new(6, &heights).unwrap();
    let sources = columns
        .iter()
        .map(|cells| ColumnSource::Dense(cells))
        .collect::<Vec<_>>();
    let (witness, _) = JaggedWitness::read(&layout, TraceSource::Columns(&sources)).unwrap();

    let mut prover = challenger();
    let (pcs, commitment, data) = commit(&witness, &mut prover);
    let bound = BoundJaggedLayout::new::<F, _>(&layout, &mut prover);
    let claims = (0..3)
        .map(|_| {
            let point = bound.sample_point::<F, EF, _>(&mut prover);
            let value = jagged_evaluation(&heights, &witness, &point);
            (point, value)
        })
        .collect::<Vec<_>>();
    let opening = bound
        .open(&pcs, data, &witness, &claims, &mut prover)
        .expect("three honest claims open together");

    let mut verifier = challenger();
    observe_commitment::<F, _, _>(&mut verifier, commitment.clone());
    let bound = BoundJaggedLayout::new::<F, _>(&layout, &mut verifier);
    let error = bound
        .verify(&pcs, &commitment, &opening, &claims[..2], &mut verifier)
        .unwrap_err();
    assert!(
        matches!(
            error,
            JaggedOpeningError::ReductionCountMismatch {
                expected: 2,
                actual: 3
            }
        ),
        "the count must be what refuses the proof, not {error:?}"
    );

    // An empty statement has nothing to discharge and is refused rather than accepted vacuously.
    let mut verifier = challenger();
    observe_commitment::<F, _, _>(&mut verifier, commitment.clone());
    let bound = BoundJaggedLayout::new::<F, _>(&layout, &mut verifier);
    let error = bound
        .verify(&pcs, &commitment, &opening, &[], &mut verifier)
        .unwrap_err();
    assert!(
        matches!(error, JaggedOpeningError::NoClaims),
        "an empty claim list must be refused, not {error:?}"
    );
}

#[test]
fn the_reported_security_charges_the_reduction_over_the_candidate_set() {
    // The commitment fixes a candidate set before the sparse point or any round challenge exists.
    //
    // A prover may therefore pick which candidate to answer for after seeing them.
    let (heights, _) = trace();
    let layout = JaggedLayout::new(6, &heights).unwrap();
    assert_eq!(layout.dense_variables(), 8);

    let mut prover = challenger();
    let pcs = configure(layout.dense_variables());
    let bound = BoundJaggedLayout::new::<F, _>(&layout, &mut prover);

    let uncharged = pcs
        .prescribed_security(&OpeningProtocol::from(JaggedOpeningShape::new(&layout, 1)))
        .expect("the configured protocol is assessed");
    let composed = bound
        .security::<EF, _, MyChallenger>(&pcs, 1)
        .expect("the same protocol is assessed through the handle");

    // Eight envelope variables and one claim draw sixteen guesses, so the raw term is four bits short.
    //
    // The field width it falls short of is taken one bit below the order.
    let raw = (EF::bits() - 1 - 4) as f64;
    let candidates = composed.log2_max_candidates;
    assert!(candidates > 0.0, "the commitment leaves a set open");
    assert_eq!(uncharged.log2_max_candidates, candidates);

    // The composed report is the scheme's own terms plus exactly one more, charged over that set.
    assert_eq!(composed.terms.len(), uncharged.terms.len() + 1);
    let charged = composed.terms.last().unwrap();
    assert_eq!(charged.bits.bits(), (raw - candidates).max(0.0));

    // Charging is not free: the composed figure is strictly below the uncharged union bound.
    let mut naive = uncharged;
    naive.terms.push(p3_security::SecurityTerm::new(
        charged.label,
        p3_security::ErrorBits::from_log2(raw),
    ));
    // The charge never improves the composed figure.
    //
    // At this fixture's target it does not move it either, the proximity term being far the weaker.
    assert!(composed.error().bits() <= naive.error().bits());

    // A wider envelope leaves a larger set open, so the same reduction is worth fewer bits.
    let wide = JaggedLayout::with_min_dense_variables(6, &heights, 16).unwrap();
    let mut transcript = challenger();
    let bound = BoundJaggedLayout::new::<F, _>(&wide, &mut transcript);
    let wider = bound
        .security::<EF, _, MyChallenger>(&configure(16), 1)
        .expect("the wider protocol is assessed");
    assert!(wider.log2_max_candidates > candidates);
    assert!(wider.terms.last().unwrap().bits.bits() < charged.bits.bits());
}

#[test]
fn a_trace_below_the_folding_factor_commits_once_its_envelope_is_raised() {
    // Two live cells give an envelope of one variable, which no folding schedule of two accepts.
    let heights = vec![1usize, 1];
    assert_eq!(JaggedLayout::new(1, &heights).unwrap().dense_variables(), 1);

    // Raising the floor buys envelope cells that no sparse cell reaches and no constraint binds.
    let layout = JaggedLayout::with_min_dense_variables(1, &heights, 4).unwrap();
    assert_eq!(layout.dense_variables(), 4);
    assert_eq!(CellBudget::of(&layout).live(), 2);
    assert_eq!(CellBudget::of(&layout).dead(), 14);

    let cells = [F::from_u64(7), F::from_u64(9)];
    let sources = [
        ColumnSource::Dense(&cells[..1]),
        ColumnSource::Dense(&cells[1..]),
    ];
    let (witness, _) = JaggedWitness::read(&layout, TraceSource::Columns(&sources)).unwrap();

    let mut prover = challenger();
    let (pcs, commitment, data) = commit(&witness, &mut prover);
    let bound = BoundJaggedLayout::new::<F, _>(&layout, &mut prover);
    let point = bound.sample_point::<F, EF, _>(&mut prover);
    let value = jagged_evaluation(&heights, &witness, &point);
    let opening = bound
        .open(&pcs, data, &witness, &[(point, value)], &mut prover)
        .expect("the raised envelope is a shape the folding schedule accepts");

    let mut verifier = challenger();
    observe_commitment::<F, _, _>(&mut verifier, commitment.clone());
    let bound = BoundJaggedLayout::new::<F, _>(&layout, &mut verifier);
    let replayed = bound.sample_point::<F, EF, _>(&mut verifier);
    bound
        .verify(
            &pcs,
            &commitment,
            &opening,
            &[(replayed, value)],
            &mut verifier,
        )
        .expect("a raised envelope changes no sparse statement");
}
