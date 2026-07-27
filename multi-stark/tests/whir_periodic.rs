//! End-to-end multilinear AIR SNARK with periodic columns, over WHIR.
//!
//! Periodic columns are public parameters derived from the AIR, never committed.
//!
//! The prover folds them into the zerocheck alongside the committed columns.
//! The verifier recomputes their multilinear extensions in closed form at the bound point.
//!
//! ```text
//!     main trace         : WHIR commitment, opened at the bound point
//!     preprocessed trace : WHIR commitment made once at setup
//!     periodic columns   : no commitment at all
//! ```

use std::borrow::Cow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::MultiStarkConfig;
use p3_multi_stark::zerocheck::ZerocheckError;
use p3_multi_stark::{
    ProverInstance, ProverInstances, VerificationError, VerifierInstance, VerifierInstances, prove,
    setup, verify,
};
use p3_sumcheck::layout::{Layout, PrefixProver, Table, Witness};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use p3_whir::{
    DomainSeparator, FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirProver,
};
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
type TestPcs = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, L>;

/// First-round folding factor.
/// It is also the per-table padding floor.
const FOLDING: usize = 2;

/// Main trace column count.
const MAIN_WIDTH: usize = 1;

/// Preprocessed trace column count.
const PREPROCESSED_WIDTH: usize = 1;

/// Period of the first periodic column.
const PERIOD_A: usize = 2;
/// Period of the second periodic column.
const PERIOD_B: usize = 4;

/// A WHIR-backed multilinear AIR configuration over BabyBear.
///
/// The main and preprocessed traces stack different column counts.
/// Their stacked polynomials therefore have different arities, one scheme each.
struct WhirConfigForTest {
    /// Scheme sized for the main stacked trace.
    pcs: TestPcs,
    /// Scheme sized for the preprocessed stacked trace.
    preprocessed_pcs: TestPcs,
}

impl MultiStarkConfig for WhirConfigForTest {
    type Val = F;
    type Challenge = EF;
    type Challenger = MyChallenger;
    type Pcs = TestPcs;

    fn pcs(&self) -> &TestPcs {
        &self.pcs
    }

    fn preprocessed_pcs(&self) -> &TestPcs {
        &self.preprocessed_pcs
    }

    fn min_num_variables(&self) -> usize {
        FOLDING
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
        // Each committed trace stacks as one polynomial.
        // Periodic columns never reach this point at all.
        L::new_witness(tables, FOLDING)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a p3_whir::WhirProverData<F, EF, MyMmcs, L>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// Fixed permutation so prover and verifier transcripts match exactly.
fn perm() -> Perm {
    let mut rng = SmallRng::seed_from_u64(0xD15EA5E);
    Perm::new_from_rng_128(&mut rng)
}

/// Per-round log-inverse rates for a stacked polynomial.
fn default_round_log_inv_rates(num_variables: usize, folding_factor: &FoldingFactor) -> Vec<usize> {
    let folding_schedule = folding_factor
        .compute_folding_schedule(num_variables)
        .expect("valid folding schedule");
    let num_rounds = folding_schedule.len() - 1;
    let mut rates = Vec::with_capacity(num_rounds);
    let mut rate = 1;
    for &folding in folding_schedule.iter().take(num_rounds) {
        rate += folding - 1;
        rates.push(rate);
    }
    rates
}

/// Build a WHIR scheme sized for a stacked polynomial of a given column count.
fn pcs_for(log_height: usize, width: usize) -> TestPcs {
    let stacked_num_variables = log_height + log2_ceil_usize(width);
    let folding_factor = FoldingFactor::Constant(FOLDING);

    let mmcs = MyMmcs::new(MyHash::new(perm()), MyCompress::new(perm()), 0);
    let params = ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        round_log_inv_rates: default_round_log_inv_rates(stacked_num_variables, &folding_factor),
        folding_factor,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };
    let whir_config = WhirConfig::new(stacked_num_variables, params).unwrap();
    TestPcs::new(whir_config, MyDft::default(), mmcs)
}

/// Build a configuration sized for a trace shape, one scheme per committed trace.
///
/// A test with no preprocessed trace still carries a preprocessed scheme.
/// It stays unused: setup commits nothing for such an AIR.
fn config_for(log_height: usize, width: usize) -> WhirConfigForTest {
    WhirConfigForTest {
        pcs: pcs_for(log_height, width),
        preprocessed_pcs: pcs_for(log_height, PREPROCESSED_WIDTH),
    }
}

/// A challenger seeded with the same domain separator on both proof and verify sides.
fn challenger(config: &WhirConfigForTest) -> MyChallenger {
    let mut challenger = MyChallenger::new(perm());
    let mut ds = DomainSeparator::new(vec![]);
    config.pcs.add_domain_separator::<8>(&mut ds);
    ds.observe_domain_separator(&mut challenger);
    challenger
}

/// The two period vectors, of different lengths.
///
/// ```text
///     column 0: [10, 20]        repeats every 2 rows
///     column 1: [1, 2, 3, 4]    repeats every 4 rows
/// ```
fn periodic_columns() -> Vec<Vec<F>> {
    vec![
        vec![F::from_u64(10), F::from_u64(20)],
        vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ],
    ]
}

/// Width-1 main AIR tied to two current-row periodic columns of different periods.
///
/// Every row asserts that the main column holds the sum of the two periodic values.
///
/// The AIR reads no next row.
/// It therefore declares an empty main next-row set.
struct PeriodicAir;

impl BaseAir<F> for PeriodicAir {
    fn width(&self) -> usize {
        MAIN_WIDTH
    }
    fn num_periodic_columns(&self) -> usize {
        periodic_columns().len()
    }
    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
        Cow::Owned(periodic_columns())
    }
    fn main_next_row_columns(&self) -> Vec<usize> {
        // Current-row only: no successor claim is needed.
        Vec::new()
    }
}

impl<AB: AirBuilder<F = F>> Air<AB> for PeriodicAir {
    fn eval(&self, builder: &mut AB) {
        // Read the single main column and both periodic values at the current row.
        let main = builder.main().current_slice()[0];
        let periodic = builder.periodic_values();
        let sum: AB::Expr = periodic[0].into() + periodic[1].into();
        builder.assert_eq(main, sum);
    }
}

/// A satisfying trace for the sum AIR.
///
///     main[i] = periodic_0[i mod 2] + periodic_1[i mod 4]
fn periodic_trace(n: usize) -> RowMajorMatrix<F> {
    let cols = periodic_columns();
    let values = (0..n)
        .map(|i| cols[0][i % PERIOD_A] + cols[1][i % PERIOD_B])
        .collect();
    RowMajorMatrix::new(values, MAIN_WIDTH)
}

#[test]
fn prove_verify_periodic_roundtrips() {
    // Invariant: a satisfying trace with periodic columns round-trips through WHIR.
    let n = 256;
    let trace = periodic_trace(n);
    let config = config_for(log2_strict_usize(n), MAIN_WIDTH);

    // This AIR has no preprocessed trace, and periodic columns are never committed.
    // Setup therefore commits nothing and yields empty keys.
    let (pk, vk) = setup(&config, &[&PeriodicAir], &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &PeriodicAir,
            Table::new(trace.transpose()),
            &pk,
            &[],
        )]),
        0,
        &mut challenger(&config),
    );
    // Nothing was committed at setup.
    // There is therefore no preprocessed opening to carry.
    assert!(proof.preprocessed_opening.is_none());

    verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(
            &PeriodicAir,
            &vk,
            log2_strict_usize(n),
            &[],
        )]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .expect("honest periodic proof must verify");
}

#[test]
fn verify_rejects_violated_periodic_constraint() {
    // Mutation: break row 0 so the main column no longer holds the periodic sum.
    //
    //     row 0 main : sum + 1
    //                  → the batched constraint is nonzero on that row
    let n = 256;
    let mut trace = periodic_trace(n);
    trace.values[0] += F::ONE;
    let config = config_for(log2_strict_usize(n), MAIN_WIDTH);

    let (pk, vk) = setup(&config, &[&PeriodicAir], &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &PeriodicAir,
            Table::new(trace.transpose()),
            &pk,
            &[],
        )]),
        0,
        &mut challenger(&config),
    );

    // The claimed zero sum cannot close against a nonzero constraint value.
    let err = verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(
            &PeriodicAir,
            &vk,
            log2_strict_usize(n),
            &[],
        )]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            VerificationError::Zerocheck(ZerocheckError::FinalSumMismatch)
        ),
        "expected zerocheck final-sum mismatch, got {err:?}"
    );
}

/// The fixed preprocessed column for a height-`n` trace.
///
///     row i holds 3 + 2 * i
fn fixed_column(n: usize) -> Vec<F> {
    (0..n).map(|i| F::from_u64(3 + 2 * i as u64)).collect()
}

/// AIR carrying all three column groups at once.
///
/// Every row asserts:
///
/// ```text
///     main[0] = preprocessed[0] + periodic[0] * periodic[1]
/// ```
///
/// The periodic factors multiply, lifting the constraint to degree two.
///
/// Nothing is read on the next row.
/// No group therefore carries a successor claim.
struct PeriodicPreprocessedAir {
    /// Trace height the preprocessed column is generated to match.
    height: usize,
}

impl BaseAir<F> for PeriodicPreprocessedAir {
    fn width(&self) -> usize {
        MAIN_WIDTH
    }
    fn preprocessed_width(&self) -> usize {
        PREPROCESSED_WIDTH
    }
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(RowMajorMatrix::new(
            fixed_column(self.height),
            PREPROCESSED_WIDTH,
        ))
    }
    fn num_periodic_columns(&self) -> usize {
        periodic_columns().len()
    }
    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
        Cow::Owned(periodic_columns())
    }
    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl<AB: AirBuilder<F = F>> Air<AB> for PeriodicPreprocessedAir {
    fn eval(&self, builder: &mut AB) {
        // One value from each of the three column groups, all on the current row.
        let main = builder.main().current_slice()[0];
        let preprocessed = builder.preprocessed().current_slice()[0];
        let periodic = builder.periodic_values();

        // The periodic factors multiply, lifting the constraint to degree two.
        let product: AB::Expr = periodic[0].into() * periodic[1].into();
        builder.assert_eq(main, preprocessed.into() + product);
    }
}

/// The satisfying main trace for the three-group AIR.
///
///     main[i] = fixed[i] + periodic_0[i mod 2] * periodic_1[i mod 4]
fn periodic_preprocessed_trace(n: usize) -> RowMajorMatrix<F> {
    let cols = periodic_columns();
    let fixed = fixed_column(n);
    let values = (0..n)
        .map(|i| fixed[i] + cols[0][i % PERIOD_A] * cols[1][i % PERIOD_B])
        .collect();
    RowMajorMatrix::new(values, MAIN_WIDTH)
}

#[test]
fn prove_verify_periodic_with_preprocessed_roundtrips() {
    // Invariant: the three column groups coexist end to end.
    //
    //     main         : committed, opened at the bound point
    //     preprocessed : committed at setup, opened at the same point
    //     periodic     : uncommitted, recomputed in closed form
    let n = 256;
    let log_height = log2_strict_usize(n);
    let air = PeriodicPreprocessedAir { height: n };
    let trace = periodic_preprocessed_trace(n);
    let config = config_for(log_height, MAIN_WIDTH);

    // Setup commits the preprocessed column and nothing else.
    let (pk, vk) = setup(&config, &[&air], &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &air,
            Table::new(trace.transpose()),
            &pk,
            &[],
        )]),
        0,
        &mut challenger(&config),
    );

    // The preprocessed commitment is opened at the bound point, hence one opening here.
    assert!(proof.preprocessed_opening.is_some());

    verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &[])]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .expect("honest three-group proof must verify");
}

#[test]
fn verify_rejects_violated_periodic_preprocessed_constraint() {
    // Mutation: break row 0 of the main trace.
    //
    //     row 0 main : fixed + product + 1
    //                  → the degree-two constraint is nonzero on that row
    let n = 256;
    let log_height = log2_strict_usize(n);
    let air = PeriodicPreprocessedAir { height: n };
    let mut trace = periodic_preprocessed_trace(n);
    trace.values[0] += F::ONE;
    let config = config_for(log_height, MAIN_WIDTH);

    let (pk, vk) = setup(&config, &[&air], &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &air,
            Table::new(trace.transpose()),
            &pk,
            &[],
        )]),
        0,
        &mut challenger(&config),
    );

    // The claimed zero sum cannot close against a nonzero constraint value.
    let err = verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &[])]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            VerificationError::Zerocheck(ZerocheckError::FinalSumMismatch)
        ),
        "expected zerocheck final-sum mismatch, got {err:?}"
    );
}
