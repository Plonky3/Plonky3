//! End-to-end multilinear AIR SNARK with periodic columns, over WHIR.
//!
//! Periodic columns are public parameters derived from the AIR, never committed.
//! The prover folds them into the zerocheck alongside the committed main trace.
//! The verifier recomputes their multilinear extensions in closed form at the bound point.
//! Only the main trace lives in a WHIR commitment.

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
use p3_multi_stark::{VerificationError, prove, setup, verify};
use p3_multilinear_util::poly::Poly;
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

/// First-round folding factor; also the per-table padding floor.
const FOLDING: usize = 2;

/// Main trace column count.
const MAIN_WIDTH: usize = 1;

/// Period of the first periodic column.
const PERIOD_A: usize = 2;
/// Period of the second periodic column.
const PERIOD_B: usize = 4;

/// A WHIR-backed multilinear AIR configuration over BabyBear.
struct WhirConfigForTest {
    /// The WHIR commitment scheme, fixed to one stacked-table arity.
    pcs: TestPcs,
}

impl MultiStarkConfig for WhirConfigForTest {
    type Val = F;
    type Challenge = EF;
    type Challenger = MyChallenger;
    type Pcs = TestPcs;

    fn pcs(&self) -> &TestPcs {
        &self.pcs
    }

    fn min_num_variables(&self) -> usize {
        FOLDING
    }

    fn build_witness(&self, columns: Vec<Poly<F>>) -> Witness<F> {
        // The main trace commits as a single stacked table; periodic columns are never committed.
        L::new_witness(vec![Table::new(columns)], FOLDING)
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

/// Build a configuration sized for a trace shape.
fn config_for(log_height: usize, width: usize) -> WhirConfigForTest {
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
    WhirConfigForTest {
        pcs: TestPcs::new(whir_config, MyDft::default(), mmcs),
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

/// The two periodic period vectors, of different lengths.
///
/// Column A repeats every two rows.
/// Column B repeats every four rows.
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
/// Every row asserts `main[0] == periodic[0] + periodic[1]`.
/// The AIR reads no next row.
/// So it declares an empty main next-row set.
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

/// A satisfying trace: `main[i] = periodic_A[i mod 2] + periodic_B[i mod 4]`.
fn periodic_trace(n: usize) -> RowMajorMatrix<F> {
    let cols = periodic_columns();
    let values = (0..n)
        .map(|i| cols[0][i % PERIOD_A] + cols[1][i % PERIOD_B])
        .collect();
    RowMajorMatrix::new(values, MAIN_WIDTH)
}

#[test]
fn prove_verify_periodic_roundtrips() {
    // A satisfying trace with periodic columns must prove and verify end to end through WHIR.
    let n = 256;
    let trace = periodic_trace(n);
    let config = config_for(log2_strict_usize(n), MAIN_WIDTH);

    // Periodic columns are not committed.
    // So setup yields empty keys.
    let (pk, vk) = setup(&config, &PeriodicAir, &mut challenger(&config));

    let proof = prove(
        &config,
        &pk,
        &PeriodicAir,
        &trace,
        &[],
        0,
        &mut challenger(&config),
    );
    // No preprocessed trace and no periodic commitment: the proof carries no preprocessed opening.
    assert!(proof.preprocessed_opening.is_none());

    verify(
        &config,
        &vk,
        &PeriodicAir,
        &proof,
        &[],
        0,
        &mut challenger(&config),
    )
    .expect("honest periodic proof must verify");
}

#[test]
fn verify_rejects_violated_periodic_constraint() {
    // Fixture state: a satisfying trace obeys `main == periodic[0] + periodic[1]` on every row.
    let n = 256;
    let mut trace = periodic_trace(n);
    // Mutation: break the first row so the closed-form periodic sum no longer matches.
    trace.values[0] += F::ONE;
    let config = config_for(log2_strict_usize(n), MAIN_WIDTH);

    let (pk, vk) = setup(&config, &PeriodicAir, &mut challenger(&config));

    let proof = prove(
        &config,
        &pk,
        &PeriodicAir,
        &trace,
        &[],
        0,
        &mut challenger(&config),
    );

    // Expected rejection: the zerocheck closes on a nonzero constraint value.
    let err = verify(
        &config,
        &vk,
        &PeriodicAir,
        &proof,
        &[],
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
