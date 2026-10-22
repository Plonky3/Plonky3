//! A multi-STARK proof over WHIR verified inside a circuit, whose own proof then verifies natively.

mod common;

use std::borrow::Cow;
use std::time::Instant;

use common::{Challenger, EF, F, Fibonacci, ValMmcs, chip, config, fibonacci, prove_and_verify};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_batch_stark::{ProverData, prove_batch, verify_batch, verify_batch_in};
use p3_circuit::tables::CircuitAir;
use p3_circuit::{Circuit, CircuitArith, MirrorChallenger};
use p3_dft::Radix2DFTSmallBatch;
use p3_field::PrimeCharacteristicRing;
use p3_koala_bear::default_koalabear_poseidon2_16;
use p3_lookup::{Count, InteractionBuilder};
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::config::MultiStarkConfig;
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, VerificationError, VerifierInstance,
    VerifierInstances, VerifyingKey, prove, setup, verify, verify_in,
};
use p3_sumcheck::OpeningBatch;
use p3_sumcheck::layout::{Layout, PrefixProver, Table, Witness};
use p3_util::log2_ceil_usize;
use p3_whir::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirProver, WhirProverData,
};

/// Variables each WHIR round folds, also the per-table padding floor.
const FOLDING: usize = 4;

/// Base-two logarithm of every inner trace height.
const LOG_HEIGHT: usize = 6;

/// The WHIR layout the inner proofs commit with.
type L = PrefixProver<F, EF>;

/// The inner commitment scheme.
type Pcs = WhirProver<EF, F, Radix2DFTSmallBatch<F>, ValMmcs, Challenger, L>;

/// A multi-STARK configuration over KoalaBear with WHIR openings.
struct InnerConfig {
    /// The scheme for the main trace, sized for its stacked arity.
    pcs: Pcs,
    /// The scheme for the preprocessed trace, sized for its stacked arity.
    preprocessed_pcs: Pcs,
}

impl MultiStarkConfig for InnerConfig {
    type Val = F;
    type Challenge = EF;
    type Challenger = Challenger;
    type Pcs = Pcs;

    fn pcs(&self) -> &Pcs {
        &self.pcs
    }

    fn preprocessed_pcs(&self) -> &Pcs {
        &self.preprocessed_pcs
    }

    fn min_num_variables(&self) -> usize {
        FOLDING
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
        L::new_witness(tables, FOLDING)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a WhirProverData<F, EF, ValMmcs, L>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// A configuration for main and preprocessed stacked polynomials of the given arities.
fn inner_config(num_variables: usize, preprocessed_num_variables: usize) -> InnerConfig {
    InnerConfig {
        pcs: whir(num_variables),
        preprocessed_pcs: whir(preprocessed_num_variables),
    }
}

/// A WHIR scheme for a stacked polynomial of `num_variables` variables.
fn whir(num_variables: usize) -> Pcs {
    let folding_factor = FoldingFactor::Constant(FOLDING);
    let schedule = folding_factor
        .compute_folding_schedule(num_variables)
        .unwrap();
    let mut rate = 1;
    let round_log_inv_rates = schedule[..schedule.len() - 1]
        .iter()
        .map(|folding| {
            rate += folding - 1;
            rate
        })
        .collect();
    let params = ProtocolParameters {
        security_level: 32,
        pow_bits: 8,
        round_log_inv_rates,
        folding_factor,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };
    let perm = default_koalabear_poseidon2_16();
    let mmcs = ValMmcs::new(
        common::Hash::new(perm.clone()),
        common::Compress::new(perm),
        0,
    );
    Pcs::new(
        WhirConfig::new(num_variables, params).unwrap(),
        Radix2DFTSmallBatch::default(),
        mmcs,
    )
}

/// A fresh inner transcript.
fn challenger() -> Challenger {
    Challenger::new(default_koalabear_poseidon2_16())
}

/// A Fibonacci AIR, a local multiset equality, or a table of fixed and periodic columns.
enum Inner {
    /// The Fibonacci recurrence with public ends.
    Fib,
    /// Column one is a permutation of column zero.
    Permutation,
    /// Column zero copies a preprocessed column, column one scales it by a periodic one.
    Fixed,
}

/// The preprocessed column of [`Inner::Fixed`].
fn fixed_column() -> Vec<F> {
    (0..1 << LOG_HEIGHT)
        .map(|row| F::from_usize(3 * row + 1))
        .collect()
}

/// The periodic column of [`Inner::Fixed`].
fn periodic_column() -> Vec<F> {
    [2, 7, 1, 8].map(F::from_u32).to_vec()
}

/// The main trace of [`Inner::Fixed`].
fn fixed_trace() -> RowMajorMatrix<F> {
    let periodic = periodic_column();
    RowMajorMatrix::new(
        fixed_column()
            .into_iter()
            .enumerate()
            .flat_map(|(row, value)| [value, value * periodic[row % periodic.len()]])
            .collect(),
        2,
    )
}

impl BaseAir<F> for Inner {
    fn width(&self) -> usize {
        2
    }

    fn num_public_values(&self) -> usize {
        match self {
            Self::Fib => 3,
            Self::Permutation | Self::Fixed => 0,
        }
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        match self {
            Self::Fib => vec![0, 1],
            Self::Permutation | Self::Fixed => Vec::new(),
        }
    }

    fn preprocessed_width(&self) -> usize {
        match self {
            Self::Fixed => 1,
            Self::Fib | Self::Permutation => 0,
        }
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        matches!(self, Self::Fixed).then(|| RowMajorMatrix::new(fixed_column(), 1))
    }

    fn num_periodic_columns(&self) -> usize {
        usize::from(matches!(self, Self::Fixed))
    }

    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
        match self {
            Self::Fixed => Cow::Owned(vec![periodic_column()]),
            Self::Fib | Self::Permutation => Cow::Owned(Vec::new()),
        }
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for Inner {
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::Fib => Fibonacci.eval(builder),
            Self::Permutation => {
                let local = builder.main().current_slice().to_vec();
                builder.push_local_interaction([
                    (vec![local[0].into()], Count::bounded(AB::Expr::ONE, 1)),
                    (vec![local[1].into()], Count::provided(-AB::Expr::ONE)),
                ]);
            }
            Self::Fixed => {
                let local = builder.main().current_slice().to_vec();
                let fixed = builder.preprocessed().current_slice()[0];
                let periodic = builder.periodic_values()[0];
                builder.assert_eq(local[0], fixed);
                builder.assert_eq(local[1], local[0] * periodic.into());
            }
        }
    }
}

/// The reversal of `0, 1, ..., n - 1` beside itself.
fn permutation_trace(n: usize) -> RowMajorMatrix<F> {
    RowMajorMatrix::new(
        (0..n)
            .flat_map(|row| [F::from_usize(row), F::from_usize(n - 1 - row)])
            .collect(),
        2,
    )
}

/// One inner batch: its AIRs, traces and public values.
struct Batch {
    airs: Vec<Inner>,
    traces: Vec<RowMajorMatrix<F>>,
    publics: Vec<Vec<F>>,
}

/// A Fibonacci batch from `(a, b)`, with permutation and fixed tables when `lookup` is set.
fn batch(a: F, b: F, lookup: bool) -> Batch {
    let (trace, publics) = fibonacci(a, b, 1 << LOG_HEIGHT);
    let mut batch = Batch {
        airs: vec![Inner::Fib],
        traces: vec![trace],
        publics: vec![publics],
    };
    if lookup {
        batch.airs.push(Inner::Permutation);
        batch.traces.push(permutation_trace(1 << LOG_HEIGHT));
        batch.publics.push(Vec::new());
        batch.airs.push(Inner::Fixed);
        batch.traces.push(fixed_trace());
        batch.publics.push(Vec::new());
    }
    batch
}

/// The inner proof of a batch, with its configuration and key.
///
/// Native verification accepts it exactly when the trace is `honest`.
fn prove_inner(
    batch: &Batch,
    honest: bool,
) -> (
    InnerConfig,
    VerifyingKey<InnerConfig>,
    MultiStarkProof<InnerConfig>,
) {
    let config = inner_config(
        log2_ceil_usize((batch.airs.len() * 2) << LOG_HEIGHT),
        LOG_HEIGHT,
    );
    let airs = batch.airs.iter().collect::<Vec<_>>();
    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();
    let proof = prove(
        &config,
        ProverInstances::new(
            batch
                .airs
                .iter()
                .zip(&batch.traces)
                .zip(&batch.publics)
                .map(|((air, trace), publics)| {
                    ProverInstance::new(air, Table::new(trace.transpose()), &pk, publics)
                })
                .collect(),
        ),
        0,
        &mut challenger(),
    )
    .unwrap();
    let native = verify(&config, instances(batch, &vk), &proof, 0, &mut challenger());
    assert_eq!(native.is_ok(), honest, "native verification: {native:?}");
    (config, vk, proof)
}

/// The verifier-side instances of a batch.
fn instances<'a>(
    batch: &'a Batch,
    vk: &'a VerifyingKey<InnerConfig>,
) -> VerifierInstances<'a, InnerConfig, Inner> {
    VerifierInstances::new(
        batch
            .airs
            .iter()
            .zip(&batch.publics)
            .map(|(air, publics)| VerifierInstance::new(air, vk, LOG_HEIGHT, publics))
            .collect(),
    )
}

/// The circuit verifying an inner proof, its public values as circuit inputs.
fn verify_in_circuit(
    batch: &Batch,
    config: &InnerConfig,
    vk: &VerifyingKey<InnerConfig>,
    proof: &MultiStarkProof<InnerConfig>,
) -> Result<Circuit<F, EF, common::Chip>, VerificationError<p3_whir::VerifierError>> {
    let mut arith = CircuitArith::<F, EF, _>::new(chip());
    let lifted = arith.public_inputs(&batch.publics);
    let lifted = lifted.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut mirror = challenger().mirror(&arith);
    verify_in(
        &mut arith,
        config,
        instances(batch, vk),
        &lifted,
        proof,
        0,
        &mut mirror,
    )?;
    drop(mirror);
    Ok(arith.build())
}

/// Print every table's height.
fn report(label: &str, circuit: &Circuit<F, EF, common::Chip>) {
    let rows = circuit
        .traces
        .iter()
        .map(|trace| trace.values.len() / trace.width)
        .collect::<Vec<_>>();
    std::println!("{label}: table heights {rows:?}");
}

#[test]
fn a_multi_stark_fibonacci_proof_verifies_inside_a_circuit() {
    let batch = batch(F::ZERO, F::ONE, false);
    let (inner, vk, proof) = prove_inner(&batch, true);
    let circuit = verify_in_circuit(&batch, &inner, &vk, &proof)
        .expect("the inner proof verifies inside the circuit");
    report("fibonacci", &circuit);
    prove_and_verify(&config(2), &circuit);
}

#[test]
fn a_multi_stark_proof_with_lookups_and_fixed_columns_verifies_inside_a_circuit() {
    let batch = batch(F::ZERO, F::ONE, true);
    let (inner, vk, proof) = prove_inner(&batch, true);
    assert!(proof.lookup.is_some());
    let circuit = verify_in_circuit(&batch, &inner, &vk, &proof)
        .expect("the inner proof verifies inside the circuit");
    report("fibonacci + lookup + preprocessed + periodic", &circuit);
    prove_and_verify(&config(2), &circuit);
}

#[test]
fn a_tampered_multi_stark_proof_is_rejected_inside_the_circuit() {
    let batch = batch(F::ZERO, F::ONE, true);
    let (config, vk, mut proof) = prove_inner(&batch, true);

    // One opened trace value moves off the committed polynomial.
    let opened = &proof.opening.evals[0];
    let mut current = opened.current().to_vec();
    current[0] += EF::ONE;
    proof.opening.evals[0] = OpeningBatch::new(current, opened.next().to_vec());

    assert!(matches!(
        verify_in_circuit(&batch, &config, &vk, &proof),
        Err(VerificationError::Opening(_))
    ));
}

#[test]
fn multi_stark_proofs_of_one_shape_give_one_verifying_key() {
    let circuits = [(F::ZERO, F::ONE), (F::new(3), F::new(5))].map(|(a, b)| {
        let batch = batch(a, b, true);
        let (config, vk, proof) = prove_inner(&batch, true);
        verify_in_circuit(&batch, &config, &vk, &proof).unwrap()
    });

    // The outer circuits differ only in their witness, so their preprocessed columns agree.
    let [one, two] = &circuits;
    assert_eq!(one.airs.len(), two.airs.len());
    for (a, b) in one.airs.iter().zip(&two.airs) {
        assert_eq!(a.preprocessed_trace(), b.preprocessed_trace());
    }
}

#[test]
fn a_violated_constraint_fails_the_closing_check_inside_the_circuit() {
    // One interior cell breaks the recurrence while both ends still match the public values.
    let mut batch = batch(F::ZERO, F::ONE, false);
    batch.traces[0].values[2 * 10] += F::ONE;
    let (config, vk, proof) = prove_inner(&batch, false);

    assert!(matches!(
        verify_in_circuit(&batch, &config, &vk, &proof),
        Err(VerificationError::Zerocheck(_))
    ));
}

/// The arities of a circuit's stacked main and preprocessed traces.
fn stacked_arities(circuit: &Circuit<F, EF, common::Chip>) -> (usize, usize) {
    let cells = |width: &dyn Fn(&CircuitAir<F, EF, common::Chip>) -> usize| {
        circuit
            .airs
            .iter()
            .zip(circuit.log_heights())
            .map(|(air, log_height)| width(air) << log_height)
            .sum::<usize>()
    };
    (
        log2_ceil_usize(cells(&|air| air.width())),
        log2_ceil_usize(cells(&|air| air.preprocessed_width())),
    )
}

#[test]
fn a_multi_stark_proof_of_a_circuit_verifies_inside_a_circuit() {
    // The first circuit verifies an inner multi-STARK proof.
    let batch = batch(F::ZERO, F::ONE, true);
    let (inner, vk, proof) = prove_inner(&batch, true);
    let mut circuit = verify_in_circuit(&batch, &inner, &vk, &proof).unwrap();
    report("circuit", &circuit);

    // Its tables, proven with batch-STARK over FRI at about the WHIR target below.
    let fri = config(16);
    let start = Instant::now();
    let fri_instances = circuit.instances::<common::Config>();
    let fri_data = ProverData::from_instances(&fri, &fri_instances).unwrap();
    let fri_proof = prove_batch(&fri, &fri_instances, &fri_data).unwrap();
    let fri_prove = start.elapsed();
    let publics = circuit.public_values();
    let start = Instant::now();
    verify_batch(&fri, &circuit.airs, &fri_proof, &publics, &fri_data.common).unwrap();
    let fri_verify = start.elapsed();
    let mut arith = CircuitArith::<F, EF, _>::new(chip());
    let lifted = arith.public_inputs(&publics);
    verify_batch_in(
        &mut arith,
        &fri,
        &circuit.airs,
        &fri_proof,
        &lifted,
        &fri_data.common,
    )
    .unwrap();
    report("batch-STARK recursion", &arith.build());

    // The same tables, padded to the folding floor and proven with a multi-STARK over WHIR.
    circuit.pad(FOLDING);
    let (main, preprocessed) = stacked_arities(&circuit);
    let outer = inner_config(main, preprocessed);
    let start = Instant::now();
    let airs = circuit.airs.iter().collect::<Vec<_>>();
    let (outer_pk, outer_vk) = setup(&outer, &airs, &mut challenger()).unwrap();
    let outer_proof = circuit
        .prove_multi_stark(&outer, &outer_pk, &mut challenger())
        .unwrap();
    let whir_prove = start.elapsed();
    report("circuit, padded", &circuit);

    let log_heights = circuit.log_heights();
    let outer_instances = || {
        VerifierInstances::new(
            circuit
                .airs
                .iter()
                .zip(&log_heights)
                .zip(&publics)
                .map(|((air, &log_height), publics)| {
                    VerifierInstance::new(air, &outer_vk, log_height, publics)
                })
                .collect(),
        )
    };
    let start = Instant::now();
    verify(
        &outer,
        outer_instances(),
        &outer_proof,
        0,
        &mut challenger(),
    )
    .expect("the circuit's multi-STARK proof verifies natively");
    let whir_verify = start.elapsed();
    std::println!(
        "batch-STARK prove {fri_prove:?} verify {fri_verify:?}; multi-STARK prove {whir_prove:?} verify {whir_verify:?}; stacked arities {main} / {preprocessed}"
    );

    // The second circuit verifies that proof, and batch-STARK proves the second circuit.
    let mut arith = CircuitArith::<F, EF, _>::new(chip());
    let lifted = arith.public_inputs(&publics);
    let lifted = lifted.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut mirror = challenger().mirror(&arith);
    verify_in(
        &mut arith,
        &outer,
        outer_instances(),
        &lifted,
        &outer_proof,
        0,
        &mut mirror,
    )
    .expect("the circuit's multi-STARK proof verifies inside a circuit");
    drop(mirror);
    let recursive = arith.build();
    report("multi-STARK recursion", &recursive);
    prove_and_verify(&config(2), &recursive);
}
