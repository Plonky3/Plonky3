use std::env;
use std::error::Error;

use p3_batch_stark::ProverData;
use p3_circuit::CircuitBuilder;
use p3_circuit_prover::common::get_airs_and_degrees_with_prep;
use p3_circuit_prover::config::KoalaBearConfig;
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, TablePacking, config,
};
use p3_field::PrimeCharacteristicRing;
/// Fibonacci circuit: Compute F(n) and prove correctness
/// Public input: expected_result (F(n))
use p3_koala_bear::KoalaBear;
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

/// Initializes a global logger with default parameters.
fn init_logger() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();
}

type F = KoalaBear;

fn main() -> Result<(), Box<dyn Error>> {
    init_logger();

    let config = config::koala_bear().build();

    let n = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let mut builder = CircuitBuilder::new();

    // Public input: expected F(n)
    let expected_result = builder.alloc_public_input("expected_result");

    // Compute F(n) iteratively
    let mut a = builder.alloc_const(F::ZERO, "F(0)");
    let mut b = builder.alloc_const(F::ONE, "F(1)");

    for _i in 2..=n {
        let next = builder.add(a, b);
        a = b;
        b = next;
    }

    // Assert computed F(n) equals expected result
    builder.connect(b, expected_result);

    builder.dump_allocation_log();

    let circuit = builder.build()?;
    let table_packing = TablePacking::new(4, 4);

    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<KoalaBearConfig, _, 1>(
            &circuit,
            &table_packing,
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    // Set public input
    let expected_fib = compute_fibonacci_classical(n);
    runner.set_public_inputs(&[expected_fib])?;

    let traces = runner.run()?;
    let prover_data = ProverData::from_airs_and_degrees(&config, &mut airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(config).with_table_packing(table_packing);

    let proof = prover.prove_all_tables(&traces, &circuit_prover_data)?;
    prover.verify_all_tables(&proof)?;
    Ok(())
}

fn compute_fibonacci_classical(n: usize) -> F {
    if n == 0 {
        return F::ZERO;
    }
    if n == 1 {
        return F::ONE;
    }

    let mut a = F::ZERO;
    let mut b = F::ONE;

    for _i in 2..=n {
        let next = a + b;
        a = b;
        b = next;
    }

    b
}
