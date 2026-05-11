//! Benchmarks: trace-to-matrix conversion and full proving.
//!
//! Instance shape and size are parameterized so results are comparable and
//! scaling is visible.

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_batch_stark::ProverData;
use p3_circuit::CircuitBuilder;
use p3_circuit_prover::air::{AluAir, ConstAir, PublicAir};
use p3_circuit_prover::common::get_airs_and_degrees_with_prep;
use p3_circuit_prover::config::KoalaBearConfig;
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, TablePacking, config,
};
use p3_field::PrimeCharacteristicRing;
use p3_koala_bear::KoalaBear;

type F = KoalaBear;

fn fib_circuit(n: usize) -> (p3_circuit::Circuit<F>, F) {
    let mut builder = CircuitBuilder::new();
    let expected_result = builder.alloc_public_input("expected_result");
    let mut a = builder.alloc_const(F::ZERO, "F(0)");
    let mut b = builder.alloc_const(F::ONE, "F(1)");
    for _ in 2..=n {
        let next = builder.add(a, b);
        a = b;
        b = next;
    }
    builder.connect(b, expected_result);
    let circuit = builder.build().unwrap();
    let expected_fib = fib_classical(n);
    (circuit, expected_fib)
}

fn fib_classical(n: usize) -> F {
    if n == 0 {
        return F::ZERO;
    }
    if n == 1 {
        return F::ONE;
    }
    let mut a = F::ZERO;
    let mut b = F::ONE;
    for _ in 2..=n {
        let next = a + b;
        a = b;
        b = next;
    }
    b
}

fn bench_trace_to_matrix(c: &mut Criterion) {
    let table_packing = TablePacking::new(1, 4);
    let lanes_p = table_packing.public_lanes();
    let lanes_a = table_packing.alu_lanes();

    let mut group = c.benchmark_group("trace_to_matrix");
    for n in [100, 500, 2000] {
        group.bench_with_input(BenchmarkId::new("fibonacci", n), &n, |b, &n| {
            b.iter(|| {
                let (circuit, expected_fib) = fib_circuit(n);
                let mut runner = circuit.runner();
                runner.set_public_inputs(&[expected_fib]).unwrap();
                let traces = runner.run().unwrap();
                let _ = black_box(ConstAir::<F, 1>::trace_to_matrix(&traces.const_trace, 1));
                let _ = black_box(PublicAir::<F, 1>::trace_to_matrix(
                    &traces.public_trace,
                    lanes_p,
                    1,
                ));
                let alu_air = AluAir::<F, 1>::new(traces.alu_trace.values.len(), lanes_a);
                let _ = black_box(alu_air.trace_to_matrix(&traces.alu_trace, 1));
            });
        });
    }
    group.finish();
}

fn bench_prove_all_tables(c: &mut Criterion) {
    let table_packing = TablePacking::new(1, 4);

    let mut group = c.benchmark_group("prove_all_tables");
    for n in [100, 500, 2000] {
        group.bench_with_input(BenchmarkId::new("fibonacci", n), &n, |b, &n| {
            b.iter(|| {
                let config = config::koala_bear().build();
                let (circuit, expected_fib) = fib_circuit(n);
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
                runner.set_public_inputs(&[expected_fib]).unwrap();
                let traces = runner.run().unwrap();
                let prover_data = ProverData::from_airs_and_degrees(&config, &mut airs, &degrees);
                let circuit_prover_data =
                    CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
                let prover =
                    BatchStarkProver::new(config).with_table_packing(table_packing.clone());
                black_box(
                    prover
                        .prove_all_tables(&traces, &circuit_prover_data)
                        .unwrap(),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_trace_to_matrix, bench_prove_all_tables);
criterion_main!(benches);
