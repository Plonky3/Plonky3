//! Benchmarks: circuit runner execution + trace building.
//!
//! Includes micro-benchmarks: execute_all (evaluation only), per-table trace
//! building, and full run. Parameterized by instance size and shape.

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_circuit::CircuitBuilder;
use p3_circuit::tables::{ConstTraceBuilder, PublicTraceBuilder};
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

/// Circuit with N Const ops only (stresses set_witness / Const path).
fn many_const_circuit(n: usize) -> (p3_circuit::Circuit<F>, F) {
    let mut builder = CircuitBuilder::new();
    let pub_in = builder.alloc_public_input("p");
    let mut prev = builder.alloc_const(F::ZERO, "c0");
    for i in 1..n {
        prev = builder.alloc_const(F::from_u64(i as u64), "c");
    }
    builder.connect(prev, pub_in);
    let circuit = builder.build().unwrap();
    let public_val = F::from_u64((n - 1) as u64);
    (circuit, public_val)
}

/// Mul-heavy: out = c^N (N muls). Stresses get_witness + set_witness on Mul path.
fn mul_heavy_circuit(n: usize) -> (p3_circuit::Circuit<F>, F) {
    let mut builder = CircuitBuilder::new();
    let expected_result = builder.alloc_public_input("expected_result");
    let c = builder.alloc_const(F::from_u64(3), "c");
    let mut acc = builder.alloc_const(F::ONE, "acc0");
    for _ in 0..n {
        acc = builder.mul(acc, c);
    }
    builder.connect(acc, expected_result);
    let circuit = builder.build().unwrap();
    let mut expected = F::ONE;
    for _ in 0..n {
        expected *= F::from_u64(3);
    }
    (circuit, expected)
}

fn bench_runner_traces(c: &mut Criterion) {
    let mut group = c.benchmark_group("runner_traces");
    for n in [100, 500, 2000, 10_000] {
        group.bench_with_input(BenchmarkId::new("fibonacci", n), &n, |b, &n| {
            b.iter(|| {
                let (circuit, expected_fib) = fib_circuit(n);
                let mut runner = circuit.runner();
                runner.set_public_inputs(&[expected_fib]).unwrap();
                black_box(runner.run().unwrap())
            });
        });
    }
    group.finish();
}

fn bench_execute_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("execute_all");
    for n in [100, 500, 2000, 10_000] {
        group.bench_with_input(BenchmarkId::new("fibonacci", n), &n, |b, &n| {
            b.iter(|| {
                let (circuit, expected_fib) = fib_circuit(n);
                let mut runner = circuit.runner();
                runner.set_public_inputs(&[expected_fib]).unwrap();
                runner.execute_all().unwrap();
                black_box(());
            });
        });
    }
    for n in [1_000, 10_000, 50_000] {
        group.bench_with_input(BenchmarkId::new("many_const", n), &n, |b, &n| {
            b.iter(|| {
                let (circuit, public_val) = many_const_circuit(n);
                let mut runner = circuit.runner();
                runner.set_public_inputs(&[public_val]).unwrap();
                runner.execute_all().unwrap();
                black_box(());
            });
        });
    }
    for n in [100, 500, 2000] {
        group.bench_with_input(BenchmarkId::new("mul_heavy", n), &n, |b, &n| {
            b.iter(|| {
                let (circuit, expected_val) = mul_heavy_circuit(n);
                let mut runner = circuit.runner();
                runner.set_public_inputs(&[expected_val]).unwrap();
                runner.execute_all().unwrap();
                black_box(());
            });
        });
    }
    group.finish();
}

fn bench_trace_build(c: &mut Criterion) {
    const N: usize = 2000;
    let (circuit, expected_fib) = fib_circuit(N);
    let mut runner = circuit.runner();
    runner.set_public_inputs(&[expected_fib]).unwrap();
    runner.execute_all().unwrap();
    let witness = runner.witness();
    let ops = runner.ops();

    let mut group = c.benchmark_group("trace_build");
    group.bench_function("const", |b| {
        b.iter(|| black_box(ConstTraceBuilder::new(ops).build().unwrap()));
    });
    group.bench_function("public", |b| {
        b.iter(|| black_box(PublicTraceBuilder::new(ops, witness).build().unwrap()));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_runner_traces,
    bench_execute_all,
    bench_trace_build,
);
criterion_main!(benches);
