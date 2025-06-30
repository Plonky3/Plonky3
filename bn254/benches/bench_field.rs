use criterion::{Criterion, criterion_group, criterion_main};
use p3_bn254::Bn254;
use p3_field_testing::bench_func::{
    benchmark_add_latency, benchmark_add_throughput, benchmark_inv, benchmark_sub_latency,
    benchmark_sub_throughput,
};
use p3_field_testing::{benchmark_halve, benchmark_mul_latency, benchmark_mul_throughput};

type F = Bn254;

fn bench_field(c: &mut Criterion) {
    let name = "BN254Fr";
    const REPS: usize = 100;
    benchmark_halve::<F, REPS>(c, name);
    benchmark_inv::<F>(c, name);

    // Note that each round of throughput has 10 operations
    // So we should have 10 * more repetitions for latency tests.
    const L_REPS: usize = 10 * REPS;
    benchmark_add_latency::<F, L_REPS>(c, name);
    benchmark_add_throughput::<F, REPS>(c, name);
    benchmark_sub_latency::<F, L_REPS>(c, name);
    benchmark_sub_throughput::<F, REPS>(c, name);
    benchmark_mul_latency::<F, L_REPS>(c, name);
    benchmark_mul_throughput::<F, REPS>(c, name);
}

criterion_group!(bn254fr_arithmetic, bench_field);
criterion_main!(bn254fr_arithmetic);
