use criterion::{Criterion, criterion_group, criterion_main};
use p3_field::extension::BinomialExtensionField;
use p3_field_testing::bench_func::{
    benchmark_inv, benchmark_mul_latency, benchmark_mul_throughput, benchmark_square,
};
use p3_field_testing::benchmark_mul;
use p3_goldilocks_monty::Goldilocks;

type EF2 = BinomialExtensionField<Goldilocks, 2>;
type EF5 = BinomialExtensionField<Goldilocks, 5>;

// Note that each round of throughput has 10 operations
// So we should have 10 * more repetitions for latency tests.
const REPS: usize = 50;
const L_REPS: usize = 10 * REPS;

fn bench_quadratic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<GoldilocksMonty, 2>";
    benchmark_square::<EF2>(c, name);
    benchmark_inv::<EF2>(c, name);
    benchmark_mul::<EF2>(c, name);
    benchmark_mul_throughput::<EF2, REPS>(c, name);
    benchmark_mul_latency::<EF2, L_REPS>(c, name);
}

fn bench_quintic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<GoldilocksMonty, 5>";
    benchmark_square::<EF5>(c, name);
    benchmark_inv::<EF5>(c, name);
    benchmark_mul::<EF5>(c, name);
    benchmark_mul_throughput::<EF5, REPS>(c, name);
    benchmark_mul_latency::<EF5, L_REPS>(c, name);
}

criterion_group!(
    bench_goldilocks_monty_ef,
    bench_quadratic_extension,
    bench_quintic_extension
);
criterion_main!(bench_goldilocks_monty_ef);
