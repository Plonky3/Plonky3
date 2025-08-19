use criterion::{Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field_testing::bench_func::{
    benchmark_add_latency, benchmark_add_throughput, benchmark_inv, benchmark_mul_latency,
    benchmark_mul_throughput, benchmark_square,
};
use p3_field_testing::{
    benchmark_base_mul_latency, benchmark_base_mul_throughput, benchmark_sub_latency,
    benchmark_sub_throughput,
};

type F = BabyBear;
type EF4 = BinomialExtensionField<BabyBear, 4>;
type EF5 = BinomialExtensionField<BabyBear, 5>;
type EF8 = BinomialExtensionField<BabyBear, 8>;

// Note that each round of throughput has 10 operations
// So we should have 10 * more repetitions for latency tests.
const REPS: usize = 100;
const L_REPS: usize = 10 * REPS;

fn bench_quartic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<BabyBear, 4>";
    benchmark_add_throughput::<EF4, REPS>(c, name);
    benchmark_add_latency::<EF4, L_REPS>(c, name);
    benchmark_sub_throughput::<EF4, REPS>(c, name);
    benchmark_sub_latency::<EF4, L_REPS>(c, name);
    benchmark_base_mul_throughput::<F, EF4, REPS>(c, name);
    benchmark_base_mul_latency::<F, EF4, L_REPS>(c, name);
    benchmark_square::<EF4>(c, name);
    benchmark_inv::<EF4>(c, name);
    benchmark_mul_throughput::<EF4, REPS>(c, name);
    benchmark_mul_latency::<EF4, L_REPS>(c, name);
}

fn bench_qunitic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<BabyBear, 5>";
    benchmark_add_throughput::<EF5, REPS>(c, name);
    benchmark_add_latency::<EF5, L_REPS>(c, name);
    benchmark_sub_throughput::<EF5, REPS>(c, name);
    benchmark_sub_latency::<EF5, L_REPS>(c, name);
    benchmark_base_mul_throughput::<F, EF5, REPS>(c, name);
    benchmark_base_mul_latency::<F, EF5, L_REPS>(c, name);
    benchmark_square::<EF5>(c, name);
    benchmark_inv::<EF5>(c, name);
    benchmark_mul_throughput::<EF5, REPS>(c, name);
    benchmark_mul_latency::<EF5, L_REPS>(c, name);
}

fn bench_octic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<BabyBear, 8>";
    benchmark_add_throughput::<EF8, REPS>(c, name);
    benchmark_add_latency::<EF8, L_REPS>(c, name);
    benchmark_sub_throughput::<EF8, REPS>(c, name);
    benchmark_sub_latency::<EF8, L_REPS>(c, name);
    benchmark_base_mul_throughput::<F, EF8, REPS>(c, name);
    benchmark_base_mul_latency::<F, EF8, L_REPS>(c, name);
    benchmark_square::<EF8>(c, name);
    benchmark_inv::<EF8>(c, name);
    benchmark_mul_throughput::<EF8, REPS>(c, name);
    benchmark_mul_latency::<EF8, L_REPS>(c, name);
}

criterion_group!(
    bench_babybear_ef,
    bench_quartic_extension,
    bench_qunitic_extension,
    bench_octic_extension
);
criterion_main!(bench_babybear_ef);
