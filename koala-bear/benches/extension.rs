use criterion::{Criterion, criterion_group, criterion_main};
use p3_field::extension::BinomialExtensionField;
use p3_field_testing::bench_func::{
    benchmark_add_latency, benchmark_add_slices, benchmark_add_throughput, benchmark_inv,
    benchmark_mul_latency, benchmark_mul_throughput, benchmark_square,
};
use p3_koala_bear::{KoalaBear, QuinticExtensionField};

type EF4 = BinomialExtensionField<KoalaBear, 4>;
type EF8 = BinomialExtensionField<KoalaBear, 8>;
type EF5 = QuinticExtensionField;

// Note that each round of throughput has 10 operations
// So we should have 10 * more repetitions for latency tests.
const REPS: usize = 100;
const L_REPS: usize = 10 * REPS;

fn bench_quartic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<KoalaBear, 4>";
    benchmark_add_throughput::<EF4, REPS>(c, name);
    benchmark_add_latency::<EF4, L_REPS>(c, name);
    benchmark_add_slices::<EF4, 8>(c, name);
    benchmark_add_slices::<EF4, 1000>(c, name);
    benchmark_square::<EF4>(c, name);
    benchmark_inv::<EF4>(c, name);
    benchmark_mul_throughput::<EF4, REPS>(c, name);
    benchmark_mul_latency::<EF4, L_REPS>(c, name);
}

fn bench_octic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<KoalaBear, 8>";
    benchmark_add_throughput::<EF8, REPS>(c, name);
    benchmark_add_latency::<EF8, L_REPS>(c, name);
    benchmark_add_slices::<EF8, 8>(c, name);
    benchmark_add_slices::<EF8, 1000>(c, name);
    benchmark_square::<EF8>(c, name);
    benchmark_inv::<EF8>(c, name);
    benchmark_mul_throughput::<EF8, REPS>(c, name);
    benchmark_mul_latency::<EF8, L_REPS>(c, name);
}

fn bench_quintic_extension(c: &mut Criterion) {
    let name = "QuinticExtensionField<KoalaBear>";
    benchmark_add_throughput::<EF5, REPS>(c, name);
    benchmark_add_latency::<EF5, L_REPS>(c, name);
    benchmark_add_slices::<EF5, 8>(c, name);
    benchmark_add_slices::<EF5, 1000>(c, name);
    benchmark_square::<EF5>(c, name);
    benchmark_inv::<EF5>(c, name);
    benchmark_mul_throughput::<EF5, REPS>(c, name);
    benchmark_mul_latency::<EF5, L_REPS>(c, name);
}

criterion_group!(
    bench_koalabear_ef,
    bench_quartic_extension,
    bench_octic_extension,
    bench_quintic_extension
);
criterion_main!(bench_koalabear_ef);
