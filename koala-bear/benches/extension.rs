use criterion::{Criterion, criterion_group, criterion_main};
use p3_field::extension::BinomialExtensionField;
use p3_field_testing::bench_func::{
    benchmark_inv, benchmark_mul_latency, benchmark_mul_throughput, benchmark_square,
};
use p3_field_testing::{benchmark_add_slices, benchmark_add_throughput};
use p3_koala_bear::KoalaBear;

type EF4 = BinomialExtensionField<KoalaBear, 4>;

// Note that each round of throughput has 10 operations
// So we should have 10 * more repetitions for latency tests.
const REPS: usize = 100;
const L_REPS: usize = 10 * REPS;

fn bench_quartic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<KoalaBear, 4>";
    benchmark_add_throughput::<EF4, REPS>(c, name);
    benchmark_add_slices::<EF4, 8>(c, name);
    benchmark_add_slices::<EF4, 1000>(c, name);
    benchmark_square::<EF4>(c, name);
    benchmark_inv::<EF4>(c, name);
    benchmark_mul_throughput::<EF4, REPS>(c, name);
    benchmark_mul_latency::<EF4, L_REPS>(c, name);
}

criterion_group!(bench_babybear_ef, bench_quartic_extension,);
criterion_main!(bench_babybear_ef);
