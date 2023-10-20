use criterion::{criterion_group, criterion_main, Criterion};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field_testing::bench_func::{benchmark_inv, benchmark_mul, benchmark_square};

type EF4 = BinomialExtensionField<BabyBear, 4>;
type EF5 = BinomialExtensionField<BabyBear, 5>;

fn bench_quartic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<BabyBear, 4>";
    benchmark_square::<EF4>(c, name);
    benchmark_inv::<EF4>(c, name);
    benchmark_mul::<EF4>(c, name);
}

fn bench_qunitic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<BabyBear, 5>";
    benchmark_square::<EF5>(c, name);
    benchmark_inv::<EF5>(c, name);
    benchmark_mul::<EF5>(c, name);
}

criterion_group!(
    bench_babybear_ef,
    bench_quartic_extension,
    bench_qunitic_extension
);
criterion_main!(bench_babybear_ef);
