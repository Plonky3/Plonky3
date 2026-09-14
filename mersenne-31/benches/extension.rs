use core::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::{BinomialExtensionField, Complex};
use p3_field_testing::bench_func::{
    benchmark_inv, benchmark_mul_latency, benchmark_mul_throughput, benchmark_square,
};
use p3_mersenne_31::{Mersenne31, PackedQM31};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type EF2 = BinomialExtensionField<Complex<Mersenne31>, 2>;
type EF3 = BinomialExtensionField<Complex<Mersenne31>, 3>;

const REPS: usize = 100;
const L_REPS: usize = 10 * REPS;

fn bench_quadratic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<Mersenne31Complex<Mersenne31>, 2>";
    benchmark_square::<EF2>(c, name);
    benchmark_inv::<EF2>(c, name);
    benchmark_mul_throughput::<EF2, REPS>(c, name);
    benchmark_mul_latency::<EF2, L_REPS>(c, name);
}

fn bench_cubic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<Mersenne31Complex<Mersenne31>, 3>";
    benchmark_square::<EF3>(c, name);
    benchmark_inv::<EF3>(c, name);
    benchmark_mul_throughput::<EF3, REPS>(c, name);
    benchmark_mul_latency::<EF3, L_REPS>(c, name);
}

fn bench_packed_qm31(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);
    let x: PackedQM31 = rng.random();

    c.bench_function("PackedQM31/square/dependent_64", |b| {
        b.iter(|| {
            let mut y = black_box(x);
            for _ in 0..64 {
                y = y.square();
            }
            black_box(y)
        });
    });
}

criterion_group!(
    bench_mersennecomplex_ef2,
    bench_quadratic_extension,
    bench_packed_qm31
);
criterion_group!(bench_mersennecomplex_ef3, bench_cubic_extension);

criterion_main!(bench_mersennecomplex_ef2, bench_mersennecomplex_ef3);
