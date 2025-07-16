use core::hint::black_box;
use criterion::{Criterion, criterion_group, criterion_main};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_field_testing::bench_func::{
    benchmark_inv, benchmark_mul_latency, benchmark_mul_throughput, benchmark_square,
};
use p3_field_testing::{benchmark_add_slices, benchmark_add_throughput, benchmark_mul};
use p3_koala_bear::KoalaBear;
use p3_monty_31::{octic_mul_packed, quartic_mul, quartic_mul_packed};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type EF4 = BinomialExtensionField<KoalaBear, 4>;
type EF8 = BinomialExtensionField<KoalaBear, 8>;

// Note that each round of throughput has 10 operations
// So we should have 10 * more repetitions for latency tests.
const REPS: usize = 100;
const L_REPS: usize = 10 * REPS;

fn bench_quartic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<KoalaBear, 4>";
    benchmark_mul::<EF4>(c, name);
    benchmark_quartic_mul_packed(c, name);
    // benchmark_add_throughput::<EF4, REPS>(c, name);
    // benchmark_add_slices::<EF4, 8>(c, name);
    // benchmark_add_slices::<EF4, 1000>(c, name);
    // benchmark_square::<EF4>(c, name);
    // benchmark_inv::<EF4>(c, name);
    benchmark_mul_throughput::<EF4, REPS>(c, name);
    benchmark_mul_latency::<EF4, L_REPS>(c, name);
}

fn bench_octic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<KoalaBear, 8>";
    benchmark_mul::<EF8>(c, name);
    benchmark_octic_mul_packed(c, name);
    //     benchmark_add_throughput::<EF8, REPS>(c, name);
    //     benchmark_add_slices::<EF8, 8>(c, name);
    //     benchmark_add_slices::<EF8, 1000>(c, name);
    //     benchmark_square::<EF8>(c, name);
    //     benchmark_inv::<EF8>(c, name);
    benchmark_mul_throughput::<EF8, REPS>(c, name);
    benchmark_mul_latency::<EF8, L_REPS>(c, name);
}

criterion_group!(
    bench_koalabear_ef,
    bench_quartic_extension,
    bench_octic_extension
);
criterion_main!(bench_koalabear_ef);

fn benchmark_quartic_mul_packed(c: &mut Criterion, name: &str) {
    let mut rng = SmallRng::seed_from_u64(1);
    let x: [KoalaBear; 4] = rng.random();
    let y: [KoalaBear; 4] = rng.random();
    let w = KoalaBear::TWO + KoalaBear::ONE;
    c.bench_function(&format!("{name} mul packed"), |b| {
        b.iter(|| {
            let mut res = [KoalaBear::ZERO; 4];
            quartic_mul_packed(black_box(&x), black_box(&y), &mut res, black_box(w));
            black_box(res)
        })
    });
}

fn benchmark_octic_mul_packed(c: &mut Criterion, name: &str) {
    let mut rng = SmallRng::seed_from_u64(1);
    let x: [KoalaBear; 8] = rng.random();
    let y: [KoalaBear; 8] = rng.random();
    let w = KoalaBear::TWO + KoalaBear::ONE;
    c.bench_function(&format!("{name} mul packed"), |b| {
        b.iter(|| {
            let mut res = [KoalaBear::ZERO; 8];
            octic_mul_packed(black_box(&x), black_box(&y), &mut res, black_box(w));
            black_box(res)
        })
    });
}
