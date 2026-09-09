use core::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_field::extension::{BinomialExtensionField, CubicTrinomialExtensionField, HasFrobenius};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_field_testing::bench_func::{
    benchmark_inv, benchmark_mul_latency, benchmark_mul_throughput, benchmark_square,
};
use p3_field_testing::benchmark_mul;
use p3_goldilocks::Goldilocks;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type EF2 = BinomialExtensionField<Goldilocks, 2>;
type EF3 = CubicTrinomialExtensionField<Goldilocks>;
type EF5 = BinomialExtensionField<Goldilocks, 5>;
type PEF2 = <EF2 as ExtensionField<Goldilocks>>::ExtensionPacking;
type PEF3 = <EF3 as ExtensionField<Goldilocks>>::ExtensionPacking;
type PEF5 = <EF5 as ExtensionField<Goldilocks>>::ExtensionPacking;

// Note that each round of throughput has 10 operations
// So we should have 10 * more repetitions for latency tests.
const REPS: usize = 50;
const L_REPS: usize = 10 * REPS;

fn bench_quadratic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<Goldilocks, 2>";
    benchmark_square::<EF2>(c, name);
    benchmark_inv::<EF2>(c, name);
    benchmark_mul::<EF2>(c, name);
    benchmark_mul_throughput::<EF2, REPS>(c, name);
    benchmark_mul_latency::<EF2, L_REPS>(c, name);

    let packed_name = "Packed BinomialExtensionField<Goldilocks, 2>";
    benchmark_mul_throughput::<PEF2, REPS>(c, packed_name);
    benchmark_mul_latency::<PEF2, L_REPS>(c, packed_name);
}

fn bench_packed_cubic_extension(c: &mut Criterion) {
    let name = "Packed CubicTrinomialExtensionField<Goldilocks>";
    benchmark_mul_throughput::<PEF3, REPS>(c, name);
    benchmark_mul_latency::<PEF3, L_REPS>(c, name);
}

fn bench_quintic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<Goldilocks, 5>";
    benchmark_square::<EF5>(c, name);
    benchmark_inv::<EF5>(c, name);
    benchmark_mul::<EF5>(c, name);
    benchmark_mul_throughput::<EF5, REPS>(c, name);
    benchmark_mul_latency::<EF5, L_REPS>(c, name);

    let packed_name = "Packed BinomialExtensionField<Goldilocks, 5>";
    benchmark_mul_throughput::<PEF5, REPS>(c, packed_name);
    benchmark_mul_latency::<PEF5, L_REPS>(c, packed_name);
}

fn bench_cubic_frobenius(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0x0F0B_31A5);
    let inputs: [EF3; 64] = core::array::from_fn(|_| rng.random());
    c.bench_function(
        "CubicTrinomialExtensionField<Goldilocks> frobenius (varying inputs)",
        |b| {
            b.iter(|| {
                let mut result = EF3::ZERO;
                for &input in &inputs {
                    result += black_box(input).frobenius();
                }
                black_box(result)
            });
        },
    );
}

fn bench_cubic_inverse(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0x01A2_B125);
    let inputs: [EF3; 64] = core::array::from_fn(|_| rng.random());
    c.bench_function(
        "CubicTrinomialExtensionField<Goldilocks> inverse (varying inputs)",
        |b| {
            b.iter(|| {
                let mut result = EF3::ZERO;
                for &input in &inputs {
                    result += black_box(input).inverse();
                }
                black_box(result)
            });
        },
    );
}

criterion_group!(
    bench_goldilocks_ef,
    bench_quadratic_extension,
    bench_packed_cubic_extension,
    bench_quintic_extension,
    bench_cubic_frobenius,
    bench_cubic_inverse
);
criterion_main!(bench_goldilocks_ef);
