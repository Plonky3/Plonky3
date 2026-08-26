//! Tower arithmetic, comparing the recursive reference routines against whatever `Mul` and
//! `square` dispatch to on the host.
//!
//! The two are the same routine below `GF(2^64)`; at 64 and 128 bits both operators take the
//! carryless-multiply fast path where the target has the instruction for it. Rerunning with
//! `RUSTFLAGS="-C target-feature=-aes"` (AArch64) or without `+pclmulqdq` (x86-64) measures the
//! same code with the fast path turned off.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_binary_field::{BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128};
use p3_field::{Field, PrimeCharacteristicRing};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Multiplications performed by a single iteration of every benchmark here.
const REPS: usize = 1000;

/// Independent chains a throughput benchmark interleaves.
const LANES: usize = 10;

/// Latency and throughput of one level, for both multiplication routines.
///
/// The latency benchmark folds a dependent chain, as `p3_field_testing::bench_func` does, so
/// each product waits on the previous one. The throughput benchmark runs [`LANES`] such chains
/// side by side over the same number of products, leaving the CPU free to overlap them.
macro_rules! bench_level {
    ($c:expr, $t:ty, $bits:literal) => {{
        let mut rng = SmallRng::seed_from_u64(1);
        let operands: Vec<$t> = (0..REPS).map(|_| rng.random::<$t>()).collect();

        let mut group = $c.benchmark_group(concat!("mul/", $bits));

        group.bench_function("reference/latency", |b| {
            b.iter(|| {
                black_box(&operands)
                    .iter()
                    .fold(<$t>::ONE, |acc, &y| acc.reference_mul(y))
            });
        });
        group.bench_function("dispatched/latency", |b| {
            b.iter(|| {
                black_box(&operands)
                    .iter()
                    .fold(<$t>::ONE, |acc, &y| acc * y)
            });
        });

        group.bench_function("reference/throughput", |b| {
            b.iter(|| {
                let mut acc = [<$t>::ONE; LANES];
                for chunk in black_box(&operands).chunks_exact(LANES) {
                    for (lane, &y) in acc.iter_mut().zip(chunk) {
                        *lane = lane.reference_mul(y);
                    }
                }
                acc
            });
        });
        group.bench_function("dispatched/throughput", |b| {
            b.iter(|| {
                let mut acc = [<$t>::ONE; LANES];
                for chunk in black_box(&operands).chunks_exact(LANES) {
                    for (lane, &y) in acc.iter_mut().zip(chunk) {
                        *lane *= y;
                    }
                }
                acc
            });
        });

        group.finish();
    }};
}

/// Latency and throughput of squaring one level, against the multiplication that computes the
/// same value.
///
/// Each step mixes the next operand into the accumulator before squaring, so the chain cannot
/// be folded away at compile time and both benchmarks pay the same `XOR`.
macro_rules! bench_square_level {
    ($c:expr, $t:ty, $bits:literal) => {{
        let mut rng = SmallRng::seed_from_u64(1);
        let operands: Vec<$t> = (0..REPS).map(|_| rng.random::<$t>()).collect();

        let mut group = $c.benchmark_group(concat!("square/", $bits));

        group.bench_function("square/latency", |b| {
            b.iter(|| {
                black_box(&operands)
                    .iter()
                    .fold(<$t>::ONE, |acc, &y| (acc + y).square())
            });
        });
        group.bench_function("mul/latency", |b| {
            b.iter(|| {
                black_box(&operands).iter().fold(<$t>::ONE, |acc, &y| {
                    let x = acc + y;
                    x * x
                })
            });
        });

        group.bench_function("square/throughput", |b| {
            b.iter(|| {
                let mut acc = [<$t>::ONE; LANES];
                for chunk in black_box(&operands).chunks_exact(LANES) {
                    for (lane, &y) in acc.iter_mut().zip(chunk) {
                        *lane = (*lane + y).square();
                    }
                }
                acc
            });
        });
        group.bench_function("mul/throughput", |b| {
            b.iter(|| {
                let mut acc = [<$t>::ONE; LANES];
                for chunk in black_box(&operands).chunks_exact(LANES) {
                    for (lane, &y) in acc.iter_mut().zip(chunk) {
                        let x = *lane + y;
                        *lane = x * x;
                    }
                }
                acc
            });
        });

        group.finish();
    }};
}

fn bench_mul(c: &mut Criterion) {
    bench_level!(c, BinaryField8, 8);
    bench_level!(c, BinaryField16, 16);
    bench_level!(c, BinaryField32, 32);
    bench_level!(c, BinaryField64, 64);
    bench_level!(c, BinaryField128, 128);
}

fn bench_square(c: &mut Criterion) {
    bench_square_level!(c, BinaryField64, 64);
    bench_square_level!(c, BinaryField128, 128);
}

fn bench_inverse(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);
    // Inversion recurses through the level below's `Mul`, so it picks up the fast path too.
    let operands: Vec<BinaryField128> = (0..REPS)
        .map(|_| {
            let x = rng.random::<BinaryField128>();
            if x.is_zero() { BinaryField128::ONE } else { x }
        })
        .collect();

    let mut group = c.benchmark_group("inverse/128");
    group.bench_function("dispatched", |b| {
        b.iter(|| {
            black_box(&operands)
                .iter()
                // Addition is `XOR`, so the fold barely adds to the inversions it accumulates.
                .fold(BinaryField128::ZERO, |acc, &y| acc + y.inverse())
        });
    });
    group.finish();
}

fn bench_mul_alpha(c: &mut Criterion) {
    // `α = X_5`, the generator of `GF(2^64)` over `GF(2^32)`, sits at bit 32 of the tower
    // representation. `TowerLevel::mul_alpha` is crate-internal, so this measures the product
    // by that element through the public operator instead.
    let mut alpha_bytes = [0u8; 16];
    alpha_bytes[4] = 1;
    let alpha = BinaryField128::from_le_bytes(alpha_bytes);

    let mut rng = SmallRng::seed_from_u64(1);
    let operands: Vec<BinaryField128> = (0..REPS).map(|_| rng.random()).collect();

    let mut group = c.benchmark_group("mul_alpha/128");
    group.bench_function("reference", |b| {
        b.iter(|| {
            black_box(&operands)
                .iter()
                .fold(BinaryField128::ZERO, |acc, &y| acc + y.reference_mul(alpha))
        });
    });
    group.bench_function("dispatched", |b| {
        b.iter(|| {
            black_box(&operands)
                .iter()
                .fold(BinaryField128::ZERO, |acc, &y| acc + y * alpha)
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_mul,
    bench_square,
    bench_inverse,
    bench_mul_alpha
);
criterion_main!(benches);
