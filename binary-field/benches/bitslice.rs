//! Bit-sliced `GF(2)`: the packed operations against the one-element-per-byte scalar path,
//! and the blocked bit transpose against the lane-by-lane reference.
//!
//! Throughput is reported in field elements per second for the arithmetic and in bits per
//! second for the transpose, so the packed and scalar rows are directly comparable.
//!
//! The packings are plain word arrays, so the vector width the arithmetic reaches is the one
//! the compiler is allowed to use. Rerunning with `RUSTFLAGS="-C target-cpu=native"` measures
//! the same code with the host's widest registers unlocked.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_binary_field::{
    Gf2, PackedGf2x8, PackedGf2x16, PackedGf2x32, PackedGf2x64, PackedGf2x128, PackedGf2x256,
    PackedGf2x512,
};
use p3_field::PrimeCharacteristicRing;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Field elements touched by one iteration of an arithmetic benchmark.
///
/// Large enough that loop overhead disappears, small enough that both operand buffers stay
/// in L2 even in the scalar layout, where one element costs a whole byte.
const ELEMENTS: usize = 1 << 16;

/// Square bit matrices transposed by one iteration of a transpose benchmark.
///
/// Fixed rather than scaled with the side, so the larger sides do proportionally more work
/// and the reported bit rate stays the comparable quantity.
const MATRICES: usize = 64;

/// Exclusive or and conjunction over a buffer of one element per byte.
///
/// This is what the proving stack does today without a packing: one whole scalar per bit.
fn bench_scalar(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);
    let left: Vec<Gf2> = (0..ELEMENTS).map(|_| rng.random()).collect();
    let right: Vec<Gf2> = (0..ELEMENTS).map(|_| rng.random()).collect();
    let mut out = vec![Gf2::ZERO; ELEMENTS];

    let mut group = c.benchmark_group("bitslice/scalar");
    group.throughput(Throughput::Elements(ELEMENTS as u64));

    group.bench_function("add", |b| {
        b.iter(|| {
            // One byte read, one byte read, one byte written, per single bit of result.
            for i in 0..ELEMENTS {
                out[i] = black_box(left[i]) + black_box(right[i]);
            }
            black_box(&out);
        });
    });

    group.bench_function("mul", |b| {
        b.iter(|| {
            for i in 0..ELEMENTS {
                out[i] = black_box(left[i]) * black_box(right[i]);
            }
            black_box(&out);
        });
    });

    group.finish();
}

/// Exclusive or and conjunction over a buffer of one element per bit, at every width.
macro_rules! bench_packed {
    ($group:expr, $name:ident, $width:literal) => {{
        let mut rng = SmallRng::seed_from_u64(1);
        let count = ELEMENTS / $width;
        let left: Vec<$name> = (0..count).map(|_| rng.random()).collect();
        let right: Vec<$name> = (0..count).map(|_| rng.random()).collect();
        let mut out = vec![$name::ZERO; count];

        $group.bench_with_input(BenchmarkId::new("add", $width), &$width, |b, _| {
            b.iter(|| {
                // One word read, one word read, one word written, per `B` bits of result.
                for i in 0..count {
                    out[i] = black_box(left[i]) + black_box(right[i]);
                }
                black_box(&out);
            });
        });

        $group.bench_with_input(BenchmarkId::new("mul", $width), &$width, |b, _| {
            b.iter(|| {
                for i in 0..count {
                    out[i] = black_box(left[i]) * black_box(right[i]);
                }
                black_box(&out);
            });
        });
    }};
}

/// The packed arithmetic, one benchmark pair per width.
fn bench_packed(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitslice/packed");
    group.throughput(Throughput::Elements(ELEMENTS as u64));

    bench_packed!(group, PackedGf2x8, 8);
    bench_packed!(group, PackedGf2x16, 16);
    bench_packed!(group, PackedGf2x32, 32);
    bench_packed!(group, PackedGf2x64, 64);
    bench_packed!(group, PackedGf2x128, 128);
    bench_packed!(group, PackedGf2x256, 256);
    bench_packed!(group, PackedGf2x512, 512);

    group.finish();
}

/// The blocked transpose against the lane-by-lane reference, at every width.
///
/// The reference reads one bit at a time through the lane accessors, which is the only other
/// way to get the same answer and therefore the honest baseline for the blocked kernel.
macro_rules! bench_transpose {
    ($group:expr, $name:ident, $width:literal) => {{
        let mut rng = SmallRng::seed_from_u64(1);
        let original: Vec<[$name; $width]> = (0..MATRICES)
            .map(|_| core::array::from_fn(|_| rng.random()))
            .collect();

        $group.throughput(Throughput::Bytes((MATRICES * $width * $width / 8) as u64));

        $group.bench_with_input(BenchmarkId::new("blocked", $width), &$width, |b, _| {
            let mut work = original.clone();
            b.iter(|| {
                for matrix in &mut work {
                    $name::transpose(black_box(&mut matrix[..]));
                }
                black_box(&work);
            });
        });

        $group.bench_with_input(BenchmarkId::new("lane_by_lane", $width), &$width, |b, _| {
            let mut work = original.clone();
            b.iter(|| {
                for matrix in &mut work {
                    // Row `r` of the result is column `r` of the input, one lane at a time.
                    let rows = black_box(&*matrix);
                    let transposed: [$name; $width] =
                        core::array::from_fn(|r| $name::from_fn(|c| rows[c].get(r)));
                    *matrix = transposed;
                }
                black_box(&work);
            });
        });
    }};
}

/// The bit transpose, one benchmark pair per side.
fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitslice/transpose");

    bench_transpose!(group, PackedGf2x8, 8);
    bench_transpose!(group, PackedGf2x64, 64);
    bench_transpose!(group, PackedGf2x128, 128);
    bench_transpose!(group, PackedGf2x256, 256);
    bench_transpose!(group, PackedGf2x512, 512);

    group.finish();
}

/// Re-reading a committed bit witness at a narrower packing width.
///
/// The view is a reinterpretation of the same bytes, so the comparison is against the only
/// alternative that does not need one: copying the lanes across.
fn bench_narrow(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);

    // A witness of a megabit, held at the widest packing.
    let witness: Vec<PackedGf2x512> = (0..(1 << 20) / 512).map(|_| rng.random()).collect();

    let mut group = c.benchmark_group("bitslice/narrow");
    group.throughput(Throughput::Bytes(witness.len() as u64 * 64));

    group.bench_function("view", |b| {
        b.iter(|| {
            // No bits move: the narrow slice borrows the same bytes.
            let narrow: &[PackedGf2x64] = PackedGf2x512::narrow_slice(black_box(&witness));
            narrow.iter().map(PackedGf2x64::count_ones).sum::<u32>()
        });
    });

    group.bench_function("copy", |b| {
        b.iter(|| {
            // The same reading, rebuilt lane by lane instead of borrowed.
            let narrow: Vec<PackedGf2x64> = black_box(&witness)
                .iter()
                .flat_map(|wide| {
                    (0..8).map(move |k| PackedGf2x64::from_fn(|i| wide.get(k * 64 + i)))
                })
                .collect();
            narrow.iter().map(PackedGf2x64::count_ones).sum::<u32>()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_scalar,
    bench_packed,
    bench_narrow,
    bench_transpose
);
criterion_main!(benches);
