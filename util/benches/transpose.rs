use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::{Packable, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks;
use p3_util::transpose::transpose;

/// Benchmark matrix sizes for BabyBear field elements (4 bytes each):
const BENCHMARK_SIZES: &[(usize, usize, &str)] = &[
    (128, 128, "small_square_128x128"),
    (1024, 1024, "medium_square_1024x1024"),
    (4096, 2048, "large_wide_4096x2048"),
    (2048, 4096, "large_tall_2048x4096"),
];

fn bench_transpose_babybear(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose_babybear");

    for &(width, height, name) in BENCHMARK_SIZES {
        let size = width * height;

        let input: Vec<_> = (0..size as u64).map(BabyBear::from_u64).collect();
        let mut output = BabyBear::zero_vec(size);

        group.bench_with_input(BenchmarkId::new("transpose_util", name), &size, |b, _| {
            b.iter(|| {
                BabyBear::transpose(black_box(&input), &mut output, width, height);
                black_box(output[0])
            });
        });
    }

    group.finish();
}

fn bench_transpose_goldilocks(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose_goldilocks");

    for &(width, height, name) in BENCHMARK_SIZES {
        let size = width * height;

        let input: Vec<_> = (0..size as u64).map(Goldilocks::from_u64).collect();
        let mut output = Goldilocks::zero_vec(size);

        group.bench_with_input(BenchmarkId::new("transpose_util", name), &size, |b, _| {
            b.iter(|| {
                Goldilocks::transpose(black_box(&input), &mut output, width, height);
                black_box(output[0])
            });
        });
    }

    group.finish();
}

// Drive the portable engine through the typed public entry point. Byte arrays
// also cover non-word element sizes. Buffers are allocated outside b.iter.
fn bench_transpose_portable<const BYTES: usize>(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose_portable");
    for width in [1, 2, 4, 8, 16, 32, 256, 4096] {
        for delta in [-1isize, 0, 1] {
            let height = ((1usize << 22) / width).checked_add_signed(delta).unwrap();
            let size = width * height;
            let input: Vec<[u8; BYTES]> = (0..size).map(|i| [i as u8; BYTES]).collect();
            let mut output = vec![[0u8; BYTES]; size];
            let name = format!("{BYTES}b_w{width}_h{height}");
            group.bench_function(name, |b| {
                b.iter(|| {
                    transpose(black_box(&input), black_box(&mut output), width, height);
                    black_box(output[0])
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_transpose_babybear,
    bench_transpose_goldilocks,
    bench_transpose_portable::<3>,
    bench_transpose_portable::<4>,
    bench_transpose_portable::<8>,
    bench_transpose_portable::<16>,
    bench_transpose_portable::<32>
);

criterion_main!(benches);
