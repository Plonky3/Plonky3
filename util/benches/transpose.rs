use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
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
        let mut output_neon = vec![BabyBear::ZERO; size];
        let mut output_crate = vec![BabyBear::ZERO; size];

        group.bench_with_input(BenchmarkId::new("transpose_util", name), &size, |b, _| {
            b.iter(|| {
                transpose(black_box(&input), &mut output_neon, width, height);
                black_box(output_neon[0])
            });
        });

        group.bench_with_input(BenchmarkId::new("transpose_crate", name), &size, |b, _| {
            b.iter(|| {
                transpose::transpose(black_box(&input), &mut output_crate, width, height);
                black_box(output_crate[0])
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_transpose_babybear);

criterion_main!(benches);
