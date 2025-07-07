use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_util::{reverse_bits, reverse_slice_index_bits};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn bench_reverse_slice_index_bits(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_slice_index_bits");
    let mut rng = SmallRng::seed_from_u64(1);
    for log_size in [1, 3, 5, 8, 16, 24, 26] {
        let size = 1 << log_size;
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let data: Vec<u64> = (0..size).map(|_| rng.random()).collect();
            b.iter(|| {
                let mut test_data = data.clone();
                reverse_slice_index_bits(black_box(&mut test_data));
                black_box(test_data)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_reverse_slice_index_bits);
criterion_main!(benches);
