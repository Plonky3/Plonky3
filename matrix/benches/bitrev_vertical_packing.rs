use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type Packed = <F as Field>::Packing;

const CONFIGS: &[(usize, usize)] = &[(10, 32), (14, 32), (18, 32)];

fn bitrev_vertically_packed_row(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitrev_vertically_packed_row");
    group.sample_size(20);

    for &(log_rows, width) in CONFIGS {
        let rows = 1usize << log_rows;
        let mut rng = SmallRng::seed_from_u64(0);
        let matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, rows, width).bit_reverse_rows();
        let start_row = rows / 2;
        let param = format!("2^{log_rows}x{width}");

        group.bench_with_input(BenchmarkId::new("row", &param), &(), |b, _| {
            b.iter(|| {
                black_box(
                    matrix
                        .vertically_packed_row::<Packed>(start_row)
                        .collect::<Vec<_>>(),
                )
            });
        });
    }
}

criterion_group!(benches, bitrev_vertically_packed_row);
criterion_main!(benches);
