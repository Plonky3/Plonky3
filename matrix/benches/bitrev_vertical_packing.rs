use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::{Field, PackedValue, Vectorized};
use p3_goldilocks::Goldilocks;
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

type F = BabyBear;
type Packed = <F as Field>::Packing;

const CONFIGS: &[(usize, usize)] = &[(10, 32), (14, 32), (18, 32)];

/// `(log_rows, width)` pairs swept row group by row group.
const SWEEP_CONFIGS: &[(usize, usize)] = &[(20, 8), (18, 32), (16, 128), (16, 512)];

/// Row offset of the second packed row, as in a quotient domain twice the trace size.
const NEXT_STEP: usize = 2;

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

/// Packs every row group of a bit-reversed matrix, together with the group `NEXT_STEP` rows
/// below it, into one reused buffer.
fn sweep_rows<V, P>(c: &mut Criterion, name: &str)
where
    V: Field,
    StandardUniform: Distribution<V>,
    P: PackedValue<Value = V>,
{
    let mut group = c.benchmark_group(format!("bitrev_vertically_packed_row_sweep/{name}"));
    group.sample_size(10);

    for &(log_rows, width) in SWEEP_CONFIGS {
        let rows = 1usize << log_rows;
        let mut rng = SmallRng::seed_from_u64(0);
        let matrix = RowMajorMatrix::<V>::rand(&mut rng, rows, width).bit_reverse_rows();
        let param = format!("2^{log_rows}x{width}");
        let mut buf = Vec::with_capacity(2 * width);

        group.bench_with_input(BenchmarkId::new("row", &param), &(), |b, _| {
            b.iter(|| {
                for r in (0..rows).step_by(P::WIDTH) {
                    buf.clear();
                    buf.extend(matrix.vertically_packed_row::<P>(r));
                    buf.extend(matrix.vertically_packed_row::<P>(r + NEXT_STEP));
                    black_box(&buf);
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("row_pair", &param), &(), |b, _| {
            b.iter(|| {
                for r in (0..rows).step_by(P::WIDTH) {
                    black_box(matrix.vertically_packed_row_pair::<P>(r, NEXT_STEP));
                }
            });
        });
    }
}

fn bitrev_vertically_packed_row_sweep(c: &mut Criterion) {
    sweep_rows::<BabyBear, <BabyBear as Field>::Packing>(c, "babybear");
    sweep_rows::<BabyBear, Vectorized<BabyBear, 2>>(c, "babybear_x2");
    sweep_rows::<Goldilocks, <Goldilocks as Field>::Packing>(c, "goldilocks");
}

criterion_group!(
    benches,
    bitrev_vertically_packed_row,
    bitrev_vertically_packed_row_sweep
);
criterion_main!(benches);
