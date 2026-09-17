use core::hint::black_box;
use core::mem::size_of;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::Field;
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::util::reverse_matrix_index_bits;
use p3_maybe_rayon::PARALLEL_ENABLED;
use p3_maybe_rayon::prelude::current_num_threads;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

fn bench_field<F: Field>(c: &mut Criterion, field_name: &str)
where
    StandardUniform: Distribution<F>,
{
    let backend = if PARALLEL_ENABLED {
        "parallel"
    } else {
        "serial"
    };
    let mut group = c.benchmark_group(format!(
        "reverse_matrix_index_bits/{field_name}/{backend}/threads={}",
        current_num_threads()
    ));
    group.sample_size(20);

    let wide_row = 64 * 1024 / size_of::<F>();
    for (log_rows, width) in [
        (10, 1),
        (14, 1),
        (17, 1),
        (10, 3),
        (13, 3),
        (17, 3),
        (5, 128),
        (6, 128),
        (7, 128),
        (14, 128),
        (17, 128),
        (3, wide_row - 1),
        (3, wide_row),
        (3, wide_row + 1),
    ] {
        let rows = 1 << log_rows;
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{rows}x{width}")),
            &(rows, width),
            |b, &(rows, width)| {
                // Defer allocation until Criterion selects this case, outside the timer.
                let mut rng = SmallRng::seed_from_u64(1);
                let mut matrix = RowMajorMatrix::<F>::rand(&mut rng, rows, width);
                b.iter(|| reverse_matrix_index_bits(black_box(&mut matrix)));
            },
        );
    }
    group.finish();
}

fn bench_reverse_matrix_index_bits(c: &mut Criterion) {
    bench_field::<BabyBear>(c, "BabyBear");
    bench_field::<Goldilocks>(c, "Goldilocks");
}

criterion_group!(benches, bench_reverse_matrix_index_bits);
criterion_main!(benches);
