use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use p3_field::{Packable, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;

fn transpose_benchmark(c: &mut Criterion) {
    const SMALL_DIMS: [(usize, usize); 4] = [(4, 4), (8, 8), (10, 10), (12, 12)];
    const LARGE_DIMS: [(usize, usize); 4] = [(20, 8), (21, 8), (22, 8), (23, 8)];

    let inner = |g: &mut BenchmarkGroup<'_, _>, dims: &[(usize, usize)]| {
        let mut rng = SmallRng::seed_from_u64(1);
        for (lg_nrows, lg_ncols) in dims {
            let nrows = 1 << lg_nrows;
            let ncols = 1 << lg_ncols;
            let mut matrix1 = RowMajorMatrix::<u32>::rand(&mut rng, nrows, ncols);
            let mut matrix2 = RowMajorMatrix::default(nrows, ncols);

            let name = format!("2^{lg_nrows} x 2^{lg_ncols}");
            g.throughput(Throughput::Bytes(
                (nrows * ncols * core::mem::size_of::<u32>()) as u64,
            ));
            g.bench_function(&name, |b| {
                b.iter(|| matrix1.transpose_packable_into(&mut matrix2));
            });

            if nrows != ncols {
                let matrix2 = RowMajorMatrix::rand(&mut rng, ncols, nrows);
                let name = format!("2^{lg_ncols} x 2^{lg_nrows}");
                g.throughput(Throughput::Bytes(
                    (nrows * ncols * core::mem::size_of::<u32>()) as u64,
                ));
                g.bench_function(&name, |b| {
                    b.iter(|| matrix2.transpose_packable_into(&mut matrix1));
                });
            }
        }
    };

    let mut g = c.benchmark_group("transpose");
    inner(&mut g, &SMALL_DIMS);
    g.sample_size(10);
    inner(&mut g, &LARGE_DIMS);
}

/// Compare allocation and destruction on identical precreated inputs. The
/// initialized baseline retains the existing dispatch and default-fill pass.
fn allocating_transpose_benchmark(c: &mut Criterion) {
    fn bench<T: Packable + core::fmt::Debug>(
        c: &mut Criterion,
        name: &str,
        make: impl Fn(usize) -> T,
    ) {
        for packable in [false, true] {
            let api = if packable { "packable" } else { "generic" };
            let mut group = c.benchmark_group(format!("transpose_allocating/{name}/{api}"));
            group.sample_size(20);
            group.warm_up_time(Duration::from_secs(1));
            group.measurement_time(Duration::from_secs(3));
            for (width, height) in [
                (256, 256),
                (1024, 1024),
                (4096, 4096),
                (17, 4096),
                (4096, 17),
                (4, 262144),
                (262144, 4),
            ] {
                let input = RowMajorMatrix::new((0..width * height).map(&make).collect(), width);
                group.throughput(Throughput::Bytes((width * height * size_of::<T>()) as u64));
                let shape = format!("{width}x{height}");
                group.bench_function(format!("initialized_owned/{shape}"), |b| {
                    b.iter(|| {
                        let input = black_box(&input);
                        let mut output =
                            RowMajorMatrix::new(vec![T::default(); width * height], height);
                        if packable {
                            input.transpose_packable_into(&mut output);
                        } else {
                            input.transpose_into(&mut output);
                        }
                        black_box(output)
                    });
                });
                group.bench_function(format!("owned/{shape}"), |b| {
                    b.iter(|| {
                        let input = black_box(&input);
                        black_box(if packable {
                            input.transpose_packable()
                        } else {
                            input.transpose()
                        })
                    });
                });
                let mut output = RowMajorMatrix::new(vec![T::default(); width * height], height);
                group.bench_function(format!("preallocated/{shape}"), |b| {
                    b.iter(|| {
                        let input = black_box(&input);
                        if packable {
                            input.transpose_packable_into(black_box(&mut output));
                        } else {
                            input.transpose_into(black_box(&mut output));
                        }
                        black_box(&output);
                    });
                });
            }
            group.finish();
        }
    }
    bench::<u32>(c, "u32", |i| i as u32);
    bench::<Goldilocks>(c, "goldilocks", Goldilocks::from_usize);
}

criterion_group!(benches, transpose_benchmark, allocating_transpose_benchmark);
criterion_main!(benches);
