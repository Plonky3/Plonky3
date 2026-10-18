//! Compare extension encoding with a full padded DFT, excluding input cloning.

use core::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2DFTSmallBatch, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_whir::WhirDomain;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

fn bench_backend<Dft: TwoAdicSubgroupDft<F> + Sync>(c: &mut Criterion, name: &str) {
    let mut group = c.benchmark_group(format!("whir/encoding/{name}"));
    for (log_len, width) in [(10, 1), (10, 8), (10, 32), (14, 1), (14, 8)] {
        for log_inv_rate in [0, 1, 2, 3, 4, 6] {
            let height = 1 << (log_len + log_inv_rate);
            let label = format!("l{log_len}_r{log_inv_rate}_w{width}");
            group.throughput(Throughput::Elements((height * width) as u64));
            for optimized in [false, true] {
                let method = if optimized {
                    "degree_aware"
                } else {
                    "full_dft"
                };
                group.bench_with_input(
                    BenchmarkId::new(method, &label),
                    &optimized,
                    |b, &optimized| {
                        // Criterion invokes this closure only for selected cases. Keep fixture
                        // allocation here so narrow filters never build the large unused inputs.
                        let mut values = EF::zero_vec(height * width);
                        for (i, value) in values[..(1 << log_len) * width].iter_mut().enumerate() {
                            *value =
                                EF::from_basis_coefficients_fn(|j| F::from_usize(i * 7 + j + 1));
                        }
                        let message = RowMajorMatrix::new(values, width);
                        let dft = Dft::default();
                        b.iter_batched(
                            || message.clone(),
                            |message| {
                                black_box(if optimized {
                                    dft.encode_extension_batch_padded(
                                        black_box(message),
                                        log_inv_rate,
                                    )
                                } else {
                                    dft.dft_algebra_batch(black_box(message))
                                })
                            },
                            BatchSize::LargeInput,
                        );
                    },
                );
            }
        }
    }
    group.finish();
}

fn bench_encoding(c: &mut Criterion) {
    bench_backend::<Radix2DFTSmallBatch<F>>(c, "small_batch");
    bench_backend::<Radix2DitParallel<F>>(c, "parallel");
}

criterion_group!(benches, bench_encoding);
criterion_main!(benches);
