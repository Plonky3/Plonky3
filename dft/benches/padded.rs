//! Compare full and zero-tail coefficient DFTs with cold and shared twiddle caches.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;

fn padded_dft(c: &mut Criterion) {
    let log_height = std::env::var("P3_PADDED_LOG_HEIGHT")
        .map(|value| value.parse::<usize>().expect("valid log height"))
        .unwrap_or(14);
    let height = 1usize.checked_shl(log_height.try_into().unwrap()).unwrap();
    let mut rng = SmallRng::seed_from_u64(1);
    for cold in [false, true] {
        let mut group = c.benchmark_group(format!(
            "padded_dft/{}/log_height={log_height}",
            if cold { "cold" } else { "warm" },
        ));
        group.sample_size(10);
        for log_inv_rate in [0, 1, 2, 3, 4, 5, 8] {
            if log_inv_rate > log_height {
                continue;
            }
            for width in [1, 4, 16, 64, 256] {
                let mut message =
                    RowMajorMatrix::<BabyBear>::rand(&mut rng, height >> log_inv_rate, width);
                message.values.resize(height * width, BabyBear::ZERO);
                for padded in [false, true] {
                    let dft = Radix2DitParallel::default();
                    // Resolve the full-height tables before measuring the warm case.
                    if !cold {
                        drop(dft.dft_batch(message.clone()));
                    }
                    group.bench_function(
                        BenchmarkId::new(
                            if padded { "padded" } else { "full" },
                            format!("rate={}/width={width}", 1usize << log_inv_rate),
                        ),
                        |b| {
                            b.iter_batched(
                                || message.clone(),
                                |input| {
                                    let dft = if cold {
                                        Radix2DitParallel::default()
                                    } else {
                                        dft.clone()
                                    };
                                    if padded {
                                        dft.dft_batch_padded(input, log_inv_rate)
                                    } else {
                                        dft.dft_batch(input)
                                    }
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
}

criterion_group!(benches, padded_dft);
criterion_main!(benches);
