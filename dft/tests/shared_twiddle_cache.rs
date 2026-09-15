//! Concurrent transforms using clones of the same twiddle cache.

#![cfg(feature = "parallel")]

use std::sync::mpsc;
use std::time::Duration;

use p3_baby_bear::BabyBear;
use p3_dft::{Radix2Dit, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::TwoAdicField;
use p3_goldilocks::Goldilocks;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

fn input<F: TwoAdicField>(log_height: usize, width: usize) -> RowMajorMatrix<F> {
    let values = (0..(1 << log_height) * width)
        .map(|i| F::from_usize(i + 1))
        .collect();
    RowMajorMatrix::new(values, width)
}

fn check_shared_cache<F: TwoAdicField + Ord>() {
    for warm_inverse in [false, true] {
        for varying_keys in [false, true] {
            let dft = Radix2DitParallel::<F>::default();
            if warm_inverse {
                dft.idft_batch(input(11, 1));
            }
            let clones: Vec<_> = (0..8).map(|_| dft.clone()).collect();
            clones.into_par_iter().enumerate().for_each(|(task, dft)| {
                // Both sizes reach the parallel power-collection threshold.
                let log_height = 11 + usize::from(varying_keys) * (task / 4);
                let shift = F::GENERATOR.exp_u64(if varying_keys {
                    1 + (task / 4) as u64
                } else {
                    1
                });
                let input = input(log_height, 1 + task % 3);
                let reference = Radix2Dit::<F>::default();
                for _ in 0..2 {
                    let (actual, expected) = match task % 4 {
                        0 => (
                            dft.dft_batch(input.clone()).to_row_major_matrix(),
                            reference.dft_batch(input.clone()),
                        ),
                        1 => (
                            dft.coset_dft_batch(input.clone(), shift)
                                .to_row_major_matrix(),
                            reference.coset_dft_batch(input.clone(), shift),
                        ),
                        2 => (
                            dft.idft_batch(input.clone()),
                            reference.idft_batch(input.clone()),
                        ),
                        _ => (
                            dft.coset_lde_batch(input.clone(), 3, shift)
                                .to_row_major_matrix(),
                            reference.coset_lde_batch(input.clone(), 3, shift),
                        ),
                    };
                    assert_eq!(actual, expected);
                }
            });
        }
    }
}

#[test]
fn cloned_dfts_complete_with_cold_and_warm_caches() {
    for threads in [1, 2, 4] {
        let (done, completed) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = std::panic::catch_unwind(|| {
                let pool = ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                pool.install(|| {
                    check_shared_cache::<BabyBear>();
                    check_shared_cache::<Goldilocks>();
                });
            });
            let _ = done.send(result);
        });
        // A hang guard, not a performance bound: debug builds on loaded CI runners take tens
        // of seconds per pool size.
        let result = completed
            .recv_timeout(Duration::from_secs(300))
            .unwrap_or_else(|error| {
                panic!(
                    "shared-cache worker did not report completion with {threads} threads: {error}"
                )
            });
        worker.join().unwrap();
        if let Err(error) = result {
            std::panic::resume_unwind(error);
        }
    }
}
