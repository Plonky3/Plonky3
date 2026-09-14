use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use p3_baby_bear::BabyBear;
use p3_dft::{NaiveDft, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_goldilocks::Goldilocks;
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::reverse_bits_len;

fn check_blocks<F: TwoAdicField, D: TwoAdicSubgroupDft<F>>(dft: &D) {
    // Copy-only, no final layers, odd/even split depths, and zero/nonzero expansion.
    for (log_h, width, added_bits) in [(0, 1, 0), (0, 3, 2), (1, 8, 1), (3, 3, 0), (4, 17, 3)] {
        let h = 1 << log_h;
        let input = RowMajorMatrix::new(
            (0..h * width).map(|i| F::from_usize(i + 1)).collect(),
            width,
        );
        let mut expected = NaiveDft.coset_lde_batch(input.clone(), added_bits, F::GENERATOR);
        expected.scale(F::TWO);
        let rows = dft.lde_output_block_rows(h, added_bits);
        assert!(rows.is_power_of_two() && expected.height().is_multiple_of(rows));
        let seen: Vec<_> = (0..expected.height() / rows)
            .map(|_| AtomicBool::new(false))
            .collect();
        // Borrowing Cell keeps the coefficient closure non-Send and non-Sync.
        let calls = Cell::new(0);
        let caller = std::thread::current().id();
        let output = dft.coset_lde_batch_with_blocks(
            input,
            added_bits,
            F::GENERATOR,
            |matrix, _| {
                assert_eq!(std::thread::current().id(), caller);
                calls.set(calls.get() + 1);
                matrix.scale(F::TWO);
            },
            |start, block| {
                assert_eq!((block.height(), block.width()), (rows, width));
                assert_eq!(start % rows, 0);
                assert!(
                    !seen[start / rows].swap(true, Ordering::Relaxed),
                    "duplicate output block"
                );
                // Compare inside the callback to detect premature publication.
                for (offset, values) in block.values.chunks_exact(width).enumerate() {
                    let natural = reverse_bits_len(start + offset, log_h + added_bits);
                    assert_eq!(
                        values,
                        &expected.values[natural * width..(natural + 1) * width]
                    );
                }
            },
        );
        assert_eq!(calls.get(), 1);
        assert!(
            seen.iter().all(|block| block.load(Ordering::Relaxed)),
            "missing output block"
        );
        assert_eq!(output.to_row_major_matrix(), expected);
    }
}

#[test]
fn parallel_blocks_match_lde_values() {
    let check = || {
        check_blocks::<BabyBear, _>(&Radix2DitParallel::default());
        check_blocks::<Goldilocks, _>(&Radix2DitParallel::default());
    };
    #[cfg(feature = "parallel")]
    for workers in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(check);
    }
    #[cfg(not(feature = "parallel"))]
    check();
}

#[test]
fn default_blocks_match_lde_values() {
    // Exercise a bit-reversed backend with one-row blocks through the default method.
    #[derive(Clone, Default)]
    struct RowBlocks;

    impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for RowBlocks {
        type Evaluations = BitReversedMatrixView<RowMajorMatrix<F>>;

        fn dft_batch(&self, mat: RowMajorMatrix<F>) -> Self::Evaluations {
            let mut output = NaiveDft.dft_batch(mat);
            p3_matrix::util::reverse_matrix_index_bits(&mut output);
            output.bit_reverse_rows()
        }

        fn lde_output_block_rows(&self, _input_height: usize, _added_bits: usize) -> usize {
            1
        }
    }

    check_blocks::<Goldilocks, _>(&NaiveDft);
    check_blocks::<Goldilocks, _>(&RowBlocks);
}

#[test]
fn consumer_panic_propagates_and_dft_is_reusable() {
    let dft = Radix2DitParallel::<Goldilocks>::default();
    for h in [1, 16] {
        let input = RowMajorMatrix::new((0..h * 3).map(Goldilocks::from_usize).collect(), 3);
        let expected = NaiveDft.coset_lde_batch(input.clone(), 3, Goldilocks::GENERATOR);
        for fail_row in [0, h] {
            let failure = catch_unwind(AssertUnwindSafe(|| {
                dft.coset_lde_batch_with_blocks(
                    input.clone(),
                    3,
                    Goldilocks::GENERATOR,
                    |_, _| {},
                    |start, _| {
                        if start == fail_row {
                            panic!("consumer failure");
                        }
                    },
                )
            }));
            assert_eq!(
                failure.unwrap_err().downcast_ref::<&str>(),
                Some(&"consumer failure")
            );

            let nonzero_rows = AtomicUsize::new(0);
            let output = dft.coset_lde_batch_with_blocks(
                input.clone(),
                3,
                Goldilocks::GENERATOR,
                |_, _| {},
                |start, block| {
                    if start < h {
                        // Coset zero must wait for every shared-input reader and its consumer.
                        assert_eq!(nonzero_rows.load(Ordering::Acquire), 7 * h);
                    } else {
                        nonzero_rows.fetch_add(block.height(), Ordering::Release);
                    }
                },
            );
            assert_eq!(output.to_row_major_matrix(), expected);
        }
    }
}
