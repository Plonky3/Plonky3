//! Property-based and edge-case tests for the DFT crate internals.
//!
//! Complements the generic harness in `field-testing/dft_testing.rs`.

use core::mem::MaybeUninit;

use p3_baby_bear::BabyBear;
use p3_dft::{
    Butterfly, DifButterfly, DifButterflyZeros, DitButterfly, NaiveDft, Radix2Bowers,
    Radix2DFTSmallBatch, Radix2Dit, Radix2DitParallel, TwiddleFreeButterfly, TwoAdicSubgroupDft,
};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Concrete field used throughout this test module.
///
/// A single 31-bit prime field is sufficient to validate the field-agnostic DFT algorithms.
type F = BabyBear;

/// Derive a deterministic random field element from a seed.
fn rand_field(seed: u64) -> F {
    let mut rng = SmallRng::seed_from_u64(seed);
    rng.random()
}

/// Build a deterministic random matrix of the given dimensions.
fn rand_matrix(seed: u64, h: usize, w: usize) -> RowMajorMatrix<F> {
    let mut rng = SmallRng::seed_from_u64(seed);
    RowMajorMatrix::<F>::rand(&mut rng, h, w)
}

/// Proptest strategy that yields an arbitrary field element.
fn arb_field_elem() -> impl Strategy<Value = F> {
    any::<u64>().prop_map(rand_field)
}

/// Proptest strategy for log_2 of the matrix height.
///
/// Range [0, 8] gives subgroup sizes 1 through 256.
/// This keeps the O(n^2) naive reference under ~65 k field ops per column.
fn arb_log_h() -> impl Strategy<Value = usize> {
    0usize..=8
}

/// Proptest strategy for the matrix width (number of polynomials in a batch).
///
/// Range [1, 17] exercises both packing-aligned and non-aligned widths.
fn arb_width() -> impl Strategy<Value = usize> {
    1usize..=17
}

/// Proptest strategy for the number of extra bits added during a low-degree extension.
///
/// Range [1, 3] tests 2x, 4x, and 8x blowup factors.
fn arb_added_bits() -> impl Strategy<Value = usize> {
    1usize..=3
}

proptest! {
    #[test]
    fn dit_butterfly_matches_definition(
        x1 in arb_field_elem(), x2 in arb_field_elem(), twiddle in arb_field_elem(),
    ) {
        // Decimation-in-time gate: (x_1 + x_2 * t, x_1 - x_2 * t).
        let (y1, y2) = DitButterfly(twiddle).apply(x1, x2);

        // Pre-multiply to reuse the product.
        let x2t = x2 * twiddle;

        // First output is the sum.
        prop_assert_eq!(y1, x1 + x2t);
        // Second output is the difference.
        prop_assert_eq!(y2, x1 - x2t);
    }

    #[test]
    fn dif_butterfly_matches_definition(
        x1 in arb_field_elem(), x2 in arb_field_elem(), twiddle in arb_field_elem(),
    ) {
        // Decimation-in-frequency gate: (x_1 + x_2, (x_1 - x_2) * t).
        let (y1, y2) = DifButterfly(twiddle).apply(x1, x2);

        // First output is the plain sum.
        prop_assert_eq!(y1, x1 + x2);
        // Second output scales the difference by the twiddle.
        prop_assert_eq!(y2, (x1 - x2) * twiddle);
    }

    #[test]
    fn twiddle_free_butterfly_matches_definition(
        x1 in arb_field_elem(), x2 in arb_field_elem(),
    ) {
        // Twiddle-free gate (twiddle = 1): (x_1 + x_2, x_1 - x_2).
        let (y1, y2) = TwiddleFreeButterfly.apply(x1, x2);

        prop_assert_eq!(y1, x1 + x2);
        prop_assert_eq!(y2, x1 - x2);
    }

    #[test]
    fn dif_butterfly_zeros_matches_definition(
        x1 in arb_field_elem(), twiddle in arb_field_elem(),
    ) {
        // Optimized gate for zero-padded input: (x_1, x_1 * t).
        let (y1, y2) = DifButterflyZeros(twiddle).apply(x1, F::ZERO);

        // First output passes through unchanged.
        prop_assert_eq!(y1, x1);
        // Second output is the input scaled by the twiddle.
        prop_assert_eq!(y2, x1 * twiddle);
    }
}

proptest! {
    #[test]
    fn twiddle_free_butterfly_involution(
        x1 in arb_field_elem(), x2 in arb_field_elem(),
    ) {
        // Applying the twiddle-free gate twice must recover the original inputs scaled by 2.
        // Proof: B(B(x_1, x_2)) = B(x_1+x_2, x_1-x_2)
        //                        = (2*x_1, 2*x_2).
        let (a, b) = TwiddleFreeButterfly.apply(x1, x2);
        let (c, d) = TwiddleFreeButterfly.apply(a, b);

        prop_assert_eq!(c, x1.double());
        prop_assert_eq!(d, x2.double());
    }

    #[test]
    fn dit_dif_composition(
        x1 in arb_field_elem(), x2 in arb_field_elem(), twiddle in arb_field_elem(),
    ) {
        // Composing DIF after DIT with the same twiddle t gives:
        //   DIF(DIT(x_1, x_2))
        //     = DIF(x_1 + x_2*t, x_1 - x_2*t)
        //     = (2*x_1, 2*x_2*t^2).
        let (y1, y2) = DitButterfly(twiddle).apply(x1, x2);
        let (z1, z2) = DifButterfly(twiddle).apply(y1, y2);

        prop_assert_eq!(z1, x1.double());
        prop_assert_eq!(z2, (x2 * twiddle * twiddle).double());
    }
}

proptest! {
    #[test]
    fn dit_apply_to_rows_matches_scalar(seed: u64, w in 1usize..=33, twiddle in arb_field_elem()) {
        // Build two random rows of width `w`.
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut row1: Vec<F> = (0..w).map(|_| rng.random()).collect();
        let mut row2: Vec<F> = (0..w).map(|_| rng.random()).collect();

        // Snapshot before mutation.
        let orig1 = row1.clone();
        let orig2 = row2.clone();

        // In-place row-level transform (uses SIMD packing internally).
        DitButterfly(twiddle).apply_to_rows(&mut row1, &mut row2);

        // Each lane must match the scalar definition.
        for i in 0..w {
            let (expected1, expected2) = DitButterfly(twiddle).apply(orig1[i], orig2[i]);
            prop_assert_eq!(row1[i], expected1, "mismatch at index {}", i);
            prop_assert_eq!(row2[i], expected2, "mismatch at index {}", i);
        }
    }

    #[test]
    fn dif_apply_to_rows_matches_scalar(seed: u64, w in 1usize..=33, twiddle in arb_field_elem()) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut row1: Vec<F> = (0..w).map(|_| rng.random()).collect();
        let mut row2: Vec<F> = (0..w).map(|_| rng.random()).collect();
        let orig1 = row1.clone();
        let orig2 = row2.clone();

        DifButterfly(twiddle).apply_to_rows(&mut row1, &mut row2);

        for i in 0..w {
            let (expected1, expected2) = DifButterfly(twiddle).apply(orig1[i], orig2[i]);
            prop_assert_eq!(row1[i], expected1, "mismatch at index {}", i);
            prop_assert_eq!(row2[i], expected2, "mismatch at index {}", i);
        }
    }

    #[test]
    fn twiddle_free_apply_to_rows_matches_scalar(seed: u64, w in 1usize..=33) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut row1: Vec<F> = (0..w).map(|_| rng.random()).collect();
        let mut row2: Vec<F> = (0..w).map(|_| rng.random()).collect();
        let orig1 = row1.clone();
        let orig2 = row2.clone();

        TwiddleFreeButterfly.apply_to_rows(&mut row1, &mut row2);

        for i in 0..w {
            let (expected1, expected2) = TwiddleFreeButterfly.apply(orig1[i], orig2[i]);
            prop_assert_eq!(row1[i], expected1, "mismatch at index {}", i);
            prop_assert_eq!(row2[i], expected2, "mismatch at index {}", i);
        }
    }

    #[test]
    fn dif_zeros_apply_to_rows_matches_scalar(seed: u64, w in 1usize..=33, twiddle in arb_field_elem()) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut row1: Vec<F> = (0..w).map(|_| rng.random()).collect();
        // Second row is all zeros, matching the zero-padded precondition.
        let mut row2: Vec<F> = F::zero_vec(w);
        let orig1 = row1.clone();

        DifButterflyZeros(twiddle).apply_to_rows(&mut row1, &mut row2);

        for i in 0..w {
            // First row passes through unchanged.
            prop_assert_eq!(row1[i], orig1[i], "row1 mismatch at {}", i);
            // Second row receives the scaled copy.
            prop_assert_eq!(row2[i], orig1[i] * twiddle, "row2 mismatch at {}", i);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn dit_apply_to_rows_oop_matches_in_place(seed: u64, w in 1usize..=33, twiddle in arb_field_elem()) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let row1: Vec<F> = (0..w).map(|_| rng.random()).collect();
        let row2: Vec<F> = (0..w).map(|_| rng.random()).collect();

        // Compute the in-place reference result.
        let mut ip1 = row1.clone();
        let mut ip2 = row2.clone();
        DitButterfly(twiddle).apply_to_rows(&mut ip1, &mut ip2);

        // Compute the out-of-place result into uninitialized buffers.
        let mut dst1: Vec<MaybeUninit<F>> = vec![MaybeUninit::uninit(); w];
        let mut dst2: Vec<MaybeUninit<F>> = vec![MaybeUninit::uninit(); w];
        DitButterfly(twiddle).apply_to_rows_oop(&row1, &mut dst1, &row2, &mut dst2);

        // Read back the initialized values.
        let oop1: Vec<F> = dst1.iter().map(|x| unsafe { x.assume_init() }).collect();
        let oop2: Vec<F> = dst2.iter().map(|x| unsafe { x.assume_init() }).collect();

        // Both paths must agree.
        prop_assert_eq!(ip1, oop1);
        prop_assert_eq!(ip2, oop2);
    }

    #[test]
    fn dif_apply_to_rows_oop_matches_in_place(seed: u64, w in 1usize..=33, twiddle in arb_field_elem()) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let row1: Vec<F> = (0..w).map(|_| rng.random()).collect();
        let row2: Vec<F> = (0..w).map(|_| rng.random()).collect();

        // In-place reference.
        let mut ip1 = row1.clone();
        let mut ip2 = row2.clone();
        DifButterfly(twiddle).apply_to_rows(&mut ip1, &mut ip2);

        // Out-of-place path.
        let mut dst1: Vec<MaybeUninit<F>> = vec![MaybeUninit::uninit(); w];
        let mut dst2: Vec<MaybeUninit<F>> = vec![MaybeUninit::uninit(); w];
        DifButterfly(twiddle).apply_to_rows_oop(&row1, &mut dst1, &row2, &mut dst2);
        let oop1: Vec<F> = dst1.iter().map(|x| unsafe { x.assume_init() }).collect();
        let oop2: Vec<F> = dst2.iter().map(|x| unsafe { x.assume_init() }).collect();

        prop_assert_eq!(ip1, oop1);
        prop_assert_eq!(ip2, oop2);
    }

    #[test]
    fn twiddle_free_apply_to_rows_oop_matches_in_place(seed: u64, w in 1usize..=33) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let row1: Vec<F> = (0..w).map(|_| rng.random()).collect();
        let row2: Vec<F> = (0..w).map(|_| rng.random()).collect();

        // In-place reference.
        let mut ip1 = row1.clone();
        let mut ip2 = row2.clone();
        TwiddleFreeButterfly.apply_to_rows(&mut ip1, &mut ip2);

        // Out-of-place path.
        let mut dst1: Vec<MaybeUninit<F>> = vec![MaybeUninit::uninit(); w];
        let mut dst2: Vec<MaybeUninit<F>> = vec![MaybeUninit::uninit(); w];
        TwiddleFreeButterfly.apply_to_rows_oop(&row1, &mut dst1, &row2, &mut dst2);
        let oop1: Vec<F> = dst1.iter().map(|x| unsafe { x.assume_init() }).collect();
        let oop2: Vec<F> = dst2.iter().map(|x| unsafe { x.assume_init() }).collect();

        prop_assert_eq!(ip1, oop1);
        prop_assert_eq!(ip2, oop2);
    }
}

proptest! {
    #[test]
    fn all_dfts_agree(seed: u64, log_h in 0usize..=7, w in arb_width()) {
        let h = 1 << log_h;
        let mat = rand_matrix(seed, h, w);

        // Compute the forward DFT with every backend.
        let naive = NaiveDft.dft_batch(mat.clone());
        let dit = Radix2Dit::default().dft_batch(mat.clone());
        let bowers = Radix2Bowers.dft_batch(mat.clone());
        let parallel = Radix2DitParallel::default().dft_batch(mat.clone()).to_row_major_matrix();
        let small_batch = Radix2DFTSmallBatch::default().dft_batch(mat);

        // All must match the naive O(n^2) reference.
        prop_assert_eq!(&naive, &dit);
        prop_assert_eq!(&naive, &bowers);
        prop_assert_eq!(&naive, &parallel);
        prop_assert_eq!(&naive, &small_batch);
    }

    #[test]
    fn all_idfts_agree(seed: u64, log_h in 0usize..=7, w in arb_width()) {
        let h = 1 << log_h;
        let mat = rand_matrix(seed, h, w);

        // Compute the inverse DFT with every backend.
        let naive = NaiveDft.idft_batch(mat.clone());
        let dit = Radix2Dit::default().idft_batch(mat.clone());
        let bowers = Radix2Bowers.idft_batch(mat.clone());
        let parallel = Radix2DitParallel::default().idft_batch(mat.clone());
        let small_batch = Radix2DFTSmallBatch::default().idft_batch(mat);

        prop_assert_eq!(&naive, &dit);
        prop_assert_eq!(&naive, &bowers);
        prop_assert_eq!(&naive, &parallel);
        prop_assert_eq!(&naive, &small_batch);
    }

    #[test]
    fn all_ldes_agree(
        seed: u64, log_h in 0usize..=7, w in arb_width(), added_bits in arb_added_bits()
    ) {
        let h = 1 << log_h;
        let mat = rand_matrix(seed, h, w);

        // Compute the low-degree extension with every backend.
        let naive = NaiveDft.lde_batch(mat.clone(), added_bits).to_row_major_matrix();
        let dit = Radix2Dit::default().lde_batch(mat.clone(), added_bits).to_row_major_matrix();
        let bowers = Radix2Bowers.lde_batch(mat.clone(), added_bits);
        let parallel = Radix2DitParallel::default().lde_batch(mat.clone(), added_bits).to_row_major_matrix();
        let small_batch = Radix2DFTSmallBatch::default().lde_batch(mat, added_bits).to_row_major_matrix();

        prop_assert_eq!(&naive, &dit);
        prop_assert_eq!(&naive, &bowers);
        prop_assert_eq!(&naive, &parallel);
        prop_assert_eq!(&naive, &small_batch);
    }

    #[test]
    fn all_coset_ldes_agree(
        seed: u64, log_h in 0usize..=6, w in 1usize..=8,
        added_bits in arb_added_bits(), shift in arb_field_elem()
    ) {
        let h = 1 << log_h;
        let mat = rand_matrix(seed, h, w);

        // Compute the coset low-degree extension with every backend.
        let naive = NaiveDft.coset_lde_batch(mat.clone(), added_bits, shift).to_row_major_matrix();
        let dit = Radix2Dit::default().coset_lde_batch(mat.clone(), added_bits, shift);
        let bowers = Radix2Bowers.coset_lde_batch(mat.clone(), added_bits, shift);
        let parallel = Radix2DitParallel::default()
            .coset_lde_batch(mat.clone(), added_bits, shift)
            .to_row_major_matrix();
        let small_batch = Radix2DFTSmallBatch::default().coset_lde_batch(mat, added_bits, shift);

        prop_assert_eq!(&naive, &dit);
        prop_assert_eq!(&naive, &bowers);
        prop_assert_eq!(&naive, &parallel);
        prop_assert_eq!(&naive, &small_batch);
    }
}

proptest! {
    #[test]
    fn radix2dit_dft_idft_roundtrip(seed: u64, log_h in arb_log_h(), w in arb_width()) {
        let dft = Radix2Dit::<F>::default();
        let original = rand_matrix(seed, 1 << log_h, w);

        // Forward transform.
        let forward = dft.dft_batch(original.clone());
        // Inverse must recover the coefficients.
        let back = dft.idft_batch(forward);

        prop_assert_eq!(original, back);
    }

    #[test]
    fn bowers_dft_idft_roundtrip(seed: u64, log_h in arb_log_h(), w in arb_width()) {
        let original = rand_matrix(seed, 1 << log_h, w);

        let forward = Radix2Bowers.dft_batch(original.clone());
        let back = Radix2Bowers.idft_batch(forward);

        prop_assert_eq!(original, back);
    }

    #[test]
    fn dit_parallel_dft_idft_roundtrip(seed: u64, log_h in arb_log_h(), w in arb_width()) {
        let dft = Radix2DitParallel::<F>::default();
        let original = rand_matrix(seed, 1 << log_h, w);

        // This backend returns a bit-reversed view; materialize it
        // before passing to the inverse.
        let forward = dft.dft_batch(original.clone()).to_row_major_matrix();
        let back = dft.idft_batch(forward);

        prop_assert_eq!(original, back);
    }

    #[test]
    fn small_batch_dft_idft_roundtrip(seed: u64, log_h in arb_log_h(), w in arb_width()) {
        let dft = Radix2DFTSmallBatch::<F>::default();
        let original = rand_matrix(seed, 1 << log_h, w);

        let forward = dft.dft_batch(original.clone());
        let back = dft.idft_batch(forward);

        prop_assert_eq!(original, back);
    }
}

proptest! {
    #[test]
    fn radix2dit_coset_roundtrip(seed: u64, log_h in arb_log_h(), w in arb_width(), shift in arb_field_elem()) {
        let dft = Radix2Dit::<F>::default();
        let original = rand_matrix(seed, 1 << log_h, w);

        // Forward coset DFT evaluates on shift * H.
        let forward = dft.coset_dft_batch(original.clone(), shift);
        // Inverse coset DFT recovers the coefficients.
        let back = dft.coset_idft_batch(forward, shift);

        prop_assert_eq!(original, back);
    }

    #[test]
    fn bowers_coset_roundtrip(seed: u64, log_h in arb_log_h(), w in arb_width(), shift in arb_field_elem()) {
        let original = rand_matrix(seed, 1 << log_h, w);

        let forward = Radix2Bowers.coset_dft_batch(original.clone(), shift);
        let back = Radix2Bowers.coset_idft_batch(forward, shift);

        prop_assert_eq!(original, back);
    }
}

proptest! {
    #[test]
    fn dft_linearity(
        seed1: u64, seed2: u64, log_h in 0usize..=6, w in 1usize..=8,
        a in arb_field_elem(), b in arb_field_elem(),
    ) {
        let dft = Radix2Dit::<F>::default();
        let h = 1 << log_h;

        // Two independent random matrices.
        let x = rand_matrix(seed1, h, w);
        let y = rand_matrix(seed2, h, w);

        // Build the linear combination a*x + b*y element-wise.
        let ax_plus_by = RowMajorMatrix::new(
            x.values.iter().zip(y.values.iter())
                .map(|(&xi, &yi)| a * xi + b * yi)
                .collect(),
            w,
        );

        // DFT of the combined input.
        let dft_combined = dft.dft_batch(ax_plus_by);

        // Same linear combination of the individual DFTs.
        let dft_x = dft.dft_batch(x);
        let dft_y = dft.dft_batch(y);
        let expected = RowMajorMatrix::new(
            dft_x.values.iter().zip(dft_y.values.iter())
                .map(|(&xi, &yi)| a * xi + b * yi)
                .collect(),
            w,
        );

        // Linearity: both must match.
        prop_assert_eq!(dft_combined, expected);
    }
}

#[test]
fn dft_height_1_is_identity() {
    // A height-1 matrix represents a degree-0 polynomial.
    // The unique subgroup of order 1 is {1}, so DFT[0] = p(1) = c_0.
    // The output must equal the input for every backend.
    let mat = RowMajorMatrix::new(vec![F::from_u8(42), F::from_u8(7)], 2);

    assert_eq!(mat, Radix2Dit::default().dft_batch(mat.clone()));
    assert_eq!(mat, Radix2Bowers.dft_batch(mat.clone()));
    assert_eq!(
        mat,
        Radix2DitParallel::default()
            .dft_batch(mat.clone())
            .to_row_major_matrix()
    );
    assert_eq!(mat, Radix2DFTSmallBatch::default().dft_batch(mat.clone()));
}

#[test]
fn dft_all_zeros() {
    // The DFT of the zero polynomial is the zero evaluation vector.
    for log_h in 0..=6 {
        let h = 1 << log_h;
        // Build a zero matrix with 3 columns.
        let mat = RowMajorMatrix::new(F::zero_vec(h * 3), 3);

        // Check two backends as representative (the cross-impl tests
        // cover all five with randomized inputs).
        let dit = Radix2Dit::default().dft_batch(mat.clone());
        assert!(dit.values.iter().all(|x| x.is_zero()), "log_h={log_h}");

        let bowers = Radix2Bowers.dft_batch(mat.clone());
        assert!(bowers.values.iter().all(|x| x.is_zero()), "log_h={log_h}");
    }
}

#[test]
fn idft_all_zeros() {
    // The inverse DFT of the zero evaluation vector is the zero
    // coefficient vector.
    for log_h in 0..=6 {
        let h = 1 << log_h;
        let mat = RowMajorMatrix::new(F::zero_vec(h * 3), 3);

        let dit = Radix2Dit::default().idft_batch(mat.clone());
        assert!(dit.values.iter().all(|x| x.is_zero()), "log_h={log_h}");

        let bowers = Radix2Bowers.idft_batch(mat.clone());
        assert!(bowers.values.iter().all(|x| x.is_zero()), "log_h={log_h}");
    }
}

#[test]
fn dft_constant_polynomial() {
    // A constant polynomial p(x) = c evaluates to c everywhere.
    // In coefficient form that is [c, 0, 0, ...].
    // Every evaluation DFT[i] = p(g^i) = c.
    let c = F::from_u8(13);
    for log_h in 0..=6 {
        let h = 1 << log_h;

        // Coefficient vector: c followed by zeros.
        let mut vals = F::zero_vec(h);
        vals[0] = c;
        let mat = RowMajorMatrix::new(vals, 1);

        let result = Radix2Dit::default().dft_batch(mat);

        // Every evaluation must equal c.
        for (i, &v) in result.values.iter().enumerate() {
            assert_eq!(v, c, "log_h={log_h}, row={i}");
        }
    }
}

#[test]
fn idft_constant_vector() {
    // The inverse of the constant-polynomial property:
    // if every evaluation is c, the coefficients are [c, 0, 0, ...].
    let c = F::from_u8(7);
    for log_h in 0..=6 {
        let h = 1 << log_h;

        // Evaluation vector: all entries equal to c.
        let mat = RowMajorMatrix::new(vec![c; h], 1);

        let result = Radix2Dit::default().idft_batch(mat);

        // First coefficient is c.
        assert_eq!(result.values[0], c, "log_h={log_h}");
        // All higher coefficients are zero.
        for (i, &v) in result.values.iter().enumerate().skip(1) {
            assert_eq!(v, F::ZERO, "log_h={log_h}, coeff[{i}] should be 0");
        }
    }
}

#[test]
fn dft_width_1_matches_naive() {
    // Width 1 means a single polynomial (not a batch).
    // This exercises the degenerate-width code path in every layer.
    for log_h in 0..=8 {
        let h = 1 << log_h;
        let mat = rand_matrix(42, h, 1);

        let expected = NaiveDft.dft_batch(mat.clone());
        let actual = Radix2Dit::default().dft_batch(mat);

        assert_eq!(expected, actual, "log_h={log_h}");
    }
}

#[test]
fn coset_lde_with_shift_one_equals_lde() {
    // When the coset shift is the multiplicative identity, the coset
    // low-degree extension must be identical to the plain (unshifted)
    // low-degree extension.
    let dft = Radix2Dit::<F>::default();
    for log_h in 1..=6 {
        let h = 1 << log_h;
        let mat = rand_matrix(123, h, 4);

        // Plain LDE with blowup factor 4 (added_bits = 2).
        let lde = dft.lde_batch(mat.clone(), 2);
        // Coset LDE with trivial shift.
        let coset_lde = dft.coset_lde_batch(mat, 2, F::ONE);

        assert_eq!(lde, coset_lde, "log_h={log_h}");
    }
}

/// Evaluate a single-column coefficient matrix as a polynomial at every
/// point in the two-adic subgroup of order `h`, using the naive
/// sum p(g^i) = sum_j c_j * (g^i)^j.
///
/// Returns a vector of length `h` with the evaluations in natural order.
fn eval_poly_on_subgroup(coeffs: &RowMajorMatrix<F>, log_h: usize) -> Vec<F> {
    let h = 1 << log_h;
    // Primitive n-th root of unity generating the subgroup.
    let g = F::two_adic_generator(log_h);

    g.powers()
        .take(h)
        .map(|point| {
            // Accumulate p(point) = sum_j c_j * point^j.
            let mut eval = F::ZERO;
            let mut point_pow = F::ONE;
            for j in 0..h {
                eval += coeffs.values[j] * point_pow;
                // Advance to the next power of the evaluation point.
                point_pow *= point;
            }
            eval
        })
        .collect()
}

proptest! {
    #[test]
    fn dft_is_polynomial_evaluation(seed: u64, log_h in 1usize..=6) {
        // The DFT of a coefficient vector [c_0, c_1, ..., c_{n-1}] must
        // equal the direct evaluation p(g^i) = sum_j c_j * (g^i)^j for
        // every point in the two-adic subgroup.
        let h = 1 << log_h;
        // Single-column random polynomial.
        let coeffs = rand_matrix(seed, h, 1);

        // Ground-truth evaluations computed by naive summation.
        let expected = eval_poly_on_subgroup(&coeffs, log_h);

        // Every backend must match the direct evaluation.
        let dit = Radix2Dit::default().dft_batch(coeffs.clone());
        for (i, &e) in expected.iter().enumerate() {
            prop_assert_eq!(dit.values[i], e, "Radix2Dit mismatch at point {}", i);
        }

        let bowers = Radix2Bowers.dft_batch(coeffs.clone());
        for (i, &e) in expected.iter().enumerate() {
            prop_assert_eq!(bowers.values[i], e, "Bowers mismatch at point {}", i);
        }

        let parallel = Radix2DitParallel::default()
            .dft_batch(coeffs.clone())
            .to_row_major_matrix();
        for (i, &e) in expected.iter().enumerate() {
            prop_assert_eq!(parallel.values[i], e, "DitParallel mismatch at point {}", i);
        }

        let small_batch = Radix2DFTSmallBatch::default().dft_batch(coeffs);
        for (i, &e) in expected.iter().enumerate() {
            prop_assert_eq!(small_batch.values[i], e, "SmallBatch mismatch at point {}", i);
        }
    }
}
