//! SIMD-aware polynomial pair for quadratic sumcheck.
//!
//! This module implements a data structure that manages two multilinear polynomials.
//!
//! Evaluations and weights—whose pointwise product is required for the sumcheck protocol.
//!
//! # Mathematical Background
//!
//! In the sumcheck protocol, we prove knowledge of a claimed sum:
//!
//! ```text
//! S = \sum_{x \in \{0,1\}^n} f(x) \cdot w(x)
//! ```
//!
//! where:
//! - `f(x)` is the multilinear polynomial being sumchecked (evaluations).
//! - `w(x)` is the weight polynomial, typically derived from equality constraints.
//!
//! At each round, we compute a univariate polynomial `h(X)` that represents the partial sum
//! over remaining variables. For quadratic sumcheck, `h(X)` is degree-2.

use core::marker::PhantomData;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::constraints::Constraint;
use crate::sumcheck::strategy::SumcheckStrategy;
use crate::sumcheck::{SumcheckData, extrapolate_01inf};

/// A paired representation of evaluation and weight polynomials for quadratic sumcheck.
///
/// This enum stores two multilinear polynomials:
/// - `evals` (the polynomial being sumchecked)
/// - `weights` (the constraint weights),
///
/// in either SIMD-packed or scalar format, depending on the polynomial size.
///
/// # Memory Layout
///
/// In packed mode, evaluations are organized as:
///
/// ```text
/// Logical view:   [f(0,0,...,0), f(0,0,...,1), ..., f(1,1,...,1)]
///                 \---------------- 2^n elements ---------------/
///
/// Packed view:    [Pack_0, Pack_1, ..., Pack_{2^n/W - 1}]
///                 \--2^{n - log_2(W)} packed elements--/
///
/// where W = SIMD_WIDTH and each Pack_i contains W consecutive field elements.
/// ```
///
/// # Variants
///
/// - Packed: Uses SIMD-packed extension field elements for large polynomials.
///   Each packed element contains `F::Packing::WIDTH` consecutive evaluations, enabling
///   parallel arithmetic across SIMD lanes.
///
/// - Small: Uses scalar extension field elements for small polynomials
///   where SIMD overhead would exceed the benefit.
///
/// # Transition Logic
///
/// The representation transitions from `Packed` to `Small` when:
///
/// ```text
/// num_variables <= log_2(SIMD_WIDTH)
/// ```
///
/// This occurs after sufficient rounds of folding reduce the polynomial size below the
/// SIMD efficiency threshold.
#[derive(Debug, Clone)]
enum MaybePacked<F: Field, EF: ExtensionField<F>> {
    /// SIMD-packed representation for large polynomials.
    ///
    /// Each element in `evals` and `weights` is an `EF::ExtensionPacking`, which holds
    /// `F::Packing::WIDTH` extension field elements packed into SIMD lanes.
    ///
    /// # Memory Efficiency
    ///
    /// For a polynomial with `2^n` evaluations and SIMD width `W`:
    /// - Stored elements: `2^{n - log_2(W)}`
    /// - Memory per element: `sizeof(EF) * W`
    /// - Total memory: `2^n * sizeof(EF)` (same as scalar, but with SIMD alignment)
    Packed {
        /// Packed evaluations of the polynomial `f(x)` being sumchecked.
        ///
        /// Layout: `evals[i]` contains logical evaluations at indices `[i*W, (i+1)*W)`.
        evals: Poly<EF::ExtensionPacking>,

        /// Packed evaluations of the weight polynomial `w(x)`.
        ///
        /// Derived from equality constraints and challenge batching.
        weights: Poly<EF::ExtensionPacking>,
    },

    /// Scalar representation for small polynomials.
    ///
    /// Each element in `evals` and `weights` is a single extension field element.
    ///
    /// Used when the polynomial is too small for SIMD packing to be beneficial.
    Unpacked {
        /// Scalar evaluations of the polynomial `f(x)` being sumchecked.
        evals: Poly<EF>,

        /// Scalar evaluations of the weight polynomial `w(x)`.
        weights: Poly<EF>,
    },
}

/// Paired evaluation and weight polynomials, tagged by a sumcheck strategy.
///
/// # Contents
///
/// - Backing data kept in either SIMD-packed or scalar form.
/// - Strategy type tag that drives round-level dispatch.
///
/// # Role of the strategy type
///
/// Hot-path operations dispatch through strategy-level associated functions:
///
/// - Variable binding: prefix-first or suffix-first folding.
/// - Round coefficients: differ in which side of the hypercube is summed.
///
/// Both strategy structs are zero-sized, so only the type is carried — no
/// runtime value is needed to pick the dispatch.
#[derive(Debug, Clone)]
pub struct ProductPolynomial<F: Field, EF: ExtensionField<F>, St: SumcheckStrategy> {
    /// Paired polynomial data, SIMD-packed for large inputs and scalar otherwise.
    inner: MaybePacked<F, EF>,
    /// Ties the sumcheck strategy into the type without storing a runtime value.
    _strategy: PhantomData<St>,
}

impl<F: Field, EF: ExtensionField<F>, St: SumcheckStrategy> ProductPolynomial<F, EF, St> {
    /// Creates a packed variant and runs an immediate transition check.
    ///
    /// # Arguments
    ///
    /// - `evals`   — packed evaluations of the sumchecked polynomial.
    /// - `weights` — packed evaluations of the weight polynomial.
    ///
    /// # Panics
    ///
    /// - Evaluation and weight polynomials must share the same arity.
    pub fn new_packed(
        evals: Poly<EF::ExtensionPacking>,
        weights: Poly<EF::ExtensionPacking>,
    ) -> Self {
        // Paired polynomials must share the same variable space.
        assert_eq!(evals.num_vars(), weights.num_vars());

        // Wrap the packed pair; the strategy type tag is zero-sized.
        let mut poly = Self {
            inner: MaybePacked::Packed { evals, weights },
            _strategy: PhantomData,
        };

        // Corner case: if the input is already small, switch to scalar mode.
        poly.transition();
        poly
    }

    /// Creates a scalar variant for polynomials too small for SIMD packing.
    ///
    /// # Arguments
    ///
    /// - `evals`   — scalar evaluations of the sumchecked polynomial.
    /// - `weights` — scalar evaluations of the weight polynomial.
    pub const fn new_unpacked(evals: Poly<EF>, weights: Poly<EF>) -> Self {
        Self {
            inner: MaybePacked::Unpacked { evals, weights },
            _strategy: PhantomData,
        }
    }

    /// Returns the number of variables in the multilinear polynomials.
    ///
    /// This is the logical number of variables, accounting for SIMD packing.
    ///
    /// # Computation
    ///
    /// - **Packed**: `stored_variables + log_2(SIMD_WIDTH)`
    /// - **Small**: `stored_variables`
    pub fn num_vars(&self) -> usize {
        match &self.inner {
            MaybePacked::Packed { evals, weights } => {
                // Get the number of variables in the packed representation.
                let k = evals.num_vars();
                assert_eq!(k, weights.num_vars());

                // Add back the variables absorbed by SIMD packing.
                k + log2_strict_usize(F::Packing::WIDTH)
            }
            MaybePacked::Unpacked { evals, weights } => {
                let k = evals.num_vars();
                assert_eq!(k, weights.num_vars());
                k
            }
        }
    }

    /// Evaluates the polynomial `f(x)` at a given multilinear point.
    ///
    /// This computes `f(point)` where `point \in EF^n`.
    ///
    /// # Arguments
    ///
    /// * `point` - The evaluation point as a [`Point`].
    pub fn eval(&self, point: &Point<EF>) -> EF {
        match &self.inner {
            MaybePacked::Packed { evals, .. } => evals.eval_packed(point),
            MaybePacked::Unpacked { evals, .. } => evals.eval_ext::<F>(point),
        }
    }

    /// Folds both polynomials by binding the first variable to a challenge.
    ///
    /// This is the core operation of each sumcheck round. After receiving a challenge `r`,
    /// we reduce the polynomial from `n` variables to `n-1` variables by setting `X_1 = r`.
    ///
    /// # Mathematical Operation
    ///
    /// For a multilinear polynomial `p(X_1, X_2, ..., X_n)`:
    ///
    /// ```text
    /// p'(X_2, ..., X_n) = p(r, X_2, ..., X_n)
    ///                   = p(0, X_2, ..., X_n) + r * (p(1, X_2, ..., X_n) - p(0, X_2, ..., X_n))
    /// ```
    ///
    /// This linear interpolation is applied independently to both `evals` and `weights`.
    ///
    /// # Arguments
    ///
    /// * `r` - The verifier's challenge for this round.
    fn compress(&mut self, r: EF) {
        match &mut self.inner {
            // Apply folding to both packed polynomials.
            //
            // The compress operation handles SIMD lanes correctly.
            MaybePacked::Packed { evals, weights } => {
                St::fix_var(evals, r);
                St::fix_var(weights, r);
            }
            // Apply folding to both scalar polynomials.
            MaybePacked::Unpacked { evals, weights } => {
                St::fix_var(evals, r);
                St::fix_var(weights, r);
            }
        }
    }

    /// Transitions from packed to scalar mode if the polynomial is small enough.
    ///
    /// This is called after each fold operation to check if we should switch representations.
    /// The transition occurs when the packed representation has only a single element
    /// (i.e., `num_variables() == 0` in the packed view).
    ///
    /// # Transition Condition
    ///
    /// ```text
    /// if packed_num_variables == 0:
    ///     -> Unpack to scalar and switch to Small variant
    /// ```
    ///
    /// # Why Transition?
    ///
    /// When only one packed element remains, SIMD operations become pure overhead:
    /// - No parallelism benefit (only one "lane group" of work)
    /// - Extra unpacking/repacking costs per operation
    ///
    /// Scalar mode eliminates this overhead for the final rounds.
    fn transition(&mut self) {
        if let MaybePacked::Packed { evals, weights } = &mut self.inner {
            // Check if we've folded down to a single packed element.
            let k = evals.num_vars();
            assert_eq!(k, weights.num_vars());

            if k == 0 {
                // Unpack the single packed element into SIMD_WIDTH scalar elements.
                //
                // Extract individual extension field elements from the packed representation.
                let evals =
                    EF::ExtensionPacking::to_ext_iter(evals.as_slice().iter().copied()).collect();
                let weights =
                    EF::ExtensionPacking::to_ext_iter(weights.as_slice().iter().copied()).collect();

                // Replace self with the scalar variant.
                *self = Self::new_unpacked(Poly::new(evals), Poly::new(weights));
            }
        }
    }

    /// Executes one round of the quadratic sumcheck protocol.
    ///
    /// This is the main method that:
    /// 1. Computes the sumcheck polynomial coefficients `(h(0), h(inf))`.
    /// 2. Commits them to the Fiat-Shamir transcript.
    /// 3. Receives a challenge from the verifier.
    /// 4. Folds both polynomials using the challenge.
    /// 5. Updates the running sum.
    ///
    /// # Sumcheck Polynomial
    ///
    /// At each round, we send a univariate quadratic polynomial:
    ///
    /// ```text
    ///     h(X) = h(0) * (1 - X) + h(1) * X + h(inf) * X * (X - 1)
    /// ```
    ///
    /// where:
    /// We only send `(h(0), h(inf))` since `h(1)` is derivable by the verifier.
    ///
    /// # Arguments
    ///
    /// * `sumcheck_data` - Storage for polynomial evaluations sent to verifier.
    /// * `challenger` - Fiat-Shamir challenger for transcript operations.
    /// * `sum` - Current claimed sum (updated after this round).
    /// * `pow_bits` - Proof-of-work difficulty (0 to disable).
    ///
    /// # Returns
    ///
    /// The verifier's challenge `r \in EF` for this round.
    #[instrument(skip_all)]
    pub fn round<Challenger>(
        &mut self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        sum: &mut EF,
        pow_bits: usize,
    ) -> EF
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Step 1: Compute sumcheck polynomial coefficients.
        //
        // The strategy differs based on representation to maximize SIMD utilization.
        let (c0, c_inf) = match &self.inner {
            MaybePacked::Packed { evals, weights } => {
                // Compute coefficients using packed arithmetic.
                // Each operation processes SIMD_WIDTH elements in parallel

                let (c0, c_inf) = St::sumcheck_coefficients(evals.as_slice(), weights.as_slice());

                // Horizontal reduction: sum across all SIMD lanes to get scalar result.
                //
                // The packed computation gives us one result per lane.
                // We need the sum across all lanes as the final coefficient.
                (
                    EF::ExtensionPacking::to_ext_iter([c0]).sum(),
                    EF::ExtensionPacking::to_ext_iter([c_inf]).sum(),
                )
            }
            MaybePacked::Unpacked { evals, weights } => {
                // Compute coefficients directly on scalar elements.
                St::sumcheck_coefficients(evals.as_slice(), weights.as_slice())
            }
        };

        // Step 2-4: Commit to transcript, do PoW, and receive challenge.
        let r = sumcheck_data.observe_and_sample(challenger, c0, c_inf, pow_bits);

        // Step 5: Fold both polynomials using the challenge.
        self.compress(r);

        // Step 6: Update the claimed sum.
        //
        // h(r) = h(0)*(1-r) + h(1)*r + h(inf)*r*(r-1)
        // where h(1) = claimed_sum - h(0).
        *sum = extrapolate_01inf(c0, *sum - c0, c_inf, r);

        // Sanity check: the updated sum should equal the inner product after folding.
        debug_assert_eq!(*sum, self.dot_product());

        // Step 7: Check if we should transition to scalar mode.
        //
        // After folding, the polynomial may be small enough that scalar operations
        // are more efficient than packed operations.
        self.transition();

        r
    }

    /// Extracts the evaluation polynomial as a scalar [`Poly`].
    ///
    /// This unpacks the evaluations if in packed mode.
    ///
    /// # Returns
    ///
    /// A copy of the evaluations in scalar extension field format.
    pub fn evals(&self) -> Poly<EF> {
        match &self.inner {
            MaybePacked::Packed { evals, .. } => Poly::new(
                EF::ExtensionPacking::to_ext_iter(evals.as_slice().iter().copied()).collect(),
            ),
            MaybePacked::Unpacked { evals, .. } => evals.clone(),
        }
    }

    /// Incorporates new constraints into the weight polynomial.
    ///
    /// This is used when additional constraints need to be folded into the sumcheck
    /// after initial construction (e.g., from STIR challenges).
    ///
    /// # Arguments
    ///
    /// * `sum` - Running sum to update with new constraint contributions.
    /// * `constraint` - The constraint to combine into weights.
    pub fn combine(&mut self, sum: &mut EF, constraint: &Constraint<F, EF>) {
        match &mut self.inner {
            MaybePacked::Packed { weights, .. } => {
                constraint.combine_packed(weights, sum);
            }
            MaybePacked::Unpacked { weights, .. } => {
                constraint.combine(weights, sum);
            }
        }
    }

    /// Computes the dot product of evaluations and weights.
    ///
    /// This computes:
    ///
    /// ```text
    ///     \sum_{x \in \{0,1\}^n} evals(x) * weights(x)
    /// ```
    ///
    /// which should equal the current claimed sum at any point in the protocol.
    ///
    /// # Returns
    ///
    /// The dot product of `evals` and `weights`.
    pub fn dot_product(&self) -> EF {
        match &self.inner {
            MaybePacked::Packed { evals, weights } => {
                // Compute packed dot product (SIMD parallel multiply-accumulate).
                let sum_packed = dot_product(evals.iter().copied(), weights.iter().copied());

                // Horizontal sum to reduce packed result to scalar.
                EF::ExtensionPacking::to_ext_iter([sum_packed]).sum()
            }
            MaybePacked::Unpacked { evals, weights } => {
                // Direct scalar dot product.
                dot_product(evals.iter().copied(), weights.iter().copied())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::sumcheck::strategy::{PrefixSumcheck, sumcheck_coefficients_prefix};

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type TestChallenger = DuplexChallenger<F, Perm, 16, 8>;

    /// Creates a test challenger with a deterministic seed.
    fn make_challenger() -> TestChallenger {
        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(42));
        DuplexChallenger::new(perm)
    }

    #[test]
    fn test_num_variables_small_variant() {
        // Create a Small variant with 3 variables (8 evaluations).
        let evals = Poly::new(vec![EF::ONE; 8]);
        let weights = Poly::new(vec![EF::TWO; 8]);

        // Force Small variant by using new_unpacked directly.
        let poly = ProductPolynomial::<F, EF, PrefixSumcheck>::new_unpacked(evals, weights);

        // The logical number of variables should be 3 (since 2^3 = 8).
        assert_eq!(poly.num_vars(), 3);
    }

    #[test]
    fn test_dot_product_manual_calculation() {
        // Create Small variant with known values and verify dot product.
        //
        // dot_product = Σ_i evals[i] * weights[i]
        //             = e0*w0 + e1*w1 + e2*w2 + e3*w3
        let e0 = EF::from_u64(1);
        let e1 = EF::from_u64(2);
        let e2 = EF::from_u64(3);
        let e3 = EF::from_u64(4);
        let w0 = EF::from_u64(5);
        let w1 = EF::from_u64(6);
        let w2 = EF::from_u64(7);
        let w3 = EF::from_u64(8);

        let evals = Poly::new(vec![e0, e1, e2, e3]);
        let weights = Poly::new(vec![w0, w1, w2, w3]);

        let poly = ProductPolynomial::<F, EF, PrefixSumcheck>::new_unpacked(evals, weights);

        // dot_product = e0*w0 + e1*w1 + e2*w2 + e3*w3
        let expected = e0 * w0 + e1 * w1 + e2 * w2 + e3 * w3;
        assert_eq!(poly.dot_product(), expected);
    }

    #[test]
    fn test_sumcheck_coefficients_manual_calculation() {
        // Test the sumcheck coefficient computation with manual verification.
        //
        // For a 1-variable polynomial (2 evaluations):
        //   evals   = [e0, e1] where f(0) = e0, f(1) = e1
        //   weights = [w0, w1] where g(0) = w0, g(1) = w1
        //
        // sumcheck_coefficients returns (h(0), h(inf)) where:
        //   h(0)   = f(0) * g(0)     = e0 * w0
        //   h(inf) = (e1-e0)*(w1-w0)   (leading coefficient)
        let e0 = EF::from_u64(3);
        let e1 = EF::from_u64(7);
        let w0 = EF::from_u64(2);
        let w1 = EF::from_u64(5);

        let evals = Poly::new(vec![e0, e1]);
        let weights = Poly::new(vec![w0, w1]);

        let (h0, h_inf) = sumcheck_coefficients_prefix(evals.as_slice(), weights.as_slice());

        // h(0) = e0 * w0
        let expected_h0 = e0 * w0;
        assert_eq!(h0, expected_h0);

        // h(inf) = (e1 - e0) * (w1 - w0)  (leading coefficient)
        let expected_h_inf = (e1 - e0) * (w1 - w0);
        assert_eq!(h_inf, expected_h_inf);

        // Verify consistency: h(0) + h(1) should equal the claimed sum.
        // h(0) = e0 * w0
        // h(1) = e1 * w1
        // sum = e0*w0 + e1*w1
        let h_1 = e1 * w1;
        let sum = e0 * w0 + e1 * w1;
        assert_eq!(h0 + h_1, sum);
    }

    #[test]
    fn test_compress_manual_calculation() {
        // Test the compress (folding) operation with manual verification.
        //
        // Initial state: 2-variable polynomial (4 evaluations)
        //   evals   = [e0, e1, e2, e3] representing f(x0, x1)
        //   weights = [w0, w1, w2, w3] representing g(x0, x1)
        //
        // Memory layout:
        //   f(0,0) = e0, f(0,1) = e1 (lo half, x0 = 0)
        //   f(1,0) = e2, f(1,1) = e3 (hi half, x0 = 1)
        //
        // Folding binds x0 to challenge r:
        //   f'(x1) = f(0,x1) + r * (f(1,x1) - f(0,x1))
        //
        // So:
        //   e'0 = e0 + r * (e2 - e0)
        //   e'1 = e1 + r * (e3 - e1)
        //   w'0 = w0 + r * (w2 - w0)
        //   w'1 = w1 + r * (w3 - w1)
        let e0 = EF::from_u64(1);
        let e1 = EF::from_u64(2);
        let e2 = EF::from_u64(5);
        let e3 = EF::from_u64(8);
        let w0 = EF::from_u64(3);
        let w1 = EF::from_u64(4);
        let w2 = EF::from_u64(6);
        let w3 = EF::from_u64(7);

        let evals = Poly::new(vec![e0, e1, e2, e3]);
        let weights = Poly::new(vec![w0, w1, w2, w3]);

        let mut poly = ProductPolynomial::<F, EF, PrefixSumcheck>::new_unpacked(evals, weights);

        // Initial dot product: sum = e0*w0 + e1*w1 + e2*w2 + e3*w3
        let initial_sum = e0 * w0 + e1 * w1 + e2 * w2 + e3 * w3;
        assert_eq!(poly.dot_product(), initial_sum);

        // Fold with challenge r.
        let r = EF::from_u64(2);
        poly.compress(r);

        let folded_evals = poly.evals();

        // e'0 = e0 + r * (e2 - e0)
        // e'1 = e1 + r * (e3 - e1)
        let expected_e0 = e0 + r * (e2 - e0);
        let expected_e1 = e1 + r * (e3 - e1);

        assert_eq!(folded_evals.as_slice(), &[expected_e0, expected_e1]);

        // After folding, dot_product equals h(r) where h is the sumcheck polynomial:
        //   h(X) = h(0) + b*X + a*X^2
        //   h(0)  = e0*w0 + e1*w1
        //   h(inf) = a = (e2-e0)*(w2-w0) + (e3-e1)*(w3-w1)  (leading coefficient)
        //   h(1)  = e2*w2 + e3*w3
        //   b     = h(1) - h(0) - a
        //   h(r)  = h(0) + b*r + a*r^2
        let h_0 = e0 * w0 + e1 * w1;
        let a = (e2 - e0) * (w2 - w0) + (e3 - e1) * (w3 - w1);
        let h_1 = e2 * w2 + e3 * w3;
        let b = h_1 - h_0 - a;
        let h_r = h_0 + b * r + a * r.square();

        assert_eq!(poly.dot_product(), h_r);
    }

    #[test]
    fn test_eval_multilinear_interpolation() {
        // Test eval() with non-boolean points using multilinear interpolation.
        //
        // For a 2-variable polynomial f(x0, x1):
        //   f(x0, x1) = f(0,0)*(1-x0)*(1-x1) + f(0,1)*(1-x0)*x1
        //             + f(1,0)*x0*(1-x1)     + f(1,1)*x0*x1
        //
        // With evals = [e0, e1, e2, e3]:
        //   f(0,0) = e0, f(0,1) = e1, f(1,0) = e2, f(1,1) = e3
        let e0 = EF::from_u64(2);
        let e1 = EF::from_u64(5);
        let e2 = EF::from_u64(3);
        let e3 = EF::from_u64(11);

        let evals = Poly::new(vec![e0, e1, e2, e3]);
        let weights = Poly::new(vec![EF::ONE; 4]);

        let poly = ProductPolynomial::<F, EF, PrefixSumcheck>::new_unpacked(evals, weights);

        // Evaluate at (x0, x1):
        //   f(x0, x1) = e0*(1-x0)*(1-x1) + e1*(1-x0)*x1 + e2*x0*(1-x1) + e3*x0*x1
        let x0 = EF::from_u64(3);
        let x1 = EF::from_u64(4);
        let point = Point::new(vec![x0, x1]);

        let one = EF::ONE;
        let expected = e0 * (one - x0) * (one - x1)
            + e1 * (one - x0) * x1
            + e2 * x0 * (one - x1)
            + e3 * x0 * x1;

        assert_eq!(poly.eval(&point), expected);
    }

    #[test]
    fn test_transition_from_packed_to_small() {
        // Create a Packed variant that will transition to Small after sufficient folding.
        //
        // The SIMD threshold is log_2(F::Packing::WIDTH).
        // We need a polynomial large enough to start in Packed mode.
        type EP = <EF as ExtensionField<F>>::ExtensionPacking;

        let simd_width = <F as Field>::Packing::WIDTH;
        let simd_log = log2_strict_usize(simd_width);

        // Start with simd_log + 2 variables (e.g., if simd_width=16, start with 6 vars = 64 evals).
        // This gives us 4 packed elements initially (2 stored variables).
        let num_vars = simd_log + 2;
        let num_evals = 1 << num_vars;

        // Create scalar evaluations and pack them
        let evals_scalar = vec![EF::ONE; num_evals];
        let weights_scalar = vec![EF::ONE; num_evals];

        let packed_evals = Poly::new(
            evals_scalar
                .chunks(simd_width)
                .map(EP::from_ext_slice)
                .collect(),
        );
        let packed_weights = Poly::new(
            weights_scalar
                .chunks(simd_width)
                .map(EP::from_ext_slice)
                .collect(),
        );

        let mut poly =
            ProductPolynomial::<F, EF, PrefixSumcheck>::new_packed(packed_evals, packed_weights);

        // Initially should be Packed with correct internal structure.
        match &poly.inner {
            MaybePacked::Packed {
                evals: packed_evals,
                weights: packed_weights,
            } => {
                // Should have num_evals / simd_width = 4 packed elements.
                let expected_packed_len = num_evals / simd_width;
                assert_eq!(packed_evals.num_evals(), expected_packed_len);
                assert_eq!(packed_weights.num_evals(), expected_packed_len);
            }
            MaybePacked::Unpacked { .. } => {
                panic!("Expected Packed variant initially");
            }
        }
        assert_eq!(poly.num_vars(), num_vars);

        // Fold twice to reduce to simd_log variables (threshold for transition).
        for _ in 0..2 {
            let challenge = EF::from_u64(7);
            poly.compress(challenge);
            poly.transition();
        }

        // After two folds: simd_log variables, which triggers transition to Small.
        match &poly.inner {
            MaybePacked::Unpacked { evals, weights } => {
                // Should have 2^simd_log = simd_width scalar elements.
                assert_eq!(evals.num_evals(), simd_width);
                assert_eq!(weights.num_evals(), simd_width);
            }
            MaybePacked::Packed { .. } => {
                panic!("Expected Small variant after transition");
            }
        }
        assert_eq!(poly.num_vars(), simd_log);
    }

    #[test]
    fn test_new_packed_with_single_element_transitions() {
        // If we create a Packed variant with just 1 packed element (0 stored variables),
        // it should immediately transition to Small.
        //
        // This happens when packed evals has exactly 1 element.
        type EP = <EF as ExtensionField<F>>::ExtensionPacking;

        // Get the actual SIMD width to create properly sized arrays.
        let simd_width = <F as Field>::Packing::WIDTH;

        // Create a single packed element containing simd_width extension field elements.
        let evals_scalar: Vec<EF> = (0..simd_width).map(|i| EF::from_u64(i as u64)).collect();
        let weights_scalar: Vec<EF> = (0..simd_width)
            .map(|i| EF::from_u64(100 + i as u64))
            .collect();

        let evals = Poly::new(vec![EP::from_ext_slice(&evals_scalar)]);
        let weights = Poly::new(vec![EP::from_ext_slice(&weights_scalar)]);

        let poly = ProductPolynomial::<F, EF, PrefixSumcheck>::new_packed(evals, weights);

        // Should have transitioned to Small with correct values.
        match &poly.inner {
            MaybePacked::Unpacked {
                evals: small_evals,
                weights: small_weights,
            } => {
                // Verify the unpacked values match the original scalars.
                assert_eq!(small_evals.as_slice(), &evals_scalar);
                assert_eq!(small_weights.as_slice(), &weights_scalar);
            }
            MaybePacked::Packed { .. } => {
                panic!("Expected Small variant after transition from single packed element");
            }
        }

        // Should have log_2(simd_width) variables.
        assert_eq!(poly.num_vars(), log2_strict_usize(simd_width));
    }

    #[test]
    fn test_round_updates_sum_correctly() {
        // Test the round() function which is the core sumcheck protocol.
        //
        // The round function should:
        // 1. Compute sumcheck coefficients (h(0), h(inf))
        // 2. Update the claimed sum to h(r) where r is the challenge
        // 3. Fold both polynomials
        // 4. Return the challenge r
        //
        // Verify: after round(), dot_product() == updated sum
        let e0 = EF::from_u64(2);
        let e1 = EF::from_u64(5);
        let e2 = EF::from_u64(3);
        let e3 = EF::from_u64(7);
        let w0 = EF::from_u64(1);
        let w1 = EF::from_u64(4);
        let w2 = EF::from_u64(2);
        let w3 = EF::from_u64(6);

        let evals = Poly::new(vec![e0, e1, e2, e3]);
        let weights = Poly::new(vec![w0, w1, w2, w3]);

        let mut poly = ProductPolynomial::<F, EF, PrefixSumcheck>::new_unpacked(evals, weights);

        // Initial sum = e0*w0 + e1*w1 + e2*w2 + e3*w3
        let mut sum = e0 * w0 + e1 * w1 + e2 * w2 + e3 * w3;
        assert_eq!(poly.dot_product(), sum);

        // Perform one round of sumcheck.
        let mut sumcheck_data = SumcheckData::default();
        let mut challenger = make_challenger();

        let _r = poly.round(&mut sumcheck_data, &mut challenger, &mut sum, 0);

        // After round:
        // 1. sum should be updated to h(r)
        // 2. dot_product should equal the updated sum
        assert_eq!(poly.dot_product(), sum);

        // Verify sumcheck_data was populated with polynomial evaluations.
        assert!(!sumcheck_data.polynomial_evaluations.is_empty());
    }

    #[test]
    fn test_round_multiple_rounds() {
        // Test multiple rounds of sumcheck to verify protocol consistency.
        //
        // After each round:
        // - Number of variables decreases by 1
        // - dot_product() == sum
        let mut rng = SmallRng::seed_from_u64(123);
        let num_vars = 4;
        let num_evals = 1 << num_vars;

        let evals: Vec<EF> = (0..num_evals).map(|_| EF::from_u64(rng.random())).collect();
        let weights: Vec<EF> = (0..num_evals).map(|_| EF::from_u64(rng.random())).collect();

        let mut poly = ProductPolynomial::<F, EF, PrefixSumcheck>::new_unpacked(
            Poly::new(evals),
            Poly::new(weights),
        );

        let mut sum = poly.dot_product();
        let mut sumcheck_data = SumcheckData::default();
        let mut challenger = make_challenger();

        // Perform all rounds except the last (need at least 1 evaluation left).
        for expected_vars in (1..=num_vars).rev() {
            assert_eq!(poly.num_vars(), expected_vars);

            let _ = poly.round(&mut sumcheck_data, &mut challenger, &mut sum, 0);

            // Invariant: dot_product == sum after each round.
            assert_eq!(poly.dot_product(), sum);
        }

        // After all rounds, should have 0 variables (1 evaluation).
        assert_eq!(poly.num_vars(), 0);
    }

    #[test]
    fn test_dot_product_packed_matches_scalar() {
        // Verify that Packed and Small variants compute the same dot product.
        type EP = <EF as ExtensionField<F>>::ExtensionPacking;

        let simd_width = <F as Field>::Packing::WIDTH;
        let num_vars = log2_strict_usize(simd_width) + 1;
        let num_evals = 1 << num_vars;

        let mut rng = SmallRng::seed_from_u64(456);
        let evals_scalar: Vec<EF> = (0..num_evals).map(|_| EF::from_u64(rng.random())).collect();
        let weights_scalar: Vec<EF> = (0..num_evals).map(|_| EF::from_u64(rng.random())).collect();

        // Compute expected dot product manually.
        let expected: EF = evals_scalar
            .iter()
            .zip(weights_scalar.iter())
            .map(|(&e, &w)| e * w)
            .sum();

        // Create Small variant and verify.
        let small_poly = ProductPolynomial::<F, EF, PrefixSumcheck>::new_unpacked(
            Poly::new(evals_scalar.clone()),
            Poly::new(weights_scalar.clone()),
        );
        assert_eq!(small_poly.dot_product(), expected);

        // Create Packed variant and verify.
        let packed_evals = Poly::new(
            evals_scalar
                .chunks(simd_width)
                .map(EP::from_ext_slice)
                .collect(),
        );
        let packed_weights = Poly::new(
            weights_scalar
                .chunks(simd_width)
                .map(EP::from_ext_slice)
                .collect(),
        );

        let packed_poly =
            ProductPolynomial::<F, EF, PrefixSumcheck>::new_packed(packed_evals, packed_weights);
        assert_eq!(packed_poly.dot_product(), expected);
    }

    #[test]
    fn test_evals_extraction() {
        // Test that evals() returns correct values for both variants.
        let e0 = EF::from_u64(10);
        let e1 = EF::from_u64(20);
        let e2 = EF::from_u64(30);
        let e3 = EF::from_u64(40);

        let evals = Poly::new(vec![e0, e1, e2, e3]);
        let weights = Poly::new(vec![EF::ONE; 4]);

        let poly = ProductPolynomial::<F, EF, PrefixSumcheck>::new_unpacked(evals, weights);

        let extracted = poly.evals();
        assert_eq!(extracted.as_slice(), &[e0, e1, e2, e3]);
    }

    #[test]
    fn test_combine_updates_weights_and_sum() {
        // Test that combine() correctly incorporates new constraints.
        //
        // The combine function should:
        // 1. Update the weight polynomial with new constraint contributions
        // 2. Update the running sum accordingly
        use crate::constraints::Constraint;
        use crate::constraints::statement::EqStatement;

        let num_vars = 2;
        let evals = Poly::new(vec![EF::ONE; 4]);
        let weights = Poly::new(vec![EF::ONE; 4]);

        let mut poly =
            ProductPolynomial::<F, EF, PrefixSumcheck>::new_unpacked(evals.clone(), weights);

        // Initial state: dot_product = 4 (all ones)
        let initial_dot = poly.dot_product();
        assert_eq!(initial_dot, EF::from_u64(4));

        // Create an EqStatement with one constraint.
        let mut eq_statement = EqStatement::initialize(num_vars);
        let point = Point::new(vec![EF::from_u64(2), EF::from_u64(3)]);
        let eval = evals.eval_ext::<F>(&point);
        eq_statement.add_evaluated_constraint(point, eval);

        // Create constraint with the eq_statement.
        let challenge = EF::from_u64(7);
        let constraint = Constraint::<F, EF>::new_eq_only(challenge, eq_statement);

        let mut sum = poly.dot_product();
        poly.combine(&mut sum, &constraint);

        // After combining, the weights may have changed.
        // The exact behavior depends on the constraint implementation.
        // We verify the invariant: dot_product reflects the combined state.
        assert_eq!(poly.dot_product(), sum);
    }

    #[test]
    fn test_eval_at_boolean_points() {
        // Test eval() at boolean points (0 and 1 coordinates).
        //
        // For multilinear polynomial over boolean hypercube,
        // eval at boolean point should return the stored evaluation.
        let e00 = EF::from_u64(1);
        let e01 = EF::from_u64(2);
        let e10 = EF::from_u64(3);
        let e11 = EF::from_u64(4);

        let evals = Poly::new(vec![e00, e01, e10, e11]);
        let weights = Poly::new(vec![EF::ONE; 4]);

        let poly = ProductPolynomial::<F, EF, PrefixSumcheck>::new_unpacked(evals, weights);

        // Evaluate at (0, 0) -> should return e00
        let point_00 = Point::new(vec![EF::ZERO, EF::ZERO]);
        assert_eq!(poly.eval(&point_00), e00);

        // Evaluate at (0, 1) -> should return e01
        let point_01 = Point::new(vec![EF::ZERO, EF::ONE]);
        assert_eq!(poly.eval(&point_01), e01);

        // Evaluate at (1, 0) -> should return e10
        let point_10 = Point::new(vec![EF::ONE, EF::ZERO]);
        assert_eq!(poly.eval(&point_10), e10);

        // Evaluate at (1, 1) -> should return e11
        let point_11 = Point::new(vec![EF::ONE, EF::ONE]);
        assert_eq!(poly.eval(&point_11), e11);
    }

    proptest! {
        /// Verify that dot_product is consistent across random inputs.
        #[test]
        fn prop_dot_product_consistency(seed in 0u64..1000) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_vars = 3;
            let num_evals = 1 << num_vars;

            let evals: Vec<EF> = (0..num_evals)
                .map(|_| EF::from_u64(u64::from(rng.random::<u32>())))
                .collect();
            let weights: Vec<EF> = (0..num_evals)
                .map(|_| EF::from_u64(u64::from(rng.random::<u32>())))
                .collect();

            let poly = ProductPolynomial::<F, EF,PrefixSumcheck>::new_unpacked(
                Poly::new(evals.clone()),
                Poly::new(weights.clone()),
            );

            // Manual computation
            let expected: EF = evals
                .iter()
                .zip(weights.iter())
                .map(|(&e, &w)| e * w)
                .sum();

            prop_assert_eq!(poly.dot_product(), expected);
        }

        /// Verify that compress maintains the sumcheck invariant.
        #[test]
        fn prop_compress_maintains_invariant(seed in 0u64..1000, challenge_val in 1u64..100) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_vars = 3;
            let num_evals = 1 << num_vars;

            let evals: Vec<EF> = (0..num_evals)
                .map(|_| EF::from_u64(u64::from(rng.random::<u32>())))
                .collect();
            let weights: Vec<EF> = (0..num_evals)
                .map(|_| EF::from_u64(u64::from(rng.random::<u32>())))
                .collect();

            let mut poly = ProductPolynomial::<F, EF,PrefixSumcheck>::new_unpacked(
                Poly::new(evals),
                Poly::new(weights),
            );

            // Compute sumcheck coefficients before folding.
            // Returns (h(0), h(inf)) where h is the univariate
            // polynomial h(X) = sum_{b in {0,1}^{n-1}} f(X, b) * w(X, b).
            let (c0, c_inf) = match &poly.inner {
                MaybePacked::Unpacked {
                    evals: small_evals,
                    weights: small_weights,
                } => sumcheck_coefficients_prefix(small_evals.as_slice(), small_weights.as_slice()),
                MaybePacked::Packed { .. } => unreachable!(),
            };

            // The sumcheck relation: h(0) + h(1) = claimed_sum
            // So h(1) = claimed_sum - h(0) = claimed_sum - c0
            let initial_sum = poly.dot_product();
            let c1 = initial_sum - c0;

            // Fold with challenge r.
            let r = EF::from_u64(challenge_val);
            poly.compress(r);

            // Use interpolation to compute h(r) from h(0), h(1), h(inf).
            let h_r = extrapolate_01inf(c0, c1, c_inf, r);

            // After folding, dot_product should equal h(r).
            prop_assert_eq!(poly.dot_product(), h_r);
        }

        /// Verify that round() maintains the sumcheck invariant.
        #[test]
        fn prop_round_maintains_invariant(seed in 0u64..1000) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_vars = 4;
            let num_evals = 1 << num_vars;

            let evals: Vec<EF> = (0..num_evals)
                .map(|_| EF::from_u64(u64::from(rng.random::<u32>())))
                .collect();
            let weights: Vec<EF> = (0..num_evals)
                .map(|_| EF::from_u64(u64::from(rng.random::<u32>())))
                .collect();

            let mut poly = ProductPolynomial::<F, EF,PrefixSumcheck>::new_unpacked(
                Poly::new(evals),
                Poly::new(weights),
            );

            let mut sum = poly.dot_product();
            let mut sumcheck_data = SumcheckData::default();

            // Use seed to create challenger for reproducibility.
            let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(seed + 1000));
            let mut challenger: TestChallenger = DuplexChallenger::new(perm);

            // Perform one round.
            let _ = poly.round(&mut sumcheck_data, &mut challenger, &mut sum, 0);

            // Invariant: dot_product == sum after round.
            prop_assert_eq!(poly.dot_product(), sum);
        }
    }
}
