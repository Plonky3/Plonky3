//! Sumcheck prover: constructs and executes the sumcheck protocol for multilinear polynomials.

use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_multilinear_util::evals::Poly;
use p3_multilinear_util::multilinear::Point;
use p3_util::log2_strict_usize;

use crate::constraints::Constraint;
use crate::constraints::statement::EqStatement;
use crate::constraints::statement::initial::{InitialStatement, InitialStatementInner};
use crate::sumcheck::lagrange::lagrange_weights_012_multi;
use crate::sumcheck::product_polynomial::{ProductPolynomial, sumcheck_coefficients_cross};
use crate::sumcheck::svo::SplitEq;
use crate::sumcheck::{SumcheckData, extrapolate_012};

/// Prover state for the sumcheck protocol over a multilinear polynomial.
///
/// Holds the partially-folded polynomial pair `(f, w)` and the current claimed sum.
///
/// # Invariant
///
/// At every point during the protocol:
///
/// ```text
/// sum == sum_{x in {0,1}^n} f(x) * w(x)
/// ```
///
/// where `n` is the number of remaining unbound variables.
/// It decreases by one per round as variables are bound to verifier challenges.
#[derive(Debug, Clone)]
pub struct SumcheckProver<F: Field, EF: ExtensionField<F>> {
    /// Paired evaluation and weight polynomials for the quadratic sumcheck.
    ///
    /// Stores both `f(x)` and `w(x)` in either SIMD-packed or scalar format.
    /// The format is chosen automatically based on polynomial size.
    pub poly: ProductPolynomial<F, EF>,

    /// Current claimed sum for the remaining variables.
    ///
    /// Tracks `sum_{x in {0,1}^n} f(x) * w(x)`.
    ///
    /// After each round binding variable `X_i` to challenge `r_i`, updated via:
    ///
    /// ```text
    /// sum := h(r_i)  where  h(X) = c_0 + c_1 * X + c_2 * X^2
    /// ```
    pub sum: EF,
}

impl<F, EF> SumcheckProver<F, EF>
where
    F: Field + Ord,
    EF: ExtensionField<F>,
{
    /// Constructs a sumcheck instance using the classic approach with scalar arithmetic.
    ///
    /// Fallback path for small polynomials where SIMD packing is not beneficial.
    ///
    /// # Algorithm
    ///
    /// 1. Sample a batching challenge `alpha` from the Fiat-Shamir transcript.
    /// 2. Build `w(x)` from equality constraints via random linear combination.
    /// 3. Compute `(c_0, c_2)` for the first round.
    /// 4. Commit to the transcript, perform proof-of-work, receive challenge `r`.
    /// 5. Fold both `f` and `w` by binding the first variable to `r`.
    /// 6. Run the remaining rounds via the standard one-variable-per-round protocol.
    ///
    /// # Returns
    ///
    /// - The partially-folded prover state.
    /// - The verifier challenges `(r_1, ..., r_{folding_factor})`.
    #[tracing::instrument(skip_all)]
    fn new_classic_small<Challenger>(
        poly: &Poly<F>,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        statement: &EqStatement<EF>,
    ) -> (Self, Point<EF>)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Sample a batching challenge for combining multiple equality constraints
        // into a single weight polynomial via random linear combination.
        let alpha: EF = challenger.sample_algebra_element();

        let k = poly.num_vars();

        // Initialize a zero weight polynomial with the same number of variables.
        let mut weights = Poly::zero(k);

        // The claimed sum starts at zero.
        // It will be accumulated from the equality constraints below.
        let mut sum = EF::ZERO;

        // Populate the weight polynomial from equality constraints.
        // INITIALIZED=false means the weights buffer is freshly zeroed,
        // so we write directly rather than accumulate.
        statement.combine_hypercube::<F, false>(&mut weights, &mut sum, alpha);

        // Compute the constant and quadratic coefficients of the round polynomial:
        //   c_0 = h(0) = sum_{b in {0,1}^{k-1}} f(0, b) * w(0, b)
        //   c_2 = h(2) = sum_{b in {0,1}^{k-1}} f(2, b) * w(2, b)
        //
        // The linear coefficient c_1 is derived by the verifier from:
        //   h(0) + h(1) = claimed_sum
        let (c0, c2) = sumcheck_coefficients_cross(poly, &weights);

        // Commit (c_0, c_2) to the Fiat-Shamir transcript.
        // Perform any required proof-of-work grinding.
        // Receive the verifier's challenge `r`.
        let r = sumcheck_data.observe_and_sample(challenger, c0, c2, pow_bits);

        // Fold the weight polynomial by binding its first variable to `r`:
        //   w'(x_2, ..., x_k) = w(0, x_2, ...) + r * (w(1, ...) - w(0, ...))
        weights.fix_lo_var_mut(r);

        // Fold the evaluation polynomial similarly.
        // Promote base field evaluations into extension field elements during the fold.
        let evals = poly.fix_lo_var(r);

        // Update the claimed sum to h(r) using quadratic extrapolation.
        // h(1) = sum - c_0, since h(0) + h(1) = sum.
        sum = extrapolate_012(c0, sum - c0, c2, r);

        // Wrap the folded polynomials into a paired polynomial (scalar variant).
        let mut poly = ProductPolynomial::<F, EF>::new_small(evals, weights);

        // Verify the core sumcheck invariant.
        debug_assert_eq!(poly.dot_product(), sum);

        // Collect all verifier challenges.
        // The first challenge `r` was computed above.
        // The remaining come from subsequent rounds.
        let rs = core::iter::once(r)
            .chain(
                (1..folding_factor)
                    .map(|_| poly.round(sumcheck_data, challenger, &mut sum, pow_bits)),
            )
            .collect();

        (Self { poly, sum }, Point::new(rs))
    }

    /// Constructs a sumcheck instance using the classic approach with SIMD-packed arithmetic.
    ///
    /// Primary path for large polynomials.
    /// Processes multiple evaluations in parallel using SIMD lanes.
    ///
    /// # Algorithm
    ///
    /// Same as the scalar classic approach, except:
    /// - The weight polynomial uses packed entries (fewer elements, each holding SIMD_WIDTH values).
    /// - The evaluation polynomial is packed into SIMD lanes before computing coefficients.
    /// - A horizontal reduction (sum across SIMD lanes) produces scalar `(c_0, c_2)`.
    ///
    /// # Returns
    ///
    /// - The partially-folded prover state.
    /// - The verifier challenges `(r_1, ..., r_{folding_factor})`.
    #[tracing::instrument(skip_all)]
    fn new_classic_packed<Challenger>(
        poly: &Poly<F>,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        statement: &EqStatement<EF>,
    ) -> (Self, Point<EF>)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Sample a batching challenge for combining multiple constraints.
        let alpha: EF = challenger.sample_algebra_element();

        let k = poly.num_vars();

        // The packed representation absorbs the last k_pack variables into SIMD lanes.
        // So the weight polynomial needs only 2^{k - k_pack} packed entries.
        let k_pack = log2_strict_usize(F::Packing::WIDTH);

        // Initialize a zero weight polynomial in packed representation.
        let mut weights = Poly::zero(k - k_pack);
        let mut sum = EF::ZERO;

        // Populate packed weights from equality constraints.
        statement.combine_hypercube_packed::<F, false>(&mut weights, &mut sum, alpha);

        // Pack the base-field evaluations into SIMD lanes.
        // Each packed element holds SIMD_WIDTH consecutive evaluations.
        let poly_packed = Poly::new(F::Packing::pack_slice(poly.as_slice()).to_vec());

        // Compute sumcheck coefficients in packed arithmetic.
        // The result is still in packed form (one value per SIMD lane).
        let (c0, c2) = sumcheck_coefficients_cross(&poly_packed, &weights);

        // Sum across all SIMD lanes to produce scalar coefficients.
        // The sumcheck polynomial is a sum over ALL evaluation points, not per-lane.
        let c0 = EF::ExtensionPacking::to_ext_iter([c0]).sum();
        let c2 = EF::ExtensionPacking::to_ext_iter([c2]).sum();

        // Commit (c_0, c_2) to the transcript and receive the challenge.
        let r = sumcheck_data.observe_and_sample(challenger, c0, c2, pow_bits);

        // Fold the packed weight polynomial by binding the first variable to `r`.
        weights.fix_lo_var_mut(r);

        // Fold the base-field evaluations and promote into packed extension field form.
        // Uses compress_lo_to_packed with a single-variable point to fold and pack.
        let evals = poly.compress_lo_to_packed(&Point::new(alloc::vec![r]), EF::ONE);

        // Update the claimed sum to h(r) via quadratic extrapolation.
        sum = extrapolate_012(c0, sum - c0, c2, r);

        // Wrap into a paired polynomial (packed variant).
        // The constructor checks whether the data is small enough for scalar mode.
        let mut poly = ProductPolynomial::<F, EF>::new_packed(evals, weights);

        // Verify the sumcheck invariant.
        debug_assert_eq!(poly.dot_product(), sum);

        // Collect all verifier challenges.
        let rs = core::iter::once(r)
            .chain(
                (1..folding_factor)
                    .map(|_| poly.round(sumcheck_data, challenger, &mut sum, pow_bits)),
            )
            .collect();

        (Self { poly, sum }, Point::new(rs))
    }

    /// Constructs a sumcheck instance using the Split-Value Optimization (Algorithm 5).
    ///
    /// Used when the sumcheck polynomial has the form:
    ///
    /// ```text
    /// g(X) = sum_j alpha^j * eq(w_j, X) * p(X)
    /// ```
    ///
    /// Exploits the factorization of the equality polynomial
    /// to avoid building the full `2^l`-sized equality table.
    ///
    /// # Algorithm
    ///
    /// For each round `i`:
    ///
    /// 1. Compute Lagrange weights from challenges collected so far.
    /// 2. For each constraint, compute its contribution to `c_0` and `c_2`
    ///    via dot products of accumulators with Lagrange weights, scaled by `alpha^j`.
    /// 3. Commit `(c_0, c_2)` and receive challenge `r_i`.
    /// 4. Update the claimed sum via quadratic extrapolation.
    ///
    /// After all rounds, the polynomial and weights are materialized in packed form.
    ///
    /// # Performance
    ///
    /// Avoids the `O(2^l)` cost of the full equality table.
    /// Uses `O(2^{l/2})` pre-computed accumulators per constraint instead.
    ///
    /// # Panics
    ///
    /// - If the folding factor is 0 or exceeds the number of variables.
    /// - If any constraint has a mismatched number of variables.
    /// - If there are not enough variables for packed representation after folding.
    #[tracing::instrument(skip_all)]
    pub fn new_svo<Challenger>(
        poly: &Poly<F>,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        statements: &[SplitEq<F, EF>],
    ) -> (Self, Point<EF>)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert_ne!(folding_factor, 0);

        // Sample the batching challenge for combining multiple split eq constraints.
        let alpha: EF = challenger.sample_algebra_element();

        let k = poly.num_vars();
        assert!(
            folding_factor <= k,
            "number of rounds must be less than or equal to instance size"
        );

        // All constraints must operate over the same number of variables.
        for statement in statements {
            assert_eq!(statement.num_variables(), k);
        }

        let k_pack = log2_strict_usize(F::Packing::WIDTH);

        // After folding, the remaining polynomial must have enough variables
        // for packed representation.
        assert!(k >= 2 * k_pack + folding_factor);

        // Initialize the claimed sum as the random linear combination:
        //   sum = sum_j alpha^j * eval_j
        let mut sum = statements
            .iter()
            .zip(alpha.powers())
            .map(|(statement, alpha)| statement.eval * alpha)
            .sum::<EF>();

        // Gather the pre-computed accumulator tables for all constraints.
        // Each provides a 2D array indexed by [round][{0, 2}]:
        // - [round][0] stores data for computing c_0
        // - [round][1] stores data for computing c_2
        let accumulators = statements
            .iter()
            .map(SplitEq::accumulators)
            .collect::<Vec<_>>();

        // Collect verifier challenges across all SVO rounds.
        let mut rs = Vec::with_capacity(folding_factor);
        tracing::info_span!("svo rounds").in_scope(|| {
            for round_idx in 0..folding_factor {
                // Initialize round polynomial coefficients to zero.
                let (mut c0, mut c2): (EF, EF) = Default::default();

                // Compute Lagrange interpolation weights from previous challenges.
                // These reconstruct round polynomial coefficients from the
                // pre-computed accumulators without materializing the full equality table.
                let weights = lagrange_weights_012_multi(rs.as_slice());

                // Accumulate each constraint's contribution, scaled by alpha^j.
                for (accumulators, alpha) in accumulators.iter().zip(alpha.powers()) {
                    // Accumulators for computing h(0) and h(2) at this round.
                    let acc0 = &accumulators[round_idx][0];
                    let acc2 = &accumulators[round_idx][1];

                    // Dot product with Lagrange weights reconstructs the coefficient.
                    c0 += alpha
                        * dot_product::<EF, _, _>(acc0.iter().copied(), weights.iter().copied());
                    c2 += alpha
                        * dot_product::<EF, _, _>(acc2.iter().copied(), weights.iter().copied());
                }

                // Commit (c_0, c_2) to the transcript and receive challenge r_i.
                let r = sumcheck_data.observe_and_sample(challenger, c0, c2, pow_bits);

                // Update the claimed sum to h(r) using quadratic extrapolation.
                sum = extrapolate_012(c0, sum - c0, c2, r);

                // Record this round's challenge for the next round's Lagrange weights.
                rs.push(r);
            }
        });

        // Materialize the evaluation polynomial by folding the original base-field
        // evaluations through all collected challenges at once.
        let poly = poly.compress_lo_to_packed(&Point::new(rs.clone()), EF::ONE);

        // Materialize the weight polynomial in packed form by combining all split eq
        // constraints into a single packed weight array.
        let mut weights = Poly::<EF::ExtensionPacking>::zero(poly.num_vars());
        SplitEq::combine_into_packed(statements, weights.as_mut_slice(), alpha, &rs);

        // Wrap into a paired polynomial (packed) for subsequent standard rounds.
        let poly = ProductPolynomial::<F, EF>::new_packed(poly, weights);

        // Verify the sumcheck invariant after materialization.
        debug_assert_eq!(poly.dot_product(), sum);
        (Self { poly, sum }, Point::new(rs))
    }

    /// Entry point: constructs a sumcheck prover from base-field evaluations.
    ///
    /// Dispatches to the appropriate proving strategy based on the statement:
    ///
    /// - **SVO**: For statements with pre-split equality polynomials.
    /// - **Classic packed**: For large polynomials where SIMD is beneficial.
    /// - **Classic scalar**: Fallback for very small instances.
    ///
    /// # Returns
    ///
    /// - The partially-folded prover state, ready for further rounds.
    /// - The verifier challenges `(r_1, ..., r_{folding_factor})`.
    ///
    /// # Panics
    ///
    /// - If the folding factor is 0 or exceeds the number of variables.
    /// - For SVO: if the folding factor does not match the pre-configured split point.
    #[tracing::instrument(skip_all)]
    pub fn from_base_evals<Challenger>(
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        statement: &InitialStatement<F, EF>,
    ) -> (Self, Point<EF>)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let k = statement.num_variables();
        assert_ne!(folding_factor, 0, "number of rounds must be non-zero");
        assert!(
            folding_factor <= k,
            "number of rounds must be less than or equal to instance size"
        );

        // Threshold for choosing packed vs. scalar.
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let poly = &statement.poly;
        match &statement.inner {
            // SVO path: the statement has pre-split equality polynomials.
            InitialStatementInner::Svo { split_eqs, l0 } => {
                // The folding factor must match the pre-configured split point.
                assert_eq!(*l0, folding_factor);
                // Ensure enough variables remain after SVO rounds for packed mode.
                assert!(k > 2 * k_pack + folding_factor);
                Self::new_svo(
                    poly,
                    sumcheck_data,
                    challenger,
                    folding_factor,
                    pow_bits,
                    split_eqs,
                )
            }
            // Classic path: choose packed or scalar based on polynomial size.
            InitialStatementInner::Classic(statement) => {
                if k > k_pack {
                    // Large enough for SIMD packing to be beneficial.
                    Self::new_classic_packed(
                        poly,
                        sumcheck_data,
                        challenger,
                        folding_factor,
                        pow_bits,
                        statement,
                    )
                } else {
                    // Too small for SIMD; use scalar arithmetic.
                    Self::new_classic_small(
                        poly,
                        sumcheck_data,
                        challenger,
                        folding_factor,
                        pow_bits,
                        statement,
                    )
                }
            }
        }
    }

    /// Returns the number of remaining (unbound) variables.
    ///
    /// Decreases by one per round.
    /// Reaches zero when the polynomial is fully evaluated at a single point.
    pub fn num_variables(&self) -> usize {
        self.poly.num_variables()
    }

    /// Extracts the current evaluation polynomial as scalar extension field elements.
    ///
    /// If the internal representation is SIMD-packed, unpacks all lanes first.
    #[tracing::instrument(skip_all)]
    pub fn evals(&self) -> Poly<EF> {
        self.poly.evals()
    }

    /// Evaluates `f` at a given multilinear point via interpolation.
    ///
    /// The weight polynomial is not involved in this evaluation.
    pub fn eval(&self, point: &Point<EF>) -> EF {
        self.poly.eval(point)
    }

    /// Runs additional sumcheck rounds, optionally incorporating new constraints.
    ///
    /// Two phases:
    ///
    /// 1. **Constraint folding** (optional):
    ///    If a constraint is provided, fold it into the weight polynomial
    ///    and update the claimed sum before any rounds.
    ///    Used to incorporate STIR challenges between batches.
    ///
    /// 2. **Round execution**:
    ///    Performs `folding_factor` rounds of one-variable-per-round sumcheck.
    ///    Each round computes `(c_0, c_2)`, commits, receives a challenge, and folds.
    ///
    /// # Returns
    ///
    /// The verifier challenges `(r_1, ..., r_{folding_factor})` from this batch.
    ///
    /// # Panics
    ///
    /// If `folding_factor` exceeds the current number of remaining variables.
    #[tracing::instrument(skip_all)]
    pub fn compute_sumcheck_polynomials<Challenger>(
        &mut self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        constraint: Option<Constraint<F, EF>>,
    ) -> Point<EF>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // If a new constraint is provided, fold it into the weight polynomial
        // and update the claimed sum before starting rounds.
        if let Some(constraint) = constraint {
            self.poly.combine(&mut self.sum, &constraint);
        }

        // Execute rounds of the standard sumcheck protocol.
        // Each call computes coefficients, commits, receives a challenge, and folds.
        let res = (0..folding_factor)
            .map(|_| {
                self.poly
                    .round(sumcheck_data, challenger, &mut self.sum, pow_bits)
            })
            .collect();

        // Return the collected verifier challenges.
        Point::new(res)
    }
}
