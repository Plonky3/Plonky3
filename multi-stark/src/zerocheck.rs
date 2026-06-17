//! AIR zerocheck: prove the alpha-batched constraint vanishes on every trace row.
//!
//! Evaluating the alpha-batched constraint on each of the `2^k` trace rows yields a multilinear polynomial `g`.
//! Vanishing on every row is equivalent, for a random point `tau`, to a single sum being zero:
//!
//! ```text
//!     sum_x eq(tau, x) * g(x) = 0
//! ```
//!
//! The generic-degree sumcheck proves that sum.
//! A zerocheck always claims zero, so the verifier rejects any proof that claims a different sum.

use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, AirLayout, BaseAir, SymbolicAirBuilder};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::{GenericDegreeError, GenericDegreeProof, RoundProver};
use p3_util::log2_strict_usize;
use thiserror::Error;

use crate::folder::MultilinearFolder;
use crate::metadata::ConstraintMetadata;
use crate::selectors::BoundaryEvals;

/// Reasons the zerocheck verifier rejects a proof.
#[derive(Debug, Error)]
pub enum ZerocheckError {
    /// The inner sumcheck transcript failed to verify.
    #[error("zerocheck sumcheck: {0}")]
    Sumcheck(GenericDegreeError),
    /// The proof claimed a nonzero sum, but a zerocheck always sums to zero.
    #[error("zerocheck claimed sum is nonzero")]
    NonZeroClaimedSum,
    /// An opening vector did not carry exactly one value per main column.
    #[error("zerocheck opening count mismatch: expected {expected}, got {actual}")]
    OpeningCountMismatch {
        /// Number of main columns the AIR declares.
        expected: usize,
        /// Number of opened values the proof carries.
        actual: usize,
    },
    /// The reduced sum did not match the constraint evaluated at the random point.
    #[error("zerocheck final sum does not match the constraint at the challenge point")]
    FinalSumMismatch,
}

/// Opening claims and the sumcheck transcript produced by the zerocheck prover.
#[derive(Clone, Debug)]
pub struct ZerocheckProof<F, EF> {
    /// Generic-degree sumcheck transcript for `sum_x eq(tau, x) * g(x) = 0`.
    pub sumcheck: GenericDegreeProof<F, EF>,
    /// Each main column's multilinear extension at the sumcheck point.
    pub local_at_point: Vec<EF>,
    /// Each main column's repeat-last successor at the sumcheck point.
    pub next_at_point: Vec<EF>,
}

/// An AIR zerocheck instance.
///
/// Bundles the AIR and the grinding parameter shared by the prover and verifier.
/// The field and transcript are chosen per call, so one instance serves any field.
#[derive(Debug)]
pub struct AirZerocheck<'a, A> {
    /// AIR whose alpha-batched constraint is checked.
    air: &'a A,
    /// Grinding difficulty per sumcheck round, or `0` to skip.
    pow_bits: usize,
}

impl<'a, A> AirZerocheck<'a, A> {
    /// Create a zerocheck instance for an AIR.
    ///
    /// # Arguments
    ///
    /// - `air`: the AIR whose constraints are checked.
    /// - `pow_bits`: grinding difficulty per sumcheck round, or `0` to skip.
    pub const fn new(air: &'a A, pow_bits: usize) -> Self {
        Self { air, pow_bits }
    }

    /// Derive the AIR layout and reject column kinds this version cannot prove.
    ///
    /// Version one proves a single main trace.
    /// Preprocessed and periodic columns are not wired in yet.
    /// An AIR that declared either would be evaluated against empty data and silently mis-proved.
    ///
    /// # Returns
    ///
    /// The AIR layout, carrying the main width and public-value count.
    ///
    /// # Panics
    ///
    /// Panics if the AIR declares any preprocessed or periodic columns.
    fn layout<F>(&self) -> AirLayout
    where
        F: Field,
        A: BaseAir<F>,
    {
        let layout = AirLayout::from_air::<F>(self.air);

        // Preprocessed columns need their own committed table and opening claims.
        assert_eq!(
            layout.preprocessed_width, 0,
            "zerocheck does not support preprocessed columns yet"
        );

        // Periodic columns need closed-form evaluation threaded into the folder.
        assert_eq!(
            layout.num_periodic_columns, 0,
            "zerocheck does not support periodic columns yet"
        );

        layout
    }

    /// Per-round degree of the zerocheck sumcheck.
    ///
    /// The summed integrand `eq(tau, x) * g(x)` has per-variable degree `max_constraint_degree + 2`.
    ///
    /// The `+ 2` is two single-degree corrections:
    /// - `eq(tau, x)` is multilinear, so it adds one,
    /// - the transition selector is degree one but counts as zero symbolically, so it adds one.
    ///
    /// Sound only while every column and selector has per-variable degree at most one, which standard AIRs satisfy.
    ///
    /// # Arguments
    ///
    /// - `layout`: column widths and public-value counts sizing the symbolic pass.
    fn sumcheck_degree<F, EF>(&self, layout: AirLayout) -> usize
    where
        F: Field,
        EF: ExtensionField<F>,
        A: Air<SymbolicAirBuilder<F, EF>>,
    {
        // Symbolic max constraint degree, then the plus-two integrand correction.
        let metadata = ConstraintMetadata::from_air::<F, EF, A>(self.air, layout);
        metadata.max_constraint_degree + 2
    }

    /// Prove that the AIR's alpha-batched constraint vanishes on every trace row.
    ///
    /// The caller must observe the trace commitment into the challenger before this call.
    /// The public values are observed here, so the caller need not observe them.
    ///
    /// # Arguments
    ///
    /// - `trace`: the execution trace, one column per AIR column, height a power of two.
    /// - `public_values`: public inputs forwarded to the AIR.
    /// - `challenger`: the Fiat-Shamir transcript.
    ///
    /// # Returns
    ///
    /// The sumcheck transcript and the column opening claims at the sumcheck point.
    ///
    /// # Panics
    ///
    /// Panics if the AIR declares preprocessed or periodic columns, which this version does not support.
    pub fn prove<F, EF, Challenger>(
        &self,
        trace: &RowMajorMatrix<F>,
        public_values: &[F],
        challenger: &mut Challenger,
    ) -> ZerocheckProof<F, EF>
    where
        F: Field,
        EF: ExtensionField<F>,
        A: for<'b> Air<MultilinearFolder<'b, F, EF>> + Air<SymbolicAirBuilder<F, EF>>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let log_height = log2_strict_usize(trace.height());

        // Reject column kinds this version cannot prove, and read the layout once.
        let layout = self.layout::<F>();

        // Per-round sumcheck degree from the symbolic constraint degree.
        let degree = self.sumcheck_degree::<F, EF>(layout);

        // Bind the public values before any challenge depends on them.
        // The trace commitment must already be in the transcript.
        challenger.observe_algebra_slice(public_values);

        // Draw the batching scalar then the zerocheck point, in that fixed order.
        let (alpha, tau) = sample_zerocheck_challenges(challenger, log_height);

        let mut state = RoundState::new(self.air, public_values, alpha, &tau, trace, degree);

        // The claim is zero: every constraint must vanish on the hypercube.
        let (sumcheck, _point) =
            state.prove::<F, _>(challenger, log_height, degree, self.pow_bits, EF::ZERO);

        // After every variable is bound, each table holds its evaluation at the point.
        let local_at_point = state.local.iter().map(|p| p.as_slice()[0]).collect();
        let next_at_point = state.next.iter().map(|p| p.as_slice()[0]).collect();

        ZerocheckProof {
            sumcheck,
            local_at_point,
            next_at_point,
        }
    }

    /// Verify a zerocheck proof.
    ///
    /// The opened column values are trusted at this layer.
    /// Binding them to a commitment is the job of the polynomial commitment scheme in a later step.
    ///
    /// The caller must observe the trace commitment into the challenger before this call.
    /// The public values are observed here, so the caller need not observe them.
    ///
    /// # Arguments
    ///
    /// - `proof`: the zerocheck proof.
    /// - `log_height`: base-two logarithm of the trace height.
    /// - `public_values`: public inputs forwarded to the AIR.
    /// - `challenger`: the Fiat-Shamir transcript.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    /// - an opening vector does not carry one value per main column,
    /// - the claimed sum is nonzero,
    /// - the inner sumcheck transcript fails to verify,
    /// - the reduced sum does not match the constraint at the random point.
    ///
    /// # Panics
    ///
    /// Panics if the AIR declares preprocessed or periodic columns, which this version does not support.
    pub fn verify<F, EF, Challenger>(
        &self,
        proof: &ZerocheckProof<F, EF>,
        log_height: usize,
        public_values: &[F],
        challenger: &mut Challenger,
    ) -> Result<(), ZerocheckError>
    where
        F: Field,
        EF: ExtensionField<F>,
        A: for<'b> Air<MultilinearFolder<'b, F, EF>> + Air<SymbolicAirBuilder<F, EF>>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Reject column kinds this version cannot prove, and read the layout once.
        let layout = self.layout::<F>();
        let width = layout.main_width;

        // Each main column must contribute exactly one current-row opening.
        if proof.local_at_point.len() != width {
            return Err(ZerocheckError::OpeningCountMismatch {
                expected: width,
                actual: proof.local_at_point.len(),
            });
        }

        // Each main column must contribute exactly one next-row opening.
        if proof.next_at_point.len() != width {
            return Err(ZerocheckError::OpeningCountMismatch {
                expected: width,
                actual: proof.next_at_point.len(),
            });
        }

        // A zerocheck always claims the sum is zero.
        // A nonzero claim is a forgery vector: it would let an unsatisfied trace pass.
        if proof.sumcheck.claimed_sum != EF::ZERO {
            return Err(ZerocheckError::NonZeroClaimedSum);
        }

        let degree = self.sumcheck_degree::<F, EF>(layout);

        // Bind the public values before any challenge depends on them.
        // The trace commitment must already be in the transcript.
        challenger.observe_algebra_slice(public_values);

        // Draw the same batching scalar and zerocheck point the prover drew.
        let (alpha, tau) = sample_zerocheck_challenges(challenger, log_height);

        let (point, final_sum) = proof
            .sumcheck
            .verify(challenger, log_height, degree, self.pow_bits)
            .map_err(ZerocheckError::Sumcheck)?;

        // Recompute the integrand at the random point from the opened column values.
        let boundary = BoundaryEvals::at(point.as_slice());
        let g = MultilinearFolder::new(
            &proof.local_at_point,
            &proof.next_at_point,
            boundary,
            public_values,
            alpha,
        )
        .eval_air(self.air);
        let eq_at_point = Point::eval_eq(&tau, point.as_slice());

        // Close the protocol: the reduced sum must equal the integrand at the point.
        if final_sum != eq_at_point * g {
            return Err(ZerocheckError::FinalSumMismatch);
        }
        Ok(())
    }
}

/// Draw the constraint-batching scalar and the zerocheck point, in that order.
///
/// Prover and verifier call this identically so their transcripts stay in lockstep.
fn sample_zerocheck_challenges<F, EF, Challenger>(
    challenger: &mut Challenger,
    log_height: usize,
) -> (EF, Vec<EF>)
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    let alpha = challenger.sample_algebra_element();
    let tau = (0..log_height)
        .map(|_| challenger.sample_algebra_element())
        .collect();
    (alpha, tau)
}

/// Sumcheck prover state for the AIR zerocheck.
///
/// Every table shares the same hypercube and is bound one variable per round.
struct RoundState<'a, A, F, EF> {
    /// AIR whose alpha-batched constraint is being evaluated.
    air: &'a A,
    /// Public inputs forwarded to the AIR.
    public_values: &'a [F],
    /// Random scalar batching the AIR constraints.
    alpha: EF,
    /// Zerocheck weight `eq(tau, x)` over the remaining hypercube.
    eq: Poly<EF>,
    /// First-row selector table, `1` at row zero and `0` elsewhere.
    first: Poly<EF>,
    /// Last-row selector table, `1` at the final row and `0` elsewhere.
    last: Poly<EF>,
    /// Transition selector table, `0` at the final row and `1` elsewhere.
    transition: Poly<EF>,
    /// Current-row values of each main column.
    local: Vec<Poly<EF>>,
    /// Next-row values of each main column, with repeat-last at the final row.
    next: Vec<Poly<EF>>,
    /// Per-round sumcheck degree.
    degree: usize,
}

impl<'a, A, F, EF> RoundState<'a, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Build the prover state from a trace and a sampled zerocheck point.
    fn new(
        air: &'a A,
        public_values: &'a [F],
        alpha: EF,
        tau: &[EF],
        trace: &RowMajorMatrix<F>,
        degree: usize,
    ) -> Self {
        let height = trace.height();
        let width = trace.width;

        let mut local = Vec::with_capacity(width);
        let mut next = Vec::with_capacity(width);
        for c in 0..width {
            // Lift column `c` into the extension field in row order.
            let column: Vec<EF> = (0..height)
                .map(|i| EF::from(trace.values[i * width + c]))
                .collect();
            // Successor column: row i reads row i+1, and the final row repeats itself.
            let mut successor = Vec::with_capacity(height);
            successor.extend_from_slice(&column[1..]);
            successor.push(column[height - 1]);
            local.push(Poly::new(column));
            next.push(Poly::new(successor));
        }

        // eq(tau, x) over the hypercube.
        let eq = Poly::new_from_point(tau, EF::ONE);

        // Selector tables as plain indicators over the hypercube.
        let mut first = EF::zero_vec(height);
        first[0] = EF::ONE;
        let mut last = EF::zero_vec(height);
        last[height - 1] = EF::ONE;
        let mut transition = vec![EF::ONE; height];
        transition[height - 1] = EF::ZERO;

        Self {
            air,
            public_values,
            alpha,
            eq,
            first: Poly::new(first),
            last: Poly::new(last),
            transition: Poly::new(transition),
            local,
            next,
            degree,
        }
    }
}

impl<'a, A, F, EF> RoundProver<EF> for RoundState<'a, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: for<'b> Air<MultilinearFolder<'b, F, EF>>,
{
    fn fold(&mut self, r: EF) {
        self.eq.fix_prefix_var_mut(r);
        self.first.fix_prefix_var_mut(r);
        self.last.fix_prefix_var_mut(r);
        self.transition.fix_prefix_var_mut(r);
        for p in &mut self.local {
            p.fix_prefix_var_mut(r);
        }
        for p in &mut self.next {
            p.fix_prefix_var_mut(r);
        }
    }

    fn round_poly(&self) -> Vec<EF> {
        let width = self.local.len();

        // Reused per-row buffers handed to the folder.
        let mut local_row = EF::zero_vec(width);
        let mut next_row = EF::zero_vec(width);

        // Transmit h at nodes 0, 2, 3, ..., degree; the verifier recovers h(1).
        let mut out = Vec::with_capacity(self.degree);
        for node in core::iter::once(0).chain(2..=self.degree) {
            let z = EF::from_usize(node);

            // Bind the current variable to z in every table.
            let eq_z = self.eq.fix_prefix_var(z);
            let first_z = self.first.fix_prefix_var(z);
            let last_z = self.last.fix_prefix_var(z);
            let transition_z = self.transition.fix_prefix_var(z);
            let local_z: Vec<Poly<EF>> = self.local.iter().map(|p| p.fix_prefix_var(z)).collect();
            let next_z: Vec<Poly<EF>> = self.next.iter().map(|p| p.fix_prefix_var(z)).collect();

            // Sum the weighted constraint over the remaining hypercube.
            let mut acc = EF::ZERO;
            for s in 0..eq_z.num_evals() {
                for c in 0..width {
                    local_row[c] = local_z[c].as_slice()[s];
                    next_row[c] = next_z[c].as_slice()[s];
                }
                let boundary = BoundaryEvals {
                    first: first_z.as_slice()[s],
                    last: last_z.as_slice()[s],
                    transition: transition_z.as_slice()[s],
                };
                let g = MultilinearFolder::new(
                    &local_row,
                    &next_row,
                    boundary,
                    self.public_values,
                    self.alpha,
                )
                .eval_air(self.air);
                acc += eq_z.as_slice()[s] * g;
            }
            out.push(acc);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::borrow::Borrow;

    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    const NUM_COLS: usize = 2;

    fn fresh_challenger() -> Ch {
        // Fixed seed so prover and verifier transcripts match exactly.
        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
        let perm = Perm::new_from_rng_128(&mut rng);
        Ch::new(perm)
    }

    /// Fibonacci AIR.
    ///
    /// - first row: `left == public[0]` and `right == public[1]`
    /// - transition: `next.left == right` and `next.right == left + right`
    /// - last row: `right == public[2]`
    struct FibAir;

    struct FibRow<T> {
        left: T,
        right: T,
    }

    impl<T> Borrow<FibRow<T>> for [T] {
        fn borrow(&self) -> &FibRow<T> {
            // Safety: two fields of type T in declaration order match the layout of [T; 2].
            debug_assert_eq!(self.len(), NUM_COLS);
            let ptr = self.as_ptr() as *const FibRow<T>;
            unsafe { &*ptr }
        }
    }

    impl<X> BaseAir<X> for FibAir {
        fn width(&self) -> usize {
            NUM_COLS
        }
        fn num_public_values(&self) -> usize {
            3
        }
    }

    impl<AB: AirBuilder> Air<AB> for FibAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let pis = builder.public_values();
            let (a, b, x) = (pis[0], pis[1], pis[2]);

            let local: &FibRow<AB::Var> = main.current_slice().borrow();
            let next: &FibRow<AB::Var> = main.next_slice().borrow();

            let mut first = builder.when_first_row();
            first.assert_eq(local.left, a);
            first.assert_eq(local.right, b);

            let mut trans = builder.when_transition();
            trans.assert_eq(local.right, next.left);
            trans.assert_eq(local.left + local.right, next.right);

            builder.when_last_row().assert_eq(local.right, x);
        }
    }

    /// Build a length-`n` Fibonacci trace seeded with `(0, 1)`.
    fn fib_trace(n: usize) -> RowMajorMatrix<F> {
        let mut left = F::ZERO;
        let mut right = F::ONE;
        let mut values = Vec::with_capacity(NUM_COLS * n);
        for _ in 0..n {
            values.push(left);
            values.push(right);
            let next_left = right;
            let next_right = left + right;
            left = next_left;
            right = next_right;
        }
        RowMajorMatrix::new(values, NUM_COLS)
    }

    /// Public inputs `(F_0, F_1, F_n)` for the length-`n` trace.
    fn fib_public_values(n: usize) -> [F; 3] {
        let trace = fib_trace(n);
        let last = trace.values[(n - 1) * NUM_COLS + 1];
        [F::ZERO, F::ONE, last]
    }

    #[test]
    fn zerocheck_accepts_valid_fibonacci() {
        // A satisfying trace must prove and verify, with the final sum matching
        // the constraint evaluated at the random point.
        let n = 8;
        let trace = fib_trace(n);
        let pis = fib_public_values(n);
        let zerocheck = AirZerocheck::new(&FibAir, 0);

        let mut prover_challenger = fresh_challenger();
        let proof = zerocheck.prove::<F, EF, _>(&trace, &pis, &mut prover_challenger);

        let mut verifier_challenger = fresh_challenger();
        zerocheck
            .verify::<F, EF, _>(&proof, log2_strict_usize(n), &pis, &mut verifier_challenger)
            .expect("valid trace must verify");
    }

    #[test]
    fn zerocheck_round_polys_have_expected_shape() {
        // Degree-consistency: the proof has one round per variable, and each round
        // carries exactly `degree` transmitted evaluations.
        let n = 8;
        let trace = fib_trace(n);
        let pis = fib_public_values(n);

        let mut challenger = fresh_challenger();
        let proof = AirZerocheck::new(&FibAir, 0).prove::<F, EF, _>(&trace, &pis, &mut challenger);

        // Fibonacci has symbolic max degree 2, so the per-round degree is 4.
        let layout = AirLayout::from_air::<F>(&FibAir);
        let degree = AirZerocheck::new(&FibAir, 0).sumcheck_degree::<F, EF>(layout);
        assert_eq!(degree, 4);
        assert_eq!(proof.sumcheck.num_rounds(), log2_strict_usize(n));
        for round in &proof.sumcheck.round_polys {
            assert_eq!(round.len(), degree);
        }
    }

    #[test]
    fn zerocheck_rejects_violated_constraint() {
        // Flip one trace cell so a constraint no longer holds.
        // The claimed sum of zero is then false and the final check must reject.
        let n = 8;
        let mut trace = fib_trace(n);
        trace.values[2 * NUM_COLS] += F::ONE;
        let pis = fib_public_values(n);
        let zerocheck = AirZerocheck::new(&FibAir, 0);

        let mut prover_challenger = fresh_challenger();
        let proof = zerocheck.prove::<F, EF, _>(&trace, &pis, &mut prover_challenger);

        let mut verifier_challenger = fresh_challenger();
        let err = zerocheck
            .verify::<F, EF, _>(&proof, log2_strict_usize(n), &pis, &mut verifier_challenger)
            .unwrap_err();
        assert!(matches!(err, ZerocheckError::FinalSumMismatch));
    }

    #[test]
    fn zerocheck_rejects_tampered_opening() {
        // Corrupt an opening claim; the final check must reject.
        let n = 8;
        let trace = fib_trace(n);
        let pis = fib_public_values(n);
        let zerocheck = AirZerocheck::new(&FibAir, 0);

        let mut prover_challenger = fresh_challenger();
        let mut proof = zerocheck.prove::<F, EF, _>(&trace, &pis, &mut prover_challenger);
        proof.local_at_point[0] += EF::ONE;

        let mut verifier_challenger = fresh_challenger();
        let err = zerocheck
            .verify::<F, EF, _>(&proof, log2_strict_usize(n), &pis, &mut verifier_challenger)
            .unwrap_err();
        assert!(matches!(err, ZerocheckError::FinalSumMismatch));
    }

    #[test]
    fn zerocheck_rejects_nonzero_claimed_sum() {
        // A zerocheck always claims the sum is zero.
        //
        // Mutation: take an honest proof and bump its claimed sum off zero.
        let n = 8;
        let trace = fib_trace(n);
        let pis = fib_public_values(n);
        let zerocheck = AirZerocheck::new(&FibAir, 0);

        let mut prover_challenger = fresh_challenger();
        let mut proof = zerocheck.prove::<F, EF, _>(&trace, &pis, &mut prover_challenger);

        // Declare a nonzero sum; the verifier must reject before any further work.
        proof.sumcheck.claimed_sum += EF::ONE;

        let mut verifier_challenger = fresh_challenger();
        let err = zerocheck
            .verify::<F, EF, _>(&proof, log2_strict_usize(n), &pis, &mut verifier_challenger)
            .unwrap_err();
        assert!(matches!(err, ZerocheckError::NonZeroClaimedSum));
    }

    #[test]
    fn zerocheck_rejects_wrong_opening_count() {
        // Each of the two main columns must contribute exactly one current-row opening.
        //
        // Fixture state: width-2 AIR, so two local openings are expected.
        //
        // Mutation: drop one local opening.
        //
        //     local openings: [col_0]        (len 1)
        //     expected width: 2
        //     -> 1 != 2 -> reject
        let n = 8;
        let trace = fib_trace(n);
        let pis = fib_public_values(n);
        let zerocheck = AirZerocheck::new(&FibAir, 0);

        let mut prover_challenger = fresh_challenger();
        let mut proof = zerocheck.prove::<F, EF, _>(&trace, &pis, &mut prover_challenger);

        // Remove one opened value so the count no longer matches the AIR width.
        proof.local_at_point.pop();

        let mut verifier_challenger = fresh_challenger();
        let err = zerocheck
            .verify::<F, EF, _>(&proof, log2_strict_usize(n), &pis, &mut verifier_challenger)
            .unwrap_err();
        assert!(matches!(
            err,
            ZerocheckError::OpeningCountMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }
}
