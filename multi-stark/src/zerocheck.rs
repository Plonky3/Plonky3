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

use alloc::vec::Vec;

use p3_air::{Air, AirLayout, BaseAir, SymbolicAirBuilder, get_all_symbolic_constraints};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_sumcheck::generic_degree::{GenericDegreeError, GenericDegreeProof, RoundProver};
use p3_util::log2_strict_usize;
use thiserror::Error;

use crate::folder::MultilinearFolder;
use crate::opening::OpeningClaims;
use crate::rounds::RoundStateBase;
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
    /// The current-row openings did not carry exactly one value per main column.
    #[error("zerocheck current-row opening count mismatch: expected {expected}, got {actual}")]
    OpeningCountMismatch {
        /// Number of main columns the AIR declares.
        expected: usize,
        /// Number of current-row values the proof carries.
        actual: usize,
    },
    /// The next-row openings did not carry exactly one value per next-row column.
    #[error("zerocheck next-row opening count mismatch: expected {expected}, got {actual}")]
    NextOpeningCountMismatch {
        /// Number of columns the AIR reads on the next row.
        expected: usize,
        /// Number of next-row values the proof carries.
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
    /// Current-row value of every main column at the sumcheck point, in column order.
    ///
    /// These are the `Eq` opening claims.
    pub local: Vec<EF>,
    /// Repeat-last successor value of each column the AIR reads on the next row.
    ///
    /// These are the `Next` opening claims, in the AIR's declared next-row order.
    pub next: Vec<EF>,
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
    /// The summed integrand is `eq(tau, x) * g(x)`.
    /// Its per-variable degree is the constraint per-variable degree, plus one for the multilinear weight.
    ///
    /// The constraint degree comes from one of two sources:
    /// - a constant-time hint supplied by the AIR, when present;
    /// - otherwise a symbolic pass that scores each constraint at domain size two.
    ///
    /// At domain size two every column and every boundary selector scores degree one, so the symbolic value is exact.
    ///
    /// The prover and the verifier both call this, so they always agree on the degree.
    /// The hint must be at least the true per-variable degree:
    /// - a smaller value under-determines the round polynomial and breaks soundness;
    /// - a larger value only inflates the proof and the per-row work.
    ///
    /// A debug assertion pins the hint against the symbolic value.
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
        // Largest per-variable degree among the asserted constraints, scored at domain size two.
        // No periodic columns reach this point, so the periodic-column lengths are empty.
        let symbolic_constraint_degree = || {
            let (base, ext) = get_all_symbolic_constraints::<F, EF, A>(self.air, layout);
            let base_degree = base
                .iter()
                .map(|c| c.poly_degree(2, &[]))
                .max()
                .unwrap_or(0);
            let ext_degree = ext.iter().map(|c| c.poly_degree(2, &[])).max().unwrap_or(0);
            base_degree.max(ext_degree)
        };

        if let Some(degree) = self.air.max_constraint_degree() {
            // A hint below the true constraint degree drops evaluations from each round polynomial.
            // Reject such a hint in debug builds.
            debug_assert!(
                degree >= symbolic_constraint_degree(),
                "max_constraint_degree hint is below the symbolic constraint degree"
            );
            return degree + 1;
        }

        // Add one for the multilinear eq weight.
        symbolic_constraint_degree() + 1
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
    #[tracing::instrument(skip_all)]
    pub fn prove<F, EF, Challenger>(
        &self,
        trace: &RowMajorMatrix<F>,
        public_values: &[F],
        challenger: &mut Challenger,
    ) -> (ZerocheckProof<F, EF>, Point<EF>)
    where
        F: Field,
        EF: ExtensionField<F>,
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>
            + for<'b> Air<MultilinearFolder<'b, F, F::Packing, EF::ExtensionPacking>>
            + for<'b> Air<MultilinearFolder<'b, F, EF, EF>>
            + Air<SymbolicAirBuilder<F, EF>>,
        EF::ExtensionPacking: From<EF> + From<F::Packing>,
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

        let state = RoundStateBase::<'_, _, F, EF>::new(
            self.air,
            public_values,
            alpha,
            &Point::new(tau),
            trace,
            degree,
        );

        // Bind the transcript to the claimed sum so the challenges depend on the statement.
        challenger.observe_algebra_element(EF::ZERO);

        let mut sumcheck = GenericDegreeProof {
            claimed_sum: EF::ZERO,
            round_polys: Vec::with_capacity(log_height),
            pow_witnesses: Vec::with_capacity(if self.pow_bits > 0 { log_height } else { 0 }),
        };
        let mut challenges = Vec::with_capacity(log_height);

        let evals = state.round_poly();
        challenger.observe_algebra_slice(&evals);

        // Optional proof-of-work; raises the cost of grinding a favorable challenge.
        if self.pow_bits > 0 {
            sumcheck.pow_witnesses.push(challenger.grind(self.pow_bits));
        }

        let r: EF = challenger.sample_algebra_element();
        let mut state = state.fold(r);

        sumcheck.round_polys.push(evals);
        challenges.push(r);

        let (proof_rest, point_rest) =
            state.prove(challenger, log_height - 1, degree, self.pow_bits, EF::ZERO);
        challenges.extend_from_slice(point_rest.as_slice());

        sumcheck.round_polys.extend(proof_rest.round_polys);
        sumcheck.pow_witnesses.extend(proof_rest.pow_witnesses);

        let (local, next, _) = state.evals();
        let next = self
            .air
            .main_next_row_columns()
            .iter()
            .map(|&column| next[column])
            .collect();

        (
            ZerocheckProof {
                sumcheck,
                local,
                next,
            },
            Point::new(challenges),
        )
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
    /// - the current-row openings do not carry one value per main column,
    /// - the next-row openings do not carry one value per next-row column,
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
    ) -> Result<Point<EF>, ZerocheckError>
    where
        F: Field,
        EF: ExtensionField<F>,
        A: for<'b> Air<MultilinearFolder<'b, F, EF, EF>> + Air<SymbolicAirBuilder<F, EF>>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Reject column kinds this version cannot prove, and read the layout once.
        let layout = self.layout::<F>();
        let width = layout.main_width;

        // Only the columns the AIR reads on the next row carry a successor claim.
        let next_columns = self.air.main_next_row_columns();

        // Every main column must contribute exactly one current-row opening.
        if proof.local.len() != width {
            return Err(ZerocheckError::OpeningCountMismatch {
                expected: width,
                actual: proof.local.len(),
            });
        }

        // Every next-row column must contribute exactly one successor opening.
        if proof.next.len() != next_columns.len() {
            return Err(ZerocheckError::NextOpeningCountMismatch {
                expected: next_columns.len(),
                actual: proof.next.len(),
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

        // Reduce the proof to opening claims at the bound point.
        let claims = OpeningClaims::new(point, proof.local.clone(), &next_columns, &proof.next);

        // Reconstruct the rows the folder reads, then recompute the alpha-batched constraint.
        let next_row = claims.next_row(width);
        let boundary = BoundaryEvals::at(claims.point.as_slice());
        let g = MultilinearFolder::new(&claims.local, &next_row, boundary, public_values, alpha)
            .eval_air(self.air);
        let eq_at_point = Point::eval_eq(&tau, claims.point.as_slice());

        // Close the protocol: the reduced sum must equal the integrand at the point.
        if final_sum != eq_at_point * g {
            return Err(ZerocheckError::FinalSumMismatch);
        }
        Ok(claims.point)
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

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::vec::Vec;
    use core::borrow::Borrow;

    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_baby_bear::{
        BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS, BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16,
        BABYBEAR_S_BOX_DEGREE, BabyBear, GenericPoseidon2LinearLayersBabyBear, Poseidon2BabyBear,
    };
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use p3_poseidon2_air::{Poseidon2Air, RoundConstants};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    const NUM_COLS: usize = 2;
    const POSEIDON2_WIDTH: usize = 16;
    const POSEIDON2_SBOX_DEGREE: u64 = BABYBEAR_S_BOX_DEGREE;
    const POSEIDON2_SBOX_REGISTERS: usize = 1;
    const POSEIDON2_HALF_FULL_ROUNDS: usize = BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS;
    const POSEIDON2_PARTIAL_ROUNDS: usize = BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16;

    type BabyBearPoseidon2Air = Poseidon2Air<
        F,
        GenericPoseidon2LinearLayersBabyBear,
        POSEIDON2_WIDTH,
        POSEIDON2_SBOX_DEGREE,
        POSEIDON2_SBOX_REGISTERS,
        POSEIDON2_HALF_FULL_ROUNDS,
        POSEIDON2_PARTIAL_ROUNDS,
    >;

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
        let (proof, _) = zerocheck.prove::<F, EF, _>(&trace, &pis, &mut prover_challenger);

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
        let (proof, point) =
            AirZerocheck::new(&FibAir, 0).prove::<F, EF, _>(&trace, &pis, &mut challenger);

        // Each Fibonacci constraint is per-variable degree 2 (a degree-1 selector
        // times a degree-1 column), so the eq-weighted integrand is degree 3.
        let layout = AirLayout::from_air::<F>(&FibAir);
        let degree = AirZerocheck::new(&FibAir, 0).sumcheck_degree::<F, EF>(layout);
        assert_eq!(degree, 3);
        assert_eq!(proof.sumcheck.num_rounds(), log2_strict_usize(n));
        assert_eq!(point.num_variables(), log2_strict_usize(n));
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
        let (proof, _) = zerocheck.prove::<F, EF, _>(&trace, &pis, &mut prover_challenger);

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
        let (mut proof, _) = zerocheck.prove::<F, EF, _>(&trace, &pis, &mut prover_challenger);
        proof.local[0] += EF::ONE;

        let mut verifier_challenger = fresh_challenger();
        let err = zerocheck
            .verify::<F, EF, _>(&proof, log2_strict_usize(n), &pis, &mut verifier_challenger)
            .unwrap_err();
        assert!(matches!(err, ZerocheckError::FinalSumMismatch));
    }

    #[test]
    fn zerocheck_rejects_tampered_next_opening() {
        // Corrupt a next-row (successor) opening; the final check must reject.
        // Fibonacci reads both columns on the next row, so a next claim exists to corrupt.
        let n = 8;
        let trace = fib_trace(n);
        let pis = fib_public_values(n);
        let zerocheck = AirZerocheck::new(&FibAir, 0);

        let mut prover_challenger = fresh_challenger();
        let (mut proof, _) = zerocheck.prove::<F, EF, _>(&trace, &pis, &mut prover_challenger);
        proof.next[0] += EF::ONE;

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
        let (mut proof, _) = zerocheck.prove::<F, EF, _>(&trace, &pis, &mut prover_challenger);

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
        let (mut proof, _) = zerocheck.prove::<F, EF, _>(&trace, &pis, &mut prover_challenger);

        // Remove one opened value so the count no longer matches the AIR width.
        proof.local.pop();

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

    /// Width-2 AIR that holds column 0 constant and reads only column 0 on the next row.
    ///
    /// It declares a next-row subset, so only one column needs a successor claim.
    struct ConstColAir;

    impl<X> BaseAir<X> for ConstColAir {
        fn width(&self) -> usize {
            2
        }
        fn main_next_row_columns(&self) -> Vec<usize> {
            // Only column 0 is read on the next row; column 1 is current-row only.
            alloc::vec![0]
        }
    }

    impl<AB: AirBuilder> Air<AB> for ConstColAir {
        fn eval(&self, builder: &mut AB) {
            // Bind the current row and the single shifted entry the constraint reads.
            let main = builder.main();
            let local0 = main.current_slice()[0];
            let next0 = main.next_slice()[0];

            // Column 0 keeps its value from one row to the next.
            builder.when_transition().assert_eq(local0, next0);
        }
    }

    /// Trace whose column 0 is the constant `5` and column 1 counts up by row.
    fn const_col_trace(n: usize) -> RowMajorMatrix<F> {
        let mut values = Vec::with_capacity(2 * n);
        for i in 0..n {
            // Column 0 is constant, so the transition constraint always holds.
            values.push(F::from_u64(5));
            // Column 1 is unconstrained filler.
            values.push(F::from_u64(i as u64));
        }
        RowMajorMatrix::new(values, 2)
    }

    #[test]
    fn next_claims_cover_only_declared_columns() {
        // This AIR commits two columns but reads only column 0 on the next row.
        // So it needs a successor claim for that one column, not for both.
        //
        //     current-row (Eq) claims : column 0, column 1   -> 2
        //     next-row    (Next) claim: column 0              -> 1
        let n = 8;
        let trace = const_col_trace(n);
        let zerocheck = AirZerocheck::new(&ConstColAir, 0);

        let mut prover_challenger = fresh_challenger();
        let (proof, _) = zerocheck.prove::<F, EF, _>(&trace, &[], &mut prover_challenger);

        // Two committed columns yield two current-row claims.
        assert_eq!(proof.local.len(), 2);
        // Only the read-ahead column yields a next-row claim.
        assert_eq!(proof.next.len(), 1);

        // The reduction still verifies end to end.
        let mut verifier_challenger = fresh_challenger();
        zerocheck
            .verify::<F, EF, _>(&proof, log2_strict_usize(n), &[], &mut verifier_challenger)
            .expect("subset-next AIR must verify");
    }

    #[test]
    fn zerocheck_poseidon2() {
        // Invariant on the Poseidon2 permutation AIR:
        //   - each current-row opening equals the column multilinear at the bound point;
        //   - each next-row opening equals the shifted column at that point;
        //   - prover and verifier bind the same sumcheck point.
        //
        // Trace height is the only axis, swept exhaustively over 1..10.
        for num_vars in 1..10 {
            // Trace height 2^num_vars, i.e. one hashed input per row.
            let num_hashes = 1 << num_vars;

            // Deterministic round constants, so the trace and transcript are reproducible.
            let mut rng = SmallRng::seed_from_u64(1);
            let constants = RoundConstants::from_rng(&mut rng);
            let air: BabyBearPoseidon2Air = Poseidon2Air::new(constants);

            // Witness trace: each row is one full permutation, satisfying every constraint.
            let trace =
                tracing::info_span!("zerocheck_poseidon2_generate_trace", num_vars, num_hashes)
                    .in_scope(|| air.generate_trace_rows(num_hashes, 0));

            // Prove the alpha-batched constraint vanishes on every row.
            let zerocheck = AirZerocheck::new(&air, 0);
            let mut prover_challenger = fresh_challenger();
            let (proof, point_prover) =
                zerocheck.prove::<F, EF, _>(&trace, &[], &mut prover_challenger);

            // Reference columns: one multilinear per trace column, in row order.
            //
            //     row-major trace --transpose--> one row per column
            let columns = trace.transpose();
            let columns = columns
                .row_slices()
                .map(|col| Poly::new(col.to_vec()))
                .collect::<Vec<_>>();

            // Each current-row opening must equal the column multilinear at the bound point.
            columns
                .iter()
                .zip(proof.local.iter())
                .for_each(|(col, &local)| {
                    assert_eq!(col.eval_base(&point_prover), local);
                });

            // Each next-row opening must equal the same column shifted by one row.
            air.main_next_row_columns()
                .iter()
                .zip(proof.next.iter())
                .for_each(|(&column, &next)| {
                    assert_eq!(columns[column].eval_next_base(&point_prover), next);
                });

            // Replaying the transcript must reproduce the prover's bound point exactly.
            let mut verifier_challenger = fresh_challenger();
            let point_verifier = zerocheck
                .verify::<F, EF, _>(&proof, num_vars, &[], &mut verifier_challenger)
                .unwrap();
            assert_eq!(point_prover, point_verifier);
        }
    }
}
