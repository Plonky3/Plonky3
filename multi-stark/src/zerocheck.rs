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

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use p3_air::{Air, AirLayout, SymbolicAirBuilder, get_all_symbolic_constraints};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::{GenericDegreeError, GenericDegreeProof, RoundPolyInterpolator};
use p3_sumcheck::layout::Table;
use thiserror::Error;

use crate::folder::MultilinearFolder;
use crate::opening::{OpeningClaims, TableOpening};
use crate::packed_ext::PackedExt;
use crate::rounds::{RoundStateBase, RoundStateExt, Stage};
use crate::selectors::{BoundaryEvals, periodic_evals_at};

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
    /// The preprocessed current-row openings did not carry one value per preprocessed column.
    #[error(
        "zerocheck preprocessed current-row opening count mismatch: expected {expected}, got {actual}"
    )]
    PreprocessedOpeningCountMismatch {
        /// Number of preprocessed columns the AIR declares.
        expected: usize,
        /// Number of preprocessed current-row values the proof carries.
        actual: usize,
    },
    /// The preprocessed next-row openings did not carry one value per preprocessed next-row column.
    #[error(
        "zerocheck preprocessed next-row opening count mismatch: expected {expected}, got {actual}"
    )]
    PreprocessedNextOpeningCountMismatch {
        /// Number of preprocessed columns the AIR reads on the next row.
        expected: usize,
        /// Number of preprocessed next-row values the proof carries.
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
    /// Current-row values at the sumcheck point, grouped by input AIR order.
    ///
    /// `local[i]` contains one value per main column of `airs[i]`.
    pub local: Vec<Vec<EF>>,
    /// Repeat-last successor values at the sumcheck point, grouped by input AIR order.
    ///
    /// `next[i]` contains one value per column declared by `airs[i].main_next_row_columns()`.
    pub next: Vec<Vec<EF>>,
    /// Current-row preprocessed values at the sumcheck point, grouped by input AIR order.
    pub preprocessed_local: Vec<Vec<EF>>,
    /// Repeat-last preprocessed successor values, grouped by input AIR order.
    pub preprocessed_next: Vec<Vec<EF>>,
}

/// One AIR's opening values at its sub-point, held together while scattering back to caller order.
#[derive(Clone)]
struct AirOpenings<EF> {
    /// Current-row value of each main column.
    local: Vec<EF>,
    /// Successor value of each main column the AIR reads on the next row.
    next: Vec<EF>,
    /// Current-row value of each preprocessed column.
    preprocessed_local: Vec<EF>,
    /// Successor value of each preprocessed column the AIR reads on the next row.
    preprocessed_next: Vec<EF>,
}

/// A batched AIR zerocheck instance.
///
/// Bundles the AIRs and the grinding parameter shared by the prover and verifier.
/// The field and transcript are chosen per call, so one instance serves any field.
#[derive(Debug)]
pub struct AirZerocheck<'a, A> {
    /// AIRs whose alpha-batched constraints are checked.
    airs: &'a [&'a A],
    /// Grinding difficulty per sumcheck round, or `0` to skip.
    pow_bits: usize,
}

/// Per-round degree of an AIR's zerocheck sumcheck.
///
/// The integrand is `eq(tau, x) * g(x)`.
/// Its per-variable degree is the constraint degree plus one for the multilinear eq weight.
fn sumcheck_degree<F, EF, A>(air: &A) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    air_degree(air) + 1
}

/// Per-variable constraint degree of an AIR, with the eq weight not yet applied.
///
/// The degree comes from one of two sources:
/// - a constant-time hint supplied by the AIR, when present;
/// - otherwise a symbolic pass that scores each constraint at domain size two.
///
/// At domain size two every column and boundary selector scores degree one, so the symbolic value
/// is exact.
///
/// The hint must be at least the true degree:
/// - a smaller hint drops evaluations from each round polynomial and breaks soundness;
/// - a larger hint only inflates the proof and the per-row work.
///
/// A debug assertion pins the hint against the symbolic value.
fn air_degree<F, EF, A>(air: &A) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let layout = AirLayout::from_air::<F>(air);

    // Largest per-variable degree among the asserted constraints, scored at domain size two.
    //
    // A periodic column is materialized as a full-height multilinear polynomial.
    // So it scores per-variable degree one, just like any trace column.
    // Passing empty periodic lengths yields exactly that at domain size two.
    let symbolic_constraint_degree = || {
        let (base, ext) = get_all_symbolic_constraints::<F, EF, A>(air, layout);
        let base_degree = base
            .iter()
            .map(|c| c.poly_degree(2, &[]))
            .max()
            .unwrap_or(0);
        let ext_degree = ext.iter().map(|c| c.poly_degree(2, &[])).max().unwrap_or(0);
        base_degree.max(ext_degree)
    };

    if let Some(degree) = air.max_constraint_degree() {
        // A hint below the true constraint degree drops evaluations from each round polynomial.
        // Reject such a hint in debug builds.
        debug_assert!(
            degree >= symbolic_constraint_degree(),
            "max_constraint_degree hint is below the symbolic constraint degree"
        );
        return degree;
    }

    // No hint: fall back to the symbolic constraint degree.
    symbolic_constraint_degree()
}

impl<'a, A> AirZerocheck<'a, A> {
    /// Create a zerocheck instance for AIRs that may have different trace heights.
    ///
    /// AIRs are batched in the order supplied here; proof opening claims use the same order.
    pub const fn new(airs: &'a [&'a A], pow_bits: usize) -> Self {
        Self { airs, pow_bits }
    }

    /// Prove that every AIR in the batch vanishes on its trace.
    ///
    /// `tables[i]`, `preprocessed[i]`, and `public_values[i]` correspond to `airs[i]`.
    ///
    /// The caller must observe every trace commitment into the challenger before this call.
    /// The public values are observed here, so the caller need not observe them.
    ///
    /// Periodic columns, if any, are folded into the sumcheck but produce no opening claim.
    ///
    /// # Panics
    ///
    /// Panics if the input lengths disagree with the number of AIRs.
    /// Panics if a periodic column's period is not a power of two dividing the trace height.
    /// Panics if any trace height is less than two.
    #[tracing::instrument(skip_all)]
    pub fn prove<F, EF, Challenger>(
        &self,
        preprocessed: &[Option<&Table<F>>],
        tables: &[&Table<F>],
        public_values: &[&[F]],
        challenger: &mut Challenger,
    ) -> (ZerocheckProof<F, EF>, Point<EF>)
    where
        F: Field,
        EF: ExtensionField<F>,
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>
            + for<'b> Air<
                MultilinearFolder<
                    'b,
                    F,
                    <F as Field>::Packing,
                    <EF as ExtensionField<F>>::ExtensionPacking,
                >,
            > + for<'b> Air<MultilinearFolder<'b, F, EF, EF>>
            + for<'b> Air<
                MultilinearFolder<
                    'b,
                    F,
                    PackedExt<F, <EF as ExtensionField<F>>::ExtensionPacking>,
                    PackedExt<F, <EF as ExtensionField<F>>::ExtensionPacking>,
                >,
            > + Air<SymbolicAirBuilder<F, EF>>,
        <EF as ExtensionField<F>>::ExtensionPacking: From<EF> + From<<F as Field>::Packing>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert!(!self.airs.is_empty());
        assert_eq!(self.airs.len(), tables.len(),);
        assert_eq!(self.airs.len(), preprocessed.len(),);
        assert_eq!(self.airs.len(), public_values.len(),);

        let degrees = self
            .airs
            .iter()
            .zip(tables.iter())
            .zip(preprocessed.iter())
            .zip(public_values.iter())
            .map(|(((&air, table), preprocessed), public_values)| {
                let layout = AirLayout::from_air::<F>(air);
                // A height-1 trace has zero variables and never activates a stage.
                // Reject it here, matching the verifier's `log_height > 0` guard.
                assert!(
                    table.num_variables() > 0,
                    "zerocheck requires each trace height to be at least two"
                );
                assert_eq!(table.num_polys(), layout.main_width);
                assert_eq!(
                    preprocessed.map_or(0, |table| table.num_polys()),
                    layout.preprocessed_width
                );
                if let Some(preprocessed) = preprocessed {
                    assert_eq!(preprocessed.num_variables(), table.num_variables());
                }
                assert_eq!(public_values.len(), air.num_public_values());
                air_degree::<F, EF, A>(air)
            })
            .collect::<Vec<_>>();
        let max_degree = degrees.iter().copied().max().unwrap();

        // Bucket AIR indices by trace height.
        // The map gives deterministic height order; stages are built largest-first below.
        let mut indices_by_height = BTreeMap::<usize, Vec<usize>>::new();
        tables.iter().enumerate().for_each(|(index, table)| {
            indices_by_height
                .entry(table.num_variables())
                .or_default()
                .push(index);
        });

        let log_height = indices_by_height.keys().copied().max().unwrap();

        // Each stage contains the AIRs sharing one trace height.
        // Original AIR indices stay attached so final openings can return in caller order.
        let stages = indices_by_height
            .into_iter()
            .rev()
            .map(|(_, indices)| {
                let tables = indices.iter().map(|&i| tables[i]).collect::<Vec<_>>();
                let preprocessed = indices.iter().map(|&i| preprocessed[i]).collect::<Vec<_>>();
                let airs = indices.iter().map(|&i| self.airs[i]).collect::<Vec<_>>();
                let public_values = indices
                    .iter()
                    .map(|&i| public_values[i])
                    .collect::<Vec<_>>();
                let degrees = indices.iter().map(|&i| degrees[i]).collect::<Vec<_>>();
                Stage::new(
                    &airs,
                    &public_values,
                    &indices,
                    &preprocessed,
                    &tables,
                    &degrees,
                )
            })
            .collect::<Vec<_>>();

        // Bind all public values before any challenge depends on them.
        // Trace commitments must already be in the transcript.
        for values in public_values {
            challenger.observe_algebra_slice(values);
        }

        let (alpha, beta, tau) =
            sample_zerocheck_challenges::<F, EF, Challenger>(challenger, log_height);
        let tau = Point::new(tau);
        // Beta batches AIR contributions in caller order.
        // When a stage activates, it selects the beta powers for its original AIR indices.
        let beta_powers = beta.powers().collect_n(self.airs.len());

        // Bind the transcript to the claimed sum so the challenges depend on the statement.
        challenger.observe_algebra_element(EF::ZERO);

        let mut proof = GenericDegreeProof {
            claimed_sum: EF::ZERO,
            round_polys: Vec::with_capacity(log_height),
            pow_witnesses: Vec::with_capacity(if self.pow_bits > 0 { log_height } else { 0 }),
        };

        let mut challenges = Vec::with_capacity(log_height);
        // Active stages live as folded extension states.
        // claims[i] is the current reduced claim for states[i].
        let mut states = Vec::<RoundStateExt<'_, '_, A, F, EF>>::new();
        let mut claims = Vec::<EF>::new();

        // All stages share the same global sumcheck point.
        // eq_prefix covers folded rounds; eq_suffix covers the tail still inside each state.
        let mut eq_prefix = EF::ONE;
        let mut eq_suffix = Poly::new_from_point(&tau.as_slice()[1..], EF::ONE);

        let interpolator = RoundPolyInterpolator::new(max_degree);

        // Barycentric interpolators, indexed by internal degree, built once.
        // A lower-degree stage is extrapolated up to the batch's max degree.
        // Reusing prebuilt weights avoids recomputing them every round.
        let interpolators = (0..=max_degree)
            .map(RoundPolyInterpolator::<EF>::new)
            .collect::<Vec<_>>();

        let mut next_stage = 0;
        for round in 0..log_height {
            let num_vars = log_height - round;
            let tau_round = tau.as_slice()[round];
            let tau_round_inv = tau_round.inverse();

            let mut round_poly_acc = EF::zero_vec(max_degree);
            let mut round_polys = Vec::with_capacity(states.len());

            // Existing stages already live over the extension field.
            // Extend each stage's internal round polynomial to the global degree and accumulate it.
            for (state, &claim) in states.iter_mut().zip(claims.iter()) {
                let round_poly = state.round_poly(&eq_suffix);
                let q1 = (claim - (EF::ONE - tau_round) * round_poly[0]) * tau_round_inv;
                let unweighted_claim = round_poly[0] + q1;
                let round_poly = interpolators[round_poly.len()].extend_evals(
                    &round_poly,
                    unweighted_claim,
                    max_degree,
                );
                EF::add_slices(&mut round_poly_acc, &round_poly);
                round_polys.push(round_poly);
            }

            // A stage activates when the global cube reaches its trace height.
            // Its base-field trace is evaluated once, extended to the global degree, then folded.
            let mut new_state = None;
            if next_stage < stages.len() && num_vars == stages[next_stage].num_vars {
                let stage = &stages[next_stage];
                let tau = Point::new(tau.as_slice()[round..].to_vec());
                let betas = stage
                    .indices
                    .iter()
                    .map(|&air_index| beta_powers[air_index])
                    .collect::<Vec<_>>();
                let mut state = RoundStateBase::new(&stages[next_stage], alpha, betas, &tau);
                let round_poly = state.round_poly(&eq_suffix);
                // A stage's first round runs on an unfolded trace, so its claim starts at zero.
                let q1 = (EF::ZERO - (EF::ONE - tau_round) * round_poly[0]) * tau_round_inv;
                let unweighted_claim = round_poly[0] + q1;
                let round_poly = interpolators[round_poly.len()].extend_evals(
                    &round_poly,
                    unweighted_claim,
                    max_degree,
                );
                EF::add_slices(&mut round_poly_acc, &round_poly);
                new_state = Some((state, round_poly));
                next_stage += 1;
            }

            // The verifier sees one global sumcheck round.
            // Convert the accumulated internal q-evals back to eq-weighted standard evals.
            let claim = claims.iter().copied().sum::<EF>();
            let (standard_evals, _) = standard_round_from_q_evals(
                &interpolator,
                &round_poly_acc,
                claim,
                eq_prefix,
                tau.as_slice()[round],
            );

            challenger.observe_algebra_slice(&standard_evals);
            proof.round_polys.push(standard_evals);

            if self.pow_bits > 0 {
                proof.pow_witnesses.push(challenger.grind(self.pow_bits));
            }

            let r: EF = challenger.sample_algebra_element();
            challenges.push(r);

            // Fold every already-active state at the sampled challenge.
            // Update its reduced claim using the same internal round polynomial used above.
            for ((state, claim), round_poly) in states
                .iter_mut()
                .zip(claims.iter_mut())
                .zip(round_polys.iter())
            {
                let q1 = (*claim - (EF::ONE - tau_round) * round_poly[0]) * tau_round_inv;
                let unweighted_claim = round_poly[0] + q1;
                *claim = interpolator.eval(round_poly, unweighted_claim, r);
                state.fold(r);
            }

            // The newly activated stage joins the active list only after this round.
            // Its first claim starts from the zerocheck claim, which is zero.
            if let Some((state, round_poly)) = new_state {
                let q1 = (EF::ZERO - (EF::ONE - tau_round) * round_poly[0]) * tau_round_inv;
                let unweighted_claim = round_poly[0] + q1;
                claims.push(interpolator.eval(&round_poly, unweighted_claim, r));
                states.push(state.fold(r));
            }

            // Advance the shared equality factors to the next round.
            // The prefix absorbs r; the suffix drops the variable just bound.
            eq_prefix *= Point::eval_eq(&[tau_round], &[r]);
            if round + 1 < log_height {
                eq_suffix.sum_prefix_var_mut();
            }
        }

        // States are ordered by activation height, not caller AIR order.
        // Scatter each AIR's openings back to its original index, then read them out in order.
        let mut openings: Vec<Option<AirOpenings<EF>>> = alloc::vec![None; self.airs.len()];
        for (stage, state) in stages.iter().zip(states) {
            let (local, all_next, _) = state.evals();
            let mut column_offset = 0;
            for (((&air_index, table), preprocessed), &air) in stage
                .indices
                .iter()
                .zip(stage.tables.iter())
                .zip(stage.preprocessed.iter())
                .zip(stage.airs.iter())
            {
                // Main columns occupy the first span of this AIR's merged block.
                let main_offset = column_offset;
                let main_end = main_offset + table.num_polys();
                let next = air
                    .main_next_row_columns()
                    .into_iter()
                    .map(|column| all_next[main_offset + column])
                    .collect();
                column_offset = main_end;

                // Preprocessed columns follow immediately after the main columns.
                let preprocessed_offset = column_offset;
                let preprocessed_end =
                    preprocessed_offset + preprocessed.map_or(0, |table| table.num_polys());
                let preprocessed_next = air
                    .preprocessed_next_row_columns()
                    .into_iter()
                    .map(|column| all_next[preprocessed_offset + column])
                    .collect();
                column_offset = preprocessed_end;

                // Periodic columns come last in this AIR's block.
                // They carry no opening claim, so skip past them to reach the next AIR.
                column_offset += air.num_periodic_columns();

                openings[air_index] = Some(AirOpenings {
                    local: local[main_offset..main_end].to_vec(),
                    next,
                    preprocessed_local: local[preprocessed_offset..preprocessed_end].to_vec(),
                    preprocessed_next,
                });
            }
        }

        // Split the per-AIR openings into the four proof vectors, in caller order.
        let mut local = Vec::with_capacity(self.airs.len());
        let mut next = Vec::with_capacity(self.airs.len());
        let mut preprocessed_local = Vec::with_capacity(self.airs.len());
        let mut preprocessed_next = Vec::with_capacity(self.airs.len());
        for opening in openings {
            let opening = opening.expect("every AIR must have openings");
            local.push(opening.local);
            next.push(opening.next);
            preprocessed_local.push(opening.preprocessed_local);
            preprocessed_next.push(opening.preprocessed_next);
        }

        (
            ZerocheckProof {
                sumcheck: proof,
                local,
                next,
                preprocessed_local,
                preprocessed_next,
            },
            Point::new(challenges),
        )
    }

    /// Verify a zerocheck proof.
    ///
    /// The opened column values are trusted at this layer.
    /// Binding them to a commitment is the job of the polynomial commitment scheme in a later step.
    ///
    /// The caller must observe every trace commitment into the challenger before this call.
    /// The public values are observed here, so the caller need not observe them.
    ///
    /// # Arguments
    ///
    /// - `proof`: the zerocheck proof.
    /// - `log_heights`: base-two logarithm of each AIR's trace height, in caller order.
    /// - `public_values`: public inputs forwarded to each AIR, in caller order.
    /// - `challenger`: the Fiat-Shamir transcript.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    /// - the per-AIR opening groups do not number one per AIR,
    /// - an AIR's current-row openings do not carry one value per main column,
    /// - an AIR's next-row openings do not carry one value per next-row column,
    /// - the claimed sum is nonzero,
    /// - the inner sumcheck transcript fails to verify,
    /// - the reduced sum does not match the constraints at the random point.
    pub fn verify<F, EF, Challenger>(
        &self,
        proof: &ZerocheckProof<F, EF>,
        log_heights: &[usize],
        public_values: &[&[F]],
        challenger: &mut Challenger,
    ) -> Result<Point<EF>, ZerocheckError>
    where
        F: Field,
        EF: ExtensionField<F>,
        A: for<'b> Air<MultilinearFolder<'b, F, EF, EF>> + Air<SymbolicAirBuilder<F, EF>>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert!(!self.airs.is_empty(), "zerocheck requires at least one AIR");
        assert_eq!(self.airs.len(), public_values.len(),);
        assert_eq!(self.airs.len(), log_heights.len(),);
        assert!(log_heights.iter().all(|&log_height| log_height > 0),);

        for (&air, public_values) in self.airs.iter().zip(public_values.iter()) {
            let layout = AirLayout::from_air::<F>(air);
            assert_ne!(layout.main_width, 0);
            assert_eq!(public_values.len(), air.num_public_values());
        }

        let next_columns = self
            .airs
            .iter()
            .map(|&air| air.main_next_row_columns())
            .collect::<Vec<_>>();
        let preprocessed_next_columns = self
            .airs
            .iter()
            .map(|&air| air.preprocessed_next_row_columns())
            .collect::<Vec<_>>();
        if proof.local.len() != self.airs.len() {
            return Err(ZerocheckError::OpeningCountMismatch {
                expected: self.airs.len(),
                actual: proof.local.len(),
            });
        }
        // Every next-row column must contribute exactly one successor opening.
        if proof.next.len() != self.airs.len() {
            return Err(ZerocheckError::NextOpeningCountMismatch {
                expected: self.airs.len(),
                actual: proof.next.len(),
            });
        }
        if proof.preprocessed_local.len() != self.airs.len() {
            return Err(ZerocheckError::PreprocessedOpeningCountMismatch {
                expected: self.airs.len(),
                actual: proof.preprocessed_local.len(),
            });
        }
        if proof.preprocessed_next.len() != self.airs.len() {
            return Err(ZerocheckError::PreprocessedNextOpeningCountMismatch {
                expected: self.airs.len(),
                actual: proof.preprocessed_next.len(),
            });
        }

        // Verify the sumcheck reduction, then close on the proof's own opened values.
        let reduction = self.verify_reduction::<F, EF, _>(
            &proof.sumcheck,
            log_heights,
            public_values,
            challenger,
        )?;
        let main = proof
            .local
            .iter()
            .zip(next_columns.iter())
            .zip(proof.next.iter())
            .map(|((local, next_columns), next_values)| {
                TableOpening::new(local, next_columns, next_values)
            })
            .collect::<Vec<_>>();
        let preprocessed = proof
            .preprocessed_local
            .iter()
            .zip(preprocessed_next_columns.iter())
            .zip(proof.preprocessed_next.iter())
            .map(|((local, next_columns), next_values)| {
                TableOpening::new(local, next_columns, next_values)
            })
            .collect::<Vec<_>>();
        self.check_constraint::<F, EF>(
            &reduction,
            &main,
            &preprocessed,
            log_heights,
            public_values,
        )?;
        Ok(reduction.point)
    }

    /// Verify the zerocheck sumcheck and return the data the final check needs.
    ///
    /// This stops short of recomputing the constraint.
    /// The opened column values are not yet known.
    /// The committed verifier opens them through a commitment scheme.
    ///
    /// The caller must observe every trace commitment into the challenger before this call.
    /// The public values are observed here.
    /// The caller therefore does not observe them separately.
    ///
    /// # Arguments
    ///
    /// - Zerocheck sumcheck transcript.
    /// - Base-two logarithm of each AIR's trace height, in caller order.
    /// - Public inputs forwarded to each AIR, in caller order.
    /// - Fiat-Shamir transcript.
    ///
    /// # Errors
    ///
    /// Returns an error when the claimed sum is nonzero.
    /// Returns an error when the sumcheck transcript fails to verify.
    pub fn verify_reduction<F, EF, Challenger>(
        &self,
        sumcheck: &GenericDegreeProof<F, EF>,
        log_heights: &[usize],
        public_values: &[&[F]],
        challenger: &mut Challenger,
    ) -> Result<ZerocheckReduction<EF>, ZerocheckError>
    where
        F: Field,
        EF: ExtensionField<F>,
        A: Air<SymbolicAirBuilder<F, EF>>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert!(!self.airs.is_empty(), "zerocheck requires at least one AIR");
        assert_eq!(self.airs.len(), public_values.len(),);
        assert_eq!(self.airs.len(), log_heights.len(),);
        assert!(log_heights.iter().all(|&log_height| log_height > 0),);

        if sumcheck.claimed_sum != EF::ZERO {
            return Err(ZerocheckError::NonZeroClaimedSum);
        }

        let max_log_height = log_heights.iter().copied().max().unwrap();
        let degree = self
            .airs
            .iter()
            .map(|&air| sumcheck_degree::<F, EF, A>(air))
            .max()
            .unwrap();

        // Bind the public values before any challenge depends on them.
        // The trace commitment must already be in the transcript.
        for values in public_values {
            challenger.observe_algebra_slice(values);
        }

        // Draw the same constraint scalar, AIR batching scalar, and zerocheck point the prover drew.
        let (alpha, beta, tau) =
            sample_zerocheck_challenges::<F, EF, Challenger>(challenger, max_log_height);

        let (point, final_sum) = sumcheck
            .verify(challenger, max_log_height, degree, self.pow_bits)
            .map_err(ZerocheckError::Sumcheck)?;

        Ok(ZerocheckReduction {
            alpha,
            beta,
            tau,
            point,
            final_sum,
        })
    }

    /// Close the zerocheck: recompute the alpha-batched constraints and match the reduced sum.
    ///
    /// Opened values are grouped in the same order as `self.airs`.
    /// They may come from the proof itself or from commitment openings.
    pub fn check_constraint<F, EF>(
        &self,
        reduction: &ZerocheckReduction<EF>,
        main: &[TableOpening<'_, EF>],
        preprocessed: &[TableOpening<'_, EF>],
        log_heights: &[usize],
        public_values: &[&[F]],
    ) -> Result<(), ZerocheckError>
    where
        F: Field,
        EF: ExtensionField<F>,
        A: for<'b> Air<MultilinearFolder<'b, F, EF, EF>> + Air<SymbolicAirBuilder<F, EF>>,
    {
        assert!(!self.airs.is_empty(), "zerocheck requires at least one AIR");
        assert_eq!(self.airs.len(), public_values.len(),);
        assert_eq!(self.airs.len(), log_heights.len(),);
        assert!(log_heights.iter().all(|&log_height| log_height > 0),);

        if main.len() != self.airs.len() {
            return Err(ZerocheckError::OpeningCountMismatch {
                expected: self.airs.len(),
                actual: main.len(),
            });
        }
        if preprocessed.len() != self.airs.len() {
            return Err(ZerocheckError::PreprocessedOpeningCountMismatch {
                expected: self.airs.len(),
                actual: preprocessed.len(),
            });
        }

        let max_log_height = reduction.tau.len();
        assert_eq!(reduction.point.as_slice().len(), max_log_height);

        let mut g = EF::ZERO;
        for (air_index, ((((&air, &log_height), main), preprocessed), beta)) in self
            .airs
            .iter()
            .zip(log_heights.iter())
            .zip(main.iter())
            .zip(preprocessed.iter())
            .zip(reduction.beta.powers())
            .enumerate()
        {
            let layout = AirLayout::from_air::<F>(air);
            assert_ne!(layout.main_width, 0);
            assert_eq!(public_values[air_index].len(), air.num_public_values());

            if main.local.len() != layout.main_width {
                return Err(ZerocheckError::OpeningCountMismatch {
                    expected: layout.main_width,
                    actual: main.local.len(),
                });
            }
            if main.next_values.len() != main.next_columns.len() {
                return Err(ZerocheckError::NextOpeningCountMismatch {
                    expected: main.next_columns.len(),
                    actual: main.next_values.len(),
                });
            }
            if preprocessed.local.len() != layout.preprocessed_width {
                return Err(ZerocheckError::PreprocessedOpeningCountMismatch {
                    expected: layout.preprocessed_width,
                    actual: preprocessed.local.len(),
                });
            }
            if preprocessed.next_values.len() != preprocessed.next_columns.len() {
                return Err(ZerocheckError::PreprocessedNextOpeningCountMismatch {
                    expected: preprocessed.next_columns.len(),
                    actual: preprocessed.next_values.len(),
                });
            }

            let activation_round = max_log_height - log_height;
            let point = Point::new(reduction.point.as_slice()[activation_round..].to_vec());
            let boundary = BoundaryEvals::at(point.as_slice());
            let claims = OpeningClaims::new(
                point.clone(),
                main.local.to_vec(),
                main.next_columns,
                main.next_values,
            );
            let next_row = claims.next_row(layout.main_width);
            let preprocessed_claims = OpeningClaims::new(
                point,
                preprocessed.local.to_vec(),
                preprocessed.next_columns,
                preprocessed.next_values,
            );
            let preprocessed_next_row = preprocessed_claims.next_row(layout.preprocessed_width);

            // Periodic columns are not committed.
            // Recompute each column's multilinear extension at the bound point, in closed form.
            let periodic_columns = air.periodic_columns();
            let periodic = periodic_evals_at::<F, EF>(&periodic_columns, claims.point.as_slice());

            let air_g = MultilinearFolder::new(
                &claims.local,
                &next_row,
                boundary,
                public_values[air_index],
                reduction.alpha,
            )
            .with_preprocessed(&preprocessed_claims.local, &preprocessed_next_row)
            .with_periodic(&periodic)
            .eval_air(air);
            g += beta * air_g;
        }

        let eq_at_point = Point::eval_eq(&reduction.tau, reduction.point.as_slice());
        if reduction.final_sum != eq_at_point * g {
            return Err(ZerocheckError::FinalSumMismatch);
        }
        Ok(())
    }
}

/// Data yielded before the opened values are known.
///
/// The reduction verifier produces it from the sumcheck transcript.
/// The closing check consumes it together with the opened column values.
#[derive(Clone, Debug)]
pub struct ZerocheckReduction<EF> {
    /// Random scalar batching the AIR constraints.
    pub alpha: EF,
    /// Random scalar batching AIRs together in caller order.
    pub beta: EF,
    /// Zerocheck point sampled before the sumcheck.
    pub tau: Vec<EF>,
    /// Bound sumcheck point with every variable fixed to one challenge.
    pub point: Point<EF>,
    /// Reduced sum bound by the sumcheck.
    pub final_sum: EF,
}

/// Rebuild the verifier-facing round message from the prover's internal evaluations.
///
/// # Overview
///
/// The optimized prover evaluates an internal polynomial with the active equality
/// factor stripped out, which lowers its degree by one:
/// ```text
/// q(X) = sum over x of eq(suffix, x) * g(prefix, X, x)
/// ```
///
/// The verifier still expects the full weighted round polynomial:
/// ```text
/// s(X) = eq_prefix * eq(tau, X) * q(X)
/// ```
///
/// This restores `s` from the transmitted internal evaluations.
///
/// # Why this lives here
///
/// The weighting and the inter-round claim relation are specific to this zerocheck.
/// The generic interpolator stays unaware of equality factors, so this glue stays out of it.
///
/// Node one is dropped from the message because the verifier recovers it from the claim.
///
/// # Arguments
///
/// - `interpolator`: barycentric helper for the internal degree, reused across rounds.
/// - `evals`: internal evaluations at nodes `0, 2, 3, ..., deg - 1`.
/// - `claim`: previous round's reduced value, tying `q(0)` and `q(1)` together.
/// - `eq_prefix`: product of equality factors at the already-bound challenges.
/// - `tau`: this round's zerocheck point coordinate; must be nonzero.
///
/// # Returns
///
/// - the weighted evaluations `s(0), s(2), ..., s(deg)` sent to the verifier;
/// - the unweighted sum `q(0) + q(1)` used to reduce the internal claim next round.
///
/// # Panics
///
/// Panics if `tau` is zero, since recovering `q(1)` divides by it.
/// The caller samples `tau` from the nonzero elements, so this does not occur.
fn standard_round_from_q_evals<EF>(
    interpolator: &RoundPolyInterpolator<EF>,
    evals: &[EF],
    claim: EF,
    eq_prefix: EF,
    tau: EF,
) -> (Vec<EF>, EF)
where
    EF: Field,
{
    // Recover the dropped node one from the inter-round relation.
    //
    //     claim = (1 - tau) * q(0) + tau * q(1)
    //  => q(1)  = (claim - (1 - tau) * q(0)) / tau
    //
    // The unweighted sum q(0) + q(1) is what reduces the internal claim next round.
    let unweighted_sum = evals[0] + (claim - (EF::ONE - tau) * evals[0]) * tau.inverse();

    // The weighted message carries one extra degree from the equality factor.
    // Extrapolate the internal polynomial to that top node.
    let degree = evals.len() + 1;
    let last = interpolator.eval(evals, unweighted_sum, EF::from_usize(degree));

    // Weight every transmitted node, skipping node one as the verifier rebuilds it.
    //
    //     nodes : 0, 2, 3, ..., deg
    //     values: q(0), q(2), ..., q(deg - 1), q(deg)
    //     s(node) = eq_prefix * eq(tau, node) * q(node)
    let standard_evals = (0..=degree)
        .filter(|&node| node != 1)
        .zip(evals.iter().copied().chain(core::iter::once(last)))
        .map(|(node, q)| eq_prefix * Point::eval_eq(&[tau], &[EF::from_usize(node)]) * q)
        .collect();

    (standard_evals, unweighted_sum)
}

/// Draw the constraint-batching scalar, AIR-batching scalar, and zerocheck point, in that order.
///
/// Prover and verifier call this identically so their transcripts stay in lockstep.
///
/// Each zerocheck point coordinate is drawn nonzero.
/// The prover divides by a coordinate when rebuilding a round message, so a zero would divide by zero.
/// Resampling on both sides keeps the transcripts aligned, and a zero draw has negligible probability.
fn sample_zerocheck_challenges<F, EF, Challenger>(
    challenger: &mut Challenger,
    log_height: usize,
) -> (EF, EF, Vec<EF>)
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    // The batching scalar may take any value.
    let alpha = challenger.sample_algebra_element();
    let beta = challenger.sample_algebra_element();

    // Draw each point coordinate, resampling past a zero.
    let tau = (0..log_height)
        .map(|_| {
            let mut coord: EF = challenger.sample_algebra_element();
            while coord.is_zero() {
                coord = challenger.sample_algebra_element();
            }
            coord
        })
        .collect();
    (alpha, beta, tau)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::borrow::Cow;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::borrow::Borrow;

    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_baby_bear::{
        BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS, BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16,
        BABYBEAR_S_BOX_DEGREE, BabyBear, GenericPoseidon2LinearLayersBabyBear, Poseidon2BabyBear,
    };
    use p3_blake3_air::Blake3Air;
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use p3_poseidon2_air::{Poseidon2Air, RoundConstants};
    use p3_util::log2_strict_usize;
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

    fn prove_traces<A>(
        zerocheck: &AirZerocheck<'_, A>,
        traces: &[&RowMajorMatrix<F>],
        public_values: &[&[F]],
        challenger: &mut Ch,
    ) -> (ZerocheckProof<F, EF>, Point<EF>)
    where
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>
            + for<'b> Air<
                MultilinearFolder<
                    'b,
                    F,
                    <F as Field>::Packing,
                    <EF as ExtensionField<F>>::ExtensionPacking,
                >,
            > + for<'b> Air<MultilinearFolder<'b, F, EF, EF>>
            + for<'b> Air<
                MultilinearFolder<
                    'b,
                    F,
                    PackedExt<F, <EF as ExtensionField<F>>::ExtensionPacking>,
                    PackedExt<F, <EF as ExtensionField<F>>::ExtensionPacking>,
                >,
            > + Air<SymbolicAirBuilder<F, EF>>,
        <EF as ExtensionField<F>>::ExtensionPacking: From<EF> + From<<F as Field>::Packing>,
    {
        let tables = traces
            .iter()
            .map(|trace| Table::new(trace.transpose()))
            .collect::<Vec<_>>();
        let preprocessed = tables.iter().map(|_| None).collect::<Vec<_>>();
        let table_refs = tables.iter().collect::<Vec<_>>();
        zerocheck.prove::<F, EF, _>(&preprocessed, &table_refs, public_values, challenger)
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
        let air = FibAir;
        let airs = [&air];
        let zerocheck = AirZerocheck::new(&airs, 0);

        let mut prover_challenger = fresh_challenger();
        let traces = [&trace];
        let public_values = [&pis[..]];
        let (proof, _) = prove_traces(&zerocheck, &traces, &public_values, &mut prover_challenger);

        let mut verifier_challenger = fresh_challenger();
        let log_heights = [log2_strict_usize(n)];
        zerocheck
            .verify::<F, EF, _>(
                &proof,
                &log_heights,
                &public_values,
                &mut verifier_challenger,
            )
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
        let air = FibAir;
        let airs = [&air];
        let zerocheck = AirZerocheck::new(&airs, 0);
        let traces = [&trace];
        let public_values = [&pis[..]];
        let (proof, point) = prove_traces(&zerocheck, &traces, &public_values, &mut challenger);

        // Each Fibonacci constraint is per-variable degree 2 (a degree-1 selector
        // times a degree-1 column), so the eq-weighted integrand is degree 3.
        let degree = sumcheck_degree::<F, EF, FibAir>(&air);
        assert_eq!(degree, 3);
        assert_eq!(proof.sumcheck.num_rounds(), log2_strict_usize(n));
        assert_eq!(point.num_variables(), log2_strict_usize(n));
        for round in &proof.sumcheck.round_polys {
            assert_eq!(round.len(), degree);
        }
    }

    #[test]
    fn zerocheck_round_trip_with_grinding() {
        // The prover inlines its own grind / observe / sample loop per round.
        // Every test above runs with pow_bits = 0, so the grinding branch is otherwise unexercised.
        //
        // Fixture state:
        //
        //     trace height : 8  -> log_height = 3 rounds
        //     pow_bits     : 4  -> one witness ground per round
        //
        // Invariant:
        //
        //     a valid trace proves and verifies, and the proof carries one witness per round.
        let n = 8;
        let pow_bits = 4;
        let trace = fib_trace(n);
        let pis = fib_public_values(n);
        let air = FibAir;
        let airs = [&air];
        let zerocheck = AirZerocheck::new(&airs, pow_bits);

        // Prove with grinding enabled.
        let mut prover_challenger = fresh_challenger();
        let traces = [&trace];
        let public_values = [&pis[..]];
        let (proof, _) = prove_traces(&zerocheck, &traces, &public_values, &mut prover_challenger);

        // Grinding emits exactly one witness per sumcheck round.
        assert_eq!(proof.sumcheck.pow_witnesses.len(), log2_strict_usize(n));

        // The verifier re-checks each round's witness while replaying the transcript.
        let mut verifier_challenger = fresh_challenger();
        let log_heights = [log2_strict_usize(n)];
        zerocheck
            .verify::<F, EF, _>(
                &proof,
                &log_heights,
                &public_values,
                &mut verifier_challenger,
            )
            .expect("valid trace must verify with grinding");
    }

    #[test]
    fn zerocheck_rejects_violated_constraint() {
        // Flip one trace cell so a constraint no longer holds.
        // The claimed sum of zero is then false and the final check must reject.
        let n = 8;
        let mut trace = fib_trace(n);
        trace.values[2 * NUM_COLS] += F::ONE;
        let pis = fib_public_values(n);
        let air = FibAir;
        let airs = [&air];
        let zerocheck = AirZerocheck::new(&airs, 0);

        let mut prover_challenger = fresh_challenger();
        let traces = [&trace];
        let public_values = [&pis[..]];
        let (proof, _) = prove_traces(&zerocheck, &traces, &public_values, &mut prover_challenger);

        let mut verifier_challenger = fresh_challenger();
        let log_heights = [log2_strict_usize(n)];
        let err = zerocheck
            .verify::<F, EF, _>(
                &proof,
                &log_heights,
                &public_values,
                &mut verifier_challenger,
            )
            .unwrap_err();
        assert!(matches!(err, ZerocheckError::FinalSumMismatch));
    }

    #[test]
    fn zerocheck_rejects_tampered_opening() {
        // Corrupt an opening claim; the final check must reject.
        let n = 8;
        let trace = fib_trace(n);
        let pis = fib_public_values(n);
        let air = FibAir;
        let airs = [&air];
        let zerocheck = AirZerocheck::new(&airs, 0);

        let mut prover_challenger = fresh_challenger();
        let traces = [&trace];
        let public_values = [&pis[..]];
        let (mut proof, _) =
            prove_traces(&zerocheck, &traces, &public_values, &mut prover_challenger);
        proof.local[0][0] += EF::ONE;

        let mut verifier_challenger = fresh_challenger();
        let log_heights = [log2_strict_usize(n)];
        let err = zerocheck
            .verify::<F, EF, _>(
                &proof,
                &log_heights,
                &public_values,
                &mut verifier_challenger,
            )
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
        let air = FibAir;
        let airs = [&air];
        let zerocheck = AirZerocheck::new(&airs, 0);

        let mut prover_challenger = fresh_challenger();
        let traces = [&trace];
        let public_values = [&pis[..]];
        let (mut proof, _) =
            prove_traces(&zerocheck, &traces, &public_values, &mut prover_challenger);
        proof.next[0][0] += EF::ONE;

        let mut verifier_challenger = fresh_challenger();
        let log_heights = [log2_strict_usize(n)];
        let err = zerocheck
            .verify::<F, EF, _>(
                &proof,
                &log_heights,
                &public_values,
                &mut verifier_challenger,
            )
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
        let air = FibAir;
        let airs = [&air];
        let zerocheck = AirZerocheck::new(&airs, 0);

        let mut prover_challenger = fresh_challenger();
        let traces = [&trace];
        let public_values = [&pis[..]];
        let (mut proof, _) =
            prove_traces(&zerocheck, &traces, &public_values, &mut prover_challenger);

        // Declare a nonzero sum; the verifier must reject before any further work.
        proof.sumcheck.claimed_sum += EF::ONE;

        let mut verifier_challenger = fresh_challenger();
        let log_heights = [log2_strict_usize(n)];
        let err = zerocheck
            .verify::<F, EF, _>(
                &proof,
                &log_heights,
                &public_values,
                &mut verifier_challenger,
            )
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
        let air = FibAir;
        let airs = [&air];
        let zerocheck = AirZerocheck::new(&airs, 0);

        let mut prover_challenger = fresh_challenger();
        let traces = [&trace];
        let public_values = [&pis[..]];
        let (mut proof, _) =
            prove_traces(&zerocheck, &traces, &public_values, &mut prover_challenger);

        // Remove one opened value so the count no longer matches the AIR width.
        proof.local[0].pop();

        let mut verifier_challenger = fresh_challenger();
        let log_heights = [log2_strict_usize(n)];
        let err = zerocheck
            .verify::<F, EF, _>(
                &proof,
                &log_heights,
                &public_values,
                &mut verifier_challenger,
            )
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
        let air = ConstColAir;
        let airs = [&air];
        let zerocheck = AirZerocheck::new(&airs, 0);

        let mut prover_challenger = fresh_challenger();
        let traces = [&trace];
        let public_values = [&[] as &[F]];
        let (proof, _) = prove_traces(&zerocheck, &traces, &public_values, &mut prover_challenger);

        // Two committed columns yield two current-row claims.
        assert_eq!(proof.local.len(), 1);
        assert_eq!(proof.local[0].len(), 2);
        // Only the read-ahead column yields a next-row claim.
        assert_eq!(proof.next.len(), 1);
        assert_eq!(proof.next[0].len(), 1);

        // The reduction still verifies end to end.
        let mut verifier_challenger = fresh_challenger();
        let log_heights = [log2_strict_usize(n)];
        zerocheck
            .verify::<F, EF, _>(
                &proof,
                &log_heights,
                &public_values,
                &mut verifier_challenger,
            )
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
                    .in_scope(|| air.generate_random_trace_rows(num_hashes, 0));

            // Prove the alpha-batched constraint vanishes on every row.
            let airs = [&air];
            let zerocheck = AirZerocheck::new(&airs, 0);
            let mut prover_challenger = fresh_challenger();
            let traces = [&trace];
            let public_values = [&[] as &[F]];
            let (proof, point_prover) =
                prove_traces(&zerocheck, &traces, &public_values, &mut prover_challenger);

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
                .zip(proof.local[0].iter())
                .for_each(|(col, &local)| {
                    assert_eq!(col.eval_base(&point_prover), local);
                });

            // Each next-row opening must equal the same column shifted by one row.
            air.main_next_row_columns()
                .iter()
                .zip(proof.next[0].iter())
                .for_each(|(&column, &next)| {
                    assert_eq!(columns[column].eval_next_base(&point_prover), next);
                });

            // Replaying the transcript must reproduce the prover's bound point exactly.
            let mut verifier_challenger = fresh_challenger();
            let log_heights = [num_vars];
            let point_verifier = zerocheck
                .verify::<F, EF, _>(
                    &proof,
                    &log_heights,
                    &public_values,
                    &mut verifier_challenger,
                )
                .unwrap();
            assert_eq!(point_prover, point_verifier);
        }
    }

    #[allow(clippy::large_enum_variant)]
    enum MixedAir {
        Poseidon2(BabyBearPoseidon2Air),
        Blake3(Blake3Air),
        Fib(FibAir),
    }

    impl BaseAir<F> for MixedAir {
        fn width(&self) -> usize {
            match self {
                Self::Poseidon2(air) => <BabyBearPoseidon2Air as BaseAir<F>>::width(air),
                Self::Blake3(air) => <Blake3Air as BaseAir<F>>::width(air),
                Self::Fib(air) => <FibAir as BaseAir<F>>::width(air),
            }
        }

        fn num_public_values(&self) -> usize {
            match self {
                Self::Poseidon2(air) => {
                    <BabyBearPoseidon2Air as BaseAir<F>>::num_public_values(air)
                }
                Self::Blake3(air) => <Blake3Air as BaseAir<F>>::num_public_values(air),
                Self::Fib(air) => <FibAir as BaseAir<F>>::num_public_values(air),
            }
        }

        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            match self {
                Self::Poseidon2(air) => {
                    <BabyBearPoseidon2Air as BaseAir<F>>::preprocessed_trace(air)
                }
                Self::Blake3(air) => <Blake3Air as BaseAir<F>>::preprocessed_trace(air),
                Self::Fib(air) => <FibAir as BaseAir<F>>::preprocessed_trace(air),
            }
        }

        fn preprocessed_width(&self) -> usize {
            match self {
                Self::Poseidon2(air) => {
                    <BabyBearPoseidon2Air as BaseAir<F>>::preprocessed_width(air)
                }
                Self::Blake3(air) => <Blake3Air as BaseAir<F>>::preprocessed_width(air),
                Self::Fib(air) => <FibAir as BaseAir<F>>::preprocessed_width(air),
            }
        }

        fn main_next_row_columns(&self) -> Vec<usize> {
            match self {
                Self::Poseidon2(air) => {
                    <BabyBearPoseidon2Air as BaseAir<F>>::main_next_row_columns(air)
                }
                Self::Blake3(air) => <Blake3Air as BaseAir<F>>::main_next_row_columns(air),
                Self::Fib(air) => <FibAir as BaseAir<F>>::main_next_row_columns(air),
            }
        }

        fn preprocessed_next_row_columns(&self) -> Vec<usize> {
            match self {
                Self::Poseidon2(air) => {
                    <BabyBearPoseidon2Air as BaseAir<F>>::preprocessed_next_row_columns(air)
                }
                Self::Blake3(air) => <Blake3Air as BaseAir<F>>::preprocessed_next_row_columns(air),
                Self::Fib(air) => <FibAir as BaseAir<F>>::preprocessed_next_row_columns(air),
            }
        }

        fn num_constraints(&self) -> Option<usize> {
            match self {
                Self::Poseidon2(air) => <BabyBearPoseidon2Air as BaseAir<F>>::num_constraints(air),
                Self::Blake3(air) => <Blake3Air as BaseAir<F>>::num_constraints(air),
                Self::Fib(air) => <FibAir as BaseAir<F>>::num_constraints(air),
            }
        }

        fn max_constraint_degree(&self) -> Option<usize> {
            match self {
                Self::Poseidon2(air) => {
                    <BabyBearPoseidon2Air as BaseAir<F>>::max_constraint_degree(air)
                }
                Self::Blake3(air) => <Blake3Air as BaseAir<F>>::max_constraint_degree(air),
                Self::Fib(air) => <FibAir as BaseAir<F>>::max_constraint_degree(air),
            }
        }
    }

    impl<AB> Air<AB> for MixedAir
    where
        AB: AirBuilder<F = F>,
        BabyBearPoseidon2Air: Air<AB>,
        Blake3Air: Air<AB>,
        FibAir: Air<AB>,
    {
        fn eval(&self, builder: &mut AB) {
            match self {
                Self::Poseidon2(air) => air.eval(builder),
                Self::Blake3(air) => air.eval(builder),
                Self::Fib(air) => air.eval(builder),
            }
        }
    }

    #[test]
    fn staged_zerocheck_mixed_poseidon2_blake3_fib() {
        for num_vars in 1..10 {
            let mut rng = SmallRng::seed_from_u64(1);
            let poseidon_constants = RoundConstants::from_rng(&mut rng);
            let poseidon_air: BabyBearPoseidon2Air = Poseidon2Air::new(poseidon_constants);

            let mut airs = Vec::<MixedAir>::with_capacity(3 * num_vars);
            let mut traces = Vec::with_capacity(3 * num_vars);
            let mut public_values = Vec::<Vec<F>>::with_capacity(3 * num_vars);
            let mut log_heights = Vec::with_capacity(3 * num_vars);

            for log_height in (1..=num_vars).rev() {
                let height = 1 << log_height;

                airs.push(MixedAir::Poseidon2(poseidon_air.clone()));
                traces.push(
                    tracing::info_span!(
                        "staged_zerocheck_poseidon2_stage_trace",
                        num_vars,
                        log_height
                    )
                    .in_scope(|| poseidon_air.generate_random_trace_rows(height, 0)),
                );
                public_values.push(Vec::new());
                log_heights.push(log_height);

                airs.push(MixedAir::Blake3(Blake3Air {}));
                traces.push(
                    tracing::info_span!(
                        "staged_zerocheck_blake3_stage_trace",
                        num_vars,
                        log_height
                    )
                    .in_scope(|| Blake3Air {}.generate_random_trace_rows(height, 0)),
                );
                public_values.push(Vec::new());
                log_heights.push(log_height);

                airs.push(MixedAir::Fib(FibAir));
                traces.push(fib_trace(height));
                public_values.push(fib_public_values(height).to_vec());
                log_heights.push(log_height);
            }

            let air_refs = airs.iter().collect::<Vec<_>>();
            let trace_refs = traces.iter().collect::<Vec<_>>();
            let public_refs = public_values
                .iter()
                .map(|values| &values[..])
                .collect::<Vec<_>>();
            let zerocheck = AirZerocheck::new(&air_refs, 0);

            let mut prover_challenger = fresh_challenger();
            let (proof, point_prover) = prove_traces(
                &zerocheck,
                &trace_refs,
                &public_refs,
                &mut prover_challenger,
            );

            assert_eq!(proof.sumcheck.num_rounds(), num_vars);
            let max_degree = airs
                .iter()
                .map(sumcheck_degree::<F, EF, MixedAir>)
                .max()
                .unwrap();
            for round in &proof.sumcheck.round_polys {
                assert_eq!(round.len(), max_degree);
            }
            assert_eq!(proof.local.len(), airs.len());
            assert_eq!(proof.next.len(), airs.len());

            for ((((air, trace), &log_height), local), next) in airs
                .iter()
                .zip(traces.iter())
                .zip(log_heights.iter())
                .zip(proof.local.iter())
                .zip(proof.next.iter())
            {
                let activation_round = num_vars - log_height;
                let point = Point::new(point_prover.as_slice()[activation_round..].to_vec());
                assert_eq!(local.len(), trace.width());

                let columns = trace.transpose();
                let columns = columns
                    .row_slices()
                    .map(|col| Poly::new(col.to_vec()))
                    .collect::<Vec<_>>();

                columns.iter().zip(local.iter()).for_each(|(col, &local)| {
                    assert_eq!(col.eval_base(&point), local);
                });

                let next_columns = air.main_next_row_columns();
                assert_eq!(next.len(), next_columns.len());
                next_columns
                    .iter()
                    .zip(next.iter())
                    .for_each(|(&column, &next)| {
                        assert_eq!(columns[column].eval_next_base(&point), next);
                    });
            }

            let mut verifier_challenger = fresh_challenger();
            let point_verifier = zerocheck
                .verify::<F, EF, _>(&proof, &log_heights, &public_refs, &mut verifier_challenger)
                .unwrap();
            assert_eq!(point_prover, point_verifier);
        }
    }

    /// Width-1 main AIR tied to a fixed width-1 preprocessed column.
    ///
    /// With the main trace equal to the preprocessed trace on every row, both constraints vanish:
    ///
    /// - first row: main current value equals preprocessed current value
    /// - transition: main next value equals preprocessed next value
    ///
    /// The transition reads both the main and the preprocessed next row, so every opening path runs.
    struct PreprocessedAir;

    impl<X> BaseAir<X> for PreprocessedAir {
        fn width(&self) -> usize {
            1
        }
        fn preprocessed_width(&self) -> usize {
            1
        }
    }

    impl<AB: AirBuilder> Air<AB> for PreprocessedAir {
        fn eval(&self, builder: &mut AB) {
            // Read the current and next entry of each single-column window.
            let main = builder.main();
            let main_local = main.current_slice()[0];
            let main_next = main.next_slice()[0];
            let preprocessed = builder.preprocessed();
            let preprocessed_local = preprocessed.current_slice()[0];
            let preprocessed_next = preprocessed.next_slice()[0];

            // The main column equals the fixed column at the boundary and along every step.
            builder
                .when_first_row()
                .assert_eq(main_local, preprocessed_local);
            builder
                .when_transition()
                .assert_eq(main_next, preprocessed_next);
        }
    }

    /// A satisfying main / preprocessed pair: both columns hold the same fixed values.
    fn preprocessed_pair(n: usize) -> (RowMajorMatrix<F>, RowMajorMatrix<F>) {
        // Any injective column works; an odd arithmetic progression keeps the values distinct.
        let values: Vec<F> = (0..n).map(|i| F::from_u64(3 + 2 * i as u64)).collect();
        (
            RowMajorMatrix::new(values.clone(), 1),
            RowMajorMatrix::new(values, 1),
        )
    }

    /// Build a preprocessed batch of two tall AIRs plus one short AIR.
    ///
    /// The two tall AIRs land in one stage, so the second carries nonzero column offsets.
    /// The short AIR activates one round later, exercising staging with preprocessed columns.
    ///
    /// ```text
    ///     stage tall (num_vars):     air0, air1
    ///     stage short (num_vars-1):  air2   (activates one round later)
    /// ```
    fn preprocessed_batch(
        num_vars: usize,
    ) -> ([usize; 3], Vec<RowMajorMatrix<F>>, Vec<RowMajorMatrix<F>>) {
        // Two AIRs at the tall height share a stage; one AIR sits one height below.
        let log_heights = [num_vars, num_vars, num_vars - 1];
        // Split each satisfying pair into its main matrix and its preprocessed matrix.
        let (mains, preprocessed) = log_heights
            .iter()
            .map(|&log_height| preprocessed_pair(1 << log_height))
            .unzip();
        (log_heights, mains, preprocessed)
    }

    #[test]
    fn staged_zerocheck_preprocessed_multi_air() {
        for num_vars in 2..6 {
            // Fixture: three preprocessed AIRs, two sharing the tall stage, one activating late.
            let (log_heights, mains, preprocessed) = preprocessed_batch(num_vars);
            let air = PreprocessedAir;
            let airs = [&air, &air, &air];

            // Transpose each main and preprocessed matrix into the column-major table layout.
            let main_tables = mains
                .iter()
                .map(|main| Table::new(main.transpose()))
                .collect::<Vec<_>>();
            let preprocessed_tables = preprocessed
                .iter()
                .map(|preprocessed| Table::new(preprocessed.transpose()))
                .collect::<Vec<_>>();
            let table_refs = main_tables.iter().collect::<Vec<_>>();
            let preprocessed_refs = preprocessed_tables.iter().map(Some).collect::<Vec<_>>();
            let empty: &[F] = &[];
            let public_values = alloc::vec![empty; 3];

            let zerocheck = AirZerocheck::new(&airs, 0);
            let mut prover_challenger = fresh_challenger();
            let (proof, point) = zerocheck.prove::<F, EF, _>(
                &preprocessed_refs,
                &table_refs,
                &public_values,
                &mut prover_challenger,
            );

            // One opening group per AIR, in caller order.
            assert_eq!(proof.preprocessed_local.len(), 3);
            assert_eq!(proof.preprocessed_next.len(), 3);

            // Each preprocessed opening equals the preprocessed column at that AIR's sub-point.
            for (air_index, &log_height) in log_heights.iter().enumerate() {
                // A shorter AIR binds only the last `log_height` coordinates of the global point.
                let activation_round = num_vars - log_height;
                let sub_point = Point::new(point.as_slice()[activation_round..].to_vec());
                let column = preprocessed_tables[air_index].poly(0);
                assert_eq!(
                    proof.preprocessed_local[air_index],
                    [column.eval_base(&sub_point)]
                );
                assert_eq!(
                    proof.preprocessed_next[air_index],
                    [column.eval_next_base(&sub_point)]
                );
            }

            // The whole batch verifies end to end.
            let mut verifier_challenger = fresh_challenger();
            zerocheck
                .verify::<F, EF, _>(
                    &proof,
                    &log_heights,
                    &public_values,
                    &mut verifier_challenger,
                )
                .expect("preprocessed batch must verify");
        }
    }

    #[test]
    fn staged_zerocheck_rejects_tampered_preprocessed_in_batch() {
        // Invariant: corrupting one AIR's preprocessed opening in a batch must fail the close.
        //
        // Mutation: bump the current-row preprocessed opening of the second tall AIR.
        //
        //     air1 shares the tall stage with air0, so its columns sit at a nonzero offset.
        let num_vars = 4;
        let (log_heights, mains, preprocessed) = preprocessed_batch(num_vars);
        let air = PreprocessedAir;
        let airs = [&air, &air, &air];

        let main_tables = mains
            .iter()
            .map(|main| Table::new(main.transpose()))
            .collect::<Vec<_>>();
        let preprocessed_tables = preprocessed
            .iter()
            .map(|preprocessed| Table::new(preprocessed.transpose()))
            .collect::<Vec<_>>();
        let table_refs = main_tables.iter().collect::<Vec<_>>();
        let preprocessed_refs = preprocessed_tables.iter().map(Some).collect::<Vec<_>>();
        let empty: &[F] = &[];
        let public_values = alloc::vec![empty; 3];

        let zerocheck = AirZerocheck::new(&airs, 0);
        let mut prover_challenger = fresh_challenger();
        let (mut proof, _) = zerocheck.prove::<F, EF, _>(
            &preprocessed_refs,
            &table_refs,
            &public_values,
            &mut prover_challenger,
        );

        // Corrupt the second AIR's preprocessed current-row opening.
        proof.preprocessed_local[1][0] += EF::ONE;

        let mut verifier_challenger = fresh_challenger();
        let err = zerocheck
            .verify::<F, EF, _>(
                &proof,
                &log_heights,
                &public_values,
                &mut verifier_challenger,
            )
            .unwrap_err();
        assert!(matches!(err, ZerocheckError::FinalSumMismatch));
    }

    /// Build a Fibonacci batch of two tall AIRs plus one short AIR.
    ///
    /// The two tall AIRs share a stage; the short one activates a round later.
    fn fib_batch(num_vars: usize) -> ([usize; 3], Vec<RowMajorMatrix<F>>, Vec<Vec<F>>) {
        let log_heights = [num_vars, num_vars, num_vars - 1];
        let traces = log_heights
            .iter()
            .map(|&log_height| fib_trace(1 << log_height))
            .collect::<Vec<_>>();
        let public_values = log_heights
            .iter()
            .map(|&log_height| fib_public_values(1 << log_height).to_vec())
            .collect::<Vec<_>>();
        (log_heights, traces, public_values)
    }

    #[test]
    fn staged_zerocheck_rejects_violated_air_in_batch() {
        // Invariant: one unsatisfied AIR in a batch makes the batched final check reject.
        //
        // Mutation: break a transition row of the second tall AIR, leaving the others valid.
        //
        //     beta batches the AIRs, so a single nonzero constraint sum survives w.h.p.
        let num_vars = 4;
        let (log_heights, mut traces, public_values) = fib_batch(num_vars);

        // Break the transition constraint at row 2 of the second AIR.
        traces[1].values[2 * NUM_COLS] += F::ONE;

        let air = FibAir;
        let airs = [&air, &air, &air];
        let trace_refs = traces.iter().collect::<Vec<_>>();
        let public_refs = public_values
            .iter()
            .map(|values| &values[..])
            .collect::<Vec<_>>();
        let zerocheck = AirZerocheck::new(&airs, 0);

        let mut prover_challenger = fresh_challenger();
        let (proof, _) = prove_traces(
            &zerocheck,
            &trace_refs,
            &public_refs,
            &mut prover_challenger,
        );

        let mut verifier_challenger = fresh_challenger();
        let err = zerocheck
            .verify::<F, EF, _>(&proof, &log_heights, &public_refs, &mut verifier_challenger)
            .unwrap_err();
        assert!(matches!(err, ZerocheckError::FinalSumMismatch));
    }

    #[test]
    fn staged_zerocheck_rejects_tampered_opening_in_late_stage() {
        // Invariant: tampering a late-activating AIR's opening is caught at its own sub-point.
        //
        // Mutation: bump a current-row opening of the short AIR, which activates one round late.
        let num_vars = 4;
        let (log_heights, traces, public_values) = fib_batch(num_vars);

        let air = FibAir;
        let airs = [&air, &air, &air];
        let trace_refs = traces.iter().collect::<Vec<_>>();
        let public_refs = public_values
            .iter()
            .map(|values| &values[..])
            .collect::<Vec<_>>();
        let zerocheck = AirZerocheck::new(&airs, 0);

        let mut prover_challenger = fresh_challenger();
        let (mut proof, _) = prove_traces(
            &zerocheck,
            &trace_refs,
            &public_refs,
            &mut prover_challenger,
        );

        // The short AIR is the last entry; corrupt its first current-row opening.
        proof.local[2][0] += EF::ONE;

        let mut verifier_challenger = fresh_challenger();
        let err = zerocheck
            .verify::<F, EF, _>(&proof, &log_heights, &public_refs, &mut verifier_challenger)
            .unwrap_err();
        assert!(matches!(err, ZerocheckError::FinalSumMismatch));
    }

    #[test]
    #[should_panic = "at least two"]
    fn zerocheck_rejects_height_one_trace() {
        // A single-row trace has zero sumcheck variables, so no stage ever activates.
        // The prover rejects it up front rather than failing deep inside the fold.
        let trace = fib_trace(1);
        let pis = fib_public_values(1);
        let air = FibAir;
        let airs = [&air];
        let zerocheck = AirZerocheck::new(&airs, 0);

        let traces = [&trace];
        let public_values = [&pis[..]];
        let mut challenger = fresh_challenger();
        let _ = prove_traces(&zerocheck, &traces, &public_values, &mut challenger);
    }

    /// Period of the first periodic column.
    const PERIOD_A: usize = 2;
    /// Period of the second periodic column.
    const PERIOD_B: usize = 4;

    /// The two periodic period vectors, of different lengths.
    ///
    /// Column A repeats every two rows.
    /// Column B repeats every four rows.
    fn periodic_columns() -> Vec<Vec<F>> {
        vec![
            vec![F::from_u64(10), F::from_u64(20)],
            vec![
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4),
            ],
        ]
    }

    /// Width-1 main AIR tied to two current-row periodic columns of different periods.
    ///
    /// Every row asserts `main[0] == periodic[0] + periodic[1]`.
    /// The AIR reads no next row.
    /// So it declares an empty main next-row set.
    struct PeriodicAir;

    impl BaseAir<F> for PeriodicAir {
        fn width(&self) -> usize {
            1
        }
        fn num_periodic_columns(&self) -> usize {
            periodic_columns().len()
        }
        fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
            Cow::Owned(periodic_columns())
        }
        fn main_next_row_columns(&self) -> Vec<usize> {
            // Current-row only: no successor claim is needed.
            Vec::new()
        }
    }

    impl<AB: AirBuilder<F = F>> Air<AB> for PeriodicAir {
        fn eval(&self, builder: &mut AB) {
            // Read the single main column and both periodic values at the current row.
            let main = builder.main().current_slice()[0];
            let periodic = builder.periodic_values();
            let sum: AB::Expr = periodic[0].into() + periodic[1].into();
            builder.assert_eq(main, sum);
        }
    }

    /// A satisfying trace: `main[i] = periodic_A[i mod 2] + periodic_B[i mod 4]`.
    fn periodic_trace(n: usize) -> RowMajorMatrix<F> {
        let cols = periodic_columns();
        let values = (0..n)
            .map(|i| cols[0][i % PERIOD_A] + cols[1][i % PERIOD_B])
            .collect();
        RowMajorMatrix::new(values, 1)
    }

    #[test]
    fn zerocheck_accepts_periodic() {
        // Invariant on the periodic AIR, swept over trace heights that both periods divide:
        //   - the closed-form periodic value equals the materialized full-column MLE at the point;
        //   - periodic columns produce no opening claim;
        //   - a satisfying trace proves and verifies.
        //
        // num_vars starts at 2 so the height is a multiple of the larger period (4).
        let air = PeriodicAir;
        let airs = [&air];
        let empty: &[F] = &[];
        let public_values: [&[F]; 1] = [empty];

        for num_vars in 2..10 {
            let n = 1 << num_vars;
            let trace = periodic_trace(n);
            let zerocheck = AirZerocheck::new(&airs, 0);

            let mut prover_challenger = fresh_challenger();
            let (proof, point) = prove_traces(
                &zerocheck,
                &[&trace],
                &public_values,
                &mut prover_challenger,
            );

            // Periodic columns are uncommitted, and this AIR reads no next row.
            // So the single opening group carries no preprocessed and no next-row values.
            assert!(proof.preprocessed_local[0].is_empty());
            assert!(proof.next[0].is_empty());

            // The closed-form value must equal the full-height column folded to the same point.
            let cols = periodic_columns();
            let closed = periodic_evals_at::<F, EF>(&cols, point.as_slice());
            for (col, &value) in cols.iter().zip(closed.iter()) {
                let full = Poly::new((0..n).map(|i| col[i % col.len()]).collect::<Vec<_>>());
                assert_eq!(full.eval_base(&point), value);
            }

            let mut verifier_challenger = fresh_challenger();
            zerocheck
                .verify::<F, EF, _>(
                    &proof,
                    &[num_vars],
                    &public_values,
                    &mut verifier_challenger,
                )
                .expect("periodic AIR must verify");
        }
    }

    #[test]
    fn zerocheck_rejects_violated_periodic_constraint() {
        // Break one row so `main == periodic[0] + periodic[1]` no longer holds.
        // The claimed sum of zero is then false.
        // So the final check must reject.
        let num_vars = 3;
        let n = 1 << num_vars;
        let mut trace = periodic_trace(n);
        trace.values[0] += F::ONE;
        let air = PeriodicAir;
        let airs = [&air];
        let empty: &[F] = &[];
        let public_values: [&[F]; 1] = [empty];
        let zerocheck = AirZerocheck::new(&airs, 0);

        let mut prover_challenger = fresh_challenger();
        let (proof, _) = prove_traces(
            &zerocheck,
            &[&trace],
            &public_values,
            &mut prover_challenger,
        );

        let mut verifier_challenger = fresh_challenger();
        let err = zerocheck
            .verify::<F, EF, _>(
                &proof,
                &[num_vars],
                &public_values,
                &mut verifier_challenger,
            )
            .unwrap_err();
        assert!(matches!(err, ZerocheckError::FinalSumMismatch));
    }
}
