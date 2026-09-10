//! Provers for the stacked layout, specialised per residual-sumcheck binding mode.
//!
//! # Modules
//!
//! - Prefix prover: SVO-accumulator preprocessing, packed handoff.
//! - Suffix prover: SVO-accumulator preprocessing, unpacked handoff.

mod claims;
mod prefix;
mod suffix;

use alloc::vec::Vec;

pub use claims::StackedClaims;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{Encoder, Mmcs};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::DenseMatrix;
use p3_multilinear_util::point::Point;
pub use prefix::PrefixProver;
pub use suffix::SuffixProver;

use crate::SumcheckData;
use crate::commit::commit_base;
use crate::layout::transcript::{
    BatchingShape, LayoutBinding, OpeningProverTranscript, OpeningShape, PointSource,
    VirtualProverTranscript, VirtualShape, prover_batching_challenge,
};
use crate::layout::{LayoutStrategy, Table, Witness};
use crate::strategy::{SumcheckProver, VariableOrder};
use crate::table::{OpeningEvals, OpeningRequest};

/// Stacked-sumcheck prover layout
pub trait Layout<F: Field, EF: ExtensionField<F>>: Sized {
    /// Builds this layout from a committed witness.
    fn from_witness(witness: Witness<F>) -> Self;

    /// Builds a witness structure for this layout from source tables.
    fn new_witness(tables: Vec<Table<F>>, folding: usize) -> Witness<F>;

    /// Returns the shared claim state recorded against the stacked polynomial.
    fn claims(&self) -> &StackedClaims<F, EF>;

    /// Commits to the witness and returns the layout.
    ///
    /// # Arguments
    ///
    /// - `encoder`                — linear code used to encode the codeword.
    /// - `mmcs`                   — Merkle commitment scheme over the base field.
    /// - `challenger`             — Fiat–Shamir transcript; absorbs the Merkle root.
    /// - `witness`                — stacked committed polynomial plus its tables.
    /// - `folding`                — folding factor consumed by the first WHIR round.
    /// - `starting_log_inv_rate`  — initial log-inverse rate of the RS code.
    ///
    /// # Transcript
    ///
    /// The default body absorbs exactly one value, the Merkle root it returns.
    ///
    /// An override owes the sponge that same single absorb, and no other absorb, sample or grind.
    ///
    /// A verifier never reaches this method, so it absorbs that one root in its place.
    /// An absorbed table height would desync the two sides with no step out of place on either.
    fn commit<E, MT, Challenger>(
        encoder: &E,
        mmcs: &MT,
        challenger: &mut Challenger,
        witness: Witness<F>,
        folding: usize,
        starting_log_inv_rate: usize,
    ) -> (Self, MT::Commitment, MT::ProverData<DenseMatrix<F>>)
    where
        E: Encoder<F>,
        MT: Mmcs<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        // Encode and Merkle-commit the stacked polynomial in the mode's variable order.
        let (root, prover_data) = commit_base(
            Self::variable_order(),
            encoder,
            mmcs,
            challenger,
            &witness.poly,
            folding,
            starting_log_inv_rate,
        );

        // The witness is consumed into the layout once its codeword is committed.
        (Self::from_witness(witness), root, prover_data)
    }

    /// Returns the total number of concrete openings recorded so far.
    fn num_claims(&self) -> usize {
        self.claims().num_claims()
    }

    /// Returns the number of out-of-domain claims recorded on the stacked polynomial.
    fn num_virtual_claims(&self) -> usize {
        self.claims().num_virtual_claims()
    }

    /// Returns the verifier strategy required to replay this committed layout.
    fn strategy() -> LayoutStrategy;

    /// Returns the variable order.
    fn variable_order() -> VariableOrder {
        Self::strategy().variable_order
    }

    /// Returns the number of variables of first round
    fn folding(&self) -> usize {
        self.claims().folding()
    }

    /// Returns the number of variables of the stacked polynomial.
    fn num_variables(&self) -> usize {
        self.claims().num_variables()
    }

    /// Returns the number of variables of table `id`.
    fn num_variables_table(&self, id: usize) -> usize {
        self.claims().num_variables_table(id)
    }

    /// Returns source table `id`.
    fn table(&self, id: usize) -> &Table<F> {
        self.claims().table(id)
    }

    /// Records opening claims for the selected columns of one table at a drawn point.
    ///
    /// # Overview
    ///
    /// - The local-frame opening point is drawn from the transcript.
    /// - Direct openings evaluate a column at that point.
    /// - Successor openings evaluate the repeat-last view at the same point.
    /// - The returned evaluations list every direct opening first.
    ///
    /// # Transcript
    ///
    /// One sub-transcript spans the whole call.
    ///
    /// ```text
    ///     draw the point  ->  evaluate  ->  bind the evaluations
    /// ```
    ///
    /// The point is drawn before the values it is evaluated at exist.
    ///
    /// The values are bound before the surrounding protocol draws anything else.
    ///
    /// # Arguments
    ///
    /// - Index of the table whose columns are opened.
    /// - Column indices opened directly and through the successor view.
    /// - Sponge of the surrounding protocol, borrowed for this call.
    ///
    /// # Panics
    ///
    /// When the request names no column at all.
    fn eval<Ch>(
        &mut self,
        table_idx: usize,
        batch: &OpeningRequest,
        challenger: &mut Ch,
    ) -> OpeningEvals<EF>
    where
        F: TranscriptField,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Opening nothing would silently record an empty claim.
        //
        // The schedule is the caller's own.
        //
        // An empty request is therefore a caller bug.
        assert!(
            !batch.is_empty(),
            "opening schedule must name at least one column"
        );

        // Both column counts come from the schedule, never from a value in flight.
        let shape = OpeningShape::new(
            LayoutBinding::new(self.num_variables(), Self::strategy()),
            table_idx,
            self.num_variables_table(table_idx),
            batch.current().len(),
            batch.next().len(),
            PointSource::Drawn,
        );
        let mut transcript = OpeningProverTranscript::<Ch, F, EF>::new(challenger, shape);

        // The standalone-PCS convention: the verifier picks the evaluation point.
        let point = transcript.point();

        // Evaluate at the freshly drawn point, then bind what was found.
        let evals = self.record_opening(table_idx, batch, &point);
        transcript.evaluations(&evals);

        // Require that every described step was played.
        transcript.finish();

        evals
    }

    /// Records opening claims for the selected columns of one table at a given point.
    ///
    /// The caller supplies the local-frame opening point instead of drawing one.
    ///
    /// An outer protocol that fixes the point opens its columns here.
    ///
    /// # Overview
    ///
    /// - Direct openings evaluate a column at the supplied point.
    /// - Successor openings evaluate the repeat-last view at the same point.
    /// - The returned evaluations list every direct opening first.
    ///
    /// # Soundness
    ///
    /// This call binds the evaluations.
    ///
    /// It binds nothing else.
    ///
    /// The point must already have been drawn from, or bound to, the same sponge.
    ///
    /// A caller choosing its own point owes the transcript that binding.
    ///
    /// # Arguments
    ///
    /// - Index of the table whose columns are opened.
    /// - Column indices opened directly and through the successor view.
    /// - Local-frame opening point.
    /// - Sponge of the surrounding protocol, borrowed for this call.
    ///
    /// # Panics
    ///
    /// When the request names no column at all.
    fn eval_at<Ch>(
        &mut self,
        table_idx: usize,
        batch: &OpeningRequest,
        point: &Point<EF>,
        challenger: &mut Ch,
    ) -> OpeningEvals<EF>
    where
        F: TranscriptField,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Opening nothing would silently record an empty claim.
        assert!(
            !batch.is_empty(),
            "opening schedule must name at least one column"
        );

        // A caller-fixed point contributes no step.
        //
        // This description therefore holds no challenge.
        let shape = OpeningShape::new(
            LayoutBinding::new(self.num_variables(), Self::strategy()),
            table_idx,
            self.num_variables_table(table_idx),
            batch.current().len(),
            batch.next().len(),
            PointSource::Given,
        );
        let mut transcript = OpeningProverTranscript::<Ch, F, EF>::new(challenger, shape);

        // Evaluate at the supplied point, then bind what was found.
        let evals = self.record_opening(table_idx, batch, point);
        transcript.evaluations(&evals);

        // Require that every described step was played.
        transcript.finish();

        evals
    }

    /// Evaluates the selected columns of one table and records the resulting claim.
    ///
    /// This is the arithmetic half of an opening, with no transcript of its own.
    ///
    /// # Overview
    ///
    /// - Direct openings evaluate a column at the supplied point.
    /// - Successor openings evaluate the repeat-last view at the same point.
    /// - The claim is appended to this table's list in insertion order.
    ///
    /// # Returns
    ///
    /// Every direct evaluation first, then every successor evaluation.
    ///
    /// # Arguments
    ///
    /// - Index of the table whose columns are opened.
    /// - Column indices opened directly and through the successor view.
    /// - Local-frame opening point, one coordinate per table variable.
    fn record_opening(
        &mut self,
        table_idx: usize,
        batch: &OpeningRequest,
        point: &Point<EF>,
    ) -> OpeningEvals<EF>;

    /// Records an out-of-domain evaluation of the full stacked polynomial.
    ///
    /// # Overview
    ///
    /// WHIR pins the stacked polynomial at a fresh point for soundness amplification.
    ///
    /// The point covers every stacked variable.
    ///
    /// The claim is therefore not tied to any one column.
    ///
    /// # Transcript
    ///
    /// One sub-transcript spans the whole call.
    ///
    /// ```text
    ///     draw the point  ->  evaluate  ->  bind the evaluation
    /// ```
    ///
    /// # Arguments
    ///
    /// - Sponge of the surrounding protocol, borrowed for this call.
    ///
    /// # Returns
    ///
    /// The claimed evaluation.
    ///
    /// The caller sends it in its own proof.
    fn add_virtual_eval<Ch>(&mut self, challenger: &mut Ch) -> EF
    where
        F: TranscriptField,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // The stacked arity is the whole configuration of this component.
        let shape = VirtualShape::new(LayoutBinding::new(self.num_variables(), Self::strategy()));
        let mut transcript = VirtualProverTranscript::<Ch, F, EF>::new(challenger, shape);

        // Draw first.
        //
        // The claimed value can then not be chosen to suit the point.
        let point = transcript.point();

        // Evaluate at the drawn point, then bind what was found.
        let eval = self.record_virtual(&point);
        transcript.evaluation(eval);

        // Require that every described step was played.
        transcript.finish();

        eval
    }

    /// Evaluates the stacked polynomial at a point and records the resulting claim.
    ///
    /// The arithmetic half of an out-of-domain claim.
    ///
    /// It keeps no transcript of its own.
    ///
    /// # Arguments
    ///
    /// - Point covering every stacked variable.
    ///
    /// # Returns
    ///
    /// The stacked polynomial's evaluation at that point.
    fn record_virtual(&mut self, point: &Point<EF>) -> EF;

    /// Draws the challenge that collapses every recorded claim into a single one.
    ///
    /// # Overview
    ///
    /// Each claim is weighted by a successive power of the drawn challenge.
    ///
    /// ```text
    ///     sum = sum_i  alpha^i * eval_i
    /// ```
    ///
    /// Direct and successor openings take the low powers, in insertion order.
    ///
    /// Out-of-domain claims continue the sequence after them.
    ///
    /// # Transcript
    ///
    /// One sub-transcript spans the draw.
    ///
    /// Both claim counts reach its seed.
    ///
    /// A run that recorded a different number of claims lands elsewhere.
    ///
    /// # Arguments
    ///
    /// - Sponge of the surrounding protocol, borrowed for the draw.
    fn batching_challenge<Ch>(&self, challenger: &mut Ch) -> EF
    where
        F: TranscriptField,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Both counts come from the claims this run recorded, never from a proof.
        let shape = BatchingShape::new(
            LayoutBinding::new(self.num_variables(), Self::strategy()),
            self.num_claims(),
            self.num_virtual_claims(),
        );
        prover_batching_challenge::<Ch, F, EF>(challenger, shape)
    }

    /// Processes initial rounds of sumcheck and returns the residual sumcheck prover.
    ///
    /// # Returns
    ///
    /// - Residual sumcheck prover over the unpacked product polynomial.
    /// - Folding challenges sampled during preprocessing.
    fn into_sumcheck<Ch>(
        self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> (SumcheckProver<F, EF>, Point<EF>)
    where
        F: TranscriptField,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>;
}

#[cfg(test)]
pub(super) mod test_utils {
    //! Shared fixtures and end-to-end roundtrip runners used by both mode tests.
    //!
    //! - Fixed two-table witness reused by every scenario.
    //! - Per-mode runners that drive prover and verifier side-by-side.
    //! - Proptest strategy for random opening schedules.

    use alloc::vec;
    use alloc::vec::Vec;

    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::point::Point;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::SumcheckData;
    use crate::layout::{Layout, LayoutStrategy, Table, TableShape, Verifier, Witness};
    use crate::strategy::{Basis, VariableOrder};
    use crate::table::{OpeningBatch, OpeningEvals, OpeningRequest};
    use crate::tests::*;

    /// Preprocessing rounds each end-to-end test consumes on both sides.
    pub(crate) const FOLDING: usize = 4;
    /// Proof-of-work difficulty for the intermediate sumcheck phase.
    pub(crate) const POW_BITS: usize = 10;
    /// STIR-style constraint points sampled for the intermediate round.
    pub(crate) const ROUND_EQ_POINTS: usize = 10;
    /// STIR-style selector points sampled for the intermediate round.
    pub(crate) const ROUND_SEL_POINTS: usize = 100;

    /// Opening schedule with columns in ascending order inside each claim.
    pub(crate) const ASCENDING_POLYS: &[(usize, &[usize])] =
        &[(0, &[0, 1]), (0, &[0]), (1, &[0, 1]), (1, &[1])];

    /// Opening schedule with non-ascending columns in the first claim.
    ///
    /// Targets the alpha / partial-eval misalignment that used to bite suffix mode.
    pub(crate) const NON_ASCENDING_POLYS: &[(usize, &[usize])] =
        &[(0, &[1, 0]), (0, &[0]), (1, &[0, 1]), (1, &[1])];

    /// Builds the fixed two-table witness shared by every roundtrip test.
    ///
    /// # Layout
    ///
    /// - Table 0: arity 9, two columns.
    /// - Table 1: arity 10, two columns.
    pub(crate) fn build_tables() -> Vec<Table<F>> {
        let mut rng = SmallRng::seed_from_u64(1);
        // Table at index 1 in the insertion order: arity 10, two columns.
        let a = Table::rand(&mut rng, 2, 10);
        // Table at index 0 in the insertion order: arity 9, two columns.
        let b = Table::rand(&mut rng, 2, 9);
        vec![b, a]
    }

    /// Returns the per-table shape used by the verifier side.
    pub(crate) fn table_shapes() -> Vec<TableShape> {
        vec![TableShape::new(9, 2), TableShape::new(10, 2)]
    }

    /// Proptest strategy: random opening schedules over the fixed witness.
    ///
    /// # Shape
    ///
    /// - 1..=6 calls in total.
    /// - Each call picks a random table index (0 or 1) and a non-empty
    ///   subset of its 2 columns, possibly in any order.
    pub(crate) fn arb_opening_schedule() -> impl Strategy<Value = Vec<(usize, Vec<usize>)>> {
        // One call: (table_idx in {0, 1}, random non-empty permutation of [0, 1]).
        let one_call =
            (0usize..2, prop::collection::vec(0usize..2, 1..=2)).prop_map(|(table_idx, polys)| {
                // Deduplicate while preserving first-seen order so the opening
                // list stays valid (columns open at most once per claim).
                let mut seen = [false; 2];
                let dedup: Vec<usize> = polys
                    .into_iter()
                    .filter(|&p| {
                        let first = !seen[p];
                        seen[p] = true;
                        first
                    })
                    .collect();
                (table_idx, dedup)
            });
        prop::collection::vec(one_call, 1..=6)
    }

    /// Runs the verifier side and the final batched-sum check.
    ///
    /// Mode-agnostic: the binding direction is passed in as a runtime tag used
    /// only to evaluate the batched-constraint polynomial. Every caller-provided
    /// value comes straight from the prover side.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn verify_roundtrip(
        strategy: LayoutStrategy,
        shapes: &[TableShape],
        stacked_num_variables: usize,
        opening_claims: Vec<(usize, OpeningRequest, OpeningEvals<EF>)>,
        virtual_eval: EF,
        proof0: &SumcheckData<F, EF>,
        proof1: &SumcheckData<F, EF>,
        proof2: &SumcheckData<F, EF>,
        intermediate_evals: &[EF],
        expected_randomness: &Point<EF>,
        final_folded_value: EF,
    ) {
        // Fresh challenger: verifier must stay in lockstep with the prover transcript.
        let mut verifier_challenger = challenger();
        let mut verifier: Verifier<F, EF> = Verifier::new(shapes, strategy);

        // Re-sample the same opening points and record the claimed evaluations.
        // `add_claim` samples the point + absorbs the evals internally, mirroring
        // the prover's `eval`.
        for (table_idx, batch, evals) in opening_claims {
            verifier
                .add_claim(table_idx, &batch, &evals, &mut verifier_challenger)
                .unwrap();
        }

        // Re-sample the virtual claim too; mirrors the prover's `add_virtual_eval`.
        verifier.add_virtual_eval(virtual_eval, &mut verifier_challenger);

        // Draw the batching challenge through the layout, exactly as the prover does.
        let alpha = verifier.batching_challenge(&mut verifier_challenger);
        // Build the initial constraint and seed the running sum from it.
        let initial_constraint = verifier.constraint(alpha);
        let mut sum = EF::ZERO;
        initial_constraint.combine_evals(&mut sum);
        assert_eq!(sum, verifier.sum(alpha));

        // Drive the three-proof transcript verification.
        let mut constraints = vec![initial_constraint];
        let mut verifier_challenge = Point::new(vec![]);
        verifier_challenge.extend(
            &proof0
                .verify_rounds(
                    &mut verifier_challenger,
                    &mut sum,
                    FOLDING,
                    0,
                    Basis::Evaluation,
                )
                .unwrap(),
        );

        let intermediate_constraint = read_constraint(
            &mut verifier_challenger,
            intermediate_evals,
            stacked_num_variables - FOLDING,
            ROUND_EQ_POINTS,
            ROUND_SEL_POINTS,
        );
        intermediate_constraint.combine_evals(&mut sum);
        constraints.push(intermediate_constraint);
        verifier_challenge.extend(
            &proof1
                .verify_rounds(
                    &mut verifier_challenger,
                    &mut sum,
                    FOLDING,
                    POW_BITS,
                    Basis::Evaluation,
                )
                .unwrap(),
        );
        verifier_challenge.extend(
            &proof2
                .verify_rounds(
                    &mut verifier_challenger,
                    &mut sum,
                    stacked_num_variables - 2 * FOLDING,
                    0,
                    Basis::Evaluation,
                )
                .unwrap(),
        );

        // Final invariants:
        //     - Prover and verifier agreed on the same randomness.
        //     - Batched sum equals the final folded value times the batched weights.
        assert_eq!(expected_randomness, &verifier_challenge);
        let weights = strategy
            .variable_order
            .eval_constraints_poly(&constraints, &verifier_challenge);
        assert_eq!(sum, final_folded_value * weights);
    }

    /// Drives the intermediate + final sumcheck phases on the residual prover.
    ///
    /// Returns the two extra transcript chunks, the intermediate evals, and
    /// the final folded value. Extends `prover_randomness` with every
    /// challenge sampled during these two phases.
    pub(crate) fn drive_intermediate_and_final(
        prover: &mut crate::strategy::SumcheckProver<F, EF>,
        prover_challenger: &mut MyChallenger,
        prover_randomness: &mut Point<EF>,
        stacked_num_variables: usize,
    ) -> (SumcheckData<F, EF>, SumcheckData<F, EF>, Vec<EF>, EF) {
        // Intermediate phase: build a STIR-style constraint and absorb it.
        let mut intermediate_evals: Vec<EF> = Vec::new();
        let constraint = make_constraint_ext(
            prover_challenger,
            &mut intermediate_evals,
            prover.num_variables(),
            ROUND_EQ_POINTS,
            ROUND_SEL_POINTS,
            &prover.evals(),
        );

        let mut proof1 = SumcheckData::<F, EF>::default();
        prover_randomness.extend(&prover.compute_sumcheck_polynomials(
            &mut proof1,
            prover_challenger,
            FOLDING,
            POW_BITS,
            Some(constraint),
        ));
        let remaining_vars = stacked_num_variables - FOLDING - FOLDING;
        assert_eq!(proof1.num_rounds(), FOLDING);
        assert_eq!(prover.num_variables(), remaining_vars);

        // Final phase: fold the remaining variables down to a constant.
        let mut proof2 = SumcheckData::<F, EF>::default();
        prover_randomness.extend(&prover.compute_sumcheck_polynomials(
            &mut proof2,
            prover_challenger,
            remaining_vars,
            0,
            None,
        ));
        assert_eq!(proof2.num_rounds(), remaining_vars);
        assert_eq!(prover.num_variables(), 0);

        let final_folded_value = prover.evals().as_constant().unwrap();
        (proof1, proof2, intermediate_evals, final_folded_value)
    }

    /// Replays an opening schedule against a prover.
    ///
    /// Returns one triple of table index, opening request, and evaluations per scheduled call.
    ///
    /// The prover samples the point and absorbs the evaluations as a side effect.
    ///
    /// This keeps the transcript in lockstep with the verifier replay on the other side.
    fn replay_schedule<F>(
        calls: &[(usize, &[usize])],
        mut step: F,
    ) -> Vec<(usize, OpeningRequest, OpeningEvals<EF>)>
    where
        F: FnMut(usize, &OpeningRequest) -> OpeningEvals<EF>,
    {
        calls
            .iter()
            .map(|&(table_idx, polys)| {
                // This helper only exercises current openings, so the next group is empty.
                let batch = OpeningBatch::new(polys.to_vec(), Vec::new());
                // Drive the prover, which samples and absorbs through the transcript.
                let evals = step(table_idx, &batch);
                // Keep the request alongside its evals so the verifier can mirror the draws.
                (table_idx, batch, evals)
            })
            .collect()
    }

    /// Runs the full mode-generic roundtrip with verifier metadata from the layout.
    pub(crate) fn run_roundtrip_test<L>(
        witness: Witness<F>,
        shapes: &[TableShape],
        calls: &[(usize, &[usize])],
    ) where
        L: Layout<F, EF>,
    {
        let mut prover_challenger = challenger();
        let stacked_num_variables = witness.num_variables();
        // Snapshot the stacked polynomial before the witness is consumed.
        let stacked_poly = witness.poly().clone();

        // Prover: build the selected layout, record openings, add a virtual claim.
        let mut prover_state = L::from_witness(witness);
        let strategy = L::strategy();
        let order = strategy.variable_order;
        let opening_claims = replay_schedule(calls, |t, batch| {
            prover_state.eval(t, batch, &mut prover_challenger)
        });
        let virtual_eval = prover_state.add_virtual_eval(&mut prover_challenger);

        // Preprocessing: consume FOLDING rounds, hand off the residual prover.
        let mut proof0 = SumcheckData::<F, EF>::default();
        let (mut prover, mut prover_randomness) =
            prover_state.into_sumcheck(&mut proof0, 0, &mut prover_challenger);
        assert_eq!(proof0.num_rounds(), FOLDING);
        assert_eq!(prover.num_variables(), stacked_num_variables - FOLDING);

        // Intermediate + final rounds (mode-agnostic once the residual prover exists).
        let (proof1, proof2, intermediate_evals, final_folded_value) = drive_intermediate_and_final(
            &mut prover,
            &mut prover_challenger,
            &mut prover_randomness,
            stacked_num_variables,
        );

        let final_eval = match order {
            VariableOrder::Prefix => stacked_poly.eval_base(&prover_randomness),
            VariableOrder::Suffix => stacked_poly.eval_base(&prover_randomness.reversed()),
        };
        assert_eq!(final_eval, final_folded_value);

        verify_roundtrip(
            strategy,
            shapes,
            stacked_num_variables,
            opening_claims,
            virtual_eval,
            &proof0,
            &proof1,
            &proof2,
            &intermediate_evals,
            &prover_randomness,
            final_folded_value,
        );
    }

    /// Minimum source-table arity used by the shape proptest.
    ///
    /// # Constraints
    ///
    /// - Must be at least the preprocessing depth (every table arity >= FOLDING).
    /// - Must be at least `log2(packing_width)` so prefix mode accepts it.
    ///   BabyBear packing widths on current targets peak at 16, giving `k_pack = 4`.
    const SHAPE_MIN_ARITY: usize = 4;

    /// Upper bound on per-table arity in the shape proptest.
    const SHAPE_MAX_ARITY: usize = 8;

    /// Upper bound on the number of columns per table.
    const SHAPE_MAX_COLUMNS: usize = 3;

    /// Upper bound on the number of tables per witness.
    const SHAPE_MAX_TABLES: usize = 3;

    /// Upper bound on the number of opening calls in the generated schedule.
    const SHAPE_MAX_CALLS: usize = 5;

    /// One `(arity, column_count)` pair per source table.
    pub(crate) type WitnessShape = Vec<(usize, usize)>;

    /// One `(table_index, column_subset)` pair per opening call.
    pub(crate) type OpeningSchedule = Vec<(usize, Vec<usize>)>;

    /// Builds a deterministic witness matching the given `(arity, column_count)` shape.
    ///
    /// # Arguments
    ///
    /// - `shape` — one `(arity, column_count)` pair per source table.
    pub(crate) fn tables_from_shape(shape: &[(usize, usize)]) -> Vec<Table<F>> {
        // Fixed seed: every proptest case gets reproducible polynomial evaluations.
        let mut rng = SmallRng::seed_from_u64(42);

        // One table per (arity, column_count) pair; each column is a random polynomial.
        shape
            .iter()
            .map(|&(arity, num_cols)| Table::rand(&mut rng, num_cols, arity))
            .collect()
    }

    /// Mirrors a `(arity, column_count)` shape onto the verifier-side table shapes.
    pub(crate) fn table_shapes_from(shape: &[(usize, usize)]) -> Vec<TableShape> {
        shape
            .iter()
            .map(|&(arity, num_cols)| TableShape::new(arity, num_cols))
            .collect()
    }

    /// Proptest strategy: random witness shape paired with a valid opening schedule.
    ///
    /// # Shape
    ///
    /// - 1..=`SHAPE_MAX_TABLES` source tables.
    /// - Each table: arity in `[SHAPE_MIN_ARITY, SHAPE_MAX_ARITY]`, column count
    ///   in `[1, SHAPE_MAX_COLUMNS]`.
    ///
    /// # Schedule
    ///
    /// - 1..=`SHAPE_MAX_CALLS` opening calls over the generated witness.
    /// - Every call picks an existing table index and a non-empty, de-duplicated
    ///   subset of that table's columns (columns may appear in any order).
    pub(crate) fn arb_witness_and_schedule()
    -> impl Strategy<Value = (WitnessShape, OpeningSchedule)> {
        // Step 1: pick the witness shape.
        //
        // Stacked arity must accommodate two phases of FOLDING rounds plus the
        // final fold-to-constant phase, so total size must exceed `2^(2 * FOLDING - 1)`.
        let shape = prop::collection::vec(
            (
                SHAPE_MIN_ARITY..=SHAPE_MAX_ARITY,
                1usize..=SHAPE_MAX_COLUMNS,
            ),
            1..=SHAPE_MAX_TABLES,
        )
        .prop_filter(
            "stacked polynomial must support two phases of FOLDING rounds",
            |shape| {
                let total: usize = shape.iter().map(|&(a, c)| (1usize << a) * c).sum();
                // log2_ceil(total) >= 2 * FOLDING.
                total > (1usize << (2 * FOLDING - 1))
            },
        );

        shape.prop_flat_map(|shape| {
            // Step 2: given the shape, pick a schedule that respects it.
            let num_tables = shape.len();
            let shape_for_calls = shape.clone();

            // One call: pick a table, then a non-empty unique column subset.
            let one_call = (0..num_tables).prop_flat_map(move |t_idx| {
                let num_cols = shape_for_calls[t_idx].1;
                // Draw a random column sequence of length 1..=num_cols, then deduplicate
                // while preserving first-seen order so the opening list stays valid.
                prop::collection::vec(0..num_cols, 1..=num_cols).prop_map(move |cols| {
                    let mut seen = vec![false; num_cols];
                    let dedup: Vec<usize> = cols
                        .into_iter()
                        .filter(|&c| {
                            let first = !seen[c];
                            seen[c] = true;
                            first
                        })
                        .collect();
                    (t_idx, dedup)
                })
            });

            // Step 3: bundle the schedule back with its originating shape.
            prop::collection::vec(one_call, 1..=SHAPE_MAX_CALLS)
                .prop_map(move |sched| (shape.clone(), sched))
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;

    use super::test_utils::{
        ASCENDING_POLYS, NON_ASCENDING_POLYS, POW_BITS, ROUND_EQ_POINTS, ROUND_SEL_POINTS,
        arb_opening_schedule, arb_witness_and_schedule, drive_intermediate_and_final,
        table_shapes_from,
    };
    use super::{PrefixProver, SuffixProver};
    use crate::SumcheckData;
    use crate::layout::prover::test_utils::{
        FOLDING, build_tables, run_roundtrip_test, table_shapes, tables_from_shape,
    };
    use crate::layout::{Layout, Verifier};
    use crate::strategy::Basis;
    use crate::table::{OpeningBatch, OpeningEvals};
    use crate::tests::*;

    #[test]
    fn num_claims_counts_every_recorded_opening() {
        fn run_num_claims_test_with<L>(witness: crate::layout::Witness<F>)
        where
            L: Layout<F, EF>,
        {
            let mut prover = L::from_witness(witness);
            assert_eq!(prover.num_claims(), 0);

            let mut ch = challenger();
            prover.eval(0, &OpeningBatch::new(vec![0, 1], Vec::new()), &mut ch);
            assert_eq!(prover.num_claims(), 2);

            prover.eval(1, &OpeningBatch::new(vec![0], Vec::new()), &mut ch);
            assert_eq!(prover.num_claims(), 3);
        }

        run_num_claims_test_with::<SuffixProver<F, EF>>(SuffixProver::<F, EF>::new_witness(
            build_tables(),
            FOLDING,
        ));
        run_num_claims_test_with::<PrefixProver<F, EF>>(PrefixProver::<F, EF>::new_witness(
            build_tables(),
            FOLDING,
        ));
    }

    #[test]
    fn eval_current_preserves_order() {
        // Invariant: returned evals follow the requested column order, not a sorted order.
        fn run_eval_current_test_with<L>()
        where
            L: Layout<F, EF>,
        {
            // Two identical provers over the same tables and folding depth.
            let mut prover = L::from_witness(L::new_witness(build_tables(), FOLDING));
            let mut reversed = L::from_witness(L::new_witness(build_tables(), FOLDING));

            // Independent transcripts seeded identically, so draws match.
            let mut prover_ch = challenger();
            let mut reversed_ch = challenger();

            // Request columns [1, 0]: evals must come back in that exact order.
            let evals = prover.eval(
                0,
                &OpeningBatch::new(vec![1, 0], Vec::new()),
                &mut prover_ch,
            );
            // Request the same columns in swapped order [0, 1].
            let reversed_evals = reversed.eval(
                0,
                &OpeningBatch::new(vec![0, 1], Vec::new()),
                &mut reversed_ch,
            );

            // Swapping the request order swaps the eval order:
            //     [eval(col 1), eval(col 0)] == reverse([eval(col 0), eval(col 1)])
            assert_eq!(
                evals.to_vec(),
                reversed_evals
                    .to_vec()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
            );
            // Both record the same number of claims regardless of order.
            assert_eq!(prover.num_claims(), reversed.num_claims());
        }

        // Exercise both binding orders.
        run_eval_current_test_with::<SuffixProver<F, EF>>();
        run_eval_current_test_with::<PrefixProver<F, EF>>();
    }

    #[test]
    fn prefix_eval_accepts_next() {
        // Invariant: a batch with only a next opening (no current openings) is accepted.
        let mut prover = PrefixProver::<F, EF>::from_witness(PrefixProver::<F, EF>::new_witness(
            build_tables(),
            FOLDING,
        ));
        let mut ch = challenger();

        // Request: current = [], next = [col 0]  → one next opening, zero current.
        let evals = prover.eval(0, &OpeningBatch::new(Vec::new(), vec![0]), &mut ch);
        // One next opening yields one eval and records one claim.
        assert_eq!(evals.len(), 1);
        assert_eq!(prover.num_claims(), 1);
    }

    #[test]
    fn roundtrip_ascending_polys() {
        run_roundtrip_test::<PrefixProver<F, EF>>(
            PrefixProver::<F, EF>::new_witness(build_tables(), FOLDING),
            &table_shapes(),
            ASCENDING_POLYS,
        );

        run_roundtrip_test::<SuffixProver<F, EF>>(
            SuffixProver::<F, EF>::new_witness(build_tables(), FOLDING),
            &table_shapes(),
            ASCENDING_POLYS,
        );
    }

    #[test]
    fn roundtrip_non_ascending_polys() {
        run_roundtrip_test::<PrefixProver<F, EF>>(
            PrefixProver::<F, EF>::new_witness(build_tables(), FOLDING),
            &table_shapes(),
            NON_ASCENDING_POLYS,
        );

        run_roundtrip_test::<SuffixProver<F, EF>>(
            SuffixProver::<F, EF>::new_witness(build_tables(), FOLDING),
            &table_shapes(),
            NON_ASCENDING_POLYS,
        );
    }

    #[test]
    fn suffix_roundtrip_mixed_eq_next_requests() {
        // Invariant: a full prove/verify roundtrip agrees when batches mix
        // current and next openings, under suffix-first variable binding.
        let witness = SuffixProver::<F, EF>::new_witness(build_tables(), FOLDING);
        let shapes = table_shapes();
        let stacked_num_variables = witness.num_variables();
        // Keep a copy of the stacked polynomial to cross-check the final fold.
        let stacked_poly = witness.poly().clone();
        let strategy = SuffixProver::<F, EF>::strategy();

        // Mixed schedule: each tuple is (table, current columns, next columns).
        //     table 0: current [0], next [0, 1]
        //     table 1: current [],  next [0]
        //     table 0: current [1], next [0]
        let schedule = [
            (0, vec![0], vec![0, 1]),
            (1, vec![], vec![0]),
            (0, vec![1], vec![0]),
        ];

        // Prover side: sample points and absorb evals through the transcript.
        let mut prover_challenger = challenger();
        let mut prover_state = SuffixProver::<F, EF>::from_witness(witness);
        let opening_claims = schedule
            .iter()
            .map(|(table_idx, current, next)| {
                // Pack current and next column requests into one batch.
                let batch = OpeningBatch::new(current.clone(), next.clone());
                let evals = prover_state.eval(*table_idx, &batch, &mut prover_challenger);
                // Retain request and evals so the verifier can replay identically.
                (*table_idx, batch, evals)
            })
            .collect::<Vec<_>>();
        // One virtual claim over the full stacked polynomial.
        let virtual_eval = prover_state.add_virtual_eval(&mut prover_challenger);

        // First sumcheck stage folds the SVO rounds and writes their proof.
        let mut proof0 = SumcheckData::<F, EF>::default();
        let (mut prover, mut prover_randomness) =
            prover_state.into_sumcheck(&mut proof0, 0, &mut prover_challenger);
        // The first stage consumes exactly the folding-depth rounds.
        assert_eq!(proof0.num_rounds(), FOLDING);
        assert_eq!(prover.num_variables(), stacked_num_variables - FOLDING);

        // Remaining stages drive the prover to a single folded value.
        let (proof1, proof2, intermediate_evals, final_folded_value) = drive_intermediate_and_final(
            &mut prover,
            &mut prover_challenger,
            &mut prover_randomness,
            stacked_num_variables,
        );

        // Suffix binding reverses the challenge order for the direct evaluation check.
        let final_eval = stacked_poly.eval_base(&prover_randomness.reversed());
        assert_eq!(final_eval, final_folded_value);

        // Verifier side: fresh transcript, mirror every prover absorption.
        let mut verifier_challenger = challenger();
        let mut verifier = Verifier::<F, EF>::new(&shapes, strategy);
        for (table_idx, batch, evals) in opening_claims {
            verifier
                .add_claim(table_idx, &batch, &evals, &mut verifier_challenger)
                .unwrap();
        }
        verifier.add_virtual_eval(virtual_eval, &mut verifier_challenger);

        // Batching challenge and the initial constraint over all recorded claims.
        // The verifier draws it through the layout, exactly as the prover does.
        let alpha = verifier.batching_challenge(&mut verifier_challenger);
        let initial_constraint = verifier.constraint(alpha);
        let mut sum = EF::ZERO;
        initial_constraint.combine_evals(&mut sum);
        // The constraint's combined value must equal the alpha-batched claim sum.
        assert_eq!(sum, verifier.sum(alpha));

        // Collect each stage's constraint and replay the challenges it folded.
        let mut constraints = vec![initial_constraint];
        let mut verifier_challenge = Point::new(vec![]);
        verifier_challenge.extend(
            &proof0
                .verify_rounds(
                    &mut verifier_challenger,
                    &mut sum,
                    FOLDING,
                    0,
                    Basis::Evaluation,
                )
                .unwrap(),
        );

        // Rebuild the intermediate-stage constraint from its transcript reads.
        let intermediate_constraint = read_constraint(
            &mut verifier_challenger,
            &intermediate_evals,
            stacked_num_variables - FOLDING,
            ROUND_EQ_POINTS,
            ROUND_SEL_POINTS,
        );
        intermediate_constraint.combine_evals(&mut sum);
        constraints.push(intermediate_constraint);
        // The grinding stage carries proof-of-work bits; the final stage none.
        verifier_challenge.extend(
            &proof1
                .verify_rounds(
                    &mut verifier_challenger,
                    &mut sum,
                    FOLDING,
                    POW_BITS,
                    Basis::Evaluation,
                )
                .unwrap(),
        );
        verifier_challenge.extend(
            &proof2
                .verify_rounds(
                    &mut verifier_challenger,
                    &mut sum,
                    stacked_num_variables - 2 * FOLDING,
                    0,
                    Basis::Evaluation,
                )
                .unwrap(),
        );

        // Both sides must have folded the identical challenge vector.
        assert_eq!(prover_randomness, verifier_challenge);
        // Final identity: running sum equals folded value times the constraint weights.
        let weights = strategy
            .variable_order
            .eval_constraints_poly(&constraints, &verifier_challenge);
        assert_eq!(sum, final_folded_value * weights);
    }

    #[test]
    fn prefix_roundtrip_mixed_eq_next_requests() {
        // Invariant: same mixed-batch roundtrip as the suffix case, but under
        // prefix-first variable binding, so no challenge reversal is needed.
        let witness = PrefixProver::<F, EF>::new_witness(build_tables(), FOLDING);
        let shapes = table_shapes();
        let stacked_num_variables = witness.num_variables();
        let stacked_poly = witness.poly().clone();
        let strategy = PrefixProver::<F, EF>::strategy();

        // Mixed schedule: each tuple is (table, current columns, next columns).
        //     table 0: current [0], next [1]
        //     table 1: current [],  next [0]
        //     table 0: current [1], next [0]
        let schedule = [
            (0, vec![0], vec![1]),
            (1, vec![], vec![0]),
            (0, vec![1], vec![0]),
        ];

        // Prover side: sample points and absorb evals through the transcript.
        let mut prover_challenger = challenger();
        let mut prover_state = PrefixProver::<F, EF>::from_witness(witness);
        let opening_claims = schedule
            .iter()
            .map(|(table_idx, current, next)| {
                // Pack current and next column requests into one batch.
                let batch = OpeningBatch::new(current.clone(), next.clone());
                let evals = prover_state.eval(*table_idx, &batch, &mut prover_challenger);
                (*table_idx, batch, evals)
            })
            .collect::<Vec<_>>();
        let virtual_eval = prover_state.add_virtual_eval(&mut prover_challenger);

        // First sumcheck stage folds the SVO rounds and writes their proof.
        let mut proof0 = SumcheckData::<F, EF>::default();
        let (mut prover, mut prover_randomness) =
            prover_state.into_sumcheck(&mut proof0, 0, &mut prover_challenger);
        assert_eq!(proof0.num_rounds(), FOLDING);
        assert_eq!(prover.num_variables(), stacked_num_variables - FOLDING);

        // Remaining stages drive the prover to a single folded value.
        let (proof1, proof2, intermediate_evals, final_folded_value) = drive_intermediate_and_final(
            &mut prover,
            &mut prover_challenger,
            &mut prover_randomness,
            stacked_num_variables,
        );

        // Prefix binding evaluates directly at the folded challenges, no reversal.
        let final_eval = stacked_poly.eval_base(&prover_randomness);
        assert_eq!(final_eval, final_folded_value);

        // Verifier side: fresh transcript, mirror every prover absorption.
        let mut verifier_challenger = challenger();
        let mut verifier = Verifier::<F, EF>::new(&shapes, strategy);
        for (table_idx, batch, evals) in opening_claims {
            verifier
                .add_claim(table_idx, &batch, &evals, &mut verifier_challenger)
                .unwrap();
        }
        verifier.add_virtual_eval(virtual_eval, &mut verifier_challenger);

        // Batching challenge and the initial constraint over all recorded claims.
        // The verifier draws it through the layout, exactly as the prover does.
        let alpha = verifier.batching_challenge(&mut verifier_challenger);
        let initial_constraint = verifier.constraint(alpha);
        let mut sum = EF::ZERO;
        initial_constraint.combine_evals(&mut sum);
        // The constraint's combined value must equal the alpha-batched claim sum.
        assert_eq!(sum, verifier.sum(alpha));

        // Collect each stage's constraint and replay the challenges it folded.
        let mut constraints = vec![initial_constraint];
        let mut verifier_challenge = Point::new(vec![]);
        verifier_challenge.extend(
            &proof0
                .verify_rounds(
                    &mut verifier_challenger,
                    &mut sum,
                    FOLDING,
                    0,
                    Basis::Evaluation,
                )
                .unwrap(),
        );

        // Rebuild the intermediate-stage constraint from its transcript reads.
        let intermediate_constraint = read_constraint(
            &mut verifier_challenger,
            &intermediate_evals,
            stacked_num_variables - FOLDING,
            ROUND_EQ_POINTS,
            ROUND_SEL_POINTS,
        );
        intermediate_constraint.combine_evals(&mut sum);
        constraints.push(intermediate_constraint);
        // The grinding stage carries proof-of-work bits; the final stage none.
        verifier_challenge.extend(
            &proof1
                .verify_rounds(
                    &mut verifier_challenger,
                    &mut sum,
                    FOLDING,
                    POW_BITS,
                    Basis::Evaluation,
                )
                .unwrap(),
        );
        verifier_challenge.extend(
            &proof2
                .verify_rounds(
                    &mut verifier_challenger,
                    &mut sum,
                    stacked_num_variables - 2 * FOLDING,
                    0,
                    Basis::Evaluation,
                )
                .unwrap(),
        );

        // Both sides must have folded the identical challenge vector.
        assert_eq!(prover_randomness, verifier_challenge);
        // Final identity: running sum equals folded value times the constraint weights.
        let weights = strategy
            .variable_order
            .eval_constraints_poly(&constraints, &verifier_challenge);
        assert_eq!(sum, final_folded_value * weights);
    }

    #[test]
    fn prefix_next_is_slot_local_not_full_stacked_next() {
        // Invariant: the repeat-last successor view must be taken per-column
        // inside its own slot, never across the interleaved stacked layout.
        //
        // Why: stacking interleaves columns, so the index "one past" in the
        // full layout lands on a neighbouring column, not the next row of one
        // column. The two successor evaluations must therefore differ.

        // Two 2-variable columns with distinct values.
        let col0 = Poly::new(vec![
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
        ]);
        let col1 = Poly::new(vec![
            F::from_u64(13),
            F::from_u64(17),
            F::from_u64(19),
            F::from_u64(23),
        ]);

        // Interleave the two columns into one 3-variable stacked polynomial.
        //     stacked[2*i]   = col0[i]   (even slots)
        //     stacked[2*i+1] = col1[i]   (odd slots)
        let mut stacked = F::zero_vec(8);
        for local_idx in 0..4 {
            stacked[local_idx << 1] = col0.as_slice()[local_idx];
            stacked[(local_idx << 1) | 1] = col1.as_slice()[local_idx];
        }
        let stacked = Poly::new(stacked);

        // Slot-local point over the two column variables.
        let local_point = Point::new(vec![EF::from_u64(29), EF::from_u64(31)]);
        // Selector coordinate 0 lifts that point into col0's even slots.
        let selector_point = Point::new(vec![EF::ZERO]);
        let mut full_point = local_point.clone();
        full_point.extend(&selector_point);

        // Successor inside col0 alone vs successor across the stacked layout.
        let slot_local_next = col0.eval_next_base(&local_point);
        let full_stacked_next = stacked.eval_next_base(&full_point);

        // They must disagree: the stacked successor crosses into col1.
        assert_ne!(slot_local_next, full_stacked_next);
    }

    fn run_shape_test<L>(shape: &[(usize, usize)], schedule: &[(usize, Vec<usize>)])
    where
        L: Layout<F, EF>,
    {
        let witness = L::new_witness(tables_from_shape(shape), FOLDING);
        let shapes = table_shapes_from(shape);
        let borrowed: Vec<(usize, &[usize])> = schedule
            .iter()
            .map(|(t, polys)| (*t, polys.as_slice()))
            .collect();
        run_roundtrip_test::<L>(witness, &shapes, &borrowed);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 16, ..ProptestConfig::default() })]

        // Invariant:
        //     Every valid opening schedule over the fixed two-table witness
        //     roundtrips through the protocol without prover/verifier divergence.
        //
        // Coverage: includes non-ascending column orders that previously exposed
        // alpha / partial-eval alignment bugs.
        #[test]
        fn roundtrip_proptest(schedule in arb_opening_schedule()) {
            let borrowed: Vec<(usize, &[usize])> = schedule
                .iter()
                .map(|(t, polys)| (*t, polys.as_slice()))
                .collect();

            run_roundtrip_test::<PrefixProver<F, EF>>(
                PrefixProver::<F, EF>::new_witness(build_tables(), FOLDING),
                &table_shapes(),
                &borrowed,
            );
            run_roundtrip_test::<SuffixProver<F, EF>>(
                SuffixProver::<F, EF>::new_witness(build_tables(), FOLDING),
                &table_shapes(),
                &borrowed,
            );
        }
    }

    /// One recorded batch.
    ///
    /// - Index of the table it opens.
    /// - Columns it requested.
    /// - Evaluations it claimed.
    type RecordedClaim = (usize, OpeningBatch<usize>, OpeningEvals<EF>);

    /// Opening schedule that exercises both column groups on both fixture tables.
    ///
    /// Returned as owned requests so a mutation test can perturb the evaluations.
    fn mixed_schedule() -> Vec<(usize, OpeningBatch<usize>)> {
        // Table 0 has arity 9 and two columns.
        //
        // Table 1 has arity 10 and two columns.
        //
        //     table 0: direct [0, 1], successor [1]
        //     table 1: direct [0],    successor [0, 1]
        vec![
            (0, OpeningBatch::new(vec![0, 1], vec![1])),
            (1, OpeningBatch::new(vec![0], vec![0, 1])),
        ]
    }

    /// Records the mixed schedule plus one out-of-domain claim on a fresh prover.
    ///
    /// # Returns
    ///
    /// - The recorded claims.
    /// - The out-of-domain evaluation.
    /// - The batching challenge the prover reached after them.
    fn record_mixed_claims<L>() -> (Vec<RecordedClaim>, EF, EF)
    where
        L: Layout<F, EF>,
    {
        // Fresh prover over the fixed two-table fixture.
        let mut prover = L::from_witness(L::new_witness(build_tables(), FOLDING));
        let mut ch = challenger();

        // Each call draws its own point and binds its own evaluations.
        let claims: Vec<_> = mixed_schedule()
            .into_iter()
            .map(|(table_idx, batch)| {
                let evals = prover.eval(table_idx, &batch, &mut ch);
                (table_idx, batch, evals)
            })
            .collect();

        // One out-of-domain claim on the whole stacked polynomial.
        //
        // It follows the concrete ones.
        let virtual_eval = prover.add_virtual_eval(&mut ch);

        // The challenge that would collapse every claim recorded above.
        let alpha = prover.batching_challenge(&mut ch);

        (claims, virtual_eval, alpha)
    }

    /// Replays a recorded claim list on a fresh verifier.
    ///
    /// Returns the batching challenge that replay reaches.
    ///
    /// Every draw comes from the verifier's own transcript.
    ///
    /// A value that fails to bind shows up as a challenge the prover never saw.
    fn replay_mixed_claims<L>(claims: &[RecordedClaim], virtual_eval: EF) -> EF
    where
        L: Layout<F, EF>,
    {
        // Fresh verifier over the same table shapes and the same layout strategy.
        let mut ch = challenger();
        let mut verifier = Verifier::<F, EF>::new(&table_shapes(), L::strategy());

        // Mirror every concrete batch, in the order the prover recorded them.
        for (table_idx, batch, evals) in claims {
            verifier
                .add_claim(*table_idx, batch, evals, &mut ch)
                .unwrap();
        }

        // Mirror the out-of-domain claim that followed them.
        verifier.add_virtual_eval(virtual_eval, &mut ch);

        verifier.batching_challenge(&mut ch)
    }

    /// Replays a claim list whose direct evaluation at the given position is perturbed.
    fn replay_with_perturbed_direct<L>(
        claims: &[RecordedClaim],
        virtual_eval: EF,
        claim: usize,
        position: usize,
    ) -> EF
    where
        L: Layout<F, EF>,
    {
        let mut tampered = claims.to_vec();
        // Bump one direct evaluation by one.
        //
        // Every other value is left alone.
        let mut current = tampered[claim].2.current().to_vec();
        current[position] += EF::ONE;
        tampered[claim].2 = OpeningBatch::new(current, tampered[claim].2.next().to_vec());
        replay_mixed_claims::<L>(&tampered, virtual_eval)
    }

    /// Replays a claim list with one perturbed successor evaluation.
    ///
    /// The position selects which one.
    fn replay_with_perturbed_successor<L>(
        claims: &[RecordedClaim],
        virtual_eval: EF,
        claim: usize,
        position: usize,
    ) -> EF
    where
        L: Layout<F, EF>,
    {
        let mut tampered = claims.to_vec();
        // Bump one successor evaluation by one.
        //
        // Every other value is left alone.
        let mut next = tampered[claim].2.next().to_vec();
        next[position] += EF::ONE;
        tampered[claim].2 = OpeningBatch::new(tampered[claim].2.current().to_vec(), next);
        replay_mixed_claims::<L>(&tampered, virtual_eval)
    }

    #[test]
    fn both_sides_of_the_claim_phase_reach_one_batching_challenge() {
        // Invariant:
        //     Prover and verifier walk the same descriptions.
        //     Both therefore reach one batching challenge.
        //
        // Fixture state:
        //     table 0 (arity 9, 2 cols):  direct [0, 1], successor [1]
        //     table 1 (arity 10, 2 cols): direct [0],    successor [0, 1]
        //     plus one out-of-domain claim on the stacked polynomial.
        fn run<L>()
        where
            L: Layout<F, EF>,
        {
            let (claims, virtual_eval, prover_alpha) = record_mixed_claims::<L>();
            let verifier_alpha = replay_mixed_claims::<L>(&claims, virtual_eval);
            assert_eq!(prover_alpha, verifier_alpha);
        }

        // Both binding orders record the same claims through the same descriptions.
        run::<PrefixProver<F, EF>>();
        run::<SuffixProver<F, EF>>();
    }

    #[test]
    fn a_perturbed_direct_evaluation_breaks_the_claim_phase() {
        // Invariant:
        //     Every direct evaluation is bound before the batching challenge.
        //     Changing one leaves the verifier folding a claim nobody proved.
        //
        // Fixture state:
        //     claim 0 on table 0 carries direct evaluations at positions 0 and 1.
        //
        // Mutation:
        //     claim 0 direct group:  [e_0, e_1]  ->  [e_0 + 1, e_1]
        //                                                  ^
        //     then:                  [e_0, e_1]  ->  [e_0, e_1 + 1]
        //                                                       ^
        //     -> the replayed batching challenge must move in both cases
        fn run<L>()
        where
            L: Layout<F, EF>,
        {
            let (claims, virtual_eval, alpha) = record_mixed_claims::<L>();

            // Position 0 of the first claim's direct group.
            assert_ne!(
                alpha,
                replay_with_perturbed_direct::<L>(&claims, virtual_eval, 0, 0)
            );

            // Position 1 of the same group.
            //
            // No single position therefore carries the binding.
            assert_ne!(
                alpha,
                replay_with_perturbed_direct::<L>(&claims, virtual_eval, 0, 1)
            );
        }

        run::<PrefixProver<F, EF>>();
        run::<SuffixProver<F, EF>>();
    }

    #[test]
    fn a_perturbed_successor_evaluation_breaks_the_claim_phase() {
        // Invariant:
        //     The successor group is bound under its own step.
        //     It is as tightly bound as the direct group.
        //
        // Fixture state:
        //     claim 1 on table 1 carries successor evaluations at positions 0 and 1.
        //
        // Mutation:
        //     claim 1 successor group:  [e_0, e_1]  ->  [e_0 + 1, e_1]
        //                                                    ^
        //     then:                    [e_0, e_1]  ->  [e_0, e_1 + 1]
        //                                                         ^
        //     -> the replayed batching challenge must move in both cases
        fn run<L>()
        where
            L: Layout<F, EF>,
        {
            let (claims, virtual_eval, alpha) = record_mixed_claims::<L>();

            assert_ne!(
                alpha,
                replay_with_perturbed_successor::<L>(&claims, virtual_eval, 1, 0)
            );
            assert_ne!(
                alpha,
                replay_with_perturbed_successor::<L>(&claims, virtual_eval, 1, 1)
            );
        }

        run::<PrefixProver<F, EF>>();
        run::<SuffixProver<F, EF>>();
    }

    #[test]
    fn a_perturbed_out_of_domain_evaluation_breaks_the_claim_phase() {
        // Invariant:
        //     The out-of-domain evaluation is bound too, even though it names no column.
        //
        // Fixture state:
        //     one out-of-domain claim recorded after the two concrete batches.
        //
        // Mutation:
        //     out-of-domain evaluation:  v  ->  v + 1
        //     -> the replayed batching challenge must move
        fn run<L>()
        where
            L: Layout<F, EF>,
        {
            let (claims, virtual_eval, alpha) = record_mixed_claims::<L>();
            assert_ne!(
                alpha,
                replay_mixed_claims::<L>(&claims, virtual_eval + EF::ONE)
            );
        }

        run::<PrefixProver<F, EF>>();
        run::<SuffixProver<F, EF>>();
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 16, ..ProptestConfig::default() })]

        // Invariant:
        //     Agreement holds over random schedules.
        //     It is not special to the fixed mixed schedule above.
        //
        //     coverage: both binding orders, 1..=6 calls, any column order
        #[test]
        fn claim_phase_agrees_over_random_schedules(schedule in arb_opening_schedule()) {
            fn run<L>(schedule: &[(usize, Vec<usize>)])
            where
                L: Layout<F, EF>,
            {
                // Prover side: record every scheduled batch, then draw the challenge.
                let mut prover = L::from_witness(L::new_witness(build_tables(), FOLDING));
                let mut prover_ch = challenger();
                let claims: Vec<_> = schedule
                    .iter()
                    .map(|(table_idx, polys)| {
                        // This strategy generates direct openings only.
                        let batch = OpeningBatch::new(polys.clone(), Vec::new());
                        let evals = prover.eval(*table_idx, &batch, &mut prover_ch);
                        (*table_idx, batch, evals)
                    })
                    .collect();
                let prover_alpha = prover.batching_challenge(&mut prover_ch);

                // Verifier side: mirror the same batches from the same schedule.
                let mut verifier_ch = challenger();
                let mut verifier = Verifier::<F, EF>::new(&table_shapes(), L::strategy());
                for (table_idx, batch, evals) in &claims {
                    verifier
                        .add_claim(*table_idx, batch, evals, &mut verifier_ch)
                        .unwrap();
                }

                assert_eq!(prover_alpha, verifier.batching_challenge(&mut verifier_ch));
            }

            run::<PrefixProver<F, EF>>(&schedule);
            run::<SuffixProver<F, EF>>(&schedule);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 8, ..ProptestConfig::default() })]

        // Invariant:
        //     Roundtrip agreement holds for valid generated witness shapes, not
        //     only the fixed two-table fixture.
        #[test]
        fn roundtrip_shape_proptest((shape, schedule) in arb_witness_and_schedule()) {
            run_shape_test::<PrefixProver<F, EF>>(&shape, &schedule);
            run_shape_test::<SuffixProver<F, EF>>(&shape, &schedule);
        }
    }
}
