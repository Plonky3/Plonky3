//! Prefix-mode stacked-sumcheck prover.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField, dot_product};
use p3_matrix::dense::DenseMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;

use crate::pcs::committer::writer::commit_base;
use crate::sumcheck::lagrange::lagrange_weights_01inf_multi;
use crate::sumcheck::layout::opening::Opening;
use crate::sumcheck::layout::prover::Layout;
use crate::sumcheck::layout::witness::{Table, TablePlacement};
use crate::sumcheck::layout::{LayoutStrategy, ProverMultiClaim, ProverVirtualClaim, Witness};
use crate::sumcheck::product_polynomial::ProductPolynomial;
use crate::sumcheck::strategy::{SumcheckProver, VariableOrder};
use crate::sumcheck::svo::{SvoPoint, calculate_accumulators_batch};
use crate::sumcheck::{Claim, SumcheckData, extrapolate_01inf};

/// Stacked-sumcheck prover with prefix-first variable binding.
///
/// # Flow
///
/// - Round one runs in SIMD-packed form.
/// - Every later round runs on the residual product polynomial.
#[derive(Debug, Clone)]
pub struct PrefixProver<F: Field, EF: ExtensionField<F>> {
    /// Source tables behind the stacked polynomial.
    pub(crate) tables: Vec<Table<F>>,
    /// Per-table placement metadata inside the stacked polynomial.
    pub(crate) placements: Vec<TablePlacement>,
    /// Number of variables of the stacked polynomial.
    pub(crate) num_variables: usize,
    /// Number of preprocessing rounds consumed before residual sumcheck.
    pub(crate) folding: usize,
    /// Stacked committed polynomial.
    pub(crate) poly: Poly<F>,
    /// Concrete claims recorded per source table.
    ///
    /// # Invariants
    ///
    /// - Every opening stored here is tied to a concrete source column.
    /// - Virtual openings never enter this map.
    /// - Claims are appended in insertion order.
    pub(crate) claim_map: Vec<Vec<ProverMultiClaim<F, EF>>>,
    /// Virtual claims sampled directly on the stacked polynomial.
    pub(crate) virtual_claims: Vec<ProverVirtualClaim<EF>>,
}

impl<F: TwoAdicField, EF: ExtensionField<F>> Layout<F, EF> for PrefixProver<F, EF> {
    fn from_witness(witness: Witness<F>) -> Self {
        // Move the witness fields out so the prover owns them outright.
        let parts = witness.into_parts();
        // One claim list per source table; virtual claims live in their own bucket.
        let num_tables = parts.tables.len();
        Self {
            tables: parts.tables,
            placements: parts.placements,
            num_variables: parts.num_variables,
            folding: parts.folding,
            poly: parts.poly,
            claim_map: (0..num_tables).map(|_| Vec::new()).collect(),
            virtual_claims: Vec::new(),
        }
    }

    fn new_witness(tables: Vec<Table<F>>, folding: usize) -> Witness<F> {
        Witness::new_interleaved(tables, folding)
    }

    fn commit<Dft, MT, Challenger>(
        dft: &Dft,
        mmcs: &MT,
        challenger: &mut Challenger,
        witness: Witness<F>,
        folding: usize,
        starting_log_inv_rate: usize,
    ) -> (Self, MT::Commitment, MT::ProverData<DenseMatrix<F>>)
    where
        Dft: TwoAdicSubgroupDft<F>,
        MT: Mmcs<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        let (root, prover_data) = commit_base(
            Self::variable_order(),
            dft,
            mmcs,
            challenger,
            &witness.poly,
            folding,
            starting_log_inv_rate,
        );

        (Self::from_witness(witness), root, prover_data)
    }

    fn folding(&self) -> usize {
        self.folding
    }

    /// Returns the number of variables of the stacked polynomial.
    fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns the number of variables of table `id`.
    fn num_variables_table(&self, id: usize) -> usize {
        self.tables[id].num_variables()
    }

    /// Records opening claims for the selected columns of `table_idx`.
    ///
    /// # Arguments
    ///
    /// - `table_idx`  — source table index.
    /// - `polys`      — columns to open; must be non-empty.
    /// - `challenger` — Fiat–Shamir transcript.
    ///
    /// # Fiat–Shamir
    ///
    /// - Samples the opening point internally from the challenger.
    /// - Absorbs the evaluations into the transcript before returning.
    /// - The verifier's `add_claim` performs the symmetric absorption.
    ///
    /// # Panics
    ///
    /// - Columns list must be non-empty.
    #[tracing::instrument(skip_all)]
    fn eval<Ch>(&mut self, table_idx: usize, polys: &[usize], challenger: &mut Ch) -> Vec<EF>
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Precondition: opening nothing would silently push an empty ProverMultiClaim.
        assert!(
            !polys.is_empty(),
            "opening schedule must name at least one column"
        );

        // Sample the local-frame opening point from the transcript.
        let table = &self.tables[table_idx];
        let point = Point::expand_from_univariate(
            challenger.sample_algebra_element(),
            table.num_variables(),
        );

        // Factorise the point once; every selected column reuses it.
        let point = SvoPoint::new_packed(self.folding, &point);

        // Evaluate each column at the SVO point; split into (opening, eval).
        let (openings, evals): (Vec<_>, Vec<EF>) = polys
            .iter()
            .map(|&poly_idx| {
                let (eval, partial_evals) = point.eval(table.poly(poly_idx));
                let opening = Opening {
                    poly_idx: Some(poly_idx),
                    eval,
                    data: partial_evals,
                };
                (opening, eval)
            })
            .unzip();

        // Bind the evaluations into the transcript; the verifier absorbs the same bytes.
        challenger.observe_algebra_slice(&evals);

        // Store the batch for the later sumcheck reduction.
        self.claim_map[table_idx].push(ProverMultiClaim::new(point, openings));

        evals
    }

    /// Samples a virtual evaluation on the full stacked polynomial.
    ///
    /// # Why
    ///
    /// The WHIR protocol occasionally pins the stacked polynomial at a fresh
    /// random point for soundness amplification. Prefix mode evaluates the
    /// stacked polynomial directly — no per-column weighting needed.
    #[tracing::instrument(skip_all)]
    fn add_virtual_eval<Ch>(&mut self, challenger: &mut Ch) -> EF
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Sample a challenge point covering every stacked variable.
        let point =
            Point::expand_from_univariate(challenger.sample_algebra_element(), self.num_variables);

        let mut eval = EF::ZERO;
        let mut openings = Vec::new();
        let mut weights = Vec::new();

        for placement in &self.placements {
            let table = &self.tables[placement.idx()];
            for (poly_idx, selector) in placement.selectors().iter().enumerate() {
                let poly = table.poly(poly_idx);

                let (local_part, selector_part) = point.split_at(table.num_variables());

                let weight =
                    Point::eval_eq::<EF>(selector.point().as_slice(), selector_part.as_slice());

                let local_svo = SvoPoint::new_packed(self.folding, &local_part);
                let (column_eval, partial_evals) = local_svo.eval(poly);

                eval += weight * column_eval;
                openings.push(Opening {
                    poly_idx: None,
                    eval: column_eval,
                    data: partial_evals,
                });
                weights.push(weight);
            }
        }

        let accumulators = calculate_accumulators_batch(
            &ProverMultiClaim::new(
                SvoPoint::new_unpacked(self.folding, &point, VariableOrder::Prefix),
                openings,
            ),
            &weights,
        );

        // Commit the evaluation to the transcript.
        challenger.observe_algebra_element(eval);
        self.virtual_claims.push(Claim {
            point,
            eval,
            data: accumulators,
        });

        eval
    }

    /// Finalises preprocessing and returns the residual sumcheck prover.
    ///
    /// # Returns
    ///
    /// - Residual sumcheck prover over the packed product polynomial.
    /// - Folding challenges sampled during preprocessing.
    ///
    /// # Algorithm
    ///
    /// ```text
    ///     Phase | Action
    ///     ------+-----------------------------------------------
    ///       1   | Sample the batching challenge  a.
    ///       2   | running sum  = sum_{i}  a^i * eval_i.
    ///       3   | weight poly  = sum_{i}  a^i * eq(z_i, X).
    ///       4   | Fold round 1 in SIMD-packed arithmetic.
    ///       5   | Drive rounds 2..folding on the product polynomial.
    /// ```
    ///
    /// # Precondition
    ///
    /// - Each table's arity is at least  log_2(W), with W the packing width.
    /// - Guarantees every per-slot packed accumulation spans a whole packed element.
    #[tracing::instrument(skip_all)]
    fn into_sumcheck<Ch>(
        self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> (SumcheckProver<F, EF>, Point<EF>)
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Sanity: preprocessing cannot consume more rounds than the stacked arity.
        assert!(self.folding <= self.num_variables);

        let alpha: EF = challenger.sample_algebra_element();
        let n_claims = self.num_claims();

        let mut alphas = alpha.powers();
        let accumulators: Vec<_> = self
            .placements
            .iter()
            .flat_map(|placement| self.claim_map[placement.idx()].iter())
            .map(|claim| {
                let per_claim: Vec<EF> = alphas.by_ref().take(claim.len()).collect();
                calculate_accumulators_batch(claim, &per_claim)
            })
            .collect();

        let mut sum = self.sum(alpha);
        let mut rs = Vec::new();

        for round_idx in 0..self.folding {
            let weights = lagrange_weights_01inf_multi(&rs);

            let mut c0 = EF::ZERO;
            let mut c_inf = EF::ZERO;

            for accs in &accumulators {
                c0 += dot_product::<EF, _, _>(
                    accs[round_idx][0].iter().copied(),
                    weights.iter().copied(),
                );
                c_inf += dot_product::<EF, _, _>(
                    accs[round_idx][1].iter().copied(),
                    weights.iter().copied(),
                );
            }

            for (vc, alpha_i) in self
                .virtual_claims
                .iter()
                .zip(alpha.powers().skip(n_claims))
            {
                let vc_accs = &vc.data;
                c0 += alpha_i
                    * dot_product::<EF, _, _>(
                        vc_accs[round_idx][0].iter().copied(),
                        weights.iter().copied(),
                    );
                c_inf += alpha_i
                    * dot_product::<EF, _, _>(
                        vc_accs[round_idx][1].iter().copied(),
                        weights.iter().copied(),
                    );
            }

            let r = sumcheck_data.observe_and_sample(challenger, c0, c_inf, pow_bits);
            sum = extrapolate_01inf(c0, sum - c0, c_inf, r);
            rs.push(r);
        }

        let rs = Point::new(rs);
        let compressed = tracing::info_span!("compress_prefix_to_packed")
            .in_scope(|| self.poly.compress_prefix_to_packed(&rs, EF::ONE));

        let weights = self.combine_eqs(&rs, alpha).pack::<F, EF>();
        let prod_poly =
            ProductPolynomial::<F, EF>::new_packed(VariableOrder::Prefix, compressed, weights);
        debug_assert_eq!(prod_poly.dot_product(), sum);

        (SumcheckProver::new(prod_poly, sum), rs)
    }

    /// Returns the total number of concrete openings recorded so far.
    fn num_claims(&self) -> usize {
        self.claim_map
            .iter()
            .flat_map(|claims| claims.iter().map(ProverMultiClaim::len))
            .sum()
    }

    fn strategy() -> LayoutStrategy {
        LayoutStrategy::new(true, VariableOrder::Prefix)
    }
}

impl<F: TwoAdicField, EF: ExtensionField<F>> PrefixProver<F, EF> {
    /// Computes the batched claimed sum from concrete and virtual openings.
    ///
    /// # Identity
    ///
    /// ```text
    ///     sum = sum_{i}  alpha^i * eval_i
    /// ```
    ///
    /// # Alpha ordering
    ///
    /// Powers of `alpha` are handed out in insertion order:
    ///
    /// - Outer: placements, in the order the witness laid them out.
    /// - Middle: claims recorded against that placement's source table.
    /// - Inner: openings inside each claim, in the order they were recorded.
    ///
    /// # Virtual claims
    ///
    /// - Virtual evaluations continue the same alpha sequence.
    /// - They start at `alpha^n`, with `n` the total concrete opening count.
    ///
    /// # Verifier agreement
    ///
    /// The verifier walks its claim registry with the same three-loop order,
    /// so both sides assign the same `alpha^i` to the same claim point.
    fn sum(&self, alpha: EF) -> EF {
        let mut sum = EF::ZERO;
        let mut alphas = alpha.powers();

        // Concrete openings: three loops, no filter.
        for placement in &self.placements {
            for claim in &self.claim_map[placement.idx()] {
                for opening in claim.openings() {
                    sum += opening.eval() * alphas.next().unwrap();
                }
            }
        }

        // Virtual claims continue the alpha sequence right after the concrete ones.
        sum += dot_product::<EF, _, _>(
            self.virtual_claims.iter().map(Claim::eval),
            alpha.powers().skip(self.num_claims()),
        );

        sum
    }

    /// Builds the residual equality-weight polynomial after the prefix SVO rounds.
    #[tracing::instrument(skip_all)]
    fn combine_eqs(&self, rs: &Point<EF>, alpha: EF) -> Poly<EF> {
        assert_eq!(rs.num_variables(), self.folding);
        let mut out = Poly::<EF>::zero(self.num_variables - rs.num_variables());

        let mut alphas = alpha.powers();

        for placement in &self.placements {
            let local_rest_variables =
                self.num_variables_table(placement.idx()) - rs.num_variables();
            for claim in &self.claim_map[placement.idx()] {
                for opening in claim.openings() {
                    let col = opening.poly_idx().unwrap();
                    let selector = &placement.selectors()[col];
                    let mut local = Poly::<EF>::zero(local_rest_variables);
                    claim
                        .point()
                        .accumulate_into(local.as_mut_slice(), rs, alphas.next().unwrap());

                    for (local_idx, &value) in local.as_slice().iter().enumerate() {
                        let dst = (local_idx << selector.num_variables()) | selector.index();
                        out.as_mut_slice()[dst] += value;
                    }
                }
            }
        }

        let mut alpha_i = alpha.exp_u64(self.num_claims() as u64);
        for claim in &self.virtual_claims {
            let (svo, rest) = claim.point.split_at(rs.num_variables());
            let scale = alpha_i * Point::eval_eq(svo.as_slice(), rs.as_slice());
            SplitEq::new_unpacked(&rest, scale).accumulate_into(out.as_mut_slice(), None);
            alpha_i *= alpha;
        }

        out
    }
}
