//! Direct batched compiler for the WARP root IOP.
//!
//! The parent root prover first records the WARP VACC/DACC transcript as
//! typed opening claims on committed RS oracles. This module is the compact
//! WHIR-facing reduction for those claims when every oracle is represented by
//! the same WARP/WHIR Reed-Solomon code `C`.
//!
//! The algebra follows the same shape as WHIR's linear-Sigma IOP. For an
//! oracle message `w` and codeword `u = C(w)`, each recorded WARP query is
//! converted into a linear statement over the message:
//!
//! ```text
//!     <a_q, w> = y.
//! ```
//!
//! For an RS codeword index claim `u[j] = y`, `a_q` is the message-domain
//! linear functional that evaluates the encoded word at `j`. For a multilinear
//! codeword claim `hat u(z) = y`, `a_q` is the message-domain pullback of the
//! equality polynomial `eq_z`. This is exactly the bridge we need between the
//! WARP paper's RS accumulator claims and the WHIR paper's constrained-RS
//! proximity/opening proof: WARP supplies the linear claims, WHIR authenticates
//! one final batched opening against the committed RS oracles.
//!
//! Fiat-Shamir order in this file is deliberately explicit:
//! 1. commitments are observed by the caller before this reduction starts,
//! 2. every statement is absorbed before its per-oracle batching challenge
//!    `gamma` is sampled,
//! 3. the cross-oracle batching challenge `rho` is sampled only after all
//!    per-oracle targets are bound,
//! 4. the final virtual opening value is absorbed before WHIR opens it.
//!
//! This module does not decide WARP step validity by itself. Soundness comes
//! from the surrounding root IOP recording every VACC/DACC and finalizer claim,
//! then using this compiler to prove those recorded linear claims against the
//! same committed RS codewords.

use super::*;

/// Residual oracle material for the older per-oracle fallback path.
///
/// It stores an owned multilinear table after the generic linear-Sigma
/// reduction has already reduced that oracle to one residual opening.
pub(super) enum NativeWarpBatchedResidualPoly<F, EF> {
    Base(Poly<F>),
    Extension(Poly<EF>),
}

/// Borrowed message-domain oracle used by the compact root reduction.
///
/// The important invariant is that these slices are message evaluations
/// `w`, not freshly encoded codewords. The WARP root compiler has already
/// converted codeword claims through `C` into message-domain weights, so the
/// sumcheck below runs over the message hypercube and avoids a second RS code.
pub(super) enum NativeWarpDirectBatchedResidualPoly<'a, F, EF> {
    Base(&'a [F]),
    Extension(&'a [EF]),
}

const COMPACT_ROOT_EVAL_PAR_THRESHOLD: usize = 1 << 14;

enum NativeWarpCompactEvalPoly<'a, F, EF> {
    Base(&'a [F]),
    ExtensionBorrowed(&'a [EF]),
    ExtensionOwned(Poly<EF>),
}

impl<'a, F, EF> NativeWarpCompactEvalPoly<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F> + Send + Sync,
{
    /// Build a mutable sumcheck view over the borrowed message oracle.
    ///
    /// The view only allocates after the first folding challenge is sampled.
    /// That keeps the prover on borrowed slices until the WHIR/Spartan-style
    /// multilinear folding `f'(x) = f(0, x) + r * (f(1, x) - f(0, x))` has to
    /// materialize the next round.
    fn from_direct(poly: &'a NativeWarpDirectBatchedResidualPoly<'a, F, EF>) -> Self {
        match poly {
            NativeWarpDirectBatchedResidualPoly::Base(values) => Self::Base(values),
            NativeWarpDirectBatchedResidualPoly::Extension(values) => {
                Self::ExtensionBorrowed(values)
            }
        }
    }

    fn sumcheck_coefficients(&self, weights: &[EF]) -> (EF, EF) {
        match self {
            Self::Base(values) => VariableOrder::Prefix.sumcheck_coefficients(*values, weights),
            Self::ExtensionBorrowed(values) => {
                VariableOrder::Prefix.sumcheck_coefficients(*values, weights)
            }
            Self::ExtensionOwned(poly) => {
                VariableOrder::Prefix.sumcheck_coefficients(poly.as_slice(), weights)
            }
        }
    }

    /// Fix the next prefix variable to the verifier challenge `r`.
    ///
    /// This is the standard multilinear sumcheck folding. For base-field
    /// messages we lift into the extension field at the first fold because
    /// `r` is sampled in the extension field.
    fn fix_prefix_var_mut(&mut self, r: EF) {
        match self {
            Self::Base(values) => {
                *self = Self::ExtensionOwned(fix_base_prefix_to_extension::<F, EF>(values, r));
            }
            Self::ExtensionBorrowed(values) => {
                *self = Self::ExtensionOwned(fix_extension_prefix_to_owned::<EF>(values, r));
            }
            Self::ExtensionOwned(poly) => poly.fix_prefix_var_mut(r),
        }
    }

    /// Return the single terminal value left after all sumcheck folds.
    fn final_value(&self) -> EF {
        match self {
            Self::Base(values) => EF::from(values[0]),
            Self::ExtensionBorrowed(values) => values[0],
            Self::ExtensionOwned(poly) => poly.as_slice()[0],
        }
    }
}

/// Fold a base-field table by one extension-field sumcheck challenge.
fn fix_base_prefix_to_extension<F, EF>(values: &[F], r: EF) -> Poly<EF>
where
    F: Field,
    EF: ExtensionField<F> + Send + Sync,
{
    let mid = values.len() / 2;
    let (lo, hi) = values.split_at(mid);
    let folded = if values.len() >= COMPACT_ROOT_EVAL_PAR_THRESHOLD {
        lo.par_iter()
            .zip(hi.par_iter())
            .map(|(&a0, &a1)| EF::from(a0) + r * EF::from(a1 - a0))
            .collect()
    } else {
        lo.iter()
            .zip(hi.iter())
            .map(|(&a0, &a1)| EF::from(a0) + r * EF::from(a1 - a0))
            .collect()
    };
    Poly::new(folded)
}

/// Fold an extension-field table by one sumcheck challenge.
fn fix_extension_prefix_to_owned<EF>(values: &[EF], r: EF) -> Poly<EF>
where
    EF: Field,
{
    let mid = values.len() / 2;
    let (lo, hi) = values.split_at(mid);
    let folded = if values.len() >= COMPACT_ROOT_EVAL_PAR_THRESHOLD {
        lo.par_iter()
            .zip(hi.par_iter())
            .map(|(&a0, &a1)| a0 + r * (a1 - a0))
            .collect()
    } else {
        lo.iter()
            .zip(hi.iter())
            .map(|(&a0, &a1)| a0 + r * (a1 - a0))
            .collect()
    };
    Poly::new(folded)
}

/// Commitment variant for one batched residual oracle.
///
/// `SharedBase` is the optimized WHIR path for many base-message columns under
/// one Merkle root: the verifier opens one row and receives the selected
/// column instead of authenticating independent roots for every WARP oracle.
pub(super) enum NativeWarpBatchedResidualCommitment<Comm> {
    Base(Comm),
    Extension(Comm),
    SharedBase {
        root: Comm,
        column: usize,
        width: usize,
    },
}

/// Prover-side deferred WHIR oracle corresponding to
/// [`NativeWarpBatchedResidualCommitment`].
pub(super) enum NativeWarpBatchedResidualProverOracle<F, EF, MT, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    Base(WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>),
    Extension(WhirExtensionDeferredProverData<F, EF, MT, DIGEST_ELEMS>),
    SharedBase {
        shared: Arc<WhirSharedBaseDeferredProverData<F, EF, MT, DIGEST_ELEMS>>,
        column: usize,
        width: usize,
    },
}

/// Linear claims for one committed WARP root oracle, before cross-oracle
/// batching.
///
/// A statement contains constraints of the form `<a_q, w> = y` for one message
/// oracle `w`. Multiple constraints on the same oracle are folded by a
/// per-oracle challenge `gamma` into
///
/// ```text
///     <sum_i gamma^i a_i, w> = sum_i gamma^i y_i.
/// ```
///
/// The prover and verifier both derive `gamma` after absorbing the exact query
/// shape and target values, so a malicious prover cannot choose the batching
/// weights before committing to the WARP claims being batched.
#[derive(Clone, Debug)]
pub(super) struct NativeWarpCompactRootStatement<EF> {
    num_variables: usize,
    constraints: Vec<NativeWarpCompactRootConstraint<EF>>,
}

#[derive(Clone, Debug)]
struct NativeWarpCompactRootConstraint<EF> {
    query: NativeWarpCompactRootQuery<EF>,
    target: EF,
}

#[derive(Clone, Debug)]
enum NativeWarpCompactRootQuery<EF> {
    Index(usize),
    Mle(Vec<EF>),
}

impl<EF> NativeWarpCompactRootStatement<EF>
where
    EF: Field,
{
    pub(super) fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            constraints: Vec::new(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    pub(super) fn add_index(&mut self, index: usize, target: EF) {
        self.constraints.push(NativeWarpCompactRootConstraint {
            query: NativeWarpCompactRootQuery::Index(index),
            target,
        });
    }

    pub(super) fn add_mle(&mut self, point: Vec<EF>, target: EF) {
        self.constraints.push(NativeWarpCompactRootConstraint {
            query: NativeWarpCompactRootQuery::Mle(point),
            target,
        });
    }

    fn observe_and_sample_gamma<F, Challenger>(&self, challenger: &mut Challenger) -> EF
    where
        F: Field + PrimeCharacteristicRing,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + CanObserve<F>,
    {
        // Bind the statement shape and targets before sampling `gamma`.
        // This matches the linear-Sigma batching discipline: the random linear
        // combination is verifier-derived after the individual claims are
        // fixed in the Fiat-Shamir transcript.
        challenger.observe(F::from_usize(self.num_variables));
        challenger.observe(F::from_usize(self.constraints.len()));
        for constraint in &self.constraints {
            match &constraint.query {
                NativeWarpCompactRootQuery::Index(index) => {
                    challenger.observe(F::ZERO);
                    challenger.observe(F::from_usize(*index));
                }
                NativeWarpCompactRootQuery::Mle(point) => {
                    challenger.observe(F::ONE);
                    challenger.observe(F::from_usize(point.len()));
                    for &coord in point {
                        challenger.observe_algebra_element(coord);
                    }
                }
            }
            challenger.observe_algebra_element(constraint.target);
        }
        challenger.sample_algebra_element()
    }

    fn batched_target(&self, gamma: EF) -> EF {
        let mut scale = EF::ONE;
        let mut target = EF::ZERO;
        for constraint in &self.constraints {
            target += scale * constraint.target;
            scale *= gamma;
        }
        target
    }

    /// Evaluate the batched query weight at the final sumcheck point by using
    /// an already-encoded equality vector.
    ///
    /// If the query is over codeword coordinates, the linear functional is the
    /// pullback of the codeword equality polynomial through the RS encoder:
    ///
    /// ```text
    ///     a_q = C^T eq_q,  so  <a_q, w> = <eq_q, C(w)>.
    /// ```
    ///
    /// The caller supplies `encoded_message_eq = C(eq_r)`, where `r` is the
    /// final message-domain sumcheck point. This lets the verifier compute the
    /// coefficient for the final WHIR opening without reconstructing every
    /// dense query weight independently.
    fn batched_weight_eval_from_encoded_eq<F>(
        &self,
        encoded_message_eq: &[EF],
        encoded_message_eq_poly: &Poly<EF>,
        gamma: EF,
    ) -> EF
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        let mut scale = EF::ONE;
        let mut value = EF::ZERO;
        for constraint in &self.constraints {
            let local = match &constraint.query {
                NativeWarpCompactRootQuery::Index(index) => encoded_message_eq[*index],
                NativeWarpCompactRootQuery::Mle(point) => {
                    encoded_message_eq_poly.eval_ext::<F>(&Point::new(point.clone()))
                }
            };
            value += scale * local;
            scale *= gamma;
        }
        value
    }

    /// Fast verifier path for systematic RS queries that land on the message
    /// subspace.
    ///
    /// In systematic mode, a codeword index divisible by the blowup stride is
    /// literally a message-coordinate query. For those WARP claims we avoid
    /// encoding `eq_r` and use the multilinear equality value directly:
    ///
    /// ```text
    ///     eq_r(b) = prod_i (r_i b_i + (1-r_i)(1-b_i)).
    /// ```
    fn batched_weight_eval_from_message_eq_point<F, Dft>(
        &self,
        code: &ReedSolomonCode<F, Dft>,
        message_point: &[EF],
        gamma: EF,
    ) -> Option<EF>
    where
        F: TwoAdicField,
        Dft: TwoAdicSubgroupDft<F>,
    {
        if !code.is_systematic() {
            return None;
        }
        let stride = 1 << code.log_inv_rate();
        let mut scale = EF::ONE;
        let mut value = EF::ZERO;
        for constraint in &self.constraints {
            let local = match &constraint.query {
                NativeWarpCompactRootQuery::Index(index) => {
                    if !index.is_multiple_of(stride) {
                        return None;
                    }
                    eval_eq_at_hypercube_index(message_point, index / stride)
                }
                NativeWarpCompactRootQuery::Mle(point) => {
                    let (prefix, suffix) = point.split_at(code.log_msg_len());
                    if point.len() != code.log_codeword_len()
                        || suffix.iter().any(|&coord| coord != EF::ZERO)
                    {
                        return None;
                    }
                    eval_eq_point(message_point, prefix)
                }
            };
            value += scale * local;
            scale *= gamma;
        }
        Some(value)
    }
}

/// Convert all recorded WARP root claims into message-domain weight
/// polynomials.
///
/// This is the main "one RS code" bridge. WARP records claims about RS
/// codewords `u = C(w)`; WHIR opens the committed message polynomial `w`.
/// This function computes, for each statement, the message weight `a` such
/// that `<a, w>` equals the batched WARP target. Systematic message-subspace
/// claims use a sparse fast path; general codeword/MLE claims use the RS
/// adjoint batch helper from [`ReedSolomonCode`].
fn compact_batched_root_weights<F, EF, Dft>(
    code: &ReedSolomonCode<F, Dft>,
    statements: &[NativeWarpCompactRootStatement<EF>],
    gammas: &[EF],
) -> Result<Vec<Poly<EF>>, LinearSigmaReductionError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + Send + Sync,
    Dft: TwoAdicSubgroupDft<F>,
{
    if statements.len() != gammas.len() {
        return Err(LinearSigmaReductionError::ArityMismatch {
            expected: statements.len(),
            actual: gammas.len(),
        });
    }

    let stride = 1 << code.log_inv_rate();
    let mut weights = (0..statements.len()).map(|_| None).collect::<Vec<_>>();
    let mut dense_statement_indices = Vec::new();
    let mut dense_query_columns = Vec::new();

    for (statement_index, (statement, &gamma)) in statements.iter().zip(gammas).enumerate() {
        if statement.constraints.is_empty() {
            return Err(LinearSigmaReductionError::EmptyStatement);
        }
        if statement.num_variables != code.log_msg_len() {
            return Err(LinearSigmaReductionError::ArityMismatch {
                expected: statement.num_variables,
                actual: code.log_msg_len(),
            });
        }

        let all_systematic_indices = code.is_systematic()
            && statement.constraints.iter().all(|constraint| {
                matches!(
                    constraint.query,
                    NativeWarpCompactRootQuery::Index(index) if index.is_multiple_of(stride)
                )
            });
        if all_systematic_indices {
            let mut message_weights = EF::zero_vec(code.msg_len());
            let mut scale = EF::ONE;
            for constraint in &statement.constraints {
                let NativeWarpCompactRootQuery::Index(index) = constraint.query else {
                    unreachable!("all queries are systematic indices");
                };
                message_weights[index / stride] += scale;
                scale *= gamma;
            }
            weights[statement_index] = Some(Poly::new(message_weights));
            continue;
        }

        let mut codeword_query = EF::zero_vec(code.codeword_len());
        let mut scale = EF::ONE;
        for constraint in &statement.constraints {
            match &constraint.query {
                NativeWarpCompactRootQuery::Index(index) => {
                    codeword_query[*index] += scale;
                }
                NativeWarpCompactRootQuery::Mle(point) => {
                    if point.len() != code.log_codeword_len() {
                        return Err(LinearSigmaReductionError::ArityMismatch {
                            expected: code.log_codeword_len(),
                            actual: point.len(),
                        });
                    }
                    let eq = Poly::<EF>::new_from_point(point, scale);
                    for (slot, &value) in codeword_query.iter_mut().zip(eq.as_slice()) {
                        *slot += value;
                    }
                }
            }
            scale *= gamma;
        }
        dense_statement_indices.push(statement_index);
        dense_query_columns.push(codeword_query);
    }

    if !dense_query_columns.is_empty() {
        let width = dense_query_columns.len();
        let mut matrix_values = EF::zero_vec(code.codeword_len() * width);
        for (col, query) in dense_query_columns.iter().enumerate() {
            for (row, &value) in query.iter().enumerate() {
                matrix_values[row * width + col] = value;
            }
        }
        let message_weights =
            code.codeword_query_weights_batch(RowMajorMatrix::new(matrix_values, width));
        for (col, &statement_index) in dense_statement_indices.iter().enumerate() {
            let column = (0..code.msg_len())
                .map(|row| message_weights.values[row * width + col])
                .collect::<Vec<_>>();
            weights[statement_index] = Some(Poly::new(column));
        }
    }

    weights
        .into_iter()
        .map(|weight| weight.ok_or(LinearSigmaReductionError::EmptyStatement))
        .collect()
}

/// Prove all compact WARP root statements with one batched sumcheck and one
/// batched WHIR opening claim.
///
/// For oracle messages `w_k`, per-oracle batched weights `a_k`, and targets
/// `t_k`, the prover first checks the WARP-derived linear equations
///
/// ```text
///     <a_k, w_k> = t_k.
/// ```
///
/// Then it samples `rho` and proves the single multilinear sum
///
/// ```text
///     sum_k rho^k <a_k, w_k> = sum_k rho^k t_k.
/// ```
///
/// The sumcheck leaves a random message point `r` and coefficients
/// `c_k = rho^k a_k(r)`. WHIR then authenticates the residual opening
///
/// ```text
///     sum_k c_k w_k(r) = v.
/// ```
///
/// That final batched opening is the only PCS/WHIR opening needed by the root
/// proof when all statements are in this compact path.
pub(super) fn prove_compact_batched_root_reduction<F, EF, Dft, Challenger>(
    code: &ReedSolomonCode<F, Dft>,
    statements: &[NativeWarpCompactRootStatement<EF>],
    polys: &[NativeWarpDirectBatchedResidualPoly<'_, F, EF>],
    challenger: &mut Challenger,
    pow_bits: usize,
) -> Result<
    (
        BatchedLinearSigmaReductionProof<F, EF>,
        BatchedLinearSigmaOpeningClaim<EF>,
    ),
    LinearSigmaReductionError,
>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F> + TwoAdicField + Send + Sync,
    Dft: TwoAdicSubgroupDft<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if statements.is_empty() {
        return Err(LinearSigmaReductionError::EmptyStatement);
    }
    if statements.len() != polys.len() {
        return Err(LinearSigmaReductionError::ArityMismatch {
            expected: statements.len(),
            actual: polys.len(),
        });
    }

    challenger.observe(F::from_usize(statements.len()));
    let mut gammas = Vec::with_capacity(statements.len());
    let mut targets = Vec::with_capacity(statements.len());
    for statement in statements {
        let gamma = statement.observe_and_sample_gamma::<F, _>(challenger);
        gammas.push(gamma);
        targets.push(statement.batched_target(gamma));
    }

    let mut evals = Vec::with_capacity(statements.len());
    let mut weights = compact_batched_root_weights(code, statements, &gammas)?;
    for ((poly, weight), &target) in polys.iter().zip(&weights).zip(&targets) {
        match poly {
            NativeWarpDirectBatchedResidualPoly::Base(values) => {
                if values.len() != code.msg_len() {
                    return Err(LinearSigmaReductionError::ArityMismatch {
                        expected: code.log_msg_len(),
                        actual: log2_strict_usize(values.len()),
                    });
                }
                let actual = weight
                    .as_slice()
                    .iter()
                    .zip(values.iter())
                    .map(|(&weight, &value)| weight * value)
                    .sum::<EF>();
                if actual != target {
                    return Err(LinearSigmaReductionError::UnsatisfiedStatement);
                }
                evals.push(NativeWarpCompactEvalPoly::from_direct(poly));
            }
            NativeWarpDirectBatchedResidualPoly::Extension(values) => {
                if values.len() != code.msg_len() {
                    return Err(LinearSigmaReductionError::ArityMismatch {
                        expected: code.log_msg_len(),
                        actual: log2_strict_usize(values.len()),
                    });
                }
                let actual = weight
                    .as_slice()
                    .iter()
                    .zip(values.iter())
                    .map(|(&weight, &value)| weight * value)
                    .sum::<EF>();
                if actual != target {
                    return Err(LinearSigmaReductionError::UnsatisfiedStatement);
                }
                evals.push(NativeWarpCompactEvalPoly::from_direct(poly));
            }
        }
    }

    let rho: EF = challenger.sample_algebra_element();
    let scales = rho.powers().take(statements.len()).collect();
    let mut claimed_sum = scales
        .iter()
        .zip(&targets)
        .map(|(&scale, &target)| scale * target)
        .sum::<EF>();

    let mut sumcheck = SumcheckData::default();
    let mut point = Vec::with_capacity(code.log_msg_len());
    for _ in 0..code.log_msg_len() {
        let mut c0 = EF::ZERO;
        let mut c_inf = EF::ZERO;
        for ((evals, weights), &scale) in evals.iter().zip(weights.iter()).zip(&scales) {
            let (local_c0, local_c_inf) = evals.sumcheck_coefficients(weights.as_slice());
            c0 += scale * local_c0;
            c_inf += scale * local_c_inf;
        }

        let r = sumcheck.observe_and_sample(challenger, c0, c_inf, pow_bits);
        for (evals, weights) in evals.iter_mut().zip(weights.iter_mut()) {
            evals.fix_prefix_var_mut(r);
            weights.fix_prefix_var_mut(r);
        }
        claimed_sum = extrapolate_01inf(c0, claimed_sum - c0, c_inf, r);
        point.push(r);
    }

    let coeffs = scales
        .iter()
        .zip(&weights)
        .map(|(&scale, weights)| scale * weights.as_slice()[0])
        .collect::<Vec<_>>();
    let virtual_eval = coeffs
        .iter()
        .zip(&evals)
        .map(|(&coeff, evals)| coeff * evals.final_value())
        .sum::<EF>();
    if claimed_sum != virtual_eval {
        return Err(LinearSigmaReductionError::FinalCheckFailed);
    }

    challenger.observe_algebra_element(virtual_eval);

    Ok((
        BatchedLinearSigmaReductionProof {
            sumcheck,
            virtual_eval,
        },
        BatchedLinearSigmaOpeningClaim {
            point: Point::new(point),
            coeffs,
            value: virtual_eval,
        },
    ))
}

/// Verifier counterpart of [`prove_compact_batched_root_reduction`].
///
/// The verifier repeats the same Fiat-Shamir transcript, recomputes the
/// batched targets, verifies the sumcheck, and then computes the residual
/// coefficients `c_k = rho^k a_k(r)`. The returned
/// [`BatchedLinearSigmaOpeningClaim`] is deliberately not accepted on its own:
/// the caller must pass it to WHIR so the committed RS oracles are actually
/// opened at `r`.
pub(super) fn verify_compact_batched_root_reduction<F, EF, Dft, Challenger>(
    code: &ReedSolomonCode<F, Dft>,
    statements: &[NativeWarpCompactRootStatement<EF>],
    proof: &BatchedLinearSigmaReductionProof<F, EF>,
    challenger: &mut Challenger,
    pow_bits: usize,
) -> Result<BatchedLinearSigmaOpeningClaim<EF>, LinearSigmaReductionError>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F> + TwoAdicField + Send + Sync,
    Dft: TwoAdicSubgroupDft<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if statements.is_empty() {
        return Err(LinearSigmaReductionError::EmptyStatement);
    }
    if proof.sumcheck.num_rounds() != code.log_msg_len() {
        return Err(SumcheckError::RoundCountMismatch {
            expected: code.log_msg_len(),
            actual: proof.sumcheck.num_rounds(),
        }
        .into());
    }

    challenger.observe(F::from_usize(statements.len()));
    let mut gammas = Vec::with_capacity(statements.len());
    let mut targets = Vec::with_capacity(statements.len());
    for statement in statements {
        let gamma = statement.observe_and_sample_gamma::<F, _>(challenger);
        gammas.push(gamma);
        targets.push(statement.batched_target(gamma));
    }

    let rho: EF = challenger.sample_algebra_element();
    let scales = rho.powers().take(statements.len()).collect();
    let mut claimed_sum = scales
        .iter()
        .zip(&targets)
        .map(|(&scale, &target)| scale * target)
        .sum::<EF>();

    let point = proof
        .sumcheck
        .verify_rounds(challenger, &mut claimed_sum, pow_bits)?;
    challenger.observe_algebra_element(proof.virtual_eval);
    if claimed_sum != proof.virtual_eval {
        return Err(LinearSigmaReductionError::FinalCheckFailed);
    }

    let mut encoded_message_eq = None;
    let mut encoded_message_eq_poly = None;
    let coeffs = scales
        .iter()
        .zip(&gammas)
        .zip(statements)
        .map(|((&scale, &gamma), statement)| {
            let local = statement
                .batched_weight_eval_from_message_eq_point::<F, Dft>(code, point.as_slice(), gamma)
                .unwrap_or_else(|| {
                    let encoded = encoded_message_eq.get_or_insert_with(|| {
                        let message_eq = Poly::<EF>::new_from_point(point.as_slice(), EF::ONE);
                        code.encode_algebra(message_eq.as_slice())
                    });
                    let poly =
                        encoded_message_eq_poly.get_or_insert_with(|| Poly::new(encoded.clone()));
                    statement.batched_weight_eval_from_encoded_eq::<F>(encoded, poly, gamma)
                });
            scale * local
        })
        .collect::<Vec<_>>();

    Ok(BatchedLinearSigmaOpeningClaim {
        point,
        coeffs,
        value: proof.virtual_eval,
    })
}

/// Build a generic equality statement for an already-reduced residual claim.
///
/// This supports the fallback path where an oracle was reduced separately
/// before the root compiler attempts to batch the remaining residual openings.
pub(super) fn residual_eq_statement<F, EF>(
    residual: &NativeWarpWhirRootResidualClaim<EF>,
) -> LinearSigmaStatement<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut eq = EqStatement::initialize(residual.opening.point.num_variables());
    eq.add_evaluated_constraint(residual.opening.point.clone(), residual.opening.value);
    let mut statement = LinearSigmaStatement::initialize(residual.opening.point.num_variables());
    statement.add_constraint(LinearSigmaConstraint::from_eq_statement::<F>(&eq, EF::ONE));
    statement
}

/// Absorb the full typed WARP root commitment into Fiat-Shamir.
///
/// The oracle id, length, field tag, and underlying WHIR commitment payload are
/// all bound. This prevents reusing a valid WHIR commitment for a different
/// WARP transcript position, field kind, or arity.
pub(super) fn observe_native_root_commitment<F, Challenger, Comm>(
    challenger: &mut Challenger,
    commitment: &RootIopBoundCommitment<NativeWarpWhirRootCommitment<Comm>>,
) where
    F: Field + PrimeCharacteristicRing,
    Comm: Clone,
    Challenger: FieldChallenger<F> + CanObserve<Comm>,
{
    challenger.observe(F::from_usize(commitment.oracle_id));
    challenger.observe(F::from_usize(commitment.log_len));
    challenger.observe(F::from_usize(match commitment.field {
        RootIopOracleField::Base => 0,
        RootIopOracleField::Extension => 1,
    }));
    commitment
        .commitment
        .observe_payload_into::<F, _>(challenger);
}

/// Evaluate the multilinear equality polynomial `eq_lhs(rhs)`.
fn eval_eq_point<EF: Field>(lhs: &[EF], rhs: &[EF]) -> EF {
    debug_assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(&l, &r)| l * r + (EF::ONE - l) * (EF::ONE - r))
        .product()
}

/// Evaluate `eq_point(b_index)` for a Boolean hypercube vertex.
fn eval_eq_at_hypercube_index<EF: Field>(point: &[EF], index: usize) -> EF {
    let num_variables = point.len();
    point
        .iter()
        .enumerate()
        .map(|(bit, &coord)| {
            if (index >> (num_variables - 1 - bit)) & 1 == 1 {
                coord
            } else {
                EF::ONE - coord
            }
        })
        .product()
}
