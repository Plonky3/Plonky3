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

/// Lazy verifier cache for evaluating residual WARP weights at the final
/// sumcheck point.
///
/// The verifier needs values of the form `a_q(r)`, where `a_q` is the
/// message-domain linear functional obtained from a WARP codeword query `q`
/// through the RS adjoint. By duality,
///
/// ```text
///     a_q(r) = <C^T q, eq_r> = <q, C(eq_r)>.
/// ```
///
/// Older code materialized the full encoded vector `C(eq_r)` as soon as any
/// query left the systematic message subspace. This cache keeps that fallback
/// for true codeword-MLE claims, but answers sparse index claims by evaluating
/// only the required systematic RS cosets.
struct EncodedMessageEqCache<'a, F, EF, Dft>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    code: &'a ReedSolomonCode<F, Dft>,
    message_point: &'a [EF],
    message_eq: Option<Vec<EF>>,
    full_encoded: Option<Vec<EF>>,
    systematic_cosets: Vec<Option<Vec<EF>>>,
}

impl<'a, F, EF, Dft> EncodedMessageEqCache<'a, F, EF, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + Send + Sync,
    Dft: TwoAdicSubgroupDft<F>,
{
    fn new(code: &'a ReedSolomonCode<F, Dft>, message_point: &'a [EF]) -> Self {
        let coset_count = if code.is_systematic() {
            1 << code.log_inv_rate()
        } else {
            0
        };
        Self {
            code,
            message_point,
            message_eq: None,
            full_encoded: None,
            systematic_cosets: (0..coset_count).map(|_| None).collect(),
        }
    }

    fn message_eq(&mut self) -> &[EF] {
        self.message_eq.get_or_insert_with(|| {
            Poly::<EF>::new_from_point(self.message_point, EF::ONE).into_evals()
        })
    }

    fn full_encoded(&mut self) -> &[EF] {
        if self.full_encoded.is_none() {
            let message_eq = self.message_eq().to_vec();
            self.full_encoded = Some(self.code.encode_algebra(&message_eq));
        }
        self.full_encoded.as_ref().expect("encoded eq is set")
    }

    fn systematic_coset(&mut self, residue: usize) -> &[EF] {
        debug_assert!(self.code.is_systematic());
        if residue == 0 {
            return self.message_eq();
        }

        if self.systematic_cosets[residue].is_none() {
            let message_eq = self.message_eq().to_vec();
            self.systematic_cosets[residue] = Some(
                self.code
                    .encode_algebra_systematic_coset(&message_eq, residue),
            );
        }
        self.systematic_cosets[residue]
            .as_ref()
            .expect("systematic coset is set")
    }

    fn codeword_index_value(&mut self, index: usize) -> EF {
        if self.code.is_systematic() {
            let stride = 1 << self.code.log_inv_rate();
            let residue = index & (stride - 1);
            let row = index >> self.code.log_inv_rate();
            self.systematic_coset(residue)[row]
        } else {
            self.full_encoded()[index]
        }
    }

    fn codeword_mle_value(&mut self, point: &[EF]) -> EF {
        let point = Point::new(point.to_vec());
        Poly::<EF>::eval_ext_slice::<F>(self.full_encoded(), &point)
    }
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

    /// Evaluate the batched query weight at the final sumcheck point.
    ///
    /// Each constraint is handled independently:
    ///
    /// - systematic message-subspace claims use the ordinary multilinear
    ///   equality value `eq_r(b)`;
    /// - sparse non-systematic index claims read only the needed systematic RS
    ///   coset of `C(eq_r)`;
    /// - full codeword-MLE claims fall back to the dense encoded equality
    ///   vector, but evaluate it by borrowed slice instead of cloning it into
    ///   a temporary polynomial.
    fn batched_weight_eval_at_sumcheck_point<F, Dft>(
        &self,
        code: &ReedSolomonCode<F, Dft>,
        message_point: &[EF],
        gamma: EF,
        cache: &mut EncodedMessageEqCache<'_, F, EF, Dft>,
    ) -> EF
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + Send + Sync,
        Dft: TwoAdicSubgroupDft<F>,
    {
        let stride = 1 << code.log_inv_rate();
        let mut scale = EF::ONE;
        let mut value = EF::ZERO;
        for constraint in &self.constraints {
            let local = match &constraint.query {
                NativeWarpCompactRootQuery::Index(index) => {
                    if code.is_systematic() && index.is_multiple_of(stride) {
                        eval_eq_at_hypercube_index(message_point, index / stride)
                    } else {
                        cache.codeword_index_value(*index)
                    }
                }
                NativeWarpCompactRootQuery::Mle(point) => {
                    if code.is_systematic() && point.len() == code.log_codeword_len() {
                        let (prefix, suffix) = point.split_at(code.log_msg_len());
                        if suffix.iter().all(|&coord| coord == EF::ZERO) {
                            eval_eq_point(message_point, prefix)
                        } else {
                            cache.codeword_mle_value(point)
                        }
                    } else if point.len() == code.log_msg_len() {
                        eval_eq_point(message_point, point)
                    } else {
                        cache.codeword_mle_value(point)
                    }
                }
            };
            value += scale * local;
            scale *= gamma;
        }
        value
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

    let mut eq_cache = EncodedMessageEqCache::new(code, point.as_slice());
    let coeffs = scales
        .iter()
        .zip(&gammas)
        .zip(statements)
        .map(|((&scale, &gamma), statement)| {
            let local = statement.batched_weight_eval_at_sumcheck_point::<F, Dft>(
                code,
                point.as_slice(),
                gamma,
                &mut eq_cache,
            );
            scale * local
        })
        .collect::<Vec<_>>();

    Ok(BatchedLinearSigmaOpeningClaim {
        point,
        coeffs,
        value: proof.virtual_eval,
    })
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

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Dft = Radix2DFTSmallBatch<F>;

    #[test]
    fn compact_root_verifier_weights_match_dense_rs_duality() {
        let code = ReedSolomonCode::new_systematic(3, 2, Dft::default());
        let mut statement = NativeWarpCompactRootStatement::initialize(code.log_msg_len());
        statement.add_index(0, EF::from_u64(11));
        statement.add_index(1, EF::from_u64(13));
        statement.add_index(6, EF::from_u64(17));
        let codeword_point = vec![
            EF::from_u64(3),
            EF::from_u64(5),
            EF::from_u64(7),
            EF::from_u64(11),
            EF::from_u64(13),
        ];
        statement.add_mle(codeword_point.clone(), EF::from_u64(19));

        let sumcheck_point = vec![EF::from_u64(23), EF::from_u64(29), EF::from_u64(31)];
        let gamma = EF::from_u64(37);
        let mut cache = EncodedMessageEqCache::new(&code, &sumcheck_point);
        let actual = statement.batched_weight_eval_at_sumcheck_point::<F, Dft>(
            &code,
            &sumcheck_point,
            gamma,
            &mut cache,
        );

        let message_eq = Poly::<EF>::new_from_point(&sumcheck_point, EF::ONE);
        let encoded_eq = code.encode_algebra(message_eq.as_slice());
        let encoded_eq_poly = Poly::new(encoded_eq.clone());
        let mut scale = EF::ONE;
        let mut expected = EF::ZERO;
        for constraint in &statement.constraints {
            let local = match &constraint.query {
                NativeWarpCompactRootQuery::Index(index) => encoded_eq[*index],
                NativeWarpCompactRootQuery::Mle(point) => {
                    encoded_eq_poly.eval_ext::<F>(&Point::new(point.clone()))
                }
            };
            expected += scale * local;
            scale *= gamma;
        }

        assert_eq!(actual, expected);
    }
}
