//! Linear Sigma-IOP constraints for WHIR's constrained-RS compiler.
//!
//! WHIR's Section 7 compiler reduces linear Sigma-IOP queries to
//! constrained Reed-Solomon proximity checks. This module provides the
//! explicit constraint object for the linear case:
//!
//! ```text
//!     sum_{b in {0,1}^m} a(b) * f_hat(b) = sigma
//! ```
//!
//! Equality openings are a special case with `a(b) = eq(z, b)`.

use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::constraints::statement::EqStatement;
use crate::sumcheck::product_polynomial::ProductPolynomial;
use crate::sumcheck::strategy::{SumcheckProver, VariableOrder};
use crate::sumcheck::{SumcheckData, SumcheckError, extrapolate_01inf};

/// Errors from reducing a linear Sigma statement to one residual oracle opening.
#[derive(Error, Debug)]
pub enum LinearSigmaReductionError {
    /// The statement has no constraints to prove.
    #[error("linear Sigma statement is empty")]
    EmptyStatement,

    /// The oracle arity does not match the statement arity.
    #[error("linear Sigma arity mismatch: expected {expected}, got {actual}")]
    ArityMismatch { expected: usize, actual: usize },

    /// The honest prover's oracle does not satisfy the sampled batched statement.
    #[error("linear Sigma statement is not satisfied by the oracle")]
    UnsatisfiedStatement,

    /// The sumcheck transcript is malformed.
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError),

    /// The final sumcheck claim is inconsistent with the residual oracle opening.
    #[error("linear Sigma final check failed")]
    FinalCheckFailed,
}

/// Sumcheck transcript reducing a linear Sigma claim to one oracle evaluation.
///
/// This is not a complete PCS proof by itself. Verification returns the
/// residual claim `f(r) = oracle_eval`; a later constrained-RS/PCS layer must
/// bind that value to the committed oracle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearSigmaReductionProof<F, EF> {
    /// Sumcheck round messages for `sum_b weights(b) * f(b)`.
    pub sumcheck: SumcheckData<F, EF>,
    /// Claimed value of the oracle at the verifier-derived final point.
    pub oracle_eval: EF,
}

/// Residual opening claim emitted by a successful linear Sigma reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearSigmaOpeningClaim<EF> {
    /// Verifier-derived point `r`.
    pub point: Point<EF>,
    /// Claimed oracle value `f(r)`.
    pub value: EF,
}

/// Sumcheck transcript reducing several linear Sigma claims over several
/// same-arity oracles to one virtual oracle opening.
///
/// Verification returns coefficients `c_i` and a point `r` such that the next
/// PCS layer must prove:
///
/// ```text
///     sum_i c_i * f_i(r) = virtual_eval
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchedLinearSigmaReductionProof<F, EF> {
    /// Sumcheck round messages for
    /// `sum_x sum_i rho_i * weights_i(x) * f_i(x)`.
    pub sumcheck: SumcheckData<F, EF>,
    /// Claimed virtual opening at the verifier-derived final point.
    pub virtual_eval: EF,
}

/// Virtual opening claim emitted by a batched linear Sigma reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedLinearSigmaOpeningClaim<EF> {
    /// Verifier-derived point shared by every oracle opening.
    pub point: Point<EF>,
    /// Per-oracle opening coefficients at `point`.
    pub coeffs: Vec<EF>,
    /// Claimed value of `sum_i coeffs[i] * f_i(point)`.
    pub value: EF,
}

/// Prover-side oracle participating in a batched linear Sigma reduction.
pub struct BatchedLinearSigmaProverOracle<'a, F, EF> {
    /// Public linear Sigma statement for this oracle.
    pub statement: &'a LinearSigmaStatement<EF>,
    /// Oracle evaluations over the Boolean hypercube.
    pub values: BatchedLinearSigmaOracleValues<'a, F, EF>,
}

/// Field of a prover-side oracle participating in a batched reduction.
pub enum BatchedLinearSigmaOracleValues<'a, F, EF> {
    /// Base-field oracle evaluations.
    Base(&'a Poly<F>),
    /// Extension-field oracle evaluations.
    Extension(&'a Poly<EF>),
}

impl<'a, F, EF> BatchedLinearSigmaProverOracle<'a, F, EF> {
    /// Create a base-field batched reduction oracle.
    #[must_use]
    pub const fn base(statement: &'a LinearSigmaStatement<EF>, values: &'a Poly<F>) -> Self {
        Self {
            statement,
            values: BatchedLinearSigmaOracleValues::Base(values),
        }
    }

    /// Create an extension-field batched reduction oracle.
    #[must_use]
    pub const fn extension(statement: &'a LinearSigmaStatement<EF>, values: &'a Poly<EF>) -> Self {
        Self {
            statement,
            values: BatchedLinearSigmaOracleValues::Extension(values),
        }
    }
}

/// Prove a batched linear Sigma reduction across several same-arity oracles.
///
/// This is the algebraic compiler layer needed when several already committed
/// oracles have separate linear claims, but the PCS should authenticate only
/// one same-point virtual opening. The caller must bind every oracle commitment
/// into `challenger` before calling this function.
pub fn prove_batched_linear_sigma_reduction<F, EF, Challenger>(
    oracles: &[BatchedLinearSigmaProverOracle<'_, F, EF>],
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
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let (mut evals, mut weights, scales, mut claimed_sum, num_variables) =
        prepare_batched_linear_sigma_prover(oracles, challenger)?;

    let mut sumcheck = SumcheckData::default();
    let mut point = Vec::with_capacity(num_variables);
    for _ in 0..num_variables {
        let mut c0 = EF::ZERO;
        let mut c_inf = EF::ZERO;
        for ((evals, weights), &scale) in evals.iter().zip_eq(weights.iter()).zip_eq(&scales) {
            let (local_c0, local_c_inf) =
                VariableOrder::Prefix.sumcheck_coefficients(evals.as_slice(), weights.as_slice());
            c0 += scale * local_c0;
            c_inf += scale * local_c_inf;
        }

        let r = sumcheck.observe_and_sample(challenger, c0, c_inf, pow_bits);
        for (evals, weights) in evals.iter_mut().zip_eq(weights.iter_mut()) {
            evals.fix_prefix_var_mut(r);
            weights.fix_prefix_var_mut(r);
        }
        claimed_sum = extrapolate_01inf(c0, claimed_sum - c0, c_inf, r);
        point.push(r);
    }

    let coeffs = scales
        .iter()
        .zip_eq(&weights)
        .map(|(&scale, weights)| scale * weights.as_slice()[0])
        .collect::<Vec<_>>();
    let virtual_eval = coeffs
        .iter()
        .zip_eq(&evals)
        .map(|(&coeff, evals)| coeff * evals.as_slice()[0])
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

/// Verify a batched linear Sigma reduction and return the virtual same-point
/// opening claim to be checked by a PCS.
pub fn verify_batched_linear_sigma_reduction<F, EF, Challenger>(
    statements: &[&LinearSigmaStatement<EF>],
    proof: &BatchedLinearSigmaReductionProof<F, EF>,
    challenger: &mut Challenger,
    pow_bits: usize,
) -> Result<BatchedLinearSigmaOpeningClaim<EF>, LinearSigmaReductionError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let (constraints, scales, mut claimed_sum, num_variables) =
        prepare_batched_linear_sigma_verifier::<F, EF, _>(statements, challenger)?;
    if proof.sumcheck.num_rounds() != num_variables {
        return Err(SumcheckError::RoundCountMismatch {
            expected: num_variables,
            actual: proof.sumcheck.num_rounds(),
        }
        .into());
    }

    let point = proof
        .sumcheck
        .verify_rounds(challenger, &mut claimed_sum, pow_bits)?;
    challenger.observe_algebra_element(proof.virtual_eval);

    if claimed_sum != proof.virtual_eval {
        return Err(LinearSigmaReductionError::FinalCheckFailed);
    }

    let coeffs = scales
        .iter()
        .zip_eq(&constraints)
        .map(|(&scale, constraint)| scale * constraint.weights.eval_ext::<F>(&point))
        .collect::<Vec<_>>();

    Ok(BatchedLinearSigmaOpeningClaim {
        point,
        coeffs,
        value: proof.virtual_eval,
    })
}

type BatchedProverState<EF> = (Vec<Poly<EF>>, Vec<Poly<EF>>, Vec<EF>, EF, usize);
type BatchedVerifierState<EF> = (Vec<LinearSigmaConstraint<EF>>, Vec<EF>, EF, usize);

fn prepare_batched_linear_sigma_prover<F, EF, Challenger>(
    oracles: &[BatchedLinearSigmaProverOracle<'_, F, EF>],
    challenger: &mut Challenger,
) -> Result<BatchedProverState<EF>, LinearSigmaReductionError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if oracles.is_empty() {
        return Err(LinearSigmaReductionError::EmptyStatement);
    }
    challenger.observe(F::from_u64(oracles.len() as u64));

    let num_variables = oracles[0].statement.num_variables();
    let mut evals = Vec::with_capacity(oracles.len());
    let mut constraints = Vec::with_capacity(oracles.len());
    for oracle in oracles {
        if oracle.statement.num_variables() != num_variables {
            return Err(LinearSigmaReductionError::ArityMismatch {
                expected: num_variables,
                actual: oracle.statement.num_variables(),
            });
        }
        let batched = oracle
            .statement
            .sample_batched_constraint::<F, _>(challenger)?;
        match oracle.values {
            BatchedLinearSigmaOracleValues::Base(poly) => {
                if poly.num_variables() != num_variables {
                    return Err(LinearSigmaReductionError::ArityMismatch {
                        expected: num_variables,
                        actual: poly.num_variables(),
                    });
                }
                if batched.evaluate_base(poly) != batched.target {
                    return Err(LinearSigmaReductionError::UnsatisfiedStatement);
                }
                evals.push(Poly::new(
                    poly.as_slice()
                        .iter()
                        .map(|&value| EF::from(value))
                        .collect(),
                ));
            }
            BatchedLinearSigmaOracleValues::Extension(poly) => {
                if poly.num_variables() != num_variables {
                    return Err(LinearSigmaReductionError::ArityMismatch {
                        expected: num_variables,
                        actual: poly.num_variables(),
                    });
                }
                if batched.evaluate_ext(poly) != batched.target {
                    return Err(LinearSigmaReductionError::UnsatisfiedStatement);
                }
                evals.push(poly.clone());
            }
        }
        constraints.push(batched);
    }

    let rho: EF = challenger.sample_algebra_element();
    let scales: Vec<_> = rho.powers().take(oracles.len()).collect();
    let claimed_sum = scales
        .iter()
        .zip_eq(&constraints)
        .map(|(&scale, constraint)| scale * constraint.target)
        .sum::<EF>();
    let weights = constraints
        .into_iter()
        .map(|constraint| constraint.weights)
        .collect::<Vec<_>>();

    Ok((evals, weights, scales, claimed_sum, num_variables))
}

fn prepare_batched_linear_sigma_verifier<F, EF, Challenger>(
    statements: &[&LinearSigmaStatement<EF>],
    challenger: &mut Challenger,
) -> Result<BatchedVerifierState<EF>, LinearSigmaReductionError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if statements.is_empty() {
        return Err(LinearSigmaReductionError::EmptyStatement);
    }
    challenger.observe(F::from_u64(statements.len() as u64));

    let num_variables = statements[0].num_variables();
    let mut constraints = Vec::with_capacity(statements.len());
    for statement in statements {
        if statement.num_variables() != num_variables {
            return Err(LinearSigmaReductionError::ArityMismatch {
                expected: num_variables,
                actual: statement.num_variables(),
            });
        }
        constraints.push(statement.sample_batched_constraint::<F, _>(challenger)?);
    }

    let rho: EF = challenger.sample_algebra_element();
    let scales: Vec<_> = rho.powers().take(statements.len()).collect();
    let claimed_sum = scales
        .iter()
        .zip_eq(&constraints)
        .map(|(&scale, constraint)| scale * constraint.target)
        .sum::<EF>();

    Ok((constraints, scales, claimed_sum, num_variables))
}

/// One linear Sigma-IOP query over a multilinear oracle.
///
/// The `weights` polynomial is stored by evaluations over `{0,1}^m`.
/// The constraint accepts an oracle polynomial `f` when:
///
/// ```text
///     sum_b weights[b] * f(b) == target
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearSigmaConstraint<EF> {
    /// Weight polynomial evaluations over the Boolean hypercube.
    pub weights: Poly<EF>,
    /// Claimed weighted sum.
    pub target: EF,
}

impl<EF: Field> LinearSigmaConstraint<EF> {
    /// Create a linear Sigma constraint.
    ///
    /// # Panics
    ///
    /// Panics if `weights` is empty, which would not define a hypercube.
    #[must_use]
    pub fn new(weights: Poly<EF>, target: EF) -> Self {
        assert!(!weights.as_slice().is_empty(), "empty Sigma weight table");
        Self { weights, target }
    }

    /// Number of variables in the hypercube.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.weights.num_variables()
    }

    /// Build the linear-Sigma form of a batched equality statement.
    ///
    /// This uses the same batching kernel as [`EqStatement`], so
    /// `p(z_i) = v_i` batched with `gamma` becomes:
    ///
    /// ```text
    ///     sum_b (sum_i gamma^i * eq(z_i, b)) * p(b)
    ///       = sum_i gamma^i * v_i
    /// ```
    #[must_use]
    pub fn from_eq_statement<Base>(statement: &EqStatement<EF>, gamma: EF) -> Self
    where
        Base: Field,
        EF: ExtensionField<Base>,
    {
        let mut weights = Poly::zero(statement.num_variables());
        let mut target = EF::ZERO;
        statement.combine_hypercube::<Base, false>(&mut weights, &mut target, gamma);
        Self::new(weights, target)
    }

    /// Evaluate the weighted sum against a base-field oracle.
    ///
    /// # Panics
    ///
    /// Panics if `poly` and `weights` are over different hypercube sizes.
    #[must_use]
    pub fn evaluate_base<Base>(&self, poly: &Poly<Base>) -> EF
    where
        Base: Field,
        EF: ExtensionField<Base>,
    {
        assert_eq!(
            self.weights.num_variables(),
            poly.num_variables(),
            "linear Sigma constraint arity mismatch",
        );
        self.weights
            .as_slice()
            .iter()
            .zip_eq(poly.as_slice())
            .map(|(&weight, &value)| weight * value)
            .sum()
    }

    /// Evaluate the weighted sum against an extension-field oracle.
    ///
    /// # Panics
    ///
    /// Panics if `poly` and `weights` are over different hypercube sizes.
    #[must_use]
    pub fn evaluate_ext(&self, poly: &Poly<EF>) -> EF {
        assert_eq!(
            self.weights.num_variables(),
            poly.num_variables(),
            "linear Sigma constraint arity mismatch",
        );
        self.weights
            .as_slice()
            .iter()
            .zip_eq(poly.as_slice())
            .map(|(&weight, &value)| weight * value)
            .sum()
    }

    /// Verify this constraint against a base-field oracle.
    #[must_use]
    pub fn verify_base<Base>(&self, poly: &Poly<Base>) -> bool
    where
        Base: Field,
        EF: ExtensionField<Base>,
    {
        self.evaluate_base(poly) == self.target
    }

    /// Verify this constraint against an extension-field oracle.
    #[must_use]
    pub fn verify_ext(&self, poly: &Poly<EF>) -> bool {
        self.evaluate_ext(poly) == self.target
    }

    /// Add `scale * other` into this constraint.
    ///
    /// This is the local form of WHIR Construction 5.5's random linear
    /// batching for multi-constrained RS statements.
    ///
    /// # Panics
    ///
    /// Panics if both constraints do not use the same hypercube arity.
    pub fn add_scaled(&mut self, scale: EF, other: &Self) {
        assert_eq!(
            self.num_variables(),
            other.num_variables(),
            "linear Sigma batching arity mismatch",
        );
        for (slot, &rhs) in self
            .weights
            .as_mut_slice()
            .iter_mut()
            .zip_eq(other.weights.as_slice())
        {
            *slot += scale * rhs;
        }
        self.target += scale * other.target;
    }
}

/// A list of linear Sigma constraints over the same oracle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearSigmaStatement<EF> {
    num_variables: usize,
    constraints: Vec<LinearSigmaConstraint<EF>>,
}

impl<EF: Field> LinearSigmaStatement<EF> {
    /// Create an empty statement for `num_variables` variables.
    #[must_use]
    pub const fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            constraints: Vec::new(),
        }
    }

    /// Number of variables shared by every constraint.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Number of constraints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Whether the statement is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Iterate over constraints.
    pub fn iter(&self) -> impl Iterator<Item = &LinearSigmaConstraint<EF>> {
        self.constraints.iter()
    }

    /// Add a constraint.
    ///
    /// # Panics
    ///
    /// Panics if the constraint arity differs from the statement arity.
    pub fn add_constraint(&mut self, constraint: LinearSigmaConstraint<EF>) {
        assert_eq!(
            self.num_variables,
            constraint.num_variables(),
            "linear Sigma statement arity mismatch",
        );
        self.constraints.push(constraint);
    }

    /// Batch all constraints with powers of `gamma`.
    ///
    /// Returns `None` for an empty statement. Otherwise returns:
    ///
    /// ```text
    ///     sum_i gamma^i * constraint_i
    /// ```
    #[must_use]
    pub fn batch(&self, gamma: EF) -> Option<LinearSigmaConstraint<EF>> {
        let mut constraints = self.constraints.iter();
        let first = constraints.next()?.clone();
        let mut batched = first;
        for (scale, constraint) in gamma.powers().skip(1).zip(constraints) {
            batched.add_scaled(scale, constraint);
        }
        Some(batched)
    }

    /// Verify every constraint against a base-field oracle.
    #[must_use]
    pub fn verify_base<Base>(&self, poly: &Poly<Base>) -> bool
    where
        Base: Field,
        EF: ExtensionField<Base>,
    {
        self.constraints
            .iter()
            .all(|constraint| constraint.verify_base(poly))
    }

    /// Verify every constraint against an extension-field oracle.
    #[must_use]
    pub fn verify_ext(&self, poly: &Poly<EF>) -> bool {
        self.constraints
            .iter()
            .all(|constraint| constraint.verify_ext(poly))
    }

    /// Prove this statement against a base-field oracle by reducing it to one
    /// residual oracle opening.
    ///
    /// The Fiat-Shamir order is:
    ///
    /// 1. sample the random batching challenge for all linear constraints,
    /// 2. run the full quadratic sumcheck over the batched product
    ///    `weights(b) * f(b)`,
    /// 3. absorb the residual oracle value `f(r)`.
    ///
    /// The caller must ensure the oracle commitment and all public statement
    /// data have already been bound into `challenger`.
    pub fn prove_reduction_base<F, Challenger>(
        &self,
        poly: &Poly<F>,
        challenger: &mut Challenger,
        pow_bits: usize,
    ) -> Result<
        (
            LinearSigmaReductionProof<F, EF>,
            LinearSigmaOpeningClaim<EF>,
        ),
        LinearSigmaReductionError,
    >
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.check_oracle_arity(poly.num_variables())?;
        let batched = self.sample_batched_constraint::<F, _>(challenger)?;
        if batched.evaluate_base(poly) != batched.target {
            return Err(LinearSigmaReductionError::UnsatisfiedStatement);
        }

        let evals = Poly::new(
            poly.as_slice()
                .iter()
                .map(|&value| EF::from(value))
                .collect(),
        );
        self.prove_batched_reduction(evals, batched, challenger, pow_bits)
    }

    /// Prove this statement against an extension-field oracle by reducing it to
    /// one residual oracle opening.
    pub fn prove_reduction_ext<F, Challenger>(
        &self,
        poly: &Poly<EF>,
        challenger: &mut Challenger,
        pow_bits: usize,
    ) -> Result<
        (
            LinearSigmaReductionProof<F, EF>,
            LinearSigmaOpeningClaim<EF>,
        ),
        LinearSigmaReductionError,
    >
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.check_oracle_arity(poly.num_variables())?;
        let batched = self.sample_batched_constraint::<F, _>(challenger)?;
        if batched.evaluate_ext(poly) != batched.target {
            return Err(LinearSigmaReductionError::UnsatisfiedStatement);
        }

        self.prove_batched_reduction(poly.clone(), batched, challenger, pow_bits)
    }

    /// Verify a linear Sigma reduction and return the residual oracle opening
    /// claim that still has to be checked against the committed oracle.
    pub fn verify_reduction<F, Challenger>(
        &self,
        proof: &LinearSigmaReductionProof<F, EF>,
        challenger: &mut Challenger,
        pow_bits: usize,
    ) -> Result<LinearSigmaOpeningClaim<EF>, LinearSigmaReductionError>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let batched = self.sample_batched_constraint::<F, _>(challenger)?;
        if proof.sumcheck.num_rounds() != self.num_variables {
            return Err(SumcheckError::RoundCountMismatch {
                expected: self.num_variables,
                actual: proof.sumcheck.num_rounds(),
            }
            .into());
        }

        let mut claimed_sum = batched.target;
        let point = proof
            .sumcheck
            .verify_rounds(challenger, &mut claimed_sum, pow_bits)?;
        challenger.observe_algebra_element(proof.oracle_eval);

        let weight_eval = batched.weights.eval_ext::<F>(&point);
        if claimed_sum != weight_eval * proof.oracle_eval {
            return Err(LinearSigmaReductionError::FinalCheckFailed);
        }

        Ok(LinearSigmaOpeningClaim {
            point,
            value: proof.oracle_eval,
        })
    }

    fn check_oracle_arity(&self, actual: usize) -> Result<(), LinearSigmaReductionError> {
        if self.num_variables == actual {
            Ok(())
        } else {
            Err(LinearSigmaReductionError::ArityMismatch {
                expected: self.num_variables,
                actual,
            })
        }
    }

    fn sample_batched_constraint<F, Challenger>(
        &self,
        challenger: &mut Challenger,
    ) -> Result<LinearSigmaConstraint<EF>, LinearSigmaReductionError>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.observe_public_statement::<F, _>(challenger);
        let gamma: EF = challenger.sample_algebra_element();
        self.batch(gamma)
            .ok_or(LinearSigmaReductionError::EmptyStatement)
    }

    fn observe_public_statement<F, Challenger>(&self, challenger: &mut Challenger)
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + CanObserve<F>,
    {
        challenger.observe(F::from_u64(self.num_variables as u64));
        challenger.observe(F::from_u64(self.constraints.len() as u64));
        for constraint in &self.constraints {
            challenger.observe(F::from_u64(constraint.weights.as_slice().len() as u64));
            for &weight in constraint.weights.as_slice() {
                challenger.observe_algebra_element(weight);
            }
            challenger.observe_algebra_element(constraint.target);
        }
    }

    fn prove_batched_reduction<F, Challenger>(
        &self,
        evals: Poly<EF>,
        batched: LinearSigmaConstraint<EF>,
        challenger: &mut Challenger,
        pow_bits: usize,
    ) -> Result<
        (
            LinearSigmaReductionProof<F, EF>,
            LinearSigmaOpeningClaim<EF>,
        ),
        LinearSigmaReductionError,
    >
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let product =
            ProductPolynomial::<F, EF>::new_unpacked(VariableOrder::Prefix, evals, batched.weights);
        let mut prover = SumcheckProver::new(product, batched.target);
        let mut sumcheck = SumcheckData::default();
        let point = prover.compute_sumcheck_polynomials(
            &mut sumcheck,
            challenger,
            self.num_variables,
            pow_bits,
            None,
        );
        debug_assert_eq!(prover.num_variables(), 0);

        let final_evals = prover.evals();
        let oracle_eval = final_evals.as_slice()[0];
        challenger.observe_algebra_element(oracle_eval);

        Ok((
            LinearSigmaReductionProof {
                sumcheck,
                oracle_eval,
            },
            LinearSigmaOpeningClaim {
                point,
                value: oracle_eval,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::point::Point;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type TestChallenger = DuplexChallenger<F, Perm, 16, 8>;

    fn challenger() -> TestChallenger {
        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
        DuplexChallenger::new(perm)
    }

    fn sample_poly() -> Poly<F> {
        Poly::new(vec![
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
        ])
    }

    #[test]
    fn equality_claim_is_linear_sigma_constraint() {
        let poly = sample_poly();
        let point = Point::new(vec![EF::from_u64(2), EF::from_u64(4)]);
        let value = poly.eval_base(&point);

        let mut eq = EqStatement::initialize(2);
        eq.add_evaluated_constraint(point.clone(), value);

        let constraint = LinearSigmaConstraint::from_eq_statement::<F>(&eq, EF::from_u64(9));
        let expected_weights = Poly::<EF>::new_from_point(point.as_slice(), EF::ONE);

        assert_eq!(constraint.weights, expected_weights);
        assert_eq!(constraint.target, value);
        assert!(constraint.verify_base(&poly));
    }

    #[test]
    fn batched_equality_claims_match_eq_statement_combination() {
        let poly = sample_poly();
        let point0 = Point::new(vec![EF::from_u64(2), EF::from_u64(4)]);
        let point1 = Point::new(vec![EF::from_u64(6), EF::from_u64(8)]);
        let value0 = poly.eval_base(&point0);
        let value1 = poly.eval_base(&point1);
        let gamma = EF::from_u64(13);

        let mut eq = EqStatement::initialize(2);
        eq.add_evaluated_constraint(point0, value0);
        eq.add_evaluated_constraint(point1, value1);

        let constraint = LinearSigmaConstraint::from_eq_statement::<F>(&eq, gamma);
        let mut expected_weights = Poly::zero(2);
        let mut expected_target = EF::ZERO;
        eq.combine_hypercube::<F, false>(&mut expected_weights, &mut expected_target, gamma);

        assert_eq!(constraint.weights, expected_weights);
        assert_eq!(constraint.target, expected_target);
        assert!(constraint.verify_base(&poly));
    }

    #[test]
    fn linear_sigma_rejects_wrong_target() {
        let poly = sample_poly();
        let point = Point::new(vec![EF::from_u64(2), EF::from_u64(4)]);
        let value = poly.eval_base(&point);

        let mut eq = EqStatement::initialize(2);
        eq.add_evaluated_constraint(point, value + EF::ONE);

        let constraint = LinearSigmaConstraint::from_eq_statement::<F>(&eq, EF::from_u64(9));
        assert!(!constraint.verify_base(&poly));
    }

    #[test]
    fn statement_batches_constraints_with_gamma_powers() {
        let poly = sample_poly();
        let point0 = Point::new(vec![EF::from_u64(2), EF::from_u64(4)]);
        let point1 = Point::new(vec![EF::from_u64(6), EF::from_u64(8)]);
        let value0 = poly.eval_base(&point0);
        let value1 = poly.eval_base(&point1);

        let mut eq0 = EqStatement::initialize(2);
        eq0.add_evaluated_constraint(point0, value0);
        let mut eq1 = EqStatement::initialize(2);
        eq1.add_evaluated_constraint(point1, value1);

        let c0 = LinearSigmaConstraint::from_eq_statement::<F>(&eq0, EF::from_u64(3));
        let c1 = LinearSigmaConstraint::from_eq_statement::<F>(&eq1, EF::from_u64(5));

        let mut statement = LinearSigmaStatement::initialize(2);
        statement.add_constraint(c0.clone());
        statement.add_constraint(c1.clone());

        let gamma = EF::from_u64(17);
        let batched = statement.batch(gamma).expect("non-empty statement");

        let mut expected = c0;
        expected.add_scaled(gamma, &c1);

        assert_eq!(batched, expected);
        assert!(statement.verify_base(&poly));
        assert!(batched.verify_base(&poly));
    }

    #[test]
    fn batched_multi_oracle_reduction_opens_one_virtual_claim() {
        let base_poly = sample_poly();
        let ext_poly = Poly::new(vec![
            EF::from_u64(17),
            EF::from_u64(19),
            EF::from_u64(23),
            EF::from_u64(29),
        ]);
        let base_point = Point::new(vec![EF::from_u64(3), EF::from_u64(5)]);
        let ext_point = Point::new(vec![EF::from_u64(7), EF::from_u64(11)]);

        let mut base_eq = EqStatement::initialize(2);
        base_eq.add_evaluated_constraint(base_point.clone(), base_poly.eval_base(&base_point));
        let mut base_statement = LinearSigmaStatement::initialize(2);
        base_statement.add_constraint(LinearSigmaConstraint::from_eq_statement::<F>(
            &base_eq,
            EF::ONE,
        ));

        let mut ext_eq = EqStatement::initialize(2);
        ext_eq.add_evaluated_constraint(ext_point.clone(), ext_poly.eval_ext::<F>(&ext_point));
        let mut ext_statement = LinearSigmaStatement::initialize(2);
        ext_statement.add_constraint(LinearSigmaConstraint::from_eq_statement::<F>(
            &ext_eq,
            EF::ONE,
        ));

        let oracles = vec![
            BatchedLinearSigmaProverOracle::base(&base_statement, &base_poly),
            BatchedLinearSigmaProverOracle::extension(&ext_statement, &ext_poly),
        ];
        let mut prover_challenger = challenger();
        let (proof, opening) =
            prove_batched_linear_sigma_reduction::<F, EF, _>(&oracles, &mut prover_challenger, 0)
                .expect("batched reduction proof");

        let statements = vec![&base_statement, &ext_statement];
        let mut verifier_challenger = challenger();
        let verified = verify_batched_linear_sigma_reduction::<F, EF, _>(
            &statements,
            &proof,
            &mut verifier_challenger,
            0,
        )
        .expect("batched reduction verification");

        assert_eq!(opening, verified);
        let expected = opening.coeffs[0] * base_poly.eval_base(&opening.point)
            + opening.coeffs[1] * ext_poly.eval_ext::<F>(&opening.point);
        assert_eq!(opening.value, expected);
    }

    #[test]
    fn batched_multi_oracle_reduction_rejects_tampered_virtual_eval() {
        let base_poly = sample_poly();
        let ext_poly = Poly::new(vec![
            EF::from_u64(31),
            EF::from_u64(37),
            EF::from_u64(41),
            EF::from_u64(43),
        ]);
        let point = Point::new(vec![EF::from_u64(3), EF::from_u64(5)]);

        let mut base_eq = EqStatement::initialize(2);
        base_eq.add_evaluated_constraint(point.clone(), base_poly.eval_base(&point));
        let mut base_statement = LinearSigmaStatement::initialize(2);
        base_statement.add_constraint(LinearSigmaConstraint::from_eq_statement::<F>(
            &base_eq,
            EF::ONE,
        ));

        let mut ext_eq = EqStatement::initialize(2);
        ext_eq.add_evaluated_constraint(point.clone(), ext_poly.eval_ext::<F>(&point));
        let mut ext_statement = LinearSigmaStatement::initialize(2);
        ext_statement.add_constraint(LinearSigmaConstraint::from_eq_statement::<F>(
            &ext_eq,
            EF::ONE,
        ));

        let oracles = vec![
            BatchedLinearSigmaProverOracle::base(&base_statement, &base_poly),
            BatchedLinearSigmaProverOracle::extension(&ext_statement, &ext_poly),
        ];
        let (mut proof, _) =
            prove_batched_linear_sigma_reduction::<F, EF, _>(&oracles, &mut challenger(), 0)
                .expect("batched reduction proof");
        proof.virtual_eval += EF::ONE;

        let statements = vec![&base_statement, &ext_statement];
        assert!(
            verify_batched_linear_sigma_reduction::<F, EF, _>(
                &statements,
                &proof,
                &mut challenger(),
                0,
            )
            .is_err()
        );
    }

    #[test]
    fn reduction_proves_and_verifies_base_oracle() {
        let poly = sample_poly();
        let point0 = Point::new(vec![EF::from_u64(2), EF::from_u64(4)]);
        let point1 = Point::new(vec![EF::from_u64(6), EF::from_u64(8)]);

        let mut statement = LinearSigmaStatement::initialize(2);
        for point in [point0, point1] {
            let mut eq = EqStatement::initialize(2);
            eq.add_evaluated_constraint(point.clone(), poly.eval_base(&point));
            statement.add_constraint(LinearSigmaConstraint::from_eq_statement::<F>(&eq, EF::ONE));
        }

        let mut prover_challenger = challenger();
        let mut verifier_challenger = challenger();
        let (proof, claim) = statement
            .prove_reduction_base::<F, _>(&poly, &mut prover_challenger, 0)
            .expect("honest linear Sigma reduction");

        let verified_claim = statement
            .verify_reduction::<F, _>(&proof, &mut verifier_challenger, 0)
            .expect("linear Sigma reduction verification");

        assert_eq!(claim, verified_claim);
        assert_eq!(claim.value, poly.eval_base(&claim.point));
    }

    #[test]
    fn reduction_proves_and_verifies_extension_oracle() {
        let poly = Poly::new(vec![
            EF::from_u64(3),
            EF::from_u64(5),
            EF::from_u64(7),
            EF::from_u64(11),
        ]);
        let point = Point::new(vec![EF::from_u64(9), EF::from_u64(10)]);
        let mut eq = EqStatement::initialize(2);
        eq.add_evaluated_constraint(
            point,
            poly.eval_ext::<F>(&Point::new(vec![EF::from_u64(9), EF::from_u64(10)])),
        );

        let mut statement = LinearSigmaStatement::initialize(2);
        statement.add_constraint(LinearSigmaConstraint::from_eq_statement::<F>(&eq, EF::ONE));

        let mut prover_challenger = challenger();
        let mut verifier_challenger = challenger();
        let (proof, claim) = statement
            .prove_reduction_ext::<F, _>(&poly, &mut prover_challenger, 0)
            .expect("honest extension linear Sigma reduction");
        let verified_claim = statement
            .verify_reduction::<F, _>(&proof, &mut verifier_challenger, 0)
            .expect("extension linear Sigma reduction verification");

        assert_eq!(claim, verified_claim);
        assert_eq!(claim.value, poly.eval_ext::<F>(&claim.point));
    }

    #[test]
    fn reduction_rejects_unsatisfied_oracle_at_prove_time() {
        let poly = sample_poly();
        let weights = Poly::new(vec![EF::ONE; 4]);
        let wrong_target = poly
            .as_slice()
            .iter()
            .map(|&value| EF::from(value))
            .sum::<EF>()
            + EF::ONE;
        let mut statement = LinearSigmaStatement::initialize(2);
        statement.add_constraint(LinearSigmaConstraint::new(weights, wrong_target));

        let err = statement
            .prove_reduction_base::<F, _>(&poly, &mut challenger(), 0)
            .expect_err("wrong target should not produce an honest proof");
        assert!(matches!(
            err,
            LinearSigmaReductionError::UnsatisfiedStatement
        ));
    }

    #[test]
    fn reduction_rejects_tampered_residual_opening() {
        let poly = sample_poly();
        let target = poly
            .as_slice()
            .iter()
            .map(|&value| EF::from(value))
            .sum::<EF>();
        let mut statement = LinearSigmaStatement::initialize(2);
        statement.add_constraint(LinearSigmaConstraint::new(
            Poly::new(vec![EF::ONE; 4]),
            target,
        ));

        let (mut proof, _) = statement
            .prove_reduction_base::<F, _>(&poly, &mut challenger(), 0)
            .expect("honest linear Sigma reduction");
        proof.oracle_eval += EF::ONE;

        let err = statement
            .verify_reduction::<F, _>(&proof, &mut challenger(), 0)
            .expect_err("tampered residual opening should fail");
        assert!(matches!(err, LinearSigmaReductionError::FinalCheckFailed));
    }
}
