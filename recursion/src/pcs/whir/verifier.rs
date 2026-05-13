//! Recursive verifier helpers for WHIR multilinear proofs.
//!
//! The full WHIR verifier is built from three pieces:
//! 1. Fiat-Shamir transcript replay,
//! 2. sumcheck verification for the linearized constraints,
//! 3. STIR/MMCS query verification for Reed-Solomon proximity.
//!
//! This file starts with the reusable sumcheck part. It is deliberately small
//! and formula-level: the circuit computes the same degree-two extrapolation as
//! native WHIR and updates the claimed sum after every Fiat-Shamir challenge.

use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::{format, vec};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_circuit::ops::Poseidon2Config;
use p3_circuit::{CircuitBuilder, CircuitBuilderError, NonPrimitiveOpId};
use p3_commit::Mmcs;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::log2_strict_usize;
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{RoundConfig, WhirConfig};

use super::{WhirProofVerificationTargets, WhirQueryOpeningTargets, WhirSumcheckDataTargets};
use crate::Target;
use crate::pcs::{verify_batch_circuit, verify_batch_circuit_from_extension_opened};
use crate::traits::{RecursiveChallenger, RecursiveMmcs};
use crate::verifier::{ObservableCommitment, VerificationError};

/// Native WHIR caps each transcript bit squeeze at this many bits.
///
/// Keep this equal to `p3_whir::pcs::utils::MAX_SAMPLE_BITS`; that constant is
/// private in the native crate, so the recursive verifier mirrors it here.
const WHIR_MAX_SAMPLE_BITS: usize = 20;

/// Commitment data parsed from a WHIR proof inside the circuit.
///
/// Native WHIR parses a commitment by observing the MMCS root, sampling OOD
/// points from Fiat-Shamir, expanding each sampled univariate point into a
/// multilinear point, and absorbing the claimed OOD answer. The recursive
/// verifier keeps the same data in target form so the later constraint
/// linearization can combine it with the public statement.
#[derive(Clone)]
pub struct WhirParsedCommitmentTargets<Comm> {
    /// Commitment root targets.
    pub root: Comm,
    /// OOD points expanded as `[y^(2^(n-1)), ..., y^2, y]`.
    pub ood_points: Vec<Vec<Target>>,
    /// OOD answers sent by the prover.
    pub ood_answers: Vec<Target>,
}

/// Circuit-side equality statement for WHIR's multilinear opening claims.
///
/// Native WHIR batches constraints `f(z_i) = v_i` with powers of a transcript
/// challenge `gamma`. This target form stores the same points and values so the
/// recursive verifier can update the claimed sum and later evaluate the batched
/// weight polynomial at the final folding point.
#[derive(Clone, Debug)]
pub struct WhirEqStatementTargets {
    /// Number of variables in the committed multilinear polynomial.
    pub num_variables: usize,
    /// Multilinear opening points `z_i`.
    pub points: Vec<Vec<Target>>,
    /// Claimed evaluations `v_i`.
    pub evaluations: Vec<Target>,
}

impl WhirEqStatementTargets {
    /// Construct an empty equality statement with fixed arity.
    pub const fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            points: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Append one evaluated equality claim.
    pub fn add_evaluated_constraint(&mut self, point: Vec<Target>, evaluation: Target) {
        assert_eq!(
            point.len(),
            self.num_variables,
            "WHIR eq point arity mismatch"
        );
        self.points.push(point);
        self.evaluations.push(evaluation);
    }

    /// Concatenate another equality statement of the same arity.
    pub fn concatenate(&mut self, other: &Self) {
        assert_eq!(
            self.num_variables, other.num_variables,
            "WHIR eq statement arity mismatch"
        );
        self.points.extend_from_slice(&other.points);
        self.evaluations.extend_from_slice(&other.evaluations);
    }

    /// Number of equality constraints.
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.points.len(), self.evaluations.len());
        self.points.len()
    }

    /// Add `sum_i gamma^i * v_i` to the running claimed evaluation.
    pub fn combine_evals<EF>(
        &self,
        circuit: &mut CircuitBuilder<EF>,
        claimed_eval: &mut Target,
        gamma: Target,
    ) where
        EF: p3_field::Field,
    {
        let mut power = circuit.define_const(EF::ONE);
        for &evaluation in &self.evaluations {
            *claimed_eval = circuit.mul_add(power, evaluation, *claimed_eval);
            power = circuit.mul(power, gamma);
        }
    }
}

/// Circuit-side select statement for WHIR's constrained-RS query claims.
///
/// Select claims use the WHIR polynomial
/// `select(pow(z), x) = prod_i (x_i * (z^{2^i} - 1) + 1)` over Boolean
/// hypercube points. They are batched after the equality claims with the same
/// challenge powers, matching native `SelectStatement::combine_evals`.
#[derive(Clone, Debug)]
pub struct WhirSelectStatementTargets {
    /// Number of multilinear variables.
    pub num_variables: usize,
    /// Univariate select variables `z_j`.
    pub vars: Vec<Target>,
    /// Claimed evaluations at the selected positions.
    pub evaluations: Vec<Target>,
}

impl WhirSelectStatementTargets {
    /// Construct an empty select statement with fixed arity.
    pub const fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            vars: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Construct a select statement from variables and values.
    pub fn new(num_variables: usize, vars: Vec<Target>, evaluations: Vec<Target>) -> Self {
        assert_eq!(
            vars.len(),
            evaluations.len(),
            "WHIR select statement length mismatch"
        );
        Self {
            num_variables,
            vars,
            evaluations,
        }
    }

    /// Add `sum_j gamma^(shift+j) * s_j` to the running claimed evaluation.
    pub fn combine_evals<EF>(
        &self,
        circuit: &mut CircuitBuilder<EF>,
        claimed_eval: &mut Target,
        gamma: Target,
        shift: usize,
    ) where
        EF: p3_field::Field,
    {
        let mut power = circuit.define_const(EF::ONE);
        for _ in 0..shift {
            power = circuit.mul(power, gamma);
        }
        for &evaluation in &self.evaluations {
            *claimed_eval = circuit.mul_add(power, evaluation, *claimed_eval);
            power = circuit.mul(power, gamma);
        }
    }
}

/// Circuit-side WHIR constraint batch.
///
/// This mirrors native `p3_whir::constraints::Constraint`: equality claims use
/// powers `gamma^0..`, select claims continue at `gamma^n_eq..`, and both
/// contribute to one claimed sum checked by WHIR's quadratic sumcheck.
#[derive(Clone, Debug)]
pub struct WhirConstraintTargets {
    /// Batching challenge sampled after OOD binding.
    pub challenge: Target,
    /// Equality-style multilinear opening constraints.
    pub eq_statement: WhirEqStatementTargets,
    /// Select-style constrained-RS query constraints.
    pub sel_statement: WhirSelectStatementTargets,
}

impl WhirConstraintTargets {
    /// Construct a WHIR constraint batch.
    pub fn new(
        challenge: Target,
        eq_statement: WhirEqStatementTargets,
        sel_statement: WhirSelectStatementTargets,
    ) -> Self {
        assert_eq!(
            eq_statement.num_variables, sel_statement.num_variables,
            "WHIR constraint arity mismatch"
        );
        Self {
            challenge,
            eq_statement,
            sel_statement,
        }
    }

    /// Number of variables in the constraint polynomial.
    pub const fn num_variables(&self) -> usize {
        self.eq_statement.num_variables
    }

    /// Add this constraint's expected evaluations to `claimed_eval`.
    pub fn combine_evals<EF>(&self, circuit: &mut CircuitBuilder<EF>, claimed_eval: &mut Target)
    where
        EF: p3_field::Field,
    {
        self.eq_statement
            .combine_evals(circuit, claimed_eval, self.challenge);
        self.sel_statement.combine_evals(
            circuit,
            claimed_eval,
            self.challenge,
            self.eq_statement.len(),
        );
    }
}

/// Expand a univariate OOD challenge into the multilinear point used by WHIR.
///
/// This matches `Point::expand_from_univariate`: for `n` variables and sampled
/// `y`, WHIR checks the multilinear polynomial at
/// `[y^(2^(n-1)), ..., y^2, y]`. The big-endian convention is relevant for the
/// final equality-polynomial checks, so the circuit fixes it here.
pub fn expand_from_univariate_circuit<EF>(
    circuit: &mut CircuitBuilder<EF>,
    point: Target,
    num_variables: usize,
) -> Vec<Target>
where
    EF: p3_field::Field,
{
    let zero = circuit.define_const(EF::ZERO);
    let mut expanded = vec![zero; num_variables];
    let mut cur = point;
    for i in (0..num_variables).rev() {
        expanded[i] = cur;
        cur = circuit.mul(cur, cur);
    }
    expanded
}

/// Observe a WHIR commitment and its OOD answers in Fiat-Shamir order.
///
/// This is the recursive analogue of native
/// `ParsedCommitment::parse_with_round`, parameterized over the recursive
/// commitment target type. It does not yet build the combined WHIR constraint;
/// it returns the parsed OOD points/answers so the caller can do that with the
/// same linearization as the native verifier.
pub fn observe_whir_commitment_and_ood_circuit<BF, EF, C, Comm>(
    circuit: &mut CircuitBuilder<EF>,
    challenger: &mut C,
    root: &Comm,
    ood_answers: &[Target],
    num_variables: usize,
    ood_samples: usize,
) -> Result<WhirParsedCommitmentTargets<Comm>, VerificationError>
where
    BF: PrimeField64,
    EF: ExtensionField<BF>,
    C: RecursiveChallenger<BF, EF>,
    Comm: ObservableCommitment + Clone,
{
    if ood_answers.len() != ood_samples {
        return Err(VerificationError::InvalidProofShape(format!(
            "WHIR OOD answer count mismatch: expected {}, got {}",
            ood_samples,
            ood_answers.len()
        )));
    }

    challenger.observe_slice(circuit, &root.to_observation_targets());

    let mut ood_points = Vec::with_capacity(ood_samples);
    let mut parsed_answers = Vec::with_capacity(ood_samples);
    for &answer in ood_answers {
        let point = challenger.sample_ext(circuit);
        ood_points.push(expand_from_univariate_circuit(
            circuit,
            point,
            num_variables,
        ));
        challenger.observe_ext(circuit, answer);
        parsed_answers.push(answer);
    }

    Ok(WhirParsedCommitmentTargets {
        root: root.clone(),
        ood_points,
        ood_answers: parsed_answers,
    })
}

/// Evaluate the WHIR sumcheck round polynomial at the Fiat-Shamir challenge.
///
/// Native WHIR sends `h(0)` and `h(infinity)`; the verifier derives
/// `h(1) = claimed_sum - h(0)` from the sumcheck constraint and evaluates
///
/// ```text
/// h(r) = h(0) * (1 - r) + h(1) * r + h(infinity) * r * (r - 1).
/// ```
///
/// This is the same formula as `p3_whir::sumcheck::extrapolate_01inf`, but
/// expressed over circuit targets.
pub fn extrapolate_01inf_circuit<EF>(
    circuit: &mut CircuitBuilder<EF>,
    h0: Target,
    h1: Target,
    h_inf: Target,
    r: Target,
) -> Target
where
    EF: p3_field::Field,
{
    let one = circuit.define_const(EF::ONE);
    let one_minus_r = circuit.sub(one, r);
    let h0_term = circuit.mul(h0, one_minus_r);
    let h1_term = circuit.mul(h1, r);
    let r_minus_one = circuit.sub(r, one);
    let r_times_r_minus_one = circuit.mul(r, r_minus_one);
    let h_inf_term = circuit.mul(h_inf, r_times_r_minus_one);
    let finite_terms = circuit.add(h0_term, h1_term);
    circuit.add(finite_terms, h_inf_term)
}

/// Evaluate `eq(point, challenge)` over circuit targets.
///
/// This uses the same identity as WHIR's native equality statement:
///
/// ```text
/// eq(a, b) = prod_i (a_i b_i + (1 - a_i)(1 - b_i))
///          = prod_i (1 + 2 a_i b_i - a_i - b_i).
/// ```
pub fn eval_eq_poly_circuit<EF>(
    circuit: &mut CircuitBuilder<EF>,
    point: &[Target],
    challenge: &[Target],
) -> Target
where
    EF: p3_field::Field,
{
    assert_eq!(point.len(), challenge.len(), "WHIR eq arity mismatch");
    let one = circuit.define_const(EF::ONE);
    let two = circuit.define_const(EF::TWO);
    let factors = point
        .iter()
        .zip(challenge)
        .map(|(&lhs, &rhs)| {
            let lhs_rhs = circuit.mul(lhs, rhs);
            let twice_lhs_rhs = circuit.mul(two, lhs_rhs);
            let without_lhs = circuit.sub(twice_lhs_rhs, lhs);
            let without_rhs = circuit.sub(without_lhs, rhs);
            circuit.add(one, without_rhs)
        })
        .collect::<Vec<_>>();
    circuit.mul_many(&factors)
}

/// Evaluate WHIR's select polynomial for a univariate point `var`.
///
/// Native WHIR expands `var` as `(var, var^2, var^4, ...)`; with the
/// big-endian point convention this is evaluated by walking the local challenge
/// in reverse order.
pub fn eval_select_poly_circuit<EF>(
    circuit: &mut CircuitBuilder<EF>,
    local_challenge: &[Target],
    mut var: Target,
) -> Target
where
    EF: p3_field::Field,
{
    let one = circuit.define_const(EF::ONE);
    let mut factors = Vec::with_capacity(local_challenge.len());
    for &r in local_challenge.iter().rev() {
        let var_minus_one = circuit.sub(var, one);
        factors.push(circuit.mul_add(r, var_minus_one, one));
        var = circuit.mul(var, var);
    }
    circuit.mul_many(&factors)
}

/// Evaluate native WHIR's prefix-order batched constraint polynomial.
///
/// WHIR folds variables in prefix order, so each constraint of arity `k`
/// sees the last `k` variables of the full folding randomness in their original
/// order. This mirrors `VariableOrder::Prefix.eval_constraints_poly`.
pub fn eval_prefix_constraints_poly_circuit<EF>(
    circuit: &mut CircuitBuilder<EF>,
    constraints: &[WhirConstraintTargets],
    folding_randomness: &[Target],
) -> Target
where
    EF: p3_field::Field,
{
    let mut total = circuit.define_const(EF::ZERO);

    for constraint in constraints {
        let k = constraint.num_variables();
        assert!(
            k <= folding_randomness.len(),
            "WHIR folding randomness shorter than constraint arity"
        );
        let local_challenge = &folding_randomness[folding_randomness.len() - k..];

        let mut power = circuit.define_const(EF::ONE);
        for point in &constraint.eq_statement.points {
            let eq = eval_eq_poly_circuit(circuit, point, local_challenge);
            total = circuit.mul_add(power, eq, total);
            power = circuit.mul(power, constraint.challenge);
        }
        for &var in &constraint.sel_statement.vars {
            let select = eval_select_poly_circuit(circuit, local_challenge, var);
            total = circuit.mul_add(power, select, total);
            power = circuit.mul(power, constraint.challenge);
        }
    }

    total
}

/// Replay WHIR sumcheck rounds in-circuit and return their sampled challenges.
///
/// The caller owns the current `claimed_sum` target. This function absorbs each
/// sent pair `(h_i(0), h_i(infinity))`, checks the optional PoW witness, samples
/// `r_i`, and replaces `claimed_sum` with `h_i(r_i)`.
pub fn verify_whir_sumcheck_rounds_circuit<BF, EF, C>(
    circuit: &mut CircuitBuilder<EF>,
    challenger: &mut C,
    sumcheck: &WhirSumcheckDataTargets<BF, EF>,
    claimed_sum: &mut Target,
    pow_bits: usize,
) -> Result<Vec<Target>, CircuitBuilderError>
where
    BF: PrimeField64,
    EF: ExtensionField<BF>,
    C: RecursiveChallenger<BF, EF>,
{
    if pow_bits > 0 {
        assert_eq!(
            sumcheck.pow_witnesses.len(),
            sumcheck.polynomial_evaluations.len(),
            "WHIR sumcheck PoW witness count must match the number of rounds"
        );
    }

    let mut randomness = Vec::with_capacity(sumcheck.polynomial_evaluations.len());
    for (round, &[h0, h_inf]) in sumcheck.polynomial_evaluations.iter().enumerate() {
        challenger.observe_ext_slice(circuit, &[h0, h_inf]);

        if pow_bits > 0 {
            challenger.check_pow_witness(circuit, pow_bits, sumcheck.pow_witnesses[round])?;
        }

        let r = challenger.sample_ext(circuit);
        let h1 = circuit.sub(*claimed_sum, h0);
        *claimed_sum = extrapolate_01inf_circuit(circuit, h0, h1, h_inf, r);
        randomness.push(r);
    }

    Ok(randomness)
}

/// Replay the optional final WHIR sumcheck.
///
/// Native WHIR omits this object when the number of direct final rounds is zero.
/// The recursive verifier follows the same convention.
pub fn verify_whir_final_sumcheck_rounds_circuit<BF, EF, C>(
    circuit: &mut CircuitBuilder<EF>,
    challenger: &mut C,
    final_sumcheck: Option<&WhirSumcheckDataTargets<BF, EF>>,
    claimed_sum: &mut Target,
    rounds: usize,
    pow_bits: usize,
) -> Result<Vec<Target>, CircuitBuilderError>
where
    BF: PrimeField64,
    EF: ExtensionField<BF>,
    C: RecursiveChallenger<BF, EF>,
{
    if rounds == 0 {
        return Ok(vec![]);
    }

    let sumcheck = final_sumcheck.expect("WHIR final sumcheck is required for nonzero rounds");
    assert_eq!(
        sumcheck.polynomial_evaluations.len(),
        rounds,
        "WHIR final sumcheck round count must match verifier parameters"
    );
    verify_whir_sumcheck_rounds_circuit(circuit, challenger, sumcheck, claimed_sum, pow_bits)
}

/// Absorb the native WHIR domain separator through the recursive challenger.
///
/// Native `WhirPcs::verify` first builds `DomainSeparator` from the derived
/// `WhirConfig`, then absorbs the encoded field elements before parsing any
/// commitment. The recursive verifier does the same operation with constants
/// in the circuit, so all later OOD, combination, sumcheck, PoW, and STIR
/// challenges are bound to the exact native WHIR protocol shape.
pub fn observe_whir_domain_separator_circuit<
    BF,
    EF,
    C,
    MT,
    NativeChallenger,
    const DIGEST_ELEMS: usize,
>(
    circuit: &mut CircuitBuilder<EF>,
    challenger: &mut C,
    config: &WhirConfig<EF, BF, MT, NativeChallenger>,
) where
    BF: TwoAdicField + PrimeField64,
    EF: ExtensionField<BF> + TwoAdicField,
    C: RecursiveChallenger<BF, EF>,
    MT: Mmcs<BF>,
    NativeChallenger: FieldChallenger<BF> + GrindingChallenger<Witness = BF>,
{
    let mut ds = DomainSeparator::new(vec![]);
    ds.commit_statement::<MT, NativeChallenger, DIGEST_ELEMS>(config);
    ds.add_whir_proof::<MT, NativeChallenger, DIGEST_ELEMS>(config);

    for &element in ds.pattern() {
        let element_target = circuit.define_const(EF::from(element));
        challenger.observe(circuit, element_target);
    }
}

/// Pack lifted base-field targets into extension-field rate elements.
///
/// The recursive MMCS verifier consumes Merkle caps in the same row format as
/// the native Poseidon2 MMCS. When the permutation hashes packed extension
/// limbs, adjacent lifted base elements must be recomposed as one extension
/// target; when the permutation is the D=1 base-field hash, the lifted values
/// remain one-per-rate-slot.
fn pack_lifted_to_ext<BF, EF>(builder: &mut CircuitBuilder<EF>, lifted: &[Target]) -> Vec<Target>
where
    BF: Field,
    EF: ExtensionField<BF> + BasedVectorSpace<BF>,
{
    if lifted.is_empty() {
        return Vec::new();
    }

    let d = EF::DIMENSION;
    let basis_consts = (0..d)
        .map(|i| {
            let mut coeffs = vec![BF::ZERO; d];
            coeffs[i] = BF::ONE;
            builder.define_const(
                EF::from_basis_coefficients_slice(&coeffs).expect("valid extension basis"),
            )
        })
        .collect::<Vec<_>>();
    let zero = builder.define_const(EF::ZERO);

    lifted
        .chunks(d)
        .map(|chunk| {
            let mut packed = chunk[0];
            for j in 1..d {
                let val = if j < chunk.len() { chunk[j] } else { zero };
                packed = builder.mul_add(val, basis_consts[j], packed);
            }
            packed
        })
        .collect()
}

/// Convert observable commitment targets into MMCS cap rows.
fn commitment_cap_rows_from_lifted<BF, EF>(
    builder: &mut CircuitBuilder<EF>,
    perm_config: Poseidon2Config,
    lifted: &[Target],
) -> Vec<Vec<Target>>
where
    BF: Field,
    EF: ExtensionField<BF> + BasedVectorSpace<BF>,
{
    let rate_ext = perm_config.rate_ext();
    if perm_config.d() == 1 && EF::DIMENSION > 1 {
        lifted.chunks(rate_ext).map(|c| c.to_vec()).collect()
    } else {
        let packed = pack_lifted_to_ext::<BF, EF>(builder, lifted);
        packed.chunks(rate_ext).map(|c| c.to_vec()).collect()
    }
}

fn assert_same_observable_commitment<EF, Comm>(
    circuit: &mut CircuitBuilder<EF>,
    lhs: &Comm,
    rhs: &Comm,
) -> Result<(), VerificationError>
where
    EF: Field,
    Comm: ObservableCommitment,
{
    let lhs = lhs.to_observation_targets();
    let rhs = rhs.to_observation_targets();
    if lhs.len() != rhs.len() {
        return Err(VerificationError::InvalidProofShape(format!(
            "WHIR commitment target length mismatch: {} vs {}",
            lhs.len(),
            rhs.len()
        )));
    }
    for (a, b) in lhs.into_iter().zip(rhs) {
        circuit.connect(a, b);
    }
    Ok(())
}

fn ood_eq_statement_from_parsed<Comm>(
    parsed: &WhirParsedCommitmentTargets<Comm>,
    num_variables: usize,
) -> WhirEqStatementTargets {
    let mut statement = WhirEqStatementTargets::initialize(num_variables);
    for (point, &answer) in parsed.ood_points.iter().zip(&parsed.ood_answers) {
        statement.add_evaluated_constraint(point.clone(), answer);
    }
    statement
}

/// Evaluate a multilinear table at `point` using WHIR's lexicographic order.
///
/// Native `Poly::eval_ext` interpolates along the last variable first, then
/// moves toward the prefix variables. This in-circuit version follows the same
/// adjacent-pair fold order.
pub fn eval_multilinear_poly_circuit<EF>(
    circuit: &mut CircuitBuilder<EF>,
    evals: &[Target],
    point: &[Target],
) -> Target
where
    EF: Field,
{
    assert_eq!(evals.len(), 1 << point.len(), "WHIR MLE arity mismatch");
    if evals.len() == 1 {
        return evals[0];
    }

    let mut layer = evals.to_vec();
    for &r in point.iter().rev() {
        let mut next = Vec::with_capacity(layer.len() / 2);
        for pair in layer.chunks_exact(2) {
            let diff = circuit.sub(pair[1], pair[0]);
            next.push(circuit.mul_add(r, diff, pair[0]));
        }
        layer = next;
    }
    debug_assert_eq!(layer.len(), 1);
    layer[0]
}

fn touch_private_targets<EF>(circuit: &mut CircuitBuilder<EF>, targets: &[Target])
where
    EF: Field,
{
    for &target in targets {
        // Allocated public/private witness targets must be consumed by some
        // constrained expression, otherwise the circuit-prover witness bus sees
        // a creator row with no matching reader. `target - target = 0` is a
        // semantic no-op but forces the lowerer to emit balanced ALU reads of
        // the target without creating a dangling output witness.
        let zero = circuit.sub(target, target);
        circuit.assert_zero(zero);
    }
}

fn touch_sumcheck_targets<BF, EF>(
    circuit: &mut CircuitBuilder<EF>,
    sumcheck: &WhirSumcheckDataTargets<BF, EF>,
) where
    BF: Field,
    EF: ExtensionField<BF>,
{
    for evals in &sumcheck.polynomial_evaluations {
        touch_private_targets(circuit, evals);
    }
    touch_private_targets(circuit, &sumcheck.pow_witnesses);
}

fn touch_query_targets<BF, EF, RecMmcs>(
    circuit: &mut CircuitBuilder<EF>,
    query: &WhirQueryOpeningTargets<BF, EF, RecMmcs>,
) where
    BF: Field,
    EF: ExtensionField<BF>,
    RecMmcs: RecursiveMmcs<BF, EF>,
{
    match query {
        WhirQueryOpeningTargets::Base { values, .. }
        | WhirQueryOpeningTargets::Extension { values, .. } => {
            touch_private_targets(circuit, values);
        }
        WhirQueryOpeningTargets::SharedBase { values, .. } => {
            for row in values {
                touch_private_targets(circuit, row);
            }
        }
        WhirQueryOpeningTargets::SharedExtension { values, .. } => {
            for row in values {
                touch_private_targets(circuit, row);
            }
        }
        WhirQueryOpeningTargets::Batched { openings } => {
            for opening in openings {
                touch_query_targets::<BF, EF, RecMmcs>(circuit, opening);
            }
        }
    }
}

fn bitvec_less_than_circuit<EF>(
    circuit: &mut CircuitBuilder<EF>,
    lhs_le: &[Target],
    rhs_le: &[Target],
) -> Target
where
    EF: Field,
{
    debug_assert_eq!(lhs_le.len(), rhs_le.len());
    let one = circuit.define_const(EF::ONE);
    let two = circuit.define_const(EF::TWO);
    let mut eq_prefix = one;
    let mut less = circuit.define_const(EF::ZERO);

    for (&a, &b) in lhs_le.iter().zip(rhs_le).rev() {
        let one_minus_a = circuit.sub(one, a);
        let a0_b1 = circuit.mul(one_minus_a, b);
        less = circuit.mul_add(eq_prefix, a0_b1, less);

        let ab = circuit.mul(a, b);
        let twice_ab = circuit.mul(two, ab);
        let without_a = circuit.sub(twice_ab, a);
        let without_b = circuit.sub(without_a, b);
        let eq_bit = circuit.add(one, without_b);
        eq_prefix = circuit.mul(eq_prefix, eq_bit);
    }

    less
}

fn sort_and_reject_duplicate_index_bits<EF>(
    circuit: &mut CircuitBuilder<EF>,
    mut indices: Vec<Vec<Target>>,
) -> Vec<Vec<Target>>
where
    EF: Field,
{
    if indices.len() <= 1 {
        return indices;
    }

    for end in (1..indices.len()).rev() {
        for i in 0..end {
            let left = indices[i].clone();
            let right = indices[i + 1].clone();
            let left_lt_right = bitvec_less_than_circuit(circuit, &left, &right);
            let mut min_bits = Vec::with_capacity(left.len());
            let mut max_bits = Vec::with_capacity(left.len());
            for (&a, &b) in left.iter().zip(&right) {
                let a_minus_b = circuit.sub(a, b);
                let b_minus_a = circuit.sub(b, a);
                min_bits.push(circuit.mul_add(left_lt_right, a_minus_b, b));
                max_bits.push(circuit.mul_add(left_lt_right, b_minus_a, a));
            }
            indices[i] = min_bits;
            indices[i + 1] = max_bits;
        }
    }

    let one = circuit.define_const(EF::ONE);
    for pair in indices.windows(2) {
        let strictly_less = bitvec_less_than_circuit(circuit, &pair[0], &pair[1]);
        let not_less = circuit.sub(one, strictly_less);
        circuit.assert_zero(not_less);
    }

    indices
}

fn assert_boolean_bits<EF>(circuit: &mut CircuitBuilder<EF>, bits: &[Target])
where
    EF: Field,
{
    let one = circuit.define_const(EF::ONE);
    for &bit in bits {
        let bit_minus_one = circuit.sub(bit, one);
        let product = circuit.mul(bit, bit_minus_one);
        circuit.assert_zero(product);
    }
}

fn bitvec_equal_indicator<EF>(
    circuit: &mut CircuitBuilder<EF>,
    lhs_le: &[Target],
    rhs_le: &[Target],
) -> Target
where
    EF: Field,
{
    debug_assert_eq!(lhs_le.len(), rhs_le.len());
    let one = circuit.define_const(EF::ONE);
    let mut eq = one;
    for (&a, &b) in lhs_le.iter().zip(rhs_le) {
        let diff = circuit.sub(a, b);
        let diff_sq = circuit.mul(diff, diff);
        let same_bit = circuit.sub(one, diff_sq);
        eq = circuit.mul(eq, same_bit);
    }
    eq
}

fn constrain_deduped_query_indices<EF>(
    circuit: &mut CircuitBuilder<EF>,
    sampled_bits: &[Vec<Target>],
    deduped_bits: &[Vec<Target>],
) -> Result<(), VerificationError>
where
    EF: Field,
{
    if deduped_bits.len() > sampled_bits.len() {
        return Err(VerificationError::InvalidProofShape(format!(
            "WHIR deduplicated query count exceeds sampled count: sampled {}, deduped {}",
            sampled_bits.len(),
            deduped_bits.len()
        )));
    }
    let bit_len = sampled_bits.first().map_or(0, Vec::len);
    for bits in deduped_bits {
        if bits.len() != bit_len {
            return Err(VerificationError::InvalidProofShape(
                "WHIR query index bit length mismatch".to_string(),
            ));
        }
        assert_boolean_bits(circuit, bits);
    }
    let sorted = sort_and_reject_duplicate_index_bits(circuit, deduped_bits.to_vec());
    for (original, sorted) in deduped_bits.iter().zip(sorted.iter()) {
        for (&a, &b) in original.iter().zip(sorted) {
            let diff = circuit.sub(a, b);
            circuit.assert_zero(diff);
        }
    }

    let one = circuit.define_const(EF::ONE);
    for sampled in sampled_bits {
        let mut multiplicity = circuit.define_const(EF::ZERO);
        for deduped in deduped_bits {
            let eq = bitvec_equal_indicator(circuit, sampled, deduped);
            multiplicity = circuit.add(multiplicity, eq);
        }
        let diff = circuit.sub(multiplicity, one);
        circuit.assert_zero(diff);
    }

    Ok(())
}

fn sample_stir_query_index_bits_circuit<BF, EF, C>(
    circuit: &mut CircuitBuilder<EF>,
    challenger: &mut C,
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
) -> Result<Vec<Vec<Target>>, CircuitBuilderError>
where
    BF: PrimeField64,
    EF: ExtensionField<BF>,
    C: RecursiveChallenger<BF, EF>,
{
    let folded_domain_size = domain_size >> folding_factor;
    let domain_size_bits = log2_strict_usize(folded_domain_size);
    let max_bits_per_call = (BF::bits() - 1).min(WHIR_MAX_SAMPLE_BITS);
    let total_bits_needed = num_queries * domain_size_bits;
    let mut queries = Vec::with_capacity(num_queries);

    if total_bits_needed <= max_bits_per_call {
        let all_bits = challenger.sample_bits(circuit, total_bits_needed)?;
        for query in 0..num_queries {
            let start = query * domain_size_bits;
            queries.push(all_bits[start..start + domain_size_bits].to_vec());
        }
    } else {
        let queries_per_batch = max_bits_per_call / domain_size_bits;
        if queries_per_batch >= 2 {
            let mut remaining = num_queries;
            while remaining > 0 {
                let batch_size = remaining.min(queries_per_batch);
                let batch_bits = batch_size * domain_size_bits;
                let all_bits = challenger.sample_bits(circuit, batch_bits)?;
                for query in 0..batch_size {
                    let start = query * domain_size_bits;
                    queries.push(all_bits[start..start + domain_size_bits].to_vec());
                }
                remaining -= batch_size;
            }
        } else {
            for _ in 0..num_queries {
                queries.push(challenger.sample_bits(circuit, domain_size_bits)?);
            }
        }
    }

    Ok(queries)
}

fn folded_domain_var_from_index_bits<BF, EF>(
    circuit: &mut CircuitBuilder<EF>,
    folded_domain_gen: BF,
    index_bits_le: &[Target],
) -> Target
where
    BF: Field,
    EF: ExtensionField<BF>,
{
    let one = circuit.define_const(EF::ONE);
    let mut var = one;
    for (i, &bit) in index_bits_le.iter().enumerate() {
        let power = folded_domain_gen.exp_u64(1_u64 << i);
        let factor_if_one = circuit.define_const(EF::from(power));
        let factor_minus_one = circuit.sub(factor_if_one, one);
        let factor = circuit.mul_add(bit, factor_minus_one, one);
        var = circuit.mul(var, factor);
    }
    var
}

fn verify_single_root_query_opening_circuit<BF, EF, RecMmcs, Comm>(
    circuit: &mut CircuitBuilder<EF>,
    perm_config: Poseidon2Config,
    root: &Comm,
    dimensions: &[Dimensions],
    index_bits: &[Target],
    query: &WhirQueryOpeningTargets<BF, EF, RecMmcs>,
) -> Result<(Vec<Target>, Vec<NonPrimitiveOpId>), VerificationError>
where
    BF: TwoAdicField + PrimeField64,
    EF: ExtensionField<BF> + BasedVectorSpace<BF>,
    RecMmcs: RecursiveMmcs<BF, EF>,
    Comm: ObservableCommitment,
{
    let commitment_cap = commitment_cap_rows_from_lifted::<BF, EF>(
        circuit,
        perm_config,
        &root.to_observation_targets(),
    );

    match query {
        WhirQueryOpeningTargets::Base { values, .. } => {
            touch_private_targets(circuit, values);
            let op_ids = verify_batch_circuit::<BF, EF>(
                circuit,
                perm_config,
                &commitment_cap,
                dimensions,
                index_bits,
                core::slice::from_ref(values),
            )?;
            Ok((values.clone(), op_ids))
        }
        WhirQueryOpeningTargets::Extension { values, .. } => {
            touch_private_targets(circuit, values);
            let op_ids = verify_batch_circuit_from_extension_opened::<BF, EF>(
                circuit,
                perm_config,
                &commitment_cap,
                dimensions,
                index_bits,
                core::slice::from_ref(values),
            )?;
            Ok((values.clone(), op_ids))
        }
        WhirQueryOpeningTargets::SharedBase { .. }
        | WhirQueryOpeningTargets::SharedExtension { .. }
        | WhirQueryOpeningTargets::Batched { .. } => Err(VerificationError::InvalidProofShape(
            "single-root WHIR verifier received a batched/shared query opening".to_string(),
        )),
    }
}

fn verify_whir_stir_challenges_circuit<BF, EF, C, RecMmcs, Comm>(
    circuit: &mut CircuitBuilder<EF>,
    challenger: &mut C,
    perm_config: Poseidon2Config,
    params: &RoundConfig<BF>,
    root: &Comm,
    queries: &[WhirQueryOpeningTargets<BF, EF, RecMmcs>],
    query_index_bits: &[Vec<Target>],
    pow_witness: Target,
    folding_randomness: &[Target],
    is_intermediate_round: bool,
) -> Result<(WhirSelectStatementTargets, Vec<NonPrimitiveOpId>), VerificationError>
where
    BF: TwoAdicField + PrimeField64,
    EF: ExtensionField<BF> + BasedVectorSpace<BF>,
    C: RecursiveChallenger<BF, EF>,
    RecMmcs: RecursiveMmcs<BF, EF>,
    Comm: ObservableCommitment,
{
    if params.pow_bits > 0 {
        challenger.check_pow_witness(circuit, params.pow_bits, pow_witness)?;
    }

    if is_intermediate_round {
        let _ = challenger.sample(circuit);
    }

    if queries.len() != query_index_bits.len() {
        return Err(VerificationError::InvalidProofShape(format!(
            "WHIR STIR query/opening count mismatch: indices {}, openings {}",
            query_index_bits.len(),
            queries.len(),
        )));
    }

    let sampled_bits = sample_stir_query_index_bits_circuit::<BF, EF, C>(
        circuit,
        challenger,
        params.domain_size,
        params.folding_factor,
        params.num_queries,
    )?;
    constrain_deduped_query_indices(circuit, &sampled_bits, query_index_bits)?;
    let dimensions = vec![Dimensions {
        height: params.domain_size >> params.folding_factor,
        width: 1 << params.folding_factor,
    }];

    let mut vars = Vec::with_capacity(queries.len());
    let mut folds = Vec::with_capacity(queries.len());
    let mut op_ids = Vec::new();

    for (bits, query) in query_index_bits.iter().zip(queries) {
        let (answer, query_ops) = verify_single_root_query_opening_circuit::<BF, EF, RecMmcs, Comm>(
            circuit,
            perm_config,
            root,
            &dimensions,
            bits,
            query,
        )?;
        if answer.len() != (1 << params.folding_factor) {
            return Err(VerificationError::InvalidProofShape(format!(
                "WHIR STIR row width mismatch: expected {}, got {}",
                1 << params.folding_factor,
                answer.len()
            )));
        }
        vars.push(folded_domain_var_from_index_bits::<BF, EF>(
            circuit,
            params.folded_domain_gen,
            bits,
        ));
        folds.push(eval_multilinear_poly_circuit(
            circuit,
            &answer,
            folding_randomness,
        ));
        op_ids.extend(query_ops);
    }

    Ok((
        WhirSelectStatementTargets::new(params.num_variables, vars, folds),
        op_ids,
    ))
}

fn verify_select_statement_on_final_poly_circuit<EF>(
    circuit: &mut CircuitBuilder<EF>,
    statement: &WhirSelectStatementTargets,
    final_poly: &[Target],
) -> Result<(), VerificationError>
where
    EF: Field,
{
    if final_poly.len() != (1 << statement.num_variables) {
        return Err(VerificationError::InvalidProofShape(format!(
            "WHIR final polynomial arity mismatch: expected {}, got {}",
            1 << statement.num_variables,
            final_poly.len()
        )));
    }

    let zero = circuit.define_const(EF::ZERO);
    for (&var, &expected) in statement.vars.iter().zip(&statement.evaluations) {
        let mut value = zero;
        for &coeff in final_poly.iter().rev() {
            value = circuit.mul_add(value, var, coeff);
        }
        let diff = circuit.sub(value, expected);
        circuit.assert_zero(diff);
    }
    Ok(())
}

/// Verify one native WHIR PCS proof inside the recursive circuit.
///
/// This is the circuit analogue of `WhirPcs::verify` for the single-polynomial
/// path. It replays the native domain separator, parses the initial
/// commitment, checks every WHIR sumcheck, verifies every STIR/MMCS query, and
/// enforces the final folded-polynomial consistency equation.
#[allow(clippy::too_many_arguments)]
pub fn verify_native_whir_proof_circuit<
    BF,
    EF,
    C,
    RecMmcs,
    NativeChallenger,
    const DIGEST_ELEMS: usize,
>(
    circuit: &mut CircuitBuilder<EF>,
    challenger: &mut C,
    perm_config: Poseidon2Config,
    config: &WhirConfig<EF, BF, RecMmcs::Input, NativeChallenger>,
    targets: &WhirProofVerificationTargets<BF, EF, RecMmcs::Commitment, RecMmcs>,
) -> Result<Vec<NonPrimitiveOpId>, VerificationError>
where
    BF: TwoAdicField + PrimeField64,
    EF: ExtensionField<BF> + TwoAdicField + BasedVectorSpace<BF>,
    C: RecursiveChallenger<BF, EF>,
    RecMmcs: RecursiveMmcs<BF, EF>,
    RecMmcs::Commitment: ObservableCommitment + Clone,
    NativeChallenger: FieldChallenger<BF> + GrindingChallenger<Witness = BF>,
{
    if targets.opening_claims.len() != 1 {
        return Err(VerificationError::InvalidProofShape(
            "recursive WHIR verifier currently supports one committed polynomial".to_string(),
        ));
    }

    observe_whir_domain_separator_circuit::<
        BF,
        EF,
        C,
        RecMmcs::Input,
        NativeChallenger,
        DIGEST_ELEMS,
    >(circuit, challenger, config);

    let initial_commitment = targets.proof.initial_commitment.as_ref().ok_or_else(|| {
        VerificationError::InvalidProofShape(
            "WHIR proof is missing its initial commitment".to_string(),
        )
    })?;
    assert_same_observable_commitment(circuit, &targets.commitment, initial_commitment)?;
    touch_private_targets(circuit, &targets.proof.initial_ood_answers);
    touch_sumcheck_targets(circuit, &targets.proof.initial_sumcheck);
    for round in &targets.proof.rounds {
        touch_private_targets(circuit, &round.ood_answers);
        touch_private_targets(circuit, core::slice::from_ref(&round.pow_witness));
        for query in &round.queries {
            touch_query_targets::<BF, EF, RecMmcs>(circuit, query);
        }
        touch_sumcheck_targets(circuit, &round.sumcheck);
    }
    if let Some(final_poly) = &targets.proof.final_poly {
        touch_private_targets(circuit, final_poly);
    }
    touch_private_targets(
        circuit,
        core::slice::from_ref(&targets.proof.final_pow_witness),
    );
    for query in &targets.proof.final_queries {
        touch_query_targets::<BF, EF, RecMmcs>(circuit, query);
    }
    if let Some(final_sumcheck) = &targets.proof.final_sumcheck {
        touch_sumcheck_targets(circuit, final_sumcheck);
    }
    for round_indices in &targets.round_query_index_bits {
        for bits in round_indices {
            touch_private_targets(circuit, bits);
        }
    }
    for bits in &targets.final_query_index_bits {
        touch_private_targets(circuit, bits);
    }

    let mut constraints = Vec::new();
    let mut round_folding_randomness = Vec::new();
    let mut claimed_eval = circuit.define_const(EF::ZERO);

    let parsed_initial = observe_whir_commitment_and_ood_circuit::<BF, EF, C, _>(
        circuit,
        challenger,
        initial_commitment,
        &targets.proof.initial_ood_answers,
        config.num_variables,
        config.commitment_ood_samples,
    )?;

    let mut statement = WhirEqStatementTargets::initialize(config.num_variables);
    for claim in &targets.opening_claims[0] {
        statement.add_evaluated_constraint(claim.point.clone(), claim.value);
    }
    statement.concatenate(&ood_eq_statement_from_parsed(
        &parsed_initial,
        config.num_variables,
    ));

    let initial_constraint = WhirConstraintTargets::new(
        challenger.sample_ext(circuit),
        statement,
        WhirSelectStatementTargets::initialize(config.num_variables),
    );
    initial_constraint.combine_evals(circuit, &mut claimed_eval);
    constraints.push(initial_constraint);

    let initial_randomness = verify_whir_sumcheck_rounds_circuit::<BF, EF, C>(
        circuit,
        challenger,
        &targets.proof.initial_sumcheck,
        &mut claimed_eval,
        config.starting_folding_pow_bits,
    )?;
    round_folding_randomness.push(initial_randomness);

    let mut prev_root = parsed_initial.root;
    let mut mmcs_op_ids = Vec::new();

    if targets.proof.rounds.len() != config.n_rounds() {
        return Err(VerificationError::InvalidProofShape(format!(
            "WHIR round count mismatch: expected {}, got {}",
            config.n_rounds(),
            targets.proof.rounds.len()
        )));
    }

    for (round_index, round) in targets.proof.rounds.iter().enumerate() {
        let round_params = &config.round_parameters[round_index];
        let round_commitment = round.commitment.as_ref().ok_or_else(|| {
            VerificationError::InvalidProofShape(format!(
                "WHIR proof is missing commitment for round {round_index}"
            ))
        })?;
        let parsed_round = observe_whir_commitment_and_ood_circuit::<BF, EF, C, _>(
            circuit,
            challenger,
            round_commitment,
            &round.ood_answers,
            round_params.num_variables,
            round_params.ood_samples,
        )?;

        let (stir_statement, stir_ops) =
            verify_whir_stir_challenges_circuit::<BF, EF, C, RecMmcs, RecMmcs::Commitment>(
                circuit,
                challenger,
                perm_config,
                round_params,
                &prev_root,
                &round.queries,
                &targets.round_query_index_bits[round_index],
                round.pow_witness,
                round_folding_randomness
                    .last()
                    .expect("initial randomness exists"),
                true,
            )?;
        mmcs_op_ids.extend(stir_ops);

        let round_constraint = WhirConstraintTargets::new(
            challenger.sample_ext(circuit),
            ood_eq_statement_from_parsed(&parsed_round, round_params.num_variables),
            stir_statement,
        );
        round_constraint.combine_evals(circuit, &mut claimed_eval);
        constraints.push(round_constraint);

        let folding_randomness = verify_whir_sumcheck_rounds_circuit::<BF, EF, C>(
            circuit,
            challenger,
            &round.sumcheck,
            &mut claimed_eval,
            round_params.folding_pow_bits,
        )?;
        round_folding_randomness.push(folding_randomness);
        prev_root = parsed_round.root;
    }

    let final_poly = targets.proof.final_poly.as_ref().ok_or_else(|| {
        VerificationError::InvalidProofShape(
            "WHIR proof is missing its final polynomial".to_string(),
        )
    })?;
    touch_private_targets(circuit, final_poly);
    challenger.observe_ext_slice(circuit, final_poly);

    let final_round_config = config.final_round_config();
    let (final_stir_statement, final_stir_ops) =
        verify_whir_stir_challenges_circuit::<BF, EF, C, RecMmcs, RecMmcs::Commitment>(
            circuit,
            challenger,
            perm_config,
            &final_round_config,
            &prev_root,
            &targets.proof.final_queries,
            &targets.final_query_index_bits,
            targets.proof.final_pow_witness,
            round_folding_randomness
                .last()
                .expect("folding randomness exists"),
            false,
        )?;
    mmcs_op_ids.extend(final_stir_ops);
    verify_select_statement_on_final_poly_circuit(circuit, &final_stir_statement, final_poly)?;

    let final_sumcheck_randomness = verify_whir_final_sumcheck_rounds_circuit::<BF, EF, C>(
        circuit,
        challenger,
        targets.proof.final_sumcheck.as_ref(),
        &mut claimed_eval,
        config.final_sumcheck_rounds,
        config.final_folding_pow_bits,
    )?;
    round_folding_randomness.push(final_sumcheck_randomness.clone());

    let full_folding_randomness = round_folding_randomness
        .into_iter()
        .flat_map(IntoIterator::into_iter)
        .collect::<Vec<_>>();
    let evaluation_of_weights =
        eval_prefix_constraints_poly_circuit(circuit, &constraints, &full_folding_randomness);
    let final_value =
        eval_multilinear_poly_circuit(circuit, final_poly, &final_sumcheck_randomness);
    let expected = circuit.mul(evaluation_of_weights, final_value);
    let diff = circuit.sub(claimed_eval, expected);
    circuit.assert_zero(diff);

    Ok(mmcs_op_ids)
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_circuit::ops::Poseidon2Config;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;
    use crate::CircuitChallenger;

    type BF = BabyBear;
    type EF = BinomialExtensionField<BF, 4>;

    #[test]
    fn observe_whir_commitment_rejects_wrong_ood_answer_count() {
        let mut circuit = CircuitBuilder::<EF>::new();
        let root = circuit.define_const(EF::ZERO);
        let answer = circuit.define_const(EF::ONE);
        let mut challenger = CircuitChallenger::<16, 8, Poseidon2Config>::new_babybear();

        let result = observe_whir_commitment_and_ood_circuit::<BF, EF, _, _>(
            &mut circuit,
            &mut challenger,
            &root,
            &[answer],
            3,
            2,
        );

        match result {
            Err(VerificationError::InvalidProofShape(message)) => {
                assert!(message.contains("WHIR OOD answer count mismatch"));
            }
            Err(error) => panic!("unexpected verification error: {error:?}"),
            Ok(_) => panic!("wrong OOD answer count should be rejected"),
        }
    }
}
