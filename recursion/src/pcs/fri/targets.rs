use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::marker::PhantomData;

use p3_challenger::{CanObserve, GrindingChallenger};
use p3_circuit::symbolic::RowSelectorsTargets;
use p3_circuit::{CircuitBuilder, CircuitBuilderError, NonPrimitiveOpId};
use p3_commit::{BatchOpening, ExtensionMmcs, Mmcs, OpenedValues, PolynomialSpace};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PackedValue, PrimeCharacteristicRing, PrimeField64,
    TwoAdicField,
};
use p3_fri::{CommitPhaseProofStep, FriProof, HidingFriPcs, QueryProof, TwoAdicFriPcs};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, MerkleCap, PseudoCompressionFunction};
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

use super::{FriVerifierParams, verify_fri_circuit};
use crate::Target;
use crate::challenger::CircuitChallenger;
use crate::traits::{
    ComsWithOpeningsTargets, Recursive, RecursiveChallenger, RecursiveExtensionMmcs, RecursiveMmcs,
    RecursivePcs,
};
use crate::types::{OpenedValuesTargetsWithLookups, RecursiveLagrangeSelectors};
use crate::verifier::{ObservableCommitment, VerificationError};

/// `Recursive` version of `FriProof`.
pub struct FriProofTargets<
    F: Field,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveExtensionMmcs<F, EF>,
    InputProof: Recursive<EF>,
    Witness: Recursive<EF>,
> {
    pub commit_phase_commits: Vec<RecMmcs::Commitment>,
    pub commit_pow_witnesses: Vec<Witness>,
    pub query_proofs: Vec<QueryProofTargets<F, EF, InputProof, RecMmcs>>,
    pub final_poly: Vec<Target>,
    pub pow_witness: Witness,
    pub log_arities: Vec<usize>,
}

impl<
    F: Field,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveExtensionMmcs<F, EF>,
    InputProof: Recursive<EF>,
    Witness: Recursive<EF>,
> Recursive<EF> for FriProofTargets<F, EF, RecMmcs, InputProof, Witness>
{
    type Input = FriProof<EF, RecMmcs::Input, Witness::Input, InputProof::Input>;

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        let commit_phase_commits = input
            .commit_phase_commits
            .iter()
            .map(|commitment| RecMmcs::Commitment::new(circuit, commitment))
            .collect();

        let commit_pow_witnesses = input
            .commit_pow_witnesses
            .iter()
            .map(|witness| Witness::new(circuit, witness))
            .collect();

        let query_proofs = input
            .query_proofs
            .iter()
            .map(|query| QueryProofTargets::new(circuit, query))
            .collect();

        let final_poly = circuit
            .alloc_public_inputs(input.final_poly.len(), "FRI final polynomial coefficients");

        let log_arities = input
            .query_proofs
            .first()
            .map(|qp| {
                qp.commit_phase_openings
                    .iter()
                    .map(|o| o.log_arity as usize)
                    .collect()
            })
            .unwrap_or_default();

        Self {
            commit_phase_commits,
            commit_pow_witnesses,
            query_proofs,
            final_poly,
            pow_witness: Witness::new(circuit, &input.query_pow_witness),
            log_arities,
        }
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        let FriProof {
            commit_phase_commits,
            commit_pow_witnesses,
            query_proofs,
            final_poly,
            query_pow_witness,
        } = input;

        commit_phase_commits
            .iter()
            .flat_map(|c| RecMmcs::Commitment::get_values(c))
            .chain(
                commit_pow_witnesses
                    .iter()
                    .flat_map(|w| Witness::get_values(w)),
            )
            .chain(
                query_proofs
                    .iter()
                    .flat_map(|c| QueryProofTargets::<F, EF, InputProof, RecMmcs>::get_values(c)),
            )
            .chain(final_poly.iter().copied())
            .chain(Witness::get_values(query_pow_witness))
            .collect()
    }

    fn get_private_values(input: &Self::Input) -> Vec<EF> {
        input
            .query_proofs
            .iter()
            .flat_map(|c| QueryProofTargets::<F, EF, InputProof, RecMmcs>::get_private_values(c))
            .collect()
    }
}

/// `Recursive` version of `QueryProof`.
pub struct QueryProofTargets<
    F: Field,
    EF: ExtensionField<F>,
    InputProof: Recursive<EF>,
    RecMmcs: RecursiveExtensionMmcs<F, EF>,
> {
    pub input_proof: InputProof,
    pub commit_phase_openings: Vec<CommitPhaseProofStepTargets<F, EF, RecMmcs>>,
}

impl<
    F: Field,
    EF: ExtensionField<F>,
    InputProof: Recursive<EF>,
    RecMmcs: RecursiveExtensionMmcs<F, EF>,
> Recursive<EF> for QueryProofTargets<F, EF, InputProof, RecMmcs>
{
    type Input = QueryProof<EF, RecMmcs::Input, InputProof::Input>;

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        // Note that the iterator `lens` is updated by each call to `new`. So we can always pass the same `lens` for all structures.
        let input_proof = InputProof::new(circuit, &input.input_proof);
        let commit_phase_openings = input
            .commit_phase_openings
            .iter()
            .map(|commitment| CommitPhaseProofStepTargets::new(circuit, commitment))
            .collect();
        Self {
            input_proof,
            commit_phase_openings,
        }
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        InputProof::get_values(&input.input_proof)
            .into_iter()
            .chain(
                input
                    .commit_phase_openings
                    .iter()
                    .flat_map(|o| CommitPhaseProofStepTargets::<_, _, RecMmcs>::get_values(o)),
            )
            .collect()
    }

    fn get_private_values(input: &Self::Input) -> Vec<EF> {
        InputProof::get_private_values(&input.input_proof)
            .into_iter()
            .chain(
                input.commit_phase_openings.iter().flat_map(|o| {
                    CommitPhaseProofStepTargets::<_, _, RecMmcs>::get_private_values(o)
                }),
            )
            .collect()
    }
}

/// `Recursive` version of `CommitPhaseProofStep`.
///
/// Sibling values are stored as **lifted base field coefficients** to enable MMCS verification.
/// ExtensionMmcs commits by flattening extension elements to base field, so we need the
/// coefficients separately for hashing. Use `sibling_values_packed()` to get the packed
/// extension elements for FRI folding arithmetic.
///
/// For arity `k = 2^log_arity`, we store `k - 1` sibling values (the queried value is the
/// folded evaluation from the previous phase). Each sibling is represented by `EF::DIMENSION`
/// lifted base field coefficients, giving `(k - 1) * EF::DIMENSION` targets total.
pub struct CommitPhaseProofStepTargets<
    F: Field,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveExtensionMmcs<F, EF>,
> {
    pub log_arity: usize,
    /// Lifted base field coefficients for all (arity - 1) sibling values, flattened.
    /// Layout: [sib0_c0, sib0_c1, .., sib0_cD, sib1_c0, .., sib{a-2}_cD]
    pub sibling_coefficients: Vec<Target>,
    pub opening_proof: RecMmcs::Proof,
    _phantom: PhantomData<(F, EF)>,
}

impl<F: Field, EF: ExtensionField<F> + BasedVectorSpace<F>, RecMmcs: RecursiveExtensionMmcs<F, EF>>
    CommitPhaseProofStepTargets<F, EF, RecMmcs>
{
    /// Pack a single sibling's lifted base field coefficients into an extension element.
    ///
    /// Uses pure ALU ops (so the coefficient private inputs satisfy the "used in ALU operand"
    /// invariant) and registers the result in the circuit builder's coefficient-provenance cache
    /// via [`CircuitBuilder::hint_ext_recompose_coeffs`]. This makes a subsequent
    /// `decompose_ext_to_base_coeffs` call on the packed value a no-op rather than allocating
    /// new witnesses (critical for D=1 MMCS hashing over higher-degree extension fields).
    fn pack_one_sibling(coeffs: &[Target], circuit: &mut CircuitBuilder<EF>) -> Target {
        let basis: Vec<EF> = (0..EF::DIMENSION)
            .map(|i| EF::from_basis_coefficients_fn(|j| if i == j { F::ONE } else { F::ZERO }))
            .collect();

        let mut result = coeffs[0];
        for (i, &basis_elem) in basis.iter().enumerate().skip(1) {
            let basis_const = circuit.define_const(basis_elem);
            result = circuit.mul_add(coeffs[i], basis_const, result);
        }
        circuit.hint_ext_recompose_coeffs(result, coeffs);
        result
    }

    /// Returns all (arity - 1) sibling values as packed extension elements.
    pub fn sibling_values_packed(&self, circuit: &mut CircuitBuilder<EF>) -> Vec<Target> {
        let d = EF::DIMENSION;
        self.sibling_coefficients
            .chunks_exact(d)
            .map(|chunk| Self::pack_one_sibling(chunk, circuit))
            .collect()
    }

    /// Returns the single sibling value as a packed extension element (arity-2 convenience).
    pub fn sibling_value_packed(&self, circuit: &mut CircuitBuilder<EF>) -> Target {
        debug_assert_eq!(
            self.log_arity, 1,
            "sibling_value_packed is for arity-2 only; use sibling_values_packed for higher arity"
        );
        Self::pack_one_sibling(&self.sibling_coefficients, circuit)
    }
}

impl<F: Field, EF: ExtensionField<F> + BasedVectorSpace<F>, RecMmcs: RecursiveExtensionMmcs<F, EF>>
    Recursive<EF> for CommitPhaseProofStepTargets<F, EF, RecMmcs>
{
    type Input = CommitPhaseProofStep<EF, RecMmcs::Input>;

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        let log_arity = input.log_arity as usize;
        let arity = 1usize << log_arity;
        let num_siblings = arity - 1;
        let num_coeffs = num_siblings * EF::DIMENSION;
        let sibling_coefficients =
            circuit.alloc_private_inputs(num_coeffs, "FRI commit phase sibling coefficients");
        let opening_proof = RecMmcs::Proof::new(circuit, &input.opening_proof);
        Self {
            log_arity,
            sibling_coefficients,
            opening_proof,
            _phantom: PhantomData,
        }
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        RecMmcs::Proof::get_values(&input.opening_proof)
    }

    fn get_private_values(input: &Self::Input) -> Vec<EF> {
        let mut values: Vec<EF> = Vec::new();
        for sibling_value in &input.sibling_values {
            let coeffs = sibling_value.as_basis_coefficients_slice();
            values.extend(coeffs.iter().map(|&c| EF::from(c)));
        }
        values
    }
}

/// `Recursive` version of `BatchOpening`.
///
/// Uses **lifted representation**: each base field value is represented as a single extension
/// field element `EF([v, 0, 0, 0])`. This allows 1:1 correspondence with polynomial values
/// for arithmetic verification.
pub struct BatchOpeningTargets<F: Field, EF: ExtensionField<F>, RecMmcs: RecursiveMmcs<F, EF>> {
    /// The opened row values from each matrix in the batch.
    /// Each inner vector has one target per base field value.
    pub opened_values: Vec<Vec<Target>>,
    /// The proof showing the values are valid openings.
    pub opening_proof: RecMmcs::Proof,
}

impl<F: Field, EF: ExtensionField<F>, Inner: RecursiveMmcs<F, EF>> Recursive<EF>
    for BatchOpeningTargets<F, EF, Inner>
{
    type Input = BatchOpening<F, Inner::Input>;

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        let opened_values = input
            .opened_values
            .iter()
            .map(|values| circuit.alloc_private_inputs(values.len(), "batch opened values"))
            .collect();

        let opening_proof = Inner::Proof::new(circuit, &input.opening_proof);

        Self {
            opened_values,
            opening_proof,
        }
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        Inner::Proof::get_values(&input.opening_proof)
    }

    fn get_private_values(input: &Self::Input) -> Vec<EF> {
        input
            .opened_values
            .iter()
            .flat_map(|inner| inner.iter().map(|v| EF::from(*v)))
            .collect()
    }
}

// Now, we define the commitment schemes.

/// `MerkleCapTargets` corresponds to a Merkle cap commitment with `2^cap_height` hash entries,
/// each having `DIGEST_ELEMS` digest elements.
///
/// Uses **lifted representation**: each base field hash element is stored as a separate extension
/// field target `EF([v, 0, 0, 0])`. This is consistent with Fiat-Shamir observation.
///
/// A cap of height 0 contains a single entry (the root), while a cap of height `h` contains
/// `2^h` entries. The Fiat-Shamir transcript observes all entries sequentially.
#[derive(Clone)]
pub struct MerkleCapTargets<F, const DIGEST_ELEMS: usize> {
    pub cap_targets: Vec<[Target; DIGEST_ELEMS]>,
    _phantom: PhantomData<F>,
}

impl<F, const DIGEST_ELEMS: usize> ObservableCommitment for MerkleCapTargets<F, DIGEST_ELEMS> {
    fn to_observation_targets(&self) -> Vec<Target> {
        self.cap_targets
            .iter()
            .flat_map(|entry| entry.iter().copied())
            .collect()
    }
}

type ValMmcsCommitment<F, const DIGEST_ELEMS: usize> =
    MerkleCap<<F as PackedValue>::Value, [<F as PackedValue>::Value; DIGEST_ELEMS]>;

impl<F: Field, EF: ExtensionField<F>, const DIGEST_ELEMS: usize> Recursive<EF>
    for MerkleCapTargets<F, DIGEST_ELEMS>
{
    type Input = ValMmcsCommitment<F, DIGEST_ELEMS>;

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        let cap_targets = (0..input.num_roots())
            .map(|_| circuit.alloc_public_input_array("MMCS commitment cap entry"))
            .collect();
        Self {
            cap_targets,
            _phantom: PhantomData,
        }
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        input
            .roots()
            .iter()
            .flat_map(|entry: &[<F as PackedValue>::Value; DIGEST_ELEMS]| {
                entry.iter().map(|v| EF::from(*v))
            })
            .collect()
    }
}

/// `HashProofTargets` corresponds to a Merkle tree `Proof` in the form of a vector of hashes with `DIGEST_ELEMS` digest elements.
// TODO: This could be fully removed?
pub struct HashProofTargets<F, const DIGEST_ELEMS: usize> {
    pub hash_proof_targets: Vec<[Target; DIGEST_ELEMS]>,
    _phantom: PhantomData<F>,
}

type ValMmcsProof<PW, const DIGEST_ELEMS: usize> = Vec<[<PW as PackedValue>::Value; DIGEST_ELEMS]>;

impl<F: Field, EF: ExtensionField<F>, const DIGEST_ELEMS: usize> Recursive<EF>
    for HashProofTargets<F, DIGEST_ELEMS>
{
    type Input = ValMmcsProof<F, DIGEST_ELEMS>;

    fn new(_circuit: &mut CircuitBuilder<EF>, _input: &Self::Input) -> Self {
        // Merkle proof hashes are not allocated as circuit inputs.
        Self {
            hash_proof_targets: vec![],
            _phantom: PhantomData,
        }
    }

    fn get_values(_input: &Self::Input) -> Vec<EF> {
        vec![]
    }
}

/// In TwoAdicFriPcs, the POW witness is just a base field element.
pub struct Witness<F> {
    pub witness: Target,
    _phantom: PhantomData<F>,
}

impl<F: Field, EF: ExtensionField<F>> Recursive<EF> for Witness<F> {
    type Input = F;

    fn new(circuit: &mut CircuitBuilder<EF>, _input: &Self::Input) -> Self {
        Self {
            witness: circuit.alloc_public_input("FRI proof-of-work witness"),
            _phantom: PhantomData,
        }
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        vec![EF::from(*input)]
    }
}

/// `Recursive` version of a `MerkleTreeMmcs` where the leaf and digest elements are base field values.
pub struct RecValMmcs<F: Field, const DIGEST_ELEMS: usize, H, C>
where
    H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
        + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
        + Sync,
{
    pub hash: H,
    pub compress: C,
    _phantom: PhantomData<F>,
}

impl<F: Field, EF: ExtensionField<F>, const DIGEST_ELEMS: usize, H, C> RecursiveMmcs<F, EF>
    for RecValMmcs<F, DIGEST_ELEMS, H, C>
where
    H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
        + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
        + Sync,
    [F; DIGEST_ELEMS]: Serialize + for<'a> Deserialize<'a>,
{
    type Input = MerkleTreeMmcs<F::Packing, F::Packing, H, C, 2, DIGEST_ELEMS>;

    type Commitment = MerkleCapTargets<F, DIGEST_ELEMS>;

    type Proof = HashProofTargets<F, DIGEST_ELEMS>;
}

/// `Recursive` version of an `ExtensionFieldMmcs` where the inner `Mmcs` is a `MerkleTreeMmcs`.
pub struct RecExtensionValMmcs<
    F: Field,
    EF: ExtensionField<F>,
    const DIGEST_ELEMS: usize,
    MyMmcs: RecursiveMmcs<F, EF>,
> {
    _phantom: PhantomData<F>,
    _phantom_ef: PhantomData<EF>,
    _phantom_val: PhantomData<MyMmcs>,
}

impl<F: Field, EF: ExtensionField<F>, const DIGEST_ELEMS: usize, RecValMmcs: RecursiveMmcs<F, EF>>
    RecursiveExtensionMmcs<F, EF> for RecExtensionValMmcs<F, EF, DIGEST_ELEMS, RecValMmcs>
{
    type Input = ExtensionMmcs<F, EF, RecValMmcs::Input>;

    type Commitment = RecValMmcs::Commitment;

    type Proof = RecValMmcs::Proof;
}

pub type InputProofTargets<F, EF, Inner> = Vec<BatchOpeningTargets<F, EF, Inner>>;

pub type TwoAdicFriProofTargets<F, EF, RecMmcs, Inner> =
    FriProofTargets<F, EF, RecMmcs, InputProofTargets<F, EF, Inner>, Target>;

impl<F: Field, EF: ExtensionField<F>, Inner: RecursiveMmcs<F, EF>> Recursive<EF>
    for InputProofTargets<F, EF, Inner>
{
    type Input = Vec<BatchOpening<F, Inner::Input>>;

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        let num_batch_openings = input.len();
        let mut batch_openings = Self::with_capacity(num_batch_openings);
        for batch_opening in input.iter() {
            batch_openings.push(BatchOpeningTargets::new(circuit, batch_opening));
        }

        batch_openings
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        input
            .iter()
            .flat_map(|batch_opening| {
                BatchOpeningTargets::<F, EF, Inner>::get_values(batch_opening)
            })
            .collect()
    }

    fn get_private_values(input: &Self::Input) -> Vec<EF> {
        input
            .iter()
            .flat_map(|batch_opening| {
                BatchOpeningTargets::<F, EF, Inner>::get_private_values(batch_opening)
            })
            .collect()
    }
}

// Recursive type for the `FriProof` of `TwoAdicFriPcs`.
type RecursiveFriProof<SC, RecursiveFriMmcs, RecursiveInputProof> = FriProofTargets<
    Val<SC>,
    <SC as StarkGenericConfig>::Challenge,
    RecursiveFriMmcs,
    RecursiveInputProof,
    Witness<Val<SC>>,
>;

// Implement `RecursivePcs` for `TwoAdicFriPcs`.
impl<SC, Dft, Comm, InputMmcs, RecursiveInputMmcs, RecursiveFriMmcs, FriMmcs>
    RecursivePcs<
        SC,
        InputProofTargets<Val<SC>, SC::Challenge, RecursiveInputMmcs>,
        RecursiveFriProof<
            SC,
            RecursiveFriMmcs,
            InputProofTargets<Val<SC>, SC::Challenge, RecursiveInputMmcs>,
        >,
        Comm,
        TwoAdicMultiplicativeCoset<Val<SC>>,
    > for TwoAdicFriPcs<Val<SC>, Dft, InputMmcs, FriMmcs>
where
    SC: StarkGenericConfig,
    Val<SC>: TwoAdicField + PrimeField64,
    InputMmcs: Mmcs<Val<SC>>,
    FriMmcs: Mmcs<SC::Challenge>,
    Comm: Recursive<SC::Challenge> + ObservableCommitment,
    RecursiveInputMmcs: RecursiveMmcs<Val<SC>, SC::Challenge, Input = InputMmcs>,
    RecursiveFriMmcs: RecursiveExtensionMmcs<Val<SC>, SC::Challenge, Input = FriMmcs>,
    RecursiveFriMmcs::Commitment: ObservableCommitment,
    SC::Challenger: GrindingChallenger + CanObserve<FriMmcs::Commitment>,
{
    type VerifierParams = FriVerifierParams;
    type RecursiveProof = RecursiveFriProof<
        SC,
        RecursiveFriMmcs,
        InputProofTargets<Val<SC>, SC::Challenge, RecursiveInputMmcs>,
    >;

    /// Observes all opened values and derives PCS-specific challenges.
    fn get_challenges_circuit<
        const WIDTH: usize,
        const RATE: usize,
        C: crate::ChallengerPermConfig,
    >(
        circuit: &mut CircuitBuilder<SC::Challenge>,
        challenger: &mut CircuitChallenger<WIDTH, RATE, C>,
        fri_proof: &RecursiveFriProof<
            SC,
            RecursiveFriMmcs,
            InputProofTargets<Val<SC>, SC::Challenge, RecursiveInputMmcs>,
        >,
        _opened_values: &OpenedValuesTargetsWithLookups<SC>,
        params: &Self::VerifierParams,
    ) -> Result<Vec<Target>, CircuitBuilderError>
    where
        Val<SC>: PrimeField64,
        SC::Challenge: ExtensionField<Val<SC>>,
    {
        // NOTE: Opened values must be observed by the caller BEFORE calling this function.
        // For batch-STARK, the caller must observe in per-instance order to match native.
        // For single-STARK, the caller can use opened_values.observe() directly.

        // Sample FRI alpha (for batch opening reduction) - extension field
        let fri_alpha = challenger.sample_ext(circuit);

        // Sample FRI betas: one per commit phase
        // For each FRI commitment, observe it and sample beta
        let mut betas = Vec::with_capacity(fri_proof.commit_phase_commits.len());
        for (commit, pow) in fri_proof
            .commit_phase_commits
            .iter()
            .zip(fri_proof.commit_pow_witnesses.iter())
        {
            let commit_targets = commit.to_observation_targets();
            challenger.observe_slice(circuit, &commit_targets);
            // Check commit-phase PoW witness.
            challenger.check_pow_witness(circuit, params.commit_pow_bits, pow.witness)?;
            // Sample beta - extension field
            let beta = challenger.sample_ext(circuit);
            betas.push(beta);
        }

        // Observe final polynomial coefficients (extension field values)
        challenger.observe_ext_slice(circuit, &fri_proof.final_poly);

        // Bind the variable-arity schedule into the transcript before query grinding,
        // matching the native FRI verifier in Plonky3.
        for &log_arity in &fri_proof.log_arities {
            let log_arity_target =
                circuit.alloc_const(SC::Challenge::from_usize(log_arity), "FRI log_arity");
            challenger.observe(circuit, log_arity_target);
        }

        // Check query PoW witness.
        challenger.check_pow_witness(
            circuit,
            params.query_pow_bits,
            fri_proof.pow_witness.witness,
        )?;

        // Query indices are sampled in-circuit by verify_circuit from the challenger
        // (which is left in the correct state here) to ensure soundness.
        let mut challenges = Vec::with_capacity(1 + betas.len());
        challenges.push(fri_alpha);
        challenges.extend(betas);
        Ok(challenges)
    }

    fn verify_circuit<const WIDTH: usize, const RATE: usize, C: crate::ChallengerPermConfig>(
        &self,
        circuit: &mut CircuitBuilder<SC::Challenge>,
        challenges: &[Target],
        challenger: &mut CircuitChallenger<WIDTH, RATE, C>,
        commitments_with_opening_points: &ComsWithOpeningsTargets<
            Comm,
            TwoAdicMultiplicativeCoset<Val<SC>>,
        >,
        opening_proof: &Self::RecursiveProof,
        params: &Self::VerifierParams,
    ) -> Result<Vec<NonPrimitiveOpId>, VerificationError> {
        let FriVerifierParams {
            log_blowup,
            log_final_poly_len,
            commit_pow_bits: _,
            query_pow_bits: _,
            permutation_config,
        } = *params;
        let num_betas = opening_proof.commit_phase_commits.len();
        let num_queries = opening_proof.query_proofs.len();

        let alpha = challenges[0];
        let betas = &challenges[1..1 + num_betas];

        let total_log_reduction: usize = opening_proof.log_arities.iter().sum();
        let log_max_height = total_log_reduction + log_final_poly_len + log_blowup;

        let max_query_index_bits = Val::<SC>::bits();
        if log_max_height > max_query_index_bits {
            return Err(VerificationError::InvalidProofShape(format!(
                "log_max_height {log_max_height} exceeds base field bit width {max_query_index_bits}"
            )));
        }

        let index_bits_per_query: Vec<Vec<Target>> = (0..num_queries)
            .map(|_| challenger.sample_bits(circuit, log_max_height))
            .collect::<Result<Vec<_>, _>>()?;

        verify_fri_circuit(
            circuit,
            opening_proof,
            alpha,
            betas,
            &index_bits_per_query,
            commitments_with_opening_points,
            log_blowup,
            permutation_config,
        )
    }

    fn selectors_at_point_circuit(
        &self,
        circuit: &mut CircuitBuilder<SC::Challenge>,
        domain: &TwoAdicMultiplicativeCoset<Val<SC>>,
        point: &Target,
    ) -> RecursiveLagrangeSelectors {
        // Constants that we will need.
        let shift_inv =
            circuit.alloc_const(SC::Challenge::from(domain.shift_inverse()), "shift_inv");
        let one = circuit.alloc_const(SC::Challenge::from(Val::<SC>::ONE), "1");
        let subgroup_gen_inv = circuit.alloc_const(
            SC::Challenge::from(domain.subgroup_generator().inverse()),
            "subgroup_gen_inv",
        );

        // Unshifted and z_h
        let unshifted_point = circuit.alloc_mul(shift_inv, *point, "unshifted_point");
        let us_exp = circuit.exp_power_of_2(unshifted_point, domain.log_size());
        let z_h = circuit.alloc_sub(us_exp, one, "z_h");

        // Denominators
        let us_minus_one = circuit.alloc_sub(unshifted_point, one, "us_minus_one");
        let us_minus_gen_inv =
            circuit.alloc_sub(unshifted_point, subgroup_gen_inv, "us_minus_gen_inv");

        // Selectors
        let is_first_row = circuit.alloc_div(z_h, us_minus_one, "is_first_row");
        let is_last_row = circuit.alloc_div(z_h, us_minus_gen_inv, "is_last_row");
        let is_transition = us_minus_gen_inv;
        let inv_vanishing = circuit.alloc_div(one, z_h, "inv_vanishing");

        let row_selectors = RowSelectorsTargets {
            is_first_row,
            is_last_row,
            is_transition,
        };
        RecursiveLagrangeSelectors {
            row_selectors,
            inv_vanishing,
        }
    }

    fn create_disjoint_domain(
        &self,
        trace_domain: TwoAdicMultiplicativeCoset<Val<SC>>,
        degree: usize,
    ) -> TwoAdicMultiplicativeCoset<Val<SC>> {
        trace_domain.create_disjoint_domain(degree)
    }

    fn split_domains(
        &self,
        trace_domain: &TwoAdicMultiplicativeCoset<Val<SC>>,
        degree: usize,
    ) -> Vec<TwoAdicMultiplicativeCoset<Val<SC>>> {
        trace_domain.split_domains(degree)
    }

    fn log_size(&self, trace_domain: &TwoAdicMultiplicativeCoset<Val<SC>>) -> usize {
        trace_domain.log_size()
    }

    fn first_point(&self, trace_domain: &TwoAdicMultiplicativeCoset<Val<SC>>) -> SC::Challenge {
        trace_domain.first_point().into()
    }

    // This is not used for non-ZK FRI proofs.
    fn get_fri_random_opened_values(
        _proof: &RecursiveFriProof<
            SC,
            RecursiveFriMmcs,
            InputProofTargets<Val<SC>, SC::Challenge, RecursiveInputMmcs>,
        >,
    ) -> &[Vec<Vec<Vec<Target>>>] {
        &[]
    }
}

/// Recursive targets for the extra random opened values carried by `HidingFriPcs`.
pub struct HidingOpenedValuesTargets<EF: Field> {
    /// Layout: rounds -> matrices -> points -> values.
    pub rounds: Vec<Vec<Vec<Vec<Target>>>>,
    _phantom: PhantomData<EF>,
}

impl<EF: Field> Recursive<EF> for HidingOpenedValuesTargets<EF> {
    type Input = OpenedValues<EF>;

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        let rounds = input
            .iter()
            .map(|round| {
                round
                    .iter()
                    .map(|matrix| {
                        matrix
                            .iter()
                            .map(|point_vals| {
                                circuit.alloc_private_inputs(
                                    point_vals.len(),
                                    "hiding random opened values",
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self {
            rounds,
            _phantom: PhantomData,
        }
    }

    fn get_values(_input: &Self::Input) -> Vec<EF> {
        vec![]
    }

    fn get_private_values(input: &Self::Input) -> Vec<EF> {
        input
            .iter()
            .flat_map(|round| round.iter())
            .flat_map(|matrix| matrix.iter())
            .flat_map(|point_vals| point_vals.iter().copied())
            .collect()
    }
}

/// Recursive proof targets for `HidingFriPcs`.
///
/// This wraps:
/// 1. Random opened values split out by `HidingFriPcs`
/// 2. The inner FRI proof (same structure as `TwoAdicFriPcs`)
pub struct HidingFriProofTargets<
    F: Field,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveExtensionMmcs<F, EF>,
    InputProof: Recursive<EF>,
    PowWitness: Recursive<EF>,
> {
    pub random_opened_values: HidingOpenedValuesTargets<EF>,
    pub inner_proof: FriProofTargets<F, EF, RecMmcs, InputProof, PowWitness>,
}

impl<
    F: Field,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveExtensionMmcs<F, EF>,
    InputProof: Recursive<EF>,
    PowWitness: Recursive<EF>,
> Recursive<EF> for HidingFriProofTargets<F, EF, RecMmcs, InputProof, PowWitness>
{
    type Input = (
        OpenedValues<EF>,
        FriProof<EF, RecMmcs::Input, PowWitness::Input, InputProof::Input>,
    );

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        Self {
            random_opened_values: HidingOpenedValuesTargets::new(circuit, &input.0),
            inner_proof: FriProofTargets::new(circuit, &input.1),
        }
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        FriProofTargets::<F, EF, RecMmcs, InputProof, PowWitness>::get_values(&input.1)
    }

    fn get_private_values(input: &Self::Input) -> Vec<EF> {
        HidingOpenedValuesTargets::<EF>::get_private_values(&input.0)
            .into_iter()
            .chain(
                FriProofTargets::<F, EF, RecMmcs, InputProof, PowWitness>::get_private_values(
                    &input.1,
                ),
            )
            .collect()
    }
}

type RecursiveHidingFriProof<SC, RecursiveFriMmcs, RecursiveInputProof> = HidingFriProofTargets<
    Val<SC>,
    <SC as StarkGenericConfig>::Challenge,
    RecursiveFriMmcs,
    RecursiveInputProof,
    Witness<Val<SC>>,
>;

/// Merged commitments with opening points and random opened values.
type HidingMergedCommitments<SC, Comm> = (
    Comm,
    Vec<(
        TwoAdicMultiplicativeCoset<Val<SC>>,
        Vec<(Target, Vec<Target>)>,
    )>,
);

fn merge_hiding_random_openings<SC, Comm>(
    commitments_with_opening_points: &ComsWithOpeningsTargets<
        Comm,
        TwoAdicMultiplicativeCoset<Val<SC>>,
    >,
    random_opened_values: &[Vec<Vec<Vec<Target>>>],
) -> Result<Vec<HidingMergedCommitments<SC, Comm>>, VerificationError>
where
    SC: StarkGenericConfig,
    Val<SC>: TwoAdicField,
    Comm: Clone,
{
    if commitments_with_opening_points.len() != random_opened_values.len() {
        return Err(VerificationError::InvalidProofShape(
            "Hiding FRI proof shape mismatch: random rounds count does not match commitments"
                .to_string(),
        ));
    }

    let mut merged = Vec::with_capacity(commitments_with_opening_points.len());

    for ((commitment, mats), rand_round) in commitments_with_opening_points
        .iter()
        .zip(random_opened_values.iter())
    {
        if mats.len() != rand_round.len() {
            return Err(VerificationError::InvalidProofShape(
                "Hiding FRI proof shape mismatch: random matrices count does not match".to_string(),
            ));
        }

        let mut merged_mats = Vec::with_capacity(mats.len());
        for ((domain, points), rand_mat) in mats.iter().zip(rand_round.iter()) {
            if points.len() != rand_mat.len() {
                return Err(VerificationError::InvalidProofShape(
                    "Hiding FRI proof shape mismatch: random points count does not match"
                        .to_string(),
                ));
            }

            let mut merged_points = Vec::with_capacity(points.len());
            for ((point, vals), rand_point) in points.iter().zip(rand_mat.iter()) {
                let mut merged_vals = vals.clone();
                merged_vals.extend(rand_point.iter().copied());
                merged_points.push((*point, merged_vals));
            }

            merged_mats.push((*domain, merged_points));
        }

        merged.push((commitment.clone(), merged_mats));
    }

    Ok(merged)
}

impl<SC, Dft, Comm, InputMmcs, RecursiveInputMmcs, RecursiveFriMmcs, FriMmcs, R>
    RecursivePcs<
        SC,
        InputProofTargets<Val<SC>, SC::Challenge, RecursiveInputMmcs>,
        RecursiveHidingFriProof<
            SC,
            RecursiveFriMmcs,
            InputProofTargets<Val<SC>, SC::Challenge, RecursiveInputMmcs>,
        >,
        Comm,
        TwoAdicMultiplicativeCoset<Val<SC>>,
    > for HidingFriPcs<Val<SC>, Dft, InputMmcs, FriMmcs, R>
where
    SC: StarkGenericConfig,
    Val<SC>: TwoAdicField + PrimeField64,
    InputMmcs: Mmcs<Val<SC>>,
    FriMmcs: Mmcs<SC::Challenge>,
    Comm: Recursive<SC::Challenge> + ObservableCommitment + Clone,
    RecursiveInputMmcs: RecursiveMmcs<Val<SC>, SC::Challenge, Input = InputMmcs>,
    RecursiveFriMmcs: RecursiveExtensionMmcs<Val<SC>, SC::Challenge, Input = FriMmcs>,
    RecursiveFriMmcs::Commitment: ObservableCommitment,
    SC::Challenger: GrindingChallenger + CanObserve<FriMmcs::Commitment>,
{
    type VerifierParams = FriVerifierParams;
    type RecursiveProof = RecursiveHidingFriProof<
        SC,
        RecursiveFriMmcs,
        InputProofTargets<Val<SC>, SC::Challenge, RecursiveInputMmcs>,
    >;

    fn get_challenges_circuit<
        const WIDTH: usize,
        const RATE: usize,
        C: crate::ChallengerPermConfig,
    >(
        circuit: &mut CircuitBuilder<SC::Challenge>,
        challenger: &mut CircuitChallenger<WIDTH, RATE, C>,
        proof_targets: &Self::RecursiveProof,
        _opened_values: &OpenedValuesTargetsWithLookups<SC>,
        params: &Self::VerifierParams,
    ) -> Result<Vec<Target>, CircuitBuilderError>
    where
        Val<SC>: PrimeField64,
        SC::Challenge: ExtensionField<Val<SC>>,
    {
        let fri_proof = &proof_targets.inner_proof;

        let fri_alpha = challenger.sample_ext(circuit);

        let mut betas = Vec::with_capacity(fri_proof.commit_phase_commits.len());
        for (commit, pow) in fri_proof
            .commit_phase_commits
            .iter()
            .zip(fri_proof.commit_pow_witnesses.iter())
        {
            let commit_targets = commit.to_observation_targets();
            challenger.observe_slice(circuit, &commit_targets);
            challenger.check_pow_witness(circuit, params.commit_pow_bits, pow.witness)?;
            let beta = challenger.sample_ext(circuit);
            betas.push(beta);
        }

        challenger.observe_ext_slice(circuit, &fri_proof.final_poly);

        for &log_arity in &fri_proof.log_arities {
            let log_arity_target =
                circuit.alloc_const(SC::Challenge::from_usize(log_arity), "FRI log_arity");
            challenger.observe(circuit, log_arity_target);
        }

        challenger.check_pow_witness(
            circuit,
            params.query_pow_bits,
            fri_proof.pow_witness.witness,
        )?;

        let mut challenges = Vec::with_capacity(1 + betas.len());
        challenges.push(fri_alpha);
        challenges.extend(betas);
        Ok(challenges)
    }

    fn verify_circuit<const WIDTH: usize, const RATE: usize, C: crate::ChallengerPermConfig>(
        &self,
        circuit: &mut CircuitBuilder<SC::Challenge>,
        challenges: &[Target],
        challenger: &mut CircuitChallenger<WIDTH, RATE, C>,
        commitments_with_opening_points: &ComsWithOpeningsTargets<
            Comm,
            TwoAdicMultiplicativeCoset<Val<SC>>,
        >,
        opening_proof: &Self::RecursiveProof,
        params: &Self::VerifierParams,
    ) -> Result<Vec<NonPrimitiveOpId>, VerificationError> {
        let FriVerifierParams {
            log_blowup,
            log_final_poly_len,
            commit_pow_bits: _,
            query_pow_bits: _,
            permutation_config,
        } = *params;
        let fri_proof = &opening_proof.inner_proof;
        let num_betas = fri_proof.commit_phase_commits.len();
        let num_queries = fri_proof.query_proofs.len();

        let alpha = challenges[0];
        let betas = &challenges[1..1 + num_betas];

        let total_log_reduction: usize = fri_proof.log_arities.iter().sum();
        let log_max_height = total_log_reduction + log_final_poly_len + log_blowup;

        let max_query_index_bits = Val::<SC>::bits();
        if log_max_height > max_query_index_bits {
            return Err(VerificationError::InvalidProofShape(format!(
                "log_max_height {log_max_height} exceeds base field bit width {max_query_index_bits}"
            )));
        }

        let index_bits_per_query: Vec<Vec<Target>> = (0..num_queries)
            .map(|_| challenger.sample_bits(circuit, log_max_height))
            .collect::<Result<Vec<_>, _>>()?;

        let merged_commitments = merge_hiding_random_openings::<SC, Comm>(
            commitments_with_opening_points,
            &opening_proof.random_opened_values.rounds,
        )?;

        verify_fri_circuit(
            circuit,
            fri_proof,
            alpha,
            betas,
            &index_bits_per_query,
            &merged_commitments,
            log_blowup,
            permutation_config,
        )
    }

    fn selectors_at_point_circuit(
        &self,
        circuit: &mut CircuitBuilder<SC::Challenge>,
        domain: &TwoAdicMultiplicativeCoset<Val<SC>>,
        point: &Target,
    ) -> RecursiveLagrangeSelectors {
        let shift_inv =
            circuit.alloc_const(SC::Challenge::from(domain.shift_inverse()), "shift_inv");
        let one = circuit.alloc_const(SC::Challenge::from(Val::<SC>::ONE), "1");
        let subgroup_gen_inv = circuit.alloc_const(
            SC::Challenge::from(domain.subgroup_generator().inverse()),
            "subgroup_gen_inv",
        );

        let unshifted_point = circuit.alloc_mul(shift_inv, *point, "unshifted_point");
        let us_exp = circuit.exp_power_of_2(unshifted_point, domain.log_size());
        let z_h = circuit.alloc_sub(us_exp, one, "z_h");

        let us_minus_one = circuit.alloc_sub(unshifted_point, one, "us_minus_one");
        let us_minus_gen_inv =
            circuit.alloc_sub(unshifted_point, subgroup_gen_inv, "us_minus_gen_inv");

        let is_first_row = circuit.alloc_div(z_h, us_minus_one, "is_first_row");
        let is_last_row = circuit.alloc_div(z_h, us_minus_gen_inv, "is_last_row");
        let is_transition = us_minus_gen_inv;
        let inv_vanishing = circuit.alloc_div(one, z_h, "inv_vanishing");

        let row_selectors = RowSelectorsTargets {
            is_first_row,
            is_last_row,
            is_transition,
        };
        RecursiveLagrangeSelectors {
            row_selectors,
            inv_vanishing,
        }
    }

    fn create_disjoint_domain(
        &self,
        trace_domain: TwoAdicMultiplicativeCoset<Val<SC>>,
        degree: usize,
    ) -> TwoAdicMultiplicativeCoset<Val<SC>> {
        trace_domain.create_disjoint_domain(degree)
    }

    fn split_domains(
        &self,
        trace_domain: &TwoAdicMultiplicativeCoset<Val<SC>>,
        degree: usize,
    ) -> Vec<TwoAdicMultiplicativeCoset<Val<SC>>> {
        trace_domain.split_domains(degree)
    }

    fn log_size(&self, trace_domain: &TwoAdicMultiplicativeCoset<Val<SC>>) -> usize {
        trace_domain.log_size()
    }

    fn first_point(&self, trace_domain: &TwoAdicMultiplicativeCoset<Val<SC>>) -> SC::Challenge {
        trace_domain.first_point().into()
    }

    fn get_fri_random_opened_values(
        proof: &RecursiveHidingFriProof<
            SC,
            RecursiveFriMmcs,
            InputProofTargets<Val<SC>, SC::Challenge, RecursiveInputMmcs>,
        >,
    ) -> &[Vec<Vec<Vec<Target>>>] {
        &proof.random_opened_values.rounds
    }
}
