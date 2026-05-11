//! This module provides type-safe builders and helper functions
//! for constructing public inputs for recursive verification circuits.

use alloc::vec::Vec;

use p3_batch_stark::{BatchProof, CommonData};
use p3_circuit::CircuitBuilder;
use p3_commit::Pcs;
use p3_field::{BasedVectorSpace, Field, PrimeField64};
use p3_uni_stark::{Proof, StarkGenericConfig, Val};

use crate::traits::Recursive;
use crate::{BatchProofTargets, CommonDataTargets, ProofTargets};

/// Builder for constructing a flat public input vector in a canonical order.
///
/// This type accumulates field elements in insertion order.
///
/// **NOTE:** The caller is responsible for calling the methods in the same logical
/// order in which the circuit allocated its public input targets.
///
/// The key invariant is:
///
/// - The order of calls to this builder must match the order in which the
///   circuit created its public input targets.
///
/// If this invariant is broken, the verifier circuit will read incorrect
/// values and verification will fail, even if the proof is valid.
///
/// # Typical usage
///
/// ```ignore
/// let inputs = PublicInputBuilder::new()
///     .add_proof_values(proof_values)
///     .add_challenge(alpha)
///     .add_challenges(betas)
///     .build();
/// ```
#[derive(Default)]
pub struct PublicInputBuilder<F: Field> {
    /// Accumulated public input values in exact insertion order.
    inputs: Vec<F>,
}

impl<F: Field> PublicInputBuilder<F> {
    /// Creates a new empty builder with no accumulated inputs.
    ///
    /// The internal vector starts empty and grows as values are added.
    pub const fn new() -> Self {
        Self { inputs: Vec::new() }
    }

    /// Appends a sequence of proof-related values.
    ///
    /// These values are usually extracted from a proof or commitment structure.
    /// They may represent commitments, opened evaluations, FRI data, and
    /// other components that must appear as public inputs.
    ///
    /// # Parameters
    /// - `values`: An iterable collection of field elements to append.
    ///
    /// # Returns
    /// A mutable reference to `self` for method chaining.
    pub fn add_proof_values(&mut self, values: impl IntoIterator<Item = F>) -> &mut Self {
        // Append all provided values at the end of the internal buffer.
        self.inputs.extend(values);
        self
    }

    /// Appends a single challenge value.
    ///
    /// Challenges are random-looking values produced by Fiat–Shamir
    /// procedure on the verifier side.
    ///
    /// # Parameters
    /// - `challenge`: The challenge field element to append.
    ///
    /// # Returns
    /// A mutable reference to `self` for method chaining.
    pub fn add_challenge(&mut self, challenge: F) -> &mut Self {
        // Push one challenge at the end of the buffer.
        self.inputs.push(challenge);
        self
    }

    /// Appends a sequence of challenge values.
    ///
    /// The order of the iterator is preserved in the public input vector.
    ///
    /// # Parameters
    /// - `challenges`: An iterable collection of challenge elements.
    ///
    /// # Returns
    /// A mutable reference to `self` for method chaining.
    pub fn add_challenges(&mut self, challenges: impl IntoIterator<Item = F>) -> &mut Self {
        self.inputs.extend(challenges);
        self
    }

    /// Appends the bit decomposition of a query index.
    ///
    /// A FRI query index is decomposed into its binary representation.
    ///
    /// The bits are stored in little-endian order:
    /// - bit 0 (least significant bit) comes first.
    ///
    /// More precisely, for an integer index `i` we append bits:
    ///
    /// ```text
    ///     i = Σ(j=0 to k-1) b_j · 2^j
    /// ```
    ///
    /// where b_j ∈ {0, 1} and k = `F::bits()` (the bit width of the base field).
    ///
    /// # Parameters
    /// - `index`: The query index as a field element.
    ///
    /// # Returns
    /// A mutable reference to `self` for method chaining.
    pub fn add_query_index(&mut self, index: F) -> &mut Self
    where
        F: PrimeField64,
    {
        // Interpret the field element as a canonical 64-bit integer.
        let index_usize = index.as_canonical_u64() as usize;

        // For each bit position k in [0, MAX_QUERY_INDEX_BITS):
        for k in 0..F::bits() {
            // Extract bit k: shift right by k, then mask with 1.
            let bit = if (index_usize >> k) & 1 == 1 {
                F::ONE
            } else {
                F::ZERO
            };
            // Append the bit as a field element (0 or 1).
            self.inputs.push(bit);
        }

        self
    }

    /// Appends pre-decomposed query index bits.
    ///
    /// Use this when the bit decomposition has already been performed elsewhere.
    ///
    /// The bits must be given in little-endian order (LSB first), and each bit
    /// should be encoded as field element 0 or 1.
    ///
    /// # Parameters
    /// - `bits`: An iterable collection of bit elements (each 0 or 1).
    ///
    /// # Returns
    /// A mutable reference to `self` for method chaining.
    pub fn add_query_index_bits(&mut self, bits: impl IntoIterator<Item = F>) -> &mut Self {
        // Simply append the provided bits without modification.
        self.inputs.extend(bits);
        self
    }

    /// Returns the current number of accumulated inputs.
    ///
    /// This is the length of the internal `inputs` vector.
    pub const fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Returns `true` if no inputs have been added yet.
    ///
    /// This is equivalent to `self.len() == 0`.
    pub const fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    /// Consumes the builder and returns the final public input vector.
    ///
    /// The returned vector contains all accumulated values in insertion order.
    ///
    /// No further modifications can be made through this builder.
    pub fn build(self) -> Vec<F> {
        self.inputs
    }
}

/// Opening data for a single polynomial commitment.
///
/// In a polynomial commitment scheme, a commitment binds to a polynomial.
///
/// An opening provides evidence that the committed polynomial evaluates to
/// specific values at specific points.
#[derive(Clone, Debug)]
pub struct CommitmentOpening<F: Field> {
    /// The commitment value in the field representation used by the circuit.
    pub commitment: F,

    /// Opened evaluation points and their corresponding values.
    ///
    /// Each entry is a pair `(z, v)` where:
    /// - `z` is an evaluation point,
    /// - `v` is the vector of polynomial values at `z`.
    pub opened_points: Vec<(F, Vec<F>)>,
}

/// Helper structure for constructing public inputs for FRI-only circuits.
pub struct FriVerifierInputs<F: Field> {
    /// Field values extracted from the FRI proof.
    pub fri_proof_values: Vec<F>,

    /// The batching challenge α used to combine multiple polynomials.
    ///
    /// For polynomials p_i(x), the combined polynomial is:
    ///
    /// ```text
    ///     p_comb(x) = Σ_i α^i · p_i(x)
    /// ```
    pub alpha: F,

    /// FRI folding challenges.
    ///
    /// In each FRI round, the polynomial is folded using a challenge β:
    ///
    /// ```text
    ///     p'(x) = p_even(x) + β · p_odd(x)
    /// ```
    pub betas: Vec<F>,

    /// Query index bits for each query, in little-endian order.
    ///
    /// Each inner vector:
    /// - has length `F::bits()` (the bit width of the base field),
    /// - encodes one query index as bits 0 or 1.
    pub query_index_bits: Vec<Vec<F>>,

    /// Commitment openings for all committed polynomials involved in FRI.
    pub commitment_openings: Vec<CommitmentOpening<F>>,
}

impl<F: Field> FriVerifierInputs<F> {
    /// Flattens all FRI-related data into a single public input vector.
    ///
    /// # Canonical input order
    /// 1. FRI proof values,
    /// 2. Batching challenge α,
    /// 3. FRI folding challenges β,
    /// 4. Query index bits for each query,
    /// 5. Commitment openings:
    ///    - Commitment,
    ///    - For each opening: `(z, f(z))` values.
    ///
    /// # Returns
    /// A vector of field elements ready for circuit execution.
    pub fn build(self) -> Vec<F> {
        // Start from an empty builder.
        let mut builder = PublicInputBuilder::new();

        // 1. Add all values extracted from the FRI proof structure.
        builder.add_proof_values(self.fri_proof_values);

        // 2. Add the α challenge for polynomial batching.
        builder.add_challenge(self.alpha);

        // 3. Add all β challenges for FRI folding.
        builder.add_challenges(self.betas);

        // 4. Add query index bits for each query.
        for bits in self.query_index_bits {
            debug_assert!(
                bits.len() <= F::bits(),
                "query index bit length exceeds limit"
            );

            builder.add_query_index_bits(bits);
        }

        // 5. Add commitment openings in a fixed layout.
        for opening in self.commitment_openings {
            // First add the commitment value itself.
            builder.add_challenge(opening.commitment);

            // Then, for each opened point, add (z, values at z).
            for (z, values) in opening.opened_points {
                builder.add_challenge(z);
                builder.add_proof_values(values);
            }
        }

        // Return the flattened public input vector.
        builder.build()
    }
}

/// Helper structure for constructing public inputs for full STARK verification.
///
/// The output is a vector of extension field elements fed to a verifier circuit.
pub struct StarkVerifierInputs<F, EF>
where
    F: Field + PrimeField64,
    EF: Field + BasedVectorSpace<F> + From<F>,
{
    /// Public input values for the AIR being verified, in the base field.
    ///
    /// These encode the statement being proven.
    pub air_public_values: Vec<F>,

    /// Values extracted from the proof in the extension field.
    ///
    /// These encodings usually include:
    /// - trace commitments,
    /// - opened values,
    /// - FRI-related data.
    pub proof_values: Vec<EF>,

    /// Values extracted from the preprocessed commitment, if present.
    ///
    /// These are also in the extension field.
    pub preprocessed: Vec<EF>,

    /// All Fiat–Shamir challenges used by the verifier.
    ///
    /// This typically includes:
    /// - batching challenge,
    /// - evaluation points for openings,
    /// - FRI folding challenges,
    /// - query indices.
    pub challenges: Vec<EF>,

    /// Number of FRI queries in the proof.
    pub num_queries: usize,
}

impl<F, EF> StarkVerifierInputs<F, EF>
where
    F: Field + PrimeField64,
    EF: Field + BasedVectorSpace<F> + From<F>,
{
    /// Flattens all STARK-related data into a single public input vector.
    ///
    /// # Canonical input order
    /// 1. AIR public values,
    /// 2. Proof values,
    /// 3. Preprocessed commitment values.
    ///
    /// # Returns
    /// A vector of extension field elements ready for circuit execution.
    pub fn build(self) -> Vec<EF> {
        // Use a builder over extension field elements.
        let mut builder = PublicInputBuilder::new();

        // 1. Lift AIR public values from base field to extension field.
        builder.add_proof_values(self.air_public_values.iter().map(|&v| v.into()));

        // 2. Add proof values (already in extension field).
        builder.add_proof_values(self.proof_values);

        // 3. Add preprocessed commitment values.
        builder.add_proof_values(self.preprocessed);

        builder.build()
    }
}

/// Constructs public inputs for a batch (multi-instance) STARK verification circuit.
///
/// Batch verification proves several AIR instances at once.
/// All instances share the same challenges, but each instance has its own AIR public values.
///
/// # Parameters
/// - `air_public_values`: For each instance, its AIR public values in `F`.
/// - `proof_values`: Values extracted from the batch proof in `EF`.
/// - `common_data`: Common data values.
///
/// # Returns
/// A vector of extension field elements suitable for a batch verifier circuit.
pub fn construct_batch_stark_verifier_inputs<F, EF>(
    air_public_values: &[Vec<F>],
    proof_values: &[EF],
    common_data: &[EF],
) -> Vec<EF>
where
    F: Field + PrimeField64,
    EF: Field + BasedVectorSpace<F> + From<F>,
{
    // Builder over extension field elements.
    let mut builder = PublicInputBuilder::new();

    // Add public values for each AIR instance, lifting from F to EF.
    for instance_pv in air_public_values {
        builder.add_proof_values(instance_pv.iter().map(|&v| v.into()));
    }

    // Add proof values (already in extension field).
    builder.add_proof_values(proof_values.iter().copied());

    // Add common_data values.
    builder.add_proof_values(common_data.iter().copied());

    builder.build()
}

/// Two-phase builder for single-instance STARK verification circuits.
///
/// This builder ties together:
/// 1. Circuit construction phase:
///    - allocate public input targets in the circuit,
///    - record their structure inside this builder.
/// 2. Execution phase:
///    - pack concrete values into a public input vector in exactly the same
///      order as the allocation order.
///
/// The central invariant is that target allocation and value packing are
/// perfectly aligned in both order and shape.
///
/// # Usage Pattern
///
/// ```ignore
/// // Phase 1: Circuit building
/// let mut circuit = CircuitBuilder::new();
/// let verifier = StarkVerifierInputsBuilder::allocate(&mut circuit, &proof, pis.len());
/// verify_circuit(config, air, &mut circuit, &verifier.proof_targets, &verifier.air_public_targets, ...)?;
/// let built_circuit = circuit.build()?;
///
/// // Phase 2: Execution
/// let public_inputs = verifier.pack_public_values(&pis, &proof, &None);
/// runner.set_public_inputs(&public_inputs)?;
/// ```
pub struct StarkVerifierInputsBuilder<SC, Comm, OpeningProof>
where
    SC: StarkGenericConfig,
    Comm: Recursive<
            SC::Challenge,
            Input = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
        >,
    OpeningProof:
        Recursive<SC::Challenge, Input = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof>,
{
    /// Public input targets for the AIR public values.
    ///
    /// The element at index `i` corresponds to the `i`-th AIR public value.
    pub air_public_targets: Vec<crate::Target>,

    /// Targets representing the entire proof structure.
    ///
    /// This includes trace commitments, opened evaluations, and FRI data.
    pub proof_targets: ProofTargets<SC, Comm, OpeningProof>,

    /// Target representation of the preprocessed commitment, if present.
    ///
    /// `None` when the AIR has no preprocessed columns.
    pub preprocessed_commit: Option<Comm>,
}

/// Type alias for the commitment type inside a STARK configuration.
///
/// This extracts the commitment type of the PCS used by the configuration.
type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;

impl<SC, Comm, OpeningProof> StarkVerifierInputsBuilder<SC, Comm, OpeningProof>
where
    SC: StarkGenericConfig,
    Comm: Recursive<
            SC::Challenge,
            Input = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
        >,
    OpeningProof:
        Recursive<SC::Challenge, Input = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof>,
{
    /// Allocates all public input targets during circuit construction.
    ///
    /// This method only inspects the *shape* of the proof and preprocessed
    /// commitment. It does not depend on any concrete values.
    ///
    /// # Parameters
    /// - `circuit`: Circuit builder in which targets are allocated.
    /// - `proof`: A reference proof used solely to determine the target shape.
    /// - `preprocessed_commit`: Optional reference preprocessed commitment.
    /// - `num_air_public_inputs`: Number of AIR public inputs.
    ///
    /// # Returns
    /// A `StarkVerifierInputsBuilder` that remembers the allocation layout.
    pub fn allocate(
        circuit: &mut CircuitBuilder<SC::Challenge>,
        proof: &Proof<SC>,
        preprocessed_commit: Option<&Com<SC>>,
        num_air_public_inputs: usize,
    ) -> Self {
        // Allocate public input targets for the AIR public values.
        //
        // These targets come first in the overall public input ordering.
        let air_public_targets = (0..num_air_public_inputs)
            .map(|_| circuit.public_input())
            .collect();

        // Allocate targets for all proof components based on the reference proof.
        let proof_targets = ProofTargets::new(circuit, proof);

        // Allocate targets for the preprocessed commitment if it exists.
        //
        // The commitment wrapper knows how to mirror the structure as targets.
        let preprocessed_commit = preprocessed_commit
            .as_ref()
            .map(|prep_comm| Comm::new(circuit, prep_comm));

        Self {
            air_public_targets,
            proof_targets,
            preprocessed_commit,
        }
    }

    /// Packs concrete values into public inputs in the canonical order.
    ///
    /// This must be called in the execution phase, after proof data is known.
    ///
    /// The output vector is consistent with the target allocation order used in `allocate`.
    ///
    /// # Parameters
    /// - `air_public_values`: AIR public values in the base field.
    /// - `proof`: Actual proof whose values are extracted.
    /// - `preprocessed_commit`: Actual preprocessed commitment, if any.
    ///
    /// # Returns
    /// A public input vector ready to be passed to the verifier circuit.
    pub fn pack_public_values(
        &self,
        air_public_values: &[Val<SC>],
        proof: &Proof<SC>,
        preprocessed_commit: &Option<Com<SC>>,
    ) -> Vec<SC::Challenge>
    where
        Val<SC>: PrimeField64,
        SC::Challenge: BasedVectorSpace<Val<SC>> + From<Val<SC>>,
    {
        // Extract extension-field values from the proof structure.
        //
        // The internal order is compatible with the way the proof targets were created.
        let proof_values = ProofTargets::<SC, Comm, OpeningProof>::get_values(proof);

        // Extract values from the preprocessed commitment, if it exists.
        //
        // If there is no preprocessed commitment, use an empty vector.
        let preprocessed = preprocessed_commit
            .as_ref()
            .map_or_else(Vec::new, |prep_comm| Comm::get_values(prep_comm));

        // Combine all components into a single public input vector.
        StarkVerifierInputs {
            air_public_values: air_public_values.to_vec(),
            proof_values: proof_values.to_vec(),
            preprocessed: preprocessed.to_vec(),
            challenges: Vec::new(),
            num_queries: 0,
        }
        .build()
    }

    /// Pack private input values (opened values, FRI siblings, etc.) for the verifier circuit.
    pub fn pack_private_values(&self, proof: &Proof<SC>) -> Vec<SC::Challenge>
    where
        Val<SC>: PrimeField64,
        SC::Challenge: BasedVectorSpace<Val<SC>> + From<Val<SC>>,
    {
        ProofTargets::<SC, Comm, OpeningProof>::get_private_values(proof)
    }

    /// Pack both public and private input values for the verifier circuit.
    pub fn pack_values(
        &self,
        air_public_values: &[Val<SC>],
        proof: &Proof<SC>,
        preprocessed_commit: &Option<Com<SC>>,
    ) -> (Vec<SC::Challenge>, Vec<SC::Challenge>)
    where
        Val<SC>: PrimeField64,
        SC::Challenge: BasedVectorSpace<Val<SC>> + From<Val<SC>>,
    {
        let public_values = self.pack_public_values(air_public_values, proof, preprocessed_commit);
        let private_values = self.pack_private_values(proof);

        (public_values, private_values)
    }
}

/// Two-phase builder for batch (multi-instance) STARK verification circuits.
///
/// This is the batch analogue of the single-instance inputs builder.
///
/// It manages:
/// - per-instance AIR public input targets,
/// - batch proof targets,
///
/// and later packs concrete values into a flat public input vector.
pub struct BatchStarkVerifierInputsBuilder<SC, Comm, OpeningProof>
where
    SC: StarkGenericConfig,
    Comm: Recursive<
            SC::Challenge,
            Input = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
        >,
    OpeningProof:
        Recursive<SC::Challenge, Input = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof>,
{
    /// Per-instance public input targets.
    ///
    /// `air_public_targets[i]` is the list of public input targets for instance `i`.
    pub air_public_targets: Vec<Vec<crate::Target>>,

    /// Targets representing the batch proof structure.
    ///
    /// This includes all commitments and openings for all instances.
    pub proof_targets: BatchProofTargets<SC, Comm, OpeningProof>,
    /// Allocated common data targets (if any).
    pub common_data: CommonDataTargets<SC, Comm>,
}

impl<SC, Comm, OpeningProof> BatchStarkVerifierInputsBuilder<SC, Comm, OpeningProof>
where
    SC: StarkGenericConfig,
    Comm: Recursive<
            SC::Challenge,
            Input = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
        >,
    OpeningProof:
        Recursive<SC::Challenge, Input = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof>,
{
    /// Allocates public input targets for batch verification.
    ///
    /// # Parameters
    /// - `circuit`: Circuit builder into which targets are allocated.
    /// - `proof`: Reference batch proof used only for its structure.
    /// - `air_public_counts`: Number of public inputs per AIR instance.
    ///
    /// # Panics
    /// Panics if `air_public_counts.len()` does not match the number of instances in the batch proof.
    pub fn allocate(
        circuit: &mut CircuitBuilder<SC::Challenge>,
        proof: &BatchProof<SC>,
        common_data: &CommonData<SC>,
        air_public_counts: &[usize],
    ) -> Self {
        // Ensure we have one public count per instance.
        assert_eq!(
            air_public_counts.len(),
            proof.opened_values.instances.len(),
            "public input count must match number of instances"
        );

        // For each instance, allocate `count` public input targets.
        let air_public_targets = air_public_counts
            .iter()
            .map(|&count| (0..count).map(|_| circuit.public_input()).collect())
            .collect();

        // Allocate targets for the batch proof structure, based on the reference proof.
        let proof_targets = BatchProofTargets::new(circuit, proof);

        let common_data = CommonDataTargets::<SC, Comm>::new(circuit, common_data);

        Self {
            air_public_targets,
            proof_targets,
            common_data,
        }
    }

    /// Packs concrete values into public inputs for batch verification.
    ///
    /// # Parameters
    /// - `air_public_values`: AIR public values for each instance.
    /// - `proof`: Actual batch proof.
    /// - `common`: Common data for the batch proof.
    ///
    /// # Returns
    /// A flattened public input vector ready for the batch verifier circuit.
    pub fn pack_public_values(
        &self,
        air_public_values: &[Vec<Val<SC>>],
        proof: &BatchProof<SC>,
        common: &CommonData<SC>,
    ) -> Vec<SC::Challenge>
    where
        Val<SC>: PrimeField64,
        SC::Challenge: BasedVectorSpace<Val<SC>> + From<Val<SC>>,
    {
        // Extract extension-field values from the batch proof structure.
        //
        // The internal order matches the structure used when targets were created.
        let common_data = CommonDataTargets::<SC, Comm>::get_values(common);
        let proof_values = BatchProofTargets::<SC, Comm, OpeningProof>::get_values(proof);

        // Combine AIR public values and proof values into a single public input vector.
        construct_batch_stark_verifier_inputs(air_public_values, &proof_values, &common_data)
    }

    /// Pack private input values (opened values, FRI siblings, etc.) for the verifier circuit.
    pub fn pack_private_values(&self, proof: &BatchProof<SC>) -> Vec<SC::Challenge>
    where
        Val<SC>: PrimeField64,
        SC::Challenge: BasedVectorSpace<Val<SC>> + From<Val<SC>>,
    {
        BatchProofTargets::<SC, Comm, OpeningProof>::get_private_values(proof)
    }

    /// Pack both public and private input values for the verifier circuit.
    pub fn pack_values(
        &self,
        air_public_values: &[Vec<Val<SC>>],
        proof: &BatchProof<SC>,
        common: &CommonData<SC>,
    ) -> (Vec<SC::Challenge>, Vec<SC::Challenge>)
    where
        Val<SC>: PrimeField64,
        SC::Challenge: BasedVectorSpace<Val<SC>> + From<Val<SC>>,
    {
        let public_values = self.pack_public_values(air_public_values, proof, common);
        let private_values = self.pack_private_values(proof);

        (public_values, private_values)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_public_input_builder() {
        let mut builder = PublicInputBuilder::<BabyBear>::new();

        // Initially empty.
        assert_eq!(builder.len(), 0);
        assert!(builder.is_empty());

        // Add various values.
        builder
            .add_proof_values([BabyBear::from_u32(1), BabyBear::from_u32(2)])
            .add_challenge(BabyBear::from_u32(3))
            .add_challenges([BabyBear::from_u32(4), BabyBear::from_u32(5)]);

        // Verify accumulated count.
        assert_eq!(builder.len(), 5);
        assert!(!builder.is_empty());

        // Build and verify contents.
        let inputs = builder.build();
        assert_eq!(inputs.len(), 5);
        assert_eq!(inputs[0], BabyBear::from_u32(1));
        assert_eq!(inputs[4], BabyBear::from_u32(5));
    }

    #[test]
    fn test_query_index_bit_decomposition() {
        let mut builder = PublicInputBuilder::<BabyBear>::new();

        // Index 5 = 0b101 in binary.
        builder.add_query_index(BabyBear::from_u32(5));

        let inputs = builder.build();

        // Should have exactly `F::bits()` bits.
        assert_eq!(inputs.len(), BabyBear::bits());

        // Verify little-endian bit pattern: 101 means bits are [1, 0, 1, 0, 0, ...].
        assert_eq!(inputs[0], BabyBear::ONE); // bit 0 (LSB)
        assert_eq!(inputs[1], BabyBear::ZERO); // bit 1
        assert_eq!(inputs[2], BabyBear::ONE); // bit 2

        // Remaining bits should all be zero.
        for &bit in &inputs[3..] {
            assert_eq!(bit, BabyBear::ZERO);
        }
    }

    /// Strategy for generating random BabyBear field elements.
    fn field_element() -> impl Strategy<Value = BabyBear> {
        any::<u32>().prop_map(BabyBear::from_u32)
    }

    proptest! {
        #[test]
        fn build_preserves_order(vals in prop::collection::vec(field_element(), 1..20)) {
            let mut builder = PublicInputBuilder::<BabyBear>::new();
            builder.add_proof_values(vals.clone());

            let result = builder.build();

            // Length must match.
            prop_assert_eq!(result.len(), vals.len());

            // Each element must be in the same position.
            for (i, &val) in vals.iter().enumerate() {
                prop_assert_eq!(result[i], val);
            }
        }

        #[test]
        fn chaining_preserves_order(
            vals1 in prop::collection::vec(field_element(), 1..10),
            challenge in field_element(),
            vals2 in prop::collection::vec(field_element(), 1..10)
        ) {
            let mut builder = PublicInputBuilder::<BabyBear>::new();

            // Chain multiple additions.
            builder
                .add_proof_values(vals1.clone())
                .add_challenge(challenge)
                .add_challenges(vals2.clone());

            let result = builder.build();

            // Verify total length.
            let expected_len = vals1.len() + 1 + vals2.len();
            prop_assert_eq!(result.len(), expected_len);

            // Verify vals1 appears first, in order.
            for (i, &val) in vals1.iter().enumerate() {
                prop_assert_eq!(result[i], val, "vals1 order");
            }

            // Verify challenge appears after vals1.
            prop_assert_eq!(result[vals1.len()], challenge, "challenge position");

            // Verify vals2 appears after challenge, in order.
            for (i, &val) in vals2.iter().enumerate() {
                prop_assert_eq!(result[vals1.len() + 1 + i], val, "vals2 order");
            }
        }
    }
}
