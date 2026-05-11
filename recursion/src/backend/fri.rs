//! FRI PCS backend for the unified recursion API.

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::{format, vec};

use p3_circuit::{CircuitBuilder, CircuitRunner, NonPrimitiveOpId};
use p3_circuit_prover::batch_stark_prover::{
    poseidon2_air_builders, poseidon2_air_builders_d5, poseidon2_preprocessor,
    poseidon2_table_provers_d5, recompose_air_builders, recompose_preprocessor,
};
use p3_circuit_prover::common::{NpoAirBuilder, NpoPreprocessor};
use p3_circuit_prover::config::StarkField;
use p3_circuit_prover::field_params::ExtractBinomialW;
use p3_circuit_prover::{
    ConstraintProfile, Poseidon2Preprocessor, Poseidon2Prover, Poseidon2ProverD2,
    RecomposePreprocessor, TableProver, recompose_table_provers,
};
use p3_commit::Pcs;
use p3_field::extension::BinomiallyExtendable;
use p3_field::{Algebra, BasedVectorSpace, ExtensionField, PrimeCharacteristicRing, PrimeField64};
use p3_lookup::logup::LogUpGadget;
use p3_uni_stark::{StarkGenericConfig, SymbolicExpressionExt, Val};

use crate::ops::Poseidon2Config;
use crate::public_inputs::{BatchStarkVerifierInputsBuilder, StarkVerifierInputsBuilder};
use crate::recursion::{PcsRecursionBackend, RecursionInput, VerifierCircuitResult};
use crate::traits::RecursiveAir;
use crate::verifier::{
    ObservableCommitment, VerificationError, verify_p3_batch_proof_circuit,
    verify_p3_uni_proof_circuit,
};
use crate::{ChallengerPermConfig, Recursive, RecursivePcs};

/// Config that uses FRI with Merkle-tree MMCS and fixed constants (WIDTH, RATE, DIGEST_ELEMS).
/// Implement this for your StarkConfig to use [`FriRecursionBackend`].
pub trait FriRecursionConfig: StarkGenericConfig + Sized
where
    Self::Pcs: RecursivePcs<
            Self,
            Self::InputProof,
            Self::OpeningProof,
            Self::Commitment,
            <Self::Pcs as Pcs<Self::Challenge, Self::Challenger>>::Domain,
        >,
{
    /// Commitment type used in the verifier circuit (e.g. HashTargets).
    type Commitment: Recursive<
            Self::Challenge,
            Input = <Self::Pcs as Pcs<Self::Challenge, Self::Challenger>>::Commitment,
        > + Clone
        + ObservableCommitment;

    /// Input proof type for the PCS (e.g. batch opening targets for FRI).
    type InputProof: Recursive<Self::Challenge>;

    /// Opening proof type used in the verifier circuit (e.g. FRI proof targets).
    type OpeningProof: Recursive<
            Self::Challenge,
            Input = <Self::Pcs as Pcs<Self::Challenge, Self::Challenger>>::Proof,
        >;

    /// Raw FRI opening proof type (value type, not circuit targets). Used to set private data.
    type RawOpeningProof;

    /// Number of field elements in a single Merkle digest (e.g. 8 for BabyBear with Poseidon2).
    const DIGEST_ELEMS: usize;

    /// Invoke a closure with the FRI opening proof extracted from the recursion input.
    fn with_fri_opening_proof<'a, A, R>(
        prev: &RecursionInput<'a, Self, A>,
        f: impl FnOnce(&Self::RawOpeningProof) -> R,
    ) -> R
    where
        A: RecursiveAir<Val<Self>, Self::Challenge, LogUpGadget>;

    /// Prepare the circuit for verification (e.g. enable challenger permutation and NPOs). Called by the backend before building the verifier.
    fn prepare_circuit_for_verification(
        &self,
        circuit: &mut CircuitBuilder<Self::Challenge>,
    ) -> Result<(), VerificationError>;

    /// Return the PCS verifier params (e.g. FRI params). The config must hold these and return a reference.
    #[allow(clippy::type_complexity)]
    fn pcs_verifier_params(
        &self,
    ) -> &<Self::Pcs as RecursivePcs<
        Self,
        Self::InputProof,
        Self::OpeningProof,
        Self::Commitment,
        <Self::Pcs as Pcs<Self::Challenge, Self::Challenger>>::Domain,
    >>::VerifierParams;

    /// Set FRI Merkle path private data on the runner. Implement by calling
    /// [`crate::pcs::set_fri_mmcs_private_data`] with your concrete MMCS/hasher types.
    fn set_fri_private_data(
        runner: &mut CircuitRunner<'_, Self::Challenge>,
        op_ids: &[NonPrimitiveOpId],
        opening_proof: &Self::RawOpeningProof,
    ) -> Result<(), &'static str>;
}

/// FRI-based recursion backend, holding the challenger permutation config.
/// The verifier params come from the config via [`FriRecursionConfig::pcs_verifier_params`].
/// `WIDTH` and `RATE` are the permutation circuit parameters (typically 16 and 8).
// TODO: Make this generic over the challenger permutation config.
#[derive(Clone)]
pub struct FriRecursionBackend<const WIDTH: usize = 16, const RATE: usize = 8> {
    /// Poseidon2 configuration used for the Fiat-Shamir challenger permutation circuit.
    pub challenger_perm_config: Poseidon2Config,
    /// Number of recompose operations packed per AIR row.
    ///
    /// Increasing this reduces the recompose table height proportionally.
    /// Must be kept in sync between prover and verifier. Defaults to 1.
    pub recompose_lanes: usize,
}

impl<const WIDTH: usize, const RATE: usize> FriRecursionBackend<WIDTH, RATE> {
    /// Create a new backend with the given challenger permutation configuration.
    pub const fn new(challenger_perm_config: Poseidon2Config) -> Self {
        Self {
            challenger_perm_config,
            recompose_lanes: 1,
        }
    }

    /// Override the number of recompose operations packed per AIR row.
    pub const fn with_recompose_lanes(mut self, lanes: usize) -> Self {
        self.recompose_lanes = if lanes < 1 { 1 } else { lanes };
        self
    }

    /// Tag this backend for a fixed batch/extension degree `D` (typically `2` or `4`).
    pub const fn for_extension_degree<const D: usize>(
        self,
    ) -> FriRecursionBackendForExt<D, WIDTH, RATE> {
        FriRecursionBackendForExt(self)
    }

    /// For KoalaBear quintic extension (`D = 5`). Use when `SC::Challenge` is
    /// `QuinticTrinomialExtensionField<KoalaBear>`.
    ///
    /// # Panics
    ///
    /// Panics if `challenger_perm_config.d() != 1`. The quintic challenger operates
    /// entirely in the base field, so the Poseidon2 config must be a D=1 variant
    /// (e.g. `KoalaBearD1Width16`). Passing a D=4 config would cause a dimension
    /// mismatch when the trace generator copies 5-element quintic coefficients into
    /// 4-slot base-field state slices.
    pub const fn new_d5(
        challenger_perm_config: Poseidon2Config,
    ) -> FriRecursionBackendD5<WIDTH, RATE> {
        assert!(
            challenger_perm_config.d() == 1,
            "new_d5 requires a D=1 (base-field) Poseidon2 config; \
             the quintic challenger operates in the base field"
        );
        FriRecursionBackendD5(Self::new(challenger_perm_config))
    }
}

/// FRI recursion backend tagged with batch/extension field degree `D` (e.g. `2` or `4`).
#[derive(Clone)]
pub struct FriRecursionBackendForExt<const D: usize, const WIDTH: usize = 16, const RATE: usize = 8>(
    /// The inner backend holding the challenger permutation config.
    pub(crate) FriRecursionBackend<WIDTH, RATE>,
);

/// FRI backend for KoalaBear quintic extension (`D = 5`).
#[derive(Clone)]
pub struct FriRecursionBackendD5<const WIDTH: usize = 16, const RATE: usize = 8>(
    /// The inner backend holding the challenger permutation config.
    pub(crate) FriRecursionBackend<WIDTH, RATE>,
);

/// Verifier result from the FRI backend: either uni-stark or batch-stark builder + op_ids.
pub enum FriVerifierResult<SC>
where
    SC: FriRecursionConfig,
    SC::Pcs: RecursivePcs<
            SC,
            SC::InputProof,
            SC::OpeningProof,
            SC::Commitment,
            <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain,
        >,
{
    /// Result for a single-instance (uni-STARK) input proof.
    UniStark(
        StarkVerifierInputsBuilder<SC, SC::Commitment, SC::OpeningProof>,
        Vec<NonPrimitiveOpId>,
    ),
    /// Result for a batch-STARK input proof.
    BatchStark(
        BatchStarkVerifierInputsBuilder<SC, SC::Commitment, SC::OpeningProof>,
        Vec<NonPrimitiveOpId>,
    ),
}

impl<SC, A> VerifierCircuitResult<SC, A> for FriVerifierResult<SC>
where
    SC: FriRecursionConfig,
    SC::Pcs: RecursivePcs<
            SC,
            SC::InputProof,
            SC::OpeningProof,
            SC::Commitment,
            <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain,
        >,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    Val<SC>: PrimeField64,
    SC::Challenge: BasedVectorSpace<Val<SC>> + From<Val<SC>>,
{
    fn pack_public_inputs(
        &self,
        prev: &RecursionInput<'_, SC, A>,
    ) -> Result<Vec<SC::Challenge>, VerificationError> {
        match (self, prev) {
            (
                Self::UniStark(builder, _),
                RecursionInput::UniStark {
                    proof,
                    public_inputs,
                    preprocessed_commit,
                    ..
                },
            ) => Ok(builder.pack_public_values(public_inputs, proof, preprocessed_commit)),
            (
                Self::BatchStark(builder, _),
                RecursionInput::BatchStark {
                    proof,
                    common_data,
                    table_public_inputs,
                },
            ) => Ok(builder.pack_public_values(table_public_inputs, &proof.proof, common_data)),
            _ => Err(VerificationError::InvalidProofShape(
                "RecursionInput variant does not match verifier result".to_string(),
            )),
        }
    }

    fn pack_private_inputs(
        &self,
        prev: &RecursionInput<'_, SC, A>,
    ) -> Result<Vec<SC::Challenge>, VerificationError> {
        match (self, prev) {
            (Self::UniStark(builder, _), RecursionInput::UniStark { proof, .. }) => {
                Ok(builder.pack_private_values(proof))
            }
            (Self::BatchStark(builder, _), RecursionInput::BatchStark { proof, .. }) => {
                Ok(builder.pack_private_values(&proof.proof))
            }
            _ => Err(VerificationError::InvalidProofShape(
                "RecursionInput variant does not match verifier result".to_string(),
            )),
        }
    }

    fn op_ids(&self) -> &[NonPrimitiveOpId] {
        match self {
            Self::UniStark(_, ids) | Self::BatchStark(_, ids) => ids,
        }
    }
}

fn build_verifier_circuit_impl<SC, A, const WIDTH: usize, const RATE: usize>(
    backend: &FriRecursionBackend<WIDTH, RATE>,
    prev: &RecursionInput<'_, SC, A>,
    config: &SC,
    circuit: &mut CircuitBuilder<SC::Challenge>,
    non_primitive_provers: &[Box<dyn TableProver<SC>>],
) -> Result<FriVerifierResult<SC>, VerificationError>
where
    SC: FriRecursionConfig + Send + Sync + 'static,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    Val<SC>: PrimeField64,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + PrimeCharacteristicRing
        + ExtractBinomialW<Val<SC>>,
    <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain: Clone,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        From<p3_uni_stark::SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
    SC::Pcs: RecursivePcs<
            SC,
            SC::InputProof,
            SC::OpeningProof,
            SC::Commitment,
            <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain,
        >,
{
    match prev {
        RecursionInput::UniStark {
            proof,
            air,
            public_inputs,
            preprocessed_commit,
        } => {
            let verifier_inputs =
                StarkVerifierInputsBuilder::<SC, SC::Commitment, SC::OpeningProof>::allocate(
                    circuit,
                    proof,
                    preprocessed_commit.as_ref(),
                    public_inputs.len(),
                );
            let op_ids = verify_p3_uni_proof_circuit::<
                A,
                SC,
                SC::Commitment,
                SC::InputProof,
                SC::OpeningProof,
                _,
                WIDTH,
                RATE,
            >(
                config,
                air,
                circuit,
                &verifier_inputs.proof_targets,
                &verifier_inputs.air_public_targets,
                &verifier_inputs.preprocessed_commit,
                config.pcs_verifier_params(),
                backend.challenger_perm_config,
            )?;
            Ok(FriVerifierResult::UniStark(verifier_inputs, op_ids))
        }
        RecursionInput::BatchStark {
            proof,
            common_data,
            table_public_inputs: _,
        } => {
            let lookup_gadget = LogUpGadget::new();
            let (verifier_inputs, op_ids) = match proof.ext_degree {
                1 => verify_p3_batch_proof_circuit::<
                    SC,
                    SC::Commitment,
                    SC::InputProof,
                    SC::OpeningProof,
                    _,
                    _,
                    WIDTH,
                    RATE,
                    1,
                >(
                    config,
                    circuit,
                    proof,
                    config.pcs_verifier_params(),
                    common_data,
                    &lookup_gadget,
                    backend.challenger_perm_config,
                    non_primitive_provers,
                )?,
                2 => verify_p3_batch_proof_circuit::<
                    SC,
                    SC::Commitment,
                    SC::InputProof,
                    SC::OpeningProof,
                    _,
                    _,
                    WIDTH,
                    RATE,
                    2,
                >(
                    config,
                    circuit,
                    proof,
                    config.pcs_verifier_params(),
                    common_data,
                    &lookup_gadget,
                    backend.challenger_perm_config,
                    non_primitive_provers,
                )?,
                4 => verify_p3_batch_proof_circuit::<
                    SC,
                    SC::Commitment,
                    SC::InputProof,
                    SC::OpeningProof,
                    _,
                    _,
                    WIDTH,
                    RATE,
                    4,
                >(
                    config,
                    circuit,
                    proof,
                    config.pcs_verifier_params(),
                    common_data,
                    &lookup_gadget,
                    backend.challenger_perm_config,
                    non_primitive_provers,
                )?,
                5 => verify_p3_batch_proof_circuit::<
                    SC,
                    SC::Commitment,
                    SC::InputProof,
                    SC::OpeningProof,
                    _,
                    _,
                    WIDTH,
                    RATE,
                    5,
                >(
                    config,
                    circuit,
                    proof,
                    config.pcs_verifier_params(),
                    common_data,
                    &lookup_gadget,
                    backend.challenger_perm_config,
                    non_primitive_provers,
                )?,
                d => {
                    return Err(VerificationError::InvalidProofShape(format!(
                        "unsupported batch proof ext_degree {}",
                        d
                    )));
                }
            };
            Ok(FriVerifierResult::BatchStark(verifier_inputs, op_ids))
        }
    }
}

impl<SC, A, const WIDTH: usize, const RATE: usize> PcsRecursionBackend<SC, A, 2>
    for FriRecursionBackendForExt<2, WIDTH, RATE>
where
    SC: FriRecursionConfig + Send + Sync + 'static,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    Val<SC>: PrimeField64 + BinomiallyExtendable<2> + StarkField,
    Poseidon2Preprocessor: NpoPreprocessor<Val<SC>>,
    RecomposePreprocessor: NpoPreprocessor<Val<SC>>,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + PrimeCharacteristicRing
        + ExtractBinomialW<Val<SC>>,
    <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain: Clone,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        From<p3_uni_stark::SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
    SC::Pcs: RecursivePcs<
            SC,
            SC::InputProof,
            SC::OpeningProof,
            SC::Commitment,
            <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain,
        >,
{
    type VerifierResult = FriVerifierResult<SC>;

    fn prepare_circuit(
        &self,
        config: &SC,
        circuit: &mut CircuitBuilder<SC::Challenge>,
    ) -> Result<(), VerificationError> {
        config.prepare_circuit_for_verification(circuit)
    }

    fn build_verifier_circuit(
        &self,
        prev: &RecursionInput<'_, SC, A>,
        config: &SC,
        circuit: &mut CircuitBuilder<SC::Challenge>,
    ) -> Result<Self::VerifierResult, VerificationError> {
        let provers = match prev {
            RecursionInput::BatchStark { proof, .. } => {
                PcsRecursionBackend::<SC, A, 2>::non_primitive_provers(self, proof.ext_degree)
            }
            _ => Vec::new(),
        };
        build_verifier_circuit_impl(&self.0, prev, config, circuit, &provers)
    }

    fn set_private_data(
        &self,
        config: &SC,
        runner: &mut CircuitRunner<'_, SC::Challenge>,
        op_ids: &[NonPrimitiveOpId],
        prev: &RecursionInput<'_, SC, A>,
    ) -> Result<(), &'static str> {
        let _ = config;
        SC::with_fri_opening_proof(prev, |opening_proof| {
            SC::set_fri_private_data(runner, op_ids, opening_proof)
        })
    }

    fn challenger_perm_config(&self) -> Option<Box<dyn ChallengerPermConfig>> {
        Some(Box::new(self.0.challenger_perm_config))
    }

    fn non_primitive_preprocessors(&self) -> Vec<Box<dyn NpoPreprocessor<Val<SC>>>> {
        let cl = self.0.challenger_perm_config.d() != 2;
        vec![
            poseidon2_preprocessor::<Val<SC>>(),
            recompose_preprocessor::<Val<SC>>(cl),
        ]
    }

    fn non_primitive_provers(&self, ext_degree: usize) -> Vec<Box<dyn TableProver<SC>>> {
        if ext_degree == 2 {
            let cl = self.0.challenger_perm_config.d() != 2;
            let mut provers: Vec<Box<dyn TableProver<SC>>> = vec![Box::new(
                Poseidon2ProverD2::new(self.0.challenger_perm_config, ConstraintProfile::Standard),
            )];
            provers.extend(recompose_table_provers::<SC, 2>(self.0.recompose_lanes, cl));
            provers
        } else {
            Vec::new()
        }
    }

    fn non_primitive_air_builders(&self) -> Vec<Box<dyn NpoAirBuilder<SC, 2>>> {
        let cl = self.0.challenger_perm_config.d() != 2;
        let mut builders = poseidon2_air_builders::<SC, 2>();
        builders.extend(recompose_air_builders::<SC, 2>(self.0.recompose_lanes, cl));
        builders
    }
}

impl<SC, A, const WIDTH: usize, const RATE: usize> PcsRecursionBackend<SC, A, 4>
    for FriRecursionBackendForExt<4, WIDTH, RATE>
where
    SC: FriRecursionConfig + Send + Sync + 'static,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    Val<SC>: PrimeField64 + BinomiallyExtendable<4> + StarkField,
    Poseidon2Preprocessor: NpoPreprocessor<Val<SC>>,
    RecomposePreprocessor: NpoPreprocessor<Val<SC>>,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + PrimeCharacteristicRing
        + ExtractBinomialW<Val<SC>>,
    <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain: Clone,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        From<p3_uni_stark::SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
    SC::Pcs: RecursivePcs<
            SC,
            SC::InputProof,
            SC::OpeningProof,
            SC::Commitment,
            <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain,
        >,
{
    type VerifierResult = FriVerifierResult<SC>;

    fn prepare_circuit(
        &self,
        config: &SC,
        circuit: &mut CircuitBuilder<SC::Challenge>,
    ) -> Result<(), VerificationError> {
        config.prepare_circuit_for_verification(circuit)
    }

    fn build_verifier_circuit(
        &self,
        prev: &RecursionInput<'_, SC, A>,
        config: &SC,
        circuit: &mut CircuitBuilder<SC::Challenge>,
    ) -> Result<Self::VerifierResult, VerificationError> {
        let provers = match prev {
            RecursionInput::BatchStark { proof, .. } => {
                PcsRecursionBackend::<SC, A, 4>::non_primitive_provers(self, proof.ext_degree)
            }
            _ => Vec::new(),
        };
        build_verifier_circuit_impl(&self.0, prev, config, circuit, &provers)
    }

    fn set_private_data(
        &self,
        config: &SC,
        runner: &mut CircuitRunner<'_, SC::Challenge>,
        op_ids: &[NonPrimitiveOpId],
        prev: &RecursionInput<'_, SC, A>,
    ) -> Result<(), &'static str> {
        let _ = config;
        SC::with_fri_opening_proof(prev, |opening_proof| {
            SC::set_fri_private_data(runner, op_ids, opening_proof)
        })
    }

    fn challenger_perm_config(&self) -> Option<Box<dyn ChallengerPermConfig>> {
        Some(Box::new(self.0.challenger_perm_config))
    }

    fn non_primitive_preprocessors(&self) -> Vec<Box<dyn NpoPreprocessor<Val<SC>>>> {
        let cl = self.0.challenger_perm_config.d() != 4;
        vec![
            poseidon2_preprocessor::<Val<SC>>(),
            recompose_preprocessor::<Val<SC>>(cl),
        ]
    }

    fn non_primitive_provers(&self, ext_degree: usize) -> Vec<Box<dyn TableProver<SC>>> {
        if ext_degree == 4 {
            let cl = self.0.challenger_perm_config.d() != 4;
            let mut provers: Vec<Box<dyn TableProver<SC>>> = vec![Box::new(Poseidon2Prover::new(
                self.0.challenger_perm_config,
                ConstraintProfile::Standard,
            ))];
            provers.extend(recompose_table_provers::<SC, 4>(self.0.recompose_lanes, cl));
            provers
        } else {
            Vec::new()
        }
    }

    fn non_primitive_air_builders(&self) -> Vec<Box<dyn NpoAirBuilder<SC, 4>>> {
        let cl = self.0.challenger_perm_config.d() != 4;
        let mut builders = poseidon2_air_builders::<SC, 4>();
        builders.extend(recompose_air_builders::<SC, 4>(self.0.recompose_lanes, cl));
        builders
    }
}

impl<SC, A, const WIDTH: usize, const RATE: usize> PcsRecursionBackend<SC, A, 5>
    for FriRecursionBackendD5<WIDTH, RATE>
where
    SC: FriRecursionConfig + Send + Sync + 'static,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    Val<SC>: PrimeField64 + StarkField + BinomiallyExtendable<4>,
    Poseidon2Preprocessor: NpoPreprocessor<Val<SC>>,
    RecomposePreprocessor: NpoPreprocessor<Val<SC>>,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + PrimeCharacteristicRing
        + ExtractBinomialW<Val<SC>>,
    <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain: Clone,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        From<p3_uni_stark::SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
    SC::Pcs: RecursivePcs<
            SC,
            SC::InputProof,
            SC::OpeningProof,
            SC::Commitment,
            <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain,
        >,
{
    type VerifierResult = FriVerifierResult<SC>;

    fn prepare_circuit(
        &self,
        config: &SC,
        circuit: &mut CircuitBuilder<SC::Challenge>,
    ) -> Result<(), VerificationError> {
        config.prepare_circuit_for_verification(circuit)
    }

    fn build_verifier_circuit(
        &self,
        prev: &RecursionInput<'_, SC, A>,
        config: &SC,
        circuit: &mut CircuitBuilder<SC::Challenge>,
    ) -> Result<Self::VerifierResult, VerificationError> {
        let provers = match prev {
            RecursionInput::BatchStark { proof, .. } => {
                PcsRecursionBackend::<SC, A, 5>::non_primitive_provers(self, proof.ext_degree)
            }
            _ => Vec::new(),
        };
        build_verifier_circuit_impl(&self.0, prev, config, circuit, &provers)
    }

    fn set_private_data(
        &self,
        config: &SC,
        runner: &mut CircuitRunner<'_, SC::Challenge>,
        op_ids: &[NonPrimitiveOpId],
        prev: &RecursionInput<'_, SC, A>,
    ) -> Result<(), &'static str> {
        let _ = config;
        SC::with_fri_opening_proof(prev, |opening_proof| {
            SC::set_fri_private_data(runner, op_ids, opening_proof)
        })
    }

    fn challenger_perm_config(&self) -> Option<Box<dyn ChallengerPermConfig>> {
        Some(Box::new(self.0.challenger_perm_config))
    }

    fn non_primitive_preprocessors(&self) -> Vec<Box<dyn NpoPreprocessor<Val<SC>>>> {
        let cl = self.0.challenger_perm_config.d() != 5;
        vec![
            poseidon2_preprocessor::<Val<SC>>(),
            recompose_preprocessor::<Val<SC>>(cl),
        ]
    }

    fn non_primitive_provers(&self, ext_degree: usize) -> Vec<Box<dyn TableProver<SC>>> {
        if ext_degree == 5 {
            let cl = self.0.challenger_perm_config.d() != 5;
            let mut provers = poseidon2_table_provers_d5(self.0.challenger_perm_config);
            provers.extend(recompose_table_provers::<SC, 5>(self.0.recompose_lanes, cl));
            provers
        } else {
            Vec::new()
        }
    }

    fn non_primitive_air_builders(&self) -> Vec<Box<dyn NpoAirBuilder<SC, 5>>> {
        let cl = self.0.challenger_perm_config.d() != 5;
        let mut builders = poseidon2_air_builders_d5();
        builders.extend(recompose_air_builders::<SC, 5>(self.0.recompose_lanes, cl));
        builders
    }
}
