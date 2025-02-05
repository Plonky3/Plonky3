use p3_air::{Air, AirBuilder, BaseAir};
use p3_blake3_air::Blake3Air;
use p3_challenger::FieldChallenger;
use p3_commit::PolynomialSpace;
use p3_field::{ExtensionField, Field, PrimeField64};
use p3_keccak_air::KeccakAir;
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use p3_poseidon2_air::VectorizedPoseidon2Air;
use p3_uni_stark::{
    DebugConstraintBuilder, ProverConstraintFolder, StarkGenericConfig, SymbolicAirBuilder,
    SymbolicExpression, VerifierConstraintFolder,
};
use rand::distributions::Standard;
use rand::prelude::Distribution;

/// An enum containing the three different AIR's.
///
/// This implements `AIR` by passing to whatever the contained struct is.
pub enum ProofObjective<
    F: Field,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> {
    Blake3(Blake3Air),
    Keccak(KeccakAir),
    Poseidon2(
        VectorizedPoseidon2Air<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            VECTOR_LEN,
        >,
    ),
}

/// An AIR for a hash function used for example proofs and benchmarking.
///
/// A key feature is the ability to randomly generate a trace which proves
/// the output of some number of hashes using a given hash function.
pub trait ExampleHashAir<F: Field, SC: StarkGenericConfig>:
    BaseAir<F>
    + for<'a> Air<DebugConstraintBuilder<'a, F>>
    + Air<SymbolicAirBuilder<F>>
    + for<'a> Air<ProverConstraintFolder<'a, SC>>
    + for<'a> Air<VerifierConstraintFolder<'a, SC>>
{
    fn generate_trace_rows(
        &self,
        num_hashes: usize,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F>
    where
        Standard: Distribution<F>;
}

impl<
        F: Field,
        LinearLayers: Sync,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
        const VECTOR_LEN: usize,
    > BaseAir<F>
    for ProofObjective<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    #[inline]
    fn width(&self) -> usize {
        match self {
            ProofObjective::Blake3(b3_air) => <Blake3Air as BaseAir<F>>::width(b3_air),
            ProofObjective::Poseidon2(p2_air) => p2_air.width(),
            ProofObjective::Keccak(k_air) => <KeccakAir as BaseAir<F>>::width(k_air),
        }
    }
}

impl<
        AB: AirBuilder,
        LinearLayers: GenericPoseidon2LinearLayers<AB::Expr, WIDTH>,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
        const VECTOR_LEN: usize,
    > Air<AB>
    for ProofObjective<
        AB::F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        match self {
            ProofObjective::Blake3(b3_air) => b3_air.eval(builder),
            ProofObjective::Poseidon2(p2_air) => p2_air.eval(builder),
            ProofObjective::Keccak(k_air) => k_air.eval(builder),
        }
    }
}

impl<
        F: PrimeField64,
        Domain: PolynomialSpace<Val = F>,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
        Pcs: p3_commit::Pcs<EF, Challenger, Domain = Domain>,
        SC: StarkGenericConfig<Pcs = Pcs, Challenge = EF, Challenger = Challenger>,
        LinearLayers: GenericPoseidon2LinearLayers<F, WIDTH>
            + GenericPoseidon2LinearLayers<SymbolicExpression<F>, WIDTH>
            + GenericPoseidon2LinearLayers<<F as Field>::Packing, WIDTH>
            + GenericPoseidon2LinearLayers<EF, WIDTH>,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
        const VECTOR_LEN: usize,
    > ExampleHashAir<F, SC>
    for ProofObjective<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    #[inline]
    fn generate_trace_rows(
        &self,
        num_hashes: usize,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F>
    where
        Standard: Distribution<F>,
    {
        match self {
            ProofObjective::Blake3(b3_air) => {
                b3_air.generate_trace_rows(num_hashes, extra_capacity_bits)
            }
            ProofObjective::Poseidon2(p2_air) => {
                p2_air.generate_vectorized_trace_rows(num_hashes, extra_capacity_bits)
            }
            ProofObjective::Keccak(k_air) => {
                k_air.generate_trace_rows(num_hashes, extra_capacity_bits)
            }
        }
    }
}
