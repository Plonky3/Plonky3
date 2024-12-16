use p3_air::{Air, AirBuilder, BaseAir};
use p3_blake3_air::Blake3Air;
use p3_field::{Field, PrimeField64};
use p3_keccak_air::KeccakAir;
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use p3_poseidon2_air::VectorizedPoseidon2Air;
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

impl<
        F: PrimeField64,
        LinearLayers: GenericPoseidon2LinearLayers<F, WIDTH>,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
        const VECTOR_LEN: usize,
    >
    ProofObjective<
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
    pub fn generate_trace_rows(&self, num_hashes: usize) -> RowMajorMatrix<F>
    where
        Standard: Distribution<F>,
    {
        match self {
            ProofObjective::Blake3(ref b3_air) => b3_air.generate_trace_rows(num_hashes),
            ProofObjective::Poseidon2(ref p2_air) => {
                p2_air.generate_vectorized_trace_rows(num_hashes)
            }
            ProofObjective::Keccak(ref k_air) => k_air.generate_trace_rows(num_hashes),
        }
    }
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
            ProofObjective::Blake3(ref b3_air) => <Blake3Air as BaseAir<F>>::width(b3_air),
            ProofObjective::Poseidon2(ref p2_air) => p2_air.width(),
            ProofObjective::Keccak(ref k_air) => <KeccakAir as BaseAir<F>>::width(k_air),
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
            ProofObjective::Blake3(ref b3_air) => b3_air.eval(builder),
            ProofObjective::Poseidon2(ref p2_air) => p2_air.eval(builder),
            ProofObjective::Keccak(ref k_air) => k_air.eval(builder),
        }
    }
}
