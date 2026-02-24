//! Wrapper structs for the Poseidon1 internal/external layers on MontyField31.
//!
//! Used when no SIMD architecture (AVX2/AVX512/NEON) is available.

use core::marker::PhantomData;

use p3_poseidon::external::{FullRoundConstants, FullRoundLayerConstructor};
use p3_poseidon::internal::{PartialRoundConstants, PartialRoundLayerConstructor};

use crate::{FieldParameters, MontyField31, MontyParameters, PartialRoundBaseParameters};

/// The internal (partial round) layer of the Poseidon1 permutation for Monty31 fields.
#[derive(Debug, Clone)]
pub struct Poseidon1InternalLayerMonty31<
    MP: MontyParameters,
    const WIDTH: usize,
    ILP: PartialRoundBaseParameters<MP, WIDTH>,
> {
    pub(crate) internal_constants: PartialRoundConstants<MontyField31<MP>, WIDTH>,
    _phantom: PhantomData<ILP>,
}

/// The external (full round) layer of the Poseidon1 permutation for Monty31 fields.
#[derive(Debug, Clone)]
pub struct Poseidon1ExternalLayerMonty31<MP: MontyParameters, const WIDTH: usize> {
    pub(crate) external_constants: FullRoundConstants<MontyField31<MP>, WIDTH>,
}

impl<FP: FieldParameters, const WIDTH: usize, ILP: PartialRoundBaseParameters<FP, WIDTH>>
    PartialRoundLayerConstructor<MontyField31<FP>, WIDTH>
    for Poseidon1InternalLayerMonty31<FP, WIDTH, ILP>
{
    fn new_from_constants(
        internal_constants: PartialRoundConstants<MontyField31<FP>, WIDTH>,
    ) -> Self {
        Self {
            internal_constants,
            _phantom: PhantomData,
        }
    }
}

impl<FP: FieldParameters, const WIDTH: usize> FullRoundLayerConstructor<MontyField31<FP>, WIDTH>
    for Poseidon1ExternalLayerMonty31<FP, WIDTH>
{
    fn new_from_constants(external_constants: FullRoundConstants<MontyField31<FP>, WIDTH>) -> Self {
        Self { external_constants }
    }
}
