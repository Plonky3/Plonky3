//! NEON-optimized Poseidon layers for MontyField31.
//!
//! Currently delegates to the scalar implementation. SIMD optimization of the
//! sparse matrix multiplication is a future enhancement.

use core::marker::PhantomData;

use p3_poseidon::external::{FullRoundConstants, FullRoundLayerConstructor};
use p3_poseidon::internal::{PartialRoundConstants, PartialRoundLayerConstructor};

use crate::{FieldParameters, MontyField31, PackedMontyParameters, PartialRoundBaseParameters};

/// NEON-specific trait for Poseidon partial round parameters.
pub trait PartialRoundParametersNeon<PMP: PackedMontyParameters, const WIDTH: usize>:
    Clone + Sync
{
}

/// The internal (partial round) layer of Poseidon for NEON-packed MontyField31.
#[derive(Debug, Clone)]
pub struct PoseidonInternalLayerMonty31<
    PMP: PackedMontyParameters,
    const WIDTH: usize,
    ILP: PartialRoundBaseParameters<PMP, WIDTH>,
> {
    pub(crate) internal_constants: PartialRoundConstants<MontyField31<PMP>, WIDTH>,
    _phantom: PhantomData<ILP>,
}

/// The external (full round) layer of Poseidon for NEON-packed MontyField31.
#[derive(Debug, Clone)]
pub struct PoseidonExternalLayerMonty31<PMP: PackedMontyParameters, const WIDTH: usize> {
    pub(crate) external_constants: FullRoundConstants<MontyField31<PMP>, WIDTH>,
}

impl<FP: FieldParameters, const WIDTH: usize, ILP: PartialRoundBaseParameters<FP, WIDTH>>
    PartialRoundLayerConstructor<MontyField31<FP>, WIDTH>
    for PoseidonInternalLayerMonty31<FP, WIDTH, ILP>
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
    for PoseidonExternalLayerMonty31<FP, WIDTH>
{
    fn new_from_constants(external_constants: FullRoundConstants<MontyField31<FP>, WIDTH>) -> Self {
        Self { external_constants }
    }
}
