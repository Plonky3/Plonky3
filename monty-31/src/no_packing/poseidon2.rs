use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_poseidon2::{ExternalLayerConstants, ExternalLayerConstructor, InternalLayerConstructor};

use crate::{FieldParameters, InternalLayerBaseParameters, MontyField31, MontyParameters};

#[derive(Debug, Clone, Default)]
pub struct Poseidon2InternalLayerMonty31<
    MP: MontyParameters,
    const WIDTH: usize,
    ILP: InternalLayerBaseParameters<MP, WIDTH>,
> {
    pub(crate) internal_constants: Vec<MontyField31<MP>>,
    _phantom: PhantomData<ILP>,
}

#[derive(Default, Clone)]
pub struct Poseidon2ExternalLayerMonty31<MP: MontyParameters, const WIDTH: usize> {
    pub(crate) initial_external_constants: Vec<[MontyField31<MP>; WIDTH]>,
    pub(crate) final_external_constants: Vec<[MontyField31<MP>; WIDTH]>,
}

impl<FP: FieldParameters, const WIDTH: usize, ILP: InternalLayerBaseParameters<FP, WIDTH>>
    InternalLayerConstructor<MontyField31<FP>> for Poseidon2InternalLayerMonty31<FP, WIDTH, ILP>
{
    fn new_from_constants(internal_constants: Vec<MontyField31<FP>>) -> Self {
        Self {
            internal_constants,
            _phantom: PhantomData,
        }
    }
}

impl<FP: FieldParameters, const WIDTH: usize> ExternalLayerConstructor<MontyField31<FP>, WIDTH>
    for Poseidon2ExternalLayerMonty31<FP, WIDTH>
{
    fn new_from_constants(
        external_constants: ExternalLayerConstants<MontyField31<FP>, WIDTH>,
    ) -> Self {
        let initial_external_constants = external_constants.get_initial_constants().clone();
        let final_external_constants = external_constants.get_terminal_constants().clone();

        Self {
            initial_external_constants,
            final_external_constants,
        }
    }
}