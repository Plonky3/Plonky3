use core::marker::PhantomData;

use p3_field::Field;
use p3_poseidon::external::{
    FullRoundLayer, full_round_initial_permute_state, full_round_terminal_permute_state,
};
use p3_poseidon::generic::GenericPoseidonLinearLayers;
use p3_poseidon::internal::{PartialRoundLayer, partial_permute_state};

use crate::{
    FieldParameters, MontyField31, MontyParameters, PoseidonExternalLayerMonty31,
    PoseidonInternalLayerMonty31, RelativelyPrimePower,
};

/// Trait for Poseidon partial round scalar operations.
pub trait PartialRoundBaseParameters<MP: MontyParameters, const WIDTH: usize>:
    Clone + Sync
{
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub trait PartialRoundParameters<FP: FieldParameters, const WIDTH: usize>:
    PartialRoundBaseParameters<FP, WIDTH> + crate::PartialRoundParametersNeon<FP, WIDTH>
{
}
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub trait PartialRoundParameters<FP: FieldParameters, const WIDTH: usize>:
    PartialRoundBaseParameters<FP, WIDTH> + crate::PartialRoundParametersAVX2<FP, WIDTH>
{
}
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub trait PartialRoundParameters<FP: FieldParameters, const WIDTH: usize>:
    PartialRoundBaseParameters<FP, WIDTH> + crate::PartialRoundParametersAVX512<FP, WIDTH>
{
}
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ),
    all(target_arch = "x86_64", target_feature = "avx512f"),
)))]
pub trait PartialRoundParameters<FP: FieldParameters, const WIDTH: usize>:
    PartialRoundBaseParameters<FP, WIDTH>
{
}

impl<FP, const WIDTH: usize, P1P, const D: u64> PartialRoundLayer<MontyField31<FP>, WIDTH, D>
    for PoseidonInternalLayerMonty31<FP, WIDTH, P1P>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    P1P: PartialRoundParameters<FP, WIDTH>,
{
    fn permute_state(&self, state: &mut [MontyField31<FP>; WIDTH]) {
        partial_permute_state::<MontyField31<FP>, MontyField31<FP>, WIDTH, D>(
            state,
            &self.internal_constants,
        );
    }
}

impl<FP, const WIDTH: usize, const D: u64> FullRoundLayer<MontyField31<FP>, WIDTH, D>
    for PoseidonExternalLayerMonty31<FP, WIDTH>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
{
    fn permute_state_initial(&self, state: &mut [MontyField31<FP>; WIDTH]) {
        full_round_initial_permute_state::<MontyField31<FP>, MontyField31<FP>, WIDTH, D>(
            state,
            &self.external_constants,
        );
    }

    fn permute_state_terminal(&self, state: &mut [MontyField31<FP>; WIDTH]) {
        full_round_terminal_permute_state::<MontyField31<FP>, MontyField31<FP>, WIDTH, D>(
            state,
            &self.external_constants,
        );
    }
}

/// Generic Poseidon linear layers for MontyField31.
pub struct GenericPoseidonLinearLayersMonty31<FP, PRBP> {
    _phantom1: PhantomData<FP>,
    _phantom2: PhantomData<PRBP>,
}

impl<FP, PRBP, F, const WIDTH: usize> GenericPoseidonLinearLayers<F, WIDTH>
    for GenericPoseidonLinearLayersMonty31<FP, PRBP>
where
    FP: FieldParameters,
    PRBP: PartialRoundBaseParameters<FP, WIDTH>,
    F: Field,
{
}
