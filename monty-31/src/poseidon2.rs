use core::marker::PhantomData;

use p3_field::{InjectiveMonomial, PrimeCharacteristicRing};
use p3_poseidon2::{
    ExternalLayer, GenericPoseidon2LinearLayers, InternalLayer, MDSMat4, add_rc_and_sbox_generic,
    external_initial_permute_state, external_terminal_permute_state,
};

use crate::{
    FieldParameters, MontyField31, MontyParameters, Poseidon2ExternalLayerMonty31,
    Poseidon2InternalLayerMonty31, RelativelyPrimePower,
};

/// Trait which handles the Poseidon2 internal layers.
///
/// Everything needed to compute multiplication by a `WIDTH x WIDTH` diffusion matrix whose monty form is `1 + Diag(vec)`.
/// vec is assumed to be of the form `[-2, ...]` with all entries after the first being small powers of `2`.
pub trait InternalLayerBaseParameters<MP: MontyParameters, const WIDTH: usize>:
    Clone + Sync
{
    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul<R: PrimeCharacteristicRing>(state: &mut [R; WIDTH], sum: R);

    /// Perform the matrix multiplication corresponding to the internal linear
    /// layer.
    fn generic_internal_linear_layer<R: PrimeCharacteristicRing>(state: &mut [R; WIDTH]) {
        // We mostly delegate to internal_layer_mat_mul but have to handle state[0] separately.
        let part_sum: R = state[1..].iter().cloned().sum();
        let full_sum = part_sum.clone() + state[0].clone();
        state[0] = part_sum - state[0].clone();
        Self::internal_layer_mat_mul(state, full_sum);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub trait InternalLayerParameters<FP: FieldParameters, const WIDTH: usize>:
    InternalLayerBaseParameters<FP, WIDTH> + crate::InternalLayerParametersNeon<FP, WIDTH>
{
}
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub trait InternalLayerParameters<FP: FieldParameters, const WIDTH: usize>:
    InternalLayerBaseParameters<FP, WIDTH> + crate::InternalLayerParametersAVX2<FP, WIDTH>
{
}
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub trait InternalLayerParameters<FP: FieldParameters, const WIDTH: usize>:
    InternalLayerBaseParameters<FP, WIDTH> + crate::InternalLayerParametersAVX512<FP, WIDTH>
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
pub trait InternalLayerParameters<FP: FieldParameters, const WIDTH: usize>:
    InternalLayerBaseParameters<FP, WIDTH>
{
}

impl<FP, const WIDTH: usize, P2P, const D: u64> InternalLayer<MontyField31<FP>, WIDTH, D>
    for Poseidon2InternalLayerMonty31<FP, WIDTH, P2P>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    P2P: InternalLayerParameters<FP, WIDTH>,
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [MontyField31<FP>; WIDTH]) {
        self.internal_constants.iter().for_each(|rc| {
            state[0] += *rc;
            state[0] = state[0].injective_exp_n();
            let part_sum: MontyField31<FP> = state[1..].iter().copied().sum();
            let full_sum = part_sum + state[0];
            state[0] = part_sum - state[0];
            P2P::internal_layer_mat_mul(state, full_sum);
        })
    }
}

impl<FP, const WIDTH: usize, const D: u64> ExternalLayer<MontyField31<FP>, WIDTH, D>
    for Poseidon2ExternalLayerMonty31<FP, WIDTH>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
{
    /// Perform the initial external layers of the Poseidon2 permutation on the given state.
    fn permute_state_initial(&self, state: &mut [MontyField31<FP>; WIDTH]) {
        external_initial_permute_state(
            state,
            self.external_constants.get_initial_constants(),
            add_rc_and_sbox_generic,
            &MDSMat4,
        );
    }

    /// Perform the terminal external layers of the Poseidon2 permutation on the given state.
    fn permute_state_terminal(&self, state: &mut [MontyField31<FP>; WIDTH]) {
        external_terminal_permute_state(
            state,
            self.external_constants.get_terminal_constants(),
            add_rc_and_sbox_generic,
            &MDSMat4,
        );
    }
}

/// An implementation of the matrix multiplications in the internal and external layers of Poseidon2.
///
/// This can act on `[A; WIDTH]` for any ring implementing `Algebra<MontyField31<FP>>`.
/// This will usually be slower than the Poseidon2 permutation built from `Poseidon2InternalLayerMonty31` and
/// `Poseidon2ExternalLayerMonty31` but it does work in more cases.
pub struct GenericPoseidon2LinearLayersMonty31<FP, ILBP> {
    _phantom1: PhantomData<FP>,
    _phantom2: PhantomData<ILBP>,
}

impl<FP, ILBP, const WIDTH: usize> GenericPoseidon2LinearLayers<WIDTH>
    for GenericPoseidon2LinearLayersMonty31<FP, ILBP>
where
    FP: FieldParameters,
    ILBP: InternalLayerBaseParameters<FP, WIDTH>,
{
    fn internal_linear_layer<R: PrimeCharacteristicRing>(state: &mut [R; WIDTH]) {
        ILBP::generic_internal_linear_layer(state);
    }
}
