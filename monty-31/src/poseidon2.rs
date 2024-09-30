use p3_field::AbstractField;
use p3_poseidon2::{
    external_initial_permute_state, external_terminal_permute_state, ExternalLayer, InternalLayer,
    MDSMat4,
};

use crate::{
    FieldParameters, MontyField31, MontyParameters, Poseidon2ExternalLayerMonty31,
    Poseidon2InternalLayerMonty31,
};

/// Everything needed to compute multiplication by a WIDTH x WIDTH diffusion matrix whose monty form is 1 + Diag(vec).
/// vec is assumed to be of the form [-2, ...] with all entries after the first being small powers of 2.
pub trait InternalLayerBaseParameters<MP: MontyParameters, const WIDTH: usize>:
    Clone + Sync
{
    // Most of the time, ArrayLike will be [u8; WIDTH - 1].
    type ArrayLike: AsRef<[MontyField31<MP>]> + Sized;

    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul(state: &mut [MontyField31<MP>; WIDTH], sum: MontyField31<MP>);
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub trait InternalLayerParameters<FP: FieldParameters, const WIDTH: usize>:
    InternalLayerBaseParameters<FP, WIDTH> + crate::InternalLayerParametersNeon<WIDTH>
{
}
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(feature = "nightly-features", target_feature = "avx512f"))
))]
pub trait InternalLayerParameters<FP: FieldParameters, const WIDTH: usize>:
    InternalLayerBaseParameters<FP, WIDTH> + crate::InternalLayerParametersAVX2<WIDTH>
{
}
#[cfg(all(
    feature = "nightly-features",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
pub trait InternalLayerParameters<FP: FieldParameters, const WIDTH: usize>:
    InternalLayerBaseParameters<FP, WIDTH> + crate::InternalLayerParametersAVX512<WIDTH>
{
}
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(feature = "nightly-features", target_feature = "avx512f"))
    ),
    all(
        feature = "nightly-features",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ),
)))]
pub trait InternalLayerParameters<FP: FieldParameters, const WIDTH: usize>:
    InternalLayerBaseParameters<FP, WIDTH>
{
}

impl<FP, const WIDTH: usize, P2P, const D: u64> InternalLayer<MontyField31<FP>, WIDTH, D>
    for Poseidon2InternalLayerMonty31<FP, WIDTH, P2P>
where
    FP: FieldParameters,
    P2P: InternalLayerParameters<FP, WIDTH>,
{
    type InternalState = [MontyField31<FP>; WIDTH];

    /// Compute a collection of Poseidon2 internal layers.
    /// One layer for every constant supplied.
    fn permute_state(&self, state: &mut Self::InternalState) {
        self.internal_constants.iter().for_each(|rc| {
            state[0] += *rc;
            state[0] = state[0].exp_const_u64::<D>();
            let part_sum: MontyField31<FP> = state[1..].iter().cloned().sum();
            let full_sum = part_sum + state[0];
            state[0] = part_sum - state[0];
            P2P::internal_layer_mat_mul(state, full_sum);
        })
    }
}

impl<FP, const WIDTH: usize, const D: u64> ExternalLayer<MontyField31<FP>, WIDTH, D>
    for Poseidon2ExternalLayerMonty31<FP, WIDTH>
where
    FP: FieldParameters,
{
    type InternalState = [MontyField31<FP>; WIDTH];

    /// Compute the first half of the Poseidon2 external layers.
    fn permute_state_initial(&self, mut state: [MontyField31<FP>; WIDTH]) -> Self::InternalState {
        external_initial_permute_state::<MontyField31<FP>, MDSMat4, WIDTH, D>(
            &mut state,
            &self.initial_external_constants,
            &MDSMat4,
        );
        state
    }

    /// Compute the second half of the Poseidon2 external layers.
    fn permute_state_terminal(&self, mut state: Self::InternalState) -> [MontyField31<FP>; WIDTH] {
        external_terminal_permute_state::<MontyField31<FP>, MDSMat4, WIDTH, D>(
            &mut state,
            &self.terminal_external_constants,
            &MDSMat4,
        );
        state
    }
}
