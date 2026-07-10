use p3_field::InjectiveMonomial;
use p3_poseidon2::{
    ExternalLayer, InternalLayer, MDSMat4, add_rc_and_sbox_generic, external_initial_permute_state,
    external_terminal_permute_state,
};

use super::PackedMontyField31Sve;
use crate::{
    FieldParameters, InternalLayerParameters, Poseidon2ExternalLayerMonty31,
    Poseidon2InternalLayerMonty31, RelativelyPrimePower,
};

impl<FP, const WIDTH: usize, P2P, const D: u64> InternalLayer<PackedMontyField31Sve<FP>, WIDTH, D>
    for Poseidon2InternalLayerMonty31<FP, WIDTH, P2P>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    P2P: InternalLayerParameters<FP, WIDTH>,
{
    fn permute_state(&self, state: &mut [PackedMontyField31Sve<FP>; WIDTH]) {
        self.internal_constants.iter().for_each(|rc| {
            state[0] += *rc;
            state[0] = state[0].injective_exp_n();
            let part_sum: PackedMontyField31Sve<FP> = state[1..].iter().copied().sum();
            let full_sum = part_sum + state[0];
            state[0] = part_sum - state[0];
            P2P::internal_layer_mat_mul(state, full_sum);
        });
    }
}

impl<FP, const WIDTH: usize, const D: u64> ExternalLayer<PackedMontyField31Sve<FP>, WIDTH, D>
    for Poseidon2ExternalLayerMonty31<FP, WIDTH>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
{
    fn permute_state_initial(&self, state: &mut [PackedMontyField31Sve<FP>; WIDTH]) {
        external_initial_permute_state(
            state,
            self.external_constants.get_initial_constants(),
            add_rc_and_sbox_generic,
            &MDSMat4,
        );
    }

    fn permute_state_terminal(&self, state: &mut [PackedMontyField31Sve<FP>; WIDTH]) {
        external_terminal_permute_state(
            state,
            self.external_constants.get_terminal_constants(),
            add_rc_and_sbox_generic,
            &MDSMat4,
        );
    }
}
