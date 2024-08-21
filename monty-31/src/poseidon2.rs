use core::marker::PhantomData;

use p3_field::AbstractField;
use p3_poseidon2::{
    external_final_permute_state, external_initial_permute_state, ExternalLayer, InternalLayer,
    MDSMat4, NoPackedImplementation,
};

use crate::{monty_reduce, FieldParameters, MontyField31, MontyParameters};

/// Everything needed to compute multiplication by a WIDTH x WIDTH diffusion matrix whose monty form is 1 + Diag(vec).
/// vec is assumed to be of the form [-2, ...] with all entries after the first being small powers of 2.
pub trait InternalLayerParameters<FP: FieldParameters, const WIDTH: usize>: Clone + Sync {
    // Most of the time, ArrayLike will be [u8; WIDTH - 1].
    type ArrayLike: AsRef<[u8]> + Sized;

    // We only need to save the powers and can ignore the initial element.
    const INTERNAL_DIAG_SHIFTS: Self::ArrayLike;

    /// Implements multiplication by the diffusion matrix 1 + Diag(vec) using a delayed reduction strategy.
    fn permute_state(state: &mut [MontyField31<FP>; WIDTH]) {
        let part_sum: u64 = state.iter().skip(1).map(|x| x.value as u64).sum();
        let full_sum = part_sum + (state[0].value as u64);
        let s0 = part_sum + (-state[0]).value as u64;
        state[0] = MontyField31::new_monty(monty_reduce::<FP>(s0));

        for i in 0..Self::INTERNAL_DIAG_SHIFTS.as_ref().len() {
            let si =
                full_sum + ((state[i + 1].value as u64) << Self::INTERNAL_DIAG_SHIFTS.as_ref()[i]);
            state[i + 1] = MontyField31::new_monty(monty_reduce::<FP>(si));
        }
    }
}

/// Some code needed by the PackedField implementation can be shared between the different WIDTHS and architectures.
/// This will likely be deleted once we have vectorized implementations.
pub trait PackedFieldPoseidon2Helpers<MP: MontyParameters> {
    const MONTY_INVERSE: MontyField31<MP> = MontyField31::new_monty(1);
}

#[derive(Debug, Clone, Default)]
pub struct Poseidon2InternalLayerMonty31<P2P>
where
    P2P: Clone,
{
    _phantom: PhantomData<P2P>,
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
impl<P2P> NoPackedImplementation for Poseidon2InternalLayerMonty31<P2P> where P2P: Clone + Sync {}

impl<FP, const WIDTH: usize, P2P, const D: u64> InternalLayer<MontyField31<FP>, WIDTH, D>
    for Poseidon2InternalLayerMonty31<P2P>
where
    FP: FieldParameters,
    P2P: InternalLayerParameters<FP, WIDTH>,
{
    type InternalState = [MontyField31<FP>; WIDTH];

    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        internal_constants: &[MontyField31<FP>],
        _packed_internal_constants: &[Self::ConstantsType],
    ) {
        internal_constants.iter().for_each(|rc| {
            state[0] += *rc;
            state[0] = state[0].exp_const_u64::<D>();
            P2P::permute_state(state);
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Poseidon2ExternalLayerMonty31<const WIDTH: usize> {}

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
impl<const WIDTH: usize> NoPackedImplementation for Poseidon2ExternalLayerMonty31<WIDTH> {}

impl<FP, const WIDTH: usize, const D: u64> ExternalLayer<MontyField31<FP>, WIDTH, D>
    for Poseidon2ExternalLayerMonty31<WIDTH>
where
    FP: FieldParameters,
{
    type InternalState = [MontyField31<FP>; WIDTH];

    fn permute_state_initial(
        &self,
        mut state: [MontyField31<FP>; WIDTH],
        initial_external_constants: &[[<MontyField31<FP> as p3_field::AbstractField>::F; WIDTH]],
        _initial_external_packed_constants: &[Self::ConstantsType],
    ) -> Self::InternalState {
        external_initial_permute_state::<MontyField31<FP>, MDSMat4, WIDTH, D>(
            &mut state,
            initial_external_constants,
            &MDSMat4,
        );
        state
    }

    fn permute_state_final(
        &self,
        mut state: Self::InternalState,
        final_external_constants: &[[<MontyField31<FP> as p3_field::AbstractField>::F; WIDTH]],
        _final_external_packed_constants: &[Self::ConstantsType],
    ) -> [MontyField31<FP>; WIDTH] {
        external_final_permute_state::<MontyField31<FP>, MDSMat4, WIDTH, D>(
            &mut state,
            final_external_constants,
            &MDSMat4,
        );
        state
    }
}
