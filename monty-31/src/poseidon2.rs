use core::marker::PhantomData;

use p3_poseidon2::DiffusionPermutation;
use p3_symmetric::Permutation;

use crate::{monty_reduce, FieldParameters, MontyField31, MontyParameters};

/// Everything needed to compute multiplication by a WIDTH x WIDTH diffusion matrix whose monty form is 1 + D(v).
pub trait DiffusionMatrixParameters<FP: FieldParameters, const WIDTH: usize>: Clone + Sync {
    // Most of the time, ArrayLike will be [u8; WIDTH - 1].
    type ArrayLike: AsRef<[u8]> + Sized;

    // We assume that v[0] = -2 and all other values of v are small powers of 2.
    // Thus we simply save the powers.
    const INTERNAL_DIAG_SHIFTS: Self::ArrayLike;

    // Long term INTERNAL_DIAG_MONTY will be removed.
    // Currently we need it for the naive Packed field implementations.
    const INTERNAL_DIAG_MONTY: [MontyField31<FP>; WIDTH];

    /// Implements multiplication by the diffusion matrix 1 + D(v) using a delayed reduction strategy.
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

/// For a given field, we need to implement DiffusionMatrixParameters for several different WIDTHS.
/// Some code can be shared between the different sizes.
pub trait PackedFieldPoseidon2Helpers<MP: MontyParameters> {
    // This is currently needed for Packed Field impls.
    // It can/will be removed once we have vectorized implementations.
    const MONTY_INVERSE: MontyField31<MP> = MontyField31::new_monty(1);
}

// Would be good to try and find a way to cut down on PhantomData.
#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixMontyField31<FP, MP>
where
    FP: FieldParameters,
    MP: Clone,
{
    _phantom1: PhantomData<FP>,
    _phantom2: PhantomData<MP>,
}

impl<FP, const WIDTH: usize, MP> Permutation<[MontyField31<FP>; WIDTH]>
    for DiffusionMatrixMontyField31<FP, MP>
where
    FP: FieldParameters,
    MP: DiffusionMatrixParameters<FP, WIDTH>,
{
    #[inline]
    fn permute_mut(&self, state: &mut [MontyField31<FP>; WIDTH]) {
        MP::permute_state(state);
    }
}

impl<FP, const WIDTH: usize, MP> DiffusionPermutation<MontyField31<FP>, WIDTH>
    for DiffusionMatrixMontyField31<FP, MP>
where
    FP: FieldParameters,
    MP: DiffusionMatrixParameters<FP, WIDTH>,
{
}
