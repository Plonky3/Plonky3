use core::marker::PhantomData;

use p3_poseidon2::DiffusionPermutation;
use p3_symmetric::Permutation;

use crate::{monty_reduce, FieldParameters, MontyField31};

pub trait Poseidon2Utils<FP: FieldParameters, const WIDTH: usize> {
    type ArrayLike: AsRef<[u8]> + Sized;
    const INTERNAL_DIAG_SHIFTS: Self::ArrayLike;

    // Long term INTERNAL_DIAG_MONTY will be removed.
    // Currently we need it for the naive Packed field implementations.
    const INTERNAL_DIAG_MONTY: [MontyField31<FP>; WIDTH];

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

// Long term we could also do an implementation like the following:
// Currently this doesn't work due to the orphan rule and the fact that we need to derive
// Permutation<[_; 16]> for the various field packings. (AVX2, AVX512, NEON)
// It's also a little ugly due to the need for 2 pieces of PhantomData.

pub trait Poseidon2Monty31<FP: FieldParameters>:
    Poseidon2Utils<FP, 16> + Poseidon2Utils<FP, 24> + Clone + Sync
{
    const MONTY_INVERSE: MontyField31<FP> = MontyField31::<FP>::new_monty(1);
}

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixMontyField31<FP: FieldParameters, PU: Poseidon2Monty31<FP>> {
    _phantom1: PhantomData<FP>,
    _phantom2: PhantomData<PU>,
}

impl<FP: FieldParameters, const WIDTH: usize, PU: Poseidon2Monty31<FP>>
    Permutation<[MontyField31<FP>; WIDTH]> for DiffusionMatrixMontyField31<FP, PU>
where
    PU: Poseidon2Utils<FP, WIDTH>,
{
    #[inline]
    fn permute_mut(&self, state: &mut [MontyField31<FP>; WIDTH]) {
        PU::permute_state(state);
    }
}

impl<FP: FieldParameters, const WIDTH: usize, PU: Poseidon2Monty31<FP>>
    DiffusionPermutation<MontyField31<FP>, WIDTH> for DiffusionMatrixMontyField31<FP, PU>
where
    PU: Poseidon2Utils<FP, WIDTH>,
{
}
