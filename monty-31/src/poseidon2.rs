use core::marker::PhantomData;
use core::ops::Mul;

use p3_field::AbstractField;
use p3_poseidon2::DiffusionPermutation;
use p3_symmetric::Permutation;

use crate::{monty_reduce, FieldParameters, MontyField31, MontyParameters};

/// Everything needed to compute multiplication by a WIDTH x WIDTH diffusion matrix whose monty form is 1 + Diag(vec).
/// vec is assumed to be of the form [-2, ...] with all entries after the first being small powers of 2.
pub trait DiffusionMatrixParameters<FP: FieldParameters, const WIDTH: usize>: Clone + Sync {
    // Most of the time, ArrayLike will be [u8; WIDTH - 1].
    type ArrayLike: AsRef<[u8]> + Sized;

    // We only need to save the powers and can ignore the initial element.
    const INTERNAL_DIAG_SHIFTS: Self::ArrayLike;

    // Long term INTERNAL_DIAG_MONTY will be removed.
    // Currently it is needed for the Packed field implementations.
    const INTERNAL_DIAG_MONTY: [MontyField31<FP>; WIDTH];

    /// Implements multiplication by the diffusion matrix 1 + Diag(vec) using a delayed reduction strategy.
    #[inline]
    fn permute_state(state: &mut [MontyField31<FP>; WIDTH]) {
        let part_sum: u64 = state.iter().skip(1).map(|x| x.value as u64).sum();
        let full_sum = part_sum + (state[0].value as u64);
        let s0 = part_sum + (-state[0]).value as u64;
        state[0] = MontyField31::new_monty(monty_reduce::<FP>(s0));

        for i in 0..Self::INTERNAL_DIAG_SHIFTS.as_ref().len() {
            let shift_i = Self::INTERNAL_DIAG_SHIFTS.as_ref()[i];
            let si = full_sum + ((state[i + 1].value as u64) << shift_i);
            state[i + 1] = MontyField31::new_monty(monty_reduce::<FP>(si));
        }
    }

    /// Like `permute_state`, but works with any `AbstractField`.
    #[inline]
    fn permute_state_generic<AF: AbstractField + Mul<MontyField31<FP>, Output = AF>>(
        state: &mut [AF; WIDTH],
    ) {
        let part_sum: AF = state.iter().skip(1).cloned().sum();
        let full_sum = part_sum.clone() + state[0].clone();
        state[0] = part_sum - state[0].clone();

        for (state_i, const_i) in state.iter_mut().zip(Self::INTERNAL_DIAG_MONTY).skip(1) {
            *state_i = full_sum.clone() + state_i.clone() * const_i;
        }
    }
}

/// Some code needed by the PackedField implementation can be shared between the different WIDTHS and architectures.
/// This will likely be deleted once we have vectorized implementations.
pub trait PackedFieldPoseidon2Helpers<MP: MontyParameters> {
    const MONTY_INVERSE: MontyField31<MP> = MontyField31::new_monty(1);
}

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixMontyField31<MP>
where
    MP: Clone,
{
    _phantom: PhantomData<MP>,
}

impl<FP, const WIDTH: usize, MP> Permutation<[MontyField31<FP>; WIDTH]>
    for DiffusionMatrixMontyField31<MP>
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
    for DiffusionMatrixMontyField31<MP>
where
    FP: FieldParameters,
    MP: DiffusionMatrixParameters<FP, WIDTH>,
{
}

/// Like `DiffusionMatrixMontyField31`, but generalized to any `AbstractField`, and less efficient
/// for the concrete Monty fields.
#[derive(Debug, Clone, Default)]
pub struct GenericDiffusionMatrixMontyField31<FP, MP: Clone> {
    _phantom_fp: PhantomData<FP>,
    _phantom_mp: PhantomData<MP>,
}

impl<FP, MP: Clone> GenericDiffusionMatrixMontyField31<FP, MP> {
    pub fn new() -> Self {
        Self {
            _phantom_fp: PhantomData,
            _phantom_mp: PhantomData,
        }
    }
}

impl<AF, FP, const WIDTH: usize, MP> Permutation<[AF; WIDTH]>
    for GenericDiffusionMatrixMontyField31<FP, MP>
where
    AF: AbstractField + Mul<MontyField31<FP>, Output = AF>,
    FP: FieldParameters,
    MP: DiffusionMatrixParameters<FP, WIDTH>,
{
    fn permute_mut(&self, input: &mut [AF; WIDTH]) {
        MP::permute_state_generic(input)
    }
}

impl<AF, FP, const WIDTH: usize, MP> DiffusionPermutation<AF, WIDTH>
    for GenericDiffusionMatrixMontyField31<FP, MP>
where
    AF: AbstractField + Mul<MontyField31<FP>, Output = AF>,
    FP: FieldParameters,
    MP: DiffusionMatrixParameters<FP, WIDTH>,
{
}
