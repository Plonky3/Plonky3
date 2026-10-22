//! Permutations the circuit proves in a dedicated table.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;

use p3_air::BaseAir;
use p3_field::{PrimeCharacteristicRing, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use p3_poseidon2_air::{Poseidon2Air, RoundConstants, generate_trace_rows, num_cols};
use p3_symmetric::Permutation;

/// A permutation proven one call per row of its own table.
///
/// The table reads every input and output straight from the row it proves.
pub trait PermutationChip<F>: Clone + Debug + Sync {
    /// State width of the permutation.
    const WIDTH: usize;

    /// The AIR proving one permutation per row.
    type Air: BaseAir<F> + Clone;

    /// The AIR of the permutation table.
    fn air(&self) -> &Self::Air;

    /// The permutation of a state.
    fn permute(&self, state: &[F]) -> Vec<F>;

    /// One row per call, padded to a power of two with calls on the zero state.
    fn trace(&self, inputs: Vec<Vec<F>>) -> RowMajorMatrix<F>;

    /// Column of a row holding input `i`.
    fn input_column(&self, i: usize) -> usize;

    /// Column of a row holding output `i`.
    fn output_column(&self, i: usize) -> usize;
}

/// [`PaddingFreeSponge`](p3_symmetric::PaddingFreeSponge) over any permutation of `width` values.
///
/// Each chunk of `rate` inputs overwrites the front of the state before a permutation.
pub fn sponge<T: PrimeCharacteristicRing>(
    inputs: &[T],
    width: usize,
    rate: usize,
    out: usize,
    mut permute: impl FnMut(&[T]) -> Vec<T>,
) -> Vec<T> {
    let mut state = vec![T::ZERO; width];
    for chunk in inputs.chunks(rate) {
        state[..chunk.len()].clone_from_slice(chunk);
        state = permute(&state);
    }
    state.truncate(out);
    state
}

/// The Poseidon2 permutation, proven by [`Poseidon2Air`].
///
/// `Perm` must compute the permutation the round constants define.
pub struct Poseidon2Chip<
    F: PrimeField,
    LinearLayers,
    Perm,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    /// The native permutation, for witness values.
    perm: Perm,
    /// The round constants, for trace generation.
    constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    /// The AIR the table proves.
    air: Poseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
}

impl<
    F: PrimeField,
    LinearLayers,
    Perm: Clone,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> Clone
    for Poseidon2Chip<
        F,
        LinearLayers,
        Perm,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    fn clone(&self) -> Self {
        Self {
            perm: self.perm.clone(),
            constants: self.constants.clone(),
            air: self.air.clone(),
        }
    }
}

impl<
    F: PrimeField,
    LinearLayers,
    Perm,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> Debug
    for Poseidon2Chip<
        F,
        LinearLayers,
        Perm,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Poseidon2Chip")
            .field("width", &WIDTH)
            .finish_non_exhaustive()
    }
}

impl<
    F: PrimeField,
    LinearLayers,
    Perm,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>
    Poseidon2Chip<
        F,
        LinearLayers,
        Perm,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    /// A chip for `perm`, which the round constants define.
    pub fn new(
        perm: Perm,
        constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    ) -> Self {
        Self {
            perm,
            air: Poseidon2Air::new(constants.clone()),
            constants,
        }
    }

    /// The native permutation.
    pub const fn permutation(&self) -> &Perm {
        &self.perm
    }
}

impl<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH> + Sync,
    Perm: Permutation<[F; WIDTH]> + Debug,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> PermutationChip<F>
    for Poseidon2Chip<
        F,
        LinearLayers,
        Perm,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    const WIDTH: usize = WIDTH;

    type Air = Poseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >;

    fn air(&self) -> &Self::Air {
        &self.air
    }

    fn permute(&self, state: &[F]) -> Vec<F> {
        let state: [F; WIDTH] = state.try_into().expect("one value per state element");
        self.perm.permute(state).to_vec()
    }

    fn trace(&self, inputs: Vec<Vec<F>>) -> RowMajorMatrix<F> {
        let height = inputs.len().next_power_of_two();
        let mut states: Vec<[F; WIDTH]> = inputs
            .into_iter()
            .map(|state| state.try_into().expect("one value per state element"))
            .collect();
        states.resize(height, [F::ZERO; WIDTH]);
        generate_trace_rows::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(states, &self.constants, 0)
    }

    fn input_column(&self, i: usize) -> usize {
        // The inputs open the row.
        i
    }

    fn output_column(&self, i: usize) -> usize {
        // The outputs are the state after the last full round, which closes the row.
        num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>() - WIDTH
            + i
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::{KoalaBear, default_koalabear_poseidon2_16};
    use p3_symmetric::{CryptographicHasher, PaddingFreeSponge, Permutation};
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn sponge_matches_the_padding_free_sponge(inputs in prop::collection::vec(any::<u32>(), 0..40)) {
            let perm = default_koalabear_poseidon2_16();
            let inputs: Vec<KoalaBear> = inputs.into_iter().map(KoalaBear::new).collect();
            let expected = PaddingFreeSponge::<_, 16, 8, 8>::new(perm.clone()).hash_slice(&inputs);
            let digest = sponge(&inputs, 16, 8, 8, |state| {
                perm.permute(<[KoalaBear; 16]>::try_from(state).unwrap()).to_vec()
            });
            prop_assert_eq!(digest, expected.to_vec());
        }
    }
}
