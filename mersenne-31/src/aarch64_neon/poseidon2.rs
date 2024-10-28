//! Eventually this will hold a Vectorized Neon implementation of Poseidon2 for PackedMersenne31Neon
//! Currently this is essentially a placeholder to allow compilation on Neon devices.
//!
//! Converting the AVX2/AVX512 code across to Neon is on the TODO list.

use alloc::vec::Vec;

use p3_field::AbstractField;
use p3_poseidon2::{
    mds_light_permutation, ExternalLayer, ExternalLayerConstants, ExternalLayerConstructor,
    GenericPoseidon2LinearLayers, InternalLayer, InternalLayerConstructor, MDSMat4,
};

use crate::{GenericPoseidon2LinearLayersMersenne31, Mersenne31, PackedMersenne31Neon};

/// The internal layers of the Poseidon2 permutation.
#[derive(Debug, Clone)]
pub struct Poseidon2InternalLayerMersenne31 {
    pub(crate) internal_constants: Vec<Mersenne31>,
}

/// The external layers of the Poseidon2 permutation.
#[derive(Clone)]
pub struct Poseidon2ExternalLayerMersenne31<const WIDTH: usize> {
    pub(crate) external_constants: ExternalLayerConstants<Mersenne31, WIDTH>,
}

impl InternalLayerConstructor<PackedMersenne31Neon> for Poseidon2InternalLayerMersenne31 {
    fn new_from_constants(internal_constants: Vec<Mersenne31>) -> Self {
        Self { internal_constants }
    }
}

impl<const WIDTH: usize> ExternalLayerConstructor<PackedMersenne31Neon, WIDTH>
    for Poseidon2ExternalLayerMersenne31<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Mersenne31, WIDTH>) -> Self {
        Self { external_constants }
    }
}

impl<const WIDTH: usize, const D: u64> InternalLayer<PackedMersenne31Neon, WIDTH, D>
    for Poseidon2InternalLayerMersenne31
where
    GenericPoseidon2LinearLayersMersenne31:
        GenericPoseidon2LinearLayers<PackedMersenne31Neon, WIDTH>,
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [PackedMersenne31Neon; WIDTH]) {
        self.internal_constants.iter().for_each(|&rc| {
            state[0] += rc;
            state[0] = state[0].exp_const_u64::<D>();
            GenericPoseidon2LinearLayersMersenne31::internal_linear_layer(state);
        })
    }
}

/// Compute a collection of Poseidon2 external layers.
/// One layer for every constant supplied.
#[inline]
fn external_rounds<const WIDTH: usize, const D: u64>(
    state: &mut [PackedMersenne31Neon; WIDTH],
    packed_external_constants: &[[Mersenne31; WIDTH]],
) {
    /*
        The external layer consists of the following 2 operations:

        s -> s + rc
        s -> s^d
        s -> Ms

        Where by s^d we mean to apply this power function element wise.

        Multiplication by M is implemented efficiently in p3_poseidon2/matrix.
    */
    packed_external_constants.iter().for_each(|round_consts| {
        state
            .iter_mut()
            .zip(round_consts.iter())
            .for_each(|(val, &rc)| {
                *val += rc;
                *val = val.exp_const_u64::<D>();
            });
        mds_light_permutation(state, &MDSMat4);
    });
}

impl<const D: u64, const WIDTH: usize> ExternalLayer<PackedMersenne31Neon, WIDTH, D>
    for Poseidon2ExternalLayerMersenne31<WIDTH>
{
    /// Perform the initial external layers of the Poseidon2 permutation on the given state.
    fn permute_state_initial(&self, state: &mut [PackedMersenne31Neon; WIDTH]) {
        mds_light_permutation(state, &MDSMat4);
        external_rounds::<WIDTH, D>(state, &self.external_constants.get_initial_constants());
    }

    /// Perform the terminal external layers of the Poseidon2 permutation on the given state.
    fn permute_state_terminal(&self, state: &mut [PackedMersenne31Neon; WIDTH]) {
        external_rounds::<WIDTH, D>(state, &self.external_constants.get_terminal_constants());
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_symmetric::Permutation;
    use rand::Rng;

    use super::*;
    use crate::Poseidon2Mersenne31;

    type F = Mersenne31;
    type Perm16 = Poseidon2Mersenne31<16>;
    type Perm24 = Poseidon2Mersenne31<24>;

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_neon_poseidon2_width_16() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(&mut rng);

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut neon_input = input.map(PackedMersenne31Neon::from_f);
        poseidon2.permute_mut(&mut neon_input);

        let neon_output = neon_input.map(|x| x.0[0]);

        assert_eq!(neon_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_neon_poseidon2_width_24() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(&mut rng);

        let input: [F; 24] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut neon_input = input.map(PackedMersenne31Neon::from_f);
        poseidon2.permute_mut(&mut neon_input);

        let neon_output = neon_input.map(|x| x.0[0]);

        assert_eq!(neon_output, expected);
    }
}
