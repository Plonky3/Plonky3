use core::arch::x86_64::{self};
use p3_poseidon2::DiffusionPermutation;
use p3_symmetric::Permutation;

use crate::{DiffusionMatrixMersenne31, PackedMersenne31AVX2, P};

// I + I_PRIME must be 31.
#[inline(always)]
fn left_shift_single<const I: i32, const I_PRIME: i32>(
    val: PackedMersenne31AVX2,
) -> PackedMersenne31AVX2 {
    // Clearly there would be a nicer way to do this if const generics where better...
    unsafe {
        let input = val.to_vector();
        let hi_bits_dirty = x86_64::_mm256_slli_epi32::<I>(input);
        let lo_bits = x86_64::_mm256_srli_epi32::<I_PRIME>(input);
        let hi_bits = x86_64::_mm256_and_si256(hi_bits_dirty, P);
        let output = x86_64::_mm256_or_si256(lo_bits, hi_bits);
        PackedMersenne31AVX2::from_vector(output)
    }
}

// [-2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16];
#[inline(always)]
fn left_shift_16(state: &mut [PackedMersenne31AVX2; 16]) {
    // A little annoying how this has to be done manually.
    state[0] = -(state[0] + state[0]);
    // State 1 shift is a shift by 0 which is free.
    state[2] = state[2] + state[2];
    state[3] = left_shift_single::<2, 29>(state[3]);
    state[4] = left_shift_single::<3, 28>(state[4]);
    state[5] = left_shift_single::<4, 27>(state[5]);
    state[6] = left_shift_single::<5, 26>(state[6]);
    state[7] = left_shift_single::<6, 25>(state[7]);
    state[8] = left_shift_single::<7, 24>(state[8]);
    state[9] = left_shift_single::<8, 23>(state[9]);
    state[10] = left_shift_single::<10, 21>(state[10]);
    state[11] = left_shift_single::<12, 19>(state[11]);
    state[12] = left_shift_single::<13, 18>(state[12]);
    state[13] = left_shift_single::<14, 17>(state[13]);
    state[14] = left_shift_single::<15, 16>(state[14]);
    state[15] = left_shift_single::<16, 15>(state[15]);
}

// This looks slightly strange but the key idea is that we want to minimize dependency chains.
// The compiler doesn't realize that add is still associative so we help it out.
// Note that state[0] is involved in a large s-box immediately before this so we keep it
// separate for as long as possible.
#[inline(always)]
fn sum_16(state: &[PackedMersenne31AVX2; 16]) -> PackedMersenne31AVX2 {
    let sum23 = state[2] + state[3];
    let sum45 = state[4] + state[5];
    let sum67 = state[6] + state[7];
    let sum89 = state[8] + state[9];
    let sum1011 = state[10] + state[11];
    let sum1213 = state[12] + state[13];
    let sum1415 = state[14] + state[15];

    let sum123 = state[1] + sum23;
    let sum4567 = sum45 + sum67;
    let sum891011 = sum89 + sum1011;
    let sum12131415 = sum1213 + sum1415;

    let sum1234567 = sum123 + sum4567;
    let sum_top_half = sum891011 + sum12131415;

    let sum_all_but_0 = sum1234567 + sum_top_half;

    sum_all_but_0 + state[0]
}

// [-2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,];
#[inline(always)]
fn left_shift_24(state: &mut [PackedMersenne31AVX2; 24]) {
    // A little annoying how this has to be done manually.
    state[0] = -(state[0] + state[0]);
    // State 1 shift is a shift by 0 which is free.
    state[2] = state[2] + state[2];
    state[3] = left_shift_single::<2, 29>(state[3]);
    state[4] = left_shift_single::<3, 28>(state[4]);
    state[5] = left_shift_single::<4, 27>(state[5]);
    state[6] = left_shift_single::<5, 26>(state[6]);
    state[7] = left_shift_single::<6, 25>(state[7]);
    state[8] = left_shift_single::<7, 24>(state[8]);
    state[9] = left_shift_single::<8, 23>(state[9]);
    state[10] = left_shift_single::<9, 22>(state[10]);
    state[11] = left_shift_single::<10, 21>(state[11]);
    state[12] = left_shift_single::<11, 20>(state[12]);
    state[13] = left_shift_single::<12, 19>(state[13]);
    state[14] = left_shift_single::<13, 18>(state[14]);
    state[15] = left_shift_single::<14, 17>(state[15]);
    state[16] = left_shift_single::<15, 16>(state[16]);
    state[17] = left_shift_single::<16, 15>(state[17]);
    state[18] = left_shift_single::<17, 14>(state[18]);
    state[19] = left_shift_single::<18, 13>(state[19]);
    state[20] = left_shift_single::<19, 12>(state[20]);
    state[21] = left_shift_single::<20, 11>(state[21]);
    state[22] = left_shift_single::<21, 10>(state[22]);
    state[23] = left_shift_single::<22, 9>(state[23]);
}

// This looks slightly strange but the key idea is that we want to minimize dependency chains.
// The compiler doesn't realize that add is still associative so we help it out.
// Note that state[0] is involved in a large s-box immediately before this so we keep it
// separate for as long as possible.
#[inline(always)]
fn sum_24(state: &[PackedMersenne31AVX2; 24]) -> PackedMersenne31AVX2 {
    let sum23 = state[2] + state[3];
    let sum45 = state[4] + state[5];
    let sum67 = state[6] + state[7];
    let sum89 = state[8] + state[9];
    let sum1011 = state[10] + state[11];
    let sum1213 = state[12] + state[13];
    let sum1415 = state[14] + state[15];
    let sum1617 = state[16] + state[17];
    let sum1819 = state[18] + state[19];
    let sum2021 = state[20] + state[21];
    let sum2223 = state[22] + state[23];

    let sum123 = state[1] + sum23;
    let sum4567 = sum45 + sum67;
    let sum891011 = sum89 + sum1011;
    let sum12131415 = sum1213 + sum1415;
    let sum16171819 = sum1617 + sum1819;
    let sum20212223 = sum2021 + sum2223;

    let sum1234567 = sum123 + sum4567;
    let sum_min_third = sum891011 + sum12131415;
    let sum_top_third = sum16171819 + sum20212223;

    let sum_all_but_0 = sum1234567 + sum_min_third + sum_top_third;

    sum_all_but_0 + state[0]
}

impl Permutation<[PackedMersenne31AVX2; 16]> for DiffusionMatrixMersenne31 {
    #[inline(always)]
    fn permute_mut(&self, state: &mut [PackedMersenne31AVX2; 16]) {
        let sum = sum_16(state);
        left_shift_16(state);
        state.iter_mut().for_each(|val| *val += sum);
    }
}

impl DiffusionPermutation<PackedMersenne31AVX2, 16> for DiffusionMatrixMersenne31 {}

impl Permutation<[PackedMersenne31AVX2; 24]> for DiffusionMatrixMersenne31 {
    #[inline(always)]
    fn permute_mut(&self, state: &mut [PackedMersenne31AVX2; 24]) {
        let sum = sum_24(state);
        left_shift_24(state);
        state.iter_mut().for_each(|val| *val += sum);
    }
}

impl<const WIDTH: usize> ExternalLayer<PackedMersenne31AVX2, WIDTH, 5>
    for Poseidon2ExternalLayerMersenne31
{
    type InternalState = [PackedMersenne31AVX2; WIDTH];

    fn permute_state_initial(
        &self,
        mut state: Self::InternalState,
        initial_external_constants: &[[Mersenne31; WIDTH]],
        _packed_initial_external_constants: &[()],
    ) -> [PackedMersenne31AVX2; WIDTH] {
        external_initial_permute_state::<_, _, WIDTH, 5>(
            &mut state,
            initial_external_constants,
            &MDSMat4,
        );
        state
    }

    fn permute_state_final(
        &self,
        mut state: Self::InternalState,
        final_external_constants: &[[Mersenne31; WIDTH]],
        _packed_final_external_constants: &[()],
    ) -> [PackedMersenne31AVX2; WIDTH] {
        external_final_permute_state::<_, _, WIDTH, 5>(
            &mut state,
            final_external_constants,
            &MDSMat4,
        );
        state
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_poseidon2::Poseidon2;
    use p3_symmetric::Permutation;
    use rand::Rng;

    use super::*;

    type F = Mersenne31;
    const D: u64 = 5;
    type Perm16 =
        Poseidon2<F, Poseidon2ExternalLayerMersenne31, Poseidon2InternalLayerMersenne31, 16, D>;
    type Perm24 =
        Poseidon2<F, Poseidon2ExternalLayerMersenne31, Poseidon2InternalLayerMersenne31, 24, D>;

    /// Test that the output is the same as the scalar version on a random input of length 16.
    #[test]
    fn test_avx2_poseidon2_width_16() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            Poseidon2ExternalLayerMersenne31,
            Poseidon2InternalLayerMersenne31,
            &mut rng,
        );

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedMersenne31AVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input of length 24.
    #[test]
    fn test_avx2_poseidon2_width_24() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(
            Poseidon2ExternalLayerMersenne31,
            Poseidon2InternalLayerMersenne31,
            &mut rng,
        );

        let input: [F; 24] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedMersenne31AVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }
}
