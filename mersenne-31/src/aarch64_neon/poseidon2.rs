//! Vectorized NEON implementation of Poseidon2 for Mersenne31.

use alloc::vec::Vec;
use core::arch::aarch64::{self, uint32x4_t};
use core::mem::transmute;

use p3_field::PrimeCharacteristicRing;
use p3_poseidon2::{
    ExternalLayer, ExternalLayerConstants, ExternalLayerConstructor, InternalLayer,
    InternalLayerConstructor, MDSMat4, external_initial_permute_state,
    external_terminal_permute_state,
};

use super::packing::exp5;
use crate::{Mersenne31, PackedMersenne31Neon};

/// The prime P = 2^31 - 1 as a packed NEON vector.
const P: uint32x4_t = unsafe { transmute::<[u32; 4], _>([0x7fffffff; 4]) };

/// The internal layers of the Poseidon2 permutation for Mersenne31.
///
/// Constants are stored in standard POSITIVE form `{0, ..., P}`.
#[derive(Debug, Clone)]
pub struct Poseidon2InternalLayerMersenne31 {
    /// The scalar round constants for each internal round.
    pub(crate) internal_constants: Vec<Mersenne31>,
    /// The round constants packed into NEON vectors for vectorized computation.
    packed_internal_constants: Vec<uint32x4_t>,
}

impl InternalLayerConstructor<Mersenne31> for Poseidon2InternalLayerMersenne31 {
    fn new_from_constants(internal_constants: Vec<Mersenne31>) -> Self {
        let packed_internal_constants = internal_constants
            .iter()
            .map(|c| unsafe { aarch64::vdupq_n_u32(c.value) })
            .collect();
        Self {
            internal_constants,
            packed_internal_constants,
        }
    }
}

/// The external layers of the Poseidon2 permutation for Mersenne31.
///
/// Constants are stored in standard POSITIVE form `{0, ..., P}`.
#[derive(Clone)]
pub struct Poseidon2ExternalLayerMersenne31<const WIDTH: usize> {
    /// The scalar round constants for both initial and terminal external rounds.
    pub(crate) external_constants: ExternalLayerConstants<Mersenne31, WIDTH>,
    /// The initial external round constants packed into NEON vectors.
    packed_initial_external_constants: Vec<[uint32x4_t; WIDTH]>,
    /// The terminal external round constants packed into NEON vectors.
    packed_terminal_external_constants: Vec<[uint32x4_t; WIDTH]>,
}

impl<const WIDTH: usize> ExternalLayerConstructor<Mersenne31, WIDTH>
    for Poseidon2ExternalLayerMersenne31<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Mersenne31, WIDTH>) -> Self {
        let packed_initial_external_constants = external_constants
            .get_initial_constants()
            .iter()
            .map(|arr| arr.map(|c| unsafe { aarch64::vdupq_n_u32(c.value) }))
            .collect();
        let packed_terminal_external_constants = external_constants
            .get_terminal_constants()
            .iter()
            .map(|arr| arr.map(|c| unsafe { aarch64::vdupq_n_u32(c.value) }))
            .collect();
        Self {
            external_constants,
            packed_initial_external_constants,
            packed_terminal_external_constants,
        }
    }
}

/// Compute the map `x -> 2^I * x` on Mersenne-31 field elements using cyclic rotation.
///
/// For Mersenne-31 (P = 2^31 - 1), we have `2^31 ≡ 1 (mod P)`, which means multiplication
/// by `2^I` is equivalent to a cyclic rotation of the binary representation.
///
/// # Safety
/// `val` must be represented as a value in `{0, ..., P}`.
/// If the input does not conform to this representation, the result is undefined.
///
/// # Generic Parameters
/// - `I`: The exponent (shift amount)
/// - `I_PRIME`: Must satisfy `I + I_PRIME = 31`
#[inline(always)]
pub(crate) fn mul_2exp_i<const I: i32, const I_PRIME: i32>(
    val: PackedMersenne31Neon,
) -> PackedMersenne31Neon {
    // We want this to compile to:
    //      ushr  lo.4s, val.4s, #(31 - I)  // Shift high bits down
    //      sli   lo.4s, val.4s, #I         // Shift low bits up and insert into lo
    //      and   res.4s, lo.4s, P.4s       // Clear the dirty sign bit
    // throughput: ~0.75 cyc/vec
    // latency: 4 cyc

    const {
        assert!(I + I_PRIME == 31);
    }

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let input = val.to_vector();

        // 1. Shift high bits down to the bottom.
        let lo = aarch64::vshrq_n_u32::<I_PRIME>(input);

        // 2. Shift low bits up and INSERT them into the accumulator.
        //
        // `vsli` (Vector Shift Left and Insert) shifts `input` left by I bits,
        // then inserts the result into `lo`, preserving the low I bits of `lo`.
        //
        // Result: bits[30:I] = input[30-I:0], bits[I-1:0] = lo[I-1:0]
        let inserted = aarch64::vsliq_n_u32::<I>(lo, input);

        // 3. Clear the sign bit (which might be dirty from the shift left).
        let output = aarch64::vandq_u32(inserted, P);

        PackedMersenne31Neon::from_vector(output)
    }
}

/// We hard-code multiplication by the diagonal minus 1 of our internal matrix (1 + Diag(V)).
///
/// For Mersenne31 with WIDTH = 16, the diagonal minus 1 is:
/// `[-2] + 1 << [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16]`
///
/// i.e., the first entry is -2 and all other entries are powers of 2.
/// Note: state[0] is handled by the calling code (multiplied by -2).
#[inline(always)]
fn diagonal_mul_16(state: &mut [PackedMersenne31Neon; 16]) {
    // state[0] -> -2*state[0] is handled by the calling code.
    // state[1] -> 1*state[1] = state[1], no-op.
    state[2] = state[2] + state[2]; // *2 via addition (faster than shift)

    // For the remaining entries, use fast cyclic rotation.
    state[3] = mul_2exp_i::<2, 29>(state[3]);
    state[4] = mul_2exp_i::<3, 28>(state[4]);
    state[5] = mul_2exp_i::<4, 27>(state[5]);
    state[6] = mul_2exp_i::<5, 26>(state[6]);
    state[7] = mul_2exp_i::<6, 25>(state[7]);
    state[8] = mul_2exp_i::<7, 24>(state[8]);
    state[9] = mul_2exp_i::<8, 23>(state[9]);
    state[10] = mul_2exp_i::<10, 21>(state[10]);
    state[11] = mul_2exp_i::<12, 19>(state[11]);
    state[12] = mul_2exp_i::<13, 18>(state[12]);
    state[13] = mul_2exp_i::<14, 17>(state[13]);
    state[14] = mul_2exp_i::<15, 16>(state[14]);
    state[15] = mul_2exp_i::<16, 15>(state[15]);
}

/// We hard-code multiplication by the diagonal minus 1 of our internal matrix (1 + Diag(V)).
///
/// For Mersenne31 with WIDTH = 24, the diagonal minus 1 is:
/// `[-2] + 1 << [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]`
///
/// i.e., the first entry is -2 and all other entries are powers of 2.
/// Note: state[0] is handled by the calling code (multiplied by -2).
#[inline(always)]
fn diagonal_mul_24(state: &mut [PackedMersenne31Neon; 24]) {
    // state[0] -> -2*state[0] is handled by the calling code.
    // state[1] -> 1*state[1] = state[1], no-op.
    state[2] = state[2] + state[2]; // *2 via addition (faster than shift)

    // For the remaining entries, use fast cyclic rotation.
    state[3] = mul_2exp_i::<2, 29>(state[3]);
    state[4] = mul_2exp_i::<3, 28>(state[4]);
    state[5] = mul_2exp_i::<4, 27>(state[5]);
    state[6] = mul_2exp_i::<5, 26>(state[6]);
    state[7] = mul_2exp_i::<6, 25>(state[7]);
    state[8] = mul_2exp_i::<7, 24>(state[8]);
    state[9] = mul_2exp_i::<8, 23>(state[9]);
    state[10] = mul_2exp_i::<9, 22>(state[10]);
    state[11] = mul_2exp_i::<10, 21>(state[11]);
    state[12] = mul_2exp_i::<11, 20>(state[12]);
    state[13] = mul_2exp_i::<12, 19>(state[13]);
    state[14] = mul_2exp_i::<13, 18>(state[14]);
    state[15] = mul_2exp_i::<14, 17>(state[15]);
    state[16] = mul_2exp_i::<15, 16>(state[16]);
    state[17] = mul_2exp_i::<16, 15>(state[17]);
    state[18] = mul_2exp_i::<17, 14>(state[18]);
    state[19] = mul_2exp_i::<18, 13>(state[19]);
    state[20] = mul_2exp_i::<19, 12>(state[20]);
    state[21] = mul_2exp_i::<20, 11>(state[21]);
    state[22] = mul_2exp_i::<21, 10>(state[22]);
    state[23] = mul_2exp_i::<22, 9>(state[23]);
}

/// Compute the fused AddRoundConstant and S-Box operation: `x -> (x + rc)^5`.
///
/// # Optimization
/// 1. Adds `rc` (positive form). Result is in `[0, 2P]`.
/// 2. Performs "Min-Reduction": `min(sum, sum - P)`.
///    - If `sum < P`, `sum - P` wraps to a huge value, `min` selects `sum`.
///    - If `sum >= P`, `sum - P` is small, `min` selects `sum - P`.
///    - Cost: 2 instructions (`sub`, `min`).
/// 3. Calls `exp5`. Since input is now strictly `[0, P]`, `exp5` does not need
///    to handle signs or absolute values.
///
/// # Safety
/// - `input` must contain elements in canonical form `{0, ..., P}`.
/// - `rc` must contain round constants in positive form `{0, ..., P}`.
#[inline(always)]
fn add_rc_and_sbox(input: &mut PackedMersenne31Neon, rc: uint32x4_t) {
    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let input_vec = input.to_vector();

        // 1. Add round constant. Result in [0, 2P].
        let sum = aarch64::vaddq_u32(input_vec, rc);

        // 2. Fast Reduction to [0, P].
        // If sum >= P, we want (sum - P).
        // If sum < P, (sum - P) underflows to > P.
        // Unsigned min selects the correct modular result.
        let diff = aarch64::vsubq_u32(sum, P);
        let reduced = aarch64::vminq_u32(sum, diff);

        // 3. Apply S-box (optimized for positive inputs).
        let output = exp5(reduced);

        *input = PackedMersenne31Neon::from_vector(output);
    }
}

/// Compute a single Poseidon2 internal layer on a state of width 16.
///
/// The internal layer consists of:
/// 1. Add round constant to state[0] and apply S-box: `s0 -> (s0 + rc)^5`
/// 2. Apply linear layer: `s -> (1 + Diag(V)) * s`
///
/// The linear layer can be decomposed as:
/// - `sum = Σ s_i`
/// - `s_i -> sum + v_i * s_i`
///
/// We optimize by computing the diagonal multiplication on `s[1..]` in parallel
/// with the S-box on `s[0]`, exploiting instruction-level parallelism.
#[inline(always)]
fn internal_16(state: &mut [PackedMersenne31Neon; 16], rc: uint32x4_t) {
    // Apply AddRoundConstant and S-Box to state[0].
    add_rc_and_sbox(&mut state[0], rc);

    // Compute sum of state[1..] while S-box is executing.
    // This can run in parallel with the S-box computation.
    let sum_tail = PackedMersenne31Neon::sum_array::<15>(&state[1..]);

    // Total sum including state[0] after S-box.
    let sum = sum_tail + state[0];

    // Update state[0]: s0_new = sum + (-2) * s0 = sum - 2*s0 = (sum - s0) - s0 = sum_tail - s0
    state[0] = sum_tail - state[0];

    // Apply diagonal multiplication to the rest of the state.
    diagonal_mul_16(state);

    // Add sum to all elements except state[0] (which was handled above).
    state[1..].iter_mut().for_each(|x| *x += sum);
}

impl InternalLayer<PackedMersenne31Neon, 16, 5> for Poseidon2InternalLayerMersenne31 {
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [PackedMersenne31Neon; 16]) {
        self.packed_internal_constants
            .iter()
            .for_each(|&rc| internal_16(state, rc));
    }
}

/// Compute a single Poseidon2 internal layer on a state of width 24.
#[inline(always)]
fn internal_24(state: &mut [PackedMersenne31Neon; 24], rc: uint32x4_t) {
    // Apply AddRoundConstant and S-Box to state[0].
    add_rc_and_sbox(&mut state[0], rc);

    // Compute sum of state[1..] while S-box is executing.
    let sum_tail = PackedMersenne31Neon::sum_array::<23>(&state[1..]);

    // Total sum including state[0] after S-box.
    let sum = sum_tail + state[0];

    // Update state[0]: s0_new = sum_tail - s0 (because v_0 = -2)
    state[0] = sum_tail - state[0];

    // Apply diagonal multiplication to the rest of the state.
    diagonal_mul_24(state);

    // Add sum to all elements except state[0].
    state[1..].iter_mut().for_each(|x| *x += sum);
}

impl InternalLayer<PackedMersenne31Neon, 24, 5> for Poseidon2InternalLayerMersenne31 {
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [PackedMersenne31Neon; 24]) {
        self.packed_internal_constants
            .iter()
            .for_each(|&rc| internal_24(state, rc));
    }
}

impl<const WIDTH: usize> ExternalLayer<PackedMersenne31Neon, WIDTH, 5>
    for Poseidon2ExternalLayerMersenne31<WIDTH>
{
    /// Perform the initial external layers of the Poseidon2 permutation on the given state.
    fn permute_state_initial(&self, state: &mut [PackedMersenne31Neon; WIDTH]) {
        external_initial_permute_state(
            state,
            &self.packed_initial_external_constants,
            add_rc_and_sbox,
            &MDSMat4,
        );
    }

    /// Perform the terminal external layers of the Poseidon2 permutation on the given state.
    fn permute_state_terminal(&self, state: &mut [PackedMersenne31Neon; WIDTH]) {
        external_terminal_permute_state(
            state,
            &self.packed_terminal_external_constants,
            add_rc_and_sbox,
            &MDSMat4,
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::Poseidon2Mersenne31;

    type F = Mersenne31;
    type Perm16 = Poseidon2Mersenne31<16>;
    type Perm24 = Poseidon2Mersenne31<24>;

    /// Test that the output is the same as the scalar version on a random input of length 16.
    #[test]
    fn test_neon_poseidon2_width_16() {
        let mut rng = SmallRng::seed_from_u64(1);

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(&mut rng);

        let input: [F; 16] = rng.random();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut neon_input = input.map(Into::<PackedMersenne31Neon>::into);
        poseidon2.permute_mut(&mut neon_input);

        let neon_output = neon_input.map(|x| x.0[0]);

        assert_eq!(neon_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input of length 24.
    #[test]
    fn test_neon_poseidon2_width_24() {
        let mut rng = SmallRng::seed_from_u64(1);

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(&mut rng);

        let input: [F; 24] = rng.random();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut neon_input = input.map(Into::<PackedMersenne31Neon>::into);
        poseidon2.permute_mut(&mut neon_input);

        let neon_output = neon_input.map(|x| x.0[0]);

        assert_eq!(neon_output, expected);
    }

    /// Test mul_2exp_i with known values.
    #[test]
    fn test_mul_2exp_i() {
        // Test multiplication by 2^4 = 16
        let input = PackedMersenne31Neon::from(Mersenne31::new(5));
        let output = mul_2exp_i::<4, 27>(input);
        assert_eq!(output.0[0], Mersenne31::new(80)); // 5 * 16 = 80

        // Test multiplication by 2^8 = 256
        let input = PackedMersenne31Neon::from(Mersenne31::new(3));
        let output = mul_2exp_i::<8, 23>(input);
        assert_eq!(output.0[0], Mersenne31::new(768)); // 3 * 256 = 768
    }
}
