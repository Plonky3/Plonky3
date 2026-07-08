use alloc::vec::Vec;
use core::mem::MaybeUninit;

use p3_field::{PackedValue, PrimeCharacteristicRing, PrimeField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_maybe_rayon::prelude::*;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use tracing::instrument;

use crate::columns::{Poseidon2Cols, num_cols};
use crate::{FullRound, PartialRound, RoundConstants, SBox};

#[instrument(name = "generate vectorized Poseidon2 trace", skip_all)]
pub fn generate_vectorized_trace_rows<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
>(
    inputs: Vec<[F; WIDTH]>,
    round_constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    let n = inputs.len();
    assert!(
        n.is_multiple_of(VECTOR_LEN) && (n / VECTOR_LEN).is_power_of_two(),
        "Callers expected to pad inputs to VECTOR_LEN times a power of two"
    );

    let nrows = n.div_ceil(VECTOR_LEN);
    let ncols = num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()
        * VECTOR_LEN;
    let mut vec = Vec::with_capacity((nrows * ncols) << extra_capacity_bits);
    let trace = &mut vec.spare_capacity_mut()[..nrows * ncols];
    let trace = RowMajorMatrixViewMut::new(trace, ncols);

    let (prefix, perms, suffix) = unsafe {
        trace.values.align_to_mut::<Poseidon2Cols<
            MaybeUninit<F>,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >>()
    };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(perms.len(), n);

    generate_perms::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >(perms, inputs, round_constants);

    unsafe {
        vec.set_len(nrows * ncols);
    }

    RowMajorMatrix::new(vec, ncols)
}

// TODO: Take generic iterable
#[instrument(name = "generate Poseidon2 trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    inputs: Vec<[F; WIDTH]>,
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    let n = inputs.len();
    assert!(
        n.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );

    let ncols = num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();
    let mut vec = Vec::with_capacity((n * ncols) << extra_capacity_bits);
    let trace = &mut vec.spare_capacity_mut()[..n * ncols];
    let trace = RowMajorMatrixViewMut::new(trace, ncols);

    let (prefix, perms, suffix) = unsafe {
        trace.values.align_to_mut::<Poseidon2Cols<
            MaybeUninit<F>,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >>()
    };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(perms.len(), n);

    generate_perms::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >(perms, inputs, constants);

    unsafe {
        vec.set_len(n * ncols);
    }

    RowMajorMatrix::new(vec, ncols)
}

/// Fill every permutation's trace columns from its input.
///
/// - With SIMD packing available, process `F::Packing::WIDTH` permutations per task.
/// - A trailing group of fewer than that many permutations cannot fill a SIMD vector.
/// - That short group, and every permutation on a target without packing, runs one at a time.
///
/// The packed and one-at-a-time paths write identical trace cells, so the choice is pure throughput.
fn generate_perms<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    perms: &mut [Poseidon2Cols<
        MaybeUninit<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >],
    inputs: Vec<[F; WIDTH]>,
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
) {
    let packing_width = F::Packing::WIDTH;

    // Width 1 means the target has no SIMD packing: batching would add overhead with no gain.
    if packing_width == 1 {
        perms.par_iter_mut().zip(inputs).for_each(|(perm, input)| {
            generate_trace_rows_for_perm::<
                F,
                LinearLayers,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >(perm, input, constants);
        });
        return;
    }

    // Split the permutations into groups of `packing_width`.
    //
    //     inputs:  [ p0 p1 p2 p3 | p4 p5 p6 p7 | p8 p9 ]   (packing_width = 4)
    //               \__ batch __/ \__ batch __/ \ tail /
    //
    // Full groups run through the packed round functions.
    // The short final group appears only when the count is not a multiple of the width.
    perms
        .par_chunks_mut(packing_width)
        .zip(inputs.par_chunks(packing_width))
        .for_each(|(perm_chunk, input_chunk)| {
            if perm_chunk.len() == packing_width {
                // A full SIMD vector's worth of permutations: one packed pass covers all lanes.
                generate_trace_rows_for_perm_batch::<
                    F,
                    LinearLayers,
                    WIDTH,
                    SBOX_DEGREE,
                    SBOX_REGISTERS,
                    HALF_FULL_ROUNDS,
                    PARTIAL_ROUNDS,
                >(perm_chunk, input_chunk, constants);
            } else {
                // Fewer than a full vector: fall back to one permutation at a time.
                for (perm, &input) in perm_chunk.iter_mut().zip(input_chunk) {
                    generate_trace_rows_for_perm::<
                        F,
                        LinearLayers,
                        WIDTH,
                        SBOX_DEGREE,
                        SBOX_REGISTERS,
                        HALF_FULL_ROUNDS,
                        PARTIAL_ROUNDS,
                    >(perm, input, constants);
                }
            }
        });
}

/// `rows` will normally consist of 24 rows, with an exception for the final row.
pub fn generate_trace_rows_for_perm<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    perm: &mut Poseidon2Cols<
        MaybeUninit<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
    mut state: [F; WIDTH],
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
) {
    perm.inputs
        .iter_mut()
        .zip(state.iter())
        .for_each(|(input, &x)| {
            input.write(x);
        });

    LinearLayers::external_linear_layer(&mut state);

    for (full_round, constants) in perm
        .beginning_full_rounds
        .iter_mut()
        .zip(&constants.beginning_full_round_constants)
    {
        generate_full_round::<_, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state, full_round, constants,
        );
    }

    for (partial_round, constant) in perm
        .partial_rounds
        .iter_mut()
        .zip(&constants.partial_round_constants)
    {
        generate_partial_round::<_, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            partial_round,
            *constant,
        );
    }

    for (full_round, constants) in perm
        .ending_full_rounds
        .iter_mut()
        .zip(&constants.ending_full_round_constants)
    {
        generate_full_round::<_, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state, full_round, constants,
        );
    }
}

/// Generate `F::Packing::WIDTH` permutations at once, running the round functions over
/// `F::Packing` so that every state element processes all lanes with one field operation
/// instead of one permutation at a time.
#[inline]
fn generate_trace_rows_for_perm_batch<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    perms: &mut [Poseidon2Cols<
        MaybeUninit<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >],
    inputs: &[[F; WIDTH]],
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
) {
    let width = F::Packing::WIDTH;
    debug_assert_eq!(perms.len(), width);
    debug_assert_eq!(inputs.len(), width);

    for (perm, input) in perms.iter_mut().zip(inputs) {
        perm.inputs.iter_mut().zip(input).for_each(|(c, &x)| {
            c.write(x);
        });
    }

    let mut state: [F::Packing; WIDTH] = F::Packing::pack_columns(inputs);

    LinearLayers::external_linear_layer(&mut state);

    for (round, round_constants) in constants.beginning_full_round_constants.iter().enumerate() {
        generate_full_round_packed::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(
            &mut state,
            perms,
            |p| &mut p.beginning_full_rounds[round],
            round_constants,
        );
    }

    for (round, &round_constant) in constants.partial_round_constants.iter().enumerate() {
        generate_partial_round_packed::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(&mut state, perms, round, round_constant);
    }

    for (round, round_constants) in constants.ending_full_round_constants.iter().enumerate() {
        generate_full_round_packed::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(
            &mut state,
            perms,
            |p| &mut p.ending_full_rounds[round],
            round_constants,
        );
    }
}

#[inline]
fn generate_full_round_packed<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    state: &mut [F::Packing; WIDTH],
    perms: &mut [Poseidon2Cols<
        MaybeUninit<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >],
    full_round: impl Fn(
        &mut Poseidon2Cols<
            MaybeUninit<F>,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >,
    ) -> &mut FullRound<MaybeUninit<F>, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[F; WIDTH],
) {
    let mut sbox_registers = [[F::Packing::ZERO; SBOX_REGISTERS]; WIDTH];
    for ((s, &rc), regs) in state
        .iter_mut()
        .zip(round_constants.iter())
        .zip(sbox_registers.iter_mut())
    {
        *s += rc;
        *regs = generate_sbox_packed::<F, SBOX_DEGREE, SBOX_REGISTERS>(s);
    }

    LinearLayers::external_linear_layer(state);

    for (lane, perm) in perms.iter_mut().enumerate() {
        let full_round = full_round(perm);
        for i in 0..WIDTH {
            for (r, reg) in sbox_registers[i].iter().enumerate() {
                full_round.sbox[i].0[r].write(reg.as_slice()[lane]);
            }
            full_round.post[i].write(state[i].as_slice()[lane]);
        }
    }
}

#[inline]
fn generate_partial_round_packed<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    state: &mut [F::Packing; WIDTH],
    perms: &mut [Poseidon2Cols<
        MaybeUninit<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >],
    round: usize,
    round_constant: F,
) {
    state[0] += round_constant;
    let sbox_registers = generate_sbox_packed::<F, SBOX_DEGREE, SBOX_REGISTERS>(&mut state[0]);

    for (lane, perm) in perms.iter_mut().enumerate() {
        let partial_round = &mut perm.partial_rounds[round];
        for (r, reg) in sbox_registers.iter().enumerate() {
            partial_round.sbox.0[r].write(reg.as_slice()[lane]);
        }
        partial_round.post_sbox.write(state[0].as_slice()[lane]);
    }

    LinearLayers::internal_linear_layer(state);
}

/// Apply the S-box over `F::Packing::WIDTH` permutations at once and return the packed powers.
///
/// The caller unpacks each lane of the returned powers into that permutation's own trace columns.
#[inline]
fn generate_sbox_packed<F: PrimeField, const DEGREE: u64, const REGISTERS: usize>(
    x: &mut F::Packing,
) -> [F::Packing; REGISTERS] {
    // Evaluate x^DEGREE lane-wise and collect the packed intermediate powers.
    let (output, registers) = sbox_with_registers::<F::Packing, DEGREE, REGISTERS>(*x);

    // Advance the packed state to the S-box output.
    *x = output;

    registers
}

#[inline]
fn generate_full_round<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [F; WIDTH],
    full_round: &mut FullRound<MaybeUninit<F>, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[F; WIDTH],
) {
    // Combine addition of round constants and S-box application in a single loop
    for ((state_i, const_i), sbox_i) in state
        .iter_mut()
        .zip(round_constants.iter())
        .zip(full_round.sbox.iter_mut())
    {
        *state_i += *const_i;
        generate_sbox(sbox_i, state_i);
    }

    LinearLayers::external_linear_layer(state);
    full_round
        .post
        .iter_mut()
        .zip(*state)
        .for_each(|(post, x)| {
            post.write(x);
        });
}

#[inline]
fn generate_partial_round<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [F; WIDTH],
    partial_round: &mut PartialRound<MaybeUninit<F>, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constant: F,
) {
    state[0] += round_constant;
    generate_sbox(&mut partial_round.sbox, &mut state[0]);
    partial_round.post_sbox.write(state[0]);
    LinearLayers::internal_linear_layer(state);
}

/// Apply the S-box in place and write its committed intermediate powers into the trace row.
#[inline]
fn generate_sbox<F: PrimeField, const DEGREE: u64, const REGISTERS: usize>(
    sbox: &mut SBox<MaybeUninit<F>, DEGREE, REGISTERS>,
    x: &mut F,
) {
    // Evaluate x^DEGREE and collect the intermediate powers.
    let (output, registers) = sbox_with_registers::<F, DEGREE, REGISTERS>(*x);

    // Advance the state to the S-box output.
    *x = output;

    // Persist each intermediate power into its own trace column.
    for (column, power) in sbox.0.iter_mut().zip(registers) {
        column.write(power);
    }
}

/// Evaluate the S-box `x -> x^DEGREE` and return the intermediate powers the trace commits to.
///
/// - The non-linear step of the permutation raises each state element to a fixed power.
/// - A single constraint can only certify a power up to the AIR's max degree.
/// - Splitting the exponent across `REGISTERS` committed powers keeps every constraint at degree `DEGREE`.
///
/// This is the single source of truth for that split, shared by the scalar and packed round functions.
///
/// # Returns
/// - The S-box output `x^DEGREE`.
/// - The `REGISTERS` intermediate powers, ordered as the trace columns expect them.
///
/// # Panics
/// Panics unless `(DEGREE, REGISTERS)` is one of the supported pairs:
/// `(3, 0)`, `(5, 0)`, `(7, 0)`, `(5, 1)`, `(7, 1)`, `(11, 2)`.
#[inline]
fn sbox_with_registers<
    R: PrimeCharacteristicRing + Copy,
    const DEGREE: u64,
    const REGISTERS: usize,
>(
    x: R,
) -> (R, [R; REGISTERS]) {
    // Intermediate powers the verifier re-derives; stays empty when REGISTERS = 0.
    let mut registers = [R::ZERO; REGISTERS];

    let output = match (DEGREE, REGISTERS) {
        // x^3 is itself degree 3: no intermediate power to commit.
        (3, 0) => x.cube(),
        // x^5 and x^7 fit in one degree-5 / degree-7 constraint.
        (5, 0) => x.exp_const_u64::<5>(),
        (7, 0) => x.exp_const_u64::<7>(),
        // x^5 through x^3: commit x^3, then x^5 = x^3 * x^2.
        (5, 1) => {
            let x2 = x.square();
            let x3 = x2 * x;
            registers[0] = x3;
            x3 * x2
        }
        // x^7 through x^3: commit x^3, then x^7 = x^3 * x^3 * x.
        (7, 1) => {
            let x3 = x.cube();
            registers[0] = x3;
            x3 * x3 * x
        }
        // x^11 through x^3 and x^9: commit both, then x^11 = x^9 * x^2.
        (11, 2) => {
            let x2 = x.square();
            let x3 = x2 * x;
            let x9 = x3.cube();
            registers[0] = x3;
            registers[1] = x9;
            x9 * x2
        }
        // Any other pairing would let a constraint exceed degree DEGREE.
        _ => panic!("Unexpected (DEGREE, REGISTERS) of ({DEGREE}, {REGISTERS})"),
    };

    (output, registers)
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{
        BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS, BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16,
        BABYBEAR_S_BOX_DEGREE, BabyBear, GenericPoseidon2LinearLayersBabyBear,
    };
    use p3_koala_bear::{
        GenericPoseidon2LinearLayersKoalaBear, KOALABEAR_POSEIDON2_HALF_FULL_ROUNDS,
        KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_16, KOALABEAR_S_BOX_DEGREE, KoalaBear,
    };
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    /// Build the trace one permutation at a time, never touching the packed batch path.
    ///
    /// This mirrors the buffer setup of the public generator exactly.
    /// Any difference in the output then isolates the packed round functions, not the layout.
    fn reference_trace_rows<
        F: PrimeField,
        LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    >(
        inputs: Vec<[F; WIDTH]>,
        constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    ) -> RowMajorMatrix<F> {
        // One trace row per permutation.
        let n = inputs.len();
        let ncols =
            num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();

        // Reserve the flat backing store, then reinterpret it as typed permutation rows.
        let mut vec = Vec::with_capacity(n * ncols);
        let trace = &mut vec.spare_capacity_mut()[..n * ncols];
        let trace = RowMajorMatrixViewMut::new(trace, ncols);
        let (prefix, perms, suffix) = unsafe {
            trace.values.align_to_mut::<Poseidon2Cols<
                MaybeUninit<F>,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >>()
        };
        assert!(prefix.is_empty(), "Alignment should match");
        assert!(suffix.is_empty(), "Alignment should match");
        assert_eq!(perms.len(), n);

        // Force the one-at-a-time path for every permutation.
        perms.iter_mut().zip(inputs).for_each(|(perm, input)| {
            generate_trace_rows_for_perm::<
                F,
                LinearLayers,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >(perm, input, constants);
        });

        // Every cell is now initialized.
        unsafe {
            vec.set_len(n * ncols);
        }

        RowMajorMatrix::new(vec, ncols)
    }

    /// Assert that the packed and one-at-a-time generators produce the same trace.
    ///
    /// Packing is only a throughput transform, so both generators must agree cell for cell.
    /// The public generator takes the packed path when packing is available and the count fills full vectors.
    /// The reference always runs one permutation at a time.
    /// On a target with packing width above 1 this compares the packed path against the scalar path.
    fn check_packed_matches_scalar<
        F: PrimeField,
        LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    >(
        seed: u64,
        n: usize,
    ) where
        StandardUniform: Distribution<F> + Distribution<[F; WIDTH]>,
    {
        // Same seed drives both constants and inputs, so the run is deterministic.
        let mut rng = SmallRng::seed_from_u64(seed);
        let constants = RoundConstants::from_rng(&mut rng);
        let inputs: Vec<[F; WIDTH]> = (0..n).map(|_| rng.sample(StandardUniform)).collect();

        // Packed path (when SIMD is available) versus the one-at-a-time reference.
        let packed = generate_trace_rows::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(inputs.clone(), &constants, 0);
        let scalar = reference_trace_rows::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(inputs, &constants);

        assert_eq!(packed.values, scalar.values);
    }

    #[test]
    fn packed_matches_scalar_babybear() {
        // BabyBear uses the degree-7 S-box with one committed register (the x^7-through-x^3 arm).
        //
        //     n = 2  -> shorter than a NEON/AVX vector -> exercises the tail fallback
        //     n = 8  -> full vectors on width 4 and 8  -> exercises the packed batch path
        //     n = 16 -> multiple full vectors
        for &n in &[2, 8, 16] {
            check_packed_matches_scalar::<
                BabyBear,
                GenericPoseidon2LinearLayersBabyBear,
                16,
                BABYBEAR_S_BOX_DEGREE,
                1,
                BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS,
                BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16,
            >(n as u64, n);
        }
    }

    #[test]
    fn packed_matches_scalar_koalabear() {
        // KoalaBear uses the degree-3 S-box with zero registers (the direct x^3 arm).
        //
        // This covers the register-free branch of the shared S-box that BabyBear does not reach.
        for &n in &[2, 8, 16] {
            check_packed_matches_scalar::<
                KoalaBear,
                GenericPoseidon2LinearLayersKoalaBear,
                16,
                KOALABEAR_S_BOX_DEGREE,
                0,
                KOALABEAR_POSEIDON2_HALF_FULL_ROUNDS,
                KOALABEAR_POSEIDON2_PARTIAL_ROUNDS_16,
            >(n as u64, n);
        }
    }
}
