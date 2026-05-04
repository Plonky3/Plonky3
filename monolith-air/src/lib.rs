//! Algebraic Intermediate Representation (AIR) for the Monolith permutation.
//!
//! # Overview
//!
//! This crate arithmetizes the [Monolith] hash permutation for use in STARK proof
//! systems. Each row of the execution trace computes **one full permutation** of
//! the state.
//!
//! [Monolith]: https://eprint.iacr.org/2023/1025

#![no_std]

extern crate alloc;

mod air;
mod columns;
mod generation;

pub use air::*;
pub use columns::*;
pub use generation::*;

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::borrow::Borrow;
    use core::mem::size_of;

    use p3_air::symbolic::{AirLayout, get_max_constraint_degree};
    use p3_air::{BaseAir, check_constraints};
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
    use p3_matrix::Matrix;
    use p3_mersenne_31::Mersenne31;
    use p3_monolith::MonolithMersenne31;
    use p3_monolith::bars::mersenne31::MonolithBarsM31;
    use p3_monolith::mds::mersenne31::MonolithMdsMatrixMersenne31;
    use p3_symmetric::Permutation;
    use proptest::prelude::*;

    use super::*;

    type F = Mersenne31;

    // Monolith-31 parameters for WIDTH=16.
    const WIDTH: usize = 16;
    const NUM_FULL_ROUNDS: usize = 5;
    const NUM_BARS: usize = 8;
    const FIELD_BITS: usize = 31;

    /// Build a Monolith-31 AIR instance and return it with the Bars impl.
    fn build_monolith_air() -> (
        MonolithAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>,
        MonolithBarsM31,
    ) {
        let bars = MonolithBarsM31;
        let mds = MonolithMdsMatrixMersenne31::<6>;

        // Extract MDS matrix before moving mds into the Monolith constructor.
        let mds_matrix =
            MonolithAir::<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>::extract_mds_matrix(
                &mds,
            );

        let monolith = MonolithMersenne31::new(bars, mds);

        let air = MonolithAir::new(monolith.round_constants, mds_matrix, MERSENNE31_LIMB_BITS);
        (air, bars)
    }

    #[test]
    fn test_column_layout() {
        // num_cols() computes the expected column count from const generics.
        let expected = num_cols::<WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>();

        // size_of with T=u8 gives the actual struct size in bytes = number of fields.
        let actual = size_of::<MonolithCols<u8, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>>();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_column_count_mersenne31_16() {
        // Manually compute the expected column count for Mersenne31 WIDTH=16.
        //
        // Per round: FIELD_BITS * NUM_BARS (bits) + NUM_BARS (bar outputs) + WIDTH (post)
        let expected = WIDTH + (NUM_FULL_ROUNDS + 1) * (FIELD_BITS * NUM_BARS + NUM_BARS + WIDTH);
        let actual = num_cols::<WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_monolith_air_degree_mersenne31() {
        let (air, _bars) = build_monolith_air();

        let layout = AirLayout {
            main_width: air.width(),
            ..Default::default()
        };
        let degree = get_max_constraint_degree::<F, _>(&air, layout);

        // The maximum constraint degree is 4, from the chi S-box XOR formula:
        //   out = in + prod - 2 * in * prod
        // where prod = (1 - in_{i+1}) * in_{i+2} * in_{i+3} has degree 3,
        // and in * prod has degree 4.
        assert_eq!(degree, 4, "Expected constraint degree 4 for Monolith-31");
    }

    #[test]
    fn test_known_answer_mersenne31_16() {
        let (air, bars) = build_monolith_air();

        // Generate a single permutation trace with sequential input [0..15].
        let input: [F; WIDTH] = core::array::from_fn(F::from_usize);
        let inputs = vec![input];
        let trace = generate_trace_rows::<_, _, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>(
            inputs, &air, &bars, 0,
        );

        // One input → one row in the trace.
        assert_eq!(trace.height(), 1);

        // Compute the expected output using the reference Monolith permutation.
        let monolith: MonolithMersenne31<_, WIDTH, NUM_FULL_ROUNDS> =
            MonolithMersenne31::new(MonolithBarsM31, MonolithMdsMatrixMersenne31::<6>);
        let mut expected_state = input;
        monolith.permute_mut(&mut expected_state);

        // Extract the final post-state from the trace (last round's post).
        let local = trace.row_slice(0).expect("Trace is empty");
        let row: &MonolithCols<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS> = (*local).borrow();
        let final_post = &row.final_round.post;

        // The AIR trace output must match the reference implementation.
        assert_eq!(
            final_post, &expected_state,
            "AIR trace output does not match reference Monolith permutation"
        );
    }

    #[test]
    fn test_known_answer_paper_vectors() {
        let (air, bars) = build_monolith_air();

        // Input: [0, 1, 2, ..., 15] — standard test vector.
        let input: [F; WIDTH] = core::array::from_fn(F::from_usize);
        let inputs = vec![input];
        let trace = generate_trace_rows::<_, _, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>(
            inputs, &air, &bars, 0,
        );

        let local = trace.row_slice(0).expect("Trace is empty");
        let row: &MonolithCols<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS> = (*local).borrow();

        // Expected output from the Monolith-31 reference test vector
        // (validated against the HorizenLabs reference implementation).
        let expected = [
            609156607, 290107110, 1900746598, 1734707571, 2050994835, 1648553244, 1307647296,
            1941164548, 1707113065, 1477714255, 1170160793, 93800695, 769879348, 375548503,
            1989726444, 1349325635,
        ]
        .map(F::from_u32);

        assert_eq!(
            row.final_round.post, expected,
            "Output does not match paper/reference test vectors"
        );
    }

    #[test]
    fn test_constraint_satisfaction_sequential_inputs() {
        let (air, bars) = build_monolith_air();

        // Test with specific sequential inputs [0..15], [16..31], etc.
        let inputs: alloc::vec::Vec<[F; WIDTH]> = (0..16)
            .map(|batch| core::array::from_fn(|i| F::from_usize(batch * WIDTH + i)))
            .collect();
        let trace = generate_trace_rows::<_, _, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>(
            inputs, &air, &bars, 0,
        );

        check_constraints(&air, &trace, &[]);
    }

    #[test]
    fn test_constraint_satisfaction_zero_input() {
        let (air, bars) = build_monolith_air();

        // Test with all-zero input (edge case for Bars fixed points).
        let inputs = vec![[F::ZERO; WIDTH]];
        let trace = generate_trace_rows::<_, _, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>(
            inputs, &air, &bars, 0,
        );

        check_constraints(&air, &trace, &[]);
    }

    #[test]
    fn test_constraint_satisfaction_max_input() {
        let (air, bars) = build_monolith_air();

        // Test with maximum field values (p-1 = 2^31 - 2).
        let max_val = F::from_u32(0x7FFF_FFFE);
        let inputs = vec![[max_val; WIDTH]];
        let trace = generate_trace_rows::<_, _, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>(
            inputs, &air, &bars, 0,
        );

        check_constraints(&air, &trace, &[]);
    }

    proptest! {
        #[test]
        fn proptest_constraint_satisfaction_random(
            s0  in 0u32..Mersenne31::ORDER_U32,
            s1  in 0u32..Mersenne31::ORDER_U32,
            s2  in 0u32..Mersenne31::ORDER_U32,
            s3  in 0u32..Mersenne31::ORDER_U32,
            s4  in 0u32..Mersenne31::ORDER_U32,
            s5  in 0u32..Mersenne31::ORDER_U32,
            s6  in 0u32..Mersenne31::ORDER_U32,
            s7  in 0u32..Mersenne31::ORDER_U32,
            s8  in 0u32..Mersenne31::ORDER_U32,
            s9  in 0u32..Mersenne31::ORDER_U32,
            s10 in 0u32..Mersenne31::ORDER_U32,
            s11 in 0u32..Mersenne31::ORDER_U32,
            s12 in 0u32..Mersenne31::ORDER_U32,
            s13 in 0u32..Mersenne31::ORDER_U32,
            s14 in 0u32..Mersenne31::ORDER_U32,
            s15 in 0u32..Mersenne31::ORDER_U32,
        ) {
            let (air, bars) = build_monolith_air();

            let input: [F; WIDTH] = [
                s0, s1, s2, s3, s4, s5, s6, s7,
                s8, s9, s10, s11, s12, s13, s14, s15,
            ].map(F::from_u32);

            let trace = generate_trace_rows::<
                _, _, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS,
            >(vec![input], &air, &bars, 0);

            check_constraints(&air, &trace, &[]);
        }
    }
}
