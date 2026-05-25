use core::array;
use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::columns::{KeccakCols, NUM_KECCAK_COLS};
use crate::constants::RC_BITS;
use crate::round_flags::eval_round_flags;
use crate::{BITS_PER_LIMB, NUM_ROUNDS_MIN_1, U64_LIMBS, generate_trace_rows};

/// Assumes the field size is at least 16 bits.
#[derive(Debug)]
pub struct KeccakAir {}

impl KeccakAir {
    pub fn generate_trace_rows<F: PrimeField64>(
        &self,
        num_hashes: usize,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F> {
        let mut rng = SmallRng::seed_from_u64(1);
        let inputs = (0..num_hashes).map(|_| rng.random()).collect();
        generate_trace_rows(inputs, extra_capacity_bits)
    }
}

impl<F> BaseAir<F> for KeccakAir {
    fn width(&self) -> usize {
        NUM_KECCAK_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for KeccakAir {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        eval_round_flags(builder);

        let main = builder.main();
        let local: &KeccakCols<AB::Var> = main.current_slice().borrow();
        let next: &KeccakCols<AB::Var> = main.next_slice().borrow();

        let first_step = local.step_flags[0];
        let final_step = local.step_flags[NUM_ROUNDS_MIN_1];
        let not_final_step = AB::Expr::ONE - final_step;

        // If this is the first step, the input A must match the preimage.
        for y in 0..5 {
            for x in 0..5 {
                builder
                    .when(first_step)
                    .assert_zeros::<U64_LIMBS, _>(array::from_fn(|limb| {
                        local.preimage[y][x][limb] - local.a[y][x][limb]
                    }));
            }
        }

        // If this is not the final step, the local and next preimages must match.
        for y in 0..5 {
            for x in 0..5 {
                builder
                    .when(not_final_step.clone())
                    .when_transition()
                    .assert_zeros::<U64_LIMBS, _>(array::from_fn(|limb| {
                        local.preimage[y][x][limb] - next.preimage[y][x][limb]
                    }));
            }
        }

        // The export flag must be 0 or 1.
        builder.assert_bool(local.export);

        // If this is not the final step, the export flag must be off.
        builder
            .when(not_final_step.clone())
            .assert_zero(local.export);

        // C'[x, z] = xor(C[x, z], C[x - 1, z], C[x + 1, z - 1]).
        // Note that if all entries of C are boolean, the arithmetic generalization
        // xor3 function only outputs 0, 1 and so this check also ensures that all
        // entries of C'[x, z] are boolean.
        for x in 0..5 {
            builder.assert_bools(local.c[x]);
            builder.assert_zeros::<64, _>(array::from_fn(|z| {
                let xor = local.c[x][z].into().xor3(
                    &local.c[(x + 4) % 5][z].into(),
                    &local.c[(x + 1) % 5][(z + 63) % 64].into(),
                );
                local.c_prime[x][z] - xor
            }));
        }

        // Check that the input limbs are consistent with A' and D.
        // A[x, y, z] = xor(A'[x, y, z], D[x, y, z])
        //            = xor(A'[x, y, z], C[x - 1, z], C[x + 1, z - 1])
        //            = xor(A'[x, y, z], C[x, z], C'[x, z]).
        // The last step is valid based on the identity we checked above.
        // It isn't required, but makes this check a bit cleaner.
        // We also check that all entries of A' are bools.
        // This has the side effect of also range checking the limbs of A.
        for y in 0..5 {
            for x in 0..5 {
                let get_bit = |z: usize| {
                    local.a_prime[y][x][z]
                        .into()
                        .xor3(&local.c[x][z].into(), &local.c_prime[x][z].into())
                };

                // Check that all entries of A'[y][x] are boolean.
                builder.assert_bools(local.a_prime[y][x]);

                builder.assert_zeros::<U64_LIMBS, _>(array::from_fn(|limb| {
                    let computed_limb = (limb * BITS_PER_LIMB..(limb + 1) * BITS_PER_LIMB)
                        .rev()
                        .fold(AB::Expr::ZERO, |acc, z| {
                            // Check to ensure all entries of A' are bools.
                            acc.double() + get_bit(z)
                        });
                    computed_limb - local.a[y][x][limb]
                }));
            }
        }

        // xor_{i=0}^4 A'[x, i, z] = C'[x, z], so for each x, z,
        // diff * (diff - 2) * (diff - 4) = 0, where
        // diff = sum_{i=0}^4 A'[x, i, z] - C'[x, z]
        for x in 0..5 {
            let four = AB::Expr::TWO.double();
            builder.assert_zeros::<64, _>(array::from_fn(|z| {
                let sum: AB::Expr = (0..5).map(|y| local.a_prime[y][x][z].into()).sum();
                let diff = sum - local.c_prime[x][z];
                diff.clone() * (diff.clone() - AB::Expr::TWO) * (diff - four.clone())
            }));
        }

        // A''[x, y] = xor(B[x, y], andn(B[x + 1, y], B[x + 2, y])).
        // As B is a rotation of A', all entries must be bools and so
        // this check also range checks A''.
        for y in 0..5 {
            for x in 0..5 {
                let get_bit = |z| {
                    let andn = local
                        .b((x + 1) % 5, y, z)
                        .into()
                        .andn(&local.b((x + 2) % 5, y, z).into());
                    andn.xor(&local.b(x, y, z).into())
                };
                builder.assert_zeros::<U64_LIMBS, _>(array::from_fn(|limb| {
                    let computed_limb = (limb * BITS_PER_LIMB..(limb + 1) * BITS_PER_LIMB)
                        .rev()
                        .fold(AB::Expr::ZERO, |acc, z| acc.double() + get_bit(z));
                    computed_limb - local.a_prime_prime[y][x][limb]
                }));
            }
        }

        // A'''[0, 0] = A''[0, 0] XOR RC
        // Check to ensure the bits of A''[0, 0] are boolean.
        builder.assert_bools(local.a_prime_prime_0_0_bits);
        builder.assert_zeros::<U64_LIMBS, _>(array::from_fn(|limb| {
            let computed_a_prime_prime_0_0_limb = (limb * BITS_PER_LIMB
                ..(limb + 1) * BITS_PER_LIMB)
                .rev()
                .fold(AB::Expr::ZERO, |acc, z| {
                    acc.double() + local.a_prime_prime_0_0_bits[z]
                });
            computed_a_prime_prime_0_0_limb - local.a_prime_prime[0][0][limb]
        }));

        let get_xored_bit = |i| {
            let mut rc_bit_i = AB::Expr::ZERO;
            for (rc_bits_r, &step_flag) in RC_BITS.iter().zip(local.step_flags.iter()) {
                let this_round_constant = AB::Expr::from_bool(rc_bits_r[i] != 0);
                rc_bit_i += step_flag * this_round_constant;
            }

            rc_bit_i.xor(&AB::Expr::from(local.a_prime_prime_0_0_bits[i]))
        };

        builder.assert_zeros::<U64_LIMBS, _>(array::from_fn(|limb| {
            let computed_a_prime_prime_prime_0_0_limb = (limb * BITS_PER_LIMB
                ..(limb + 1) * BITS_PER_LIMB)
                .rev()
                .fold(AB::Expr::ZERO, |acc, z| acc.double() + get_xored_bit(z));
            computed_a_prime_prime_prime_0_0_limb - local.a_prime_prime_prime_0_0_limbs[limb]
        }));

        // Enforce that this round's output equals the next round's input.
        for x in 0..5 {
            for y in 0..5 {
                builder
                    .when_transition()
                    .when(not_final_step.clone())
                    .assert_zeros::<U64_LIMBS, _>(array::from_fn(|limb| {
                        local.a_prime_prime_prime(y, x, limb) - next.a[y][x][limb]
                    }));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_air::{check_all_constraints, check_constraints};
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;
    use proptest::prelude::*;

    use super::*;
    use crate::columns::KECCAK_COL_MAP;

    type F = Goldilocks;

    fn valid_trace(input: [u64; 25]) -> RowMajorMatrix<F> {
        generate_trace_rows(vec![input], 0)
    }

    fn trace_val(trace: &mut RowMajorMatrix<F>, row: usize, col: usize) -> &mut F {
        let w = trace.width;
        &mut trace.values[row * w + col]
    }

    fn flip_bit(val: &mut F) {
        *val = if *val == F::ZERO { F::ONE } else { F::ZERO };
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn prop_eval_satisfied_for_random_input(input in any::<[u64; 25]>()) {
            let trace = valid_trace(input);
            // Walk every row, evaluate all ~2 000 constraints.
            check_constraints(&KeccakAir {}, &trace, &[]);
        }

        #[test]
        fn prop_eval_satisfied_batch(
            a in any::<[u64; 25]>(),
            b in any::<[u64; 25]>(),
        ) {
            // Trace: rows [0..24) = perm(a), rows [24..48) = perm(b).
            // Exercises the row-23 → row-24 boundary where the
            // final-step flag must gate off the transition constraint.
            let trace = generate_trace_rows::<F>(vec![a, b], 0);
            check_constraints(&KeccakAir {}, &trace, &[]);
        }
    }

    #[test]
    fn test_corrupted_step_flag_detected() {
        let trace = &mut valid_trace([0u64; 25]);

        // Mutation: swap step_flags[0] and step_flags[1] on row 0.
        //
        //     valid:   step_flags = [1, 0, 0, ..., 0]   (round 0 active)
        //     corrupt: step_flags = [0, 1, 0, ..., 0]   (round 1 active)
        //
        // Violates the first-row constraint: step_flags[0] must be 1.
        *trace_val(trace, 0, KECCAK_COL_MAP.step_flags[0]) = F::ZERO;
        *trace_val(trace, 0, KECCAK_COL_MAP.step_flags[1]) = F::ONE;

        let report = check_all_constraints(&KeccakAir {}, trace, &[], Some(10));
        assert!(!report.is_ok());
    }

    #[test]
    fn test_corrupted_preimage_detected() {
        let trace = &mut valid_trace([0u64; 25]);

        // Mutation: change preimage[0][0][0] at row 1.
        //
        //     row 0: preimage[0][0][0] = 0x0000
        //     row 1: preimage[0][0][0] = 0xBEEF  ← corrupt
        //
        // Violates: local.preimage == next.preimage (when not final step).
        *trace_val(trace, 1, KECCAK_COL_MAP.preimage[0][0][0]) = F::from_u16(0xBEEF);

        let report = check_all_constraints(&KeccakAir {}, trace, &[], Some(10));
        assert!(!report.is_ok());
    }

    #[test]
    fn test_corrupted_c_boolean_detected() {
        let trace = &mut valid_trace([0u64; 25]);

        // Mutation: set C[0][0] = 2.
        //
        //     valid:   C[0][0] in {0, 1}
        //     corrupt: C[0][0] = 2
        //
        // Violates: assert_bools(local.c[x]).
        *trace_val(trace, 0, KECCAK_COL_MAP.c[0][0]) = F::TWO;

        let report = check_all_constraints(&KeccakAir {}, trace, &[], Some(10));
        assert!(!report.is_ok());
    }

    #[test]
    fn test_corrupted_a_prime_bit_detected() {
        let trace = &mut valid_trace([0u64; 25]);

        // Mutation: flip A'[0][0][0] (one theta-output bit).
        //
        // Breaks three constraints at once:
        // - Boolean check on A'.
        // - Limb reconstruction: sum of bits != stored limb.
        // - Parity: sum(A'[x, 0..5, z]) not in {0, 2, 4}.
        flip_bit(trace_val(trace, 0, KECCAK_COL_MAP.a_prime[0][0][0]));

        let report = check_all_constraints(&KeccakAir {}, trace, &[], Some(10));
        assert!(!report.is_ok());
    }

    #[test]
    fn test_corrupted_a_prime_prime_limb_detected() {
        let trace = &mut valid_trace([0u64; 25]);

        // Mutation: A''[0][0][0] += 1.
        //
        // The A' bits still reconstruct to the old chi result,
        // but the stored limb is now old + 1 → mismatch.
        *trace_val(trace, 0, KECCAK_COL_MAP.a_prime_prime[0][0][0]) += F::ONE;

        let report = check_all_constraints(&KeccakAir {}, trace, &[], Some(10));
        assert!(!report.is_ok());
    }

    #[test]
    fn test_corrupted_a_prime_prime_0_0_bit_detected() {
        let trace = &mut valid_trace([0u64; 25]);

        // Mutation: flip bit 0 of A''[0,0].
        //
        // The bit-decomposition no longer reconstructs to the
        // stored A''[0][0] limb.
        flip_bit(trace_val(
            trace,
            0,
            KECCAK_COL_MAP.a_prime_prime_0_0_bits[0],
        ));

        let report = check_all_constraints(&KeccakAir {}, trace, &[], Some(10));
        assert!(!report.is_ok());
    }

    #[test]
    fn test_corrupted_a_prime_prime_prime_limb_detected() {
        let trace = &mut valid_trace([0u64; 25]);

        // Mutation: A'''[0,0] limb 0 += 1.
        //
        // Breaks two constraints:
        // - Iota: A'''[0,0] != A''[0,0] XOR RC.
        // - Transition: A'''[0,0] != next row's A[0][0].
        *trace_val(trace, 0, KECCAK_COL_MAP.a_prime_prime_prime_0_0_limbs[0]) += F::ONE;

        let report = check_all_constraints(&KeccakAir {}, trace, &[], Some(10));
        assert!(!report.is_ok());
    }

    #[test]
    fn test_corrupted_round_output_to_next_input_detected() {
        let trace = &mut valid_trace([0u64; 25]);

        // Mutation: A[0][0][0] at row 1 += 1.
        //
        //     row 0 output: A'''[0][0][0] = V
        //     row 1 input:  A[0][0][0]    = V + 1  ← corrupt
        //
        // Violates: A'''[y,x,limb] == next.a[y][x][limb].
        *trace_val(trace, 1, KECCAK_COL_MAP.a[0][0][0]) += F::ONE;

        let report = check_all_constraints(&KeccakAir {}, trace, &[], Some(10));
        assert!(!report.is_ok());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_any_c_bit_corruption_detected(
            input in any::<[u64; 25]>(),
            x in 0..5usize,
            z in 0..64usize,
        ) {
            // Flip C[x][z] at row 0.  5 × 64 = 320 bits total.
            let trace = &mut valid_trace(input);
            flip_bit(trace_val(trace, 0, KECCAK_COL_MAP.c[x][z]));

            let report = check_all_constraints(&KeccakAir {}, trace, &[], Some(5));
            prop_assert!(!report.is_ok());
        }

        #[test]
        fn prop_any_a_prime_bit_corruption_detected(
            input in any::<[u64; 25]>(),
            y in 0..5usize,
            x in 0..5usize,
            z in 0..64usize,
        ) {
            // Flip A'[y][x][z] at row 0.  5 × 5 × 64 = 1 600 bits total.
            let trace = &mut valid_trace(input);
            flip_bit(trace_val(trace, 0, KECCAK_COL_MAP.a_prime[y][x][z]));

            let report = check_all_constraints(&KeccakAir {}, trace, &[], Some(5));
            prop_assert!(!report.is_ok());
        }
    }
}
