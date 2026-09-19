use core::array;
use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::columns::{KECCAK_BINARY_ROWS_PER_PERM, KeccakBinaryCols, NUM_KECCAK_BINARY_COLS};
use super::generation::generate_binary_trace_rows;
use super::rho_pi_source;
use crate::constants::RC_BITS;
use crate::{NUM_ROUNDS, NUM_ROUNDS_MIN_1};

/// Number of state bits in a row.
const NUM_STATE_BITS: usize = 1600;

/// Round-flag constraints: 25 on the first row, 23 round steps, the wrap, and exclusivity.
const NUM_FLAG_CONSTRAINTS: usize = KECCAK_BINARY_ROWS_PER_PERM + NUM_ROUNDS_MIN_1 + 2;

/// Round-flag constraints, then booleanity and the round map for every state bit.
const NUM_CONSTRAINTS: usize = NUM_FLAG_CONSTRAINTS + 2 * NUM_STATE_BITS;

/// AIR for the Keccak-f permutation over a field of characteristic 2.
///
/// Each row holds a 1600-bit state and a one-hot round flag (see [`KeccakBinaryCols`]).
/// A round row constrains the next row's state to the round map of its own state.
/// An output row leaves the next row's state free.
///
/// A round-23 row may also be followed directly by round 0, chaining two permutations
/// without an output row between them: the round-0 input is then the previous output.
///
/// The constraints read the next row but use no transition selector.
/// The last row of a valid trace is an output row, so both successor conventions hold:
///
/// ```text
///     wrap to row 0 : output row -> round 0
///     repeat last   : output row -> itself
/// ```
#[derive(Debug)]
pub struct KeccakBinaryAir {}

impl KeccakBinaryAir {
    /// Generate a trace over `num_hashes` fixed-seed random permutation inputs.
    ///
    /// This is for benches/examples only — it does not let callers supply the actual
    /// inputs being hashed. Use the free [`generate_binary_trace_rows`] function directly
    /// to prove specific inputs.
    ///
    /// # Panics
    ///
    /// - The field does not have characteristic 2.
    /// - `num_hashes` is 0.
    pub fn generate_random_trace_rows<F: Field>(
        &self,
        num_hashes: usize,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F> {
        let mut rng = SmallRng::seed_from_u64(1);
        let inputs = (0..num_hashes).map(|_| rng.random()).collect();
        generate_binary_trace_rows(inputs, extra_capacity_bits)
    }
}

impl<F> BaseAir<F> for KeccakBinaryAir {
    fn width(&self) -> usize {
        NUM_KECCAK_BINARY_COLS
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(3)
    }

    fn num_constraints(&self) -> Option<usize> {
        Some(NUM_CONSTRAINTS)
    }
}

impl<AB: AirBuilder> Air<AB> for KeccakBinaryAir {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &KeccakBinaryCols<AB::Var> = main.current_slice().borrow();
        let next: &KeccakBinaryCols<AB::Var> = main.next_slice().borrow();

        let flags = &local.round_flags;
        let next_flags = &next.round_flags;
        let output_flag = flags[NUM_ROUNDS];

        // The first row is the input row of round 0.
        builder.when_first_row().assert_one(flags[0]);
        builder
            .when_first_row()
            .assert_zeros::<NUM_ROUNDS, _>(flags[1..].try_into().unwrap());

        // Round r < 23 is followed by round r + 1.
        builder
            .assert_zeros::<NUM_ROUNDS_MIN_1, _>(array::from_fn(|r| next_flags[r + 1] + flags[r]));

        // Round 23 and an output row are each followed by round 0 or an output row.
        builder.assert_zero(
            next_flags[0] + next_flags[NUM_ROUNDS] + flags[NUM_ROUNDS_MIN_1] + output_flag,
        );

        // No row is both round 0 and an output row.
        //
        // From the one-hot first row, the flag constraints keep every successor one-hot:
        // its (round 0, output) pair sums to a bit and has product 0, so both entries are bits.
        builder.assert_zero(flags[0] * output_flag);

        // Every state bit is a bit.
        for plane in &local.a {
            for lane in plane {
                builder.assert_bools(*lane);
            }
        }

        // Theta:
        //     C[x][z]     = sum_y A[y][x][z]
        //     D[x][z]     = C[x - 1][z] + C[x + 1][z - 1]
        //     A'[y][x][z] = A[y][x][z] + D[x][z]
        let c: [[AB::Expr; 64]; 5] =
            array::from_fn(|x| array::from_fn(|z| (0..5).map(|y| local.a[y][x][z].into()).sum()));
        let d: [[AB::Expr; 64]; 5] = array::from_fn(|x| {
            array::from_fn(|z| c[(x + 4) % 5][z].clone() + c[(x + 1) % 5][(z + 63) % 64].clone())
        });
        let a_prime: [[[AB::Expr; 64]; 5]; 5] = array::from_fn(|y| {
            array::from_fn(|x| array::from_fn(|z| d[x][z].clone() + local.a[y][x][z]))
        });

        // Rho and pi: B[x][y][z] is a bit of A'.
        let b = |x: usize, y: usize, z: usize| {
            let (source_y, source_x, rot) = rho_pi_source(x, y);
            a_prime[source_y][source_x][(z + 64 - rot) % 64].clone()
        };

        // A round row maps its state to the next row's state:
        //
        //     A''[y][x][z]  = B[x][y][z] + (1 + B[x + 1][y][z]) * B[x + 2][y][z]
        //     next.A[y][x]  = A''[y][x] + [x = y = 0] * RC[r]
        //
        // The map is gated by (1 + output_flag), which is 0 exactly on an output row.
        // The round constant bit is the sum of the flags of the rounds whose constant has it set,
        // which is also 0 on an output row.
        let not_output = AB::Expr::ONE + output_flag;
        for y in 0..5 {
            for x in 0..5 {
                builder.assert_zeros::<64, _>(array::from_fn(|z| {
                    let chi =
                        b(x, y, z) + (AB::Expr::ONE + b((x + 1) % 5, y, z)) * b((x + 2) % 5, y, z);
                    let round_map = not_output.clone() * (chi + next.a[y][x][z]);
                    if x == 0 && y == 0 {
                        let rc_bit: AB::Expr = RC_BITS
                            .iter()
                            .zip(flags)
                            .filter(|(rc_bits, _)| rc_bits[z] != 0)
                            .map(|(_, &flag)| flag.into())
                            .sum();
                        round_map + rc_bit
                    } else {
                        round_map
                    }
                }));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use core::borrow::BorrowMut;
    use core::ops::Range;

    use p3_air::{
        AirLayout, DebugConstraintBuilder, check_all_constraints, check_constraints,
        get_max_constraint_degree, get_symbolic_constraints,
    };
    use p3_binary_field::BinaryField128;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrixView;
    use p3_matrix::stack::ViewPair;

    use super::*;
    use crate::binary::generation::{keccak_round, write_state};

    type F = BinaryField128;

    /// Constraint indices of the round-flag constraints.
    const FLAG_CONSTRAINTS: Range<usize> = 0..NUM_FLAG_CONSTRAINTS;

    /// Constraint indices of the state booleanity constraints.
    const BOOL_CONSTRAINTS: Range<usize> =
        NUM_FLAG_CONSTRAINTS..NUM_FLAG_CONSTRAINTS + NUM_STATE_BITS;

    fn row_values(trace: &RowMajorMatrix<F>, r: usize) -> &[F] {
        &trace.values[r * trace.width..(r + 1) * trace.width]
    }

    fn row(trace: &RowMajorMatrix<F>, r: usize) -> &KeccakBinaryCols<F> {
        row_values(trace, r).borrow()
    }

    fn row_mut(trace: &mut RowMajorMatrix<F>, r: usize) -> &mut KeccakBinaryCols<F> {
        trace.row_mut(r).borrow_mut()
    }

    /// Evaluate the AIR on every row with the multi-STARK successor convention.
    ///
    /// The successor of row `r` is row `r + 1`, and the successor of the last row is itself.
    ///
    /// # Returns
    ///
    /// Every violated constraint as `(row, constraint index)`.
    fn repeat_last_failures(trace: &RowMajorMatrix<F>) -> Vec<(usize, usize)> {
        let height = trace.height();
        (0..height)
            .flat_map(|r| {
                let successor = (r + 1).min(height - 1);
                let main = ViewPair::new(
                    RowMajorMatrixView::new_row(row_values(trace, r)),
                    RowMajorMatrixView::new_row(row_values(trace, successor)),
                );
                let preprocessed = ViewPair::new(
                    RowMajorMatrixView::new(&[], 0),
                    RowMajorMatrixView::new(&[], 0),
                );
                let mut builder = DebugConstraintBuilder::new(
                    r,
                    main,
                    preprocessed,
                    &[],
                    F::from_bool(r == 0),
                    F::from_bool(r == height - 1),
                    F::from_bool(r != height - 1),
                    &[],
                );
                KeccakBinaryAir {}.eval(&mut builder);
                builder
                    .into_failures()
                    .into_iter()
                    .map(|f| (f.row, f.constraint))
            })
            .collect()
    }

    /// Trace of one permutation input run through an arbitrary round schedule.
    ///
    /// Row `i` is the input row of round `schedule[i]`.
    /// If the height allows, the row after the schedule is the output row,
    /// and every later row is an all-zero output row.
    fn trace_for_schedule(
        input: [u64; 25],
        schedule: &[usize],
        height: usize,
    ) -> RowMajorMatrix<F> {
        let mut trace = RowMajorMatrix::new(
            F::zero_vec(height * NUM_KECCAK_BINARY_COLS),
            NUM_KECCAK_BINARY_COLS,
        );

        let mut state = input;
        for (i, &round) in schedule.iter().enumerate() {
            let cols = row_mut(&mut trace, i);
            cols.round_flags[round] = F::ONE;
            write_state(cols, &state);
            keccak_round(&mut state, round);
        }

        for i in schedule.len()..height {
            let cols = row_mut(&mut trace, i);
            cols.round_flags[NUM_ROUNDS] = F::ONE;
            if i == schedule.len() {
                write_state(cols, &state);
            }
        }

        trace
    }

    /// One round of the constrained map, evaluated on field elements rather than bits.
    fn field_round(a: &[[[F; 64]; 5]; 5], round: usize) -> [[[F; 64]; 5]; 5] {
        let c: [[F; 64]; 5] =
            array::from_fn(|x| array::from_fn(|z| (0..5).map(|y| a[y][x][z]).sum()));
        let a_prime: [[[F; 64]; 5]; 5] = array::from_fn(|y| {
            array::from_fn(|x| {
                array::from_fn(|z| a[y][x][z] + c[(x + 4) % 5][z] + c[(x + 1) % 5][(z + 63) % 64])
            })
        });
        let b = |x: usize, y: usize, z: usize| {
            let (source_y, source_x, rot) = rho_pi_source(x, y);
            a_prime[source_y][source_x][(z + 64 - rot) % 64]
        };
        array::from_fn(|y| {
            array::from_fn(|x| {
                array::from_fn(|z| {
                    let chi = b(x, y, z) + (F::ONE + b((x + 1) % 5, y, z)) * b((x + 2) % 5, y, z);
                    if x == 0 && y == 0 {
                        chi + F::from_u8(RC_BITS[round][z])
                    } else {
                        chi
                    }
                })
            })
        })
    }

    #[test]
    fn honest_traces_satisfy_constraints() {
        // Heights: 25 -> 32, 50 -> 64, 75 -> 128.
        // The debug checker wraps the last row to row 0.
        for (num_hashes, height) in [(1, 32), (2, 64), (3, 128)] {
            let trace = KeccakBinaryAir {}.generate_random_trace_rows::<F>(num_hashes, 0);
            assert_eq!(trace.height(), height);
            check_constraints(&KeccakBinaryAir {}, &trace, &[]);
        }
    }

    #[test]
    fn honest_traces_satisfy_repeat_last_successor() {
        // The last row is always an output row.
        // Read against itself, its round map is gated off and its flags step to themselves:
        //
        //     round r < 23 : next.f[r + 1] + f[r]                     = 0 + 0
        //     wrap         : next.f[0] + next.f[24] + f[23] + f[24]  = 0 + 1 + 0 + 1
        for num_hashes in 1..=3 {
            let trace = KeccakBinaryAir {}.generate_random_trace_rows::<F>(num_hashes, 0);
            let last = row(&trace, trace.height() - 1);
            assert_eq!(last.round_flags[NUM_ROUNDS], F::ONE);
            assert_eq!(repeat_last_failures(&trace), Vec::new());
        }
    }

    #[test]
    fn repeat_last_rejects_a_trace_ending_on_a_round_row() {
        // 24 rows: rounds 0..23 and no output row.
        // The last row is round 23, which cannot be its own successor.
        let schedule: Vec<usize> = (0..NUM_ROUNDS).collect();
        let trace = trace_for_schedule([0x0123_4567_89ab_cdef; 25], &schedule, NUM_ROUNDS);

        let failures = repeat_last_failures(&trace);
        assert!(!failures.is_empty());
        assert!(failures.iter().all(|&(r, _)| r == NUM_ROUNDS_MIN_1));
        assert!(failures.iter().any(|(_, c)| FLAG_CONSTRAINTS.contains(c)));
    }

    #[test]
    fn schedule_helper_matches_generator() {
        let input: [u64; 25] =
            array::from_fn(|i| (i as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let schedule: Vec<usize> = (0..NUM_ROUNDS).collect();
        let trace = trace_for_schedule(input, &schedule, 32);
        assert_eq!(
            trace.values,
            generate_binary_trace_rows::<F>(vec![input], 0).values
        );
    }

    #[test]
    fn flipped_state_bit_mid_permutation_is_rejected() {
        // Flip one bit of the round-12 input.
        // It is still a bit, but it no longer equals the round-11 output.
        let mut trace = KeccakBinaryAir {}.generate_random_trace_rows::<F>(1, 0);
        row_mut(&mut trace, 12).a[2][3][17] += F::ONE;

        let report = check_all_constraints(&KeccakBinaryAir {}, &trace, &[], None);
        assert!(!report.is_ok());
        assert!(report.failures.iter().any(|f| f.row == 11));
        assert!(
            !report
                .failures
                .iter()
                .any(|f| BOOL_CONSTRAINTS.contains(&f.constraint))
        );
        assert!(!repeat_last_failures(&trace).is_empty());
    }

    #[test]
    fn flipped_output_bit_is_rejected() {
        // Flip one bit of the output row: it no longer equals the round-23 output.
        let mut trace = KeccakBinaryAir {}.generate_random_trace_rows::<F>(1, 0);
        row_mut(&mut trace, NUM_ROUNDS).a[0][0][0] += F::ONE;

        let report = check_all_constraints(&KeccakBinaryAir {}, &trace, &[], None);
        assert!(!report.is_ok());
        assert!(report.failures.iter().all(|f| f.row == NUM_ROUNDS_MIN_1));
        assert!(!repeat_last_failures(&trace).is_empty());
    }

    #[test]
    fn non_boolean_state_is_rejected_by_booleanity() {
        let mut trace = KeccakBinaryAir {}.generate_random_trace_rows::<F>(1, 0);

        // The field-level round map reproduces the honest trace.
        for r in 0..NUM_ROUNDS {
            assert_eq!(field_round(&row(&trace, r).a, r), row(&trace, r + 1).a);
        }

        // Put a non-bit into the round-0 input and propagate it through every round.
        // Every round row then satisfies the round map exactly.
        let non_bit = F::GENERATOR;
        assert!(non_bit != F::ZERO && non_bit != F::ONE);
        row_mut(&mut trace, 0).a[1][2][3] = non_bit;
        for r in 0..NUM_ROUNDS {
            let next = field_round(&row(&trace, r).a, r);
            row_mut(&mut trace, r + 1).a = next;
        }

        // Only booleanity catches it.
        let report = check_all_constraints(&KeccakBinaryAir {}, &trace, &[], None);
        assert!(report.failures.iter().any(|f| f.row == 0));
        assert!(
            report
                .failures
                .iter()
                .all(|f| BOOL_CONSTRAINTS.contains(&f.constraint))
        );

        let failures = repeat_last_failures(&trace);
        assert!(!failures.is_empty());
        assert!(failures.iter().all(|(_, c)| BOOL_CONSTRAINTS.contains(c)));
    }

    #[test]
    fn two_round_flags_in_one_row_are_rejected() {
        // The output row also claims round 0.
        let mut trace = KeccakBinaryAir {}.generate_random_trace_rows::<F>(1, 0);
        row_mut(&mut trace, NUM_ROUNDS).round_flags[0] = F::ONE;

        let report = check_all_constraints(&KeccakBinaryAir {}, &trace, &[], None);
        assert!(!report.is_ok());
        assert!(
            report
                .failures
                .iter()
                .any(|f| FLAG_CONSTRAINTS.contains(&f.constraint))
        );
        assert!(!repeat_last_failures(&trace).is_empty());
    }

    #[test]
    fn non_bit_round_flags_are_rejected_only_by_exclusivity() {
        // Scale the second permutation's round flags by a non-bit u, with output flag 1 + u.
        // Every round row's map is scaled by u, so the honest states still satisfy it,
        // and the flag steps into and out of the permutation still balance:
        //
        //     output row -> round 0 : u + (1 + u) + 0 + 1 = 0
        //     round 23 -> output    : 0 + 1 + u + (1 + u) = 0
        //
        // Only the round-0 row's product u * (1 + u) is nonzero.
        let u = F::GENERATOR;
        assert!(u != F::ZERO && u != F::ONE);
        let mut trace = KeccakBinaryAir {}.generate_random_trace_rows::<F>(2, 0);
        for round in 0..NUM_ROUNDS {
            let flags = &mut row_mut(&mut trace, KECCAK_BINARY_ROWS_PER_PERM + round).round_flags;
            flags[round] = u;
            flags[NUM_ROUNDS] = F::ONE + u;
        }

        let exclusivity = NUM_FLAG_CONSTRAINTS - 1;
        let expected = vec![(KECCAK_BINARY_ROWS_PER_PERM, exclusivity)];
        let report = check_all_constraints(&KeccakBinaryAir {}, &trace, &[], None);
        let failures: Vec<(usize, usize)> = report
            .failures
            .iter()
            .map(|f| (f.row, f.constraint))
            .collect();
        assert_eq!(failures, expected);
        assert_eq!(repeat_last_failures(&trace), expected);
    }

    #[test]
    fn vanishing_round_flags_are_rejected_only_by_the_wrap() {
        // Clear every flag after the output row.
        // A flagless row applies the round map without a round constant, which fixes the
        // all-zero padding state, and its flags step to zero.
        // Only the output row's step into the first flagless row fails:
        //
        //     next.f[0] + next.f[24] + f[23] + f[24] = 0 + 0 + 0 + 1
        let mut trace = KeccakBinaryAir {}.generate_random_trace_rows::<F>(1, 0);
        for r in KECCAK_BINARY_ROWS_PER_PERM..trace.height() {
            row_mut(&mut trace, r).round_flags[NUM_ROUNDS] = F::ZERO;
        }

        let wrap = NUM_FLAG_CONSTRAINTS - 2;
        assert_eq!(repeat_last_failures(&trace), vec![(NUM_ROUNDS, wrap)]);
    }

    #[test]
    fn skipped_round_is_rejected() {
        // An honest execution of rounds 0..4, 6..23: every state follows its row's round map,
        // and the output row holds the 23-round result.
        // Only the flag step from round 4 to round 6 is wrong.
        let schedule: Vec<usize> = (0..NUM_ROUNDS).filter(|&r| r != 5).collect();
        let trace = trace_for_schedule([0xfedc_ba98_7654_3210; 25], &schedule, 32);

        let report = check_all_constraints(&KeccakBinaryAir {}, &trace, &[], None);
        assert!(!report.is_ok());
        assert!(report.failures.iter().all(|f| f.row == 4));
        assert!(
            report
                .failures
                .iter()
                .all(|f| FLAG_CONSTRAINTS.contains(&f.constraint))
        );
        let report_failures: Vec<(usize, usize)> = report
            .failures
            .iter()
            .map(|f| (f.row, f.constraint))
            .collect();
        assert_eq!(repeat_last_failures(&trace), report_failures);
    }

    #[test]
    fn symbolic_constraints_match_hints() {
        let air = KeccakBinaryAir {};
        let layout = AirLayout::from_air::<F>(&air);

        let constraints = get_symbolic_constraints::<F, _>(&air, layout);
        assert_eq!(constraints.len(), 3250);
        assert_eq!(Some(constraints.len()), BaseAir::<F>::num_constraints(&air));

        let degree = get_max_constraint_degree::<F, _>(&air, layout, 32);
        assert_eq!(degree, 3);
        assert_eq!(Some(degree), BaseAir::<F>::max_constraint_degree(&air));
    }
}
