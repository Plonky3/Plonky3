use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::borrow::Borrow;

use p3_air::utils::u32_to_bits_le;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::columns::{NUM_SHA256_BINARY_COLS, Sha256BinaryCols};
use super::generation::{generate_binary_trace_packed, generate_binary_trace_rows};
use super::{SCHEDULE_CARRIES, T1_CARRIES, rotr_index};
use crate::{
    BLOCK_WORDS, INPUT_WORDS, NUM_COMPRESSION_ROUNDS, SCHEDULE_EXTENSIONS, SHA256_K, STATE_WORDS,
};

/// Number of input bits: the chaining value and the message block.
pub(super) const NUM_INPUT_BITS: usize = (STATE_WORDS + BLOCK_WORDS) * 32;

/// Constraints for one addition that stores its carries: bit 0, then bits `1..31`.
const STORED_CARRY_ADD: usize = 31;

/// Constraints for one addition that reads its carries off the sum.
const SUM_CARRY_ADD: usize = 32;

/// Constraints per schedule word: two stored-carry additions and a final one.
pub(super) const CONSTRAINTS_PER_SCHEDULE_WORD: usize =
    SCHEDULE_CARRIES * STORED_CARRY_ADD + SUM_CARRY_ADD;

/// Constraints per round: `Ch`, `Maj`, `T1`, `new e = d + T1` and `new a = T1 + Σ0 + Maj`.
pub(super) const CONSTRAINTS_PER_ROUND: usize = 32
    + 32
    + (T1_CARRIES * STORED_CARRY_ADD + SUM_CARRY_ADD)
    + SUM_CARRY_ADD
    + (STORED_CARRY_ADD + SUM_CARRY_ADD);

/// Number of constraints after input booleanity: the schedule, every round and the output.
const NUM_HASH_CONSTRAINTS: usize = SCHEDULE_EXTENSIONS * CONSTRAINTS_PER_SCHEDULE_WORD
    + NUM_COMPRESSION_ROUNDS * CONSTRAINTS_PER_ROUND
    + STATE_WORDS * SUM_CARRY_ADD;

/// An AIR for the SHA-256 compression function over a field of characteristic 2.
///
/// Each row proves one compression. The AIR reads no next row and every constraint has
/// degree at most 2.
///
/// The input bits are constrained to be boolean, unless [`Self::constrain_booleanity`] is
/// cleared. Every other column is then forced to a bit by the constraints that define it, since
/// the majority of three bits is a bit.
///
/// The constraints describe SHA-256 only over a field of characteristic 2.
#[derive(Debug)]
pub struct Sha256BinaryAir {
    /// Whether the AIR constrains every input cell to be a bit.
    ///
    /// Clearing it makes the AIR report [`BaseAir::assumes_boolean_trace`]. That is sound only
    /// when the trace commitment refuses every cell outside `{0, 1}`, as a commitment to the
    /// trace's bits does. Under a commitment to field elements, nothing else keeps an input cell
    /// in `{0, 1}`.
    pub constrain_booleanity: bool,
}

impl Default for Sha256BinaryAir {
    fn default() -> Self {
        Self {
            constrain_booleanity: true,
        }
    }
}

impl Sha256BinaryAir {
    /// Generate a trace over `num_hashes` fixed-seed random compression inputs.
    ///
    /// This is for benches/examples only. Use the free [`generate_binary_trace_rows`]
    /// function directly to prove specific inputs.
    pub fn generate_random_trace_rows<F: Field>(
        &self,
        num_hashes: usize,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F> {
        let mut rng = SmallRng::seed_from_u64(1);
        let inputs = (0..num_hashes)
            .map(|_| rng.random::<[u32; INPUT_WORDS]>())
            .collect();
        generate_binary_trace_rows(inputs, extra_capacity_bits)
    }

    /// Generate a packed trace over `num_hashes` fixed-seed random compression inputs.
    ///
    /// See [`generate_binary_trace_packed`] for the packing.
    pub fn generate_random_trace_packed<F: Field>(&self, num_hashes: usize) -> RowMajorMatrix<u64> {
        let mut rng = SmallRng::seed_from_u64(1);
        let inputs = (0..num_hashes)
            .map(|_| rng.random::<[u32; INPUT_WORDS]>())
            .collect();
        generate_binary_trace_packed::<F>(inputs)
    }
}

impl<F> BaseAir<F> for Sha256BinaryAir {
    fn width(&self) -> usize {
        NUM_SHA256_BINARY_COLS
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn assumes_boolean_trace(&self) -> bool {
        !self.constrain_booleanity
    }

    fn num_constraints(&self) -> Option<usize> {
        let booleanity = if self.constrain_booleanity {
            NUM_INPUT_BITS
        } else {
            0
        };
        Some(booleanity + NUM_HASH_CONSTRAINTS)
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(2)
    }
}

impl<AB: AirBuilder> Air<AB> for Sha256BinaryAir {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &Sha256BinaryCols<AB::Var> = main.current_slice().borrow();

        if self.constrain_booleanity {
            for word in local.a_chain[..4]
                .iter()
                .chain(&local.e_chain[..4])
                .chain(&local.w[..BLOCK_WORDS])
            {
                builder.assert_bools(*word);
            }
        }

        let a_chain: [[AB::Expr; 32]; _] = local.a_chain.map(|word| word.map(Into::into));
        let e_chain: [[AB::Expr; 32]; _] = local.e_chain.map(|word| word.map(Into::into));
        let w: [[AB::Expr; 32]; _] = local.w.map(|word| word.map(Into::into));

        // W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16].
        for (i, cols) in local.schedule.iter().enumerate() {
            let t = BLOCK_WORDS + i;
            add_many(
                builder,
                &[
                    xor3_shift(&w[t - 2], [17, 19], 10),
                    w[t - 7].clone(),
                    xor3_shift(&w[t - 15], [7, 18], 3),
                    w[t - 16].clone(),
                ],
                &cols.carries,
                &w[t],
            );
        }

        for (t, cols) in local.rounds.iter().enumerate() {
            let (a, b, c, d) = (
                &a_chain[t + 3],
                &a_chain[t + 2],
                &a_chain[t + 1],
                &a_chain[t],
            );
            let (e, f, g, h) = (
                &e_chain[t + 3],
                &e_chain[t + 2],
                &e_chain[t + 1],
                &e_chain[t],
            );
            let ch: [AB::Expr; 32] = cols.ch.map(Into::into);
            let maj_word: [AB::Expr; 32] = cols.maj.map(Into::into);
            let t1: [AB::Expr; 32] = cols.t1.map(Into::into);

            // Ch(e, f, g) = e * (f + g) + g.
            for i in 0..32 {
                builder.assert_zero(
                    e[i].clone() * (f[i].clone() + g[i].clone()) + g[i].clone() + ch[i].clone(),
                );
            }
            // Maj(a, b, c) is the bitwise majority.
            for i in 0..32 {
                builder.assert_zero(
                    maj::<AB>(a[i].clone(), b[i].clone(), c[i].clone()) + maj_word[i].clone(),
                );
            }

            // T1 = h + Σ1(e) + Ch + K[t] + W[t].
            add_many(
                builder,
                &[
                    h.clone(),
                    xor3_rot(e, [6, 11, 25]),
                    ch,
                    u32_to_bits_le(SHA256_K[t]),
                    w[t].clone(),
                ],
                &cols.t1_carries,
                &t1,
            );

            // new e = d + T1.
            add_many(builder, &[d.clone(), t1.clone()], &[], &e_chain[t + 4]);

            // new a = T1 + Σ0(a) + Maj.
            add_many(
                builder,
                &[t1, xor3_rot(a, [2, 13, 22]), maj_word],
                &[cols.new_a_carries],
                &a_chain[t + 4],
            );
        }

        // H'[i] = H[i] + final working variable i. Both are chain entries: the chaining value
        // sits at the start of a chain in round-shift order, the final state at its end.
        for (i, out) in local.h_out.iter().enumerate() {
            let chain = if i < 4 { &a_chain } else { &e_chain };
            let (h_in, last) = (
                &chain[3 - i % 4],
                &chain[NUM_COMPRESSION_ROUNDS + 3 - i % 4],
            );
            add_many(
                builder,
                &[h_in.clone(), last.clone()],
                &[],
                &out.map(Into::into),
            );
        }
    }
}

/// The bits of `(x >>> r0) ^ (x >>> r1) ^ (x >>> r2)`, as in `Σ0` and `Σ1`.
fn xor3_rot<E: Clone + core::ops::Add<Output = E>>(
    x: &[E; 32],
    [r0, r1, r2]: [usize; 3],
) -> [E; 32] {
    array::from_fn(|i| {
        x[rotr_index(i, r0)].clone() + x[rotr_index(i, r1)].clone() + x[rotr_index(i, r2)].clone()
    })
}

/// The bits of `(x >>> r0) ^ (x >>> r1) ^ (x >> s)`, as in `σ0` and `σ1`.
fn xor3_shift<E: Clone + core::ops::Add<Output = E>>(
    x: &[E; 32],
    [r0, r1]: [usize; 2],
    s: usize,
) -> [E; 32] {
    array::from_fn(|i| {
        let rotated = x[rotr_index(i, r0)].clone() + x[rotr_index(i, r1)].clone();
        if i + s < 32 {
            rotated + x[i + s].clone()
        } else {
            rotated
        }
    })
}

/// The majority of three bits, `(x + z) * (y + z) + z` in characteristic 2.
fn maj<AB: AirBuilder>(x: AB::Expr, y: AB::Expr, z: AB::Expr) -> AB::Expr {
    (x + z.clone()) * (y + z.clone()) + z
}

/// Constrain `sum = words[0] + words[1] + ... mod 2^32`.
///
/// The words are added in order. `carries[j]` holds the carries into bits `1..32` of
/// addition `j`, which adds `words[j + 1]` to the partial sum of `words[0..=j]`. Each partial
/// sum is then linear in the words and those carries. The last addition stores no carries:
/// its carry into bit `i` is `sum_i + x_i + y_i`, which must be zero at bit 0 and, above that,
/// the majority of the previous bit's operands and carry.
///
/// Every operand and `sum` must be linear in the trace, so every constraint has degree 2.
fn add_many<AB: AirBuilder, const N: usize>(
    builder: &mut AB,
    words: &[[AB::Expr; 32]],
    carries: &[[AB::Var; 31]; N],
    sum: &[AB::Expr; 32],
) {
    assert_eq!(
        words.len(),
        N + 2,
        "one stored-carry addition per middle word"
    );

    let mut partial = words[0].clone();
    for (word, carries) in words[1..=N].iter().zip(carries) {
        builder.assert_zero(partial[0].clone() * word[0].clone() + carries[0]);
        for i in 1..31 {
            builder.assert_zero(
                maj::<AB>(partial[i].clone(), word[i].clone(), carries[i - 1].into()) + carries[i],
            );
        }
        partial = array::from_fn(|i| {
            if i == 0 {
                partial[0].clone() + word[0].clone()
            } else {
                partial[i].clone() + word[i].clone() + carries[i - 1]
            }
        });
    }

    let last = &words[N + 1];
    let carry: [AB::Expr; 32] =
        array::from_fn(|i| sum[i].clone() + partial[i].clone() + last[i].clone());
    builder.assert_zero(carry[0].clone());
    for i in 0..31 {
        builder.assert_zero(
            maj::<AB>(partial[i].clone(), last[i].clone(), carry[i].clone()) + carry[i + 1].clone(),
        );
    }
}
