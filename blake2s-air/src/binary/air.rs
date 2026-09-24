use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::borrow::Borrow;

use itertools::izip;
use p3_air::utils::u32_to_bits_le;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::columns::{Blake2sBinaryCols, Blake2sBinaryGCols, NUM_BLAKE2S_BINARY_COLS};
use super::generation::{
    Blake2sCompressionInput, NUM_INPUT_WORDS, generate_binary_trace_packed,
    generate_binary_trace_rows,
};
use super::{G_PER_ROUND, G_SCHEDULE, NUM_ROUNDS};
use crate::constants::{IV, SIGMA};

/// Number of input bits: chaining value, message, counter halves and the two flags.
pub(super) const NUM_INPUT_BITS: usize = NUM_INPUT_WORDS * 32;

/// Constraints per G step: two 3-operand and two 2-operand additions.
const CONSTRAINTS_PER_G: usize = 2 * (31 + 32) + 2 * 32;

/// Number of constraints excluding input booleanity: every G step.
const NUM_G_CONSTRAINTS: usize = NUM_ROUNDS * G_PER_ROUND * CONSTRAINTS_PER_G;

/// An AIR for the BLAKE2s compression function over a field of characteristic 2.
///
/// Each row proves one compression. The AIR reads no next row and every constraint has
/// degree at most 2.
///
/// The compression output is neither stored nor constrained. It is linear in the chaining
/// value and the last round's `b1`, `d1`, `b2`, `d2` words, and
/// [`Blake2sBinaryCols::compression_output`] recovers it from a row.
///
/// [`Self::default`] constrains the input bits to be boolean, and
/// [`Self::assuming_boolean_trace`] leaves that to the trace commitment. Every other column is
/// then forced to a bit by the addition constraints, since the majority of three bits is a bit.
///
/// The constraints describe BLAKE2s only over a field of characteristic 2.
#[derive(Debug)]
pub struct Blake2sBinaryAir {
    /// Whether the AIR constrains every input cell to be a bit.
    constrain_booleanity: bool,
}

impl Default for Blake2sBinaryAir {
    fn default() -> Self {
        Self {
            constrain_booleanity: true,
        }
    }
}

impl Blake2sBinaryAir {
    /// An AIR that skips the input booleanity constraints, which the commitment must then supply.
    ///
    /// The AIR reports [`BaseAir::assumes_boolean_trace`]. It is sound only under a commitment
    /// whose alphabet is one bit per cell, such as a commitment to the trace's bits, where a cell
    /// outside `{0, 1}` is not representable. Under a commitment to field elements, nothing else
    /// keeps an input cell in `{0, 1}`.
    pub const fn assuming_boolean_trace() -> Self {
        Self {
            constrain_booleanity: false,
        }
    }

    /// Generate a trace over `num_hashes` fixed-seed random compression inputs.
    ///
    /// Each row draws a random chaining value and block, counts 64 bytes per block, and is a
    /// final block that is not a last node.
    ///
    /// This is for benches/examples only. Use the free [`generate_binary_trace_rows`]
    /// function directly to prove specific inputs.
    pub fn generate_random_trace_rows<F: Field>(
        &self,
        num_hashes: usize,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F> {
        generate_binary_trace_rows(random_inputs(num_hashes), extra_capacity_bits)
    }

    /// Generate the same fixed-seed random compression inputs as
    /// [`Self::generate_random_trace_rows`], packed into one `u64` per 64 trace rows.
    ///
    /// The generic field names only the characteristic the cells are read in and does not affect
    /// the resulting bits.
    pub fn generate_random_trace_packed<F: Field>(&self, num_hashes: usize) -> RowMajorMatrix<u64> {
        generate_binary_trace_packed::<F>(random_inputs(num_hashes))
    }
}

/// The fixed-seed random inputs both trace generators share, so the two agree row for row.
fn random_inputs(num_hashes: usize) -> Vec<Blake2sCompressionInput> {
    let mut rng = SmallRng::seed_from_u64(1);
    (0..num_hashes)
        .map(|i| Blake2sCompressionInput {
            chaining_value: rng.random(),
            block: rng.random(),
            counter: 64 * (i as u64 + 1),
            last_block: true,
            last_node: false,
        })
        .collect()
}

impl<F> BaseAir<F> for Blake2sBinaryAir {
    fn width(&self) -> usize {
        NUM_BLAKE2S_BINARY_COLS
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
        Some(booleanity + NUM_G_CONSTRAINTS)
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(2)
    }
}

/// The compression state as bits, split into its four rows of four words.
///
/// The `b` words are always trace columns. The `a`, `c` and `d` words are either initial
/// values or expressions over trace columns: `d` starts as an initialization word XORed with
/// an input column, and later becomes a stored word.
struct State<AB: AirBuilder> {
    a: [[AB::Expr; 32]; 4],
    b: [[AB::Var; 32]; 4],
    c: [[AB::Expr; 32]; 4],
    d: [[AB::Expr; 32]; 4],
}

impl<AB: AirBuilder> Air<AB> for Blake2sBinaryAir {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &Blake2sBinaryCols<AB::Var> = main.current_slice().borrow();

        let parameters = [
            local.counter_low,
            local.counter_high,
            local.last_block,
            local.last_node,
        ];

        if self.constrain_booleanity {
            for word in local
                .chaining_value
                .iter()
                .chain(&local.block)
                .chain(&parameters)
            {
                builder.assert_bools(*word);
            }
        }

        let mut state = State::<AB> {
            a: array::from_fn(|i| local.chaining_value[i].map(Into::into)),
            b: array::from_fn(|i| local.chaining_value[4 + i]),
            c: array::from_fn(|i| u32_to_bits_le(IV[i])),
            // The counter and the flags enter the state XORed into the last four
            // initialization words, which is linear in the input columns.
            d: array::from_fn(|i| {
                let iv: [AB::Expr; 32] = u32_to_bits_le(IV[4 + i]);
                array::from_fn(|bit| iv[bit].clone() + parameters[i][bit])
            }),
        };

        for (round, schedule) in local.rounds.iter().zip(SIGMA) {
            for (g, (cols, slots)) in round.iter().zip(G_SCHEDULE).enumerate() {
                let (mx, my) = (
                    &local.block[schedule[2 * g]],
                    &local.block[schedule[2 * g + 1]],
                );
                eval_g(builder, &mut state, slots, mx, my, cols);
            }
        }
    }
}

/// Constrain one G step and advance `state` to its output.
///
/// `slots` holds the indices of the `a`, `b`, `c`, `d` words within their state rows.
fn eval_g<AB: AirBuilder>(
    builder: &mut AB,
    state: &mut State<AB>,
    [ia, ib, ic, id]: [usize; 4],
    mx: &[AB::Var; 32],
    my: &[AB::Var; 32],
    cols: &Blake2sBinaryGCols<AB::Var>,
) {
    let b = &state.b[ib];
    let d = &state.d[id];

    // a1 = a + b + mx, where a1 = d ^ (d1 <<< 16).
    let a1 = xor_rotl::<AB>(d, &cols.d1, 16);
    add3(builder, &state.a[ia], b, mx, &cols.add1_carries, &a1);

    // c1 = c + d1, where c1 = b ^ (b1 <<< 12).
    let c1 = xor_rotl_var::<AB>(b, &cols.b1, 12);
    add2(builder, &state.c[ic], &cols.d1, &c1);

    // a2 = a1 + b1 + my, where a2 = d1 ^ (d2 <<< 8).
    let a2 = xor_rotl_var::<AB>(&cols.d1, &cols.d2, 8);
    add3(builder, &a1, &cols.b1, my, &cols.add2_carries, &a2);

    // c2 = c1 + d2, where c2 = b1 ^ (b2 <<< 7).
    let c2 = xor_rotl_var::<AB>(&cols.b1, &cols.b2, 7);
    add2(builder, &c1, &cols.d2, &c2);

    state.a[ia] = a2;
    state.b[ib] = cols.b2;
    state.c[ic] = c2;
    state.d[id] = cols.d2.map(Into::into);
}

/// The bits of `x ^ (y <<< shift)`, for an `x` that is already an expression.
fn xor_rotl<AB: AirBuilder>(x: &[AB::Expr; 32], y: &[AB::Var; 32], shift: usize) -> [AB::Expr; 32] {
    array::from_fn(|i| x[i].clone() + y[(i + 32 - shift) % 32])
}

/// The bits of `x ^ (y <<< shift)`, for an `x` held in trace columns.
fn xor_rotl_var<AB: AirBuilder>(
    x: &[AB::Var; 32],
    y: &[AB::Var; 32],
    shift: usize,
) -> [AB::Expr; 32] {
    array::from_fn(|i| x[i] + y[(i + 32 - shift) % 32])
}

/// The majority of three bits, `(x + z) * (y + z) + z` in characteristic 2.
fn maj<AB: AirBuilder>(x: AB::Expr, y: AB::Expr, z: AB::Expr) -> AB::Expr {
    (x + z.clone()) * (y + z.clone()) + z
}

/// Constrain `sum = x + y + z mod 2^32`.
///
/// `carries[i]` is the carry into bit `i + 1` of `x + y`. The carry constraints fix
/// `t = x + y mod 2^32`, and [`add2`] then checks `t + z = sum`.
fn add3<AB: AirBuilder>(
    builder: &mut AB,
    x: &[AB::Expr; 32],
    y: &[AB::Var; 32],
    z: &[AB::Var; 32],
    carries: &[AB::Var; 31],
    sum: &[AB::Expr; 32],
) {
    builder.assert_zero(x[0].clone() * y[0] + carries[0]);
    for (x_i, &y_i, &carry_i, &carry_next) in izip!(&x[1..], &y[1..], carries, &carries[1..]) {
        builder.assert_zero(maj::<AB>(x_i.clone(), y_i.into(), carry_i.into()) + carry_next);
    }

    let t: [AB::Expr; 32] = array::from_fn(|i| {
        if i == 0 {
            x[0].clone() + y[0]
        } else {
            x[i].clone() + y[i] + carries[i - 1]
        }
    });
    add2(builder, &t, z, sum);
}

/// Constrain `sum = x + y mod 2^32` for a sum given as bits.
///
/// The carry into bit `i` is `sum_i + x_i + y_i`. It must be zero at bit 0, and above that
/// equal to the majority of the previous bit's operands and carry.
fn add2<AB: AirBuilder>(
    builder: &mut AB,
    x: &[AB::Expr; 32],
    y: &[AB::Var; 32],
    sum: &[AB::Expr; 32],
) {
    let carries: [AB::Expr; 32] = array::from_fn(|i| sum[i].clone() + x[i].clone() + y[i]);

    builder.assert_zero(carries[0].clone());
    for (x_i, &y_i, carry_i, carry_next) in izip!(x, y, &carries, &carries[1..]) {
        builder
            .assert_zero(maj::<AB>(x_i.clone(), y_i.into(), carry_i.clone()) + carry_next.clone());
    }
}
