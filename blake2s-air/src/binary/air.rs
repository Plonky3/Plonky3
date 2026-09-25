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

/// Number of input cells: 32 per input word, plus one for the finalization flag.
pub(super) const NUM_INPUT_BITS: usize = NUM_INPUT_WORDS * 32 + 1;

/// Constraints per G step.
///
/// - Each 3-operand addition costs 31 carry constraints plus 32 sum constraints.
/// - Each 2-operand addition costs 32 sum constraints.
/// - A G step has two of each.
const CONSTRAINTS_PER_G: usize = 2 * (31 + 32) + 2 * 32;

/// Number of constraints of all G steps, which excludes input booleanity.
const NUM_G_CONSTRAINTS: usize = NUM_ROUNDS * G_PER_ROUND * CONSTRAINTS_PER_G;

/// An AIR for the BLAKE2s compression function over a field of characteristic 2.
///
/// - Each row proves one compression.
/// - No constraint reads the next row.
/// - Every constraint has degree at most 2.
///
/// The compression output is neither stored nor constrained.
///
/// It is linear in the chaining value and the last round's witness words.
///
/// A row therefore recovers it without any extra column.
///
/// Soundness relies on every input cell being a bit.
///
/// The addition constraints then force every other cell to a bit.
///
/// This holds because the majority of three bits is a bit.
///
/// The constraints describe BLAKE2s only over a field of characteristic 2.
#[derive(Debug)]
pub struct Blake2sBinaryAir {
    /// Whether the AIR constrains every input cell to be a bit.
    constrain_booleanity: bool,
}

impl Default for Blake2sBinaryAir {
    /// An AIR that constrains every input cell to be a bit.
    fn default() -> Self {
        Self {
            constrain_booleanity: true,
        }
    }
}

impl Blake2sBinaryAir {
    /// An AIR that leaves input booleanity to the trace commitment.
    ///
    /// It is sound only under a commitment with one bit per cell.
    ///
    /// There, a cell outside `{0, 1}` cannot be written.
    ///
    /// Under a commitment to field elements, nothing keeps an input cell in `{0, 1}`.
    pub const fn assuming_boolean_trace() -> Self {
        Self {
            constrain_booleanity: false,
        }
    }

    /// Generate a trace over `num_hashes` random compression inputs, drawn from a fixed seed.
    ///
    /// Each row draws a random chaining value and block.
    ///
    /// Row `i` counts `64 * (i + 1)` bytes and is a final block.
    ///
    /// This is for benches and examples only.
    pub fn generate_random_trace_rows<F: Field>(
        &self,
        num_hashes: usize,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F> {
        generate_binary_trace_rows(random_inputs(num_hashes), extra_capacity_bits)
    }

    /// Generate the same random trace, packed into one `u64` per 64 trace rows.
    ///
    /// The field parameter only fixes the characteristic.
    ///
    /// It does not change the resulting bits.
    pub fn generate_random_trace_packed<F: Field>(&self, num_hashes: usize) -> RowMajorMatrix<u64> {
        generate_binary_trace_packed::<F>(&random_inputs(num_hashes))
    }
}

/// The fixed-seed random inputs shared by both random trace generators.
///
/// Sharing them makes the dense and the packed traces agree row for row.
fn random_inputs(num_hashes: usize) -> Vec<Blake2sCompressionInput> {
    // A fixed seed keeps benches reproducible.
    let mut rng = SmallRng::seed_from_u64(1);
    (0..num_hashes)
        .map(|i| Blake2sCompressionInput {
            chaining_value: rng.random(),
            block: rng.random(),
            // Row i ends a message of i + 1 full blocks.
            counter: 64 * (i as u64 + 1),
            last_block: true,
        })
        .collect()
}

impl<F> BaseAir<F> for Blake2sBinaryAir {
    fn width(&self) -> usize {
        NUM_BLAKE2S_BINARY_COLS
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        // Each row is a standalone compression.
        vec![]
    }

    fn assumes_boolean_trace(&self) -> bool {
        !self.constrain_booleanity
    }

    fn num_constraints(&self) -> Option<usize> {
        // One booleanity constraint per input cell, when enabled.
        let booleanity = if self.constrain_booleanity {
            NUM_INPUT_BITS
        } else {
            0
        };
        Some(booleanity + NUM_G_CONSTRAINTS)
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        // The majority function multiplies two linear terms.
        Some(2)
    }
}

/// The working state as bits, viewed as four rows of four words.
///
/// - The `b` words are always trace columns.
/// - The `a` and `c` words are XORs of trace columns and constants.
/// - The `d` words start as an initialization word XORed with an input, then become trace columns.
struct State<AB: AirBuilder> {
    /// The words `v[0..4]`.
    a: [[AB::Expr; 32]; 4],
    /// The words `v[4..8]`.
    b: [[AB::Var; 32]; 4],
    /// The words `v[8..12]`.
    c: [[AB::Expr; 32]; 4],
    /// The words `v[12..16]`.
    d: [[AB::Expr; 32]; 4],
}

impl<AB: AirBuilder> Air<AB> for Blake2sBinaryAir {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &Blake2sBinaryCols<AB::Var> = main.current_slice().borrow();

        // Input booleanity, in column order:
        //
        //     constraints 0..256      chaining value
        //     constraints 256..768    message block
        //     constraints 768..832    counter, low word then high word
        //     constraint  832         last-block flag
        if self.constrain_booleanity {
            for word in local
                .chaining_value
                .iter()
                .chain(&local.block)
                .chain([&local.counter_low, &local.counter_high])
            {
                builder.assert_bools(*word);
            }
            builder.assert_bool(local.last_block);
        }

        // The three words XORed into v[12..15], from RFC 7693 section 3.2.
        //
        // The flag is one cell, repeated over all 32 bits.
        // XORing a set flag into every bit inverts the word, as the RFC requires.
        //
        // v[15] takes IV[7] unchanged: section 3.2 inverts v[14] alone.
        let parameters = [
            local.counter_low,
            local.counter_high,
            [local.last_block; 32],
        ];

        // Initialize the working state.
        //
        //     v[0..8]   = h[0..8]
        //     v[8..12]  = IV[0..4]
        //     v[12..15] = IV[4..7] ^ (t_low, t_high, f)
        //     v[15]     = IV[7]
        //
        // Every word is linear in the inputs, so this costs no column.
        let mut state = State::<AB> {
            a: array::from_fn(|i| local.chaining_value[i].map(Into::into)),
            b: array::from_fn(|i| local.chaining_value[4 + i]),
            c: array::from_fn(|i| u32_to_bits_le(IV[i])),
            d: array::from_fn(|i| {
                let iv: [AB::Expr; 32] = u32_to_bits_le(IV[4 + i]);
                parameters.get(i).map_or_else(
                    || iv.clone(),
                    |word| array::from_fn(|bit| iv[bit].clone() + word[bit]),
                )
            }),
        };

        // Ten rounds of eight G steps.
        //
        // Round r feeds step g the message words at positions 2g and 2g + 1 of schedule row r.
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

/// Constrain one G step and replace the four words it mixes with their new values.
///
/// The four slots are the positions of the `a`, `b`, `c`, `d` words within their state rows.
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

    // First half.
    //
    //     a_1 = a + b + m_x,  with  a_1 = d ^ (d_1 <<< 16)
    let a1 = xor_rotl::<AB>(d, &cols.d1, 16);
    add3(builder, &state.a[ia], b, mx, &cols.add1_carries, &a1);

    //     c_1 = c + d_1,  with  c_1 = b ^ (b_1 <<< 12)
    let c1 = xor_rotl_var::<AB>(b, &cols.b1, 12);
    add2(builder, &state.c[ic], &cols.d1, &c1);

    // Second half.
    //
    //     a_2 = a_1 + b_1 + m_y,  with  a_2 = d_1 ^ (d_2 <<< 8)
    let a2 = xor_rotl_var::<AB>(&cols.d1, &cols.d2, 8);
    add3(builder, &a1, &cols.b1, my, &cols.add2_carries, &a2);

    //     c_2 = c_1 + d_2,  with  c_2 = b_1 ^ (b_2 <<< 7)
    let c2 = xor_rotl_var::<AB>(&cols.b1, &cols.b2, 7);
    add2(builder, &c1, &cols.d2, &c2);

    // The step's outputs replace its inputs in the state.
    state.a[ia] = a2;
    state.b[ib] = cols.b2;
    state.c[ic] = c2;
    state.d[id] = cols.d2.map(Into::into);
}

/// The bits of `x ^ (y <<< shift)`, where `x` is already an expression.
fn xor_rotl<AB: AirBuilder>(x: &[AB::Expr; 32], y: &[AB::Var; 32], shift: usize) -> [AB::Expr; 32] {
    // Rotating left by s moves bit i - s to bit i.
    array::from_fn(|i| x[i].clone() + y[(i + 32 - shift) % 32])
}

/// The bits of `x ^ (y <<< shift)`, where `x` is held in trace columns.
fn xor_rotl_var<AB: AirBuilder>(
    x: &[AB::Var; 32],
    y: &[AB::Var; 32],
    shift: usize,
) -> [AB::Expr; 32] {
    // Rotating left by s moves bit i - s to bit i.
    array::from_fn(|i| x[i] + y[(i + 32 - shift) % 32])
}

/// The majority of three bits, `(x + z) * (y + z) + z` in characteristic 2.
///
/// It is the carry out of a full adder with inputs `x`, `y` and carry `z`.
fn maj<AB: AirBuilder>(x: AB::Expr, y: AB::Expr, z: AB::Expr) -> AB::Expr {
    (x + z.clone()) * (y + z.clone()) + z
}

/// Constrain `sum = x + y + z mod 2^32`, with the carries of `x + y` stored in columns.
///
/// Entry `i` of the carries is the carry into bit `i + 1` of `x + y`.
fn add3<AB: AirBuilder>(
    builder: &mut AB,
    x: &[AB::Expr; 32],
    y: &[AB::Var; 32],
    z: &[AB::Var; 32],
    carries: &[AB::Var; 31],
    sum: &[AB::Expr; 32],
) {
    // The carry into bit 1 is x_0 AND y_0, since nothing carries into bit 0.
    builder.assert_zero(x[0].clone() * y[0] + carries[0]);

    // The carry into bit i + 1 is the majority of x_i, y_i and the carry into bit i.
    for (x_i, &y_i, &carry_i, &carry_next) in izip!(&x[1..], &y[1..], carries, &carries[1..]) {
        builder.assert_zero(maj::<AB>(x_i.clone(), y_i.into(), carry_i.into()) + carry_next);
    }

    // With the carries fixed, t = x + y mod 2^32 is linear:
    //
    //     t_0 = x_0 + y_0
    //     t_i = x_i + y_i + carry_i
    let t: [AB::Expr; 32] = array::from_fn(|i| {
        if i == 0 {
            x[0].clone() + y[0]
        } else {
            x[i].clone() + y[i] + carries[i - 1]
        }
    });

    // The last operand is added with carries derived from the sum.
    add2(builder, &t, z, sum);
}

/// Constrain `sum = x + y mod 2^32`, with no carry column.
///
/// Each carry is recovered from the sum as `carry_i = sum_i + x_i + y_i`.
fn add2<AB: AirBuilder>(
    builder: &mut AB,
    x: &[AB::Expr; 32],
    y: &[AB::Var; 32],
    sum: &[AB::Expr; 32],
) {
    // Recover each carry from the sum bit it produced.
    let carries: [AB::Expr; 32] = array::from_fn(|i| sum[i].clone() + x[i].clone() + y[i]);

    // Nothing carries into bit 0.
    builder.assert_zero(carries[0].clone());

    // The carry into bit i + 1 is the majority of x_i, y_i and the carry into bit i.
    //
    // Together these pin every sum bit, the top one included.
    for (x_i, &y_i, carry_i, carry_next) in izip!(x, y, &carries, &carries[1..]) {
        builder
            .assert_zero(maj::<AB>(x_i.clone(), y_i.into(), carry_i.clone()) + carry_next.clone());
    }
}
