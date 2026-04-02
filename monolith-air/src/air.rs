//! AIR constraint definitions for the Monolith permutation.
//!
//! # Overview
//!
//! This module defines `MonolithAir`, the AIR that constrains the Monolith permutation.
//!
//! The constraints ensure that each trace row represents a valid execution of the full permutation.
//!
//! # Constraint Structure
//!
//! The constraints mirror the round structure of the permutation:
//!
//! ```text
//!   inputs ──▶ initial Concrete ──▶ R rounds of (Bars → Bricks → Concrete → RC)
//!                               ──▶ final round (Bars → Bricks → Concrete)
//! ```
//!
//! Each round produces constraints that verify:
//!
//! 1. **Bit decomposition**: The committed bits are boolean and reconstruct
//!    the Bar input element.
//!
//! 2. **S-box correctness**: The committed `bars_output` values match the
//!    chi-like S-box applied to the bit-decomposed limbs.
//!
//! 3. **Bricks correctness**: The Feistel Type-3 layer `state[i] += state[i-1]^2`
//!    is applied correctly (degree-2 constraints).
//!
//! 4. **Concrete + RC correctness**: The committed `post` values equal the MDS
//!    matrix applied to the post-Bricks state, plus round constants.
//!
//! # Maximum Constraint Degree
//!
//! The maximum constraint degree is **4**, arising from the chi S-box formula:
//!
//! ```text
//!   out_j = in_{j+1} XOR ((NOT in_{j+2}) AND in_{j+3} AND in_{j+4})
//! ```
//!
//! where the XOR of a bit and a degree-3 AND product yields degree 4 over Fp.
//! All other constraints (boolean, reconstruction, Bricks, Concrete) are degree 2 or less.

use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;

use p3_air::utils::pack_bits_le;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_monolith::MonolithBars;
use p3_poseidon1::external::mds_multiply;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::columns::{MonolithCols, MonolithRoundCols, num_cols};
use crate::generation::generate_trace_rows;

/// Limb bit widths for Mersenne31 (p = 2^31 - 1).
///
/// A 31-bit field element is decomposed into 4 limbs.
/// - The first three (8-bit limbs) use the 3-input AND chi S-box,
/// - The last (one 7-bit limb) uses the 2-input AND chi S-box.
pub const MERSENNE31_LIMB_BITS: &[usize] = &[8, 8, 8, 7];

/// Limb bit widths for Goldilocks (p = 2^64 - 2^32 + 1) with 8-bit lookups.
///
/// A 64-bit field element is decomposed into 8 limbs of 8 bits each.
/// All limbs use the 3-input AND chi S-box.
pub const GOLDILOCKS_8_LIMB_BITS: &[usize] = &[8, 8, 8, 8, 8, 8, 8, 8];

/// The Monolith AIR.
///
/// Constrains one Monolith permutation per trace row.
/// The AIR evaluates polynomial constraints that verify every round of the permutation.
///
/// # Type Parameters
///
/// - `F`: The prime field.
/// - `WIDTH`: Permutation state width.
/// - `NUM_FULL_ROUNDS`: Number of rounds with round-constant addition (5).
/// - `NUM_BARS`: Number of Bar (S-box) applications per round (8 for M31, 4 for GL).
/// - `FIELD_BITS`: Bits per field element (31 for M31, 64 for GL).
///
/// # Constraint Degrees
///
/// - Boolean constraints on bits: degree 2.
/// - Bit reconstruction: degree 1.
/// - Chi S-box output verification: degree 4.
/// - Bricks (Feistel Type-3): degree 2.
/// - Concrete (MDS multiply): degree 2 (linear layer on degree-2 expressions
///   from Bricks, but applied to committed degree-1 values via post-Bars output).
/// - Round constant addition: degree 1.
///
/// Maximum constraint degree: **4** (from the chi S-box).
#[derive(Debug)]
pub struct MonolithAir<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
> {
    /// Round constants for each of the `NUM_FULL_ROUNDS` rounds.
    ///
    /// The final round (round `NUM_FULL_ROUNDS`) has no round constants.
    pub(crate) round_constants: [[F; WIDTH]; NUM_FULL_ROUNDS],

    /// Dense MDS matrix for the Concrete layer.
    ///
    /// `mds_matrix[row][col]` is the entry at row `row`, column `col`.
    /// The Concrete layer computes `state = mds_matrix * state`.
    pub(crate) mds_matrix: [[F; WIDTH]; WIDTH],

    /// Bit widths of each limb in the Bar decomposition.
    ///
    /// For Mersenne31: `[8, 8, 8, 7]` (three 8-bit + one 7-bit limb).
    /// For Goldilocks: `[8, 8, 8, 8, 8, 8, 8, 8]` (eight 8-bit limbs).
    ///
    /// The sum of all limb widths must equal `FIELD_BITS`.
    pub(crate) limb_bits: &'static [usize],
}

impl<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
> Clone for MonolithAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>
{
    fn clone(&self) -> Self {
        Self {
            round_constants: self.round_constants.clone(),
            mds_matrix: self.mds_matrix.clone(),
            limb_bits: self.limb_bits,
        }
    }
}

impl<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
> MonolithAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>
{
    /// Construct a `MonolithAir` from explicit parameters.
    ///
    /// # Arguments
    ///
    /// - `round_constants`: The `NUM_FULL_ROUNDS` round constant vectors.
    /// - `mds_matrix`: The dense MDS matrix (`mds_matrix[row][col]`).
    /// - `limb_bits`: Bit widths of each limb. Sum must equal `FIELD_BITS`.
    ///
    /// # Panics
    ///
    /// Panics if `limb_bits` does not sum to `FIELD_BITS`, or if `NUM_BARS > WIDTH`.
    pub fn new(
        round_constants: [[F; WIDTH]; NUM_FULL_ROUNDS],
        mds_matrix: [[F; WIDTH]; WIDTH],
        limb_bits: &'static [usize],
    ) -> Self {
        assert_eq!(
            limb_bits.iter().sum::<usize>(),
            FIELD_BITS,
            "limb_bits must sum to FIELD_BITS"
        );
        const {
            assert!(NUM_BARS <= WIDTH, "NUM_BARS must not exceed WIDTH");
        }
        Self {
            round_constants,
            mds_matrix,
            limb_bits,
        }
    }

    /// Extract a dense MDS matrix from any `MdsPermutation` implementation.
    ///
    /// Evaluates the MDS on each standard basis vector to produce the
    /// full `WIDTH × WIDTH` matrix.
    pub fn extract_mds_matrix<Mds>(mds: &Mds) -> [[F; WIDTH]; WIDTH]
    where
        F: Copy,
        Mds: p3_mds::MdsPermutation<F, WIDTH>,
    {
        // Compute each column of the matrix by applying MDS to basis vectors.
        let columns: [[F; WIDTH]; WIDTH] = core::array::from_fn(|col| {
            let mut basis = [F::ZERO; WIDTH];
            basis[col] = F::ONE;
            mds.permute_mut(&mut basis);
            basis
        });
        // Transpose: columns[col][row] → matrix[row][col].
        core::array::from_fn(|row| core::array::from_fn(|col| columns[col][row]))
    }

    /// Generate a trace with `num_hashes` random permutations.
    ///
    /// Uses a deterministic PRNG seeded with `1` for reproducible traces.
    ///
    /// The `extra_capacity_bits` parameter pre-allocates extra memory for
    /// the LDE (low-degree extension) blowup during proving.
    pub fn generate_trace_rows<B: MonolithBars<F, WIDTH> + Sync>(
        &self,
        num_hashes: usize,
        bars: &B,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F>
    where
        F: PrimeField64,
        StandardUniform: Distribution<[F; WIDTH]>,
    {
        // Deterministic PRNG for reproducible test inputs.
        let mut rng = SmallRng::seed_from_u64(1);
        let inputs = (0..num_hashes).map(|_| rng.random()).collect();
        generate_trace_rows(inputs, self, bars, extra_capacity_bits)
    }
}

impl<
    F: PrimeCharacteristicRing + Sync,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
> BaseAir<F> for MonolithAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>
{
    /// Returns the number of trace columns (the AIR width).
    fn width(&self) -> usize {
        num_cols::<WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>()
    }

    /// No next-row columns. Each permutation is fully constrained within one row.
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
> Air<AB> for MonolithAir<AB::F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &MonolithCols<AB::Var, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS> =
            main.current_slice().borrow();

        // Initialize the running state from the committed input columns.
        let mut state: [AB::Expr; WIDTH] = local.inputs.map(|x| x.into());

        // Initial Concrete layer: multiply the input state by the MDS matrix.
        // This is a linear operation, so the state remains degree 1.
        mds_multiply::<_, _, WIDTH>(&mut state, &self.mds_matrix);

        // Full rounds: Bars → Bricks → Concrete → AddRoundConstants.
        for round_idx in 0..NUM_FULL_ROUNDS {
            eval_round::<AB, WIDTH, NUM_BARS, FIELD_BITS>(
                &mut state,
                &local.full_rounds[round_idx],
                &self.mds_matrix,
                Some(&self.round_constants[round_idx]),
                self.limb_bits,
                builder,
            );
        }

        // Final round: Bars → Bricks → Concrete (no round constants).
        eval_round::<AB, WIDTH, NUM_BARS, FIELD_BITS>(
            &mut state,
            &local.final_round,
            &self.mds_matrix,
            None,
            self.limb_bits,
            builder,
        );
    }
}

/// Evaluate constraints for one Monolith round.
///
/// A round applies four operations in sequence:
///
/// 1. **Bars**: Verify bit decomposition and S-box for each of the `NUM_BARS`
///    elements. The remaining elements pass through unchanged.
/// 2. **Bricks**: Feistel Type-3 layer: `state[i] += state[i-1]^2` (degree 2).
/// 3. **Concrete**: Dense MDS matrix multiplication (linear, degree unchanged).
/// 4. **AddRoundConstants** (optional): Add field constants to the state.
///
/// After computing the result, constrain it against the committed `post` values,
/// then reset the running state to those committed values (degree 1).
#[inline]
fn eval_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    round: &MonolithRoundCols<AB::Var, WIDTH, NUM_BARS, FIELD_BITS>,
    mds_matrix: &[[AB::F; WIDTH]; WIDTH],
    round_constants: Option<&[AB::F; WIDTH]>,
    limb_bits: &[usize],
    builder: &mut AB,
) {
    // Bars layer constraints
    //
    // For each of the NUM_BARS elements:
    // 1. Verify bits are boolean: b * (1 - b) = 0.
    // 2. Verify reconstruction: sum(bits[i] * 2^i) == state[bar_idx].
    // 3. Verify S-box: bars_output == chi_sbox(bits) reconstructed.
    for (bar_idx, (input_bits, &bar_out)) in round
        .bars_input_bits
        .iter()
        .zip(round.bars_output.iter())
        .enumerate()
    {
        // Boolean constraints: each bit must be 0 or 1.
        builder.assert_bools(*input_bits);

        // Convert committed Var bits to Expr for algebraic evaluation.
        let bits: [AB::Expr; FIELD_BITS] = input_bits.map(|b| b.into());

        // Reconstruction constraint: the bits must reconstruct the Bar input.
        // sum(bits[i] * 2^i) == state[bar_idx]
        let reconstructed: AB::Expr = pack_bits_le(bits.iter().cloned());
        builder.assert_eq(reconstructed, state[bar_idx].clone());

        // S-box constraint: the committed bars_output must equal the chi S-box
        // applied to the bit-decomposed limbs.
        let sbox_output = eval_bar_sbox::<AB, FIELD_BITS>(&bits, limb_bits);
        builder.assert_eq(AB::Expr::from(bar_out), sbox_output);
    }

    // Build the post-Bars state:
    // - First NUM_BARS elements come from the committed bars_output (degree 1).
    // - Remaining elements are unchanged from the previous state.
    let mut post_bars = state.clone();
    for (bar_idx, &bar_out) in round.bars_output.iter().enumerate() {
        post_bars[bar_idx] = bar_out.into();
    }

    // Bricks layer
    //
    // Feistel Type-3:
    // - state'[0] = state[0],
    // - state'[i] = state[i] + state[i-1]^2.
    let mut post_bricks = core::array::from_fn(|i| {
        if i == 0 {
            post_bars[0].clone()
        } else {
            post_bars[i].clone() + post_bars[i - 1].clone().square()
        }
    });

    // Concrete layer
    //
    // Dense MDS matrix multiply. The post_bricks state has degree 2
    // (from the squaring in Bricks). MDS is linear, preserving degree 2.
    mds_multiply::<_, _, WIDTH>(&mut post_bricks, mds_matrix);

    // Round constants (if not the final round)
    if let Some(rc) = round_constants {
        for (s, c) in post_bricks.iter_mut().zip(rc.iter()) {
            *s += AB::Expr::from(c.clone());
        }
    }

    // Constrain post-state
    //
    // The computed state must equal the committed post values.
    // Then reset the running state to the committed values (degree 1).
    for (computed, &committed) in post_bricks.into_iter().zip(round.post.iter()) {
        builder.assert_eq(computed, committed);
    }

    // Reset state to committed values for the next round.
    *state = round.post.map(|x| x.into());
}

/// Evaluate the full Bar S-box on a bit-decomposed field element.
///
/// Applies the chi-like S-box independently to each limb (determined by
/// `limb_bits`), then reconstructs the output as a single field expression
/// using [`pack_bits_le`].
///
/// # Chi S-box Formula (per limb)
///
/// Rust's `u8::rotate_left(k)` maps bit position `i` in the result to bit
/// `(i - k) mod n` of the input. Combining with the final rotation:
///
/// ```text
///   // 8-bit: out[j] = x[(j-1)%n] XOR (andn(x[(j-2)%n], x[(j-3)%n]) * x[(j-4)%n])
///   // 7-bit: out[j] = x[(j-1)%n] XOR andn(x[(j-2)%n], x[(j-3)%n])
/// ```
///
/// Uses the `andn` and `xor` methods from [`PrimeCharacteristicRing`], matching
/// the same pattern used by the keccak-air for its chi step.
///
/// Maximum degree: 4 (XOR of degree-1 bit with degree-3 AND-NOT product).
fn eval_bar_sbox<AB: AirBuilder, const FIELD_BITS: usize>(
    bits: &[AB::Expr; FIELD_BITS],
    limb_bits: &[usize],
) -> AB::Expr {
    let mut result = AB::Expr::ZERO;
    let mut bit_offset = 0;

    for (limb_idx, &n) in limb_bits.iter().enumerate() {
        let x = &bits[bit_offset..bit_offset + n];

        // Last limb with < 8 bits uses 2-input AND (7-bit chi variant).
        let is_last_reduced = limb_idx == limb_bits.len() - 1 && n < 8;

        // Modular index subtraction within the limb.
        let sub = |base: usize, offset: usize| (base + n - (offset % n)) % n;

        // Compute each output bit using andn/xor, then pack into a field element.
        //
        // out[j] = x[sub(j,1)].xor(andn(x[sub(j,2)], x[sub(j,3)]) [* x[sub(j,4)]])
        let get_out_bit = |j: usize| -> AB::Expr {
            // andn(a, b) = (1 - a) * b  (degree 2)
            let andn = x[sub(j, 2)].clone().andn(&x[sub(j, 3)].clone());

            let chi_product = if is_last_reduced {
                andn // degree 2
            } else {
                andn * x[sub(j, 4)].clone() // degree 3
            };

            // xor(a, b) = a + b - 2ab  (degree += 1)
            x[sub(j, 1)].clone().xor(&chi_product)
        };

        // Pack limb output bits into a field element using Horner evaluation.
        let limb_value: AB::Expr = pack_bits_le((0..n).map(get_out_bit));

        // Shift limb value to its position and accumulate.
        let limb_shift = AB::F::from_u64(1u64 << bit_offset);
        result += limb_value * limb_shift;

        bit_offset += n;
    }

    result
}
