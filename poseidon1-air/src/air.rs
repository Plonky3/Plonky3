//! AIR constraint definitions for the Poseidon1 permutation.
//!
//! # Overview
//!
//! This module defines `Poseidon1Air`, the AIR that constrains the Poseidon1
//! permutation. The constraints ensure that each trace row represents a valid
//! execution of the full permutation.
//!
//! # Constraint Structure
//!
//! The constraints mirror the three phases of the permutation:
//!
//! ```text
//!   inputs ──▶ RF/2 full rounds ──▶ sparse partial rounds ──▶ RF/2 full rounds
//! ```
//!
//! Each round produces constraints that verify:
//!
//! 1. **Round constant addition**: The constant is added to the state (baked into
//!    the expression, not a separate constraint).
//!
//! 2. **S-box correctness**: The committed intermediate values (if any) and the
//!    S-box output satisfy the power-map relation `x → x^α`.
//!
//! 3. **MDS correctness**: In full rounds, the committed `post` values equal the
//!    MDS matrix applied to the post-S-box state. In partial rounds, the sparse
//!    matrix multiply is folded into expressions (no separate constraint).
//!
//! **Sparse Partial** Rounds: The partial rounds use the sparse matrix decomposition from Appendix B of
//! the Poseidon1 paper.

use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField, dot_product};
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon1::external::mds_multiply;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::columns::{Poseidon1Cols, num_cols};
use crate::{
    FullRound, FullRoundConstants, PartialRound, PartialRoundConstants, SBox, generate_trace_rows,
};

/// The Poseidon1 AIR.
///
/// Constrains one Poseidon1 permutation per trace row. The AIR evaluates
/// polynomial constraints that verify every round of the permutation.
///
/// # Type Parameters
///
/// - `F`: The prime field (must be at least 16 bits for S-box correctness).
/// - `WIDTH`: Permutation state width (`t` in the paper).
/// - `SBOX_DEGREE`: The S-box exponent `α`.
/// - `SBOX_REGISTERS`: Intermediate columns per S-box (depends on `SBOX_DEGREE`).
/// - `HALF_FULL_ROUNDS`: `RF/2`, number of full rounds per half.
/// - `PARTIAL_ROUNDS`: `RP`, number of partial rounds.
///
/// # Constraint Count
///
/// - Each full round produces `WIDTH` constraints (one per post-state element)
///   plus S-box constraints.
/// - Each partial round produces 1 constraint (for the S-box output) plus
///   S-box intermediate constraints.
///
/// Total:
///
/// ```text
///   full round constraints:    2 * HALF_FULL_ROUNDS * (WIDTH + WIDTH * sbox_constraints)
///   partial round constraints: PARTIAL_ROUNDS * (1 + sbox_constraints)
/// ```
#[derive(Debug)]
pub struct Poseidon1Air<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    /// Pre-computed constants for full rounds (initial + terminal) and MDS matrix.
    pub(crate) full_constants: FullRoundConstants<F, WIDTH>,

    /// Pre-computed constants for partial rounds (sparse matrix decomposition).
    pub(crate) partial_constants: PartialRoundConstants<F, WIDTH>,
}

impl<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> Clone for Poseidon1Air<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    fn clone(&self) -> Self {
        Self {
            full_constants: self.full_constants.clone(),
            partial_constants: self.partial_constants.clone(),
        }
    }
}

impl<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> Poseidon1Air<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    /// Construct a `Poseidon1Air` from pre-computed round constants.
    ///
    /// Use [`Poseidon1Constants::to_optimized`] to produce the two constant
    /// structs from raw Poseidon1 parameters.
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(
        full_constants: FullRoundConstants<F, WIDTH>,
        partial_constants: PartialRoundConstants<F, WIDTH>,
    ) -> Self {
        Self {
            full_constants,
            partial_constants,
        }
    }

    /// Generate a trace with `num_hashes` random permutations.
    ///
    /// Uses a deterministic PRNG seeded with `1` for reproducible traces.
    ///
    /// The `extra_capacity_bits` parameter pre-allocates extra memory for
    /// the LDE (low-degree extension) blowup during proving.
    pub fn generate_trace_rows(
        &self,
        num_hashes: usize,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F>
    where
        F: PrimeField,
        StandardUniform: Distribution<[F; WIDTH]>,
    {
        // Deterministic PRNG for reproducible test inputs.
        let mut rng = SmallRng::seed_from_u64(1);
        let inputs = (0..num_hashes).map(|_| rng.random()).collect();
        generate_trace_rows::<_, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>(
            inputs,
            &self.full_constants,
            &self.partial_constants,
            extra_capacity_bits,
        )
    }
}

impl<
    F: PrimeCharacteristicRing + Sync,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> BaseAir<F>
    for Poseidon1Air<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    /// Returns the number of trace columns (the AIR width).
    fn width(&self) -> usize {
        num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()
    }

    /// No next-row columns. Each permutation is fully constrained within one row.
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

/// Evaluate all Poseidon1 constraints for one trace row.
///
/// This is the core constraint evaluation function.
///
/// It maintains a running `state` expression array and walks through all three phases:
///
/// 1. Beginning full rounds
/// 2. Sparse partial rounds (first-round constants, m_i, then loop)
/// 3. Ending full rounds
///
/// At each round, the `state` expressions are updated and constrained against
/// the committed values. After constraining, the state is reset to the
/// committed values (degree 1) to prevent expression degree blowup.
pub(crate) fn eval<
    AB: AirBuilder,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    air: &Poseidon1Air<AB::F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    builder: &mut AB,
    local: &Poseidon1Cols<
        AB::Var,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
) {
    // Initialize the running state from the committed input columns.
    let mut state: [_; WIDTH] = local.inputs.map(|x| x.into());

    // Phase 1: Beginning full rounds (RF/2 rounds)
    //
    // Each round: add constants → S-box on all elements → MDS multiply.
    for round in 0..HALF_FULL_ROUNDS {
        eval_full_round::<AB, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.beginning_full_rounds[round],
            &air.full_constants.initial[round],
            &air.full_constants.dense_mds,
            builder,
        );
    }

    // Phase 2: Sparse partial rounds
    //
    // Add first-round constants (full WIDTH vector).
    for (s, c) in state
        .iter_mut()
        .zip(air.partial_constants.first_round_constants.iter())
    {
        *s += c.clone();
    }

    // Dense transition matrix m_i (applied once).
    mds_multiply(&mut state, &air.partial_constants.m_i);

    // Partial round loop: S-box on state[0], commit, scalar constant, sparse matmul.
    for round in 0..PARTIAL_ROUNDS {
        eval_sparse_partial_round::<AB, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.partial_rounds[round],
            if round < PARTIAL_ROUNDS - 1 {
                Some(air.partial_constants.round_constants[round].clone())
            } else {
                None
            },
            &air.partial_constants.sparse_first_row[round],
            &air.partial_constants.v[round],
            builder,
        );
    }

    // Phase 3: Ending full rounds (RF/2 rounds)
    for round in 0..HALF_FULL_ROUNDS {
        eval_full_round::<AB, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.ending_full_rounds[round],
            &air.full_constants.terminal[round],
            &air.full_constants.dense_mds,
            builder,
        );
    }
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> Air<AB>
    for Poseidon1Air<AB::F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        // Read the current row as a flat slice, then reinterpret as columns.
        let main = builder.main();
        let local = main.current_slice().borrow();

        eval::<_, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>(
            self, builder, local,
        );
    }
}

/// Evaluate constraints for one **full** round.
///
/// A full round applies three operations in sequence:
///
/// 1. **AddRoundConstants**: Add `round_constants[i]` to `state[i]` for all `i`.
/// 2. **S-box**: Apply `x → x^DEGREE` to every state element.
/// 3. **MDS multiply**: Multiply the state by the dense MDS matrix.
///
/// After the MDS multiply, the computed state is constrained to equal the
/// committed `post` values, then the running state is reset to those committed
/// values (degree 1).
#[inline]
fn eval_full_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    full_round: &FullRound<AB::Var, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[AB::F; WIDTH],
    mds_matrix: &[[AB::F; WIDTH]; WIDTH],
    builder: &mut AB,
) {
    // Step 1 & 2: For each state element, add the round constant and apply the S-box.
    for (i, (s, r)) in state.iter_mut().zip(round_constants.iter()).enumerate() {
        // AddRoundConstants: state[i] += rc[i].
        *s += r.clone();

        // S-box: state[i] = state[i]^DEGREE.
        // This also constrains any committed intermediate values.
        eval_sbox(&full_round.sbox[i], s, builder);
    }

    // Step 3: Multiply by the dense MDS matrix.
    mds_multiply(state, mds_matrix);

    // Constrain: computed state must equal committed post-state.
    // Then reset state to the committed values (degree 1).
    for (state_i, post_i) in state.iter_mut().zip(full_round.post) {
        builder.assert_eq(state_i.clone(), post_i);
        *state_i = post_i.into();
    }
}

/// Evaluate constraints for one **sparse partial** round.
///
/// 1. S-box on state[0].
/// 2. Commit the S-box output and reset degree.
/// 3. Add scalar round constant (except last round).
/// 4. Sparse matrix multiply: dot product for new state[0], rank-1 update for rest.
#[inline]
fn eval_sparse_partial_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    partial_round: &PartialRound<AB::Var, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constant: Option<AB::F>,
    first_row: &[AB::F; WIDTH],
    v: &[AB::F; WIDTH],
    builder: &mut AB,
) {
    // S-box on state[0].
    eval_sbox(&partial_round.sbox, &mut state[0], builder);

    // Commit the S-box output.
    let committed: AB::Expr = partial_round.post_sbox.into();
    builder.assert_eq(state[0].clone(), committed.clone());
    state[0] = committed;

    // Add scalar round constant (all rounds except last).
    if let Some(rc) = round_constant {
        state[0] += rc;
    }

    // Sparse matrix multiply.
    let old_s0 = state[0].clone();
    state[0] = dot_product(state.iter().cloned(), first_row.iter().cloned());

    for i in 1..WIDTH {
        state[i] += old_s0.clone() * v[i - 1].clone();
    }
}

/// Evaluate constraints for one S-box computation (`x → x^DEGREE`).
///
/// Depending on the `(DEGREE, REGISTERS)` pair, this function either:
///
/// - Computes the power directly (no extra constraints), or
/// - Constrains committed intermediate values that decompose the power.
///
/// # Supported Configurations
///
/// | DEGREE | REGISTERS | Strategy                                          |
/// |--------|-----------|---------------------------------------------------|
/// | 3      | 0         | Compute `x^3` directly (constraint degree 3).     |
/// | 5      | 0         | Compute `x^5` directly (constraint degree 5).     |
/// | 7      | 0         | Compute `x^7` directly (constraint degree 7).     |
/// | 5      | 1         | Commit `x^3`, constrain `x^3 = x^2 * x`,          |
/// |        |           | output `x^3 * x^2` (constraint degree 3).         |
/// | 7      | 1         | Commit `x^3`, constrain `x^3 = x * x * x`,        |
/// |        |           | output `(x^3)^2 * x` (constraint degree 3).       |
/// | 11     | 2         | Commit `x^3` and `x^9`,                           |
/// |        |           | constrain `x^3 = x^2 * x` and `x^9 = (x^3)^3`,    |
/// |        |           | output `x^9 * x^2` (constraint degree 3).         |
///
/// # Panics
///
/// Panics if `(DEGREE, REGISTERS)` is not one of the supported configurations.
#[inline]
fn eval_sbox<AB, const DEGREE: u64, const REGISTERS: usize>(
    sbox: &SBox<AB::Var, DEGREE, REGISTERS>,
    x: &mut AB::Expr,
    builder: &mut AB,
) where
    AB: AirBuilder,
{
    *x = match (DEGREE, REGISTERS) {
        // Direct computation (no intermediate columns)

        // x^3: single cube operation.
        (3, 0) => x.cube(),

        // x^5: computed directly.
        (5, 0) => x.exp_const_u64::<5>(),

        // x^7: computed directly.
        (7, 0) => x.exp_const_u64::<7>(),

        // With committed intermediates (lower constraint degree)

        // x^5 via committed x^3:
        //   Constrain: committed_x3 = x^2 * x
        //   Output:    committed_x3 * x^2 = x^5
        (5, 1) => {
            let committed_x3 = sbox.0[0].into();
            let x2 = x.square();
            builder.assert_eq(committed_x3.clone(), x2.clone() * x.clone());
            committed_x3 * x2
        }

        // x^7 via committed x^3:
        //   Constrain: committed_x3 = x^3
        //   Output:    (committed_x3)^2 * x = x^7
        (7, 1) => {
            let committed_x3 = sbox.0[0].into();
            builder.assert_eq(committed_x3.clone(), x.cube());
            committed_x3.square() * x.clone()
        }

        // x^11 via committed x^3:
        //   Constrain: committed_x3 = x^3
        //   Output:    (committed_x3)^3 * x^2 = x^11
        //   Max constraint degree: 5.
        (11, 1) => {
            let committed_x3 = sbox.0[0].into();
            builder.assert_eq(committed_x3.clone(), x.cube());
            committed_x3.cube() * x.square()
        }

        // x^11 via committed x^3 and x^9:
        //   Constrain: committed_x3 = x^2 * x
        //   Constrain: committed_x9 = (committed_x3)^3
        //   Output:    committed_x9 * x^2 = x^11
        (11, 2) => {
            let committed_x3 = sbox.0[0].into();
            let committed_x9 = sbox.0[1].into();
            let x2 = x.square();
            builder.assert_eq(committed_x3.clone(), x2.clone() * x.clone());
            builder.assert_eq(committed_x9.clone(), committed_x3.cube());
            committed_x9 * x2
        }

        _ => panic!(
            "Unexpected (DEGREE, REGISTERS) of ({}, {})",
            DEGREE, REGISTERS
        ),
    }
}
