use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{PrimeCharacteristicRing, PrimeField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon::external::mds_multiply;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::columns::{PoseidonCols, num_cols};
use crate::constants::RoundConstants;
use crate::{FullRound, PartialRound, SBox, generate_trace_rows};

/// Assumes the field size is at least 16 bits.
#[derive(Debug)]
pub struct PoseidonAir<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    pub(crate) constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
}

impl<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> Clone for PoseidonAir<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    fn clone(&self) -> Self {
        Self {
            constants: self.constants.clone(),
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
> PoseidonAir<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    pub const fn new(
        constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    ) -> Self {
        Self { constants }
    }

    pub fn generate_trace_rows(
        &self,
        num_hashes: usize,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F>
    where
        F: PrimeField,
        StandardUniform: Distribution<[F; WIDTH]>,
    {
        let mut rng = SmallRng::seed_from_u64(1);
        let inputs = (0..num_hashes).map(|_| rng.random()).collect();
        generate_trace_rows::<_, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>(
            inputs,
            &self.constants,
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
    for PoseidonAir<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

pub(crate) fn eval<
    AB: AirBuilder,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    air: &PoseidonAir<AB::F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    builder: &mut AB,
    local: &PoseidonCols<
        AB::Var,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
) {
    let mut state: [_; WIDTH] = local.inputs.map(|x| x.into());

    // No initial linear layer (unlike Poseidon2).

    for round in 0..HALF_FULL_ROUNDS {
        eval_full_round::<AB, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.beginning_full_rounds[round],
            &air.constants.beginning_full_round_constants[round],
            &air.constants.mds_matrix,
            builder,
        );
    }

    for round in 0..PARTIAL_ROUNDS {
        eval_partial_round::<AB, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.partial_rounds[round],
            &air.constants.partial_round_constants[round],
            &air.constants.mds_matrix,
            builder,
        );
    }

    // Add residual vector after all partial rounds.
    for (s, r) in state
        .iter_mut()
        .zip(air.constants.partial_round_residual.iter())
    {
        *s += r.clone();
    }

    for round in 0..HALF_FULL_ROUNDS {
        eval_full_round::<AB, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.ending_full_rounds[round],
            &air.constants.ending_full_round_constants[round],
            &air.constants.mds_matrix,
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
    for PoseidonAir<AB::F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("The matrix is empty?");
        let local = (*local).borrow();

        eval::<_, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>(
            self, builder, local,
        );
    }
}

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
    for (i, (s, r)) in state.iter_mut().zip(round_constants.iter()).enumerate() {
        *s += r.clone();
        eval_sbox(&full_round.sbox[i], s, builder);
    }
    mds_multiply(state, mds_matrix);
    for (state_i, post_i) in state.iter_mut().zip(full_round.post) {
        builder.assert_eq(state_i.clone(), post_i);
        *state_i = post_i.into();
    }
}

#[inline]
fn eval_partial_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    partial_round: &PartialRound<AB::Var, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constant: &AB::F,
    mds_matrix: &[[AB::F; WIDTH]; WIDTH],
    builder: &mut AB,
) {
    state[0] += round_constant.clone();
    eval_sbox(&partial_round.sbox, &mut state[0], builder);

    // Apply MDS to the full state.
    mds_multiply(state, mds_matrix);

    // Constrain the full post-state.
    for (state_i, post_i) in state.iter_mut().zip(partial_round.post) {
        builder.assert_eq(state_i.clone(), post_i);
        *state_i = post_i.into();
    }
}

/// Evaluates the S-box over a degree-1 expression `x`.
///
/// # Panics
///
/// This method panics if the number of `REGISTERS` is not chosen optimally for the given
/// `DEGREE` or if the `DEGREE` is not supported by the S-box. The supported degrees are
/// `3`, `5`, `7`, and `11`.
#[inline]
fn eval_sbox<AB, const DEGREE: u64, const REGISTERS: usize>(
    sbox: &SBox<AB::Var, DEGREE, REGISTERS>,
    x: &mut AB::Expr,
    builder: &mut AB,
) where
    AB: AirBuilder,
{
    *x = match (DEGREE, REGISTERS) {
        (3, 0) => x.cube(),
        (5, 0) => x.exp_const_u64::<5>(),
        (7, 0) => x.exp_const_u64::<7>(),
        (5, 1) => {
            let committed_x3 = sbox.0[0].into();
            let x2 = x.square();
            builder.assert_eq(committed_x3.clone(), x2.clone() * x.clone());
            committed_x3 * x2
        }
        (7, 1) => {
            let committed_x3 = sbox.0[0].into();
            builder.assert_eq(committed_x3.clone(), x.cube());
            committed_x3.square() * x.clone()
        }
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
