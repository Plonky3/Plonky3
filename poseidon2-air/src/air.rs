use alloc::vec::Vec;
use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::columns::{num_cols, Poseidon2Cols};
use crate::{FullRound, PartialRound, SBox};

/// Assumes the field size is at least 16 bits.
///
/// ***WARNING***: this is a stub for now, not ready to use.
#[derive(Debug)]
pub struct Poseidon2Air<
    F: Field,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    beginning_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    partial_round_constants: [F; PARTIAL_ROUNDS],
    ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
}

impl<
        F: Field,
        const WIDTH: usize,
        const SBOX_DEGREE: usize,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > Poseidon2Air<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    pub fn new_from_rng<R: Rng>(rng: &mut R) -> Self
    where
        Standard: Distribution<F> + Distribution<[F; WIDTH]>,
    {
        let beginning_full_round_constants = rng
            .sample_iter(Standard)
            .take(HALF_FULL_ROUNDS)
            .collect::<Vec<[F; WIDTH]>>()
            .try_into()
            .unwrap();
        let partial_round_constants = rng
            .sample_iter(Standard)
            .take(PARTIAL_ROUNDS)
            .collect::<Vec<F>>()
            .try_into()
            .unwrap();
        let ending_full_round_constants = rng
            .sample_iter(Standard)
            .take(HALF_FULL_ROUNDS)
            .collect::<Vec<[F; WIDTH]>>()
            .try_into()
            .unwrap();
        Self {
            beginning_full_round_constants,
            partial_round_constants,
            ending_full_round_constants,
        }
    }
}

impl<
        F: Field,
        const WIDTH: usize,
        const SBOX_DEGREE: usize,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > BaseAir<F>
    for Poseidon2Air<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()
    }
}

impl<
        AB: AirBuilder,
        const WIDTH: usize,
        const SBOX_DEGREE: usize,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > Air<AB>
    for Poseidon2Air<AB::F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Poseidon2Cols<
            AB::Var,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = (*local).borrow();

        eval(self, builder, local);
    }
}

pub(crate) fn eval<
    AB: AirBuilder,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    air: &Poseidon2Air<AB::F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    builder: &mut AB,
    local: &Poseidon2Cols<
        AB::Var,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
) {
    let mut state: [AB::Expr; WIDTH] = local.inputs.map(|x| x.into());

    // assert_eq!(
    //     L::WIDTH,
    //     WIDTH,
    //     "The WIDTH for this STARK does not match the Linear Layer WIDTH."
    // );

    // L::matmul_external(state);
    for round in 0..HALF_FULL_ROUNDS {
        eval_full_round(
            &mut state,
            &local.beginning_full_rounds[round],
            &air.beginning_full_round_constants[round],
            builder,
        );
    }

    for round in 0..PARTIAL_ROUNDS {
        eval_partial_round(
            &mut state,
            &local.partial_rounds[round],
            &air.partial_round_constants[round],
            builder,
        );
    }

    for round in 0..HALF_FULL_ROUNDS {
        eval_full_round(
            &mut state,
            &local.ending_full_rounds[round],
            &air.ending_full_round_constants[round],
            builder,
        );
    }
}

#[inline]
fn eval_full_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    full_round: &FullRound<AB::Var, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[AB::F; WIDTH],
    builder: &mut AB,
) {
    for (i, (s, r)) in state.iter_mut().zip(round_constants.iter()).enumerate() {
        *s = s.clone() + *r;
        eval_sbox(&full_round.sbox[i], s, builder);
    }
    // L::matmul_external(state);
}

#[inline]
fn eval_partial_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    partial_round: &PartialRound<AB::Var, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constant: &AB::F,
    builder: &mut AB,
) {
    state[0] = state[0].clone() + *round_constant;
    eval_sbox(&partial_round.sbox, &mut state[0], builder);
    // L::matmul_internal(state, internal_matrix_diagonal);
}

/// Evaluates the S-BOX over a degree-`1` expression `x`.
///
/// # Panics
///
/// This method panics if the number of `REGISTERS` is not chosen optimally for the given
/// `DEGREE` or if the `DEGREE` is not supported by the S-BOX. The supported degrees are
/// `3`, `5`, `7`, and `11`.
///
/// # Efficiency Note
///
/// This method computes the S-BOX by computing the cube of `x` and then successively
/// multiplying the running sum by the cube of `x` until the last multiplication where we use
/// the appropriate power to reach the final product:
///
/// ```text
/// (x^3) * (x^3) * ... * (x^k) where k = d mod 3
/// ```
///
/// The intermediate powers are stored in the auxiliary column registers. To maximize the
/// efficiency of the registers we try to do three multiplications per round. This algorithm
/// only multiplies the cube of `x` but a more optimal product would be to find the base-3
/// decomposition of the `DEGREE` and use that to generate the addition chain. Even this is not
/// the optimal number of multiplications for all possible degrees, but for the S-BOX powers we
/// are interested in for Poseidon2 (namely `3`, `5`, `7`, and `11`), we get the optimal number
/// with this algorithm. We use the following register table:
///
/// | `DEGREE` | `REGISTERS` |
/// |:--------:|:-----------:|
/// | `3`      | `1`         |
/// | `5`      | `2`         |
/// | `7`      | `3`         |
/// | `11`     | `3`         |
///
/// We record this table in [`Self::OPTIMAL_REGISTER_COUNT`] and this choice of registers is
/// enforced by this method.
#[inline]
fn eval_sbox<AB, const DEGREE: usize, const REGISTERS: usize>(
    sbox: &SBox<AB::Var, DEGREE, REGISTERS>,
    x: &mut AB::Expr,
    builder: &mut AB,
) where
    AB: AirBuilder,
{
    // assert_ne!(REGISTERS, 0, "The number of REGISTERS must be positive.");
    // assert!(DEGREE <= 11, "The DEGREE must be less than or equal to 11.");
    // assert_eq!(
    //     REGISTERS,
    //     Self::OPTIMAL_REGISTER_COUNT[DEGREE],
    //     "The number of REGISTERS must be optimal for the given DEGREE."
    // );

    let x2 = x.square();
    let x3 = x2.clone() * x.clone();
    load(sbox, 0, x3.clone(), builder);
    if REGISTERS == 1 {
        *x = sbox.0[0].into();
        return;
    }
    if DEGREE == 11 {
        (1..REGISTERS - 1).for_each(|j| load_product(sbox, j, &[0, 0, j - 1], builder));
    } else {
        (1..REGISTERS - 1).for_each(|j| load_product(sbox, j, &[0, j - 1], builder));
    }
    load_last_product(sbox, x.clone(), x2, x3, builder);
    *x = sbox.0[REGISTERS - 1].into();
}

/// Loads `value` into the `i`-th S-BOX register.
#[inline]
fn load<AB, const SBOX_DEGREE: usize, const SBOX_REGISTERS: usize>(
    _sbox: &SBox<AB::Var, SBOX_DEGREE, SBOX_REGISTERS>,
    _i: usize,
    _value: AB::Expr,
    _builder: &mut AB,
) where
    AB: AirBuilder,
{
    // builder.assert_eq(sbox.0[i].into(), value);
}

/// Loads the product over all `product` indices the into the `i`-th S-BOX register.
#[inline]
fn load_product<AB, const SBOX_DEGREE: usize, const SBOX_REGISTERS: usize>(
    _sbox: &SBox<AB::Var, SBOX_DEGREE, SBOX_REGISTERS>,
    _i: usize,
    _product: &[usize],
    _builder: &mut AB,
) where
    AB: AirBuilder,
{
    // assert!(
    //     product.len() <= 3,
    //     "Product is too big. We can only compute at most degree-3 constraints."
    // );
    // load(
    //     sbox,
    //     i,
    //     product.iter().map(|j| AB::Expr::from(self.0[*j])).product(),
    //     builder,
    // );
}

/// Loads the final product into the last S-BOX register. The final term in the product is
/// `pow(x, DEGREE % 3)`.
#[inline]
fn load_last_product<AB, const SBOX_DEGREE: usize, const SBOX_REGISTERS: usize>(
    _sbox: &SBox<AB::Var, SBOX_DEGREE, SBOX_REGISTERS>,
    _x: AB::Expr,
    _x2: AB::Expr,
    _x3: AB::Expr,
    _builder: &mut AB,
) where
    AB: AirBuilder,
{
    // load(
    //     sbox,
    //     REGISTERS - 1,
    //     [x3, x, x2][DEGREE % 3].clone() * AB::Expr::from(self.0[REGISTERS - 2]),
    //     builder,
    // );
}
