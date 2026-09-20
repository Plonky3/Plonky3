use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;
use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, PrimeCharacteristicRing, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::columns::{Poseidon2Cols, num_cols};
use crate::constants::RoundConstants;
use crate::{FullRound, PartialRound, SBox, generate_trace_rows};

/// Assumes the field size is at least 16 bits.
#[derive(Debug)]
pub struct Poseidon2Air<
    F: PrimeCharacteristicRing,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    pub(crate) constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    _phantom: PhantomData<LinearLayers>,
}

impl<
    F: PrimeCharacteristicRing,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> Clone
    for Poseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    fn clone(&self) -> Self {
        Self {
            constants: self.constants.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<
    F: PrimeCharacteristicRing,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>
    Poseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    pub const fn new(
        constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    ) -> Self {
        Self {
            constants,
            _phantom: PhantomData,
        }
    }

    /// Generate a trace over `num_hashes` fixed-seed random permutation inputs.
    ///
    /// This is for benches/examples only — it does not let callers supply the actual
    /// inputs being hashed. Use the free [`generate_trace_rows`] function directly to
    /// prove specific inputs.
    pub fn generate_random_trace_rows(
        &self,
        num_hashes: usize,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F>
    where
        F: PrimeField,
        LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
        StandardUniform: Distribution<[F; WIDTH]>,
    {
        let mut rng = SmallRng::seed_from_u64(1);
        let inputs = (0..num_hashes).map(|_| rng.random()).collect();
        generate_trace_rows::<
            _,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(inputs, &self.constants, extra_capacity_bits)
    }
}

impl<
    F: PrimeCharacteristicRing + Sync,
    LinearLayers: Sync,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> BaseAir<F>
    for Poseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    fn width(&self) -> usize {
        num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(sbox_constraint_degree(SBOX_DEGREE, SBOX_REGISTERS))
    }
}

/// The maximum degree among the constraints emitted by [`eval_sbox`] for a given
/// `(DEGREE, REGISTERS)` configuration.
pub(crate) const fn sbox_constraint_degree(degree: u64, registers: usize) -> usize {
    match (degree, registers) {
        (3, 0) => 3,
        (5, 0) => 5,
        (7, 0) => 7,
        (5, 1) | (7, 1) | (11, 2) => 3,
        _ => panic!("Unexpected (DEGREE, REGISTERS)"),
    }
}

pub(crate) fn eval<
    AB: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    air: &Poseidon2Air<
        AB::F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
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
    let mut state: [_; WIDTH] = local.inputs.map(|x| x.into());

    LinearLayers::external_linear_layer(&mut state);

    for round in 0..HALF_FULL_ROUNDS {
        eval_full_round::<_, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.beginning_full_rounds[round],
            &air.constants.beginning_full_round_constants[round],
            builder,
        );
    }

    for round in 0..PARTIAL_ROUNDS {
        eval_partial_round::<_, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.partial_rounds[round],
            &air.constants.partial_round_constants[round],
            builder,
        );
    }

    for round in 0..HALF_FULL_ROUNDS {
        eval_full_round::<_, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.ending_full_rounds[round],
            &air.constants.ending_full_round_constants[round],
            builder,
        );
    }
}

impl<
    AB: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> Air<AB>
    for Poseidon2Air<
        AB::F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice().borrow();

        eval::<_, _, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>(
            self, builder, local,
        );
    }
}

#[inline]
fn eval_full_round<
    AB: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    full_round: &FullRound<AB::Var, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[AB::F; WIDTH],
    builder: &mut AB,
) {
    for (i, (s, r)) in state.iter_mut().zip(round_constants.iter()).enumerate() {
        *s += r.dup();
        eval_sbox(&full_round.sbox[i], s, builder);
    }
    LinearLayers::external_linear_layer(state);
    for (state_i, post_i) in state.iter_mut().zip(full_round.post) {
        builder.assert_eq(state_i.clone(), post_i);
        *state_i = post_i.into();
    }
}

#[inline]
fn eval_partial_round<
    AB: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    partial_round: &PartialRound<AB::Var, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constant: &AB::F,
    builder: &mut AB,
) {
    state[0] += round_constant.dup();
    eval_sbox(&partial_round.sbox, &mut state[0], builder);

    builder.assert_eq(state[0].dup(), partial_round.post_sbox);
    state[0] = partial_round.post_sbox.into();

    LinearLayers::internal_linear_layer(state);
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
            builder.assert_eq(committed_x3.dup(), x2.dup() * x.dup());
            committed_x3 * x2
        }
        (7, 1) => {
            let committed_x3 = sbox.0[0].into();
            builder.assert_eq(committed_x3.dup(), x.cube());
            committed_x3.square() * x.dup()
        }
        (11, 2) => {
            let committed_x3 = sbox.0[0].into();
            let committed_x9 = sbox.0[1].into();
            let x2 = x.square();
            builder.assert_eq(committed_x3.dup(), x2.dup() * x.dup());
            builder.assert_eq(committed_x9.dup(), committed_x3.cube());
            committed_x9 * x2
        }
        _ => panic!(
            "Unexpected (DEGREE, REGISTERS) of ({}, {})",
            DEGREE, REGISTERS
        ),
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::borrow::BorrowMut;

    use p3_air::{check_all_constraints, check_constraints};
    use p3_baby_bear::{
        BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS, BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16,
        BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL, BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
        BABYBEAR_POSEIDON2_RC_16_INTERNAL, BABYBEAR_S_BOX_DEGREE, BabyBear,
        GenericPoseidon2LinearLayersBabyBear, default_babybear_poseidon2_16,
    };
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_symmetric::Permutation;

    use super::*;
    use crate::RoundConstants;

    type F = BabyBear;
    const WIDTH: usize = 16;
    const SBOX_DEGREE: u64 = BABYBEAR_S_BOX_DEGREE;
    const HALF_FULL_ROUNDS: usize = BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS;
    const PARTIAL_ROUNDS: usize = BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16;
    const NUM_HASHES: usize = 8;

    type Air<const REGISTERS: usize> = Poseidon2Air<
        F,
        GenericPoseidon2LinearLayersBabyBear,
        WIDTH,
        SBOX_DEGREE,
        REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >;
    type Cols<const REGISTERS: usize> =
        Poseidon2Cols<F, WIDTH, SBOX_DEGREE, REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;

    /// The AIR over the canonical BabyBear constants, so the trace can be checked against
    /// [`default_babybear_poseidon2_16`] rather than only against itself.
    fn canonical_air<const REGISTERS: usize>() -> Air<REGISTERS> {
        Poseidon2Air::new(RoundConstants::new(
            BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
            BABYBEAR_POSEIDON2_RC_16_INTERNAL,
            BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
        ))
    }

    fn inputs() -> Vec<[F; WIDTH]> {
        (0..NUM_HASHES)
            .map(|i| core::array::from_fn(|j| F::from_usize(i * WIDTH + j)))
            .collect()
    }

    fn trace<const REGISTERS: usize>(air: &Air<REGISTERS>) -> RowMajorMatrix<F> {
        generate_trace_rows::<
            _,
            GenericPoseidon2LinearLayersBabyBear,
            WIDTH,
            SBOX_DEGREE,
            REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(inputs(), &air.constants, 0)
    }

    fn row_mut<const REGISTERS: usize>(
        trace: &mut RowMajorMatrix<F>,
        row: usize,
    ) -> &mut Cols<REGISTERS> {
        let width = trace.width;
        trace.values[row * width..(row + 1) * width].borrow_mut()
    }

    fn known_answer<const REGISTERS: usize>() {
        let air = canonical_air::<REGISTERS>();
        let mut trace = trace(&air);
        let reference = default_babybear_poseidon2_16();
        for (row, input) in inputs().into_iter().enumerate() {
            let cols = row_mut::<REGISTERS>(&mut trace, row);
            assert_eq!(cols.inputs, input);
            let expected = reference.permute(input);
            assert_eq!(cols.ending_full_rounds[HALF_FULL_ROUNDS - 1].post, expected);
        }
    }

    #[test]
    fn test_known_answer_babybear_16() {
        known_answer::<0>();
        known_answer::<1>();
    }

    #[test]
    fn test_constraint_satisfaction_babybear_16() {
        let air = canonical_air::<0>();
        check_constraints(&air, &trace(&air), &[]);
        let air = canonical_air::<1>();
        check_constraints(&air, &trace(&air), &[]);
    }

    /// Apply `mutate` to row 1 of an honest trace and require at least one violated constraint.
    fn assert_mutation_detected<const REGISTERS: usize>(
        what: &str,
        mutate: impl FnOnce(&mut Cols<REGISTERS>),
    ) {
        let air = canonical_air::<REGISTERS>();
        let mut trace = trace(&air);
        mutate(row_mut::<REGISTERS>(&mut trace, 1));
        let report = check_all_constraints(&air, &trace, &[], Some(1));
        assert!(
            !report.is_ok(),
            "{what}: corrupted trace passed every constraint"
        );
    }

    #[test]
    fn test_corrupted_input_detected() {
        // An input feeds the first full round through the linear layer, so every post cell of
        // that round disagrees with it.
        assert_mutation_detected::<1>("input", |cols| cols.inputs[3] += F::ONE);
        assert_mutation_detected::<0>("input", |cols| cols.inputs[3] += F::ONE);
    }

    #[test]
    fn test_corrupted_full_round_post_detected() {
        assert_mutation_detected::<1>("beginning full-round post", |cols| {
            cols.beginning_full_rounds[1].post[5] += F::ONE;
        });
        assert_mutation_detected::<0>("ending full-round post", |cols| {
            cols.ending_full_rounds[2].post[0] += F::ONE;
        });
    }

    #[test]
    fn test_corrupted_partial_round_post_detected() {
        assert_mutation_detected::<1>("partial-round post_sbox", |cols| {
            cols.partial_rounds[6].post_sbox += F::ONE;
        });
        assert_mutation_detected::<0>("partial-round post_sbox", |cols| {
            cols.partial_rounds[0].post_sbox += F::ONE;
        });
    }

    #[test]
    fn test_corrupted_sbox_register_detected() {
        // With one register the S-box is split as x^3 · x^3 · x, so a wrong intermediate must
        // be caught by its own constraint rather than absorbed downstream.
        assert_mutation_detected::<1>("full-round sbox register", |cols| {
            cols.beginning_full_rounds[0].sbox[2].0[0] += F::ONE;
        });
        assert_mutation_detected::<1>("partial-round sbox register", |cols| {
            cols.partial_rounds[12].sbox.0[0] += F::ONE;
        });
    }

    #[test]
    fn test_corrupted_output_detected() {
        assert_mutation_detected::<1>("final output", |cols| {
            cols.ending_full_rounds[HALF_FULL_ROUNDS - 1].post[15] += F::ONE;
        });
    }

    /// A forged S-box register with every later cell recomputed from it.
    ///
    /// Single-cell corruption is always caught by the *next* constraint in the chain, so it
    /// cannot tell whether the register's own constraint exists. This trace satisfies every
    /// other constraint of the last full round by construction, so it passes if and only if
    /// the `committed_x3 == x^3` check is missing.
    #[test]
    fn test_forged_sbox_register_detected() {
        let air = canonical_air::<1>();
        let mut trace = trace(&air);
        let last = HALF_FULL_ROUNDS - 1;
        let constants = air.constants.ending_full_round_constants()[last];
        let cols = row_mut::<1>(&mut trace, 1);

        // Input to the last full round is the previous round's post.
        let mut state = cols.ending_full_rounds[last - 1].post;
        let forged_x3 = cols.ending_full_rounds[last].sbox[0].0[0] + F::ONE;
        for (i, s) in state.iter_mut().enumerate() {
            *s += constants[i];
            *s = if i == 0 {
                forged_x3.square() * *s
            } else {
                s.exp_const_u64::<7>()
            };
        }
        GenericPoseidon2LinearLayersBabyBear::external_linear_layer(&mut state);
        cols.ending_full_rounds[last].sbox[0].0[0] = forged_x3;
        cols.ending_full_rounds[last].post = state;

        let honest_output = default_babybear_poseidon2_16().permute(cols.inputs);
        assert_ne!(
            state, honest_output,
            "the forgery must change the permutation output"
        );
        let report = check_all_constraints(&air, &trace, &[], Some(1));
        assert!(
            !report.is_ok(),
            "a forged S-box register with a consistent post row passed every constraint"
        );
    }

    /// Every column of the layout is load-bearing: bumping any single cell of an honest row
    /// violates at least one constraint, so no cell is left unconstrained by `eval`.
    fn every_column_is_constrained<const REGISTERS: usize>() {
        let air = canonical_air::<REGISTERS>();
        let honest = trace(&air);
        let width = honest.width();
        let mut unconstrained = Vec::new();
        for col in 0..width {
            let mut trace = honest.clone();
            trace.values[width + col] += F::ONE;
            if check_all_constraints(&air, &trace, &[], Some(1)).is_ok() {
                unconstrained.push(col);
            }
        }
        assert!(
            unconstrained.is_empty(),
            "columns accept an arbitrary change without any constraint failing: {unconstrained:?}"
        );
    }

    #[test]
    fn test_every_column_is_constrained() {
        every_column_is_constrained::<0>();
        every_column_is_constrained::<1>();
    }
}
