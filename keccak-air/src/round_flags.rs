use core::borrow::Borrow;

use p3_air::AirBuilder;
use p3_field::AbstractField;
use p3_matrix::MatrixRowSlices;

use crate::columns::KeccakCols;
use crate::NUM_ROUNDS;

pub(crate) fn eval_round_flags<AB: AirBuilder>(builder: &mut AB) {
    let main = builder.main();
    let local: &KeccakCols<AB::Var> = main.row_slice(0).borrow();
    let next: &KeccakCols<AB::Var> = main.row_slice(1).borrow();

    let not_export = AB::Expr::one() - local.export.clone();

    // Initially, the first step flag should be 1 while the others should be 0.
    builder.when_first_row().when(not_export.clone()).assert_one(local.step_flags[0]);
    for i in 1..NUM_ROUNDS {
        builder.when_first_row().when(not_export.clone()).assert_zero(local.step_flags[i]);
    }

    // On non-export rows that aren't transitioning to export rows
    let next_not_export = AB::Expr::one() - next.export.clone();
    for i in 0..NUM_ROUNDS {
        let current_round_flag = local.step_flags[i];
        let next_round_flag = next.step_flags[(i + 1) % NUM_ROUNDS];
        builder
            .when_transition()
            .when(not_export.clone())
            .when(next_not_export.clone())
            .assert_eq(next_round_flag, current_round_flag);
    }

    // When transitioning to an export row, the next row should only have the final flag set
    builder
        .when_transition()
        .when(not_export.clone())
        .when(next.export)
        .assert_one(next.step_flags[NUM_ROUNDS - 1]);

    for i in 0..NUM_ROUNDS-1 {
        builder
            .when_transition()
            .when(not_export.clone())
            .when(next.export)
            .assert_zero(next.step_flags[i]);
    }
}