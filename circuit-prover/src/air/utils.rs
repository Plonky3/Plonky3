use alloc::vec::Vec;
use core::borrow::Borrow;
use core::iter;

use p3_air::{AirBuilder, AirLayout};
use p3_field::Field;
use p3_lookup::lookup_traits::{Direction, LookupInput};
use p3_matrix::Matrix;
use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression, SymbolicVariable};

use super::alu_columns::{AluMainLaneCols, AluPrepLaneCols, PREP_LANE_WIDTH, alu_main_lane_width};
use super::column_layout::WITNESS_LOOKUP_PREP_COL_MAP;

pub fn get_index_lookups<F: Field, const D: usize>(
    main_start: usize,
    preprocessed_start: usize,
    num_lookups: usize,
    main: &[SymbolicVariable<F>],
    preprocessed: &[SymbolicVariable<F>],
    direction: Direction,
) -> Vec<LookupInput<F>> {
    (0..num_lookups)
        .map(|i| {
            let idx = SymbolicExpression::from(
                preprocessed[preprocessed_start + WITNESS_LOOKUP_PREP_COL_MAP.witness_idx + i],
            );

            let multiplicity = SymbolicExpression::from(
                preprocessed[preprocessed_start + WITNESS_LOOKUP_PREP_COL_MAP.multiplicity],
            );

            let values = (0..D).map(|j| SymbolicExpression::from(main[main_start + i * D + j]));
            let inps = iter::once(idx).chain(values).collect::<Vec<_>>();

            (inps, multiplicity, direction)
        })
        .collect()
}

/// Get ALU lookups for the 4 operands (a, b, c, out).
///
/// Uses [`super::alu_columns`] for column offsets.
pub fn get_alu_index_lookups<F: Field, const D: usize>(
    main_start: usize,
    preprocessed_start: usize,
    main: &[SymbolicVariable<F>],
    preprocessed: &[SymbolicVariable<F>],
) -> Vec<LookupInput<F>> {
    let prep: &AluPrepLaneCols<_> =
        preprocessed[preprocessed_start..preprocessed_start + PREP_LANE_WIDTH].borrow();
    let lane_main: &AluMainLaneCols<_, D> =
        main[main_start..main_start + alu_main_lane_width::<D>()].borrow();

    let mult_a = SymbolicExpression::from(prep.mult_a);
    let mult_b = SymbolicExpression::from(prep.mult_b);
    let mult_out = SymbolicExpression::from(prep.mult_out);
    let a_is_reader = SymbolicExpression::from(prep.a_is_reader);
    let c_is_reader = SymbolicExpression::from(prep.c_is_reader);

    let eff_mult_a = mult_a.clone() * a_is_reader;
    let eff_mult_c = mult_a * c_is_reader;

    let multiplicities = [eff_mult_a, mult_b, eff_mult_c, mult_out];

    let idx_vars = [prep.a_idx, prep.b_idx, prep.c_idx, prep.out_idx];
    let operands = [&lane_main.a, &lane_main.b, &lane_main.c, &lane_main.out];

    (0..4)
        .map(|i| {
            let idx = SymbolicExpression::from(idx_vars[i]);

            let values = (0..D).map(|j| SymbolicExpression::from(operands[i][j]));
            let inps = iter::once(idx).chain(values).collect::<Vec<_>>();

            (inps, multiplicities[i].clone(), Direction::Receive)
        })
        .collect()
}

/// Helper to create symbolic air builder and extract symbolic variables for lookup generation.
///
/// Returns `(symbolic_main_local, preprocessed_local)` slices for use in lookup generation.
pub fn create_symbolic_variables<F: Field>(
    preprocessed_width: usize,
    main_width: usize,
    num_public_values: usize,
    num_permutation_cols: usize,
) -> (Vec<SymbolicVariable<F>>, Vec<SymbolicVariable<F>>) {
    let layout = AirLayout {
        preprocessed_width,
        main_width,
        num_public_values,
        permutation_width: num_permutation_cols,
        num_permutation_challenges: 0,
        num_permutation_values: 0,
        num_periodic_columns: 0,
    };
    let symbolic_air_builder = SymbolicAirBuilder::<F>::new(layout);

    let symbolic_main = symbolic_air_builder.main();
    let symbolic_main_local = symbolic_main.row_slice(0).unwrap().to_vec();

    let preprocessed = symbolic_air_builder.preprocessed().clone();
    let preprocessed_local = preprocessed
        .row_slice(0)
        .expect("The preprocessed matrix has only one row?")
        .to_vec();

    (symbolic_main_local, preprocessed_local)
}
