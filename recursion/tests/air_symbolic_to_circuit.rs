mod common;

use p3_air::{Air, BaseAir, RowWindow};
use p3_circuit::symbolic::{ColumnsTargets, RowSelectorsTargets};
use p3_circuit::{CircuitBuilder, CircuitError};
use p3_circuit_prover::air::{AluAir, ConstAir, PublicAir};
use p3_lookup::logup::LogUpGadget;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_poseidon2_air::RoundConstants;
use p3_poseidon2_circuit_air::Poseidon2CircuitAirBabyBearD4Width16;
use p3_recursion::traits::{LookupMetadata, RecursiveAir};
use p3_recursion::types::RecursiveLagrangeSelectors;
use p3_test_utils::baby_bear_params::*;
use p3_uni_stark::{SymbolicAirBuilder, VerifierConstraintFolder};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

type Challenge = F;
type ChallengeMmcs = ExtensionMmcs<F, Challenge, MyMmcs>;
type MyPcs = TwoAdicFriPcs<F, Dft, MyMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<MyPcs, Challenge, Challenger>;

fn run_recursive<A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    rng: &mut SmallRng,
) -> Result<(), CircuitError>
where
    A: BaseAir<F>
        + RecursiveAir<F, F, LogUpGadget>
        + Air<SymbolicAirBuilder<F>>
        + for<'a> Air<VerifierConstraintFolder<'a, MyConfig>>,
{
    let width = RecursiveAir::width(air);

    let mut trace_local: Vec<F> = (0..width).map(|_| rng.random()).collect();
    let mut trace_next: Vec<F> = (0..width).map(|_| rng.random()).collect();
    let mut preprocessed_local: Vec<F> = (0..preprocessed_width).map(|_| rng.random()).collect();
    let mut preprocessed_next: Vec<F> = (0..preprocessed_width).map(|_| rng.random()).collect();
    let public_values: Vec<F> = (0..num_public_values)
        .map(|_| F::from_u32(rng.next_u32()))
        .collect();
    let selectors: [F; 3] = [rng.random(), rng.random(), rng.random()];
    let alpha: F = rng.random();

    let main = VerticalPair::new(
        RowMajorMatrixView::new_row(&trace_local),
        RowMajorMatrixView::new_row(&trace_next),
    );
    let preprocessed = if preprocessed_width > 0 {
        VerticalPair::new(
            RowMajorMatrixView::new_row(&preprocessed_local),
            RowMajorMatrixView::new_row(&preprocessed_next),
        )
    } else {
        VerticalPair::new(
            RowMajorMatrixView::new(&[], 0),
            RowMajorMatrixView::new(&[], 0),
        )
    };
    let preprocessed_window =
        RowWindow::from_two_rows(preprocessed.top.values, preprocessed.bottom.values);
    let mut folder: VerifierConstraintFolder<'_, MyConfig> = VerifierConstraintFolder {
        main,
        preprocessed,
        preprocessed_window,
        periodic_values: &[],
        public_values: &public_values,
        is_first_row: selectors[0],
        is_last_row: selectors[1],
        is_transition: selectors[2],
        alpha,
        accumulator: F::ZERO,
    };
    air.eval(&mut folder);
    let folded_value = folder.accumulator;

    let mut builder = CircuitBuilder::<F>::new();
    let selector_targets = [
        builder.public_input(),
        builder.public_input(),
        builder.public_input(),
    ];
    let public_targets: Vec<_> = (0..num_public_values)
        .map(|_| builder.public_input())
        .collect();

    let pre_local_targets: Vec<_> = (0..preprocessed_width)
        .map(|_| builder.public_input())
        .collect();
    let pre_next_targets: Vec<_> = (0..preprocessed_width)
        .map(|_| builder.public_input())
        .collect();
    let local_targets: Vec<_> = (0..width).map(|_| builder.public_input()).collect();
    let next_targets: Vec<_> = (0..width).map(|_| builder.public_input()).collect();

    let row_selectors = RowSelectorsTargets {
        is_first_row: selector_targets[0],
        is_last_row: selector_targets[1],
        is_transition: selector_targets[2],
    };

    let columns = ColumnsTargets {
        challenges: &[],
        public_values: &public_targets,
        permutation_local_values: &[],
        permutation_next_values: &[],
        permutation_values: &[],
        local_prep_values: &pre_local_targets,
        next_prep_values: &pre_next_targets,
        local_values: &local_targets,
        next_values: &next_targets,
    };

    let alpha_t = builder.define_const(alpha);
    let sels = RecursiveLagrangeSelectors {
        row_selectors,
        inv_vanishing: builder.define_const(F::ONE),
    };
    let lookup_gadget = LogUpGadget {};
    let dummy_lookup_metadata = LookupMetadata {
        contexts: &[],
        lookup_data: &[],
    };
    let sum = air.eval_folded_circuit(
        &mut builder,
        &sels,
        &alpha_t,
        &dummy_lookup_metadata,
        columns,
        &lookup_gadget,
    );
    let const_target = builder.define_const(folded_value);
    builder.connect(const_target, sum);

    let mut all_public_inputs = Vec::new();
    all_public_inputs.extend_from_slice(&selectors);
    all_public_inputs.extend_from_slice(&public_values);
    all_public_inputs.append(&mut preprocessed_local);
    all_public_inputs.append(&mut preprocessed_next);
    all_public_inputs.append(&mut trace_local);
    all_public_inputs.append(&mut trace_next);

    let circuit = builder.build()?;
    let mut runner = circuit.runner();
    runner.set_public_inputs(&all_public_inputs)?;
    runner.run()?;

    Ok(())
}

#[test]
fn primitive_airs_symbolic_to_circuit() -> Result<(), CircuitError> {
    let mut rng = SmallRng::seed_from_u64(7);

    // AluAir preprocessed format (per op, multiplicity added by AIR):
    // [sel_add_vs_mul, sel_bool, sel_muladd, sel_horner, a_idx, b_idx, c_idx, out_idx]
    let alu_add_prep = vec![
        F::ONE,  // sel_add_vs_mul
        F::ZERO, // sel_bool
        F::ZERO, // sel_muladd
        F::ZERO, // sel_horner
        F::ZERO,
        F::ONE,
        F::ZERO,
        F::from_u64(2), // a_idx, b_idx, c_idx, out_idx
    ];
    let alu_air = AluAir::<F, 1>::new_with_preprocessed(1, 1, alu_add_prep, 2);
    run_recursive(&alu_air, alu_air.preprocessed_width(), 0, &mut rng)?;

    let const_air = ConstAir::<F, 1>::new_with_preprocessed(1, vec![F::from_u64(3)]);
    run_recursive(
        &const_air,
        ConstAir::<F, 1>::preprocessed_width(),
        0,
        &mut rng,
    )?;

    let public_air = PublicAir::<F, 1>::new_with_preprocessed(1, 1, vec![F::from_u64(4)]);
    run_recursive(&public_air, public_air.preprocessed_width(), 1, &mut rng)?;

    Ok(())
}

#[test]
fn poseidon2_perm_air_symbolic_to_circuit() -> Result<(), CircuitError> {
    let mut rng = SmallRng::seed_from_u64(9);

    let constants = RoundConstants::new(rng.random(), rng.random(), rng.random());
    let preprocessed_width = Poseidon2CircuitAirBabyBearD4Width16::preprocessed_width();
    let preprocessed_values: Vec<F> = (0..preprocessed_width).map(|_| rng.random()).collect();
    let air =
        Poseidon2CircuitAirBabyBearD4Width16::new_with_preprocessed(constants, preprocessed_values);

    run_recursive(&air, preprocessed_width, 0, &mut rng)
}
