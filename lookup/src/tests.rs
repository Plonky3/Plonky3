use alloc::vec;
use alloc::vec::Vec;

use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::gadgets::{GadgetConstraintContext, InteractionGadget, LogUpGadget};
use crate::{Interaction, eval_symbolic};

/// Base field type for the test
type F = BabyBear;
/// Extension field type for the test
type EF = BinomialExtensionField<F, 4>;

/// Test-specific interaction kinds

fn create_symbolic_with_degree(degree: usize) -> SymbolicExpression<F> {
    use alloc::rc::Rc;
    let x = Rc::new(SymbolicExpression::Constant(F::ONE));
    let y = Rc::new(SymbolicExpression::Constant(F::TWO));
    SymbolicExpression::Mul {
        x,
        y,
        degree_multiple: degree,
    }
}

fn create_dummy_interactions(
    num_elements_per_tuple: Vec<usize>,
    degree_per_element: Vec<Vec<usize>>,
    degree_multiplicities: Vec<usize>,
) -> Vec<Interaction<F>> {
    assert_eq!(num_elements_per_tuple.len(), degree_per_element.len());
    assert_eq!(num_elements_per_tuple.len(), degree_multiplicities.len());

    num_elements_per_tuple
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            assert_eq!(num_elements_per_tuple[i], degree_per_element[i].len());
            let values = (0..n)
                .map(|j| create_symbolic_with_degree(degree_per_element[i][j]))
                .collect();
            let multiplicity = create_symbolic_with_degree(degree_multiplicities[i]);

            Interaction {
                values,
                multiplicity,
            }
        })
        .collect()
}

#[test]
fn test_constraint_degree_calculation() {
    let gadget = LogUpGadget;

    let interactions = create_dummy_interactions(vec![1, 1], vec![vec![1], vec![1]], vec![1, 1]);
    assert_eq!(gadget.constraint_degree(&interactions), 3);

    let interactions = create_dummy_interactions(vec![1, 1], vec![vec![1], vec![1]], vec![3, 3]);
    assert_eq!(gadget.constraint_degree(&interactions), 4);

    let interactions = create_dummy_interactions(vec![1, 1], vec![vec![2], vec![3]], vec![1, 1]);
    assert_eq!(gadget.constraint_degree(&interactions), 6);

    let degrees1 = vec![1, 3, 0, 0, 0];
    let degrees2 = vec![0, 1, 0, 2, 0, 1];
    let interactions = create_dummy_interactions(vec![5, 6], vec![degrees1, degrees2], vec![5, 2]);
    assert_eq!(gadget.constraint_degree(&interactions), 7);
}

/// A mock `AirBuilder` for testing purposes that simulates constraint evaluation.
struct MockAirBuilder {
    main: RowMajorMatrix<F>,
    permutation: RowMajorMatrix<EF>,
    challenges: Vec<EF>,
    current_row: usize,
    height: usize,
}

impl MockAirBuilder {
    fn new(main: RowMajorMatrix<F>, permutation: RowMajorMatrix<EF>, challenges: Vec<EF>) -> Self {
        let height = main.height();
        Self {
            main,
            permutation,
            challenges,
            current_row: 0,
            height,
        }
    }

    fn for_row(&mut self, row: usize) {
        self.current_row = row;
    }

    fn window<T: Clone + Send + Sync + Field>(
        &self,
        trace: &RowMajorMatrix<T>,
    ) -> RowMajorMatrix<T> {
        let mut view = Vec::new();
        let local_row: Vec<T> = trace.row(self.current_row).unwrap().into_iter().collect();
        view.extend_from_slice(&local_row);
        if self.current_row + 1 < self.height {
            let next_row: Vec<T> = trace
                .row(self.current_row + 1)
                .unwrap()
                .into_iter()
                .collect();
            view.extend_from_slice(&next_row);
        } else {
            view.extend(vec![T::ZERO; trace.width()]);
        }
        RowMajorMatrix::new(view, trace.width())
    }
}

impl AirBuilder for MockAirBuilder {
    type F = F;
    type Expr = F;
    type Var = F;
    type M = RowMajorMatrix<F>;

    fn main(&self) -> Self::M {
        self.window(&self.main)
    }

    fn is_first_row(&self) -> Self::Expr {
        F::from_bool(self.current_row == 0)
    }

    fn is_last_row(&self) -> Self::Expr {
        F::from_bool(self.current_row == self.height - 1)
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert!(size > 0);
        F::from_bool(self.current_row < self.height - (size - 1))
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let val = x.into();
        assert_eq!(
            val,
            F::ZERO,
            "Constraint failed at row {}: {:?} != 0",
            self.current_row,
            val
        );
    }
}

impl ExtensionBuilder for MockAirBuilder {
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let val = x.into();
        if val != EF::ZERO {
            panic!(
                "Extension constraint failed at row {}: {:?} != 0",
                self.current_row, val
            );
        }
    }
}

impl PermutationAirBuilder for MockAirBuilder {
    type MP = RowMajorMatrix<EF>;
    type RandomVar = EF;

    fn permutation(&self) -> Self::MP {
        self.window(&self.permutation)
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.challenges
    }
}

impl PairBuilder for MockAirBuilder {
    fn preprocessed(&self) -> Self::M {
        RowMajorMatrix::new(vec![], 0)
    }
}

impl AirBuilderWithPublicValues for MockAirBuilder {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::PublicVar] {
        &[]
    }
}

/// Computes the contribution to the LogUp running sum for a single row.
fn compute_logup_contribution(
    challenges: LogUpChallenges,
    vals_read: Vec<F>,
    vals_provided: Vec<F>,
    mult: F,
) -> EF {
    let alpha = challenges.alpha;
    let beta = challenges.beta;
    let vals_read_len = vals_read.len();
    let val_read = vals_read
        .iter()
        .fold(EF::ZERO, |acc, &v| acc * beta + EF::from(v));
    let val_provided = vals_provided
        .iter()
        .fold(EF::ZERO, |acc, &v| acc * beta + EF::from(v));

    if vals_read_len == 0 {
        (alpha - EF::from(val_provided)).inverse() * EF::from(mult)
    } else {
        (alpha - EF::from(val_read)).inverse()
            - (alpha - EF::from(val_provided)).inverse() * EF::from(mult)
    }
}

#[derive(Copy, Clone)]
struct LogUpChallenges {
    alpha: EF,
    beta: EF,
}

impl LogUpChallenges {
    fn to_vec(self) -> Vec<EF> {
        vec![self.alpha, self.beta]
    }
}

/// Helper to build interactions for range check pattern
fn build_range_check_interactions(main_width: usize) -> Vec<Interaction<F>> {
    let symbolic_builder = SymbolicAirBuilder::<F>::new(0, main_width, 0);
    let symbolic_main = symbolic_builder.main();
    let symbolic_local = symbolic_main.row_slice(0).unwrap();

    let mut interactions = Vec::new();

    // For each lookup in the trace (every 3 columns: [read, provide, mult])
    for lookup_idx in 0..(main_width / 3) {
        let offset = lookup_idx * 3;
        let val = symbolic_local[offset];
        let table_val = symbolic_local[offset + 1];
        let mult = symbolic_local[offset + 2];

        // Receive interaction (read from table, positive multiplicity)
        interactions.push(Interaction {
            values: vec![val.into()],
            multiplicity: SymbolicExpression::Constant(F::ONE),
        });

        // Send interaction (provide to table, negative multiplicity)
        let mult_expr: SymbolicExpression<F> = mult.into();
        interactions.push(Interaction {
            values: vec![table_val.into()],
            multiplicity: -mult_expr,
        });
    }

    interactions
}

#[test]
fn test_eval_symbolic() {
    let mut builder = SymbolicAirBuilder::<F>::new(0, 2, 0);
    let main = builder.main();
    let (local, next) = (main.row_slice(0).unwrap(), main.row_slice(1).unwrap());

    let mul = local[0] * next[1];
    let add = local[0] + next[1];
    let sub = local[0] - next[1];
    builder.when_first_row().assert_zero(mul.clone() * add);
    builder.when_transition().assert_zero(sub - local[0]);
    builder.when_last_row().assert_zero(mul - local[0]);

    let constraints = builder.constraints();

    let mut main_flat = Vec::new();
    main_flat.extend([F::new(10), F::new(10)]);
    main_flat.extend([F::new(256), F::new(255)]);
    main_flat.extend([F::new(42), F::new(42)]);
    let main_trace = RowMajorMatrix::new(main_flat, 2);
    let perm = RowMajorMatrix::new(vec![], 0);
    let mut builder = MockAirBuilder::new(main_trace.clone(), perm, vec![]);

    for i in 0..builder.height {
        builder.for_row(i);
        let is_first_row = if i == 0 { F::ONE } else { F::ZERO };
        let is_last_row = if i == builder.height - 1 {
            F::ONE
        } else {
            F::ZERO
        };
        let is_transition = if i < builder.height - 1 {
            F::ONE
        } else {
            F::ZERO
        };

        let cloned_trace = main_trace.clone();
        let local = cloned_trace.row(i).unwrap().into_iter().collect::<Vec<F>>();
        let next = cloned_trace.row(i + 1).map_or_else(
            || vec![F::ZERO; 2],
            |row| row.into_iter().collect::<Vec<F>>(),
        );

        let mul = local[0] * next[1];
        let add = local[0] + next[1];
        let sub = local[0] - next[1];

        let first_expected_val = is_first_row * (mul * add);
        let transition_expected_val = is_transition * (sub - local[0]);
        let last_expected_val = is_last_row * (mul - local[0]);

        let first_eval = eval_symbolic(&builder, &constraints[0].clone());
        let transition_eval = eval_symbolic(&builder, &constraints[1].clone());
        let last_eval = eval_symbolic(&builder, &constraints[2].clone());

        assert_eq!(first_eval, first_expected_val);
        assert_eq!(transition_eval, transition_expected_val);
        assert_eq!(last_eval, last_expected_val);
    }
}

/// Build main and aux traces for simple range check pattern
fn build_range_check_trace(
    rows: Vec<(Vec<u32>, Vec<u32>, u32)>,
    challenges: LogUpChallenges,
    num_lookups: usize,
) -> (RowMajorMatrix<F>, RowMajorMatrix<EF>) {
    let width = 3 * num_lookups;
    let main_flat: Vec<F> = rows
        .iter()
        .flat_map(|(read, provide, mult)| {
            let mut row = Vec::new();
            for _ in 0..num_lookups {
                row.extend(read.iter().map(|&v| F::new(v)));
                row.extend(provide.iter().map(|&v| F::new(v)));
                row.push(F::new(*mult));
            }
            row
        })
        .collect();
    let main_trace = RowMajorMatrix::new(main_flat, width);

    let mut running_sum = EF::ZERO;
    let s_col: Vec<EF> = core::iter::once(EF::ZERO)
        .chain(rows.iter().map(|(read, provide, mult)| {
            running_sum += compute_logup_contribution(
                challenges,
                read.iter().map(|&v| F::new(v)).collect(),
                provide.iter().map(|&v| F::new(v)).collect(),
                F::new(*mult),
            );
            running_sum
        }))
        .take(rows.len())
        .collect();

    let aux_trace = RowMajorMatrix::new(s_col, num_lookups);
    (main_trace, aux_trace)
}

#[test]
fn test_range_check_end_to_end_valid() {
    let mut rng = SmallRng::seed_from_u64(1);
    let challenges = LogUpChallenges {
        alpha: EF::from_u32(rng.random()),
        beta: EF::from_u32(rng.random()),
    };

    let rows = vec![
        (vec![10], vec![10], 1),
        (vec![255], vec![255], 1),
        (vec![0], vec![0], 1),
        (vec![42], vec![42], 1),
        (vec![10], vec![10], 1),
    ];
    let (main_trace, aux_trace) = build_range_check_trace(rows, challenges, 1);

    let s_final = aux_trace
        .row(aux_trace.height() - 1)
        .unwrap()
        .into_iter()
        .collect::<Vec<EF>>()[0];
    let last_row_data = main_trace
        .row(main_trace.height() - 1)
        .unwrap()
        .into_iter()
        .collect::<Vec<F>>();
    let last_contribution = compute_logup_contribution(
        challenges,
        vec![last_row_data[0]],
        vec![last_row_data[1]],
        last_row_data[2],
    );

    assert_eq!(s_final + last_contribution, EF::ZERO);

    let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges.to_vec());
    let interactions = build_range_check_interactions(3);
    let gadget = LogUpGadget;

    for i in 0..builder.height {
        builder.for_row(i);
        gadget.eval_constraints(
            &mut builder,
            GadgetConstraintContext {
                interactions: &interactions,
                aux_column_index: 0,
                expected_cumulative_sum: None,
            },
        );
    }
}

#[test]
#[should_panic(expected = "Extension constraint failed")]
fn test_range_check_end_to_end_invalid() {
    let mut main_flat = Vec::new();
    main_flat.extend([F::new(10), F::new(10), F::ONE]);
    main_flat.extend([F::new(256), F::new(255), F::ONE]);
    main_flat.extend([F::new(42), F::new(42), F::ONE]);

    let main_trace = RowMajorMatrix::new(main_flat, 3);
    let challenges = LogUpChallenges {
        alpha: EF::from_u32(12345678),
        beta: EF::from_u32(87654321),
    };

    let mut s_col = Vec::with_capacity(main_trace.height());
    let mut current_s = EF::ZERO;
    for i in 0..main_trace.height() {
        let row: Vec<F> = main_trace.row(i).unwrap().into_iter().collect();
        let contribution =
            compute_logup_contribution(challenges, vec![row[0]], vec![row[1]], row[2]);
        current_s += contribution;
        s_col.push(current_s);
    }

    let aux_trace = RowMajorMatrix::new(s_col, 1);
    let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges.to_vec());
    let interactions = build_range_check_interactions(3);
    let gadget = LogUpGadget;

    for i in 0..builder.height {
        builder.for_row(i);
        gadget.eval_constraints(
            &mut builder,
            GadgetConstraintContext {
                interactions: &interactions,
                aux_column_index: 0,
                expected_cumulative_sum: None,
            },
        );
    }
}

#[test]
fn test_compute_sum_terms_logic() {
    let alpha = EF::from_u8(100);
    let beta = EF::from_u8(100);

    let elements = [vec![F::new(2)], vec![F::new(5)]];
    let multiplicities = [F::new(3), F::ONE];

    let expected_numerator = (alpha - EF::from_u8(5)) * EF::from_u8(3) + (alpha - EF::from_u8(2));
    let expected_denominator = (alpha - EF::from_u8(2)) * (alpha - EF::from_u8(5));

    let (num, den) = crate::gadgets::logup::compute_combined_sum_terms::<MockAirBuilder, F, F>(
        &elements,
        &multiplicities,
        &alpha,
        &beta,
    );

    assert_eq!(num, expected_numerator);
    assert_eq!(den, expected_denominator);
}

#[test]
#[should_panic(expected = "Extension constraint failed at row 1")]
fn test_inconsistent_witness_fails_transition() {
    let mut rng = SmallRng::seed_from_u64(1);
    let challenges = LogUpChallenges {
        alpha: EF::from_u32(rng.random()),
        beta: EF::from_u32(rng.random()),
    };

    let rows = vec![
        (vec![10], vec![10], 1),
        (vec![20], vec![20], 1),
        (vec![30], vec![30], 1),
    ];
    let (main_trace, mut aux_trace) = build_range_check_trace(rows, challenges, 1);

    let corrupted_s_val = EF::from_u8(99);
    aux_trace.values[2] = corrupted_s_val;

    let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges.to_vec());
    let interactions = build_range_check_interactions(3);
    let gadget = LogUpGadget;

    for i in 0..builder.height {
        builder.for_row(i);
        gadget.eval_constraints(
            &mut builder,
            GadgetConstraintContext {
                interactions: &interactions,
                aux_column_index: 0,
                expected_cumulative_sum: None,
            },
        );
    }
}

#[test]
#[should_panic(expected = "Extension constraint failed at row 0")]
fn test_zero_multiplicity_is_not_counted() {
    let mut main_flat = Vec::new();
    main_flat.extend([F::new(10), F::new(10), F::ZERO]);
    main_flat.extend([F::new(20), F::new(20), F::ONE]);

    let main_trace = RowMajorMatrix::new(main_flat, 3);
    let challenges = LogUpChallenges {
        alpha: EF::from_u8(123),
        beta: EF::from_u8(111),
    };

    let mut s_col = Vec::with_capacity(main_trace.height());
    let mut current_s = EF::ZERO;
    for i in 0..main_trace.height() {
        let row: Vec<F> = main_trace.row(i).unwrap().into_iter().collect();
        let contribution =
            compute_logup_contribution(challenges, vec![row[0]], vec![row[1]], row[2]);
        current_s += contribution;
        s_col.push(current_s);
    }
    let aux_trace = RowMajorMatrix::new(s_col, 1);

    let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges.to_vec());
    let interactions = build_range_check_interactions(3);
    let gadget = LogUpGadget;

    for i in 0..builder.height {
        builder.for_row(i);
        gadget.eval_constraints(
            &mut builder,
            GadgetConstraintContext {
                interactions: &interactions,
                aux_column_index: 0,
                expected_cumulative_sum: None,
            },
        );
    }
}

#[test]
fn test_empty_lookup_is_valid() {
    let main_trace = RowMajorMatrix::new(vec![], 3);
    let aux_trace = RowMajorMatrix::new(vec![], 1);
    let alpha = EF::from_u8(123);

    let mut builder = MockAirBuilder::new(main_trace, aux_trace, vec![alpha]);
    let interactions = build_range_check_interactions(3);
    let gadget = LogUpGadget;

    for i in 0..builder.height {
        builder.for_row(i);
        gadget.eval_constraints(
            &mut builder,
            GadgetConstraintContext {
                interactions: &interactions,
                aux_column_index: 0,
                expected_cumulative_sum: None,
            },
        );
    }

    let (num, den) = crate::gadgets::logup::compute_combined_sum_terms::<MockAirBuilder, F, F>(
        &[],
        &[],
        &alpha,
        &alpha,
    );
    assert_eq!(num, EF::ZERO);
    assert_eq!(den, EF::ONE);
}

#[test]
fn test_nontrivial_permutation() {
    let mut rng = SmallRng::seed_from_u64(1);
    let challenges = LogUpChallenges {
        alpha: EF::from_u32(rng.random()),
        beta: EF::from_u32(rng.random()),
    };

    let rows = vec![
        (vec![7], vec![3], 2),
        (vec![3], vec![5], 4),
        (vec![5], vec![7], 2),
        (vec![3], vec![3], 0),
        (vec![7], vec![5], 0),
        (vec![5], vec![5], 0),
        (vec![5], vec![7], 0),
        (vec![5], vec![5], 0),
    ];
    let (main_trace, aux_trace) = build_range_check_trace(rows, challenges, 1);

    let s_final = aux_trace
        .row(aux_trace.height() - 1)
        .unwrap()
        .into_iter()
        .collect::<Vec<EF>>()[0];
    let last_row_data = main_trace
        .row(main_trace.height() - 1)
        .unwrap()
        .into_iter()
        .collect::<Vec<F>>();
    let last_contribution = compute_logup_contribution(
        challenges,
        vec![last_row_data[0]],
        vec![last_row_data[1]],
        last_row_data[2],
    );

    assert_eq!(s_final + last_contribution, EF::ZERO);

    let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges.to_vec());
    let interactions = build_range_check_interactions(3);
    let gadget = LogUpGadget;

    for i in 0..builder.height {
        builder.for_row(i);
        gadget.eval_constraints(
            &mut builder,
            GadgetConstraintContext {
                interactions: &interactions,
                aux_column_index: 0,
                expected_cumulative_sum: None,
            },
        );
    }
}

#[test]
fn test_multiple_lookups_different_columns() {
    // Two independent lookups using different columns
    let first_challenges = LogUpChallenges {
        alpha: EF::from_u32(0x12345678),
        beta: EF::from_u32(0x87654321),
    };
    let second_challenges = LogUpChallenges {
        alpha: EF::from_u32(0xABCDEF01),
        beta: EF::from_u32(0x10FEDCBA),
    };

    // Build main trace with 6 columns: [read1, provide1, mult1, read2, provide2, mult2]
    let main_flat = vec![
        // Row 0
        F::new(10),
        F::new(10),
        F::ONE, // Lookup 1
        F::new(5),
        F::new(5),
        F::ONE, // Lookup 2
        // Row 1
        F::new(20),
        F::new(20),
        F::ONE, // Lookup 1
        F::new(15),
        F::new(15),
        F::ONE, // Lookup 2
        // Row 2
        F::new(30),
        F::new(30),
        F::ONE, // Lookup 1
        F::new(25),
        F::new(25),
        F::ONE, // Lookup 2
    ];
    let main_trace = RowMajorMatrix::new(main_flat, 6);

    // Build auxiliary trace with 2 columns
    let mut s1 = EF::ZERO;
    let mut s2 = EF::ZERO;
    let mut aux_flat = vec![s1, s2];

    // Row 1
    s1 += compute_logup_contribution(first_challenges, vec![F::new(10)], vec![F::new(10)], F::ONE);
    s2 += compute_logup_contribution(second_challenges, vec![F::new(5)], vec![F::new(5)], F::ONE);
    aux_flat.push(s1);
    aux_flat.push(s2);

    // Row 2
    s1 += compute_logup_contribution(first_challenges, vec![F::new(20)], vec![F::new(20)], F::ONE);
    s2 += compute_logup_contribution(
        second_challenges,
        vec![F::new(15)],
        vec![F::new(15)],
        F::ONE,
    );
    aux_flat.push(s1);
    aux_flat.push(s2);

    let aux_trace = RowMajorMatrix::new(aux_flat, 2);

    let mut builder = MockAirBuilder::new(
        main_trace,
        aux_trace,
        [first_challenges.to_vec(), second_challenges.to_vec()].concat(),
    );

    // Create interactions for both lookups
    let interactions1 = build_range_check_interactions(6);
    let lookup1_interactions = vec![interactions1[0].clone(), interactions1[1].clone()];
    let lookup2_interactions = vec![interactions1[2].clone(), interactions1[3].clone()];

    let gadget = LogUpGadget;

    for i in 0..builder.height {
        builder.for_row(i);
        gadget.eval_constraints(
            &mut builder,
            GadgetConstraintContext {
                interactions: &lookup1_interactions,
                aux_column_index: 0,
                expected_cumulative_sum: None,
            },
        );
        gadget.eval_constraints(
            &mut builder,
            GadgetConstraintContext {
                interactions: &lookup2_interactions,
                aux_column_index: 1,
                expected_cumulative_sum: None,
            },
        );
    }
}

#[test]
fn test_tuple_lookup() {
    let mut rng = SmallRng::seed_from_u64(1);
    let challenges = LogUpChallenges {
        alpha: EF::from_u32(rng.random()),
        beta: EF::from_u32(rng.random()),
    };

    let rows = [
        (vec![0, 1, 1], vec![0, 1, 1], 2),
        (vec![0, 1, 1], vec![0, 0, 0], 1),
        (vec![1, 1, 2], vec![1, 0, 1], 0),
        (vec![0, 0, 0], vec![1, 1, 2], 1),
    ];

    let main_flat: Vec<F> = rows
        .iter()
        .flat_map(|(read, provide, mult)| {
            let mut row = Vec::new();
            row.extend(read.iter().map(|&v| F::new(v)));
            row.extend(provide.iter().map(|&v| F::new(v)));
            row.push(F::new(*mult));
            row
        })
        .collect();
    let main_trace = RowMajorMatrix::new(main_flat, 7);

    let mut running_sum = EF::ZERO;
    let s_col: Vec<EF> = core::iter::once(EF::ZERO)
        .chain(rows.iter().map(|(read, provide, mult)| {
            running_sum += compute_logup_contribution(
                challenges,
                read.iter().map(|&v| F::new(v)).collect(),
                provide.iter().map(|&v| F::new(v)).collect(),
                F::new(*mult),
            );
            running_sum
        }))
        .take(rows.len())
        .collect();
    let aux_trace = RowMajorMatrix::new(s_col, 1);

    let symbolic_builder = SymbolicAirBuilder::<F>::new(0, 7, 0);
    let symbolic_main = symbolic_builder.main();
    let symbolic_local = symbolic_main.row_slice(0).unwrap();

    let mult_expr: SymbolicExpression<F> = symbolic_local[6].into();
    let interactions = vec![
        Interaction {
            values: vec![
                symbolic_local[0].into(),
                symbolic_local[1].into(),
                symbolic_local[2].into(),
            ],
            multiplicity: SymbolicExpression::Constant(F::ONE),
        },
        Interaction {
            values: vec![
                symbolic_local[3].into(),
                symbolic_local[4].into(),
                symbolic_local[5].into(),
            ],
            multiplicity: -mult_expr,
        },
    ];

    let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges.to_vec());
    let gadget = LogUpGadget;

    for i in 0..builder.height {
        builder.for_row(i);
        gadget.eval_constraints(
            &mut builder,
            GadgetConstraintContext {
                interactions: &interactions,
                aux_column_index: 0,
                expected_cumulative_sum: None,
            },
        );
    }
}

#[test]
fn test_global_lookup() {
    let mut rng = SmallRng::seed_from_u64(1);
    let global_challenges = LogUpChallenges {
        alpha: EF::from_u32(rng.random()),
        beta: EF::from_u32(rng.random()),
    };
    let challenges1 = LogUpChallenges {
        alpha: EF::from_u32(rng.random()),
        beta: EF::from_u32(rng.random()),
    };
    let challenges2 = LogUpChallenges {
        alpha: EF::from_u32(rng.random()),
        beta: EF::from_u32(rng.random()),
    };

    let rows1 = vec![
        (vec![0, 1, 1], vec![0, 0, 0], 1),
        (vec![0, 1, 1], vec![0, 1, 1], 2),
        (vec![1, 1, 2], vec![1, 1, 2], 1),
        (vec![0, 0, 0], vec![1, 0, 1], 0),
    ];
    let rows2 = vec![
        (vec![0, 1, 1], vec![0, 1, 1], 2),
        (vec![0, 1, 1], vec![0, 0, 0], 1),
        (vec![1, 1, 2], vec![1, 0, 1], 0),
        (vec![0, 0, 0], vec![1, 1, 2], 1),
    ];

    // Build traces with 2 aux columns each (local + global)
    let build_trace = |rows: &[(Vec<u32>, Vec<u32>, u32)], local_challenges, is_send: bool| {
        let main_flat: Vec<F> = rows
            .iter()
            .flat_map(|(read, provide, mult)| [read.clone(), provide.clone(), vec![*mult]].concat())
            .map(F::new)
            .collect();
        let main_trace = RowMajorMatrix::new(main_flat, 7);

        let mut local_sum = EF::ZERO;
        let mut global_sum = EF::ZERO;
        let mut aux_flat = vec![local_sum, global_sum];

        for (read, provide, mult) in rows {
            local_sum += compute_logup_contribution(
                local_challenges,
                read.iter().map(|&v| F::new(v)).collect(),
                provide.iter().map(|&v| F::new(v)).collect(),
                F::new(*mult),
            );
            let global_mult = if is_send { -F::ONE } else { F::ONE };
            global_sum += compute_logup_contribution(
                global_challenges,
                vec![],
                provide.iter().map(|&v| F::new(v)).collect(),
                global_mult,
            );
            aux_flat.push(local_sum);
            aux_flat.push(global_sum);
        }
        aux_flat.truncate(8);
        let aux_trace = RowMajorMatrix::new(aux_flat, 2);
        (main_trace, aux_trace)
    };

    let (main_trace1, aux_trace1) = build_trace(&rows1, challenges1, false);
    let (main_trace2, aux_trace2) = build_trace(&rows2, challenges2, true);

    let s_global_final1 = aux_trace1
        .row(aux_trace1.height() - 1)
        .unwrap()
        .into_iter()
        .collect::<Vec<EF>>()[1]
        + compute_logup_contribution(
            global_challenges,
            vec![],
            vec![F::new(1), F::new(0), F::new(1)],
            F::ONE,
        );
    let s_global_final2 = aux_trace2
        .row(aux_trace2.height() - 1)
        .unwrap()
        .into_iter()
        .collect::<Vec<EF>>()[1]
        + compute_logup_contribution(
            global_challenges,
            vec![],
            vec![F::new(1), F::new(1), F::new(2)],
            -F::ONE,
        );

    let symbolic_builder = SymbolicAirBuilder::<F>::new(0, 7, 0);
    let symbolic_main = symbolic_builder.main();
    let symbolic_local = symbolic_main.row_slice(0).unwrap();

    let mult_expr: SymbolicExpression<F> = symbolic_local[6].into();
    let local_interactions = vec![
        Interaction {
            values: vec![
                symbolic_local[0].into(),
                symbolic_local[1].into(),
                symbolic_local[2].into(),
            ],
            multiplicity: SymbolicExpression::Constant(F::ONE),
        },
        Interaction {
            values: vec![
                symbolic_local[3].into(),
                symbolic_local[4].into(),
                symbolic_local[5].into(),
            ],
            multiplicity: -mult_expr,
        },
    ];

    let global_interaction_receive = vec![Interaction {
        values: vec![
            symbolic_local[3].into(),
            symbolic_local[4].into(),
            symbolic_local[5].into(),
        ],
        multiplicity: SymbolicExpression::Constant(F::ONE),
    }];

    let global_interaction_send = vec![Interaction {
        values: vec![
            symbolic_local[3].into(),
            symbolic_local[4].into(),
            symbolic_local[5].into(),
        ],
        multiplicity: -SymbolicExpression::Constant(F::ONE),
    }];

    let mut builder1 = MockAirBuilder::new(
        main_trace1,
        aux_trace1,
        [challenges1.to_vec(), global_challenges.to_vec()].concat(),
    );
    let mut builder2 = MockAirBuilder::new(
        main_trace2,
        aux_trace2,
        [challenges2.to_vec(), global_challenges.to_vec()].concat(),
    );

    let gadget = LogUpGadget;

    for i in 0..builder1.height {
        builder1.for_row(i);
        gadget.eval_constraints(
            &mut builder1,
            GadgetConstraintContext {
                interactions: &local_interactions,
                aux_column_index: 0,
                expected_cumulative_sum: None,
            },
        );
        gadget.eval_constraints(
            &mut builder1,
            GadgetConstraintContext {
                interactions: &global_interaction_receive,
                aux_column_index: 1,
                expected_cumulative_sum: Some(s_global_final1),
            },
        );

        builder2.for_row(i);
        gadget.eval_constraints(
            &mut builder2,
            GadgetConstraintContext {
                interactions: &local_interactions,
                aux_column_index: 0,
                expected_cumulative_sum: None,
            },
        );
        gadget.eval_constraints(
            &mut builder2,
            GadgetConstraintContext {
                interactions: &global_interaction_send,
                aux_column_index: 1,
                expected_cumulative_sum: Some(s_global_final2),
            },
        );
    }

    gadget
        .verify_global_final_value(&[s_global_final1, s_global_final2])
        .expect("Global lookups final values should sum to 0.");
}
