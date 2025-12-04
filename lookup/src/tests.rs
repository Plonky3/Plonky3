use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, ExtensionBuilder, PairBuilder,
    PermutationAirBuilder,
};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::logup::LogUpGadget;
use crate::lookup_traits::{
    AirLookupHandler, Direction, Kind, Lookup, LookupGadget, symbolic_to_expr,
};

/// Base field type for the test
type F = BabyBear;
/// Extension field type for the test
type EF = BinomialExtensionField<F, 4>;

fn create_symbolic_with_degree(degree: usize) -> SymbolicExpression<F> {
    let x = Arc::new(SymbolicExpression::Constant(F::ONE));
    let y = Arc::new(SymbolicExpression::Constant(F::TWO));
    SymbolicExpression::Mul {
        x,
        y,
        degree_multiple: degree,
    }
}

fn create_dummy_lookup(
    num_elements_per_tuple: &[usize],
    degree_per_element: &[Vec<usize>],
    degree_multiplicities: &[usize],
) -> Lookup<F> {
    assert!(num_elements_per_tuple.len() == degree_per_element.len());
    assert!(num_elements_per_tuple.len() == degree_multiplicities.len());

    let element_exprs = num_elements_per_tuple
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            assert_eq!(num_elements_per_tuple[i], degree_per_element[i].len());
            (0..n)
                .map(|j| create_symbolic_with_degree(degree_per_element[i][j]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let multiplicities_exprs = degree_multiplicities
        .iter()
        .map(|&deg| create_symbolic_with_degree(deg))
        .collect::<Vec<_>>();

    Lookup {
        kind: Kind::Local,
        element_exprs,
        multiplicities_exprs,
        columns: vec![0],
    }
}

#[test]
fn test_constraint_degree_calculation() {
    let gadget = LogUpGadget::new();

    // Test basic constraint degree calculation
    // We have two lookup elements (each element is a single column):
    // - each element has degree 1
    // - each multiplicity has degree 1
    // - so the total degree should be 3 (1 + (1 + 1)).
    let lookup_deg_3 = create_dummy_lookup(&[1, 1], &[vec![1], vec![1]], &[1, 1]);
    assert_eq!(gadget.constraint_degree(lookup_deg_3), 3);

    // We have two lookup elements (each element is a single column):
    // - each element has degree 1
    // - each multiplicity has degree 3
    // - so the total degree should be 4 (3 + 1).
    let lookup_degree_4 = create_dummy_lookup(&[1, 1], &[vec![1], vec![1]], &[3, 3]);
    assert_eq!(gadget.constraint_degree(lookup_degree_4), 4);

    // We have two lookup elements (each element is a single column):
    // - one element has degree 2
    // - one element has degree 3
    // - each multiplicity has degree 1
    // - so the total degree should be 6 (3 + 3).
    let lookup_degree_6 = create_dummy_lookup(&[1, 1], &[vec![2], vec![3]], &[1, 1]);
    assert_eq!(gadget.constraint_degree(lookup_degree_6), 6);

    // We have two lookup elements:
    // - first element is a tuple of 5 columns
    // - second element is a tuple of 6 columns
    // - first element has degree 2
    // - second element has degree 3
    // - first multiplicity has degree 5
    // - second multiplicity has degree 2
    // - so the total degree should be 7 (5 + 2).
    let degrees1 = vec![1, 3, 0, 0, 0]; // First element of degree 3.
    let degrees2 = vec![0, 1, 0, 2, 0, 1]; // Second element of degree 2.
    let lookup_degree_7 = create_dummy_lookup(&[5, 6], &[degrees1, degrees2], &[5, 2]);
    assert_eq!(gadget.constraint_degree(lookup_degree_7), 7);
}

/// A mock `AirBuilder` for testing purposes that simulates constraint evaluation.
struct MockAirBuilder {
    /// Main trace matrix containing the execution trace data
    main: RowMajorMatrix<F>,
    /// Auxiliary trace matrix containing the LogUp running sum column
    permutation: RowMajorMatrix<EF>,
    /// Random challenges used in the LogUp argument
    challenges: Vec<EF>,
    /// Current row being evaluated during constraint checking
    current_row: usize,
    /// Total height (number of rows) in the trace
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

    // Helper to update the builder to the current row being evaluated
    fn for_row(&mut self, row: usize) {
        self.current_row = row;
    }

    // A mock windowed view for the trace matrices
    fn window<T: Clone + Send + Sync + Field>(
        &self,
        trace: &RowMajorMatrix<T>,
    ) -> RowMajorMatrix<T> {
        let mut view = Vec::new();
        // local row
        let local_row: Vec<T> = trace.row(self.current_row).unwrap().into_iter().collect();
        view.extend_from_slice(&local_row);
        // next row (if it exists)
        if self.current_row + 1 < self.height {
            let next_row: Vec<T> = trace
                .row(self.current_row + 1)
                .unwrap()
                .into_iter()
                .collect();
            view.extend_from_slice(&next_row);
        } else {
            // pad with zeros if we are on the last row
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
/// An AIR designed to perform range checks using the `LogUpGadget`.
///
/// This AIR demonstrates how to use LogUp for range checking. It supports multiple
/// independent lookups by specifying how many lookups to perform.
///
/// For `num_lookups = 1`: Main trace has 3 columns [read, provide, mult]
/// For `num_lookups = 2`: Main trace has 6 columns [read1, provide1, mult1, read2, provide2, mult2]
struct RangeCheckAir {
    /// Number of independent lookups (default: 1)
    num_lookups: usize,
    /// Number of registered lookups (for AirLookupHandler)
    cur_num_lookups: usize,
}

impl RangeCheckAir {
    fn new() -> Self {
        Self {
            num_lookups: 1,
            cur_num_lookups: 0,
        }
    }

    fn with_multiple_lookups(num_lookups: usize) -> Self {
        Self {
            num_lookups,
            cur_num_lookups: 0,
        }
    }
}

impl<AB> Air<AB> for RangeCheckAir
where
    AB: PermutationAirBuilder<F = F, EF = EF, RandomVar = EF>
        + PairBuilder
        + AirBuilderWithPublicValues,
    AB::Var: Copy + Into<AB::ExprEF>,
    AB::ExprEF: From<AB::Var> + From<F>,
    F: Copy + Into<AB::ExprEF>,
{
    fn eval(&self, _builder: &mut AB) {
        // There are no constraints, only lookups for the range checks.
    }
}

impl<F: Field> BaseAir<F> for RangeCheckAir {
    fn width(&self) -> usize {
        3 * self.num_lookups // [read, provide, mult] per lookup
    }
}

impl<AB> AirLookupHandler<AB> for RangeCheckAir
where
    AB: PermutationAirBuilder<F = F, EF = EF, RandomVar = EF>
        + PairBuilder
        + AirBuilderWithPublicValues,
    AB::Var: Copy + Into<AB::ExprEF>,
    AB::ExprEF: From<AB::Var> + From<F>,
    F: Copy + Into<AB::ExprEF>,
{
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.cur_num_lookups;
        self.cur_num_lookups += 1;

        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<AB::F>> {
        let symbolic_air_builder =
            SymbolicAirBuilder::<F>::new(0, BaseAir::<AB::F>::width(self), 0, 0, 0);

        let symbolic_main = symbolic_air_builder.main();
        let symbolic_main_local = symbolic_main.row_slice(0).unwrap();

        // Perform each lookup independently using LookupContext
        (0..self.num_lookups)
            .map(|lookup_idx| {
                let offset = lookup_idx * 3;

                // Extract columns for this lookup: [read, provide, mult]
                let val = symbolic_main_local[offset];
                let table_val = symbolic_main_local[offset + 1];
                let mult = symbolic_main_local[offset + 2];

                // Create arrays with longer lifetime for the context
                let a_elements = vec![val.into()];
                let a_multiplicities = SymbolicExpression::Constant(F::ONE);

                let b_elements = vec![table_val.into()];
                let b_multiplicities = mult.into();

                let lookup_inputs = vec![
                    (a_elements, a_multiplicities, Direction::Receive),
                    (b_elements, b_multiplicities, Direction::Send),
                ];

                // Register the local lookup.
                AirLookupHandler::<AB>::register_lookup(self, Kind::Local, &lookup_inputs)
            })
            .collect::<Vec<_>>()
    }
}

/// Computes the contribution to the LogUp running sum for a single row.
///
/// The contribution is: `1/(α - val_read) - mult/(α - val_provided)`
fn compute_logup_contribution(
    challenges: LogUpChallenges,
    vals_read: &[F],
    vals_provided: &[F],
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
        // Then we're only computing the contribution for the provided value.
        (alpha - EF::from(val_provided)).inverse() * EF::from(mult)
    } else {
        (alpha - EF::from(val_read)).inverse()
            - (alpha - EF::from(val_provided)).inverse() * EF::from(mult)
    }
}

#[derive(Copy, Clone)]
struct LogUpChallenges {
    // Random challenge for computing `alpha - val` in LogUp
    alpha: EF,
    // Random challenge for combining tuple elements (not used in single-column tests)
    beta: EF,
}

impl LogUpChallenges {
    fn to_vec(self) -> Vec<EF> {
        vec![self.alpha, self.beta]
    }
}

/// A simple builder for constructing LogUp lookup traces with arbitrary read/provide patterns.
///
/// This makes it easy to create complex test scenarios with non-trivial permutations
/// and varied multiplicities.
///
/// # Example
/// ```ignore
/// let trace = LookupTraceBuilder::new()
///     .row(3, 1, 1)  // Read 3, Provide 1 with multiplicity 1
///     .row(1, 2, 2)  // Read 1, Provide 2 with multiplicity 2
///     .row(2, 3, 1)  // Read 2, Provide 3 with multiplicity 1
///     .build();
/// ```
struct LookupTraceBuilder {
    /// Number of columns in the main trace
    width: usize,
    /// (read_values, provide_values, multiplicity)
    rows: Vec<(Vec<F>, Vec<F>, F)>,
    /// Random challenges for the local lookup argument
    local_challenges: LogUpChallenges,
    /// Random challenges for the global lookup argument
    global_challenges: Option<LogUpChallenges>,
}

impl LookupTraceBuilder {
    /// Create a new trace builder with a random challenge.
    fn new(rng: &mut SmallRng) -> Self {
        Self::new_with_width(3, rng)
    }

    /// Create a new trace builder with a random challenge for a given width.
    fn new_with_width(width: usize, rng: &mut SmallRng) -> Self {
        let local_challenges = LogUpChallenges {
            alpha: EF::from_u32(rng.random()),
            beta: EF::from_u32(rng.random()),
        };

        Self {
            width,
            rows: Vec::new(),
            local_challenges,
            global_challenges: None,
        }
    }

    /// Add a row to the trace.
    ///
    /// # Arguments
    /// * `read` - The value being read (always with multiplicity 1)
    /// * `provide` - The value being provided to the lookup table
    /// * `mult` - The multiplicity of the provided value
    fn row(mut self, reads: &[u32], provide: &[u32], mult: u32) -> Self {
        let reads_field = reads.iter().map(|&read| F::new(read)).collect::<Vec<_>>();
        let provides_field = provide
            .iter()
            .map(|&provide| F::new(provide))
            .collect::<Vec<_>>();
        self.rows.push((reads_field, provides_field, F::new(mult)));
        self
    }

    /// Build the main and auxiliary traces.
    ///
    /// Returns `(main_trace, aux_trace, alpha)` where:
    /// - `main_trace`: 3 columns [read_val, provide_val, multiplicity]
    /// - `aux_trace`: 1 column [running_sum]
    /// - `alpha`: The challenge used
    fn build(self) -> (RowMajorMatrix<F>, RowMajorMatrix<EF>, LogUpChallenges) {
        assert!(!self.rows.is_empty(), "Must have at least one row");

        // Build main trace: flatten (read, provide, mult) tuples into a single vector
        let main_flat: Vec<F> = self
            .rows
            .iter()
            .flat_map(|(read, provide, mult)| [read.clone(), provide.clone(), vec![*mult]].concat())
            .collect();
        let main_trace = RowMajorMatrix::new(main_flat, self.width);

        // Build auxiliary trace: running sum column
        // s[0] = 0, s[i+1] = s[i] + contribution_from_row_i
        let mut running_sum = EF::ZERO;
        let s_col: Vec<EF> = core::iter::once(EF::ZERO)
            .chain(self.rows.iter().map(|(read, provide, mult)| {
                running_sum +=
                    compute_logup_contribution(self.local_challenges, read, provide, *mult);
                running_sum
            }))
            .take(self.rows.len()) // Keep trace length equal to number of rows
            .collect();

        let aux_trace = RowMajorMatrix::new(s_col, 1);

        (main_trace, aux_trace, self.local_challenges)
    }

    /// Build the main and auxiliary traces.
    ///
    /// Returns `(main_trace, aux_trace, alpha)` where:
    /// - `main_trace`: 3 columns [read_val, provide_val, multiplicity]
    /// - `aux_trace`: 1 column [running_sum]
    /// - `challenges`: the challenges used
    fn build_with_global(
        self,
        direction: Direction,
    ) -> (RowMajorMatrix<F>, RowMajorMatrix<EF>, LogUpChallenges) {
        assert!(!self.rows.is_empty(), "Must have at least one row");

        // Build main trace: flatten (read, provide, mult) tuples into a single vector
        let main_flat: Vec<F> = self
            .rows
            .iter()
            .flat_map(|(read, provide, mult)| [read.clone(), provide.clone(), vec![*mult]].concat())
            .collect();
        let main_trace = RowMajorMatrix::new(main_flat, self.width);

        // Build auxiliary trace: running sum column
        // s[0] = 0, s[i+1] = s[i] + contribution_from_row_i
        let mut running_sum = EF::ZERO;
        let mut global_running_sum = EF::ZERO;
        let s_col: Vec<EF> = core::iter::once(EF::ZERO)
            .chain(core::iter::once(EF::ZERO))
            .chain(self.rows.iter().flat_map(|(read, provide, mult)| {
                running_sum +=
                    compute_logup_contribution(self.local_challenges, read, provide, *mult);
                let global_mult = direction.multiplicity(F::ONE);

                global_running_sum += compute_logup_contribution(
                    self.global_challenges.unwrap(),
                    &[],
                    provide,
                    global_mult,
                );
                vec![running_sum, global_running_sum]
            }))
            .take(2 * self.rows.len()) // 2 initial zeros + 2 values per row
            .collect();

        let aux_trace = RowMajorMatrix::new(s_col, 2);
        (main_trace, aux_trace, self.local_challenges)
    }
}

#[test]
fn test_symbolic_to_expr() {
    use p3_air::AirBuilder;
    use p3_field::PrimeCharacteristicRing;
    use p3_uni_stark::SymbolicAirBuilder;

    let mut builder = SymbolicAirBuilder::<F>::new(0, 2, 0, 0, 0);

    let main = builder.main();

    let (local, next) = (main.row_slice(0).unwrap(), main.row_slice(1).unwrap());

    let mul = local[0] * next[1];
    let add = local[0] + next[1];
    let sub = local[0] - next[1];
    builder.when_first_row().assert_zero(mul.clone() * add);
    builder.when_transition().assert_zero(sub - local[0]);
    builder.when_last_row().assert_zero(mul - local[0]);

    let constraints = builder.base_constraints();

    let mut main_flat = Vec::new();
    main_flat.extend([F::new(10), F::new(10)]);
    main_flat.extend([F::new(256), F::new(255)]);
    main_flat.extend([F::new(42), F::new(42)]);

    let main_trace = RowMajorMatrix::new(main_flat, 2);

    let perm = RowMajorMatrix::new(vec![], 0);
    let mut builder = MockAirBuilder::new(main_trace.clone(), perm, vec![]);

    for i in 0..builder.height {
        // Define the Lagrange selectors.
        builder.for_row(i);
        let is_first_row = if i == 0 { EF::ONE } else { EF::ZERO };
        let is_last_row = if i == builder.height - 1 {
            EF::ONE
        } else {
            EF::ZERO
        };
        let is_transition = if i < builder.height - 1 {
            EF::ONE
        } else {
            EF::ZERO
        };

        // Get the local and next values for row `i`.
        let cloned_trace = main_trace.clone();
        let local = cloned_trace.row(i).unwrap().into_iter().collect::<Vec<F>>();
        let next = cloned_trace.row(i + 1).map_or_else(
            || vec![F::ZERO; 2],
            |row| row.into_iter().collect::<Vec<F>>(),
        );

        // Compute the expected constraint values at row `i`.
        let mul = EF::from(local[0]) * EF::from(next[1]);
        let add = EF::from(local[0]) + EF::from(next[1]);
        let sub = EF::from(local[0]) - EF::from(next[1]);

        let first_expected_val = is_first_row * (mul * add);
        let transition_expected_val = is_transition * (sub - EF::from(local[0]));
        let last_expected_val = is_last_row * (mul - EF::from(local[0]));

        // Evaluate the constraints at row `i`.
        let first_eval = symbolic_to_expr(&builder, &constraints[0]);
        let transition_eval = symbolic_to_expr(&builder, &constraints[1]);
        let last_eval = symbolic_to_expr(&builder, &constraints[2]);

        // Assert that the evaluated constraints are correct.
        assert_eq!(first_expected_val, first_eval.into());
        assert_eq!(transition_expected_val, transition_eval.into());
        assert_eq!(last_expected_val, last_eval.into());
    }
}

#[test]
fn test_range_check_end_to_end_valid() {
    // SCENARIO: Simple range check where each value reads and provides itself.
    //
    // Values to check: [10, 255, 0, 42, 10]
    // Each row contributes 1/(α-val) - 1/(α-val) = 0, so final sum is 0.
    let mut rng = SmallRng::seed_from_u64(1);
    let (main_trace, aux_trace, challenges) = LookupTraceBuilder::new(&mut rng)
        .row(&[10], &[10], 1)
        .row(&[255], &[255], 1)
        .row(&[0], &[0], 1)
        .row(&[42], &[42], 1)
        .row(&[10], &[10], 1)
        .build();

    // The test must check the FINAL constraint: s[n-1] + c[n-1] = 0
    let s_final = aux_trace
        .row(aux_trace.height() - 1)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let last_row_data = main_trace
        .row(main_trace.height() - 1)
        .unwrap()
        .into_iter()
        .collect::<Vec<F>>();
    let last_contribution = compute_logup_contribution(
        challenges,
        &[last_row_data[0]],
        &[last_row_data[1]],
        last_row_data[2],
    );

    assert_eq!(
        s_final + last_contribution,
        EF::ZERO,
        "Total sum (s[n-1] + c[n-1]) must be zero for a valid lookup"
    );

    // Setup the AIR and builder.
    let mut air = RangeCheckAir::new();
    let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges.to_vec());

    let lookup_gadget = LogUpGadget::new();
    let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

    // Check that the lookup was created correctly.
    assert_eq!(lookups.len(), 1, "Should have one lookup defined");
    assert_eq!(
        lookups[0].columns,
        vec![0],
        "Lookup should use the first auxiliary column"
    );
    assert_eq!(
        lookups[0].element_exprs.len(),
        2,
        "Lookup should have two element tuples (read and provide)"
    );
    assert_eq!(lookups[0].kind, Kind::Local, "Lookup should be local");
    assert_eq!(air.cur_num_lookups, 1, "Should have one lookup registered");

    // Evaluate constraints for every row.
    for i in 0..builder.height {
        builder.for_row(i);
        lookups.iter().for_each(|lookup| {
            lookup_gadget.eval_local_lookup(&mut builder, lookup);
        });
    }
}

#[test]
#[should_panic(expected = "Extension constraint failed")]
fn test_range_check_end_to_end_invalid() {
    // SCENARIO: One value (256) is outside the 8-bit range [0, 255].
    // We create a trace where we read 256 but provide 255 from the table

    // Create an invalid witness: we read 256 but provide a different value (255)
    let mut main_flat = Vec::new();
    main_flat.extend([F::new(10), F::new(10), F::ONE]); // valid row
    main_flat.extend([F::new(256), F::new(255), F::ONE]); // Mismatch!
    main_flat.extend([F::new(42), F::new(42), F::ONE]); // valid row

    let main_trace = RowMajorMatrix::new(main_flat, 3);
    let challenges = LogUpChallenges {
        alpha: EF::from_u32(12345678),
        beta: EF::from_u32(87654321),
    };
    let alpha = EF::from(F::new(0x12345678));
    let beta = EF::from(F::new(0x87654321));

    // Build auxiliary running sum column - this will be non-zero at the end
    let mut s_col = Vec::with_capacity(main_trace.height());
    let mut current_s = EF::ZERO;

    for i in 0..main_trace.height() {
        let row: Vec<F> = main_trace.row(i).unwrap().into_iter().collect();
        let val_read = vec![row[0]];
        let val_provided = vec![row[1]];
        let mult = row[2];

        let contribution = compute_logup_contribution(challenges, &val_read, &val_provided, mult);
        current_s += contribution;

        // s[i] includes the contribution from row i
        s_col.push(current_s);
    }

    let aux_trace = RowMajorMatrix::new(s_col, 1);

    // The final sum should be non-zero due to the mismatch
    let final_row: Vec<EF> = aux_trace
        .row(aux_trace.height() - 1)
        .unwrap()
        .into_iter()
        .collect();
    let final_s = final_row[0];
    assert_ne!(final_s, EF::ZERO);

    // Setup the AIR and builder
    let mut air = RangeCheckAir::new();
    let mut builder = MockAirBuilder::new(main_trace, aux_trace, vec![alpha, beta]);

    let lookup_gadget = LogUpGadget::new();
    let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

    // Evaluate constraints.
    //
    // This should fail on the second row due to the non-zero final sum
    for i in 0..builder.height {
        builder.for_row(i);
        lookups.iter().for_each(|lookup| {
            lookup_gadget.eval_local_lookup(&mut builder, lookup);
        });
    }
}

#[test]
fn test_compute_sum_terms_logic() {
    let gadget = LogUpGadget::new();
    let alpha = EF::from_u8(100);
    let beta = EF::from_u8(100);

    // Elements: [2, 5], Multiplicities: [3, 1]
    //
    // We want to compute: 3/(α - 2) + 1/(α - 5)
    let elements = [vec![F::new(2)], vec![F::new(5)]];
    let multiplicities = [F::new(3), F::ONE];

    // Expected Numerator: 3 * (α - 5) + 1 * (α - 2)
    let expected_numerator = (alpha - EF::from_u8(5)) * EF::from_u8(3) + (alpha - EF::from_u8(2));
    // Expected Denominator: (α - 2) * (α - 5)
    let expected_denominator = (alpha - EF::from_u8(2)) * (alpha - EF::from_u8(5));

    let (num, den) = gadget.compute_combined_sum_terms::<MockAirBuilder, F, F>(
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
    // SCENARIO: The main trace is valid, but the prover messes up the running sum calculation.
    let mut rng = SmallRng::seed_from_u64(1);
    let (main_trace, mut aux_trace, challenges) = LookupTraceBuilder::new(&mut rng)
        .row(&[10], &[10], 1)
        .row(&[20], &[20], 1)
        .row(&[30], &[30], 1)
        .build();

    // The witness is valid so far. Let's corrupt it.
    // The transition from row 1 to 2 will be s_2 = s_1 + C_1.
    //
    // Let's set s_2 to a garbage value to make the transition fail.
    let corrupted_s_val = EF::from_u8(99);
    aux_trace.values[2] = corrupted_s_val;

    // Evaluate constraints.
    //
    // This should now fail at row 1 when checking the transition to row 2.
    let mut air = RangeCheckAir::new();
    let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges.to_vec());

    // Register the lookups.
    let lookup_gadget = LogUpGadget::new();
    let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

    // Evaluate the constraints.
    for i in 0..builder.height {
        builder.for_row(i);
        lookups.iter().for_each(|lookup| {
            lookup_gadget.eval_local_lookup(&mut builder, lookup);
        });
    }
}

#[test]
#[should_panic(expected = "Extension constraint failed at row 0")]
fn test_zero_multiplicity_is_not_counted() {
    // SCENARIO: We read `10`, but the lookup table provides it with multiplicity 0.
    // We add a second, valid row to ensure the transition and boundary constraints are tested.
    let mut main_flat = Vec::new();
    // Read 10, provide 10 with mult 0
    main_flat.extend([F::new(10), F::new(10), F::ZERO]);
    // A valid row
    main_flat.extend([F::new(20), F::new(20), F::ONE]);

    let main_trace = RowMajorMatrix::new(main_flat, 3);
    let challenges = LogUpChallenges {
        alpha: EF::from_u8(123),
        beta: EF::from_u8(111),
    };

    // Build witness. The contribution from row 0 will be non-zero.
    let mut s_col = Vec::with_capacity(main_trace.height());
    let mut current_s = EF::ZERO;
    for i in 0..main_trace.height() {
        let row: Vec<F> = main_trace.row(i).unwrap().into_iter().collect();
        let contribution = compute_logup_contribution(challenges, &[row[0]], &[row[1]], row[2]);
        current_s += contribution;

        // s[i] includes the contribution from row i
        s_col.push(current_s);
    }
    let aux_trace = RowMajorMatrix::new(s_col, 1);

    // The final value in the `s` column (s_1) should be non-zero.
    let final_s = aux_trace
        .row(main_trace.height() - 1)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    assert_ne!(final_s, EF::ZERO);

    // Evaluate constraints
    let mut air = RangeCheckAir::new();
    let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges.to_vec());

    // Register the lookups.
    let lookup_gadget = LogUpGadget::new();
    let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

    // The initial boundary constraint will fail on row 0 since s[0] is incorrect.
    //
    // It will panic on the first row (row 0).
    for i in 0..builder.height {
        builder.for_row(i);
        lookups.iter().for_each(|lookup| {
            lookup_gadget.eval_local_lookup(&mut builder, lookup);
        });
    }
}

#[test]
fn test_empty_lookup_is_valid() {
    // SCENARIO: A lookup is triggered, but both the read and provided sets are empty.
    let main_trace = RowMajorMatrix::new(vec![], 3);
    let aux_trace = RowMajorMatrix::new(vec![], 1);
    let alpha = EF::from_u8(123);

    let mut air = RangeCheckAir::new();
    let mut builder = MockAirBuilder::new(main_trace, aux_trace, vec![alpha]);

    let lookup_gadget = LogUpGadget::new();
    let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

    // This should not panic, as there are no rows to evaluate.
    for i in 0..builder.height {
        builder.for_row(i);
        lookups.iter().for_each(|lookup| {
            lookup_gadget.eval_local_lookup(&mut builder, lookup);
        });
    }

    // Also test the internal logic directly
    let gadget = LogUpGadget::new();
    let (num, den) =
        gadget.compute_combined_sum_terms::<MockAirBuilder, F, F>(&[], &[], &alpha, &alpha);
    assert_eq!(num, EF::ZERO);
    assert_eq!(den, EF::ONE);
}

#[test]
fn test_nontrivial_permutation() {
    // SCENARIO: Complex permutation with varied multiplicities
    //
    // Multiset equality check:
    // - Reads: {3: 2, 5: 4, 7: 2}  (8 values total)
    // - Provides: {3: 2, 5: 4, 7: 2}  ✓
    //
    // ┌─────┬──────┬─────────┬──────┬────────────────────────────────────┐
    // │ Row │ Read │ Provide │ Mult │ Contribution                       │
    // ├─────┼──────┼─────────┼──────┼────────────────────────────────────┤
    // │  0  │  7   │    3    │  2   │  1/(α-7) - 2/(α-3)                │
    // │  1  │  3   │    5    │  4   │  1/(α-3) - 4/(α-5)                │
    // │  2  │  5   │    7    │  2   │  1/(α-5) - 2/(α-7)                │
    // │  3  │  3   │    3    │  0   │  1/(α-3) - 0 = 1/(α-3)            │
    // │  4  │  7   │    5    │  0   │  1/(α-7) - 0 = 1/(α-7)            │
    // │  5  │  5   │    5    │  0   │  1/(α-5) - 0 = 1/(α-5)            │
    // │  6  │  5   │    7    │  0   │  1/(α-5) - 0 = 1/(α-5)            │
    // │  7  │  5   │    --   │  0   │  1/(α-5) - 0 = 1/(α-5)            │
    // └─────┴──────┴─────────┴──────┴────────────────────────────────────┘
    //
    // Total contributions:
    // Reads:    [1/(α-7) + 1/(α-3) + 1/(α-5) + 1/(α-3) + 1/(α-7) + 1/(α-5) + 1/(α-5) + 1/(α-5)]
    //         = 2/(α-7) + 2/(α-3) + 4/(α-5)
    // Provides: [2/(α-3) + 4/(α-5) + 2/(α-7) + 0 + 0 + 0 + 0 + 0]
    //         = 2/(α-3) + 4/(α-5) + 2/(α-7)
    // Difference: 0

    let mut rng = SmallRng::seed_from_u64(1);
    let (main_trace, aux_trace, challenges) = LookupTraceBuilder::new(&mut rng)
        .row(&[7], &[3], 2) // Read 7, provide {3, 3} to the table
        .row(&[3], &[5], 4) // Read 3, provide {5, 5, 5, 5} to the table (4 fives total)
        .row(&[5], &[7], 2) // Read 5, provide {7, 7} to the table
        .row(&[3], &[3], 0) // Read 3, no provides (mult=0)
        .row(&[7], &[5], 0) // Read 7, no provides (mult=0)
        .row(&[5], &[5], 0) // Read 5, no provides (mult=0)
        .row(&[5], &[7], 0) // Read 5, no provides (mult=0)
        .row(&[5], &[5], 0) // Read 5, no provides (mult=0)
        .build();

    // The test must check the FINAL constraint: s[n-1] + c[n-1] = 0
    let s_final = aux_trace
        .row(aux_trace.height() - 1)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let last_row_data = main_trace
        .row(main_trace.height() - 1)
        .unwrap()
        .into_iter()
        .collect::<Vec<F>>();
    let last_contribution = compute_logup_contribution(
        challenges,
        &[last_row_data[0]],
        &[last_row_data[1]],
        last_row_data[2],
    );

    assert_eq!(
        s_final + last_contribution,
        EF::ZERO,
        "Total sum (s[n-1] + c[n-1]) must be zero for a valid lookup"
    );

    // Setup AIR and verify all constraints
    let mut air = RangeCheckAir::new();
    let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges.to_vec());

    // Register the lookups.
    let lookup_gadget = LogUpGadget::new();
    let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

    // Evaluate constraints for every row
    for i in 0..builder.height {
        builder.for_row(i);
        lookups.iter().for_each(|lookup| {
            lookup_gadget.eval_local_lookup(&mut builder, lookup);
        });
    }
}

#[test]
fn test_multiple_lookups_different_columns() {
    // SCENARIO: Two independent lookups in the same AIR using different columns
    //
    // Lookup 1 (column 0): Simple range check [10, 20, 30]
    // Lookup 2 (column 1): Different values [5, 15, 25]

    let first_challenges = LogUpChallenges {
        alpha: EF::from_u32(0x12345678),
        beta: EF::from_u32(0x87654321),
    };
    let second_challenges = LogUpChallenges {
        alpha: EF::from_u32(0xABCDEF01),
        beta: EF::from_u32(0x10FEDCBA),
    };
    // Build main trace with 6 columns
    // Format: [read1, provide1, mult1, read2, provide2, mult2]
    let main_flat = vec![
        // Row 0
        F::new(10),
        F::new(10),
        F::ONE, // Lookup 1: read 10, provide 10
        F::new(5),
        F::new(5),
        F::ONE, // Lookup 2: read 5, provide 5
        // Row 1
        F::new(20),
        F::new(20),
        F::ONE, // Lookup 1: read 20, provide 20
        F::new(15),
        F::new(15),
        F::ONE, // Lookup 2: read 15, provide 15
        // Row 2
        F::new(30),
        F::new(30),
        F::ONE, // Lookup 1: read 30, provide 30
        F::new(25),
        F::new(25),
        F::ONE, // Lookup 2: read 25, provide 25
    ];
    let main_trace = RowMajorMatrix::new(main_flat, 6);

    // Build auxiliary trace with 2 columns (one per lookup)
    let mut s1_col = Vec::with_capacity(3); // Running sum for lookup 1
    let mut s2_col = Vec::with_capacity(3); // Running sum for lookup 2

    let mut s1 = EF::ZERO;
    let mut s2 = EF::ZERO;

    // Row 0: s[0] = 0 for both lookups
    s1_col.push(s1);
    s2_col.push(s2);

    // Row 1: Add contributions from row 0
    s1 += compute_logup_contribution(first_challenges, &[F::new(10)], &[F::new(10)], F::ONE);
    s2 += compute_logup_contribution(second_challenges, &[F::new(5)], &[F::new(5)], F::ONE);
    s1_col.push(s1);
    s2_col.push(s2);

    // Row 2: Add contributions from row 1
    s1 += compute_logup_contribution(first_challenges, &[F::new(20)], &[F::new(20)], F::ONE);
    s2 += compute_logup_contribution(second_challenges, &[F::new(15)], &[F::new(15)], F::ONE);
    s1_col.push(s1);
    s2_col.push(s2);

    // Interleave the two columns into a single flat vector
    let mut aux_flat = Vec::with_capacity(6);
    for i in 0..3 {
        aux_flat.push(s1_col[i]);
        aux_flat.push(s2_col[i]);
    }
    let aux_trace = RowMajorMatrix::new(aux_flat, 2); // 2 columns

    // Verify both final sums are zero (each lookup is valid)
    let row2_data: Vec<EF> = aux_trace.row(2).unwrap().into_iter().collect();
    let s1_final = row2_data[0];
    let s2_final = row2_data[1];

    let c1_final =
        compute_logup_contribution(first_challenges, &[F::new(30)], &[F::new(30)], F::ONE);
    let c2_final =
        compute_logup_contribution(second_challenges, &[F::new(25)], &[F::new(25)], F::ONE);

    assert_eq!(
        s1_final + c1_final,
        EF::ZERO,
        "Lookup 1 final sum must be zero"
    );
    assert_eq!(
        s2_final + c2_final,
        EF::ZERO,
        "Lookup 2 final sum must be zero"
    );

    // Setup AIR with 2 lookups and verify all constraints
    let mut air = RangeCheckAir::with_multiple_lookups(2);
    let mut builder = MockAirBuilder::new(
        main_trace,
        aux_trace,
        [first_challenges.to_vec(), second_challenges.to_vec()].concat(),
    );

    // Register lookups.
    let lookup_gadget = LogUpGadget::new();
    let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

    // Check that the lookup was created correctly.
    assert_eq!(lookups.len(), 2, "Should have two lookups defined");
    assert_eq!(
        (lookups[0].columns.clone(), lookups[1].columns.clone()),
        (vec![0], vec![1]),
        "Lookup should use the first two auxiliary column"
    );
    assert_eq!(
        lookups[0].element_exprs.len(),
        2,
        "Lookup should have two element tuples (read and provide)"
    );
    assert_eq!(lookups[0].kind, Kind::Local, "Lookup should be local");
    assert_eq!(air.cur_num_lookups, 2, "Should have two lookups registered");

    // Evaluate constraints for every row - both lookups should pass
    for i in 0..builder.height {
        builder.for_row(i);
        lookups.iter().for_each(|lookup| {
            lookup_gadget.eval_local_lookup(&mut builder, lookup);
        });
    }
}

// An AIR with 6 columns as follows:
// - 3 columns the input add operation: (inp1, inp2, sum)
// - 3 columns for the lookup table: (table_inp1, table_inp2, table_sum)
// - 1 column for the multiplicity
struct AddAir {
    // To keep track of registered lookups
    num_lookups: usize,
    with_global: (bool, Direction),
}

impl AddAir {
    fn new() -> Self {
        Self {
            num_lookups: 0,
            with_global: (false, Direction::Send),
        }
    }

    fn new_with_global(direction: Direction) -> Self {
        Self {
            num_lookups: 0,
            with_global: (true, direction),
        }
    }
}

impl<F: Field> BaseAir<F> for AddAir {
    fn width(&self) -> usize {
        7
    }
}

impl<AB> Air<AB> for AddAir
where
    AB: PermutationAirBuilder<F = F, EF = EF, RandomVar = EF>
        + PairBuilder
        + AirBuilderWithPublicValues,
    AB::Var: Copy + Into<AB::ExprEF>,
    AB::ExprEF: From<AB::Var> + From<F>,
    F: Copy + Into<AB::ExprEF>,
{
    fn eval(&self, _builder: &mut AB) {
        // No constraints, only lookups
    }
}

impl<AB> AirLookupHandler<AB> for AddAir
where
    AB: PermutationAirBuilder<F = F, EF = EF, RandomVar = EF>
        + PairBuilder
        + AirBuilderWithPublicValues,
    AB::Var: Copy + Into<AB::ExprEF>,
    AB::ExprEF: From<AB::Var> + From<F>,
    F: Copy + Into<AB::ExprEF>,
{
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookups;
        self.num_lookups += 1;

        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<AB::F>> {
        let symbolic_air_builder =
            SymbolicAirBuilder::<F>::new(0, BaseAir::<AB::F>::width(self), 0, 0, 0);

        let symbolic_main = symbolic_air_builder.main();
        let symbolic_main_local = symbolic_main.row_slice(0).unwrap();

        // Extract columns for thelookup entries: [inp1, inp2, sum]
        let inp1 = symbolic_main_local[0];
        let inp2 = symbolic_main_local[1];
        let sum = symbolic_main_local[2];

        // Form the lookup inputs.
        let a_elements = vec![inp1.into(), inp2.into(), sum.into()];
        let a_multiplicities = SymbolicExpression::Constant(F::ONE);

        // Extract columns for the LUT entries: [table_inp1, table_inp2, table_sum]
        let table_inp1 = symbolic_main_local[3];
        let table_inp2 = symbolic_main_local[4];
        let table_sum = symbolic_main_local[5];
        // Form the LUT entries.
        let b_elements = vec![table_inp1.into(), table_inp2.into(), table_sum.into()];
        let b_multiplicities = symbolic_main_local[6].into();

        let lookup_inputs = vec![
            (a_elements, a_multiplicities, Direction::Receive),
            (b_elements.clone(), b_multiplicities, Direction::Send),
        ];

        let local_lookup =
            AirLookupHandler::<AB>::register_lookup(self, Kind::Local, &lookup_inputs);

        // also need is_send
        let (is_global, direction) = self.with_global;
        if is_global {
            let lookup_inputs = vec![(b_elements, SymbolicExpression::Constant(F::ONE), direction)];
            let global_lookup = AirLookupHandler::<AB>::register_lookup(
                self,
                Kind::Global("LUT".to_string()),
                &lookup_inputs,
            );
            // Return the local and global lookups.
            return vec![local_lookup, global_lookup];
        }
        // Return the local lookup.
        vec![local_lookup]
    }
}

#[test]
fn test_tuple_lookup() {
    // SCENARIO: Lookup with 3-column tuples representing addition operations.
    // We have a lookup table with all valid additions of binary values.
    // Values to check: [(0, 1, 1), (0, 1, 1), (1, 1, 2), (0, 0, 0)]
    // Lookup table: [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)]
    let mut rng = SmallRng::seed_from_u64(1);
    let mut air = AddAir::new();
    let width = <AddAir as BaseAir<F>>::width(&air);
    let (main_trace, aux_trace, challenges) = LookupTraceBuilder::new_with_width(width, &mut rng)
        .row(&[0, 1, 1], &[0, 1, 1], 2)
        .row(&[0, 1, 1], &[0, 0, 0], 1)
        .row(&[1, 1, 2], &[1, 0, 1], 0)
        .row(&[0, 0, 0], &[1, 1, 2], 1)
        .build();

    // The test must check the FINAL constraint: s[n-1] + c[n-1] = 0
    let s_final = aux_trace
        .row(aux_trace.height() - 1)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let last_row_data = main_trace
        .row(main_trace.height() - 1)
        .unwrap()
        .into_iter()
        .collect::<Vec<F>>();

    let last_contribution = compute_logup_contribution(
        challenges,
        &last_row_data[0..3],
        &last_row_data[3..6],
        last_row_data[6],
    );

    assert_eq!(
        s_final + last_contribution,
        EF::ZERO,
        "Total sum (s[n-1] + c[n-1]) must be zero for a valid lookup"
    );

    let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges.to_vec());

    // Register the lookups.
    let lookup_gadget = LogUpGadget::new();
    let lookups = <AddAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

    // Evaluate the constraints for every row.
    for i in 0..builder.height {
        builder.for_row(i);
        lookups.iter().for_each(|lookup| {
            lookup_gadget.eval_local_lookup(&mut builder, lookup);
        });
    }
}

#[test]
fn test_global_lookup() {
    // SCENARIO: We have two `AddAir`s.
    // In each, we perform a local (tuple) lookup:
    //      - We have a lookup table with all valid additions of binary values.
    //      - Values to check: [(0, 1, 1), (0, 1, 1), (1, 1, 2), (0, 0, 0)]
    //      - Lookup table: [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)]
    // We have a global lookup between the LUTs of the two AIRs:
    //      - One AIR sends the tuple values corresponding to the addition table.
    //      - The other AIR receives them.
    //      - The two LUTs are not stored in the same order, so that it is not a trivial lookup.

    let mut rng = SmallRng::seed_from_u64(1);

    // Get challenges for the global lookup.
    let global_challenges = LogUpChallenges {
        alpha: EF::from_u32(rng.random()),
        beta: EF::from_u32(rng.random()),
    };

    // The first AIR receives LUT values from the second air.
    let mut air1 = AddAir::new_with_global(Direction::Receive);
    let width = <AddAir as BaseAir<F>>::width(&air1);
    let (main_trace1, aux_trace1, challenges1) = {
        let mut trace_builder = LookupTraceBuilder::new_with_width(width, &mut rng);
        trace_builder.global_challenges = Some(global_challenges);
        trace_builder
            .row(&[0, 1, 1], &[0, 0, 0], 1)
            .row(&[0, 1, 1], &[0, 1, 1], 2)
            .row(&[1, 1, 2], &[1, 1, 2], 1)
            .row(&[0, 0, 0], &[1, 0, 1], 0)
            .build_with_global(Direction::Receive)
    };

    // The second AIR sends LUT values to the first air.
    let mut air2 = AddAir::new_with_global(Direction::Send);
    let width = <AddAir as BaseAir<F>>::width(&air2);
    let (main_trace2, aux_trace2, challenges2) = {
        let mut trace_builder = LookupTraceBuilder::new_with_width(width, &mut rng);
        trace_builder.global_challenges = Some(global_challenges);
        trace_builder
            .row(&[0, 1, 1], &[0, 1, 1], 2)
            .row(&[0, 1, 1], &[0, 0, 0], 1)
            .row(&[1, 1, 2], &[1, 0, 1], 0)
            .row(&[0, 0, 0], &[1, 1, 2], 1)
            .build_with_global(Direction::Send)
    };

    // The test must check the FINAL constraint: s[n-1] + c[n-1] = 0
    let last_aux_trace1 = aux_trace1
        .row(aux_trace1.height() - 1)
        .unwrap()
        .into_iter()
        .collect::<Vec<EF>>();
    let s_final1 = last_aux_trace1[0];
    let last_row_data = main_trace1
        .row(main_trace1.height() - 1)
        .unwrap()
        .into_iter()
        .collect::<Vec<F>>();

    let last_contribution1 = compute_logup_contribution(
        challenges1,
        &last_row_data[0..3],
        &last_row_data[3..6],
        last_row_data[6],
    );

    assert_eq!(
        s_final1 + last_contribution1,
        EF::ZERO,
        "Total sum (s[n-1] + c[n-1]) must be zero for a valid lookup"
    );

    let last_aux_trace2 = aux_trace2
        .row(aux_trace2.height() - 1)
        .unwrap()
        .into_iter()
        .collect::<Vec<EF>>();
    // The test must check the FINAL constraint: s[n-1] + c[n-1] = 0 for the second AIR.
    let s_final2 = last_aux_trace2[0];
    let last_row_data2 = main_trace2
        .row(main_trace2.height() - 1)
        .unwrap()
        .into_iter()
        .collect::<Vec<F>>();

    let last_contribution2 = compute_logup_contribution(
        challenges2,
        &last_row_data2[0..3],
        &last_row_data2[3..6],
        last_row_data2[6],
    );

    assert_eq!(
        s_final2 + last_contribution2,
        EF::ZERO,
        "Total sum (s[n-1] + c[n-1]) must be zero for a valid lookup"
    );

    // Retrieve the final values of the auxiliary columns for global lookups.
    // The second auxiliary column corresponds to the global lookup.
    let s_global1 = last_aux_trace1[1];
    let s_global2 = last_aux_trace2[1];

    let last_global_contribution1 =
        compute_logup_contribution(global_challenges, &[], &last_row_data[3..6], F::ONE);
    let last_global_contribution2 =
        compute_logup_contribution(global_challenges, &[], &last_row_data2[3..6], -F::ONE);

    let s_global_final1 = s_global1 + last_global_contribution1;
    let s_global_final2 = s_global2 + last_global_contribution2;

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

    // Register the lookups.
    let lookup_gadget = LogUpGadget::new();
    let lookups1 = <AddAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air1);
    let lookups2 = <AddAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air2);

    assert_eq!(
        builder1.height, builder2.height,
        "Both builders should have the same height"
    );

    // Evaluate the constraints for every row``, in each table.
    for i in 0..builder1.height {
        builder1.for_row(i);
        lookups1.iter().for_each(|lookup| {
            match &lookup.kind {
                Kind::Local => lookup_gadget.eval_local_lookup(&mut builder1, lookup),
                Kind::Global(name) => {
                    assert_eq!(*name, "LUT".to_string(), "Global lookup name should match");
                    lookup_gadget.eval_global_update(&mut builder1, lookup, s_global_final1);
                }
            };
        });

        builder2.for_row(i);
        lookups2.iter().for_each(|lookup| {
            match &lookup.kind {
                Kind::Local => lookup_gadget.eval_local_lookup(&mut builder2, lookup),
                Kind::Global(name) => {
                    assert_eq!(*name, "LUT".to_string(), "Global lookup name should match");
                    lookup_gadget.eval_global_update(&mut builder2, lookup, s_global_final2);
                }
            };
        });
    }

    // Evaluate the global lookup between the two AIRs.
    lookup_gadget
        .verify_global_final_value(&[s_global_final1, s_global_final2])
        .expect("Global lookups final values should sum to 0.");
}
