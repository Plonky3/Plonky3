use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use p3_air::symbolic::{AirLayout, SymbolicAirBuilder, SymbolicExpression};
use p3_air::{
    Air, AirBuilder, BaseAir, BaseLeaf, ExtensionBuilder, PermutationAirBuilder, WindowAccess,
};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::TwoAdicFriPcs;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::Lookups;
use crate::builder::InteractionBuilder;
use crate::logup::LogUpGadget;
use crate::protocol::LookupProtocol;
use crate::types::{Kind, Lookup, LookupError, LookupTerminal};

/// Base field used in every test.
type F = BabyBear;
/// Quartic extension field.
type EF = BinomialExtensionField<F, 4>;

// Minimal stark-config used as the `SC` parameter for `generate_permutation`.
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
type Challenger = DuplexChallenger<F, Perm, 16, 8>;
type Dft = Radix2DitParallel<F>;
type MyPcs = TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs>;
type TestConfig = StarkConfig<MyPcs, EF, Challenger>;

fn create_symbolic_with_degree(degree: usize) -> SymbolicExpression<F> {
    let x = Arc::new(SymbolicExpression::Leaf(BaseLeaf::Constant(F::ONE)));
    let y = Arc::new(SymbolicExpression::Leaf(BaseLeaf::Constant(F::TWO)));
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
    assert_eq!(num_elements_per_tuple.len(), degree_per_element.len());
    assert_eq!(num_elements_per_tuple.len(), degree_multiplicities.len());

    let elements = num_elements_per_tuple
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            assert_eq!(num_elements_per_tuple[i], degree_per_element[i].len());
            (0..n)
                .map(|j| create_symbolic_with_degree(degree_per_element[i][j]))
                .collect()
        })
        .collect();

    let multiplicities = degree_multiplicities
        .iter()
        .map(|&deg| create_symbolic_with_degree(deg))
        .collect();

    Lookup {
        kind: Kind::Local,
        elements,
        multiplicities,
        column: 0,
    }
}

#[test]
fn constraint_degree_matches_formula() {
    let gadget = LogUpGadget::new();

    // Two tuples, each with one element of degree 1, multiplicities of degree 1.
    //
    // deg(U) = 1 + 1 = 2; deg(V) = max_i(deg(m_i) + sum_{j != i} deg(e_j)) = 2.
    // Constraint degree = max(deg(U) + 1, deg(V)) = 3.
    let lookup = create_dummy_lookup(&[1, 1], &[vec![1], vec![1]], &[1, 1]);
    assert_eq!(gadget.constraint_degree(&lookup), 3);

    // Three tuples, one element each of degrees 1 / 2 / 3, multiplicities of
    // degree 1.
    //
    // deg(U) = 1 + 2 + 3 = 6; max numerator term:
    //   m_0 * (e_1 * e_2) = 1 + 2 + 3 = 6
    //   m_1 * (e_0 * e_2) = 1 + 1 + 3 = 5
    //   m_2 * (e_0 * e_1) = 1 + 1 + 2 = 4
    // Constraint degree = max(6 + 1, 6) = 7.
    let lookup = create_dummy_lookup(&[1, 1, 1], &[vec![1], vec![2], vec![3]], &[1, 1, 1]);
    assert_eq!(gadget.constraint_degree(&lookup), 7);
}

#[test]
fn compute_combined_sum_terms_matches_definition() {
    // Single tuple with one element.
    //
    // Expected: V = m * 1 = m, U = (alpha - e).
    let gadget = LogUpGadget::new();
    let alpha = EF::from_u32(17);
    let beta = EF::from_u32(2);
    let elements = vec![vec![EF::from_u32(5)]];
    let multiplicities = vec![EF::from_u32(3)];
    let (num, den) = gadget.compute_combined_sum_terms::<MockAirBuilder, EF, EF>(
        &elements,
        &multiplicities,
        &alpha,
        &beta,
    );
    assert_eq!(num, multiplicities[0]);
    assert_eq!(den, alpha - elements[0][0]);
}

#[test]
fn compute_combined_sum_terms_empty_returns_zero_over_one() {
    let gadget = LogUpGadget::new();
    let (num, den) =
        gadget.compute_combined_sum_terms::<MockAirBuilder, EF, EF>(&[], &[], &EF::ONE, &EF::ONE);
    assert_eq!(num, EF::ZERO);
    assert_eq!(den, EF::ONE);
}

#[test]
fn verify_terminal_sum_zero_total_is_ok() {
    let gadget = LogUpGadget::new();
    let terminals = vec![
        Some(LookupTerminal(EF::from_u32(5))),
        Some(LookupTerminal(-EF::from_u32(5))),
        None,
    ];
    assert!(gadget.verify_terminal_sum(&terminals).is_ok());
}

#[test]
fn verify_terminal_sum_nonzero_total_is_err() {
    let gadget = LogUpGadget::new();
    let terminals = vec![
        Some(LookupTerminal(EF::from_u32(5))),
        Some(LookupTerminal(EF::from_u32(3))),
    ];
    match gadget.verify_terminal_sum(&terminals) {
        Err(LookupError::TerminalSumNonZero) => {}
        other => panic!("expected TerminalSumNonZero, got {other:?}"),
    }
}

#[test]
fn verify_terminal_sum_ignores_absent_terminals() {
    let gadget = LogUpGadget::new();
    let terminals: Vec<Option<LookupTerminal<EF>>> = vec![None, None, None];
    assert!(gadget.verify_terminal_sum(&terminals).is_ok());
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
}

impl RangeCheckAir {
    const fn new() -> Self {
        Self { num_lookups: 1 }
    }

    const fn with_lookups(num_lookups: usize) -> Self {
        Self { num_lookups }
    }
}

impl<F: Field> BaseAir<F> for RangeCheckAir {
    fn width(&self) -> usize {
        3 * self.num_lookups
    }
}

impl<AB> Air<AB> for RangeCheckAir
where
    AB: AirBuilder<F: Field> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();

        for i in 0..self.num_lookups {
            let offset = i * 3;
            let val = local[offset]; // query value
            let table_val = local[offset + 1]; // table value
            let mult = local[offset + 2]; // multiplicity

            builder.push_local_interaction(vec![
                (vec![val.into()], AB::Expr::ONE),        // query side
                (vec![table_val.into()], -(mult.into())), // table side (negated)
            ]);
        }
    }
}

#[test]
fn from_air_extracts_one_local_lookup_per_declaration() {
    let air = RangeCheckAir::with_lookups(3);
    let lookups: Lookups<F> = Lookups::from_air::<EF, _>(&air);
    assert_eq!(lookups.len(), 3);
    for (i, lookup) in lookups.iter().enumerate() {
        assert_eq!(lookup.column, i);
        assert_eq!(lookup.kind, Kind::Local);
        // One tuple per side: query and table.
        assert_eq!(lookup.elements.len(), 2);
        assert_eq!(lookup.multiplicities.len(), 2);
    }
}

/// A mock `AirBuilder` for testing purposes that simulates constraint evaluation.
struct MockAirBuilder {
    /// Main trace matrix containing the execution trace data
    main: RowMajorMatrix<F>,
    /// Empty preprocessed matrix (no preprocessed columns in this mock).
    preprocessed: RowMajorMatrix<F>,
    /// Auxiliary trace:
    /// - column `0` is the accumulator,
    /// - columns `1..N+1` are fractions.
    permutation: RowMajorMatrix<EF>,
    /// Random challenges used in the LogUp argument
    challenges: Vec<EF>,
    /// Single-element slice carrying the AIR's committed terminal.
    permutation_values: Vec<EF>,
    /// Current row being evaluated during constraint checking
    current_row: usize,
    /// Total height (number of rows) in the trace
    height: usize,
}

impl MockAirBuilder {
    fn new(
        main: RowMajorMatrix<F>,
        permutation: RowMajorMatrix<EF>,
        challenges: Vec<EF>,
        permutation_values: Vec<EF>,
    ) -> Self {
        let height = main.height();
        Self {
            main,
            preprocessed: RowMajorMatrix::new(vec![], 0),
            permutation,
            challenges,
            permutation_values,
            current_row: 0,
            height,
        }
    }

    // Helper to update the builder to the current row being evaluated
    fn for_row(&mut self, row: usize) {
        self.current_row = row;
    }

    // A mock windowed view for the trace matrices.
    fn window<T: Clone + Send + Sync + Field>(
        &self,
        trace: &RowMajorMatrix<T>,
    ) -> RowMajorMatrix<T> {
        let mut view = Vec::with_capacity(2 * trace.width());
        // local row
        view.extend(trace.row(self.current_row).unwrap());
        // next row (if it exists)
        if self.current_row + 1 < self.height {
            view.extend(trace.row(self.current_row + 1).unwrap());
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
    type PreprocessedWindow = RowMajorMatrix<F>;
    type MainWindow = RowMajorMatrix<F>;
    type PublicVar = F;
    type PeriodicVar = F;

    fn main(&self) -> Self::MainWindow {
        self.window(&self.main)
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }

    fn is_first_row(&self) -> Self::Expr {
        F::from_bool(self.current_row == 0)
    }

    fn is_last_row(&self) -> Self::Expr {
        F::from_bool(self.current_row == self.height - 1)
    }

    fn is_transition(&self) -> Self::Expr {
        F::from_bool(self.current_row < self.height - 1)
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
        assert!(
            val == EF::ZERO,
            "Extension constraint failed at row {}: {:?} != 0",
            self.current_row,
            val
        );
    }
}

impl PermutationAirBuilder for MockAirBuilder {
    type MP = RowMajorMatrix<EF>;
    type RandomVar = EF;
    type PermutationVar = EF;

    fn permutation(&self) -> Self::MP {
        self.window(&self.permutation)
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.challenges
    }

    fn permutation_values(&self) -> &[Self::PermutationVar] {
        &self.permutation_values
    }
}

/// Build a balanced range-check trace of `height` rows.
///
/// Every row reads value `v` and provides value `v` with multiplicity 1, so
/// the lookup is multiset-balanced regardless of the random challenges.
fn balanced_range_check_main(height: usize, rng: &mut SmallRng) -> RowMajorMatrix<F> {
    let mut flat = Vec::with_capacity(height * 3);
    for _ in 0..height {
        let v: u32 = rng.random();
        flat.push(F::new(v));
        flat.push(F::new(v));
        flat.push(F::ONE);
    }
    RowMajorMatrix::new(flat, 3)
}

#[test]
fn generate_permutation_balances_to_zero_terminal() {
    let mut rng = SmallRng::seed_from_u64(0x1234_5678);
    let main = balanced_range_check_main(8, &mut rng);
    let air = RangeCheckAir::new();
    let lookups: Lookups<F> = Lookups::from_air::<EF, _>(&air);

    let gadget = LogUpGadget::new();
    let challenges = vec![EF::from_u32(7), EF::from_u32(11)];

    let (aux, terminal) =
        gadget.generate_permutation::<TestConfig>(&main, &None, &[], &lookups, &challenges);

    // The aux trace carries one accumulator column plus one fraction column.
    assert_eq!(aux.width(), 2);
    assert_eq!(aux.height(), main.height());

    // First row of the accumulator is anchored to zero.
    assert_eq!(aux.row_slice(0).unwrap()[0], EF::ZERO);

    // A balanced range check yields a zero terminal with overwhelming probability.
    let terminal = terminal.expect("AIR has lookups -> terminal must be present");
    assert_eq!(terminal.0, EF::ZERO);
}

#[test]
fn eval_all_passes_on_balanced_trace() {
    let mut rng = SmallRng::seed_from_u64(0xc0ffee);
    let main = balanced_range_check_main(16, &mut rng);
    let air = RangeCheckAir::new();
    let lookups: Lookups<F> = Lookups::from_air::<EF, _>(&air);

    let gadget = LogUpGadget::new();
    let challenges = vec![EF::from_u32(0x12345678), EF::from_u32(0x87654321)];

    let (aux, terminal) =
        gadget.generate_permutation::<TestConfig>(&main, &None, &[], &lookups, &challenges);
    let terminal = terminal.unwrap();

    // Walk every row and check the lookup constraints fire without panic.
    let mut builder = MockAirBuilder::new(main, aux, challenges, vec![terminal.0]);
    for r in 0..builder.height {
        builder.for_row(r);
        gadget.eval_all(&mut builder, &lookups);
    }
}

#[test]
fn eval_all_passes_on_multi_lookup_balanced_trace() {
    let mut rng = SmallRng::seed_from_u64(0xdeadbeef);

    // Build a 4-row main trace with two independent range checks per row.
    //
    // Each row carries [q1, t1, m1, q2, t2, m2] with q_i = t_i and m_i = 1.
    let mut flat = Vec::with_capacity(4 * 6);
    for _ in 0..4 {
        for _ in 0..2 {
            let v: u32 = rng.random();
            flat.push(F::new(v));
            flat.push(F::new(v));
            flat.push(F::ONE);
        }
    }
    let main = RowMajorMatrix::new(flat, 6);

    let air = RangeCheckAir::with_lookups(2);
    let lookups: Lookups<F> = Lookups::from_air::<EF, _>(&air);
    assert_eq!(lookups.len(), 2);

    let gadget = LogUpGadget::new();
    // Two lookups -> two challenge pairs.
    let challenges = vec![
        EF::from_u32(2),
        EF::from_u32(3),
        EF::from_u32(5),
        EF::from_u32(7),
    ];

    let (aux, terminal) =
        gadget.generate_permutation::<TestConfig>(&main, &None, &[], &lookups, &challenges);
    // 1 accumulator + 2 fraction columns.
    assert_eq!(aux.width(), 3);
    let terminal = terminal.unwrap();
    assert_eq!(terminal.0, EF::ZERO);

    let mut builder = MockAirBuilder::new(main, aux, challenges, vec![terminal.0]);
    for r in 0..builder.height {
        builder.for_row(r);
        gadget.eval_all(&mut builder, &lookups);
    }
}

#[test]
#[should_panic(expected = "Extension constraint failed")]
fn eval_all_rejects_unbalanced_trace() {
    let mut rng = SmallRng::seed_from_u64(42);
    // Balanced main trace.
    let main = balanced_range_check_main(8, &mut rng);
    let air = RangeCheckAir::new();
    let lookups: Lookups<F> = Lookups::from_air::<EF, _>(&air);
    let gadget = LogUpGadget::new();
    let challenges = vec![EF::from_u32(13), EF::from_u32(17)];

    let (mut aux, terminal) =
        gadget.generate_permutation::<TestConfig>(&main, &None, &[], &lookups, &challenges);

    // Tamper with the fraction column on row 2: violates U * frac - V = 0.
    let aux_width = aux.width();
    aux.values[2 * aux_width + 1] += EF::ONE;

    let mut builder = MockAirBuilder::new(main, aux, challenges, vec![terminal.unwrap().0]);
    for r in 0..builder.height {
        builder.for_row(r);
        gadget.eval_all(&mut builder, &lookups);
    }
}

#[test]
#[should_panic(expected = "Lookup mismatch")]
fn debug_util_detects_unbalanced_multiset() {
    use crate::debug_util::*;

    // Two-row main trace [3, 4]; lookup pushes +1 multiplicity for each.
    let main_values = vec![F::from_u32(3), F::from_u32(4)];
    let main_trace = RowMajorMatrix::new(main_values, 1);

    let builder = SymbolicAirBuilder::<F>::new(AirLayout {
        main_width: 1,
        ..Default::default()
    });
    let expr = builder.main().current(0).unwrap();

    // A single local lookup with one tuple per row and multiplicity +1 always.
    //
    // No negative side: the multiset is unbalanced.
    let lookup = Lookup {
        kind: Kind::Local,
        elements: vec![vec![SymbolicExpression::Leaf(BaseLeaf::Variable(expr))]],
        multiplicities: vec![SymbolicExpression::Leaf(BaseLeaf::Constant(F::ONE))],
        column: 0,
    };

    let instance: LookupDebugInstance<'_, F> = LookupDebugInstance {
        main_trace: &main_trace,
        preprocessed_trace: &None,
        public_values: &[],
        lookups: &[lookup],
        permutation_challenges: &[],
    };
    check_lookups(&[instance]);
}

#[test]
fn eval_all_global_lookup_carries_terminal_through_permutation_values() {
    // Two AIRs share a bus: one sends, one receives.
    //
    // Each AIR's terminal is non-zero individually; the cross-AIR sum is zero.
    let height = 4;
    let bus = "ab".to_string();

    // Sender AIR main trace: send (v) with multiplicity 1.
    //
    // Receiver AIR main trace: receive (v) with multiplicity 1.
    let mut sender_flat = Vec::with_capacity(height);
    let mut receiver_flat = Vec::with_capacity(height);
    for i in 0..height {
        sender_flat.push(F::from_u32(i as u32 + 1));
        receiver_flat.push(F::from_u32(i as u32 + 1));
    }
    let sender_main = RowMajorMatrix::new(sender_flat, 1);
    let receiver_main = RowMajorMatrix::new(receiver_flat, 1);

    // Build a global lookup record for the sender AIR.
    //
    // Sender uses count = -1; receiver uses count = +1.
    let builder_s = SymbolicAirBuilder::<F>::new(AirLayout {
        main_width: 1,
        ..Default::default()
    });
    let expr_s = builder_s.main().current(0).unwrap();
    let sender_lookup = Lookup {
        kind: Kind::Global(bus.clone()),
        elements: vec![vec![SymbolicExpression::Leaf(BaseLeaf::Variable(expr_s))]],
        multiplicities: vec![SymbolicExpression::Mul {
            x: Arc::new(SymbolicExpression::Leaf(BaseLeaf::Constant(-F::ONE))),
            y: Arc::new(SymbolicExpression::Leaf(BaseLeaf::Constant(F::ONE))),
            degree_multiple: 0,
        }],
        column: 0,
    };

    let builder_r = SymbolicAirBuilder::<F>::new(AirLayout {
        main_width: 1,
        ..Default::default()
    });
    let expr_r = builder_r.main().current(0).unwrap();
    let receiver_lookup = Lookup {
        kind: Kind::Global(bus),
        elements: vec![vec![SymbolicExpression::Leaf(BaseLeaf::Variable(expr_r))]],
        multiplicities: vec![SymbolicExpression::Leaf(BaseLeaf::Constant(F::ONE))],
        column: 0,
    };

    let gadget = LogUpGadget::new();

    // Both AIRs share the same (alpha, beta) for the same bus.
    let challenges = vec![EF::from_u32(0x1234), EF::from_u32(0x5678)];

    let (sender_aux, sender_terminal) = gadget.generate_permutation::<TestConfig>(
        &sender_main,
        &None,
        &[],
        core::slice::from_ref(&sender_lookup),
        &challenges,
    );
    let (receiver_aux, receiver_terminal) = gadget.generate_permutation::<TestConfig>(
        &receiver_main,
        &None,
        &[],
        core::slice::from_ref(&receiver_lookup),
        &challenges,
    );

    let sender_terminal = sender_terminal.unwrap();
    let receiver_terminal = receiver_terminal.unwrap();

    // Individually the AIR terminals are random-looking EF values.
    //
    // Together they cancel: cross-AIR sum = 0.
    assert!(
        gadget
            .verify_terminal_sum(&[Some(sender_terminal), Some(receiver_terminal)])
            .is_ok()
    );

    // Verify per-row constraints hold for the sender.
    let mut s_builder = MockAirBuilder::new(
        sender_main,
        sender_aux,
        challenges.clone(),
        vec![sender_terminal.0],
    );
    for r in 0..s_builder.height {
        s_builder.for_row(r);
        gadget.eval_all(&mut s_builder, core::slice::from_ref(&sender_lookup));
    }

    // And for the receiver.
    let mut r_builder = MockAirBuilder::new(
        receiver_main,
        receiver_aux,
        challenges,
        vec![receiver_terminal.0],
    );
    for r in 0..r_builder.height {
        r_builder.for_row(r);
        gadget.eval_all(&mut r_builder, core::slice::from_ref(&receiver_lookup));
    }
}

#[test]
fn empty_lookups_produce_no_permutation_trace() {
    let gadget = LogUpGadget::new();
    let main = RowMajorMatrix::new(F::zero_vec(4), 1);
    let (aux, terminal) = gadget.generate_permutation::<TestConfig>(&main, &None, &[], &[], &[]);
    assert_eq!(aux.width(), 0);
    assert_eq!(aux.values.len(), 0);
    assert!(terminal.is_none());
}

// `SymbolicExpression::resolve` against a concrete row evaluator.
//
// Independent of the lookup logic: exercises symbolic-to-concrete
// expression evaluation with first-row / transition / last-row selectors.
#[test]
fn test_symbolic_to_expr() {
    let mut builder = SymbolicAirBuilder::<F>::new(AirLayout {
        main_width: 2,
        ..Default::default()
    });

    let main = builder.main();
    let (local, next) = (main.current_slice(), main.next_slice());

    let mul = local[0] * next[1];
    let add = local[0] + next[1];
    let sub = local[0] - next[1];
    builder.when_first_row().assert_zero(mul.clone() * add);
    builder.when_transition().assert_zero(sub - local[0]);
    builder.when_last_row().assert_zero(mul - local[0]);

    let constraints = builder.base_constraints();

    let main_flat = vec![
        F::new(10),
        F::new(10),
        F::new(256),
        F::new(255),
        F::new(42),
        F::new(42),
    ];

    let main_trace = RowMajorMatrix::new(main_flat, 2);

    let perm = RowMajorMatrix::new(vec![], 0);
    let mut builder = MockAirBuilder::new(main_trace.clone(), perm, vec![], vec![]);

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
        let first_eval = constraints[0].resolve(&builder);
        let transition_eval = constraints[1].resolve(&builder);
        let last_eval = constraints[2].resolve(&builder);

        // Assert that the evaluated constraints are correct.
        assert_eq!(first_expected_val, first_eval.into());
        assert_eq!(transition_expected_val, transition_eval.into());
        assert_eq!(last_expected_val, last_eval.into());
    }
}

// Multiset with a non-identity permutation of values across rows.
//
// Reads {3×2, 5×4, 7×2} are spread across rows in a different order than
// the table rows that provide them. The single-terminal must still come
// out to zero — exercises the LogUp identity beyond the trivial case
// where `read[r] == provide[r]` on every row.
//
//     row | read | provide | mult | contribution
//      0  |   7  |    3    |   2  | 1/(α-7) - 2/(α-3)
//      1  |   3  |    5    |   4  | 1/(α-3) - 4/(α-5)
//      2  |   5  |    7    |   2  | 1/(α-5) - 2/(α-7)
//      3  |   3  |    3    |   0  | 1/(α-3) - 0
//      4  |   7  |    5    |   0  | 1/(α-7) - 0
//      5  |   5  |    5    |   0  | 1/(α-5) - 0
//      6  |   5  |    7    |   0  | 1/(α-5) - 0
//      7  |   5  |    5    |   0  | 1/(α-5) - 0
//
// Read multiset:  {3×2, 5×4, 7×2}
// Table multiset: {3×2, 5×4, 7×2}  (provides weighted by mult)
// Difference:     0  →  terminal must be zero.
#[test]
fn test_nontrivial_permutation() {
    let main_flat = vec![
        F::new(7),
        F::new(3),
        F::TWO,
        F::new(3),
        F::new(5),
        F::from_u8(4),
        F::new(5),
        F::new(7),
        F::TWO,
        F::new(3),
        F::new(3),
        F::ZERO,
        F::new(7),
        F::new(5),
        F::ZERO,
        F::new(5),
        F::new(5),
        F::ZERO,
        F::new(5),
        F::new(7),
        F::ZERO,
        F::new(5),
        F::new(5),
        F::ZERO,
    ];
    let main_trace = RowMajorMatrix::new(main_flat, 3);

    let air = RangeCheckAir::new();
    let lookups: Lookups<F> = Lookups::from_air::<EF, _>(&air);
    let gadget = LogUpGadget::new();
    let challenges = vec![EF::from_u32(0x1234_5678), EF::from_u32(0x9abc_def0)];

    let (aux, terminal) =
        gadget.generate_permutation::<TestConfig>(&main_trace, &None, &[], &lookups, &challenges);
    let terminal = terminal.unwrap();

    // Balanced multiset → terminal is zero with overwhelming probability.
    assert_eq!(terminal.0, EF::ZERO);

    // Per-row constraints all hold.
    let mut builder = MockAirBuilder::new(main_trace, aux, challenges, vec![terminal.0]);
    for r in 0..builder.height {
        builder.for_row(r);
        gadget.eval_all(&mut builder, &lookups);
    }
}

// Imbalance via a zero-multiplicity row.
//
// Row 0 reads `10` but the table provides `10` with multiplicity 0 — i.e.
// the prover never provides `10`. The multiset is therefore unbalanced
// and the AIR's terminal is non-zero, so the cross-AIR check rejects.
#[test]
fn test_zero_multiplicity_is_not_counted() {
    let main_flat = vec![
        // Read 10, provide 10 with mult 0 — the read has no provider.
        F::new(10),
        F::new(10),
        F::ZERO,
        // A valid row to make the imbalance survive past row 0.
        F::new(20),
        F::new(20),
        F::ONE,
    ];
    let main_trace = RowMajorMatrix::new(main_flat, 3);

    let air = RangeCheckAir::new();
    let lookups: Lookups<F> = Lookups::from_air::<EF, _>(&air);
    let gadget = LogUpGadget::new();
    let challenges = vec![EF::from_u8(123), EF::from_u8(111)];

    let (_aux, terminal) =
        gadget.generate_permutation::<TestConfig>(&main_trace, &None, &[], &lookups, &challenges);
    let terminal = terminal.unwrap();

    // The unread `10` survives into the terminal.
    assert_ne!(terminal.0, EF::ZERO);

    // Cross-AIR check rejects.
    match gadget.verify_terminal_sum(&[Some(terminal)]) {
        Err(LookupError::TerminalSumNonZero) => {}
        other => panic!("expected TerminalSumNonZero, got {other:?}"),
    }
}

// Corruption of the accumulator column breaks the transition constraint.
//
// The prover generates a valid aux trace, then we tamper with one cell of
// the shared accumulator column. The transition `acc[r+1] - acc[r] - Σ frac_c[r] = 0`
// fails at the row immediately before the tampered cell.
#[test]
#[should_panic(expected = "Extension constraint failed")]
fn test_inconsistent_witness_fails_transition() {
    let main_flat = vec![
        F::new(10),
        F::new(10),
        F::ONE,
        F::new(20),
        F::new(20),
        F::ONE,
        F::new(30),
        F::new(30),
        F::ONE,
        F::new(40),
        F::new(40),
        F::ONE,
    ];
    let main_trace = RowMajorMatrix::new(main_flat, 3);

    let air = RangeCheckAir::new();
    let lookups: Lookups<F> = Lookups::from_air::<EF, _>(&air);
    let gadget = LogUpGadget::new();
    let challenges = vec![EF::from_u32(31), EF::from_u32(41)];

    let (mut aux, terminal) =
        gadget.generate_permutation::<TestConfig>(&main_trace, &None, &[], &lookups, &challenges);

    // Corrupt acc[2] (accumulator column, row 2) by injecting a non-zero delta.
    //
    // - The transition r=1 → r=2 reads acc[2] as `acc_next`, so it fails first.
    let aux_width = aux.width();
    aux.values[2 * aux_width] += EF::from_u8(99);

    let mut builder = MockAirBuilder::new(main_trace, aux, challenges, vec![terminal.unwrap().0]);
    for r in 0..builder.height {
        builder.for_row(r);
        gadget.eval_all(&mut builder, &lookups);
    }
}

// AIR doing tuple lookups:
// Each row carries the read tuple, the table tuple and a multiplicity.
//
// Main trace layout per row: `[in1, in2, sum, t_in1, t_in2, t_sum, mult]`.
//
// Tuples of size > 1 force the Horner combiner's `β` challenge to participate meaningfully;
// Single-element tuples leave it inactive.
struct AddAir;

impl AddAir {
    /// Build the AIR for a 3-element tuple lookup over binary additions.
    const fn new() -> Self {
        Self
    }
}

impl<F: Field> BaseAir<F> for AddAir {
    fn width(&self) -> usize {
        7
    }
}

impl<AB> Air<AB> for AddAir
where
    AB: AirBuilder<F: Field> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();

        let in1 = local[0];
        let in2 = local[1];
        let sum = local[2];
        let t_in1 = local[3];
        let t_in2 = local[4];
        let t_sum = local[5];
        let mult = local[6];

        builder.push_local_interaction(vec![
            // Read side: (in1, in2, sum) with multiplicity +1.
            (vec![in1.into(), in2.into(), sum.into()], AB::Expr::ONE),
            // Table side: (t_in1, t_in2, t_sum) with multiplicity -mult.
            (
                vec![t_in1.into(), t_in2.into(), t_sum.into()],
                -(mult.into()),
            ),
        ]);
    }
}

// Lookup over 3-element tuples.
//
// Read multiset: { (0,1,1) ×2, (1,1,2) ×1, (0,0,0) ×1 }
// Table provides (weighted by `mult`): same multiset reordered.
//
// The 3-element payload forces the Horner combiner to actually use the
// `β` challenge — single-element tuples leave `β` inactive.
#[test]
fn test_tuple_lookup() {
    let main_flat = vec![
        // [in1, in2, sum, t_in1, t_in2, t_sum, mult]
        F::new(0),
        F::new(1),
        F::new(1),
        F::new(0),
        F::new(1),
        F::new(1),
        F::TWO,
        F::new(0),
        F::new(1),
        F::new(1),
        F::new(0),
        F::new(0),
        F::new(0),
        F::ONE,
        F::new(1),
        F::new(1),
        F::TWO,
        F::new(1),
        F::new(0),
        F::new(1),
        F::ZERO,
        F::new(0),
        F::new(0),
        F::new(0),
        F::new(1),
        F::new(1),
        F::TWO,
        F::ONE,
    ];
    let main_trace = RowMajorMatrix::new(main_flat, 7);

    let air = AddAir::new();
    let lookups: Lookups<F> = Lookups::from_air::<EF, _>(&air);
    let gadget = LogUpGadget::new();
    let challenges = vec![EF::from_u32(31), EF::from_u32(41)];

    let (aux, terminal) =
        gadget.generate_permutation::<TestConfig>(&main_trace, &None, &[], &lookups, &challenges);
    let terminal = terminal.unwrap();

    // Balanced multiset → terminal zero.
    assert_eq!(terminal.0, EF::ZERO);

    // Per-row constraints all hold.
    let mut builder = MockAirBuilder::new(main_trace, aux, challenges, vec![terminal.0]);
    for r in 0..builder.height {
        builder.for_row(r);
        gadget.eval_all(&mut builder, &lookups);
    }
}
