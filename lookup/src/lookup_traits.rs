use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Neg;

use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, ExtensionBuilder, PairBuilder,
    PermutationAirBuilder,
};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::ViewPair;
use p3_uni_stark::{Entry, LookupError, StarkGenericConfig, SymbolicExpression, Val};
use serde::{Deserialize, Serialize};
use tracing::warn;

/// Data required for global lookup arguments in a multi-STARK proof.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct LookupData<F: Clone> {
    /// Name of the global lookup interaction.
    pub name: String,
    /// Index of the auxiliary column (if there are multiple auxiliary columns, this is the first one)
    pub aux_idx: usize,
    /// Expected cumulated value for a global lookup argument.
    pub expected_cumulated: F,
}

impl<F: Field> LookupData<F> {
    pub fn to_symbolic(&self) -> LookupData<SymbolicExpression<F>> {
        let expected = SymbolicExpression::Constant(self.expected_cumulated);
        LookupData {
            name: self.name.clone(),
            aux_idx: self.aux_idx,
            expected_cumulated: expected,
        }
    }
}

/// A trait for lookup argument.
pub trait LookupGadget {
    /// Returns the number of auxiliary columns needed by this lookup protocol.
    ///
    /// For example:
    /// - LogUp needs 1 column (running sum)
    fn num_aux_cols(&self) -> usize;

    /// Returns the number of challenges for each lookup argument.
    ///
    /// For example, for LogUp, this is 2:
    /// - one challenge for combining the lookup tuples,
    /// - one challenge for the running sum.
    fn num_challenges(&self) -> usize;

    /// Evaluates a local lookup argument based on the provided context.
    ///
    /// For example, in LogUp:
    /// - this checks that the running sum is updated correctly.
    /// - it checks that the final value of the running sum is 0.
    fn eval_local_lookup<AB>(&self, builder: &mut AB, context: &Lookup<AB::F>)
    where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues;

    /// Evaluates a global lookup update based on the provided context, and the expected cumulated value.
    /// This evaluation is carried out at the AIR level. We still need to check that the permutation argument holds
    /// over all AIRs involved in the interaction.
    ///
    /// For example, in LogUp:
    /// - this checks that the running sum is updated correctly.
    /// - it checks that the local final value of the running sum is equal to the value provided by the prover.
    fn eval_global_update<AB>(
        &self,
        builder: &mut AB,
        context: &Lookup<AB::F>,
        expected_cumulated: AB::ExprEF,
    ) where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues;

    /// Evalutes the lookup constraints for all provided contexts.
    ///
    /// For each context:
    /// - if it is a local lookup, evaluates it with `eval_local_lookup`.
    /// - if it is a global lookup, evaluates it with `eval_global_update`, using the expected cumulated value from `lookup_data`.
    fn eval_lookups<AB>(
        &self,
        builder: &mut AB,
        contexts: &[Lookup<AB::F>],
        // Assumed to be sorted by auxiliary_index.
        lookup_data: &[LookupData<AB::EF>],
    ) where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    {
        let mut lookup_data_iter = lookup_data.iter();
        for context in contexts.iter() {
            match &context.kind {
                Kind::Local => {
                    self.eval_local_lookup(builder, context);
                }
                Kind::Global(_) => {
                    // Find the expected cumulated value for this context.
                    let LookupData {
                        name: _,
                        aux_idx,
                        expected_cumulated,
                    } = lookup_data_iter
                        .next()
                        .expect("Expected cumulated value missing");

                    if *aux_idx != context.columns[0] {
                        panic!("Expected cumulated values not sorted by auxiliary index");
                    }
                    let expr_ef_expected = AB::ExprEF::from(*expected_cumulated);
                    self.eval_global_update(builder, context, expr_ef_expected);
                }
            }
        }
        assert!(
            lookup_data_iter.next().is_none(),
            "Too many expected cumulated values provided"
        );
    }

    /// Generates the permutation matrix for the lookup argument.
    fn generate_permutation<SC: StarkGenericConfig>(
        &self,
        main: &RowMajorMatrix<Val<SC>>,
        preprocessed: &Option<RowMajorMatrix<Val<SC>>>,
        public_values: &[Val<SC>],
        lookups: &[Lookup<Val<SC>>],
        lookup_data: &mut [LookupData<SC::Challenge>],
        permutation_challenges: &[SC::Challenge],
    ) -> RowMajorMatrix<SC::Challenge>;

    /// Evaluates the final cumulated value over all AIRs involved in the interaction,
    /// and checks that it is equal to the expected final value.
    ///
    /// For example, in LogUp:
    /// - it sums all expected cumulated values provided by each AIR within one interaction,
    /// - checks that the sum is equal to 0.
    fn verify_global_final_value<EF: Field>(
        &self,
        all_expected_cumulated: &[EF],
    ) -> Result<(), LookupError>;

    /// Computes the polynomial degree of a lookup transition constraint.
    fn constraint_degree<F: Field>(&self, context: Lookup<F>) -> usize;
}

/// Specifies whether a lookup is local to an AIR or part of a global interaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// A lookup where all entries are contained within a single AIR.
    Local,
    /// A lookup that spans multiple AIRs, identified by a unique interaction name.
    ///
    /// The interaction name is used to identify all elements that are part of the same interaction.
    Global(String),
}

/// Indicates the direction of data flow in a global lookup.
#[derive(Clone, Copy)]
pub enum Direction {
    /// Indicates that elements are being sent (contributed) to the lookup.
    Send,
    /// Indicates that elements are being received (removed) from the lookup.
    Receive,
}

impl Direction {
    /// Helper method to compute the signed multiplicity based on the direction.
    pub fn multiplicity<T: Neg<Output = T>>(&self, mult: T) -> T {
        match self {
            Self::Send => -mult,
            Self::Receive => mult,
        }
    }
}

/// A type alias for a lookup input tuple. It contains:
/// - a vector of symbolic expressions representing the elements involved in the lookup,
/// - a symbolic expression representing the multiplicity of the lookup,
/// - a direction indicating whether the elements are being sent or received.
pub type LookupInput<F> = (Vec<SymbolicExpression<F>>, SymbolicExpression<F>, Direction);

/// A structure that holds the lookup data necessary to generate lookup contexts via [`LookupTraceBuilder`]. It is shared between the prover and the verifier.
#[derive(Clone, Debug)]
pub struct Lookup<F: Field> {
    /// Type of lookup: local or global
    pub kind: Kind,
    /// Elements being read (consumed from the table). Each `Vec<SymbolicExpression<F>>` actually represents a tuple of elements that are bundled together to make one lookup.
    pub element_exprs: Vec<Vec<SymbolicExpression<F>>>,
    /// Multiplicities for the elements.
    /// Note that Lagrange selectors may not be normalized, and so cannot be used as proper filters in the multiplicities.
    pub multiplicities_exprs: Vec<SymbolicExpression<F>>,
    /// The column index in the permutation trace for this lookup's running sum
    pub columns: Vec<usize>,
}

impl<F: Field> Lookup<F> {
    /// Creates a new lookup with the specified column.
    ///
    /// # Arguments
    /// * `elements` - Elements from the either the main execution trace or a lookup table.
    /// * `multiplicities` - How many times each `element` should appear
    /// * `column` - The column index in the permutation trace for this lookup
    pub const fn new(
        kind: Kind,
        element_exprs: Vec<Vec<SymbolicExpression<F>>>,
        multiplicities_exprs: Vec<SymbolicExpression<F>>,
        columns: Vec<usize>,
    ) -> Self {
        Self {
            kind,
            element_exprs,
            multiplicities_exprs,
            columns,
        }
    }
}

/// A trait for an AIR that handles lookup arguments.
pub trait AirLookupHandler<AB>: Air<AB>
where
    AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
{
    /// Register a lookup to be used in this AIR.
    /// This method can be used before proving or verifying, as the resulting data is shared between the prover and the verifier.
    fn register_lookup(
        &mut self,
        kind: Kind,
        lookup_inputs: &[LookupInput<AB::F>],
    ) -> Lookup<AB::F> {
        let (element_exprs, multiplicities_exprs) = lookup_inputs
            .iter()
            .map(|(elems, mult, dir)| {
                let multiplicity = dir.multiplicity(mult.clone());
                (elems.clone(), multiplicity)
            })
            .unzip();

        Lookup {
            kind,
            element_exprs,
            multiplicities_exprs,
            columns: self.add_lookup_columns(),
        }
    }

    /// Updates the number of auxiliary columns to account for a new lookup column, and returns its index (or indices).
    fn add_lookup_columns(&mut self) -> Vec<usize>;

    /// Register all lookups for the current AIR and return them.
    fn get_lookups(&mut self) -> Vec<Lookup<AB::F>>;

    /// Evaluates all AIR and lookup constraints.
    fn eval<LG: LookupGadget>(
        &self,
        builder: &mut AB,
        lookups: &[Lookup<AB::F>],
        lookup_data: &[LookupData<AB::EF>],
        lookup_gadget: &LG,
    ) {
        Air::<AB>::eval(self, builder);

        if !lookups.is_empty() {
            lookup_gadget.eval_lookups(builder, lookups, lookup_data);
        }
    }
}

/// A builder to generate the lookup traces, given the main trace, public values and permutation challenges.
pub struct LookupTraceBuilder<'a, SC: StarkGenericConfig> {
    main: ViewPair<'a, Val<SC>>,
    preprocessed: Option<ViewPair<'a, Val<SC>>>,
    public_values: &'a [Val<SC>],
    permutation_challenges: &'a [SC::Challenge],
    height: usize,
    row: usize,
}

impl<'a, SC: StarkGenericConfig> LookupTraceBuilder<'a, SC> {
    pub const fn new(
        main: ViewPair<'a, Val<SC>>,
        preprocessed: Option<ViewPair<'a, Val<SC>>>,
        public_values: &'a [Val<SC>],
        permutation_challenges: &'a [SC::Challenge],
        height: usize,
        row: usize,
    ) -> Self {
        Self {
            main,
            preprocessed,
            public_values,
            permutation_challenges,
            height,
            row,
        }
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for LookupTraceBuilder<'a, SC> {
    type F = Val<SC>;
    type Expr = Val<SC>;
    type Var = Val<SC>;
    type M = ViewPair<'a, Val<SC>>;

    #[inline]
    fn main(&self) -> Self::M {
        self.main
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        Self::F::from_bool(self.row == 0)
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        Self::F::from_bool(self.row + 1 == self.height)
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            Self::F::from_bool(self.row + 1 < self.height)
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        assert!(x.into() == Self::F::ZERO);
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        for item in array {
            assert!(item.into() == Self::F::ZERO);
        }
    }
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues for LookupTraceBuilder<'_, SC> {
    type PublicVar = Val<SC>;

    #[inline]
    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<SC: StarkGenericConfig> PairBuilder for LookupTraceBuilder<'_, SC> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
            .unwrap_or_else(|| panic!("Missing preprocessed columns"))
    }
}

impl<SC: StarkGenericConfig> ExtensionBuilder for LookupTraceBuilder<'_, SC> {
    type EF = SC::Challenge;
    type ExprEF = SC::Challenge;
    type VarEF = SC::Challenge;

    fn assert_zero_ext<I: Into<Self::ExprEF>>(&mut self, x: I) {
        assert!(x.into() == SC::Challenge::ZERO);
    }
}

impl<'a, SC: StarkGenericConfig> PermutationAirBuilder for LookupTraceBuilder<'a, SC> {
    type MP = RowMajorMatrixView<'a, SC::Challenge>;

    type RandomVar = SC::Challenge;
    fn permutation(&self) -> RowMajorMatrixView<'a, SC::Challenge> {
        panic!("we should not be accessing the permutation matrix while building it");
    }

    fn permutation_randomness(&self) -> &[SC::Challenge] {
        self.permutation_challenges
    }
}

// /// Evaluates a symbolic expression in the context of an AIR builder.
///
/// Converts `SymbolicExpression<F>` to the builder's expression type `AB::Expr`.
pub fn symbolic_to_expr<AB>(builder: &AB, expr: &SymbolicExpression<AB::F>) -> AB::Expr
where
    AB: PairBuilder + AirBuilderWithPublicValues + PermutationAirBuilder,
{
    match expr {
        SymbolicExpression::Variable(v) => match v.entry {
            Entry::Main { offset } => match offset {
                0 => builder.main().row_slice(0).unwrap()[v.index].clone().into(),
                1 => builder.main().row_slice(1).unwrap()[v.index].clone().into(),
                _ => panic!("Cannot have expressions involving more than two rows."),
            },
            Entry::Public => builder.public_values()[v.index].into(),
            Entry::Preprocessed { offset } => match offset {
                0 => builder.preprocessed().row_slice(0).unwrap()[v.index]
                    .clone()
                    .into(),
                1 => builder.preprocessed().row_slice(1).unwrap()[v.index]
                    .clone()
                    .into(),
                _ => panic!("Cannot have expressions involving more than two rows."),
            },
            _ => unimplemented!("Entry type {:?} not supported in interactions", v.entry),
        },
        SymbolicExpression::IsFirstRow => {
            warn!("IsFirstRow is not normalized");
            builder.is_first_row()
        }
        SymbolicExpression::IsLastRow => {
            warn!("IsLastRow is not normalized");
            builder.is_last_row()
        }
        SymbolicExpression::IsTransition => {
            warn!("IsTransition is not normalized");
            builder.is_transition_window(2)
        }
        SymbolicExpression::Constant(c) => AB::Expr::from(*c),
        SymbolicExpression::Add { x, y, .. } => {
            symbolic_to_expr(builder, x) + symbolic_to_expr(builder, y)
        }
        SymbolicExpression::Sub { x, y, .. } => {
            symbolic_to_expr(builder, x) - symbolic_to_expr(builder, y)
        }
        SymbolicExpression::Neg { x, .. } => -symbolic_to_expr(builder, x),
        SymbolicExpression::Mul { x, y, .. } => {
            symbolic_to_expr(builder, x) * symbolic_to_expr(builder, y)
        }
    }
}

/// Wrapper around AIRs for AIRs that do not use lookups.
#[derive(Clone)]
pub struct AirNoLookup<A> {
    air: A,
}

impl<A> AirNoLookup<A> {
    pub const fn new(air: A) -> Self {
        Self { air }
    }
}

impl<F, A: BaseAir<F>> BaseAir<F> for AirNoLookup<A> {
    fn width(&self) -> usize {
        self.air.width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.air.preprocessed_trace()
    }
}

impl<AB: AirBuilder, A: Air<AB>> Air<AB> for AirNoLookup<A> {
    fn eval(&self, builder: &mut AB) {
        self.air.eval(builder);
    }
}

impl<AB: AirBuilderWithPublicValues + PairBuilder + PermutationAirBuilder, A: Air<AB>>
    AirLookupHandler<AB> for AirNoLookup<A>
{
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        vec![]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<AB::F>> {
        vec![]
    }
}

/// Empty lookup gadget for AIRs that do not use lookups.
pub struct EmptyLookupGadget;
impl LookupGadget for EmptyLookupGadget {
    fn num_aux_cols(&self) -> usize {
        0
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn eval_local_lookup<AB>(&self, _builder: &mut AB, _context: &Lookup<AB::F>)
    where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    {
    }

    fn eval_global_update<AB>(
        &self,
        _builder: &mut AB,
        _context: &Lookup<AB::F>,
        _expected_cumulated: AB::ExprEF,
    ) where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    {
    }

    fn verify_global_final_value<EF: Field>(
        &self,
        _all_expected_cumulated: &[EF],
    ) -> Result<(), LookupError> {
        Ok(())
    }

    fn constraint_degree<F: Field>(&self, _context: Lookup<F>) -> usize {
        0
    }

    fn generate_permutation<SC: StarkGenericConfig>(
        &self,
        _main: &RowMajorMatrix<Val<SC>>,
        _preprocessed: &Option<RowMajorMatrix<Val<SC>>>,
        _public_values: &[Val<SC>],
        _lookups: &[Lookup<Val<SC>>],
        _lookup_data: &mut [LookupData<SC::Challenge>],
        _permutation_challenges: &[SC::Challenge],
    ) -> RowMajorMatrix<SC::Challenge> {
        RowMajorMatrix::new(vec![], 0)
    }
}
