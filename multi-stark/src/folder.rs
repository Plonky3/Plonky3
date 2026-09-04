//! AIR constraint folder for the multilinear prover.
//!
//! - Implements the standard AIR builder interface, so any AIR runs unchanged through the folder.
//! - Drives both directions of the protocol with the same evaluator:
//!   - the prover walks the boolean hypercube row by row,
//!   - the verifier evaluates at the random sumcheck challenge.

use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, BaseAir, RowWindow};
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing, dot_product};
use p3_lookup::{Count, InteractionBuilder, InteractionSymbolicBuilder};

use crate::lookup::AirLinkInstance;
use crate::packed_ext::PackedExt;
use crate::selectors::BoundaryEvals;

/// An AIR the multilinear verifier can evaluate.
///
/// The verifier works at a single random point.
/// It therefore needs only the scalar extension-field instantiation of each folder.
/// One symbolic pass on top of that gives it the AIR's degrees and lookup declarations.
pub trait VerifierAir<F, EF>:
    BaseAir<F>
    + Air<InteractionSymbolicBuilder<F, EF>>
    + for<'a> Air<MultilinearFolder<'a, F, EF, EF>>
    + for<'a> Air<InteractionMultilinearFolder<'a, F, EF, EF>>
where
    F: Field,
    EF: ExtensionField<F>,
{
}

impl<F, EF, A> VerifierAir<F, EF> for A
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F>
        + Air<InteractionSymbolicBuilder<F, EF>>
        + for<'a> Air<MultilinearFolder<'a, F, EF, EF>>
        + for<'a> Air<InteractionMultilinearFolder<'a, F, EF, EF>>,
{
}

/// An AIR the multilinear prover can evaluate.
///
/// The prover sweeps the hypercube, so it needs every width the sumcheck passes through:
///
/// ```text
///     round 0, scalar : base-field rows
///     round 0, packed : one SIMD lane group of base-field rows
///     later,   scalar : extension-field rows
///     later,   packed : one SIMD lane group of extension-field rows
/// ```
///
/// Each width appears twice, once for ordinary constraints and once for lookup links.
pub trait ProverAir<F, EF>:
    VerifierAir<F, EF>
    + for<'a> Air<MultilinearFolder<'a, F, F, EF>>
    + for<'a> Air<MultilinearFolder<'a, F, F::Packing, EF::ExtensionPacking>>
    + for<'a> Air<
        MultilinearFolder<
            'a,
            F,
            PackedExt<F, EF::ExtensionPacking>,
            PackedExt<F, EF::ExtensionPacking>,
        >,
    > + for<'a> Air<InteractionMultilinearFolder<'a, F, F, EF>>
    + for<'a> Air<InteractionMultilinearFolder<'a, F, F::Packing, EF::ExtensionPacking>>
    + for<'a> Air<
        InteractionMultilinearFolder<
            'a,
            F,
            PackedExt<F, EF::ExtensionPacking>,
            PackedExt<F, EF::ExtensionPacking>,
        >,
    >
where
    F: Field,
    EF: ExtensionField<F>,
{
}

impl<F, EF, A> ProverAir<F, EF> for A
where
    F: Field,
    EF: ExtensionField<F>,
    A: VerifierAir<F, EF>
        + for<'a> Air<MultilinearFolder<'a, F, F, EF>>
        + for<'a> Air<MultilinearFolder<'a, F, F::Packing, EF::ExtensionPacking>>
        + for<'a> Air<
            MultilinearFolder<
                'a,
                F,
                PackedExt<F, EF::ExtensionPacking>,
                PackedExt<F, EF::ExtensionPacking>,
            >,
        > + for<'a> Air<InteractionMultilinearFolder<'a, F, F, EF>>
        + for<'a> Air<InteractionMultilinearFolder<'a, F, F::Packing, EF::ExtensionPacking>>
        + for<'a> Air<
            InteractionMultilinearFolder<
                'a,
                F,
                PackedExt<F, EF::ExtensionPacking>,
                PackedExt<F, EF::ExtensionPacking>,
            >,
        >,
{
}

/// The two independently batched expression families emitted by one AIR evaluation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FolderEvaluations<Acc> {
    /// Alpha-batched ordinary AIR constraints.
    pub(crate) constraints: Acc,
    /// Lookup-link expressions accumulated with their static GKR coefficients.
    pub(crate) interactions: Acc,
}

/// Folder shared by the prover and the verifier.
#[derive(Debug)]
pub struct MultilinearFolder<'a, F, Var, Acc> {
    /// Two-row main window holding the current and shifted-by-one rows.
    ///
    /// The shifted row carries zero in its last position.
    pub main_window: RowWindow<'a, Var>,
    /// Boundary-selector values shared by all selector accessors.
    pub boundary: BoundaryEvals<Var>,
    /// Public inputs forwarded to the AIR, always in the base field.
    pub public_values: &'a [F],
    /// Random scalar driving alpha-batching of constraints.
    pub alpha: Acc,
    /// Running alpha-batched accumulator capturing every asserted-zero constraint.
    pub accumulator: Acc,
    /// Two-row preprocessed window; zero-width when the AIR has no preprocessed columns.
    pub preprocessed_window: RowWindow<'a, Var>,
    /// Periodic column values at the current evaluation point, one per declared periodic column.
    ///
    /// Empty when the AIR declares no periodic columns.
    pub periodic_values: &'a [Var],
}

impl<'a, F, Var, Acc> MultilinearFolder<'a, F, Var, Acc>
where
    Acc: PrimeCharacteristicRing,
{
    /// Build a folder for a single AIR evaluation.
    ///
    /// The preprocessed window and the periodic values start empty.
    /// An AIR that declares either kind of column attaches them with the builder methods below.
    ///
    /// # Arguments
    ///
    /// - `local`: column values at the current row.
    /// - `next`: column values at the shifted-by-one row.
    /// - `boundary`: selector values at the same evaluation point.
    /// - `public_values`: public inputs forwarded to the AIR.
    /// - `alpha`: random scalar driving constraint batching.
    #[inline]
    pub fn new(
        local: &'a [Var],
        next: &'a [Var],
        boundary: BoundaryEvals<Var>,
        public_values: &'a [F],
        alpha: Acc,
    ) -> Self {
        Self {
            // Pair the two borrowed rows into the window the AIR reads from.
            main_window: RowWindow::from_two_rows(local, next),
            boundary,
            public_values,
            alpha,
            // Zero-width preprocessed window covers AIRs without a preprocessed trace.
            accumulator: Acc::ZERO,
            preprocessed_window: RowWindow::from_two_rows(&[], &[]),
            // No periodic columns until attached.
            periodic_values: &[],
        }
    }

    /// Attach the two-row preprocessed window read by the AIR.
    ///
    /// # Arguments
    ///
    /// - `current`: preprocessed column values at the current row.
    /// - `next`: preprocessed column values at the shifted-by-one row.
    #[inline]
    #[must_use]
    pub fn with_preprocessed(mut self, current: &'a [Var], next: &'a [Var]) -> Self {
        self.preprocessed_window = RowWindow::from_two_rows(current, next);
        self
    }

    /// Attach the periodic column values read by the AIR.
    ///
    /// # Arguments
    ///
    /// - `periodic_values`: one entry per declared periodic column, in declaration order.
    #[inline]
    #[must_use]
    pub const fn with_periodic(mut self, periodic_values: &'a [Var]) -> Self {
        self.periodic_values = periodic_values;
        self
    }

    /// Consume the folder and return the alpha-batched accumulator.
    ///
    /// # Returns
    ///
    /// The Horner fold `sum_{i=0}^{n-1} alpha^(n - 1 - i) * C_i`, where:
    ///
    /// - `C_0, ..., C_{n-1}` are the constraints asserted by the AIR in declaration order.
    /// - `n` is the total number of asserted constraints.
    #[inline]
    #[must_use]
    pub fn into_accumulator(self) -> Acc {
        self.accumulator
    }

    /// Run the AIR through this folder and return its alpha-batched constraint value.
    ///
    /// This is the terminal step of the builder: it consumes the folder.
    /// Attach preprocessed and periodic columns before calling, if the AIR reads them.
    ///
    /// # Arguments
    ///
    /// - `air`: the AIR whose constraints are evaluated at this point.
    ///
    /// # Returns
    ///
    /// The Horner fold `sum_{i=0}^{n-1} alpha^(n - 1 - i) * C_i`, where:
    ///
    /// - `C_0, ..., C_{n-1}` are the constraints asserted by the AIR in declaration order.
    /// - `n` is the total number of asserted constraints.
    #[inline]
    #[must_use]
    pub fn eval_air<A>(mut self, air: &A) -> Acc
    where
        A: Air<Self>,
        Self: AirBuilder,
    {
        air.eval(&mut self);
        self.into_accumulator()
    }
}

/// Ordinary constraints only: every lookup declaration is dropped.
///
/// An AIR that declares lookups is written against the interaction builder interface.
/// The ordinary path must therefore still accept those declarations to run such an AIR at all.
///
/// Dropping them is the intended behaviour at exactly one place.
/// That place is an interpolation node past the lookup expressions' own degree.
/// Only the ordinary constraints still contribute there.
///
/// Everywhere else the lookup-aware adapter below is used instead.
///
/// A batch whose AIRs declare lookups but which carries no lookup link is rejected up front.
/// No proof is therefore ever produced or accepted with its lookups silently discarded.
impl<'a, F, Var, Acc> InteractionBuilder for MultilinearFolder<'a, F, Var, Acc>
where
    F: PrimeCharacteristicRing + Copy + Sync,
    Var: Algebra<F> + Copy + Send + Sync,
    Acc: Algebra<Var> + Copy,
{
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        _bus_name: &str,
        _fields: impl IntoIterator<Item = E>,
        _count: impl Into<Count<Self::Expr>>,
    ) {
    }

    fn push_local_interaction(
        &mut self,
        _tuples: impl IntoIterator<Item = (Vec<Self::Expr>, Count<Self::Expr>)>,
    ) {
    }
}

impl<'a, F, Var, Acc> AirBuilder for MultilinearFolder<'a, F, Var, Acc>
where
    F: PrimeCharacteristicRing + Copy + Sync,
    Var: Algebra<F> + Copy + Send + Sync,
    Acc: Algebra<Var> + Copy,
{
    type F = F;
    type Expr = Var;
    type Var = Var;
    type MainWindow = RowWindow<'a, Var>;
    type PreprocessedWindow = RowWindow<'a, Var>;
    // Public values stay in the base field and lift into the expression type on read.
    type PublicVar = F;
    type PeriodicVar = Var;

    #[inline]
    fn main(&self) -> Self::MainWindow {
        self.main_window
    }

    #[inline]
    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed_window
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.boundary.first
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.boundary.last
    }

    #[inline]
    fn is_transition(&self) -> Self::Expr {
        self.boundary.transition
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        // Horner alpha-batching: each push updates
        //
        //     accumulator := alpha * accumulator + C_i
        //
        // After `n` pushes the accumulator collapses to
        //
        //     C_0 * alpha^(n-1) + C_1 * alpha^(n-2) + ... + C_{n-1}.
        self.accumulator = self.accumulator * self.alpha + x.into();
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }

    #[inline]
    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.periodic_values
    }
}

/// Lookup-aware adapter over the ordinary multilinear AIR folder.
///
/// One AIR evaluation yields two independent values: the ordinary constraints and the lookup link.
///
/// The link rebuilds the already-opened fraction from the AIR's own columns:
///
/// ```text
///     link = sum_b w_b * (m_b + theta * (D_b - 1))
///     D_b  = bus_prefix - sum_k beta^k * payload_bk
/// ```
///
/// Here `b` ranges over the AIR's declared tuples.
/// `m_b` is the signed multiplicity.
/// `w_b` is the equality weight selecting the block that materialized tuple `b`.
#[derive(Debug)]
pub struct InteractionMultilinearFolder<'a, F, Var, Acc> {
    /// Ordinary folder handling every non-lookup builder method.
    inner: MultilinearFolder<'a, F, Var, Acc>,
    /// Per-tuple block weights and bus offsets for the AIR being evaluated.
    link: &'a AirLinkInstance<Acc>,
    /// Coefficient `theta * beta^k` applied to payload coordinate `k`.
    ///
    /// A payload narrower than this slice simply leaves the tail unused.
    theta_beta_powers: &'a [Acc],
    /// Running sum of the lookup-link expression over the tuples seen so far.
    interaction_accumulator: Acc,
    /// Whether asserted constraints reach the inner folder.
    ///
    /// False at interpolation nodes above the ordinary constraints' own degree.
    /// Only the lookup link still contributes there.
    constraints_enabled: bool,
    /// Number of intra-AIR lookups consumed, which indexes the next one.
    local_seen: usize,
    /// Number of cross-AIR lookups consumed, which indexes the next one.
    global_seen: usize,
}

impl<'a, F, Var, Acc> InteractionMultilinearFolder<'a, F, Var, Acc>
where
    Acc: PrimeCharacteristicRing,
{
    /// Wrap an ordinary folder so the same AIR evaluation also builds the lookup link.
    ///
    /// # Arguments
    ///
    /// - `inner`: folder already carrying the trace window and selectors.
    /// - `link`: block weights and bus offsets for this AIR, in declaration order.
    /// - `theta_beta_powers`: payload coefficients shared by every AIR in the batch.
    /// - `constraints_enabled`: whether to also accumulate the ordinary constraints.
    #[inline]
    pub(crate) const fn new(
        inner: MultilinearFolder<'a, F, Var, Acc>,
        link: &'a AirLinkInstance<Acc>,
        theta_beta_powers: &'a [Acc],
        constraints_enabled: bool,
    ) -> Self {
        Self {
            inner,
            link,
            theta_beta_powers,
            interaction_accumulator: Acc::ZERO,
            constraints_enabled,
            local_seen: 0,
            global_seen: 0,
        }
    }

    /// Run the AIR once and return its ordinary and lookup expressions separately.
    #[inline]
    #[must_use]
    pub(crate) fn eval_air<A>(mut self, air: &A) -> FolderEvaluations<Acc>
    where
        A: Air<Self>,
        Self: AirBuilder,
    {
        air.eval(&mut self);
        // Both families come out of the one pass, batched independently.
        FolderEvaluations {
            constraints: self.inner.accumulator,
            interactions: self.interaction_accumulator,
        }
    }
}

impl<'a, F, Var, Acc> AirBuilder for InteractionMultilinearFolder<'a, F, Var, Acc>
where
    F: PrimeCharacteristicRing + Copy + Sync,
    Var: Algebra<F> + Copy + Send + Sync,
    Acc: Algebra<Var> + Copy,
{
    type F = F;
    type Expr = Var;
    type Var = Var;
    type MainWindow = RowWindow<'a, Var>;
    type PreprocessedWindow = RowWindow<'a, Var>;
    type PublicVar = F;
    type PeriodicVar = Var;

    #[inline]
    fn main(&self) -> Self::MainWindow {
        self.inner.main()
    }

    #[inline]
    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        self.inner.preprocessed()
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.inner.is_first_row()
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.inner.is_last_row()
    }

    #[inline]
    fn is_transition(&self) -> Self::Expr {
        self.inner.is_transition()
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        if self.constraints_enabled {
            self.inner.assert_zero(x);
        }
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.inner.public_values()
    }

    #[inline]
    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.inner.periodic_values()
    }
}

impl<'a, F, Var, Acc> InteractionBuilder for InteractionMultilinearFolder<'a, F, Var, Acc>
where
    F: PrimeCharacteristicRing + Copy + Sync,
    Var: Algebra<F> + Copy + Send + Sync,
    Acc: Algebra<Var> + Copy,
{
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        _bus_name: &str,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Count<Self::Expr>>,
    ) {
        // Declaration order is the only key: the n-th push consumes the n-th recorded lookup.
        let lookup = &self.link.global_lookups()[self.global_seen];
        self.global_seen += 1;

        // The per-row magnitude bound is a planning input, so only the signed count is folded.
        let (multiplicity, _) = count.into().into_parts();

        // theta * sum_k beta^k * payload_k, with the bus offset added by the caller-side constant.
        let theta_fingerprint: Acc = dot_product(
            self.theta_beta_powers.iter().copied(),
            fields.into_iter().map(Into::into),
        );

        // A cross-AIR lookup carries exactly one tuple, hence one block.
        self.interaction_accumulator += lookup.block_weights[0]
            * (Acc::from(multiplicity) + lookup.theta_bus_offset - theta_fingerprint);
    }

    fn push_local_interaction(
        &mut self,
        tuples: impl IntoIterator<Item = (Vec<Self::Expr>, Count<Self::Expr>)>,
    ) {
        let lookup = &self.link.local_lookups()[self.local_seen];
        self.local_seen += 1;

        // One tuple per block, paired in declaration order.
        //
        //     tuples : (fields_0, count_0), (fields_1, count_1), ...
        //     weights: w_0,                 w_1,                 ...
        let mut paired = 0;
        for ((fields, count), weight) in
            tuples.into_iter().zip(lookup.block_weights.iter().copied())
        {
            let (multiplicity, _) = count.into_parts();
            let theta_fingerprint: Acc =
                dot_product(self.theta_beta_powers.iter().copied(), fields.into_iter());
            self.interaction_accumulator +=
                weight * (Acc::from(multiplicity) + lookup.theta_bus_offset - theta_fingerprint);
            paired += 1;
        }

        // Zipping would silently drop tuples if the AIR emitted a different count this pass.
        debug_assert_eq!(paired, lookup.block_weights.len());
    }

    fn num_global_interactions(&self) -> usize {
        self.global_seen
    }

    fn num_local_interactions(&self) -> usize {
        self.local_seen
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use core::borrow::Borrow;

    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_lookup::Count;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;
    use crate::lookup::{AirLinkInstance, AirLinkLookup};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type TestFolder<'a> = MultilinearFolder<'a, F, EF, EF>;

    /// Mini Fibonacci AIR used to exercise every selector path.
    ///
    /// Constraints:
    ///
    /// - first row:
    ///   - `left == public[0]`
    ///   - `right == public[1]`
    /// - transition:
    ///   - `next.left == local.right`
    ///   - `next.right == local.left + local.right`
    /// - last row: `right == public[2]`
    struct FibAir;

    const NUM_COLS: usize = 2;

    struct FibRow<T> {
        left: T,
        right: T,
    }

    impl<T> Borrow<FibRow<T>> for [T] {
        fn borrow(&self) -> &FibRow<T> {
            // Safety: two fields of type T in declaration order match the layout of [T; 2].
            debug_assert_eq!(self.len(), NUM_COLS);
            let ptr = self.as_ptr() as *const FibRow<T>;
            unsafe { &*ptr }
        }
    }

    impl<X> BaseAir<X> for FibAir {
        fn width(&self) -> usize {
            NUM_COLS
        }
        fn num_public_values(&self) -> usize {
            3
        }
    }

    impl<AB: AirBuilder> Air<AB> for FibAir {
        fn eval(&self, builder: &mut AB) {
            // Pull the two-row window and the public inputs into local bindings.
            let main = builder.main();
            let pis = builder.public_values();
            let a = pis[0];
            let b = pis[1];
            let x = pis[2];

            let local: &FibRow<AB::Var> = main.current_slice().borrow();
            let next: &FibRow<AB::Var> = main.next_slice().borrow();

            let mut when_first = builder.when_first_row();
            when_first.assert_eq(local.left, a);
            when_first.assert_eq(local.right, b);

            let mut when_trans = builder.when_transition();
            when_trans.assert_eq(local.right, next.left);
            when_trans.assert_eq(local.left + local.right, next.right);

            builder.when_last_row().assert_eq(local.right, x);
        }
    }

    /// Build a length-`n` Fibonacci trace seeded with `(0, 1)`.
    fn fib_trace(n: usize) -> RowMajorMatrix<F> {
        assert!(n.is_power_of_two());
        let mut left = F::ZERO;
        let mut right = F::ONE;
        let mut values = Vec::with_capacity(NUM_COLS * n);
        for _ in 0..n {
            // Each row records `(left, right)` before the step.
            values.push(left);
            values.push(right);
            let next_left = right;
            let next_right = left + right;
            left = next_left;
            right = next_right;
        }
        RowMajorMatrix::new(values, NUM_COLS)
    }

    /// Build the boundary selectors as on-cube indicators for row `i` of an `m`-row trace.
    fn boundary_at_row(i: usize, m: usize) -> BoundaryEvals<EF> {
        BoundaryEvals {
            first: if i == 0 { EF::ONE } else { EF::ZERO },
            last: if i == m - 1 { EF::ONE } else { EF::ZERO },
            transition: if i == m - 1 { EF::ZERO } else { EF::ONE },
        }
    }

    /// Slice row `i` of the trace and lift its entries into the extension field.
    fn row_in_ef(trace: &RowMajorMatrix<F>, i: usize) -> Vec<EF> {
        let w = trace.width;
        trace.values[i * w..(i + 1) * w]
            .iter()
            .copied()
            .map(EF::from)
            .collect()
    }

    #[test]
    fn folder_accumulator_is_zero_on_satisfied_rows() {
        // Fixture state: an 8-row Fibonacci trace with public inputs (F_0, F_1, F_8) = (0, 1, 21).
        //
        // Invariant: at every row the folder accumulator must equal zero.
        // Every constraint either evaluates to zero or is multiplied by a zero selector.
        let n = 8usize;
        let trace = fib_trace(n);
        let pis = [F::ZERO, F::ONE, F::from_u64(21)];
        let alpha = EF::from_u64(7);

        // Walk every row and check the accumulator.
        for i in 0..n {
            let local = row_in_ef(&trace, i);
            // Convention: the shifted "next" of the last row is all zeros (no successor).
            let next: Vec<EF> = if i == n - 1 {
                EF::zero_vec(NUM_COLS)
            } else {
                row_in_ef(&trace, i + 1)
            };
            let boundary = boundary_at_row(i, n);

            let value = TestFolder::new(&local, &next, boundary, &pis, alpha).eval_air(&FibAir);
            assert_eq!(value, EF::ZERO, "row {i}: folder returned {value:?}");
        }
    }

    #[test]
    fn folder_detects_a_bad_first_row() {
        // Fixture state: valid Fibonacci trace;
        // The verifier is told the first public input is 99 instead of 0.
        //
        // Mutation: substitute `public[0]` with 99.
        //
        //     row 0:   left = 0, right = 1
        //     claim:   left = 99  ->  constraint `local.left - 99` is non-zero
        //     ----->   folder accumulator must be non-zero
        let n = 8usize;
        let trace = fib_trace(n);
        let bad_pis = [F::from_u64(99), F::ONE, F::from_u64(21)];
        let alpha = EF::from_u64(7);

        let local = row_in_ef(&trace, 0);
        let next = row_in_ef(&trace, 1);
        let boundary = boundary_at_row(0, n);

        let value = TestFolder::new(&local, &next, boundary, &bad_pis, alpha).eval_air(&FibAir);
        assert_ne!(value, EF::ZERO);
    }

    #[test]
    fn folder_alpha_batching_matches_horner_fold() {
        // Invariant: assertions accumulate as `acc = alpha * acc + C_i`.
        //
        // The AIR has five `assert_eq` calls in declaration order:
        //
        //     0: when_first_row.assert_eq(local.left,            a)          - C_0
        //     1: when_first_row.assert_eq(local.right,           b)          - C_1
        //     2: when_trans.assert_eq(local.right,               next.left)  - C_2
        //     3: when_trans.assert_eq(local.left + local.right,  next.right) - C_3
        //     4: when_last_row.assert_eq(local.right,            x)          - C_4
        //
        // After all five pushes the accumulator must equal
        //
        //     C_0 * alpha^4 + C_1 * alpha^3 + C_2 * alpha^2 + C_3 * alpha + C_4.
        //
        // Fixture state: synthetic row hitting only the transition constraints.
        //
        //     selectors: first = 0, last = 0, transition = 1
        //     -----> only C_2 and C_3 contribute; C_0, C_1, C_4 vanish
        let local = [EF::from_u64(2), EF::from_u64(3)];
        let next = [EF::from_u64(5), EF::from_u64(7)];
        let pis = [F::from_u64(2), F::from_u64(3), F::from_u64(7)];
        let alpha = EF::from_u64(11);
        let boundary = BoundaryEvals {
            first: EF::ZERO,
            last: EF::ZERO,
            transition: EF::ONE,
        };

        let value = TestFolder::new(&local, &next, boundary, &pis, alpha).eval_air(&FibAir);

        // Active constraints (the two transition checks), in declaration order:
        //
        //     C_2 = local.right - next.left               = 3 - 5     = -2
        //     C_3 = local.left + local.right - next.right = 2 + 3 - 7 = -2
        //
        // Gated constraints C_0, C_1, C_4 vanish because their selectors are zero.
        let c2 = EF::from(local[1]) - EF::from(next[0]);
        let c3 = EF::from(local[0]) + EF::from(local[1]) - EF::from(next[1]);
        let gated = [EF::ZERO, EF::ZERO, c2, c3, EF::ZERO];

        // Hand-fold the same Horner pattern the folder uses.
        let mut expected = EF::ZERO;
        for g in gated {
            expected = expected * alpha + g;
        }
        assert_eq!(value, expected);
    }

    /// Single-column AIR that ties the main column to a preprocessed and a periodic column.
    ///
    /// Both constraints fire on every row (no selector gating):
    ///
    /// - `C_0`: `main.local[0] == preprocessed.local[0]`
    /// - `C_1`: `main.local[0] == periodic[0]`
    struct AuxAir;

    impl<X> BaseAir<X> for AuxAir {
        fn width(&self) -> usize {
            1
        }
    }

    impl<AB: AirBuilder> Air<AB> for AuxAir {
        fn eval(&self, builder: &mut AB) {
            // Read each auxiliary value out before the mutable assert calls.
            let local = builder.main().current_slice()[0];
            let prep = builder.preprocessed().current_slice()[0];
            let periodic = builder.periodic_values()[0];

            builder.assert_eq(local, prep);
            builder.assert_eq(local, periodic);
        }
    }

    #[test]
    fn folder_threads_preprocessed_and_periodic_columns() {
        // Fixture state: one main column, one preprocessed column, one periodic column.
        //
        // Invariant: when all three carry the same value both constraints vanish,
        // so the accumulator is zero; perturbing either auxiliary column breaks it.
        let alpha = EF::from_u64(7);
        let boundary = BoundaryEvals {
            first: EF::ZERO,
            last: EF::ZERO,
            transition: EF::ONE,
        };

        let main_local = [EF::from_u64(5)];
        let main_next = [EF::from_u64(9)];
        let prep_local = [EF::from_u64(5)];
        let prep_next = [EF::from_u64(9)];
        let periodic = [EF::from_u64(5)];

        // Matching auxiliary columns -> both constraints are zero.
        let value = TestFolder::new(&main_local, &main_next, boundary, &[] as &[F], alpha)
            .with_preprocessed(&prep_local, &prep_next)
            .with_periodic(&periodic)
            .eval_air(&AuxAir);
        assert_eq!(value, EF::ZERO);

        // Perturbed preprocessed column -> the first constraint is non-zero.
        let bad_prep = [EF::from_u64(6)];
        let value = TestFolder::new(&main_local, &main_next, boundary, &[] as &[F], alpha)
            .with_preprocessed(&bad_prep, &prep_next)
            .with_periodic(&periodic)
            .eval_air(&AuxAir);
        assert_ne!(value, EF::ZERO);

        // Perturbed periodic column -> the second constraint is non-zero.
        let bad_periodic = [EF::from_u64(6)];
        let value = TestFolder::new(&main_local, &main_next, boundary, &[] as &[F], alpha)
            .with_preprocessed(&prep_local, &prep_next)
            .with_periodic(&bad_periodic)
            .eval_air(&AuxAir);
        assert_ne!(value, EF::ZERO);
    }

    /// Two-column AIR declaring one local lookup with two tuples, plus one ordinary constraint.
    ///
    /// - `C_0`: `main.local[0] == main.local[1]`, gated by no selector.
    /// - tuple 0: payload `[local[0]]` requested once.
    /// - tuple 1: payload `[local[1]]` provided once.
    struct LinkedAir;

    impl<X> BaseAir<X> for LinkedAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl<AB: AirBuilder + InteractionBuilder> Air<AB> for LinkedAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let a = main.current_slice()[0];
            let b = main.current_slice()[1];

            builder.assert_eq(a, b);
            builder.push_local_interaction([
                (vec![a.into()], Count::bounded(AB::Expr::ONE, 1)),
                (vec![b.into()], Count::provided(-AB::Expr::ONE)),
            ]);
        }
    }

    #[test]
    fn interaction_folder_rebuilds_the_shifted_fraction_per_tuple() {
        // Invariant: one AIR pass yields the ordinary constraints and, separately,
        //
        //     link = sum_b w_b * (m_b + theta * (D_b - 1))
        //     D_b  = bus_prefix - sum_k theta_beta_k * payload_bk / theta
        //
        // which the folder forms as `m_b + theta * (bus_prefix - 1) - theta * fingerprint_b`.
        //
        // Fixture state: payloads 5 and 9, multiplicities +1 and -1, block weights 3 and 7.
        let a = EF::from_u64(5);
        let b = EF::from_u64(9);
        let theta_bus_offset = EF::from_u64(13);
        let link = AirLinkInstance {
            num_local_lookups: 1,
            lookups: vec![AirLinkLookup {
                theta_bus_offset,
                block_weights: vec![EF::from_u64(3), EF::from_u64(7)],
            }],
        };

        // A width-one payload uses only the first coefficient.
        let theta_beta_powers = [EF::from_u64(2)];

        let boundary = BoundaryEvals {
            first: EF::ZERO,
            last: EF::ZERO,
            transition: EF::ONE,
        };
        let alpha = EF::from_u64(11);
        let local = [a, b];
        let next = [EF::ZERO, EF::ZERO];

        let folder = TestFolder::new(&local, &next, boundary, &[] as &[F], alpha);
        let evaluations =
            InteractionMultilinearFolder::new(folder, &link, &theta_beta_powers, true)
                .eval_air(&LinkedAir);

        // Ordinary constraint: the single assertion `a - b`, with nothing to batch against.
        assert_eq!(evaluations.constraints, a - b);

        // Lookup link, tuple by tuple:
        //
        //     tuple 0: 3 * ( 1 + 13 - 2 * 5)
        //     tuple 1: 7 * (-1 + 13 - 2 * 9)
        let expected = EF::from_u64(3) * (EF::ONE + theta_bus_offset - theta_beta_powers[0] * a)
            + EF::from_u64(7) * (-EF::ONE + theta_bus_offset - theta_beta_powers[0] * b);
        assert_eq!(evaluations.interactions, expected);
    }

    #[test]
    fn interaction_folder_can_drop_the_ordinary_constraints() {
        // Past the ordinary constraints' own degree only the lookup link still contributes,
        // so the folder is told to leave the constraint accumulator untouched.
        let link = AirLinkInstance {
            num_local_lookups: 1,
            lookups: vec![AirLinkLookup {
                theta_bus_offset: EF::from_u64(13),
                block_weights: vec![EF::from_u64(3), EF::from_u64(7)],
            }],
        };
        let theta_beta_powers = [EF::from_u64(2)];
        let boundary = BoundaryEvals {
            first: EF::ZERO,
            last: EF::ZERO,
            transition: EF::ONE,
        };

        // Distinct columns make the ordinary constraint non-zero if it were accumulated.
        let local = [EF::from_u64(5), EF::from_u64(9)];
        let next = [EF::ZERO, EF::ZERO];

        let folder = TestFolder::new(&local, &next, boundary, &[] as &[F], EF::from_u64(11));
        let evaluations =
            InteractionMultilinearFolder::new(folder, &link, &theta_beta_powers, false)
                .eval_air(&LinkedAir);

        // The constraint was dropped, while the link is unaffected by the switch.
        assert_eq!(evaluations.constraints, EF::ZERO);
        assert_ne!(evaluations.interactions, EF::ZERO);
    }

    #[test]
    fn ordinary_folder_drops_lookup_declarations() {
        // The ordinary folder must still accept an AIR written against the interaction
        // interface, and must return only the constraint value.
        //
        //     columns : [5, 5] -> the assertion `a - b` vanishes
        //     lookups : declared, and dropped
        let local = [EF::from_u64(5), EF::from_u64(5)];
        let next = [EF::ZERO, EF::ZERO];
        let boundary = BoundaryEvals {
            first: EF::ZERO,
            last: EF::ZERO,
            transition: EF::ONE,
        };

        let value = TestFolder::new(&local, &next, boundary, &[] as &[F], EF::from_u64(11))
            .eval_air(&LinkedAir);
        assert_eq!(value, EF::ZERO);
    }
}
