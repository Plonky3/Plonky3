//! AIR constraint folder for the multilinear prover.
//!
//! - Implements the standard AIR builder interface, so any AIR runs unchanged through the folder.
//! - Drives both directions of the protocol with the same evaluator:
//!   - the prover walks the boolean hypercube row by row,
//!   - the verifier evaluates at the random sumcheck challenge.

use p3_air::{Air, AirBuilder, RowWindow};
use p3_field::{Algebra, PrimeCharacteristicRing, dot_product};
use p3_lookup::{Count, InteractionBuilder};

use crate::lookup::AirLinkInstance;
use crate::selectors::BoundaryEvals;

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
    /// The preprocessed window and periodic values start empty.
    ///
    /// Attach them when the AIR declares those columns:
    /// - [`Self::with_preprocessed`] for preprocessed columns.
    /// - [`Self::with_periodic`] for periodic columns.
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

/// Discard lookup declarations while evaluating only ordinary AIR constraints.
///
/// Lookup-active AIRs are generic over [`InteractionBuilder`], so the ordinary
/// path still needs to accept their declarations. The interaction wrapper below
/// replaces this sink whenever lookup-link values are actually required.
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
        _tuples: impl IntoIterator<Item = (alloc::vec::Vec<Self::Expr>, Count<Self::Expr>)>,
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
/// The inner folder retains the constraint-only hot path. This adapter is
/// constructed only at interpolation nodes that need lookup-link evaluations;
/// it can then collect ordinary constraints, interactions, or both in one AIR
/// evaluation.
#[derive(Debug)]
pub struct InteractionMultilinearFolder<'a, F, Var, Acc> {
    inner: MultilinearFolder<'a, F, Var, Acc>,
    link: &'a AirLinkInstance<Acc>,
    theta_beta_powers: &'a [Acc],
    interaction_accumulator: Acc,
    constraints_enabled: bool,
    local_seen: usize,
    global_seen: usize,
}

impl<'a, F, Var, Acc> InteractionMultilinearFolder<'a, F, Var, Acc>
where
    Acc: PrimeCharacteristicRing,
{
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
        let lookup_index = self.global_seen;
        self.global_seen += 1;
        let lookup = &self.link.global_lookups()[lookup_index];
        let (multiplicity, _) = count.into().into_parts();
        let theta_fingerprint: Acc = dot_product(
            self.theta_beta_powers.iter().copied(),
            fields.into_iter().map(Into::into),
        );
        self.interaction_accumulator += lookup.block_weights[0]
            * (Acc::from(multiplicity) + lookup.theta_bus_offset - theta_fingerprint);
    }

    fn push_local_interaction(
        &mut self,
        tuples: impl IntoIterator<Item = (alloc::vec::Vec<Self::Expr>, Count<Self::Expr>)>,
    ) {
        let lookup_index = self.local_seen;
        self.local_seen += 1;
        let lookup = &self.link.local_lookups()[lookup_index];

        tuples
            .into_iter()
            .zip(lookup.block_weights.iter().copied())
            .for_each(|((fields, count), weight)| {
                let (multiplicity, _) = count.into_parts();
                let theta_fingerprint: Acc =
                    dot_product(self.theta_beta_powers.iter().copied(), fields.into_iter());
                self.interaction_accumulator += weight
                    * (Acc::from(multiplicity) + lookup.theta_bus_offset - theta_fingerprint);
            });
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
    use alloc::vec::Vec;
    use core::borrow::Borrow;

    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;

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
}
