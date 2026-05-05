//! Adapter that lifts any [`p3_air::Air`] into a [`BundledPesat`] for WARP.
//!
//! # Mapping AIR → PESAT
//!
//! Given an AIR with `c` constraints over a trace of width `w` and height
//! `H = 2^h`, the PESAT shape is:
//!
//! ```text
//!     M  = c_padded · H        (constraints, padded to next power of two)
//!     k  = w · H                (witness length = main trace cells)
//!     κ  = 0                    (v1: no public values, no preprocessed)
//!     N  = k
//!     d  = AIR max constraint degree
//! ```
//!
//! The constraint index `i ∈ [0, M)` decomposes as `i = r · c_padded + c`
//! with the row index `r` in the high bits and the constraint index `c` in
//! the low bits. The protocol layer builds `τ_eq[i] = eq(τ, i)` using
//! [`Poly::new_from_point`](p3_multilinear_util::poly::Poly::new_from_point),
//! which uses Plonky3's big-endian hypercube convention — matching this
//! decomposition.
//!
//! # v1 Limitations
//!
//! - No public values (`air.num_public_values() == 0`).
//! - No preprocessed trace.
//! - No periodic columns.
//! - `main_width` and `H` must both be powers of two so the witness length
//!   `k = main_width · H` is a power of two (required by the Reed-Solomon
//!   message length).
//!
//! These restrictions are not fundamental to WARP, but they are real API
//! boundaries for this Plonky3 specialization and must be lifted deliberately.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir, RowWindow};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::poly::Poly;

use super::{BundledPesat, PesatShape};

/// AIR-as-PESAT adapter.
///
/// Holds an AIR plus the trace dimensions; implements [`BundledPesat`] by
/// invoking [`Air::eval`] on a custom [`EvalBuilder`] per row.
#[derive(Clone, Debug)]
pub struct AirAsPesat<A, F, EF> {
    air: A,
    log_height: usize,
    /// Logical number of constraints emitted by `air.eval`. May be < `c_padded`.
    num_constraints: usize,
    /// Padded constraint count = next_power_of_two(num_constraints).
    log_constraints_padded: usize,
    main_width: usize,
    max_degree: usize,
    /// Description bytes of this AIR (for transcript binding).
    description: Vec<u8>,
    _ph: PhantomData<(F, EF)>,
}

impl<A, F, EF> AirAsPesat<A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F>,
{
    /// Create an `AirAsPesat` for the given AIR with trace height `2^log_height`.
    ///
    /// Both `air.num_constraints()` and `air.max_constraint_degree()` must
    /// return `Some(_)`. AIRs in production almost always provide both
    /// hints; if yours doesn't, override them via the `BaseAir` trait.
    ///
    /// # Panics
    ///
    /// - The AIR must report `num_public_values() == 0`.
    /// - `air.preprocessed_trace()` must be `None`.
    /// - `air.num_constraints()` must be `Some(_)` (and ≥ 1).
    /// - `air.max_constraint_degree()` must be `Some(_)` (and ≥ 1).
    /// - `main_width` and `2^log_height` must be powers of two.
    pub fn new(air: A, log_height: usize, description: Vec<u8>) -> Self {
        assert_eq!(
            air.num_public_values(),
            0,
            "AirAsPesat: AIRs with public values are not supported in v1"
        );
        assert!(
            air.preprocessed_trace().is_none(),
            "AirAsPesat: AIRs with preprocessed traces are not supported in v1"
        );
        let num_constraints = air
            .num_constraints()
            .expect("AirAsPesat: AIR must implement num_constraints()");
        assert!(num_constraints > 0, "AirAsPesat: AIR must have constraints");
        let max_degree = air
            .max_constraint_degree()
            .expect("AirAsPesat: AIR must implement max_constraint_degree()");
        assert!(
            max_degree >= 1,
            "AirAsPesat: max_constraint_degree must be ≥ 1"
        );
        let main_width = air.width();
        assert!(
            main_width.is_power_of_two(),
            "AirAsPesat: main width must be a power of two (got {main_width})"
        );
        let log_constraints_padded = num_constraints.next_power_of_two().trailing_zeros() as usize;
        Self {
            air,
            log_height,
            num_constraints,
            log_constraints_padded,
            main_width,
            max_degree,
            description,
            _ph: PhantomData,
        }
    }

    /// Returns the underlying AIR.
    #[inline]
    pub const fn air(&self) -> &A {
        &self.air
    }

    /// Trace height `H = 2^log_height`.
    #[inline]
    pub const fn height(&self) -> usize {
        1 << self.log_height
    }

    /// `log_2 H`.
    #[inline]
    pub const fn log_height(&self) -> usize {
        self.log_height
    }

    /// Number of main-trace columns.
    #[inline]
    pub const fn main_width(&self) -> usize {
        self.main_width
    }

    /// Number of constraints emitted by `air.eval` (before padding to a power
    /// of two).
    #[inline]
    pub const fn num_constraints(&self) -> usize {
        self.num_constraints
    }

    /// Padded constraint count `c_padded = 2^log_constraints_padded`.
    #[inline]
    pub const fn num_constraints_padded(&self) -> usize {
        1 << self.log_constraints_padded
    }

    /// `log_2` of the padded row-local constraint count.
    #[inline]
    pub const fn log_constraints_padded(&self) -> usize {
        self.log_constraints_padded
    }

    /// Maximum total degree reported by the wrapped AIR.
    #[inline]
    pub const fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Row columns that the wrapped AIR may read through `main().next_slice()`.
    #[inline]
    pub fn main_next_row_columns(&self) -> Vec<usize> {
        self.air.main_next_row_columns()
    }

    /// Whether the wrapped AIR reads next-row values.
    #[inline]
    pub fn uses_next_row_values(&self) -> bool {
        !self.main_next_row_columns().is_empty()
    }

    /// Whether the wrapped AIR can be handled without shifted-row openings.
    ///
    /// The WHIR-native decider sumcheck implemented in `finalize::whir`
    /// reduces terminal claims to openings of the committed witness/codeword
    /// oracle at row/column points. AIRs that read next-row values additionally
    /// need a shifted-oracle treatment.
    pub fn supports_current_row_decider_sumcheck(&self) -> bool {
        !self.uses_next_row_values()
    }

    /// Evaluate the β-constraint-weighted AIR expression at one symbolic row.
    ///
    /// `row_point` is a point in `F^log_height`, `constraint_point` is a point
    /// in `F^log_constraints_padded`, `current` contains the opened main trace
    /// column values at `row_point`, and `next` contains the shifted-row values
    /// at the same symbolic row. This returns
    ///
    /// ```text
    /// Σ_c eq(constraint_point, c) · constraint_c(current(row_point)).
    /// ```
    ///
    /// Boundary selectors are evaluated as multilinear extensions over the
    /// row hypercube.
    pub fn evaluate_row_constraint_combination(
        &self,
        row_point: &[EF],
        constraint_point: &[EF],
        current: &[EF],
        next: &[EF],
    ) -> EF
    where
        A: for<'a> Air<EvalBuilder<'a, F, EF>>,
    {
        assert_eq!(row_point.len(), self.log_height(), "row point dimension");
        assert_eq!(
            constraint_point.len(),
            self.log_constraints_padded(),
            "constraint point dimension"
        );
        assert_eq!(current.len(), self.main_width(), "current row width");
        assert_eq!(next.len(), self.main_width(), "next row width");

        let constraint_eq = Poly::<EF>::new_from_point(constraint_point, EF::ONE);
        let empty: &[EF] = &[];
        let preprocessed_window = RowWindow::from_two_rows(empty, empty);
        let is_last_row = eval_eq_at_index(row_point, self.height() - 1);
        let mut builder = EvalBuilder::<F, EF> {
            current,
            next,
            preprocessed_window,
            public_values: &[],
            is_first_row: eval_eq_at_index(row_point, 0),
            is_last_row,
            is_transition: EF::ONE - is_last_row,
            row_constraint_eq: constraint_eq.as_slice(),
            current_constraint: 0,
            accumulator: EF::ZERO,
            _ph: PhantomData,
        };
        self.air.eval(&mut builder);
        builder.accumulator
    }

    /// Same as [`Self::evaluate_row_constraint_combination`], but accepts the
    /// already-expanded constraint equality table.
    ///
    /// This avoids rebuilding `eq(constraint_point, ·)` in prover loops where
    /// the constraint point is fixed while only the row point varies.
    pub fn evaluate_row_constraint_combination_with_eq(
        &self,
        row_point: &[EF],
        constraint_eq: &[EF],
        current: &[EF],
        next: &[EF],
    ) -> EF
    where
        A: for<'a> Air<EvalBuilder<'a, F, EF>>,
    {
        assert_eq!(row_point.len(), self.log_height(), "row point dimension");
        assert_eq!(
            constraint_eq.len(),
            self.num_constraints_padded(),
            "constraint eq length"
        );
        assert_eq!(current.len(), self.main_width(), "current row width");
        assert_eq!(next.len(), self.main_width(), "next row width");

        let empty: &[EF] = &[];
        let preprocessed_window = RowWindow::from_two_rows(empty, empty);
        let is_last_row = eval_eq_at_index(row_point, self.height() - 1);
        let mut builder = EvalBuilder::<F, EF> {
            current,
            next,
            preprocessed_window,
            public_values: &[],
            is_first_row: eval_eq_at_index(row_point, 0),
            is_last_row,
            is_transition: EF::ONE - is_last_row,
            row_constraint_eq: constraint_eq,
            current_constraint: 0,
            accumulator: EF::ZERO,
            _ph: PhantomData,
        };
        self.air.eval(&mut builder);
        builder.accumulator
    }

    /// Evaluate a current-row-only AIR expression at one symbolic row.
    pub fn evaluate_current_row_constraint_combination(
        &self,
        row_point: &[EF],
        constraint_point: &[EF],
        current: &[EF],
    ) -> EF
    where
        A: for<'a> Air<EvalBuilder<'a, F, EF>>,
    {
        let next = EF::zero_vec(self.main_width());
        self.evaluate_row_constraint_combination(row_point, constraint_point, current, &next)
    }
}

fn eval_eq_at_index<F: Field>(point: &[F], index: usize) -> F {
    debug_assert!(index < (1usize << point.len()));
    point
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if (index >> (point.len() - 1 - i)) & 1 == 1 {
                x
            } else {
                F::ONE - x
            }
        })
        .product()
}

impl<A, F, EF> BundledPesat<F, EF> for AirAsPesat<A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F>
        + for<'a> Air<EvalBuilder<'a, F, EF>>
        + for<'a> Air<PerConstraintEvalBuilder<'a, F, EF>>
        + Send
        + Sync,
{
    fn shape(&self) -> PesatShape {
        PesatShape {
            log_constraints: self.log_constraints_padded + self.log_height,
            log_witness: self.main_width.trailing_zeros() as usize + self.log_height,
            explicit_len: 0,
            max_degree: self.max_degree,
        }
    }

    fn evaluate_bundled(&self, tau_eq: &[EF], z: &[EF]) -> EF {
        let shape = self.shape();
        assert_eq!(tau_eq.len(), shape.num_constraints(), "τ_eq table size");
        assert_eq!(z.len(), shape.total_vars(), "z length");

        let h = self.height();
        let w = self.main_width;
        let c_padded = self.num_constraints_padded();

        // No public values, no explicit instance — z is just the main trace.
        let main: &[EF] = z;

        let empty: &[EF] = &[];
        let preprocessed_window = RowWindow::from_two_rows(empty, empty);

        let mut total = EF::ZERO;
        for r in 0..h {
            let row_eq = &tau_eq[r * c_padded..(r + 1) * c_padded];
            let next_r = (r + 1) % h;
            let current = &main[r * w..(r + 1) * w];
            let next = &main[next_r * w..(next_r + 1) * w];

            let mut builder = EvalBuilder::<F, EF> {
                current,
                next,
                preprocessed_window,
                public_values: &[],
                is_first_row: if r == 0 { EF::ONE } else { EF::ZERO },
                is_last_row: if r == h - 1 { EF::ONE } else { EF::ZERO },
                is_transition: if r == h - 1 { EF::ZERO } else { EF::ONE },
                row_constraint_eq: row_eq,
                current_constraint: 0,
                accumulator: EF::ZERO,
                _ph: PhantomData,
            };
            self.air.eval(&mut builder);
            total += builder.accumulator;
        }
        total
    }

    fn iter_constraint_polys_at_lerp(
        &self,
        b_x_lo: &[EF],
        b_x_hi: &[EF],
        w_lo: &[EF],
        w_hi: &[EF],
    ) -> Vec<Vec<EF>> {
        let shape = self.shape();
        // v1 AirAsPesat has κ = 0; assert that and ignore b_x.
        assert_eq!(shape.explicit_len, 0, "AirAsPesat v1 expects κ = 0");
        assert_eq!(b_x_lo.len(), 0, "b_x_lo must be empty");
        assert_eq!(b_x_hi.len(), 0, "b_x_hi must be empty");
        assert_eq!(w_lo.len(), shape.witness_len(), "w_lo length");
        assert_eq!(w_hi.len(), shape.witness_len(), "w_hi length");

        let m = shape.num_constraints();
        let d = self.max_degree;
        let h = self.height();
        let w = self.main_width;
        let c_padded = self.num_constraints_padded();
        debug_assert_eq!(m, h * c_padded, "M = H · c_padded");

        let empty: &[EF] = &[];
        let preprocessed_window = RowWindow::from_two_rows(empty, empty);

        // For each α ∈ {0, …, d}: evaluate every constraint of every row.
        // Result shape: per_alpha[α_idx][global_constraint_index] : EF.
        let mut per_alpha: Vec<Vec<EF>> = (0..=d).map(|_| Vec::with_capacity(m)).collect();

        // Reused per-row scratch.
        let mut row_constraints: Vec<EF> = Vec::with_capacity(self.num_constraints);
        let mut w_alpha: Vec<EF> = vec![EF::ZERO; w_lo.len()];

        for alpha_idx in 0..=d {
            let alpha = EF::from_u64(alpha_idx as u64);
            // Build w(α) of length k.
            for ((slot, &l), &r) in w_alpha.iter_mut().zip(w_lo.iter()).zip(w_hi.iter()) {
                *slot = l + alpha * (r - l);
            }

            let alpha_row = &mut per_alpha[alpha_idx];
            for r in 0..h {
                let next_r = (r + 1) % h;
                let current = &w_alpha[r * w..(r + 1) * w];
                let next = &w_alpha[next_r * w..(next_r + 1) * w];

                row_constraints.clear();
                let mut builder = PerConstraintEvalBuilder::<F, EF> {
                    current,
                    next,
                    preprocessed_window,
                    public_values: &[],
                    is_first_row: if r == 0 { EF::ONE } else { EF::ZERO },
                    is_last_row: if r == h - 1 { EF::ONE } else { EF::ZERO },
                    is_transition: if r == h - 1 { EF::ZERO } else { EF::ONE },
                    constraints_out: &mut row_constraints,
                    _ph: PhantomData,
                };
                self.air.eval(&mut builder);

                // Pad to c_padded with zero constraints.
                debug_assert!(
                    row_constraints.len() <= c_padded,
                    "AIR emitted more constraints than declared"
                );
                while row_constraints.len() < c_padded {
                    row_constraints.push(EF::ZERO);
                }
                alpha_row.extend_from_slice(&row_constraints);
            }
            debug_assert_eq!(alpha_row.len(), m);
        }

        // Transpose + Lagrange-interpolate per constraint.
        // polys[c] has length d + 1 (coefficient form of an α-polynomial).
        let mut polys: Vec<Vec<EF>> = Vec::with_capacity(m);
        let mut evals_buf = vec![EF::ZERO; d + 1];
        for c_idx in 0..m {
            for alpha_idx in 0..=d {
                evals_buf[alpha_idx] = per_alpha[alpha_idx][c_idx];
            }
            polys.push(super::lagrange_interpolate_int_points(&evals_buf));
        }
        polys
    }

    fn description(&self) -> Vec<u8> {
        self.description.clone()
    }
}

/// AirBuilder used by [`AirAsPesat::evaluate_bundled`].
///
/// Trace cells live in `EF`. Each `assert_zero(x)` call accumulates
/// `row_constraint_eq[current_constraint] * x` into `accumulator`,
/// then advances `current_constraint`.
#[derive(Debug)]
pub struct EvalBuilder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    pub current: &'a [EF],
    pub next: &'a [EF],
    pub preprocessed_window: RowWindow<'a, EF>,
    pub public_values: &'a [F],
    pub is_first_row: EF,
    pub is_last_row: EF,
    pub is_transition: EF,
    /// Slice of `tau_eq[r * c_padded .. (r + 1) * c_padded]` for the
    /// current row; the builder reads per-constraint weights from here.
    pub row_constraint_eq: &'a [EF],
    /// Index of the next constraint to emit. Advances on each `assert_zero`.
    pub current_constraint: usize,
    /// Running EF accumulator: Σ_c row_constraint_eq[c] · constraint_c.
    pub accumulator: EF,
    _ph: PhantomData<F>,
}

impl<'a, F, EF> AirBuilder for EvalBuilder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type F = F;
    type Expr = EF;
    type Var = EF;
    type PreprocessedWindow = RowWindow<'a, EF>;
    type MainWindow = RowWindow<'a, EF>;
    type PublicVar = F;

    #[inline]
    fn main(&self) -> Self::MainWindow {
        RowWindow::from_two_rows(self.current, self.next)
    }

    #[inline]
    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed_window
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert!(size <= 2, "only two-row windows are supported, got {size}");
        self.is_transition
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x_ef: EF = x.into();
        if self.current_constraint < self.row_constraint_eq.len() {
            self.accumulator += self.row_constraint_eq[self.current_constraint] * x_ef;
        }
        self.current_constraint += 1;
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

/// AirBuilder that records each `assert_zero(x)` call as one entry of
/// `constraints_out` (instead of accumulating a τ-weighted sum like
/// [`EvalBuilder`]).
///
/// Used by [`AirAsPesat::iter_constraint_polys_at_lerp`] to extract the
/// per-constraint polynomial slices required by the Lemma 6.4 / Claim 6.5
/// §6.3 prover.
#[derive(Debug)]
pub struct PerConstraintEvalBuilder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    pub current: &'a [EF],
    pub next: &'a [EF],
    pub preprocessed_window: RowWindow<'a, EF>,
    pub public_values: &'a [F],
    pub is_first_row: EF,
    pub is_last_row: EF,
    pub is_transition: EF,
    /// Row-local list of evaluated constraint values; the builder appends one
    /// entry per `assert_zero` call.
    pub constraints_out: &'a mut Vec<EF>,
    _ph: PhantomData<F>,
}

impl<'a, F, EF> AirBuilder for PerConstraintEvalBuilder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type F = F;
    type Expr = EF;
    type Var = EF;
    type PreprocessedWindow = RowWindow<'a, EF>;
    type MainWindow = RowWindow<'a, EF>;
    type PublicVar = F;

    #[inline]
    fn main(&self) -> Self::MainWindow {
        RowWindow::from_two_rows(self.current, self.next)
    }

    #[inline]
    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed_window
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert!(size <= 2, "only two-row windows are supported, got {size}");
        self.is_transition
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints_out.push(x.into());
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
    use p3_multilinear_util::poly::Poly;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::AirAsPesat;
    use crate::relation::{BundledPesat, lagrange_interpolate_int_points};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Test AIR: each cell `c` of the row is constrained by `c · (c − 1) = 0`
    /// (Boolean). Width-`w` AIR has `w` quadratic constraints per row.
    #[derive(Clone, Debug)]
    struct BoolAir {
        width: usize,
    }

    impl<FF: Field> BaseAir<FF> for BoolAir {
        fn width(&self) -> usize {
            self.width
        }
        fn num_constraints(&self) -> Option<usize> {
            Some(self.width)
        }
        fn max_constraint_degree(&self) -> Option<usize> {
            Some(2)
        }
        fn num_public_values(&self) -> usize {
            0
        }
        fn main_next_row_columns(&self) -> Vec<usize> {
            Vec::new()
        }
    }

    impl<AB> Air<AB> for BoolAir
    where
        AB: AirBuilder,
        AB::F: Field,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row: Vec<AB::Var> = main.current_slice().to_vec();
            for cell in row.into_iter().take(self.width) {
                let cell_expr: AB::Expr = cell.into();
                builder.assert_zero(cell_expr.clone() * (cell_expr - AB::Expr::ONE));
            }
        }
    }

    /// Naive `bundled_round_poly` reference: evaluate the polynomial at
    /// `D + 1` integer α points and Lagrange-interpolate. Quadratic in `D`
    /// per evaluation but trivially correct — used as the gold standard
    /// to validate the Claim 6.5 path.
    fn naive_round_poly_via_evaluation<P, EF2>(
        pesat: &P,
        b_lo: &[EF2],
        b_hi: &[EF2],
        w_lo: &[EF2],
        w_hi: &[EF2],
    ) -> Vec<EF2>
    where
        P: BundledPesat<F, EF2>,
        EF2: ExtensionField<F>,
    {
        let shape = pesat.shape();
        let log_m = shape.log_constraints;
        let kappa = shape.explicit_len;
        let k = shape.witness_len();
        let d_plus_one = pesat.round_poly_degree() + 1;

        let mut evals = vec![EF2::ZERO; d_plus_one];
        let mut z_buf = vec![EF2::ZERO; kappa + k];
        for (alpha_idx, slot) in evals.iter_mut().enumerate() {
            let alpha = EF2::from_u64(alpha_idx as u64);
            let b_alpha: Vec<EF2> = b_lo
                .iter()
                .zip(b_hi.iter())
                .map(|(&l, &r)| l + alpha * (r - l))
                .collect();
            for ((slot, &l), &r) in z_buf[kappa..].iter_mut().zip(w_lo.iter()).zip(w_hi.iter()) {
                *slot = l + alpha * (r - l);
            }
            z_buf[..kappa].copy_from_slice(&b_alpha[log_m..]);
            let eq_table = Poly::<EF2>::new_from_point(&b_alpha[..log_m], EF2::ONE);
            *slot = pesat.evaluate_bundled(eq_table.as_slice(), &z_buf);
        }
        lagrange_interpolate_int_points(&evals)
    }

    /// Cross-validate: the new `bundled_round_poly` (Claim 6.5 / Lemma 6.4
    /// path) must produce exactly the same coefficient vector as the naive
    /// `D + 1` evaluations + Lagrange-interpolate path.
    ///
    /// Tests several `(width, log_height)` shapes — both paths consume the
    /// same `(b_lo, b_hi, w_lo, w_hi)` and the equality must be bit-exact.
    #[test]
    fn bundled_round_poly_matches_naive_for_bool_air() {
        use rand::RngExt;
        let shapes = [(2usize, 1usize), (4, 2), (4, 3), (8, 2)];
        for &(width, log_height) in &shapes {
            let air = BoolAir { width };
            let pesat: AirAsPesat<BoolAir, F, EF> =
                AirAsPesat::new(air, log_height, b"BoolAir/cross-validate".to_vec());
            let shape = pesat.shape();
            let mut rng = SmallRng::seed_from_u64(((width as u64) << 8) | log_height as u64);

            let b_lo: Vec<EF> = (0..shape.beta_len()).map(|_| rng.random::<EF>()).collect();
            let b_hi: Vec<EF> = (0..shape.beta_len()).map(|_| rng.random::<EF>()).collect();
            let w_lo: Vec<EF> = (0..shape.witness_len())
                .map(|_| rng.random::<EF>())
                .collect();
            let w_hi: Vec<EF> = (0..shape.witness_len())
                .map(|_| rng.random::<EF>())
                .collect();

            let claim_6_5_coeffs = pesat.bundled_round_poly(&b_lo, &b_hi, &w_lo, &w_hi);
            let naive_coeffs = naive_round_poly_via_evaluation(&pesat, &b_lo, &b_hi, &w_lo, &w_hi);

            assert_eq!(
                claim_6_5_coeffs.len(),
                naive_coeffs.len(),
                "coefficient lengths differ at width={width}, log_height={log_height}"
            );
            assert_eq!(
                claim_6_5_coeffs, naive_coeffs,
                "Claim 6.5 disagrees with naive at width={width}, log_height={log_height}"
            );
        }
    }
}
