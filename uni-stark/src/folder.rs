use alloc::vec::Vec;

use p3_air::{AirBuilder, ExtensionBuilder, RowWindow};
use p3_field::{Algebra, BasedVectorSpace};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::ViewPair;

use crate::{PackedChallenge, PackedVal, StarkGenericConfig, Val};

/// Buffer size for stack-allocated constraint collection in [`ProverConstraintFolder`].
///
/// Constraints are flushed via [`Algebra::batched_linear_combination`] whenever the buffer
/// fills. 64 keeps the buffer small enough for L1 cache (~1 KB per buffer for 4-lane
/// packed fields) while being large enough that most AIRs never flush mid-evaluation.
pub const FOLDER_BUF_SIZE: usize = 64;

/// Packed constraint folder for SIMD-optimized prover evaluation.
///
/// Uses packed types to evaluate constraints on multiple domain points simultaneously.
///
/// Collects constraints during `air.eval()` into fixed-size stack buffers, flushing
/// via [`Algebra::batched_linear_combination`] when full. This avoids heap allocation
/// per row-batch and keeps constraint data in cache.
#[derive(Debug)]
pub struct ProverConstraintFolder<'a, SC: StarkGenericConfig> {
    /// The [`RowMajorMatrixView`] containing rows on which the constraint polynomial is evaluated.
    pub main: RowMajorMatrixView<'a, PackedVal<SC>>,
    /// The preprocessed columns as a [`RowMajorMatrixView`].
    /// Zero-width when the AIR has no preprocessed trace.
    pub preprocessed: RowMajorMatrixView<'a, PackedVal<SC>>,
    /// Pre-built window over the preprocessed columns.
    pub preprocessed_window: RowWindow<'a, PackedVal<SC>>,
    /// Public inputs to the [AIR](`p3_air::Air`) implementation.
    pub public_values: &'a [Val<SC>],
    /// Evaluations of the first-row selector polynomial.
    /// Non-zero only on the first trace row.
    pub is_first_row: PackedVal<SC>,
    /// Evaluations of the last-row selector polynomial.
    /// Non-zero only on the last trace row.
    pub is_last_row: PackedVal<SC>,
    /// Evaluations of the transition selector polynomial.
    /// Zero only on the last trace row.
    pub is_transition: PackedVal<SC>,
    /// Stack buffer and accumulator for base-field constraints.
    pub base: ConstraintBuf<'a, PackedVal<SC>, Val<SC>, PackedChallenge<SC>>,
    /// Stack buffer and accumulator for extension-field constraints.
    pub ext: ConstraintBuf<'a, PackedChallenge<SC>, SC::Challenge>,
    /// Current constraint index being processed (debug-only bookkeeping)
    pub constraint_index: usize,
    /// Total number of constraints in the AIR (debug-only bookkeeping)
    pub constraint_count: usize,
}

/// Handles constraint verification for the verifier in a STARK system.
///
/// Similar to [`ProverConstraintFolder`] but operates on committed values rather than the full trace,
/// using a more efficient accumulation method for verification.
#[derive(Debug)]
pub struct VerifierConstraintFolder<'a, SC: StarkGenericConfig> {
    /// Pair of consecutive rows from the committed polynomial evaluations as a [`ViewPair`].
    pub main: ViewPair<'a, SC::Challenge>,
    /// The preprocessed columns as a [`ViewPair`].
    /// Zero-width when the AIR has no preprocessed trace.
    pub preprocessed: ViewPair<'a, SC::Challenge>,
    /// Pre-built window over the preprocessed columns.
    pub preprocessed_window: RowWindow<'a, SC::Challenge>,
    /// Public values that are inputs to the computation
    pub public_values: &'a [Val<SC>],
    /// Evaluations of the first-row selector polynomial.
    /// Non-zero only on the first trace row.
    pub is_first_row: SC::Challenge,
    /// Evaluations of the last-row selector polynomial.
    /// Non-zero only on the last trace row.
    pub is_last_row: SC::Challenge,
    /// Evaluations of the transition selector polynomial.
    /// Zero only on the last trace row.
    pub is_transition: SC::Challenge,
    /// Single challenge value used for constraint combination
    pub alpha: SC::Challenge,
    /// Running accumulator for all constraints
    pub accumulator: SC::Challenge,
}

impl<SC: StarkGenericConfig> ProverConstraintFolder<'_, SC> {
    /// Flush remaining constraints and return the combined result.
    #[inline]
    pub fn finalize_constraints(&mut self) -> PackedChallenge<SC> {
        debug_assert_eq!(self.constraint_index, self.constraint_count);
        self.base.flush();
        self.ext.flush();
        self.base.acc + self.ext.acc
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for ProverConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type PreprocessedWindow = RowWindow<'a, PackedVal<SC>>;
    type MainWindow = RowWindow<'a, PackedVal<SC>>;
    type PublicVar = Val<SC>;

    #[inline]
    fn main(&self) -> Self::MainWindow {
        RowWindow::from_view(&self.main)
    }

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
        self.base.push(x.into());
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        self.base.push_array(array.map(Into::into));
        self.constraint_index += N;
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

impl<SC: StarkGenericConfig> ExtensionBuilder for ProverConstraintFolder<'_, SC> {
    type EF = SC::Challenge;
    type ExprEF = PackedChallenge<SC>;
    type VarEF = PackedChallenge<SC>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.ext.push(x.into());
        self.constraint_index += 1;
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for VerifierConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = SC::Challenge;
    type Var = SC::Challenge;
    type PreprocessedWindow = RowWindow<'a, SC::Challenge>;
    type MainWindow = RowWindow<'a, SC::Challenge>;
    type PublicVar = Val<SC>;

    fn main(&self) -> Self::MainWindow {
        RowWindow::from_two_rows(self.main.top.values, self.main.bottom.values)
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed_window
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert!(size <= 2, "only two-row windows are supported, got {size}");
        self.is_transition
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.accumulator *= self.alpha;
        self.accumulator += x.into();
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

/// Fixed-size stack buffer for collecting and combining constraints.
///
/// Stores up to `N` elements, auto-flushing via [`Algebra::batched_linear_combination`]
/// into the running accumulator `acc` when the buffer is full. Holds a reference to
/// the per-dimension coefficient arrays so that [`push`](Self::push) is self-contained.
///
/// `Acc` may differ from `T` (e.g. extension-field accumulator over base-field buffer).
/// When `Acc = T` (the default), [`BasedVectorSpace`] dimension is 1 and flush
/// reduces to a single `batched_linear_combination` call.
#[derive(Debug)]
pub struct ConstraintBuf<'a, T, F, Acc = T, const N: usize = FOLDER_BUF_SIZE> {
    buf: [T; N],
    len: usize,
    start: usize,
    pub acc: Acc,
    coeffs: &'a [Vec<F>],
}

impl<'a, T, F, Acc, const N: usize> ConstraintBuf<'a, T, F, Acc, N>
where
    T: Algebra<F> + Copy,
    F: Clone,
    Acc: Algebra<T> + BasedVectorSpace<T> + Copy,
{
    pub const fn new(coeffs: &'a [Vec<F>]) -> Self {
        Self {
            buf: [T::ZERO; N],
            len: 0,
            start: 0,
            acc: Acc::ZERO,
            coeffs,
        }
    }

    /// Push a constraint value, auto-flushing when the buffer is full.
    #[inline]
    pub fn push(&mut self, val: T) {
        if self.len == N {
            self.flush();
        }
        self.buf[self.len] = val;
        self.len += 1;
    }

    /// Push a compile-time-sized batch, flushing as needed.
    #[inline]
    pub fn push_array<const M: usize>(&mut self, vals: [T; M]) {
        let mut offset = 0;
        while offset < M {
            if self.len == N {
                self.flush();
            }
            let space = N - self.len;
            let chunk = if M - offset < space {
                M - offset
            } else {
                space
            };
            self.buf[self.len..self.len + chunk].copy_from_slice(&vals[offset..offset + chunk]);
            self.len += chunk;
            offset += chunk;
        }
    }

    /// Combine buffered elements with their coefficients and accumulate.
    #[inline]
    pub fn flush(&mut self) {
        if self.len == 0 {
            return;
        }
        let buf = &self.buf[..self.len];
        let start = self.start;
        self.start += self.len;
        self.len = 0;
        self.acc += Acc::from_basis_coefficients_fn(|d| {
            T::batched_linear_combination(buf, &self.coeffs[d][start..start + buf.len()])
        });
    }
}
