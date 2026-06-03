//! Core LogUp Implementation
//!
//! ## Mathematical Foundation
//!
//! LogUp transforms the standard lookup equation:
//! ```text
//! ∏(α - a_i)^(m_i) = ∏(α - b_j)^(m'_j)
//! ```
//!
//! Into an equivalent sum-based form using logarithmic derivatives:
//! ```text
//! ∑(m_i/(α - a_i)) = ∑(m'_j/(α - b_j))
//! ```
//!
//! Where:
//! - `α` is a random challenge
//! - `m_i, m'_j` are multiplicities (how many times each element appears)
//! - The transformation eliminates expensive exponentiation operations

use alloc::vec;
use alloc::vec::Vec;

use p3_air::{ExtensionBuilder, PermutationAirBuilder, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing, dot_product};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::VerticalPair;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{StarkGenericConfig, Val};
use tracing::instrument;

use crate::protocol::LookupProtocol;
use crate::traits::LookupTraceBuilder;
use crate::types::{Lookup, LookupError, LookupTerminal};

/// Type alias for the row evaluation context used during permutation trace generation.
pub type RowEvalContext<'a, SC> = LookupTraceBuilder<'a, SC>;

/// Core LogUp gadget implementing lookup arguments via logarithmic derivatives.
///
/// The LogUp gadget transforms the multiplicative lookup constraint:
/// ```text
/// ∏(α - a_i)^(m_i) = ∏(α - b_j)^(m'_j)
/// ```
///
/// Into an equivalent additive constraint using logarithmic differentiation:
/// ```text
/// ∑(m_i/(α - a_i)) = ∑(m'_j/(α - b_j))
/// ```
///
/// Per AIR the auxiliary trace carries one fraction column per lookup plus
/// a single shared accumulator column at index 0. Writing
/// `f_c[r] = V_c(r) / U_c(r)` for the per-row rational value of lookup `c`:
/// ```text
/// acc[0]   = 0
/// acc[i+1] = acc[i] + ∑_c f_c[i]
/// ```
///
/// Note that we do not differentiate between `a` and `b` in the implementation:
/// we simply have a list of `elements` with possibly negative `multiplicities`.
///
/// Constraints are defined as:
/// - **Fraction pin** (per lookup `c`): `U_c(r) * f_c[r] - V_c(r) = 0`
/// - **Initial Constraint**: `acc[0] = 0`
/// - **Transition Constraint**: `acc[i+1] = acc[i] + ∑_c f_c[i]`
/// - **Final Constraint**: `terminal = acc[n-1] + ∑_c f_c[n-1]`
#[derive(Debug, Clone, Default)]
pub struct LogUpGadget;

impl LogUpGadget {
    /// Creates a new LogUp gadget instance.
    pub const fn new() -> Self {
        Self {}
    }

    /// Computes the combined elements for each tuple using the challenge `beta`:
    /// `combined_elements[i] = ∑elements[i][n-j] * β^j`
    fn combine_elements<AB, E>(
        &self,
        elements: &[Vec<E>],
        alpha: &AB::ExprEF,
        beta: &AB::ExprEF,
    ) -> Vec<AB::ExprEF>
    where
        AB: PermutationAirBuilder,
        E: Into<AB::ExprEF> + Clone,
    {
        elements
            .iter()
            .map(|elts| {
                // Combine the elements in the tuple using beta.
                let combined_elt = elts.iter().fold(AB::ExprEF::ZERO, |acc, elt| {
                    elt.clone().into() + acc * beta.clone()
                });

                // Compute (α - combined_elt)
                alpha.clone() - combined_elt
            })
            .collect()
    }

    /// Computes the numerator and denominator of the fraction:
    /// `∑(m_i / (α - combined_elements[i]))`, where
    /// `combined_elements[i] = ∑elements[i][n-j] * β^j`
    pub(crate) fn compute_combined_sum_terms<AB, E, M>(
        &self,
        elements: &[Vec<E>],
        multiplicities: &[M],
        alpha: &AB::ExprEF,
        beta: &AB::ExprEF,
    ) -> (AB::ExprEF, AB::ExprEF)
    where
        AB: PermutationAirBuilder,
        E: Into<AB::ExprEF> + Clone,
        M: Into<AB::ExprEF> + Clone,
    {
        if elements.is_empty() {
            return (AB::ExprEF::ZERO, AB::ExprEF::ONE);
        }

        let n = elements.len();

        // Precompute all (α - ∑e_{i, j} β^j) terms
        let terms = self.combine_elements::<AB, E>(elements, alpha, beta);

        // Build prefix products: pref[i] = ∏_{j=0}^{i-1}(α - e_j)
        let mut pref = Vec::with_capacity(n + 1);
        pref.push(AB::ExprEF::ONE);
        for t in &terms {
            pref.push(pref.last().unwrap().clone() * t.clone());
        }

        // Build suffix products: suff[i] = ∏_{j=i}^{n-1}(α - e_j)
        let mut suff = vec![AB::ExprEF::ONE; n + 1];
        for i in (0..n).rev() {
            suff[i] = suff[i + 1].clone() * terms[i].clone();
        }

        // Common denominator is the product of all terms
        let common_denominator = pref[n].clone();

        // Compute numerator: ∑(m_i * ∏_{j≠i}(α - e_j))
        //
        // The product without i is: pref[i] * suff[i+1]
        let numerator = (0..n).fold(AB::ExprEF::ZERO, |acc, i| {
            acc + multiplicities[i].clone().into() * pref[i].clone() * suff[i + 1].clone()
        });

        (numerator, common_denominator)
    }
}

impl LookupProtocol for LogUpGadget {
    fn num_challenges(&self) -> usize {
        2
    }

    /// # Mathematical Details
    ///
    /// The constraint enforces, on every row, the rational identity:
    /// ```text
    /// U_c(r) * f_c[r] - V_c(r) = 0
    /// ```
    ///
    /// where:
    /// - `f_c` is the lookup's fraction column,
    /// - `(V_c, U_c)` is the rational form
    ///
    /// ```text
    /// ∑_i(multiplicities[i] / (α - combined_elements[i])) = V_c / U_c
    /// ```
    ///
    /// - `multiplicities` may be negative,
    /// - `combined_elements[i] = ∑elements[i][n-j] * β^j`.
    fn eval_fraction<AB>(&self, builder: &mut AB, lookup: &Lookup<AB::F>)
    where
        AB: PermutationAirBuilder,
    {
        let Lookup {
            kind: _,
            elements,
            multiplicities,
            column,
        } = lookup;

        assert!(
            elements.len() == multiplicities.len(),
            "Mismatched lengths: elements and multiplicities must have same length"
        );

        let column = *column;

        // First, turn the symbolic expressions into builder expressions, for elements and multiplicities.
        let elements = elements
            .iter()
            .map(|exprs| {
                exprs
                    .iter()
                    .map(|expr| expr.resolve(builder).into())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let multiplicities = multiplicities
            .iter()
            .map(|expr| expr.resolve(builder).into())
            .collect::<Vec<_>>();

        // Access the permutation (aux) table and the per-lookup challenges.
        let permutation = builder.permutation();
        let permutation_challenges = builder.permutation_randomness();

        assert!(
            permutation_challenges.len() >= self.num_challenges() * (column + 1),
            "Insufficient permutation challenges"
        );

        // Challenge for the rational denominator.
        let alpha = permutation_challenges[self.num_challenges() * column];
        // Challenge for combining the lookup tuples.
        let beta = permutation_challenges[self.num_challenges() * column + 1];

        // Read this lookup's fraction column at the current row.
        //
        // Column 0 of the permutation trace is the shared accumulator.
        // So lookup slot `column` lives at permutation-trace column `column + 1`.
        let frac_local: AB::ExprEF = permutation.current(column + 1).unwrap().into();

        // Build the fraction:  ∑ m_i/(α - combined_elements[i])  =  numerator / denominator .
        let (numerator, common_denominator) = self
            .compute_combined_sum_terms::<AB, AB::ExprEF, AB::ExprEF>(
                &elements,
                &multiplicities,
                &alpha.into(),
                &beta.into(),
            );

        // Pin the fraction column to V / U on every row.
        //
        // The identity is cyclic in the trace domain, so it does not need a
        // transition gate. Forcing it on every row also pins the last-row
        // value used by the accumulator's terminal binding.
        builder.assert_zero_ext(common_denominator * frac_local - numerator);
    }

    /// # Mathematical Details
    ///
    /// The accumulator at column 0 of the permutation trace satisfies:
    /// - **first row**: `acc[0] = 0`
    /// - **transition**: `acc[r + 1] - acc[r] - ∑_c f_c[r] = 0`
    /// - **last row**: `terminal - acc[n - 1] - ∑_c f_c[n - 1] = 0`
    ///
    /// `terminal` is provided by the prover and committed once per AIR.
    ///
    /// The cross-AIR sum of committed terminals is checked to be zero after every AIR has been verified.
    fn eval_accumulator<AB>(
        &self,
        builder: &mut AB,
        lookups: &[Lookup<AB::F>],
        terminal: AB::ExprEF,
    ) where
        AB: PermutationAirBuilder,
    {
        // Read accumulator and per-lookup fractions before opening the
        // mutable builder calls below.
        let acc_local: AB::ExprEF;
        let acc_next: AB::ExprEF;
        let row_sum: AB::ExprEF;
        {
            let permutation = builder.permutation();
            assert!(
                permutation.current_slice().len() > lookups.len(),
                "Permutation trace has insufficient width"
            );

            // Accumulator lives at column 0.
            acc_local = permutation.current(0).unwrap().into();
            acc_next = permutation.next(0).unwrap().into();

            // Sum every lookup's fraction at the current row.
            //
            // Lookup slot `column` maps to permutation column `column + 1`.
            row_sum = lookups.iter().fold(AB::ExprEF::ZERO, |sum, lookup| {
                sum + permutation.current(lookup.column + 1).unwrap().into()
            });
        }

        // Anchor acc[0] = 0 at the start.
        builder.when_first_row().assert_zero_ext(acc_local.clone());

        // Transition: acc[r+1] = acc[r] + row_sum[r] for r in 0..n-1.
        builder
            .when_transition()
            .assert_zero_ext(acc_next - acc_local.clone() - row_sum.clone());

        // Terminal binding on the last row: terminal = acc[n-1] + row_sum[n-1].
        builder
            .when_last_row()
            .assert_zero_ext(terminal - acc_local - row_sum);
    }

    fn verify_terminal_sum<EF: Field>(
        &self,
        terminals: &[Option<LookupTerminal<EF>>],
    ) -> Result<(), LookupError> {
        let total = terminals
            .iter()
            .filter_map(|t| t.as_ref())
            .map(|t| t.0)
            .sum::<EF>();

        if !total.is_zero() {
            return Err(LookupError::TerminalSumNonZero);
        }

        Ok(())
    }

    /// The fraction-pin constraint `U_c(r) * f_c[r] - V_c(r) = 0` carries the
    /// highest degree emitted for any one lookup.
    ///
    /// In `U_c`, each combined element `e_i = ∑e_{i, j} β^j` contributes
    /// `(α - e_i)`, so the denominator degree is the sum of all
    /// combined-element degrees.
    ///
    /// In `V_c = ∑(m_i * ∏_{j≠i}(α - e_j))` we take the maximum over `i` of
    /// `deg(m_i)` plus the sum of every other combined-element degree.
    ///
    /// The constraint degree is then: `max(1 + deg(U_c), deg(V_c))`
    fn constraint_degree<F: Field>(&self, lookup: &Lookup<F>) -> usize {
        assert!(lookup.multiplicities.len() == lookup.elements.len());

        let n = lookup.multiplicities.len();

        // Compute degrees in a single pass.
        let mut degs = Vec::with_capacity(n);
        let mut deg_sum = 0;
        for elems in &lookup.elements {
            let deg = elems
                .iter()
                .map(|elt| elt.degree_multiple())
                .max()
                .unwrap_or(0);
            degs.push(deg);
            deg_sum += deg;
        }

        // Compute 1 + degree(denominator).
        let deg_denom_constr = 1 + deg_sum;

        // Compute degree(numerator).
        let deg_num = (0..n)
            .map(|i| lookup.multiplicities[i].degree_multiple() + deg_sum - degs[i])
            .max()
            .unwrap_or(0);

        deg_denom_constr.max(deg_num)
    }

    #[instrument(name = "generate lookup permutation", skip_all, level = "debug")]
    fn generate_permutation<SC: StarkGenericConfig>(
        &self,
        main: &RowMajorMatrix<Val<SC>>,
        preprocessed: &Option<RowMajorMatrix<Val<SC>>>,
        public_values: &[Val<SC>],
        lookups: &[Lookup<Val<SC>>],
        challenges: &[SC::Challenge],
    ) -> (
        RowMajorMatrix<SC::Challenge>,
        Option<LookupTerminal<SC::Challenge>>,
    ) {
        // AIRs without lookups carry no permutation trace and no terminal.
        if lookups.is_empty() {
            return (RowMajorMatrix::new(Vec::new(), 0), None);
        }

        let height = main.height();
        let num_lookups = lookups.len();
        // One accumulator column at index 0, then one fraction column per lookup.
        let width = num_lookups + 1;

        // Validate challenge count matches number of lookups.
        debug_assert_eq!(
            challenges.len(),
            lookups.len() * self.num_challenges(),
            "perm challenge count must be per-lookup"
        );

        // Slot `i` owns fraction column `i + 1`, so indices must be contiguous.
        // Contiguity also implies uniqueness, replacing the old duplicate check.
        //
        // A gap is an out-of-bounds write on untrusted data, e.g. a width-3 trace:
        //
        //     slots [0, 5] → writes col 2 here, but the constraint reads col 6
        //
        // Kept on in release: the scan is O(N) over a handful of lookups.
        for (i, lookup) in lookups.iter().enumerate() {
            assert_eq!(
                lookup.column, i,
                "lookup slot index must match slice position: \
                 lookups[{i}].column = {} (expected {i})",
                lookup.column,
            );
        }

        // 1. PRE-COMPUTE DENOMINATORS
        // We flatten all denominators from all rows/lookups into one giant vector.
        // Order: Row -> Lookup -> Element Tuple
        let denoms_per_row: usize = lookups.iter().map(|l| l.elements.len()).sum();
        let mut lookup_denom_offsets = Vec::with_capacity(lookups.len() + 1);
        lookup_denom_offsets.push(0);
        for l in lookups.iter() {
            lookup_denom_offsets
                .push(lookup_denom_offsets.last().copied().unwrap() + l.elements.len());
        }

        // Hoist the per-lookup random challenges out of the hot per-row loop.
        //
        // Each lookup uses a pair (alpha, beta) that is constant across all rows.
        // - alpha is the rational-denominator challenge,
        // - beta combines tuple elements.
        let lookup_challenges: Vec<(SC::Challenge, SC::Challenge)> = lookups
            .iter()
            .map(|lookup| {
                // Index into the flat challenge array by the lookup's slot index.
                let base = self.num_challenges() * lookup.column;
                (challenges[base], challenges[base + 1])
            })
            .collect();

        // Fused chunk-local batch inversion.
        //
        // Instead of three global passes (fill denoms -> batch invert -> build row sums)
        // with three full-height intermediate arrays alive at once, we process
        // fixed-size chunks end-to-end inside each thread.
        //
        // Per-thread working set: CHUNK_SIZE * denoms_per_row * (16 + 4) bytes,
        // which fits comfortably in L2 cache for typical parameters.
        //
        // Memory layout (flat, row-major within each chunk):
        //
        // ```text
        //   local_denoms:  [ row_0_denom_0 .. row_0_denom_D | row_1_denom_0 .. | ... ]
        //   local_mults:   [ row_0_mult_0  .. row_0_mult_D  | row_1_mult_0  .. | ... ]
        // ```
        //
        // where D = denoms_per_row (total element tuples across all lookups).

        // Matches the internal chunk size of the batch inversion routine.
        //
        // TODO: benchmark this value across different fields and architectures
        // to determine if a more general or adaptive constant is warranted.
        const CHUNK_SIZE: usize = 1024;

        // Aux trace buffer.
        //
        // Layout — row-major, width = num_lookups + 1:
        //
        //     col 0:      acc           ← filled from row_totals (parallel prefix sum, below)
        //     col i + 1:  frac_i = V/U  ← filled per row during the batch-invert chunks
        let mut aux_trace = SC::Challenge::zero_vec(height * width);

        // Per-row sum of every fraction column.
        //
        // Used as the per-row delta when building the accumulator.
        let mut row_totals = SC::Challenge::zero_vec(height);

        aux_trace
            .par_chunks_mut(CHUNK_SIZE * width)
            .zip(row_totals.par_chunks_mut(CHUNK_SIZE))
            .enumerate()
            .for_each(|(chunk_idx, (chunk_aux, chunk_row_totals))| {
                // Derive the absolute row range for this chunk.
                let start_row = chunk_idx * CHUNK_SIZE;
                // The last chunk may be shorter than CHUNK_SIZE.
                let num_rows = chunk_aux.len() / width;

                // Thread-local denominator and multiplicity buffers.
                let mut local_denoms = SC::Challenge::zero_vec(num_rows * denoms_per_row);
                let mut local_mults = Val::<SC>::zero_vec(num_rows * denoms_per_row);

                // Phase 1: Fill denominators and multiplicities for every row in this chunk.
                for local_i in 0..num_rows {
                    let i = start_row + local_i;

                    // Build the two-row window (current, next) for the main trace.
                    // Wraps around at the trace boundary.
                    let local_main_row = main.row_slice(i).unwrap();
                    let next_main_row = main.row_slice((i + 1) % height).unwrap();
                    let main_rows = VerticalPair::new(
                        RowMajorMatrixView::new_row(&local_main_row),
                        RowMajorMatrixView::new_row(&next_main_row),
                    );

                    // Same two-row window for the preprocessed trace, if present.
                    let preprocessed_rows_data = preprocessed.as_ref().map(|prep| {
                        (
                            prep.row_slice(i).unwrap(),
                            prep.row_slice((i + 1) % height).unwrap(),
                        )
                    });
                    let preprocessed_rows = match preprocessed_rows_data.as_ref() {
                        Some((local_preprocessed_row, next_preprocessed_row)) => VerticalPair::new(
                            RowMajorMatrixView::new_row(local_preprocessed_row),
                            RowMajorMatrixView::new_row(next_preprocessed_row),
                        ),
                        // Empty views when there is no preprocessed trace.
                        None => VerticalPair::new(
                            RowMajorMatrixView::new(&[], 0),
                            RowMajorMatrixView::new(&[], 0),
                        ),
                    };

                    // Concrete evaluator: resolves symbolic expressions to field values
                    // using the current row's data.
                    let row_ctx: RowEvalContext<'_, SC> = RowEvalContext::new(
                        main_rows,
                        preprocessed_rows,
                        public_values,
                        challenges,
                        height,
                        i,
                    );

                    // Walk through each lookup's element tuples and fill the flat buffers.
                    let mut offset = local_i * denoms_per_row;
                    for (lookup, &(alpha, beta)) in lookups.iter().zip(lookup_challenges.iter()) {
                        for (j, elts) in lookup.elements.iter().enumerate() {
                            // Combine tuple elements via Horner's method:
                            //   combined = e_0 * beta^{k-1} + e_1 * beta^{k-2} + ... + e_{k-1}
                            // Then store (alpha - combined) as the denominator.
                            let mut iter = elts.iter();
                            let combined_elt = iter.next().map_or(SC::Challenge::ZERO, |first| {
                                iter.fold(
                                    first.resolve(&row_ctx).into(),
                                    |acc: SC::Challenge, e| acc * beta + e.resolve(&row_ctx),
                                )
                            });
                            local_denoms[offset] = alpha - combined_elt;

                            // Store the multiplicity as a base-field element (4 bytes vs 16 for
                            // extension) to keep the buffer small and the later dot product cheap.
                            local_mults[offset] = lookup.multiplicities[j].resolve(&row_ctx);
                            offset += 1;
                        }
                    }
                }

                // Phase 2: Batch-invert all denominators in this chunk.
                // Montgomery's trick: 1 inversion + O(N) multiplications.
                let local_inverses = p3_field::batch_multiplicative_inverse(&local_denoms);
                // Free the denominator buffer immediately to reduce peak memory.
                drop(local_denoms);

                // Phase 3: emit fractions + row totals.
                //
                // For each row in the chunk:
                // - For each lookup, dot-product `(1/d_j) * m_j` over its tuples → `frac = V/U`.
                // - Write `frac` into the fraction column of `aux_trace` (slot `i` → col `i + 1`).
                // - Accumulate `frac` into the row total feeding the prefix-sum below.
                //
                // The inner multiply is extension * base (cheaper than extension * extension).
                //
                // TODO: investigate fusing batch inversion with multiplicity multiplication.
                for (local_i, row_total_slot) in chunk_row_totals.iter_mut().enumerate() {
                    let inv_base = local_i * denoms_per_row;
                    let row_offset = local_i * width;
                    let mut row_total = SC::Challenge::ZERO;
                    for lookup_idx in 0..lookups.len() {
                        // Slice out the range of denominators belonging to this lookup.
                        let start = lookup_denom_offsets[lookup_idx];
                        let end = lookup_denom_offsets[lookup_idx + 1];
                        let inv_slice = &local_inverses[inv_base + start..inv_base + end];
                        let mult_slice = &local_mults[inv_base + start..inv_base + end];
                        // Dot product: sum of (1 / denominator) * multiplicity for each tuple.
                        let frac =
                            dot_product(inv_slice.iter().copied(), mult_slice.iter().copied());
                        // Lookup slot `lookup_idx` lives at fraction column `lookup_idx + 1`.
                        chunk_aux[row_offset + lookup_idx + 1] = frac;
                        row_total += frac;
                    }
                    *row_total_slot = row_total;
                }
            });

        // Build the accumulator column from the per-row totals.
        //
        // The accumulator is an *exclusive* prefix sum of `row_totals`:
        //
        //   acc[0] = 0
        //   acc[i] = row_totals[0] + row_totals[1] + … + row_totals[i-1]
        //
        // A naive serial loop would be O(height). Instead we use a three-phase
        // parallel prefix sum, splitting the work across threads:
        //
        //   Phase A — Each thread computes a local prefix sum on its chunk.
        //   Phase B — A tiny sequential pass (one entry per thread) combines
        //             the chunk totals into global offsets.
        //   Phase C — Each thread adds its global offset back into its chunk.
        //
        // After the three phases, `row_totals` holds an *inclusive* prefix sum.
        // The accumulator column then reads `acc[i+1] = row_totals[i]`, leaving
        // `acc[0] = 0` from the initial zeroed buffer.
        let num_threads = current_num_threads();
        let chunk_size = height.div_ceil(num_threads);

        // Phase A — Local inclusive prefix sums, one chunk per thread.
        row_totals.par_chunks_mut(chunk_size).for_each(|chunk| {
            for i in 1..chunk.len() {
                chunk[i] += chunk[i - 1];
            }
        });

        // Phase B — Combine chunk totals into cumulative offsets.
        // Only as many entries as there are chunks (one per thread), so this is tiny.
        let mut offsets = SC::Challenge::zero_vec(height.div_ceil(chunk_size));
        for i in 1..offsets.len() {
            offsets[i] = offsets[i - 1] + row_totals[i * chunk_size - 1];
        }

        // Phase C — Fold global offsets back into each chunk.
        row_totals
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let offset = offsets[chunk_idx];
                if !offset.is_zero() {
                    for val in chunk.iter_mut() {
                        *val += offset;
                    }
                }
            });

        // At this point `row_totals` holds the inclusive prefix sum.
        //
        // The committed terminal equals the full sum across the trace.
        let terminal = LookupTerminal(row_totals[height - 1]);

        // Write the exclusive prefix sum into the accumulator column of aux_trace:
        //
        // - Row 0 is already zero from the buffer's initialisation;
        // - Row r > 0 gets the inclusive sum of all previous rows.
        aux_trace
            .par_chunks_mut(width)
            .skip(1)
            .enumerate()
            .for_each(|(i, row)| {
                row[0] = row_totals[i];
            });

        (RowMajorMatrix::new(aux_trace, width), Some(terminal))
    }
}
