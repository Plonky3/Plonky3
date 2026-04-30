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
use crate::types::{Kind, Lookup, LookupData, LookupError};

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
/// This is implemented using a running sum auxiliary column `s` that accumulates:
/// ```text
/// s[i+1] = s[i] + ∑(m_a/(α - a)) - ∑(m_b/(α - b))
/// ```
///
/// Note that we do not differentiate between `a` and `b` in the implementation:
/// we simply have a list of `elements` with possibly negative `multiplicities`.
///
/// Constraints are defined as:
/// - **Initial Constraint**: `s[0] = 0`
/// - **Transition Constraint**: `s[i+1] = s[i] + contribution[i]`
/// - **Final Constraint**: `s[n-1] + contribution[n-1] = 0`
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
    /// `combined_elements[i] = ∑elements[i][n-j] * β^j
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

    /// Evaluates the transition and boundary constraints for a lookup argument.
    ///
    /// # Arguments:
    /// * builder - The AIR builder to construct expressions.
    /// * lookup - The lookup context containing:
    ///     * the kind of lookup (local or global),
    ///     * elements,
    ///     * multiplicities,
    ///     * and auxiliary column index.
    /// * opt_cumulative_sum - Optional expected cumulative value for global lookups. For local lookups, this should be `None`.
    fn eval_update<AB>(
        &self,
        builder: &mut AB,
        lookup: &Lookup<AB::F>,
        opt_cumulative_sum: Option<AB::ExprEF>,
    ) where
        AB: PermutationAirBuilder,
    {
        let Lookup {
            kind,
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

        // Access the permutation (aux) table. It carries the running sum column `s`.
        let permutation = builder.permutation();

        let permutation_challenges = builder.permutation_randomness();

        assert!(
            permutation_challenges.len() >= self.num_challenges() * (column + 1),
            "Insufficient permutation challenges"
        );

        // Challenge for the running sum.
        let alpha = permutation_challenges[self.num_challenges() * column];
        // Challenge for combining the lookup tuples.
        let beta = permutation_challenges[self.num_challenges() * column + 1];

        assert!(
            permutation.current_slice().len() > column,
            "Permutation trace has insufficient width"
        );

        // Read s[i] from the local row at the specified column.
        let s_local = permutation.current(column).unwrap().into();
        // Read s[i+1] from the next row (or a zero-padded view on the last row).
        let s_next = permutation.next(column).unwrap().into();

        // Anchor s[0] = 0 at the start.
        //
        // Avoids a high-degree boundary constraint.
        // Telescoping is enforced by the last-row check (s[n−1] + contribution[n-1] = 0).
        // This keeps aux and main traces aligned in length.
        builder.when_first_row().assert_zero_ext(s_local.clone());

        // Build the fraction:  ∑ m_i/(α - combined_elements[i])  =  numerator / denominator .
        let (numerator, common_denominator) = self
            .compute_combined_sum_terms::<AB, AB::ExprEF, AB::ExprEF>(
                &elements,
                &multiplicities,
                &alpha.into(),
                &beta.into(),
            );

        if let Some(cumulative_sum) = opt_cumulative_sum {
            // If there is a `cumulative_sum`, we are in a global lookup update.
            assert!(
                matches!(kind, Kind::Global(_)),
                "Expected cumulated value provided for a non-global lookup"
            );

            // Transition constraint:
            builder.when_transition().assert_zero_ext(
                (s_next - s_local.clone()) * common_denominator.clone() - numerator.clone(),
            );

            // Final constraint:
            let final_val = (cumulative_sum - s_local) * common_denominator - numerator;
            builder.when_last_row().assert_zero_ext(final_val);
        } else {
            // If we don't have a `cumulative_sum`, we are in a local lookup update.
            assert!(
                matches!(kind, Kind::Local),
                "No expected cumulated value provided for a global lookup"
            );

            // If we are in a local lookup, the previous transition constraint doesn't have to be limited to transition rows:
            // - we are already ensuring that the first row is 0,
            // - at point `g^{n - 1}` (where `n` is the domain size), the next point is `g^0`, so that the constraint still holds
            // on the last row.
            builder.assert_zero_ext((s_next - s_local) * common_denominator - numerator);
        }
    }
}

impl LookupProtocol for LogUpGadget {
    fn num_challenges(&self) -> usize {
        2
    }

    /// # Mathematical Details
    /// The constraint enforces:
    /// ```text
    /// ∑_i(multiplicities[i] / (α - combined_elements[i])) = 0
    /// ```
    ///
    /// where `multiplicities` can be negative, and
    /// `combined_elements[i] = ∑elements[i][n-j] * β^j`.
    ///
    /// This is implemented using a running sum column that should sum to zero.
    fn eval_local<AB>(&self, builder: &mut AB, lookup: &Lookup<AB::F>)
    where
        AB: PermutationAirBuilder,
    {
        if let Kind::Global(_) = lookup.kind {
            panic!("Global lookups are not supported in local evaluation")
        }

        self.eval_update(builder, lookup, None);
    }

    /// # Mathematical Details
    /// The constraint enforces:
    /// ```text
    /// ∑_i(multiplicities[i] / (α - combined_elements[i])) = `cumulative_sum`
    /// ```
    ///
    /// where `multiplicities` can be negative, and
    /// `combined_elements[i] = ∑elements[i][n-j] * β^j`.
    ///
    /// `cumulative_sum` is provided by the prover, and the sum of all `cumulative_sum` for this global interaction
    /// should be 0. The latter is checked as the final step, after all AIRS have been verified.
    ///
    /// This is implemented using a running sum column that should sum to `cumulative_sum`.
    fn eval_global<AB>(&self, builder: &mut AB, lookup: &Lookup<AB::F>, cumulative_sum: AB::ExprEF)
    where
        AB: PermutationAirBuilder,
    {
        self.eval_update(builder, lookup, Some(cumulative_sum));
    }

    fn verify_global_sum<EF: Field>(&self, cumulative_sums: &[EF]) -> Result<(), LookupError> {
        let total = cumulative_sums.iter().copied().sum::<EF>();

        if !total.is_zero() {
            // We set the name associated to the lookup to None because we don't have access to the actual name here.
            // The actual name will be set in the verifier directly.
            return Err(LookupError::GlobalCumulativeMismatch(None));
        }

        Ok(())
    }

    /// We need to compute the degree of the transition constraint,
    /// as it is the constraint with highest degree:
    /// `(s[n + 1] - s[n]) * common_denominator - numerator = 0`
    ///
    /// But in `common_denominator`, each combined element e_i = ∑e_{i, j} β^j
    /// contributes (α - e_i). So we need to sum the degree of all
    /// combined elements to find the degree of the common denominator.
    ///
    /// `numerator = ∑(m_i * ∏_{j≠i}(α - e_j))`, where the e_j are the combined elements.
    /// So we have to compute the max of all m_i * ∏_{j≠i}(α - e_j).
    ///
    /// The constraint degree is then:
    /// `1 + max(deg(numerator), deg(common_denominator))`
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
        lookup_data: &mut [LookupData<SC::Challenge>],
        challenges: &[SC::Challenge],
    ) -> RowMajorMatrix<SC::Challenge> {
        let height = main.height();
        let width = lookups.len();

        // Validate challenge count matches number of lookups.
        debug_assert_eq!(
            challenges.len(),
            lookups.len() * self.num_challenges(),
            "perm challenge count must be per-lookup"
        );

        // Enforce uniqueness of auxiliary column indices across lookups.
        #[cfg(debug_assertions)]
        {
            use alloc::collections::btree_set::BTreeSet;

            let mut seen = BTreeSet::new();
            for ctx in lookups {
                let a = ctx.column;
                if !seen.insert(a) {
                    panic!("duplicate aux column index {a} across lookups");
                }
            }
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
        let num_lookups = lookups.len();

        // Hoist the per-lookup random challenges out of the hot per-row loop.
        // Each lookup uses a pair (alpha, beta) that is constant across all rows.
        // alpha is the running-sum challenge, beta combines tuple elements.
        let lookup_challenges: Vec<(SC::Challenge, SC::Challenge)> = lookups
            .iter()
            .map(|lookup| {
                // Index into the flat challenge array by the lookup's auxiliary column index.
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

        // Output buffer: one row-sum per (row, lookup) pair, row-major order.
        let mut row_sums = SC::Challenge::zero_vec(height * num_lookups);

        row_sums
            .par_chunks_mut(CHUNK_SIZE * num_lookups)
            .enumerate()
            .for_each(|(chunk_idx, chunk_row_sums)| {
                // Derive the absolute row range for this chunk.
                let start_row = chunk_idx * CHUNK_SIZE;
                // The last chunk may be shorter than CHUNK_SIZE.
                let num_rows = chunk_row_sums.len() / num_lookups;

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

                // Phase 3: Accumulate per-row sums: sum_j( inverse_j * multiplicity_j ) grouped by lookup.
                //
                // The multiplication is extension * base (cheaper than extension * extension).
                //
                // TODO: investigate fusing batch inversion with multiplicity multiplication.
                for local_i in 0..num_rows {
                    let inv_base = local_i * denoms_per_row;
                    for (lookup_idx, _lookup) in lookups.iter().enumerate() {
                        // Slice out the range of denominators belonging to this lookup.
                        let start = lookup_denom_offsets[lookup_idx];
                        let end = lookup_denom_offsets[lookup_idx + 1];
                        let inv_slice = &local_inverses[inv_base + start..inv_base + end];
                        let mult_slice = &local_mults[inv_base + start..inv_base + end];
                        // Dot product: sum of (1 / denominator) * multiplicity for each tuple.
                        chunk_row_sums[local_i * num_lookups + lookup_idx] =
                            dot_product(inv_slice.iter().copied(), mult_slice.iter().copied());
                    }
                }
            });

        let mut aux_trace = SC::Challenge::zero_vec(height * width);
        let mut permutation_counter = 0;

        // Each lookup column gets its own running sum.
        // Since these columns are independent, we build them one at a time.
        //
        // The running sum is an *exclusive* prefix sum of the per-row contributions:
        //
        //   s[0] = 0
        //   s[i] = row_sum[0] + row_sum[1] + … + row_sum[i-1]
        //
        // A naive serial loop would be O(height). Instead we use a three-phase
        // parallel prefix sum, splitting the work across threads:
        //
        //   Phase A — Each thread computes a local prefix sum on its chunk.
        //   Phase B — A tiny sequential pass (one entry per thread) combines
        //             the chunk totals into global offsets.
        //   Phase C — Each thread adds its global offset back into its chunk.
        //
        // After the three phases, we have an *inclusive* prefix sum.
        // Shifting by one position turns it into the exclusive sum we need.
        let num_threads = current_num_threads();
        let chunk_size = height.div_ceil(num_threads);

        // Reuse a single buffer across all lookup columns to avoid re-allocating on every iteration.
        let mut prefix = SC::Challenge::zero_vec(height);

        for (lookup_idx, lookup) in lookups.iter().enumerate() {
            let aux_column = lookup.column;

            // Fill the buffer with this column's per-row contributions.
            for (i, val) in prefix.iter_mut().enumerate() {
                *val = row_sums[i * num_lookups + lookup_idx];
            }

            // Phase A — Local inclusive prefix sums, one chunk per thread.
            prefix.par_chunks_mut(chunk_size).for_each(|chunk| {
                for i in 1..chunk.len() {
                    chunk[i] += chunk[i - 1];
                }
            });

            // Phase B — Combine chunk totals into cumulative offsets.
            // Only as many entries as there are chunks (one per thread), so this is tiny.
            let mut offsets = SC::Challenge::zero_vec(height.div_ceil(chunk_size));
            for i in 1..offsets.len() {
                offsets[i] = offsets[i - 1] + prefix[i * chunk_size - 1];
            }

            // Phase C — Fold global offsets back into each chunk.
            prefix
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

            // At this point we hold an *inclusive* prefix sum.
            //
            // The auxiliary trace needs the *exclusive* version (shifted right by one, starting at zero).
            //
            // - Row 0 is already zero from initialization;
            // - Each subsequent row gets the inclusive sum of all *previous* rows.
            aux_trace
                .par_chunks_mut(width)
                .skip(1)
                .enumerate()
                .for_each(|(i, row)| {
                    row[aux_column] = prefix[i];
                });

            // For global lookups, record the total sum across all rows.
            if matches!(lookup.kind, Kind::Global(_)) {
                lookup_data[permutation_counter].cumulative_sum = prefix[height - 1];
                permutation_counter += 1;
            }
        }

        // Check that we have updated all `lookup_data` entries.
        debug_assert_eq!(permutation_counter, lookup_data.len());
        RowMajorMatrix::new(aux_trace, width)
    }
}
