//! `TwoAdicStirPcs`: implementing the [`Pcs`] trait using STIR.
//!
//! **Commit**: given trace matrices evaluated on two-adic cosets, compute their LDEs and commit
//! to those LDEs using an `InputMmcs`.
//!
//! **Open**: alpha-batch quotient polynomials `(f_i(z) - f_i(x)) / (z - x)` into per-height
//! reduced-opening polynomials, then run STIR on each distinct LDE-height bucket.
//! The prover returns the deduplicated first-round STIR query indices alongside the IOP
//! proof; at those positions the prover also opens the input LDE matrices (via `InputMmcs`)
//! so the verifier can confirm the reduced-opening polynomial is correctly derived from the
//! committed inputs.
//!
//! **Verify**: replay the same alpha-batching from the opening values, then for each height
//! bucket call [`verify_stir_with_external_initial`]. STIR's initial oracle *is* the reduced
//! opening, which the transcript already pins through the input commitments, the claimed
//! values, and `alpha`, so it is never committed a second time: whenever STIR needs its
//! queried fibers, the verifier rebuilds them from the input MMCS openings at exactly the
//! positions STIR sampled. No hand-mirrored transcript replay is needed.

use alloc::borrow::Cow;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    BasedVectorSpace, ExtensionField, PackedFieldExtension, TwoAdicField,
    batch_multiplicative_inverse,
};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixCow, RowMajorMatrixView};
use p3_matrix::interpolation::{Interpolate, compute_adjusted_weights};
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::config::{StirConfig, StirParameters};
use crate::proof::StirProof;
use crate::prover::prove_stir_from_external_codeword;
use crate::verifier::{StirError, verify_stir_with_external_initial};

/// Batched openings of one input commitment's LDE matrices at the STIR-derived query
/// positions for one LDE-height bucket.
///
/// One multi-opening proof authenticates every opened row together, so sibling digests
/// shared between the bucket's `(query, fiber column)` positions travel once.
///
/// `None` when the commitment has no matrix at this bucket's height.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Val: Serialize, InputMmcs::MultiProof: Serialize",
    deserialize = "Val: Deserialize<'de>, InputMmcs::MultiProof: Deserialize<'de>"
))]
pub struct InputOpenings<Val: Send + Sync + Clone, InputMmcs: Mmcs<Val>> {
    /// `opened_values[k][m]` is the opened fiber-grouped row of matrix `m` at the `k`-th
    /// queried position, in the same query order the prover and verifier both derive from
    /// public data. Each such row concatenates the `2^log_folding_factor` LDE rows of one
    /// fiber, ordered by the bit-reversal of the fiber column index.
    pub opened_values: Vec<Vec<Vec<Val>>>,
    /// Compact multi-opening proof authenticating every row at once.
    pub opening_proof: InputMmcs::MultiProof,
}

/// One LDE-height class of a commitment: its own Merkle tree over the matrices at that height.
struct HeightClass<Val: Send + Sync + Clone, InputMmcs: Mmcs<Val>> {
    data: InputMmcs::ProverData<RowMajorMatrix<Val>>,
    /// Column count of each matrix in this class before fiber grouping.
    widths: Vec<usize>,
    lde_height: usize,
}

/// Prover data for [`TwoAdicStirPcs`].
///
/// Matrices are committed in fiber-grouped form — each Merkle leaf holds
/// `2^log_folding_factor` consecutive bit-reversed LDE rows, exactly the rows one first-round
/// STIR query reads — and grouped into one tree per distinct LDE height, in descending height
/// order. STIR runs an independent sub-proof per height bucket, so a single shared tree would
/// force every bucket's openings to carry the rows of every *other* height as well, purely to
/// recompute the authentication path. `placement` maps each matrix, in the order the caller
/// committed them, to its `(class, index within class)`.
pub struct StirProverData<Val: Send + Sync + Clone, InputMmcs: Mmcs<Val>> {
    classes: Vec<HeightClass<Val, InputMmcs>>,
    placement: Vec<(usize, usize)>,
}

/// Split fiber-grouped LDEs into one height class per distinct LDE height, descending.
///
/// `heights` and `widths` are the pre-grouping LDE heights and column counts, in caller order.
fn commit_by_height_class<Val, InputMmcs>(
    input_mmcs: &InputMmcs,
    grouped: Vec<RowMajorMatrix<Val>>,
    heights: Vec<usize>,
    widths: Vec<usize>,
) -> (Vec<InputMmcs::Commitment>, StirProverData<Val, InputMmcs>)
where
    Val: Send + Sync + Clone,
    InputMmcs: Mmcs<Val>,
{
    let mut class_heights = heights.clone();
    class_heights.sort_unstable();
    class_heights.dedup();
    class_heights.reverse();

    let mut class_mats = vec![Vec::new(); class_heights.len()];
    let mut class_widths = vec![Vec::new(); class_heights.len()];
    let mut placement = Vec::with_capacity(grouped.len());
    for ((mat, height), width) in grouped.into_iter().zip(heights).zip(widths) {
        let class = class_heights
            .iter()
            .position(|&h| h == height)
            .expect("every height is one of the distinct heights");
        placement.push((class, class_mats[class].len()));
        class_mats[class].push(mat);
        class_widths[class].push(width);
    }

    let (commitments, classes) = izip!(class_mats, class_widths, class_heights)
        .map(|(mats, widths, lde_height)| {
            let (commitment, data) = input_mmcs.commit(mats);
            (
                commitment,
                HeightClass {
                    data,
                    widths,
                    lde_height,
                },
            )
        })
        .unzip();

    (commitments, StirProverData { classes, placement })
}

/// Reinterpret a bit-reversed LDE as a matrix whose rows are whole STIR fibers.
///
/// A first-round query at fold-domain index `j` reads the LDE rows
/// `reverse_bits_len(j + l * 2^(log_h - log_arity), log_h)` for `l < 2^log_arity`. Writing
/// `j` into the low `log_h - log_arity` bits and `l` into the high `log_arity` bits, the
/// reversal maps that set onto the contiguous block
/// `[rev(j) * 2^log_arity, (rev(j) + 1) * 2^log_arity)`, so grouping is a pure reshape of the
/// same buffer: no row ever straddles two leaves.
fn group_fiber_rows<Val: Clone + Send + Sync>(
    lde: RowMajorMatrix<Val>,
    log_arity: usize,
) -> RowMajorMatrix<Val> {
    let width = lde.width() << log_arity;
    RowMajorMatrix::new(lde.values, width)
}

/// Recover views of the committed matrices in their pre-grouping shape and caller order.
fn lde_views<'a, Val: Send + Sync + Clone, InputMmcs: Mmcs<Val>>(
    input_mmcs: &InputMmcs,
    prover_data: &'a StirProverData<Val, InputMmcs>,
) -> Vec<RowMajorMatrixView<'a, Val>> {
    let per_class: Vec<_> = prover_data
        .classes
        .iter()
        .map(|class| input_mmcs.get_matrices(&class.data))
        .collect();
    prover_data
        .placement
        .iter()
        .map(|&(class, index)| {
            RowMajorMatrixView::new(
                per_class[class][index].values.as_slice(),
                prover_data.classes[class].widths[index],
            )
        })
        .collect()
}

/// A polynomial commitment scheme using STIR to generate opening proofs.
#[derive(Clone, Debug)]
pub struct TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs> {
    dft: Dft,
    input_mmcs: InputMmcs,
    stir: StirParameters<StirMmcs>,
    _phantom: PhantomData<Val>,
}

impl<Val, Dft, InputMmcs, StirMmcs> TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs> {
    pub const fn new(dft: Dft, input_mmcs: InputMmcs, stir: StirParameters<StirMmcs>) -> Self {
        Self {
            dft,
            input_mmcs,
            stir,
            _phantom: PhantomData,
        }
    }
}

impl<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val, Error: Sync + Debug>,
    StirMmcs: Mmcs<Challenge>,
    Challenge: ExtensionField<Val> + TwoAdicField + BasedVectorSpace<Val>,
    Challenger: FieldChallenger<Val>
        + CanObserve<InputMmcs::Commitment>
        + CanObserve<StirMmcs::Commitment>
        + GrindingChallenger<Witness = Val>
        + CanSampleUniformBits<Val>
        + Clone,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    /// One Merkle root per distinct LDE height, in descending height order.
    type Commitment = Vec<InputMmcs::Commitment>;
    type ProverData = StirProverData<Val, InputMmcs>;
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixCow<'a, Val>>;
    /// Proof structure: one entry per distinct LDE-height bucket (descending).
    ///
    /// Each bucket contains:
    /// - `stir_proof`: the STIR IOP proof for that bucket (per-round IOP messages; the initial
    ///   oracle is external, so neither its commitment nor its openings appear). The
    ///   first-round query indices are NOT serialized — the verifier re-derives them from the
    ///   transcript.
    /// - `input_openings[commit_idx]`: one shared multi-opening proof for that commitment's
    ///   rows at the bucket's first-round STIR fiber positions, in the same sorted-by-index
    ///   order the verifier reconstructs. `None` if the commitment has no matrices at this
    ///   bucket's LDE height.
    type Proof = Vec<(
        StirProof<Challenge, StirMmcs, Val>,
        Vec<Option<InputOpenings<Val, InputMmcs>>>,
    )>;
    type Error = StirError<StirMmcs::Error, InputMmcs::Error>;

    const ZK: bool = false;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        TwoAdicMultiplicativeCoset::new(Val::ONE, log2_strict_usize(degree)).unwrap()
    }

    fn log_max_lde_height(&self) -> usize {
        Val::TWO_ADICITY.saturating_sub(self.stir.log_blowup)
    }

    #[instrument(name = "STIR PCS commit", skip_all)]
    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let min_height = 1usize << self.stir.log_folding_factor;
        let mut widths = Vec::new();
        let mut heights = Vec::new();
        let ldes: Vec<_> = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                assert!(
                    evals.height() >= min_height,
                    "STIR PCS: matrix height {} is below the minimum of 2^{} (= {}) required \
                     by log_folding_factor = {}. Pad the matrix to at least this height before \
                     committing, or lower log_folding_factor.",
                    evals.height(),
                    self.stir.log_folding_factor,
                    min_height,
                    self.stir.log_folding_factor,
                );
                let shift = Val::GENERATOR / domain.shift();
                let lde = self
                    .dft
                    .coset_lde_batch(evals, self.stir.log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix();
                widths.push(lde.width());
                heights.push(lde.height());
                group_fiber_rows(lde, self.stir.log_folding_factor)
            })
            .collect();
        commit_by_height_class(&self.input_mmcs, ldes, heights, widths)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let (class, index) = prover_data.placement[idx];
        let class_data = &prover_data.classes[class];
        let grouped = self.input_mmcs.get_matrices(&class_data.data)[index];
        let lde = RowMajorMatrixView::new(grouped.values.as_slice(), class_data.widths[index]);
        if domain.shift() == Val::GENERATOR && lde.height() >= domain.size() {
            let width = lde.width();
            let values: &'a [Val] = lde.values;
            return RowMajorMatrixView::new(&values[..domain.size() * width], width)
                .as_cow()
                .bit_reverse_rows();
        }
        let poly_height = lde.height() >> self.stir.log_blowup;
        let lde_mat = lde.bit_reverse_rows().to_row_major_matrix();
        let mut coeffs = self.dft.coset_idft_batch(lde_mat, Val::GENERATOR);
        let width = coeffs.width();
        coeffs.values.truncate(poly_height * width);
        coeffs.values.resize(domain.size() * width, Val::ZERO);
        let result = self
            .dft
            .coset_dft_batch(coeffs, domain.shift())
            .to_row_major_matrix();
        let result_width = result.width();
        RowMajorMatrixCow::new(Cow::Owned(result.values), result_width).bit_reverse_rows()
    }

    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
        _num_chunks: usize,
    ) -> Vec<RowMajorMatrix<Val>> {
        let min_height = 1usize << self.stir.log_folding_factor;
        evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert!(
                    evals.height() >= min_height,
                    "STIR PCS quotient: matrix height {} is below 2^{} required by \
                     log_folding_factor = {}.",
                    evals.height(),
                    self.stir.log_folding_factor,
                    self.stir.log_folding_factor,
                );
                let shift = Val::GENERATOR / domain.shift();
                self.dft
                    .coset_lde_batch(evals, self.stir.log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect()
    }

    fn commit_ldes(&self, ldes: Vec<RowMajorMatrix<Val>>) -> (Self::Commitment, Self::ProverData) {
        let min_lde_height = 1usize << (self.stir.log_folding_factor + self.stir.log_blowup);
        let mut widths = Vec::with_capacity(ldes.len());
        let mut heights = Vec::with_capacity(ldes.len());
        let grouped: Vec<_> = ldes
            .into_iter()
            .map(|lde| {
                assert!(
                    lde.height() >= min_lde_height,
                    "STIR PCS: pre-computed LDE height {} is below 2^{} (= {}) required by \
                     log_folding_factor + log_blowup = {} + {}.",
                    lde.height(),
                    self.stir.log_folding_factor + self.stir.log_blowup,
                    min_lde_height,
                    self.stir.log_folding_factor,
                    self.stir.log_blowup,
                );
                widths.push(lde.width());
                heights.push(lde.height());
                group_fiber_rows(lde, self.stir.log_folding_factor)
            })
            .collect();
        commit_by_height_class(&self.input_mmcs, grouped, heights, widths)
    }

    #[instrument(name = "STIR PCS open", skip_all)]
    fn open(
        &self,
        commitment_data_with_opening_points: Vec<(&Self::ProverData, Vec<Vec<Challenge>>)>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        // Step 1: Compute evaluations at opening points using Lagrange interpolation.
        let mats_and_points: Vec<_> = commitment_data_with_opening_points
            .iter()
            .map(|(data, points)| (lde_views(&self.input_mmcs, data), points))
            .collect();

        let (global_max_height, global_max_width) = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats.iter().map(|m| (m.height(), m.width())))
            .reduce(|(hmax, wmax), (h, w)| (hmax.max(h), wmax.max(w)))
            .expect("No matrices supplied");
        let log_global_max_height = log2_strict_usize(global_max_height);

        // Coset for the LDE: `GENERATOR * H` in bit-reversed order.
        let coset: Vec<Val> = {
            let coset =
                TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_global_max_height).unwrap();
            let mut pts = coset.iter().collect();
            reverse_slice_index_bits(&mut pts);
            pts
        };

        let inv_denoms = compute_inverse_denominators::<Val, Challenge>(&mats_and_points, &coset);

        // Precompute adjusted barycentric weights once per opening point.
        // adjusted[i] = 1/(z - x_i) - 1/z, reused across all matrices opened at z.
        let adjusted_weights: LinearMap<Challenge, Vec<Challenge>> = inv_denoms
            .iter()
            .map(|(point, denoms)| (*point, compute_adjusted_weights(*point, denoms)))
            .collect();

        let all_opened_values: OpenedValues<Challenge> = mats_and_points
            .iter()
            .map(|(mats, points)| {
                izip!(mats.iter(), points.iter())
                    .map(|(mat, points_for_mat)| {
                        let h = mat.height() >> self.stir.log_blowup;
                        let (low_coset, _) = mat.split_rows(h);

                        points_for_mat
                            .iter()
                            .map(|&point| {
                                // Slice the precomputed adjusted weights to match this matrix's height.
                                // Zero-allocation hot path: straight to the SIMD dot product.
                                let adj = &adjusted_weights.get(&point).unwrap()[..h];
                                let ys = low_coset.interpolate_coset_with_precomputation(
                                    Val::GENERATOR,
                                    point,
                                    adj,
                                );
                                challenger.observe_algebra_slice(&ys);
                                ys
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec();

        // Step 2: Alpha-batch into a single reduced-opening vector.
        let alpha: Challenge = challenger.sample_algebra_element();
        let packed_alpha_powers =
            Challenge::ExtensionPacking::packed_ext_powers_capped(alpha, global_max_width)
                .collect_vec();
        let alpha_powers: Vec<Challenge> =
            Challenge::ExtensionPacking::to_ext_iter(packed_alpha_powers.iter().copied())
                .collect_vec();

        // `reduced[log_h]`: alpha-batched reduced-opening at log-height `log_h`. Keyed by
        // arbitrary `log_h` so the structure scales with the field's two-adicity rather than
        // a hardcoded array bound.
        let mut reduced_openings: alloc::collections::BTreeMap<usize, Vec<Challenge>> =
            alloc::collections::BTreeMap::new();
        let mut num_reduced: alloc::collections::BTreeMap<usize, usize> =
            alloc::collections::BTreeMap::new();

        for ((mats, points), opened_vals) in mats_and_points.iter().zip(&all_opened_values) {
            for ((mat, points_for_mat), opened_for_mat) in
                izip!(mats.iter(), points.iter()).zip(opened_vals.iter())
            {
                let log_h = log2_strict_usize(mat.height());
                let ro = reduced_openings
                    .entry(log_h)
                    .or_insert_with(|| vec![Challenge::ZERO; mat.height()]);

                // Precompute alpha-batched row values for this matrix (reused per point).
                let p_x_vec: Vec<Challenge> = mat
                    .rowwise_packed_dot_product::<Challenge>(&packed_alpha_powers)
                    .collect();

                for (point, ys) in points_for_mat.iter().zip(opened_for_mat.iter()) {
                    let height_count = num_reduced.entry(log_h).or_insert(0);
                    let alpha_pow_offset = alpha.exp_u64(*height_count as u64);
                    *height_count += ys.len();

                    let full_height = mat.height();
                    let inv_denom = &inv_denoms.get(point).unwrap()[..full_height];

                    let y_combined: Challenge = ys
                        .iter()
                        .zip(alpha_powers.iter())
                        .map(|(&y, &ap)| y * ap)
                        .sum();

                    ro.par_iter_mut()
                        .zip(inv_denom.par_iter().zip(p_x_vec.par_iter()))
                        .for_each(|(ro_val, (&inv_d, &p_x))| {
                            *ro_val += alpha_pow_offset * (p_x - y_combined) * inv_d;
                        });
                }
            }
        }

        // Step 3: For each non-empty height bucket (descending), run STIR on the bucket's
        // reduced opening and bind the input MMCS. Each distinct LDE height gets its own
        // STIR sub-proof. BTreeMap iterates in ascending key order, so reverse for descending.
        let mut bucket_proofs = Vec::new();

        let bucket_log_heights: Vec<usize> = reduced_openings.keys().rev().copied().collect();
        for log_h in bucket_log_heights {
            let ro = reduced_openings
                .remove(&log_h)
                .expect("present by construction");
            let bucket_height = 1usize << log_h;

            let mut ro_natural = ro;
            reverse_slice_index_bits(&mut ro_natural);

            let log_stir_degree = log_h.saturating_sub(self.stir.log_blowup).max(1);
            let stir_config = StirConfig::<Val, Challenge, StirMmcs, Challenger>::new(
                log_stir_degree,
                self.stir.clone(),
            );

            let (stir_proof, first_round_query_indices) =
                prove_stir_from_external_codeword(&stir_config, ro_natural, &self.dft, challenger);

            // Input binding for this bucket. Folding factor is constant across rounds.
            let log_arity0 = stir_config.log_folding_factor;

            let input_openings: Vec<Option<InputOpenings<Val, InputMmcs>>> =
                commitment_data_with_opening_points
                    .iter()
                    .map(|(data, _)| {
                        // Only this bucket's height class is opened; the other classes live in
                        // their own trees and never appear in this bucket's proof.
                        let class = data
                            .classes
                            .iter()
                            .find(|class| class.lde_height == bucket_height)?;

                        // Every matrix in the class shares the bucket's height, so the grouped
                        // row index is the query index itself: one index per query, and the
                        // whole fiber lives in that single row.
                        let q_globals: Vec<usize> = first_round_query_indices
                            .iter()
                            .map(|&j| reverse_bits_len(j, log_h - log_arity0))
                            .collect();

                        let (opened_values, opening_proof) =
                            self.input_mmcs.open_multi_batch(&q_globals, &class.data);
                        Some(InputOpenings {
                            opened_values,
                            opening_proof,
                        })
                    })
                    .collect();

            bucket_proofs.push((stir_proof, input_openings));
        }

        (all_opened_values, bucket_proofs)
    }

    #[instrument(name = "STIR PCS verify", skip_all)]
    fn verify(
        &self,
        commitments_with_opening_points: Vec<(
            Self::Commitment,
            Vec<(Self::Domain, Vec<(Challenge, Vec<Challenge>)>)>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // Observe all opened values to keep the transcript in sync.
        for (_, domain_claims) in &commitments_with_opening_points {
            for (_, point_claims) in domain_claims {
                for (_, opened_vals) in point_claims {
                    challenger.observe_algebra_slice(opened_vals);
                }
            }
        }

        let alpha: Challenge = challenger.sample_algebra_element();

        // Determine the set of distinct LDE-height buckets (descending) from the public domains.
        // Must match the prover's bucket iteration order.
        let bucket_log_heights: Vec<usize> = {
            let mut seen = BTreeSet::new();
            for (_, domain_claims) in &commitments_with_opening_points {
                for (domain, _) in domain_claims {
                    seen.insert(log2_strict_usize(domain.size() << self.stir.log_blowup));
                }
            }
            seen.into_iter().rev().collect()
        };

        if proof.len() != bucket_log_heights.len() {
            return Err(StirError::InvalidProofShape);
        }

        let global_max_width = commitments_with_opening_points
            .iter()
            .flat_map(|(_, domain_claims)| {
                domain_claims
                    .iter()
                    .flat_map(|(_, point_claims)| point_claims.iter().map(|(_, v)| v.len()))
            })
            .max()
            .unwrap_or(0);
        let packed_alpha_powers =
            Challenge::ExtensionPacking::packed_ext_powers_capped(alpha, global_max_width)
                .collect_vec();
        let alpha_powers: Vec<Challenge> =
            Challenge::ExtensionPacking::to_ext_iter(packed_alpha_powers.iter().copied())
                .collect_vec();

        // Precompute, for each (commit, mat, point) triple: `alpha_pow_offset`, the power of
        // `alpha` this point's contribution to the reduced opening is weighted by, and
        // `y_combined`, the alpha-batched claimed value at that point. Both are pure
        // functions of public input — independent of which bucket is being verified — so
        // computing them once here (rather than inside the per-bucket, per-query,
        // per-fiber-lane loop below) turns an `O(n_q * arity0)` recomputation per point into
        // `O(1)`. `alpha_pow_offset` is tracked per log_h via a BTreeMap so the structure
        // scales with the field's two-adicity rather than a hardcoded array length.
        let mut height_num_reduced: alloc::collections::BTreeMap<usize, usize> =
            alloc::collections::BTreeMap::new();
        let point_data: Vec<Vec<Vec<(Challenge, Challenge)>>> = commitments_with_opening_points
            .iter()
            .map(|(_, domain_claims)| {
                domain_claims
                    .iter()
                    .map(|(domain, point_claims)| {
                        let log_h = log2_strict_usize(domain.size() << self.stir.log_blowup);
                        point_claims
                            .iter()
                            .map(|(_, vals)| {
                                let height_count = height_num_reduced.entry(log_h).or_insert(0);
                                let offset = alpha.exp_u64(*height_count as u64);
                                *height_count += vals.len();

                                let y_combined: Challenge = vals
                                    .iter()
                                    .zip(alpha_powers.iter())
                                    .map(|(&y, &ap)| y * ap)
                                    .sum();

                                (offset, y_combined)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // Verify each height bucket's STIR sub-proof and input binding.
        for (bucket_idx, &log_h) in bucket_log_heights.iter().enumerate() {
            let bucket_height = 1usize << log_h;
            let (stir_proof, input_openings) = &proof[bucket_idx];

            // SHAPE CHECK: input_openings has one slot per public commitment. Without this,
            // a malicious proof could omit trailing commitments — a `zip` would silently
            // drop them, their claimed values would still be observed into the transcript
            // (above), but they'd never be MMCS-opened or included in the reduced-opening
            // accumulation, so the proof would verify against a subset of the public input.
            if input_openings.len() != commitments_with_opening_points.len() {
                return Err(StirError::InvalidProofShape);
            }

            let log_stir_degree = log_h.saturating_sub(self.stir.log_blowup).max(1);
            let stir_config = StirConfig::<Val, Challenge, StirMmcs, Challenger>::new(
                log_stir_degree,
                self.stir.clone(),
            );

            // The folding factor is constant across rounds, so the same arity applies whether
            // STIR ran with intermediate rounds or only a final round.
            let log_arity0 = stir_config.log_folding_factor;
            let arity0 = 1usize << log_arity0;

            // A queried input row sits at LDE position `p = j + l * fold_height0`, whose coset
            // point is `GENERATOR * g^p` for `g = two_adic_generator(log_h)`. Walking a fiber's
            // `arity0` lanes is therefore one exponentiation per query followed by repeated
            // multiplication by the fixed step `g^fold_height0`, an `arity0`-th root of unity.
            let domain_gen = Val::two_adic_generator(log_h);
            let fiber_step = domain_gen.exp_power_of_2(log_h - log_arity0);

            // STIR's initial oracle is the reduced opening, which is a deterministic function
            // of the input commitments, the claimed values, and `alpha` — all already in the
            // transcript. Rather than have the prover commit and open it a second time, rebuild
            // its queried fibers from the input MMCS openings on demand.
            //
            // The accumulation runs across ALL commitments: when several contribute matrices to
            // the same height bucket, `ro[p]` is their sum, so a per-commitment reconstruction
            // would be wrong.
            let reconstruct_initial_fibers =
                |first_round_unique_js: &[usize]| -> Result<Vec<Vec<Challenge>>, Self::Error> {
                    let n_q = first_round_unique_js.len();
                    let mut expected_ro = vec![Challenge::zero_vec(arity0); n_q];

                    // Distinct opening points among matrices active at this bucket height.
                    // Matrices typically share opening points (e.g. one STARK's `zeta`), so
                    // this list is usually far shorter than the matrix count.
                    let bucket_points: Vec<Challenge> = commitments_with_opening_points
                        .iter()
                        .flat_map(|(_, domain_claims)| domain_claims.iter())
                        .filter(|(domain, _)| {
                            (domain.size() << self.stir.log_blowup) == bucket_height
                        })
                        .flat_map(|(_, point_claims)| point_claims.iter().map(|(point, _)| *point))
                        .fold(Vec::new(), |mut points, point| {
                            if !points.contains(&point) {
                                points.push(point);
                            }
                            points
                        });
                    let n_bp = bucket_points.len();

                    // Every `(query, fiber lane)` needs `1 / (point - fiber_point)` for each
                    // distinct point in `bucket_points`. Collect every such difference for the
                    // whole bucket and invert them all in one batch, rather than inverting each
                    // one individually once per (query, lane, matrix, point) quadruple below.
                    let mut denom_diffs = Vec::with_capacity(n_q * arity0 * n_bp);
                    for &j in first_round_unique_js {
                        let mut fiber_point = Val::GENERATOR * domain_gen.exp_u64(j as u64);
                        for _ in 0..arity0 {
                            let fp = Challenge::from(fiber_point);
                            denom_diffs.extend(bucket_points.iter().map(|&point| point - fp));
                            fiber_point *= fiber_step;
                        }
                    }
                    let inv_denoms = batch_multiplicative_inverse(&denom_diffs);

                    for (commit_idx, ((commitment, domain_claims), per_commit_opening)) in
                        commitments_with_opening_points
                            .iter()
                            .zip(input_openings.iter())
                            .enumerate()
                    {
                        let mat_lde_heights: Vec<usize> = domain_claims
                            .iter()
                            .map(|(domain, _)| domain.size() << self.stir.log_blowup)
                            .collect();

                        let has_at_bucket = mat_lde_heights.contains(&bucket_height);

                        // SHAPE CHECK: whether an opening is present at all is determined entirely by
                        // public input. Validating up-front turns a mismatch into a clean
                        // `InvalidProofShape` instead of silently skipping a binding check.
                        let Some(opening) = per_commit_opening else {
                            if has_at_bucket {
                                return Err(StirError::InvalidProofShape);
                            }
                            continue;
                        };
                        if !has_at_bucket {
                            return Err(StirError::InvalidProofShape);
                        }

                        let mat_widths: Vec<usize> = domain_claims
                            .iter()
                            .map(|(_, point_claims)| {
                                point_claims.first().map(|(_, v)| v.len()).unwrap_or(0)
                            })
                            .collect();

                        // The commitment is one root per distinct LDE height, descending; this
                        // bucket reads only its own class, so only that class's matrices appear
                        // in the opening.
                        let mut class_heights = mat_lde_heights.clone();
                        class_heights.sort_unstable();
                        class_heights.dedup();
                        class_heights.reverse();

                        // SHAPE CHECK: the class count is determined entirely by public input.
                        if commitment.len() != class_heights.len() {
                            return Err(StirError::InvalidProofShape);
                        }
                        let class_idx = class_heights
                            .iter()
                            .position(|&h| h == bucket_height)
                            .expect("`has_at_bucket` established this class exists");
                        let class_members: Vec<usize> = mat_lde_heights
                            .iter()
                            .enumerate()
                            .filter(|&(_, &h)| h == bucket_height)
                            .map(|(mat_idx, _)| mat_idx)
                            .collect();

                        // Matrices are committed fiber-grouped: `2^log_arity0` LDE rows per
                        // committed row.
                        let dimensions: Vec<p3_matrix::Dimensions> = class_members
                            .iter()
                            .map(|&mat_idx| p3_matrix::Dimensions {
                                height: bucket_height >> log_arity0,
                                width: mat_widths[mat_idx] << log_arity0,
                            })
                            .collect();

                        // Every matrix in the class shares the bucket's height, so the grouped
                        // row index is the query index itself.
                        let q_globals: Vec<usize> = first_round_unique_js
                            .iter()
                            .map(|&j| reverse_bits_len(j, log_h - log_arity0))
                            .collect();

                        // SHAPE CHECK: opened-row count is determined entirely by public input.
                        if opening.opened_values.len() != q_globals.len() {
                            return Err(StirError::InvalidProofShape);
                        }

                        self.input_mmcs
                            .verify_multi_batch(
                                &commitment[class_idx],
                                &dimensions,
                                &q_globals,
                                &opening.opened_values,
                                &opening.opening_proof,
                            )
                            .map_err(StirError::InputError)?;

                        for q_idx in 0..n_q {
                            let row_vals_by_mat = &opening.opened_values[q_idx];

                            #[allow(clippy::needless_range_loop)]
                            for l in 0..arity0 {
                                // Fiber column `l` sits at slot `reverse_bits_len(l, log_arity0)`
                                // of the grouped row.
                                let slot = reverse_bits_len(l, log_arity0);

                                for (local_idx, &mat_idx) in class_members.iter().enumerate() {
                                    let (_, point_claims) = &domain_claims[mat_idx];

                                    // `verify_multi_batch` already pinned each grouped row to
                                    // `dimensions[local_idx].width`, so this slice is in bounds.
                                    let width = mat_widths[mat_idx];
                                    let row_vals =
                                        &row_vals_by_mat[local_idx][slot * width..][..width];
                                    let p_x: Challenge = row_vals
                                        .iter()
                                        .zip(alpha_powers.iter())
                                        .map(|(&v, &ap)| ap * v)
                                        .sum();

                                    for (point_idx, (point, _)) in point_claims.iter().enumerate() {
                                        let (alpha_pow_offset, y_combined) =
                                            point_data[commit_idx][mat_idx][point_idx];

                                        let bp_idx = bucket_points
                                            .iter()
                                            .position(|p| p == point)
                                            .expect("point is in bucket_points by construction");
                                        let inv_denom =
                                            inv_denoms[(q_idx * arity0 + l) * n_bp + bp_idx];

                                        expected_ro[q_idx][l] +=
                                            alpha_pow_offset * (p_x - y_combined) * inv_denom;
                                    }
                                }
                            }
                        }
                    }

                    Ok(expected_ro)
                };

            // Any transcript-touching step stays inside `verify_stir_with_external_initial`;
            // the closure above only reads public data and the input openings.
            verify_stir_with_external_initial(
                &stir_config,
                stir_proof,
                challenger,
                reconstruct_initial_fibers,
            )?;
        }

        Ok(())
    }
}

type MatricesAndPoints<'a, F, EF> = (Vec<RowMajorMatrixView<'a, F>>, &'a Vec<Vec<EF>>);

/// Compute `1/(z - x)` for all coset elements `x`, batched over all unique points `z`.
fn compute_inverse_denominators<'a, F: TwoAdicField, EF: ExtensionField<F>>(
    mats_and_points: &'a [MatricesAndPoints<'a, F, EF>],
    coset: &[F],
) -> LinearMap<EF, Vec<EF>> {
    // Find the maximum height for each unique opening point.
    let mut point_max_height: LinearMap<EF, usize> = LinearMap::new();
    for (mats, points) in mats_and_points {
        for (mat, points_for_mat) in mats.iter().zip(points.iter()) {
            for &point in points_for_mat {
                if let Some(existing) = point_max_height.get_mut(&point) {
                    if mat.height() > *existing {
                        *existing = mat.height();
                    }
                } else {
                    point_max_height.insert(point, mat.height());
                }
            }
        }
    }

    point_max_height
        .into_iter()
        .map(|(z, max_h)| {
            let max_h = max_h.max(1);
            let diffs: Vec<EF> = coset[..max_h].iter().map(|&x| z - EF::from(x)).collect();
            let inv_diffs = batch_multiplicative_inverse(&diffs);
            (z, inv_diffs)
        })
        .collect()
}
