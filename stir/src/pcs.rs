//! `TwoAdicStirPcs`: implementing the [`Pcs`] trait using STIR.
//!
//! **Commit**: every matrix passed to one `commit()` call is extended onto one shared LDE
//! domain, sized to the *tallest* matrix (§7, Construction 7.2's same-domain requirement) —
//! shorter matrices are extended further, at an effectively higher per-matrix blowup. All
//! matrices land in one Merkle tree per commitment, rather than one tree per distinct height.
//!
//! **Open**: alpha-batch quotient polynomials `(f_i(z) - f_i(x)) / (z - x)` into one
//! reduced-opening polynomial per *native* (pre-shared-domain) matrix height — the same
//! per-height batching as before, just with every reduced-opening now living on the shared
//! domain rather than its own native-sized one. Commitments sharing the same shared-domain
//! size are then run through one STIR instance together (as many domain sizes as distinct
//! shared domains, "buckets" below); within a bucket, if more than one native-height class is
//! present, they are merged into a single codeword via batch degree correction
//! ([`combine_on_coset`](crate::utils::combine_on_coset), §4.5's `Combine`) before STIR runs,
//! at the tallest class's degree and full proximity radius (no per-class query-count floor).
//! The prover returns the deduplicated first-round STIR query indices alongside the IOP
//! proof; at those positions the prover also opens the input LDE matrices (via `InputMmcs`)
//! so the verifier can confirm the reduced-opening polynomials are correctly derived from the
//! committed inputs.
//!
//! **Verify**: replay the same alpha-batching and `Combine` from the opening values, then for
//! each bucket call [`verify_stir_with_external_initial`](crate::verifier::verify_stir_with_external_initial).
//! STIR's initial oracle *is* the (possibly combined) reduced opening, which the transcript
//! already pins through the input commitments, the claimed values, `alpha`, and (when a
//! bucket combines more than one class) the combination challenge, so it is never committed a
//! second time: whenever STIR needs its queried fibers, the verifier rebuilds them from the
//! input MMCS openings at exactly the positions STIR sampled. No hand-mirrored transcript
//! replay is needed.

use alloc::borrow::Cow;
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
    BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, TwoAdicField,
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
use crate::prover::prove_stir_multi_from_external_codewords;
use crate::utils::{combine_on_coset, eval_degree_correction};
use crate::verifier::{StirError, verify_stir_multi_with_external_initial};

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
    /// public data. Each such row concatenates the `2^log_starting_folding_factor` LDE rows
    /// of one fiber, ordered by the bit-reversal of the fiber column index.
    pub opened_values: Vec<Vec<Vec<Val>>>,
    /// Compact multi-opening proof authenticating every row at once.
    pub opening_proof: InputMmcs::MultiProof,
}

/// Prover data for [`TwoAdicStirPcs`].
///
/// Every matrix passed to one `commit()` call is extended onto one shared LDE domain (sized
/// to the tallest matrix) and committed in fiber-grouped form in a single Merkle tree — each
/// leaf holds `2^log_starting_folding_factor` consecutive bit-reversed LDE rows, exactly the
/// rows one first-round STIR query reads. `log_native_heights[i]` is matrix `i`'s own
/// (pre-shared-domain) log2 height, in the order the caller committed them; this is what
/// distinguishes matrices for alpha-batching and `Combine` grouping once they all sit on the
/// same physical domain.
pub struct StirProverData<Val: Send + Sync + Clone, InputMmcs: Mmcs<Val>> {
    data: InputMmcs::ProverData<RowMajorMatrix<Val>>,
    /// Column count of each matrix, in the order the caller committed them.
    widths: Vec<usize>,
    /// Native (pre-shared-domain) log2 height of each matrix, in the order committed.
    log_native_heights: Vec<usize>,
    /// Log2 of the shared LDE domain every matrix in this commitment was extended onto.
    log_shared_lde_height: usize,
}

/// Commit fiber-grouped LDEs (all sharing one height) in a single tree, recording each
/// matrix's own native height and column count for later alpha-batching/`Combine` grouping.
fn commit_shared_domain<Val, InputMmcs>(
    input_mmcs: &InputMmcs,
    grouped: Vec<RowMajorMatrix<Val>>,
    log_native_heights: Vec<usize>,
    widths: Vec<usize>,
    log_shared_lde_height: usize,
) -> (InputMmcs::Commitment, StirProverData<Val, InputMmcs>)
where
    Val: Send + Sync + Clone,
    InputMmcs: Mmcs<Val>,
{
    let (commitment, data) = input_mmcs.commit(grouped);
    (
        commitment,
        StirProverData {
            data,
            widths,
            log_native_heights,
            log_shared_lde_height,
        },
    )
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
    let matrices = input_mmcs.get_matrices(&prover_data.data);
    matrices
        .into_iter()
        .zip(&prover_data.widths)
        .map(|(grouped, &width)| RowMajorMatrixView::new(grouped.values.as_slice(), width))
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

/// One bucket's `Combine` state for the verifier: the sampled combination challenge and each
/// present native height's `(r_i, gap_i)` coefficients (`None` when the bucket has only one
/// class, so no `Combine` step ran).
type BucketCombine<Challenge> = Option<(
    Challenge,
    alloc::collections::BTreeMap<usize, (Challenge, usize)>,
)>;

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
    /// One Merkle root per commitment: every matrix passed to one `commit()` call shares a
    /// single tree on the shared LDE domain (§7's same-domain requirement).
    type Commitment = InputMmcs::Commitment;
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
        let min_height = 1usize << self.stir.log_starting_folding_factor;
        let inputs: Vec<(Self::Domain, RowMajorMatrix<Val>)> = evaluations.into_iter().collect();
        assert!(
            !inputs.is_empty(),
            "STIR PCS: commit requires at least one matrix"
        );
        for (domain, evals) in &inputs {
            assert_eq!(domain.size(), evals.height());
            assert!(
                evals.height() >= min_height,
                "STIR PCS: matrix height {} is below the minimum of 2^{} (= {}) required by \
                 log_starting_folding_factor = {}. Pad the matrix to at least this height \
                 before committing, or lower log_starting_folding_factor.",
                evals.height(),
                self.stir.log_starting_folding_factor,
                min_height,
                self.stir.log_starting_folding_factor,
            );
        }
        let log_max_native_height = inputs
            .iter()
            .map(|(domain, _)| log2_strict_usize(domain.size()))
            .max()
            .expect("checked non-empty above");
        let log_shared_lde_height = log_max_native_height + self.stir.log_blowup;

        let mut widths = Vec::with_capacity(inputs.len());
        let mut log_native_heights = Vec::with_capacity(inputs.len());
        let grouped: Vec<_> = inputs
            .into_iter()
            .map(|(domain, evals)| {
                let log_native_height = log2_strict_usize(domain.size());
                let extra_bits = log_shared_lde_height - log_native_height;
                let shift = Val::GENERATOR / domain.shift();
                let lde = self
                    .dft
                    .coset_lde_batch(evals, extra_bits, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix();
                widths.push(lde.width());
                log_native_heights.push(log_native_height);
                group_fiber_rows(lde, self.stir.log_starting_folding_factor)
            })
            .collect();
        commit_shared_domain(
            &self.input_mmcs,
            grouped,
            log_native_heights,
            widths,
            log_shared_lde_height,
        )
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let grouped = self.input_mmcs.get_matrices(&prover_data.data)[idx];
        let lde = RowMajorMatrixView::new(grouped.values.as_slice(), prover_data.widths[idx]);
        if domain.shift() == Val::GENERATOR && lde.height() >= domain.size() {
            let width = lde.width();
            let values: &'a [Val] = lde.values;
            return RowMajorMatrixView::new(&values[..domain.size() * width], width)
                .as_cow()
                .bit_reverse_rows();
        }
        let poly_height = 1usize << prover_data.log_native_heights[idx];
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
        let min_height = 1usize << self.stir.log_starting_folding_factor;
        evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert!(
                    evals.height() >= min_height,
                    "STIR PCS quotient: matrix height {} is below 2^{} required by \
                     log_starting_folding_factor = {}.",
                    evals.height(),
                    self.stir.log_starting_folding_factor,
                    self.stir.log_starting_folding_factor,
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
        let min_lde_height =
            1usize << (self.stir.log_starting_folding_factor + self.stir.log_blowup);
        assert!(
            !ldes.is_empty(),
            "STIR PCS: commit_ldes requires at least one matrix"
        );
        for lde in &ldes {
            assert!(
                lde.height() >= min_lde_height,
                "STIR PCS: pre-computed LDE height {} is below 2^{} (= {}) required by \
                 log_starting_folding_factor + log_blowup = {} + {}.",
                lde.height(),
                self.stir.log_starting_folding_factor + self.stir.log_blowup,
                min_lde_height,
                self.stir.log_starting_folding_factor,
                self.stir.log_blowup,
            );
        }

        // `ldes[i]` is already bit-reversed at `2^(native_i + log_blowup)`, GENERATOR-shifted
        // (matching `get_quotient_ldes`'s output convention). Shorter ones are re-extended
        // onto the shared domain sized to the tallest.
        let log_native_heights: Vec<usize> = ldes
            .iter()
            .map(|lde| log2_strict_usize(lde.height()) - self.stir.log_blowup)
            .collect();
        let log_max_native_height = log_native_heights
            .iter()
            .copied()
            .max()
            .expect("checked non-empty above");
        let log_shared_lde_height = log_max_native_height + self.stir.log_blowup;

        let mut widths = Vec::with_capacity(ldes.len());
        let grouped: Vec<_> = ldes
            .into_iter()
            .zip(&log_native_heights)
            .map(|(lde, &log_native_height)| {
                widths.push(lde.width());
                let extended = if lde.height() == 1usize << log_shared_lde_height {
                    lde
                } else {
                    let natural_lde = lde.bit_reverse_rows().to_row_major_matrix();
                    let mut coeffs = self.dft.coset_idft_batch(natural_lde, Val::GENERATOR);
                    let width = coeffs.width();
                    coeffs
                        .values
                        .truncate((1usize << log_native_height) * width);
                    self.dft
                        .coset_lde_batch(
                            coeffs,
                            log_shared_lde_height - log_native_height,
                            Val::GENERATOR,
                        )
                        .bit_reverse_rows()
                        .to_row_major_matrix()
                };
                group_fiber_rows(extended, self.stir.log_starting_folding_factor)
            })
            .collect();
        commit_shared_domain(
            &self.input_mmcs,
            grouped,
            log_native_heights,
            widths,
            log_shared_lde_height,
        )
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
            .zip(&commitment_data_with_opening_points)
            .map(|((mats, points), (data, _))| {
                izip!(mats.iter(), points.iter(), data.log_native_heights.iter())
                    .map(|(mat, points_for_mat, &log_native_h)| {
                        let h = 1usize << log_native_h;
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

        // Step 2: Alpha-batch into one reduced-opening vector per (shared LDE domain, native
        // height) class. Every matrix in a class lives on the same physical domain (its
        // commitment's shared domain) and shares the same claimed degree, both required to
        // alpha-batch them together and, later, for `Combine` to merge classes soundly.
        let alpha: Challenge = challenger.sample_algebra_element();
        let packed_alpha_powers =
            Challenge::ExtensionPacking::packed_ext_powers_capped(alpha, global_max_width)
                .collect_vec();
        let alpha_powers: Vec<Challenge> =
            Challenge::ExtensionPacking::to_ext_iter(packed_alpha_powers.iter().copied())
                .collect_vec();

        // Keyed by `(log_shared_lde_height, log_native_height)`. The outer key selects which
        // STIR instance a class feeds; the inner key is `Combine`'s per-class degree.
        let mut reduced_openings: alloc::collections::BTreeMap<(usize, usize), Vec<Challenge>> =
            alloc::collections::BTreeMap::new();
        let mut num_reduced: alloc::collections::BTreeMap<(usize, usize), usize> =
            alloc::collections::BTreeMap::new();

        for (((mats, points), opened_vals), (data, _)) in mats_and_points
            .iter()
            .zip(&all_opened_values)
            .zip(&commitment_data_with_opening_points)
        {
            for (((mat, points_for_mat), opened_for_mat), &log_native_h) in
                izip!(mats.iter(), points.iter())
                    .zip(opened_vals.iter())
                    .zip(data.log_native_heights.iter())
            {
                let key = (data.log_shared_lde_height, log_native_h);
                let ro = reduced_openings
                    .entry(key)
                    .or_insert_with(|| vec![Challenge::ZERO; mat.height()]);

                // Precompute alpha-batched row values for this matrix (reused per point).
                let p_x_vec: Vec<Challenge> = mat
                    .rowwise_packed_dot_product::<Challenge>(&packed_alpha_powers)
                    .collect();

                for (point, ys) in points_for_mat.iter().zip(opened_for_mat.iter()) {
                    let height_count = num_reduced.entry(key).or_insert(0);
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

        // Step 3: within each distinct shared-LDE-height bucket (one physical domain, hence
        // one STIR instance), merge its native-height classes via `Combine` (§4.5) when more
        // than one is present, then run STIR on every bucket in lockstep, sharing every
        // grind across buckets, then bind the input MMCS at each bucket's query positions.
        let bucket_log_heights: Vec<usize> = {
            let mut heights: Vec<usize> = reduced_openings.keys().map(|&(h, _)| h).collect();
            heights.sort_unstable();
            heights.dedup();
            heights.reverse();
            heights
        };

        // Native-height classes present in each bucket, descending, computed once and
        // shared by the `StirConfig` construction below (which needs the class count and
        // `ell` to size round 0's `eta` for `Combine`) and `combined_bucket_codeword`
        // (which needs the same classes to actually run `Combine`).
        let bucket_native_heights: Vec<Vec<usize>> = bucket_log_heights
            .iter()
            .map(|&log_h| {
                let mut heights: Vec<usize> = reduced_openings
                    .keys()
                    .filter(|&&(h, _)| h == log_h)
                    .map(|&(_, log_d)| log_d)
                    .collect();
                heights.sort_unstable();
                heights.dedup();
                heights.reverse();
                heights
            })
            .collect();

        let stir_configs: Vec<StirConfig<Val, Challenge, StirMmcs, Challenger>> =
            bucket_log_heights
                .iter()
                .zip(&bucket_native_heights)
                .map(|(&log_h, native_heights)| {
                    let log_stir_degree = log_h.saturating_sub(self.stir.log_blowup).max(1);
                    if native_heights.len() >= 2 {
                        let log_d_star = native_heights[0];
                        let ell: u64 = native_heights.len() as u64 * ((1u64 << log_d_star) + 1)
                            - native_heights.iter().map(|&d| 1u64 << d).sum::<u64>();
                        StirConfig::<Val, Challenge, StirMmcs, Challenger>::new_with_combine(
                            log_stir_degree,
                            self.stir.clone(),
                            native_heights.len(),
                            ell,
                        )
                    } else {
                        StirConfig::<Val, Challenge, StirMmcs, Challenger>::new(
                            log_stir_degree,
                            self.stir.clone(),
                        )
                    }
                })
                .collect();
        let stir_config_refs: Vec<&StirConfig<Val, Challenge, StirMmcs, Challenger>> =
            stir_configs.iter().collect();

        let initial_codewords: Vec<Vec<Challenge>> = bucket_log_heights
            .iter()
            .map(|&log_shared_h| {
                combined_bucket_codeword::<Val, Challenge, Challenger>(
                    &reduced_openings,
                    log_shared_h,
                    challenger,
                )
            })
            .collect();

        let bucket_results = prove_stir_multi_from_external_codewords(
            &stir_config_refs,
            initial_codewords,
            &self.dft,
            challenger,
        );

        let bucket_proofs = bucket_log_heights
            .iter()
            .zip(&stir_configs)
            .zip(bucket_results)
            .map(
                |((&log_h, stir_config), (stir_proof, first_round_query_indices))| {
                    let log_arity0 = stir_config.log_starting_folding_factor;

                    let input_openings: Vec<Option<InputOpenings<Val, InputMmcs>>> =
                        commitment_data_with_opening_points
                            .iter()
                            .map(|(data, _)| {
                                // A commitment's whole shared domain either feeds this
                                // bucket or none of it does -- there is no partial case
                                // now that every matrix in a commitment shares one domain.
                                if data.log_shared_lde_height != log_h {
                                    return None;
                                }

                                let q_globals: Vec<usize> = first_round_query_indices
                                    .iter()
                                    .map(|&j| reverse_bits_len(j, log_h - log_arity0))
                                    .collect();

                                let (opened_values, opening_proof) =
                                    self.input_mmcs.open_multi_batch(&q_globals, &data.data);
                                Some(InputOpenings {
                                    opened_values,
                                    opening_proof,
                                })
                            })
                            .collect();

                    (stir_proof, input_openings)
                },
            )
            .collect();

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

        // Every matrix in one commitment shares its shared LDE domain: the tallest matrix's
        // native height plus log_blowup. This is public (derived from the claimed domains),
        // matching how the prover derived it at commit time.
        let commitment_shared_heights: Vec<usize> = commitments_with_opening_points
            .iter()
            .map(|(_, domain_claims)| {
                let log_max_native = domain_claims
                    .iter()
                    .map(|(domain, _)| log2_strict_usize(domain.size()))
                    .max()
                    .unwrap_or(0);
                log_max_native + self.stir.log_blowup
            })
            .collect();

        // Distinct outer buckets (shared LDE heights), descending. Must match the prover's
        // bucket iteration order.
        let bucket_log_heights: Vec<usize> = {
            let mut heights: Vec<usize> = commitment_shared_heights.clone();
            heights.sort_unstable();
            heights.dedup();
            heights.reverse();
            heights
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
        // `O(1)`. Keyed like the prover's `reduced_openings`, by `(log_shared_lde_height,
        // log_native_height)`, so the structure scales with the field's two-adicity rather
        // than a hardcoded array length.
        let mut class_num_reduced: alloc::collections::BTreeMap<(usize, usize), usize> =
            alloc::collections::BTreeMap::new();
        let point_data: Vec<Vec<Vec<(Challenge, Challenge)>>> = commitments_with_opening_points
            .iter()
            .zip(&commitment_shared_heights)
            .map(|((_, domain_claims), &log_shared_h)| {
                domain_claims
                    .iter()
                    .map(|(domain, point_claims)| {
                        let key = (log_shared_h, log2_strict_usize(domain.size()));
                        point_claims
                            .iter()
                            .map(|(_, vals)| {
                                let count = class_num_reduced.entry(key).or_insert(0);
                                let offset = alpha.exp_u64(*count as u64);
                                *count += vals.len();

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

        // SHAPE CHECK: every bucket's input_openings has one slot per public commitment.
        // Without this, a malicious proof could omit trailing commitments — a `zip` would
        // silently drop them, their claimed values would still be observed into the
        // transcript (above), but they'd never be MMCS-opened or included in the
        // reduced-opening accumulation, so the proof would verify against a subset of the
        // public input.
        for (_, input_openings) in proof {
            if input_openings.len() != commitments_with_opening_points.len() {
                return Err(StirError::InvalidProofShape);
            }
        }

        // Native-height classes present in each bucket, descending, computed once and
        // shared by the `StirConfig` construction below (which needs the class count and
        // `ell` to size round 0's `eta` for `Combine`) and the `Combine`-challenge sampling
        // that follows (which needs the same classes to derive its coefficients).
        let bucket_native_heights: Vec<Vec<usize>> = bucket_log_heights
            .iter()
            .map(|&log_shared_h| {
                let mut native_heights: Vec<usize> = commitments_with_opening_points
                    .iter()
                    .zip(&commitment_shared_heights)
                    .filter(|&(_, h)| *h == log_shared_h)
                    .flat_map(|((_, domain_claims), _)| {
                        domain_claims
                            .iter()
                            .map(|(domain, _)| log2_strict_usize(domain.size()))
                    })
                    .collect();
                native_heights.sort_unstable();
                native_heights.dedup();
                native_heights.reverse();
                native_heights
            })
            .collect();

        let stir_configs: Vec<StirConfig<Val, Challenge, StirMmcs, Challenger>> =
            bucket_log_heights
                .iter()
                .zip(&bucket_native_heights)
                .map(|(&log_h, native_heights)| {
                    let log_stir_degree = log_h.saturating_sub(self.stir.log_blowup).max(1);
                    if native_heights.len() >= 2 {
                        let log_d_star = native_heights[0];
                        let ell: u64 = native_heights.len() as u64 * ((1u64 << log_d_star) + 1)
                            - native_heights.iter().map(|&d| 1u64 << d).sum::<u64>();
                        StirConfig::<Val, Challenge, StirMmcs, Challenger>::new_with_combine(
                            log_stir_degree,
                            self.stir.clone(),
                            native_heights.len(),
                            ell,
                        )
                    } else {
                        StirConfig::<Val, Challenge, StirMmcs, Challenger>::new(
                            log_stir_degree,
                            self.stir.clone(),
                        )
                    }
                })
                .collect();
        let stir_config_refs: Vec<&StirConfig<Val, Challenge, StirMmcs, Challenger>> =
            stir_configs.iter().collect();
        let stir_proofs: Vec<&StirProof<Challenge, StirMmcs, Val>> =
            proof.iter().map(|(p, _)| p).collect();

        // For every bucket with more than one native-height class present, sample the
        // `Combine` challenge and derive its per-class coefficients up front, at the same
        // transcript position the prover's `combined_bucket_codeword` used (before any
        // STIR-internal transcript operations) — mirroring how `alpha` itself is sampled
        // once, up front, rather than lazily inside a bucket's closure.
        let bucket_combine: Vec<BucketCombine<Challenge>> = bucket_log_heights
            .iter()
            .zip(&bucket_native_heights)
            .map(|(_, native_heights)| {
                if native_heights.len() <= 1 {
                    return None;
                }

                let log_d_star = native_heights[0];
                let r_comb: Challenge = challenger.sample_algebra_element();
                let (_, coeffs) =
                    combine_coefficients(r_comb, log_d_star, native_heights.iter().copied());
                Some((r_comb, native_heights.iter().copied().zip(coeffs).collect()))
            })
            .collect();

        // Captured by every bucket's closure below; taking references up front lets `move`
        // give each closure its own copy of the reference rather than the whole value.
        let commitments_with_opening_points = &commitments_with_opening_points;
        let commitment_shared_heights = &commitment_shared_heights;
        let point_data = &point_data;
        let alpha_powers = &alpha_powers;
        let bucket_combine = &bucket_combine;

        // STIR's initial oracle is the (possibly `Combine`d) reduced opening, which is a
        // deterministic function of the input commitments, the claimed values, `alpha`, and
        // (when a bucket merges more than one class) the combination challenge — all already
        // in the transcript. Rather than have the prover commit and open it a second time,
        // rebuild its queried fibers from the input MMCS openings on demand.
        let initial_fibers: Vec<_> = bucket_log_heights
            .iter()
            .zip(&stir_configs)
            .zip(proof.iter().map(|(_, input_openings)| input_openings))
            .zip(bucket_combine)
            .map(|(((&log_h, stir_config), input_openings), combine_info)| {
                let bucket_height = 1usize << log_h;
                let log_arity0 = stir_config.log_starting_folding_factor;
                let arity0 = 1usize << log_arity0;

                // A queried input row sits at LDE position `p = j + l * fold_height0`, whose
                // coset point is `GENERATOR * g^p` for `g = two_adic_generator(log_h)`.
                // Walking a fiber's `arity0` lanes is therefore one exponentiation per query
                // followed by repeated multiplication by the fixed step `g^fold_height0`, an
                // `arity0`-th root of unity.
                let domain_gen = Val::two_adic_generator(log_h);
                let fiber_step = domain_gen.exp_power_of_2(log_h - log_arity0);

                move |first_round_unique_js: &[usize]| -> Result<Vec<Vec<Challenge>>, Self::Error> {
                    let n_q = first_round_unique_js.len();

                    // One accumulator per native-height class present in this bucket; merged
                    // into the final expected codeword fibers after the accumulation loop.
                    let mut expected_ro_by_class: alloc::collections::BTreeMap<
                        usize,
                        Vec<Vec<Challenge>>,
                    > = alloc::collections::BTreeMap::new();

                    // Distinct opening points among matrices active at this bucket. Matrices
                    // typically share opening points (e.g. one STARK's `zeta`), so this list
                    // is usually far shorter than the matrix count.
                    let bucket_points: Vec<Challenge> = commitments_with_opening_points
                        .iter()
                        .zip(commitment_shared_heights.iter())
                        .filter(|&(_, h)| *h == log_h)
                        .flat_map(|((_, domain_claims), _)| domain_claims.iter())
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

                    // A commitment's whole shared domain either feeds this bucket (all its
                    // matrices) or none of it does — there is no partial case now that every
                    // matrix in a commitment shares one domain.
                    for (commit_idx, ((commitment, domain_claims), per_commit_opening)) in
                        commitments_with_opening_points
                            .iter()
                            .zip(input_openings.iter())
                            .enumerate()
                    {
                        let has_at_bucket = commitment_shared_heights[commit_idx] == log_h;

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

                        // Every matrix in the commitment shares this bucket's domain, so all
                        // of them are opened. Matrices are committed fiber-grouped:
                        // `2^log_arity0` LDE rows per committed row.
                        let dimensions: Vec<p3_matrix::Dimensions> = mat_widths
                            .iter()
                            .map(|&width| p3_matrix::Dimensions {
                                height: bucket_height >> log_arity0,
                                width: width << log_arity0,
                            })
                            .collect();

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
                                commitment,
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

                                for (mat_idx, (domain, point_claims)) in
                                    domain_claims.iter().enumerate()
                                {
                                    let log_native_h = log2_strict_usize(domain.size());
                                    let width = mat_widths[mat_idx];
                                    let row_vals =
                                        &row_vals_by_mat[mat_idx][slot * width..][..width];
                                    let p_x: Challenge = row_vals
                                        .iter()
                                        .zip(alpha_powers.iter())
                                        .map(|(&v, &ap)| ap * v)
                                        .sum();

                                    let ro_class = expected_ro_by_class
                                        .entry(log_native_h)
                                        .or_insert_with(|| vec![Challenge::zero_vec(arity0); n_q]);

                                    for (point_idx, (point, _)) in point_claims.iter().enumerate() {
                                        let (alpha_pow_offset, y_combined) =
                                            point_data[commit_idx][mat_idx][point_idx];

                                        let bp_idx = bucket_points
                                            .iter()
                                            .position(|p| p == point)
                                            .expect("point is in bucket_points by construction");
                                        let inv_denom =
                                            inv_denoms[(q_idx * arity0 + l) * n_bp + bp_idx];

                                        ro_class[q_idx][l] +=
                                            alpha_pow_offset * (p_x - y_combined) * inv_denom;
                                    }
                                }
                            }
                        }
                    }

                    // Merge the per-class accumulators into the bucket's expected codeword
                    // fibers, mirroring the prover's `combine_on_coset`/`eval_degree_correction`
                    // pointwise, only at the queried fiber lanes.
                    match combine_info {
                        None => {
                            // SHAPE CHECK: exactly one class expected when no Combine ran.
                            if expected_ro_by_class.len() != 1 {
                                return Err(StirError::InvalidProofShape);
                            }
                            Ok(expected_ro_by_class.into_values().next().unwrap())
                        }
                        Some((r_comb, coeffs_by_height)) => {
                            // SHAPE CHECK: every expected class must be present.
                            if expected_ro_by_class.len() != coeffs_by_height.len() {
                                return Err(StirError::InvalidProofShape);
                            }
                            let mut combined = vec![Challenge::zero_vec(arity0); n_q];
                            for (log_native_h, ro_class) in &expected_ro_by_class {
                                let &(r_i, gap) = coeffs_by_height
                                    .get(log_native_h)
                                    .ok_or(StirError::InvalidProofShape)?;
                                for (q_idx, &j) in first_round_unique_js.iter().enumerate() {
                                    let mut fiber_point =
                                        Val::GENERATOR * domain_gen.exp_u64(j as u64);
                                    for l in 0..arity0 {
                                        let x = Challenge::from(fiber_point);
                                        combined[q_idx][l] += eval_degree_correction(
                                            r_i * ro_class[q_idx][l],
                                            x,
                                            *r_comb,
                                            gap,
                                        );
                                        fiber_point *= fiber_step;
                                    }
                                }
                            }
                            Ok(combined)
                        }
                    }
                }
            })
            .collect();

        // Any transcript-touching step stays inside `verify_stir_multi_with_external_initial`;
        // every closure above only reads public data and its own bucket's input openings.
        verify_stir_multi_with_external_initial(
            &stir_config_refs,
            &stir_proofs,
            challenger,
            initial_fibers,
        )?;

        Ok(())
    }
}

/// Definition 4.11's per-class `Combine` coefficients, for classes already sorted in
/// descending native-degree order (tallest first, so the target degree `d* := 2^log_d_star`
/// is the first class's own degree, giving it `gap = 0` and, per `r_1 := 1`, a trivial
/// coefficient).
///
/// Returns `(d*, [(r_i, gap_i)])` where `gap_i = d* - dᵢ` (a degree count, not log) is what
/// `eval_degree_correction`/`combine_on_coset` expect directly.
fn combine_coefficients<EF: Field>(
    r_comb: EF,
    log_d_star: usize,
    sorted_log_ds: impl Iterator<Item = usize>,
) -> (u64, Vec<(EF, usize)>) {
    let d_star = 1u64 << log_d_star;
    let mut running_exp = 0u64;
    let coeffs = sorted_log_ds
        .map(|log_d| {
            let r_i = r_comb.exp_u64(running_exp);
            let gap = (d_star - (1u64 << log_d)) as usize;
            running_exp += 1 + gap as u64;
            (r_i, gap)
        })
        .collect();
    (d_star, coeffs)
}

/// Merge one shared-LDE-height bucket's native-height classes into a single codeword on
/// their shared domain, via `Combine` (§4.5) when more than one class is present.
///
/// `reduced_openings` is keyed by `(log_shared_lde_height, log_native_height)`; this pulls
/// out every class at `log_shared_h`, descending by native height, and returns the natural
/// (not yet bit-reversed) combined codeword STIR should run on.
fn combined_bucket_codeword<Val, Challenge, Challenger>(
    reduced_openings: &alloc::collections::BTreeMap<(usize, usize), Vec<Challenge>>,
    log_shared_h: usize,
    challenger: &mut Challenger,
) -> Vec<Challenge>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val> + TwoAdicField,
    Challenger: FieldChallenger<Val>,
{
    let mut classes: Vec<(usize, &Vec<Challenge>)> = reduced_openings
        .iter()
        .filter(|((h, _), _)| *h == log_shared_h)
        .map(|(&(_, log_d), ro)| (log_d, ro))
        .collect();
    classes.sort_unstable_by(|(a, _), (b, _)| b.cmp(a));

    if classes.len() == 1 {
        let mut natural = classes[0].1.clone();
        reverse_slice_index_bits(&mut natural);
        return natural;
    }

    let log_d_star = classes[0].0;
    let r_comb: Challenge = challenger.sample_algebra_element();
    let (_, coeffs) =
        combine_coefficients(r_comb, log_d_star, classes.iter().map(|&(log_d, _)| log_d));

    // `combine_on_coset` indexes its inputs (and produces its output) in natural order, but
    // `reduced_openings` is bit-reversed (built from the bit-reversed LDE matrices), so each
    // class's codeword is un-reversed before combining; the combined result is then already
    // in the natural order STIR expects, with no further reversal needed.
    let natural_ros: Vec<Vec<Challenge>> = classes
        .iter()
        .map(|&(_, ro)| {
            let mut natural = ro.clone();
            reverse_slice_index_bits(&mut natural);
            natural
        })
        .collect();
    let groups: Vec<(Challenge, usize, &[Challenge])> = coeffs
        .into_iter()
        .zip(&natural_ros)
        .map(|((r_i, gap), ro)| (r_i, gap, ro.as_slice()))
        .collect();
    combine_on_coset(&groups, r_comb, Val::GENERATOR, log_shared_h)
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
