//! `TwoAdicStirPcs`: implementing the [`Pcs`] trait using STIR.
//!
//! **Commit**: the matrices passed to one `commit()` call are partitioned into *shared-domain
//! groups* of bounded native-height spread. Each group is extended onto one shared LDE domain
//! sized to that group's tallest matrix (§7, Construction 7.2's same-domain requirement) and
//! committed in its own Merkle tree, so a commitment carries one root per group. A group
//! holding a single native height needs no merging at all; a group holding several merges
//! them below.
//!
//! The two ends of the spread cap ([`TwoAdicStirPcs::with_max_log_height_spread`],
//! [`DEFAULT_MAX_LOG_HEIGHT_SPREAD`]) are the two layouts this generalizes: a cap of `0` puts
//! every distinct native height on its own domain, so `Combine` never runs and each height
//! gets its own STIR instance, while a cap at or above the committed spread puts everything on
//! one domain and merges it all. Groups also shrink on their own when `Combine`'s soundness
//! cost does not fit the challenge field, so an infeasible parameter set degrades into more
//! STIR instances rather than failing.
//!
//! **Open**: alpha-batch quotient polynomials `(f_i(z) - f_i(x)) / (z - x)` into one
//! reduced-opening polynomial per *native* matrix height, each living on its group's shared
//! domain. Groups sharing a domain size — across commitments as well as within one — are run
//! through a single STIR instance ("buckets" below, one per distinct shared LDE height);
//! within a bucket, if more than one native-height class is present, they are merged into a
//! single codeword via batch degree correction ([`crate::utils::combine_on_coset`], §4.5's
//! `Combine`) before STIR runs, at the tallest class's degree and full proximity radius (no
//! per-class query-count floor). The prover returns the deduplicated first-round STIR query
//! indices alongside the IOP proof; at those positions the prover also opens the input LDE
//! matrices (via `InputMmcs`) so the verifier can confirm the reduced-opening polynomials are
//! correctly derived from the committed inputs.
//!
//! **Verify**: reproduce the grouping from the claimed domain sizes — it is a pure function of
//! those and the PCS parameters, so no part of it travels in the proof — then replay the same
//! alpha-batching and `Combine` from the opening values, and for each bucket call
//! [`verify_stir_with_external_initial`](crate::verifier::verify_stir_with_external_initial).
//! STIR's initial oracle *is* the (possibly combined) reduced opening, which the transcript
//! already pins through the input commitments, the claimed values, `alpha`, and (when a
//! bucket combines more than one class) the combination challenge, so it is never committed a
//! second time: whenever STIR needs its queried fibers, the verifier rebuilds them from the
//! input MMCS openings at exactly the positions STIR sampled. No hand-mirrored transcript
//! replay is needed.
//!
//! **Cost profile**: merging classes onto one domain makes opening and verification cheaper —
//! one STIR instance instead of one per height class — and committing more expensive, and the
//! spread cap is what bounds the second. A matrix at native height `2^h` in a group whose
//! tallest is `2^H` pays a `2^(H - h + log_blowup)` blowup instead of `2^log_blowup`, in both
//! its DFT and its share of that group's Merkle tree (the tree is `2^(H + log_blowup)` rows
//! deep and carries every group member's full width in each leaf, so hashing goes from
//! `Σᵢ 2^(hᵢ + b)·widthᵢ` to `2^(H + b)·Σᵢ widthᵢ`). Capping the spread caps `H - h`, so a
//! short matrix committed alongside a much taller one lands in its own group and pays its own
//! blowup rather than the tallest one's.

use alloc::borrow::Cow;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;

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
use spin::RwLock;
use tracing::instrument;

use crate::config::{StirConfig, StirConfigError, StirParameters};
use crate::proof::StirProof;
use crate::prover::prove_stir_multi_from_external_codewords;
use crate::utils::combine_on_coset;
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

/// One shared-LDE-domain group of a commitment: its own Merkle tree over the matrices whose
/// native heights the partition placed together.
///
/// Every matrix here is extended onto the same domain (sized to the group's tallest) and
/// committed in fiber-grouped form — each leaf holds `2^log_starting_folding_factor`
/// consecutive bit-reversed LDE rows, exactly the rows one first-round STIR query reads.
struct DomainGroup<Val: Send + Sync + Clone, InputMmcs: Mmcs<Val>> {
    data: InputMmcs::ProverData<RowMajorMatrix<Val>>,
    /// Column count of each matrix in this group, in the order the caller committed them.
    widths: Vec<usize>,
    /// Native (pre-extension) log2 height of each matrix in this group, same order. This is
    /// what distinguishes matrices for alpha-batching and `Combine` grouping once they all
    /// sit on the same physical domain.
    log_native_heights: Vec<usize>,
    /// Log2 of the shared LDE domain this group's matrices were extended onto.
    log_lde_height: usize,
}

/// Prover data for [`TwoAdicStirPcs`].
///
/// The matrices passed to one `commit()` call are partitioned into shared-domain groups of
/// bounded height spread ([`TwoAdicStirPcs::with_max_log_height_spread`]), each committed in its
/// own tree. A group of one native height runs no `Combine` at all; a group spanning several
/// merges them on its shared domain per §7's same-domain requirement.
pub struct StirProverData<Val: Send + Sync + Clone, InputMmcs: Mmcs<Val>> {
    /// Groups in descending LDE height, matching the commitment's root order.
    groups: Vec<DomainGroup<Val, InputMmcs>>,
    /// `placement[i] = (group index, index within that group)` for the caller's matrix `i`.
    placement: Vec<(usize, usize)>,
}

impl<Val: Send + Sync + Clone, InputMmcs: Mmcs<Val>> StirProverData<Val, InputMmcs> {
    /// `(native log2 height, log2 LDE height of the group holding it)` per matrix, in the
    /// order the caller committed them.
    fn matrix_layout(&self) -> Vec<(usize, usize)> {
        self.placement
            .iter()
            .map(|&(group_idx, idx)| {
                let group = &self.groups[group_idx];
                (group.log_native_heights[idx], group.log_lde_height)
            })
            .collect()
    }

    /// The group committed on the domain of size `2^log_lde_height`, if this commitment has
    /// one. Group LDE heights are distinct, so at most one can match.
    fn group_at(&self, log_lde_height: usize) -> Option<&DomainGroup<Val, InputMmcs>> {
        self.groups
            .iter()
            .find(|group| group.log_lde_height == log_lde_height)
    }
}

/// How one commitment's matrices are partitioned across shared LDE domains.
///
/// Derived identically by the prover (from the committed heights) and the verifier (from the
/// claimed domain sizes), so no part of it travels in the proof.
struct GroupPlan {
    /// Log2 LDE height of each group, descending.
    log_lde_heights: Vec<usize>,
    /// Group index of each matrix, in the order the caller supplied them.
    group_of_matrix: Vec<usize>,
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
///
/// Matrices live in per-group trees, so the caller order is reassembled through
/// [`StirProverData::placement`].
fn lde_views<'a, Val: Send + Sync + Clone, InputMmcs: Mmcs<Val>>(
    input_mmcs: &InputMmcs,
    prover_data: &'a StirProverData<Val, InputMmcs>,
) -> Vec<RowMajorMatrixView<'a, Val>> {
    let per_group: Vec<Vec<&'a RowMajorMatrix<Val>>> = prover_data
        .groups
        .iter()
        .map(|group| input_mmcs.get_matrices(&group.data))
        .collect();

    prover_data
        .placement
        .iter()
        .map(|&(group_idx, idx)| {
            let grouped = per_group[group_idx][idx];
            let width = prover_data.groups[group_idx].widths[idx];
            RowMajorMatrixView::new(grouped.values.as_slice(), width)
        })
        .collect()
}

/// Key identifying a derived STIR config: the bucket's degree plus, when §7's `Combine` runs,
/// the class count and multiplicity that size round 0's `eta`.
///
/// `(log_stir_degree, 1, 0)` is the canonical no-`Combine` key; it cannot collide with a
/// `Combine` key, since [`StirConfig::try_new_with_combine`] rejects `num_classes < 2`.
type StirConfigKey = (usize, usize, u64);

/// STIR configs derived on demand, memoized by [`StirConfigKey`].
type StirConfigCache<Val, Challenge, StirMmcs, Challenger> = Arc<
    RwLock<
        alloc::collections::BTreeMap<
            StirConfigKey,
            Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>>,
        >,
    >,
>;

/// Cap on memoized configs.
///
/// The key space is bounded by the base field's two-adicity and the height shapes a caller
/// actually commits, but `verify` derives its keys from claim shapes, so a hard cap keeps a
/// pathological caller from growing the map without bound. Past the cap, derivation still
/// returns a correct config — just an unmemoized one.
const CONFIG_CACHE_CAPACITY: usize = 256;

/// Default cap on the height spread sharing one LDE domain.
///
/// Merging native-height classes onto one domain trades commit work for proof size: a matrix
/// `s` octaves below its group's tallest pays a `2^s` larger blowup, while §7's `Combine`
/// removes a whole STIR instance and its query-count floor. Three octaves is the spread of
/// the shape that trade was measured on; past it the extra DFT and Merkle work grows without
/// bound while `Combine`'s return does not, so wider spreads get their own domain.
pub const DEFAULT_MAX_LOG_HEIGHT_SPREAD: usize = 3;

/// A polynomial commitment scheme using STIR to generate opening proofs.
#[derive(Clone, Debug)]
pub struct TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger> {
    dft: Dft,
    input_mmcs: InputMmcs,
    stir: StirParameters<StirMmcs>,
    /// Maximum `h_max - h_min`, in octaves, among the native heights sharing one LDE domain.
    ///
    /// `0` puts every distinct native height on its own domain, so `Combine` never runs and
    /// each height gets its own STIR instance. A value at or above the committed spread puts
    /// everything on one domain. See [`DEFAULT_MAX_LOG_HEIGHT_SPREAD`].
    max_log_height_spread: usize,
    /// `StirConfig::try_new` runs an 80-iteration floating-point bisection per stage to
    /// derive sound round parameters. `open`/`verify` re-derive it per LDE-height bucket, and
    /// bucket shapes recur across calls and across proofs of the same statement, so caching
    /// them here avoids repeating that derivation every time. `Combine` configs are keyed by
    /// their `(num_classes, ell)` alongside the degree: with matrices sharing one domain, a
    /// commitment holding several native heights always takes the `Combine` branch, so a
    /// degree-only key would miss on exactly the shape this PCS exists for.
    config_cache: StirConfigCache<Val, Challenge, StirMmcs, Challenger>,
}

impl<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
    TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
{
    pub fn new(dft: Dft, input_mmcs: InputMmcs, stir: StirParameters<StirMmcs>) -> Self {
        Self {
            dft,
            input_mmcs,
            stir,
            max_log_height_spread: DEFAULT_MAX_LOG_HEIGHT_SPREAD,
            config_cache: Arc::new(RwLock::new(alloc::collections::BTreeMap::new())),
        }
    }

    /// Override how wide a native-height spread may share one LDE domain.
    ///
    /// Both sides of a proof must agree on this, since it decides the commit layout and how
    /// many STIR instances a proof holds; it is not carried in the proof.
    #[must_use]
    pub const fn with_max_log_height_spread(mut self, max_log_height_spread: usize) -> Self {
        self.max_log_height_spread = max_log_height_spread;
        self
    }

    /// Commit one tree per shared-domain group.
    ///
    /// `plan` assigns matrices to groups; `grouped[i]` is matrix `i`'s fiber-grouped LDE,
    /// already extended onto the domain of its own group.
    fn commit_groups(
        &self,
        plan: &GroupPlan,
        grouped: Vec<RowMajorMatrix<Val>>,
        log_native_heights: &[usize],
        widths: &[usize],
    ) -> (Vec<InputMmcs::Commitment>, StirProverData<Val, InputMmcs>)
    where
        Val: Send + Sync + Clone,
        InputMmcs: Mmcs<Val>,
    {
        let num_groups = plan.log_lde_heights.len();
        let mut per_group: Vec<Vec<RowMajorMatrix<Val>>> = vec![Vec::new(); num_groups];
        let mut per_group_heights: Vec<Vec<usize>> = vec![Vec::new(); num_groups];
        let mut per_group_widths: Vec<Vec<usize>> = vec![Vec::new(); num_groups];
        let mut placement = Vec::with_capacity(grouped.len());

        for (matrix_idx, matrix) in grouped.into_iter().enumerate() {
            let group_idx = plan.group_of_matrix[matrix_idx];
            placement.push((group_idx, per_group[group_idx].len()));
            per_group[group_idx].push(matrix);
            per_group_heights[group_idx].push(log_native_heights[matrix_idx]);
            per_group_widths[group_idx].push(widths[matrix_idx]);
        }

        let (commitments, groups) = izip!(
            per_group,
            per_group_widths,
            per_group_heights,
            &plan.log_lde_heights
        )
        .map(|(matrices, widths, log_native_heights, &log_lde_height)| {
            let (commitment, data) = self.input_mmcs.commit(matrices);
            (
                commitment,
                DomainGroup {
                    data,
                    widths,
                    log_native_heights,
                    log_lde_height,
                },
            )
        })
        .unzip();

        (commitments, StirProverData { groups, placement })
    }
}

impl<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
    TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    StirMmcs: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger<Witness = Val>,
{
    /// Returns the derived STIR config for one bucket, computing and caching it on first use.
    ///
    /// `combine` carries the bucket's `(num_classes, ell)` when more than one native-height
    /// class shares its domain and §7's `Combine` therefore runs, and is `None` otherwise.
    fn get_or_try_compute_stir_config(
        &self,
        log_stir_degree: usize,
        combine: Option<(usize, u64)>,
    ) -> Result<Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>>, StirConfigError> {
        let key: StirConfigKey = combine.map_or((log_stir_degree, 1, 0), |(classes, ell)| {
            (log_stir_degree, classes, ell)
        });

        if let Some(config) = self.config_cache.read().get(&key) {
            return Ok(config.clone());
        }

        // Derived before the write guard is taken: `spin::RwLock` does not park, so holding it
        // across the bisection would make a thread missing on *any* key busy-spin for the
        // whole derivation — under rayon, possibly while the holder is descheduled. The
        // derivation is idempotent, so a racing duplicate is harmless: the loser's `Arc` is
        // simply dropped in favour of whichever landed first.
        let config = Arc::new(match combine {
            Some((num_classes, ell)) => StirConfig::try_new_with_combine(
                log_stir_degree,
                self.stir.clone(),
                num_classes,
                ell,
            )?,
            None => StirConfig::try_new(log_stir_degree, self.stir.clone())?,
        });

        let mut cache = self.config_cache.write();
        if cache.len() >= CONFIG_CACHE_CAPACITY && !cache.contains_key(&key) {
            return Ok(config);
        }
        Ok(cache.entry(key).or_insert(config).clone())
    }

    /// Like [`Self::get_or_try_compute_stir_config`], but panics on an infeasible config —
    /// for use on the prover side, where `open` cannot return a `Result`.
    fn get_or_compute_stir_config(
        &self,
        log_stir_degree: usize,
        combine: Option<(usize, u64)>,
    ) -> Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>> {
        self.get_or_try_compute_stir_config(log_stir_degree, combine)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Log2 STIR degree of a bucket whose shared LDE domain has size `2^log_lde_height`.
    fn log_stir_degree(&self, log_lde_height: usize) -> usize {
        log_lde_height.saturating_sub(self.stir.log_blowup).max(1)
    }

    /// Widest band of native heights below `tallest`, in octaves, that may share `tallest`'s
    /// LDE domain.
    ///
    /// Growth stops at the first of two limits: the configured spread cap, which bounds the
    /// extra blowup a short matrix pays for sitting on a taller group's domain; and `Combine`
    /// feasibility, which is what makes an infeasible parameter set degrade into more STIR
    /// instances rather than failing.
    ///
    /// Feasibility is probed against the *whole* band `[tallest - w, tallest]` rather than the
    /// heights a particular commitment happens to hold. That matters because a bucket pools
    /// every group topped at `tallest`, across all commitments opened together, and merges
    /// their classes into one `Combine`. Adding a class raises Lemma 4.13's `ell` by
    /// `2^tallest + 1 - 2^h > 0`, so the full band maximizes `ell` over every subset that
    /// could form, and a width feasible for it is feasible for whatever union actually shows
    /// up. Since the width depends only on `tallest` and this PCS's parameters, every
    /// commitment independently agrees on it.
    ///
    /// A width of `0` runs no `Combine` at all, so this always terminates: in the worst case
    /// every distinct height gets its own domain and its own STIR instance.
    ///
    /// The configs derived while probing are served from the same cache the bucket
    /// construction later reads, so a repeated shape pays for them once.
    fn combine_band_width(&self, tallest: usize) -> usize {
        let log_stir_degree = self.log_stir_degree(tallest + self.stir.log_blowup);
        let max_width = self.max_log_height_spread.min(tallest);

        let mut width = 0;
        while width < max_width {
            let band: Vec<usize> = (0..=width + 1).map(|i| tallest - i).collect();
            if self
                .get_or_try_compute_stir_config(log_stir_degree, Self::combine_key(&band))
                .is_err()
            {
                break;
            }
            width += 1;
        }
        width
    }

    /// Partition distinct native heights, descending, into shared-domain groups.
    ///
    /// Returns each group's size, so group `g` covers the slice starting after the previous
    /// groups. Each group takes every remaining height inside the band
    /// [`Self::combine_band_width`] admits below its tallest.
    fn partition_native_heights(&self, descending: &[usize]) -> Vec<usize> {
        let mut sizes = Vec::new();
        let mut start = 0;

        while start < descending.len() {
            let tallest = descending[start];
            let floor = tallest - self.combine_band_width(tallest);
            let size = descending[start..]
                .iter()
                .take_while(|&&h| h >= floor)
                .count();
            sizes.push(size);
            start += size;
        }

        sizes
    }

    /// Assign a commitment's matrices to shared LDE domains.
    ///
    /// Depends only on the multiset of native heights and this PCS's parameters, so the
    /// verifier reproduces it exactly from the claimed domain sizes — the layout is never
    /// carried in the proof, and a prover that used a different one fails the input MMCS
    /// check, whose dimensions it fixes.
    fn plan_groups(&self, log_native_heights: &[usize]) -> GroupPlan {
        let mut distinct: Vec<usize> = log_native_heights.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        distinct.reverse();

        let sizes = self.partition_native_heights(&distinct);

        // Group index of each distinct native height, then of each matrix through it.
        let mut group_of_height: alloc::collections::BTreeMap<usize, usize> =
            alloc::collections::BTreeMap::new();
        let mut log_lde_heights = Vec::with_capacity(sizes.len());
        let mut offset = 0;
        for (group_idx, size) in sizes.into_iter().enumerate() {
            log_lde_heights.push(distinct[offset] + self.stir.log_blowup);
            for &log_native_h in &distinct[offset..offset + size] {
                group_of_height.insert(log_native_h, group_idx);
            }
            offset += size;
        }

        let group_of_matrix = log_native_heights
            .iter()
            .map(|log_native_h| group_of_height[log_native_h])
            .collect();

        GroupPlan {
            log_lde_heights,
            group_of_matrix,
        }
    }

    /// A bucket's `Combine` key: `None` when only one native-height class shares the domain.
    ///
    /// `ell` is Lemma 4.13's multiplicity `num_classes·(d* + 1) − Σᵢ dᵢ`, with `d*` the
    /// tallest class's degree (`native_heights` is descending).
    fn combine_key(native_heights: &[usize]) -> Option<(usize, u64)> {
        (native_heights.len() >= 2).then(|| {
            let d_star = 1u64 << native_heights[0];
            let ell = native_heights.len() as u64 * (d_star + 1)
                - native_heights.iter().map(|&d| 1u64 << d).sum::<u64>();
            (native_heights.len(), ell)
        })
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
    for TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
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
    /// One Merkle root per shared-domain group, in descending LDE height.
    ///
    /// The matrices of one `commit()` call are partitioned into groups of bounded height
    /// spread, each extended onto its own shared domain (§7's same-domain requirement applies
    /// within a group, not across the whole commitment) and committed in its own tree. A
    /// commitment whose heights all fit one group therefore has a single root, and callers
    /// observe the roots in order.
    type Commitment = Vec<InputMmcs::Commitment>;
    type ProverData = StirProverData<Val, InputMmcs>;
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixCow<'a, Val>>;
    /// Proof structure: one entry per distinct shared LDE height across every commitment's
    /// groups (descending). A commitment contributes to one entry per group it holds.
    ///
    /// Each bucket contains:
    /// - `stir_proof`: the STIR IOP proof for that bucket (per-round IOP messages; the initial
    ///   oracle is external, so neither its commitment nor its openings appear). The
    ///   first-round query indices are NOT serialized — the verifier re-derives them from the
    ///   transcript.
    /// - `input_openings[commit_idx]`: one shared multi-opening proof for that commitment's
    ///   rows at the bucket's first-round STIR fiber positions, in the same sorted-by-index
    ///   order the verifier reconstructs. `None` if the commitment has no group at this
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
        let log_native_heights: Vec<usize> = inputs
            .iter()
            .map(|(domain, _)| log2_strict_usize(domain.size()))
            .collect();
        let plan = self.plan_groups(&log_native_heights);

        let mut widths = Vec::with_capacity(inputs.len());
        let grouped: Vec<_> = inputs
            .into_iter()
            .zip(&log_native_heights)
            .zip(&plan.group_of_matrix)
            .map(|(((domain, evals), &log_native_height), &group_idx)| {
                // Effective per-matrix blowup: `log_blowup` for the tallest matrix in the
                // group, and one extra bit per octave of height below it — which is what the
                // spread cap bounds. See the module-level cost note.
                let extra_bits = plan.log_lde_heights[group_idx] - log_native_height;
                let shift = Val::GENERATOR / domain.shift();
                let lde = self
                    .dft
                    .coset_lde_batch(evals, extra_bits, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix();
                widths.push(lde.width());
                group_fiber_rows(lde, self.stir.log_starting_folding_factor)
            })
            .collect();
        self.commit_groups(&plan, grouped, &log_native_heights, &widths)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let (group_idx, idx_in_group) = prover_data.placement[idx];
        let group = &prover_data.groups[group_idx];
        let grouped = self.input_mmcs.get_matrices(&group.data)[idx_in_group];
        let lde = RowMajorMatrixView::new(grouped.values.as_slice(), group.widths[idx_in_group]);
        if domain.shift() == Val::GENERATOR && lde.height() >= domain.size() {
            let width = lde.width();
            let values: &'a [Val] = lde.values;
            return RowMajorMatrixView::new(&values[..domain.size() * width], width)
                .as_cow()
                .bit_reverse_rows();
        }
        let poly_height = 1usize << group.log_native_heights[idx_in_group];
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
        let plan = self.plan_groups(&log_native_heights);

        let mut widths = Vec::with_capacity(ldes.len());
        let grouped: Vec<_> = ldes
            .into_iter()
            .zip(&log_native_heights)
            .zip(&plan.group_of_matrix)
            .map(|((lde, &log_native_height), &group_idx)| {
                widths.push(lde.width());
                let log_lde_height = plan.log_lde_heights[group_idx];
                let extended = if lde.height() == 1usize << log_lde_height {
                    lde
                } else {
                    let natural_lde = lde.bit_reverse_rows().to_row_major_matrix();
                    let mut coeffs = self.dft.coset_idft_batch(natural_lde, Val::GENERATOR);
                    let width = coeffs.width();
                    coeffs
                        .values
                        .truncate((1usize << log_native_height) * width);
                    self.dft
                        .coset_lde_batch(coeffs, log_lde_height - log_native_height, Val::GENERATOR)
                        .bit_reverse_rows()
                        .to_row_major_matrix()
                };
                group_fiber_rows(extended, self.stir.log_starting_folding_factor)
            })
            .collect();
        self.commit_groups(&plan, grouped, &log_native_heights, &widths)
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

        // `(native height, group LDE height)` per matrix, in caller order: the first selects
        // the `Combine` class, the second selects which STIR instance that class feeds.
        let matrix_layouts: Vec<Vec<(usize, usize)>> = commitment_data_with_opening_points
            .iter()
            .map(|(data, _)| data.matrix_layout())
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
            .zip(&matrix_layouts)
            .map(|((mats, points), layout)| {
                izip!(mats.iter(), points.iter(), layout.iter())
                    .map(|(mat, points_for_mat, &(log_native_h, _))| {
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

        for (((mats, points), opened_vals), layout) in mats_and_points
            .iter()
            .zip(&all_opened_values)
            .zip(&matrix_layouts)
        {
            for (((mat, points_for_mat), opened_for_mat), &(log_native_h, log_lde_h)) in
                izip!(mats.iter(), points.iter())
                    .zip(opened_vals.iter())
                    .zip(layout.iter())
            {
                // A matrix opened at no points would contribute nothing to the reduced
                // opening, but the verifier still counts it as a native-height class (it reads
                // class membership off the claimed domains), so skipping it here would emit a
                // proof that cannot verify. `verify` rejects the same shape up front; this is
                // the prover-side mirror.
                assert!(
                    !points_for_mat.is_empty(),
                    "STIR PCS: matrix at native height 2^{log_native_h} was opened at no \
                     points; every committed matrix must be opened at least once"
                );

                let key = (log_lde_h, log_native_h);
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

        let stir_configs: Vec<Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>>> =
            bucket_log_heights
                .iter()
                .zip(&bucket_native_heights)
                .map(|(&log_h, native_heights)| {
                    let log_stir_degree = self.log_stir_degree(log_h);
                    self.get_or_compute_stir_config(
                        log_stir_degree,
                        Self::combine_key(native_heights),
                    )
                })
                .collect();
        let stir_config_refs: Vec<&StirConfig<Val, Challenge, StirMmcs, Challenger>> =
            stir_configs.iter().map(AsRef::as_ref).collect();

        let initial_codewords: Vec<Vec<Challenge>> = bucket_log_heights
            .iter()
            .map(|&log_shared_h| {
                combined_bucket_codeword::<Val, Challenge, Challenger>(
                    &mut reduced_openings,
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
                                // Each group has its own tree on its own domain, so a bucket
                                // reads exactly the group committed at its LDE height —
                                // never a partial slice of one, and nothing from the others.
                                let group = data.group_at(log_h)?;

                                let q_globals: Vec<usize> = first_round_query_indices
                                    .iter()
                                    .map(|&j| reverse_bits_len(j, log_h - log_arity0))
                                    .collect();

                                let (opened_values, opening_proof) =
                                    self.input_mmcs.open_multi_batch(&q_globals, &group.data);
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
        // SHAPE CHECK, before anything reaches the transcript: a matrix opened at no points
        // carries no claim to pin its width, and the prover skips it entirely — so it would
        // not create a native-height class where the verifier, reading class membership off
        // the claimed domains, still counts one. That disagreement decides whether `Combine`
        // runs and therefore whether `r_comb` is drawn, so it has to be settled before the
        // transcript can fork on it. Depends only on the public claims (mirrors
        // `fri::verifier::FriError::MatrixWithoutOpeningPoints`).
        for (commitment, domain_claims) in commitments_with_opening_points
            .iter()
            .enumerate()
            .map(|(commit_idx, (_, domain_claims))| (commit_idx, domain_claims))
        {
            for (matrix, (_, point_claims)) in domain_claims.iter().enumerate() {
                if point_claims.is_empty() {
                    return Err(StirError::MatrixWithoutOpeningPoints { commitment, matrix });
                }
            }
        }

        // Observe all opened values to keep the transcript in sync.
        for (_, domain_claims) in &commitments_with_opening_points {
            for (_, point_claims) in domain_claims {
                for (_, opened_vals) in point_claims {
                    challenger.observe_algebra_slice(opened_vals);
                }
            }
        }

        let alpha: Challenge = challenger.sample_algebra_element();

        // Reproduce each commitment's shared-domain layout from the claimed domain sizes,
        // exactly as `commit` derived it from the committed ones. Nothing about the layout
        // travels in the proof: it is a function of the claimed heights and this PCS's
        // parameters, and a prover that used a different one fixed different MMCS dimensions
        // and fails the input opening check below.
        let plans: Vec<GroupPlan> = commitments_with_opening_points
            .iter()
            .map(|(_, domain_claims)| {
                let log_native_heights: Vec<usize> = domain_claims
                    .iter()
                    .map(|(domain, _)| log2_strict_usize(domain.size()))
                    .collect();
                self.plan_groups(&log_native_heights)
            })
            .collect();

        // SHAPE CHECK: one Merkle root per group of the layout the claims imply.
        for ((commitment, _), plan) in commitments_with_opening_points.iter().zip(&plans) {
            if commitment.len() != plan.log_lde_heights.len() {
                return Err(StirError::InvalidProofShape);
            }
        }

        // Log2 LDE height of the domain each matrix sits on, in claimed order.
        let matrix_lde_heights: Vec<Vec<usize>> = plans
            .iter()
            .map(|plan| {
                plan.group_of_matrix
                    .iter()
                    .map(|&group_idx| plan.log_lde_heights[group_idx])
                    .collect()
            })
            .collect();

        // Distinct outer buckets (shared LDE heights), descending. Must match the prover's
        // bucket iteration order.
        let bucket_log_heights: Vec<usize> = {
            let mut heights: Vec<usize> = plans
                .iter()
                .flat_map(|plan| plan.log_lde_heights.iter().copied())
                .collect();
            heights.sort_unstable();
            heights.dedup();
            heights.reverse();
            heights
        };

        if proof.len() != bucket_log_heights.len() {
            return Err(StirError::InvalidProofShape);
        }

        // Which of a commitment's groups (hence which of its Merkle roots) feeds each
        // bucket, if any. Group LDE heights within a commitment are distinct, so this is
        // at most one group per bucket.
        let bucket_group_indices: Vec<Vec<Option<usize>>> = bucket_log_heights
            .iter()
            .map(|&log_h| {
                plans
                    .iter()
                    .map(|plan| plan.log_lde_heights.iter().position(|&h| h == log_h))
                    .collect()
            })
            .collect();

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
            .zip(&matrix_lde_heights)
            .map(|((_, domain_claims), lde_heights)| {
                domain_claims
                    .iter()
                    .zip(lde_heights)
                    .map(|((domain, point_claims), &log_lde_h)| {
                        let key = (log_lde_h, log2_strict_usize(domain.size()));
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
                    .zip(&matrix_lde_heights)
                    .flat_map(|((_, domain_claims), lde_heights)| {
                        domain_claims
                            .iter()
                            .zip(lde_heights)
                            .filter(move |&(_, &log_lde_h)| log_lde_h == log_shared_h)
                            .map(|((domain, _), _)| log2_strict_usize(domain.size()))
                    })
                    .collect();
                native_heights.sort_unstable();
                native_heights.dedup();
                native_heights.reverse();
                native_heights
            })
            .collect();

        let stir_configs: Vec<Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>>> =
            bucket_log_heights
                .iter()
                .zip(&bucket_native_heights)
                .map(|(&log_h, native_heights)| {
                    let log_stir_degree = self.log_stir_degree(log_h);
                    self.get_or_try_compute_stir_config(
                        log_stir_degree,
                        Self::combine_key(native_heights),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(StirError::Config)?;
        let stir_config_refs: Vec<&StirConfig<Val, Challenge, StirMmcs, Challenger>> =
            stir_configs.iter().map(AsRef::as_ref).collect();
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
                let coeffs =
                    combine_coefficients(r_comb, log_d_star, native_heights.iter().copied());
                Some((r_comb, native_heights.iter().copied().zip(coeffs).collect()))
            })
            .collect();

        // Captured by every bucket's closure below; taking references up front lets `move`
        // give each closure its own copy of the reference rather than the whole value.
        let commitments_with_opening_points = &commitments_with_opening_points;
        let matrix_lde_heights = &matrix_lde_heights;
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
            .zip(&bucket_native_heights)
            .zip(&bucket_group_indices)
            .map(
                |(
                    ((((&log_h, stir_config), input_openings), combine_info), native_heights),
                    group_indices,
                )| {
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

                    // One accumulator per native-height class present in this bucket, indexed
                    // as `native_heights` is (descending); merged into the final expected
                    // codeword fibers after the accumulation loop.
                    let mut expected_ro_by_class: Vec<Vec<Vec<Challenge>>> =
                        vec![vec![Challenge::zero_vec(arity0); n_q]; native_heights.len()];

                    // Distinct opening points among matrices active at this bucket. Matrices
                    // typically share opening points (e.g. one STARK's `zeta`), so this list
                    // is usually far shorter than the matrix count.
                    let bucket_points: Vec<Challenge> = commitments_with_opening_points
                        .iter()
                        .zip(matrix_lde_heights.iter())
                        .flat_map(|((_, domain_claims), lde_heights)| {
                            domain_claims
                                .iter()
                                .zip(lde_heights)
                                .filter(move |&(_, &h)| h == log_h)
                                .map(|(claim, _)| claim)
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

                    // A commitment feeds this bucket through at most one of its groups: the
                    // one committed on this domain. Its other groups live in other trees at
                    // other heights and belong to other buckets.
                    for (commit_idx, ((commitment, domain_claims), per_commit_opening)) in
                        commitments_with_opening_points
                            .iter()
                            .zip(input_openings.iter())
                            .enumerate()
                    {
                        let group_idx = group_indices[commit_idx];

                        let Some(opening) = per_commit_opening else {
                            if group_idx.is_some() {
                                return Err(StirError::InvalidProofShape);
                            }
                            continue;
                        };
                        let Some(group_idx) = group_idx else {
                            return Err(StirError::InvalidProofShape);
                        };

                        // The commitment's matrices on this domain, in the order they were
                        // committed to its tree: caller order, filtered to this group.
                        let group_mats: Vec<usize> = matrix_lde_heights[commit_idx]
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, &h)| (h == log_h).then_some(idx))
                            .collect();

                        // Pin each matrix's width to its claimed evaluation count, never to
                        // the proof. Every matrix has at least one claim — the up-front check
                        // in `verify` rejects otherwise before the transcript is touched.
                        let mat_widths: Vec<usize> = group_mats
                            .iter()
                            .map(|&idx| {
                                domain_claims[idx]
                                    .1
                                    .first()
                                    .map(|(_, v)| v.len())
                                    .expect("rejected up front in verify")
                            })
                            .collect();

                        // A matrix's native-height class, and each of its opening points'
                        // slot in `bucket_points`, are fixed for the whole commitment. Both
                        // are resolved once here rather than once per
                        // `(query, lane, matrix, point)`, which is where the innermost loop
                        // below would otherwise re-scan `bucket_points` linearly.
                        let mat_class_indices: Vec<usize> = group_mats
                            .iter()
                            .map(|&idx| {
                                let log_native_h = log2_strict_usize(domain_claims[idx].0.size());
                                native_heights
                                    .iter()
                                    .position(|&h| h == log_native_h)
                                    .expect("bucket_native_heights is built from these claims")
                            })
                            .collect();
                        let mat_point_slots: Vec<Vec<usize>> = group_mats
                            .iter()
                            .map(|&idx| {
                                domain_claims[idx]
                                    .1
                                    .iter()
                                    .map(|(point, _)| {
                                        bucket_points
                                            .iter()
                                            .position(|p| p == point)
                                            .expect("point is in bucket_points by construction")
                                    })
                                    .collect()
                            })
                            .collect();

                        // Matrices are committed fiber-grouped: `2^log_arity0` LDE rows per
                        // committed row.
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
                                &commitment[group_idx],
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

                                for (mat_idx, point_slots) in
                                    mat_point_slots.iter().enumerate()
                                {
                                    // `mat_idx` indexes this group's tree, and therefore the
                                    // opened rows; `point_data` is keyed by the commitment's
                                    // full claim order, which `group_mats` maps back to.
                                    let claim_idx = group_mats[mat_idx];
                                    let width = mat_widths[mat_idx];
                                    let row_vals =
                                        &row_vals_by_mat[mat_idx][slot * width..][..width];
                                    let p_x: Challenge = row_vals
                                        .iter()
                                        .zip(alpha_powers.iter())
                                        .map(|(&v, &ap)| ap * v)
                                        .sum();

                                    let ro_class =
                                        &mut expected_ro_by_class[mat_class_indices[mat_idx]];

                                    for (point_idx, &bp_idx) in point_slots.iter().enumerate() {
                                        let (alpha_pow_offset, y_combined) =
                                            point_data[commit_idx][claim_idx][point_idx];
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
                            Ok(expected_ro_by_class.into_iter().next().unwrap())
                        }
                        Some((r_comb, coeffs_by_height)) => {
                            // SHAPE CHECK: every expected class must be present.
                            if expected_ro_by_class.len() != coeffs_by_height.len() {
                                return Err(StirError::InvalidProofShape);
                            }

                            // Pointwise mirror of the prover's `combine_on_coset`, evaluated
                            // only at the queried lanes. The `1 − r_comb·x` denominators do
                            // not depend on the class, so they are swept once for the whole
                            // fiber set and inverted in a single batch rather than once per
                            // `(class, query, lane)`.
                            let mut fiber_steps = Vec::with_capacity(n_q * arity0);
                            let mut denoms = Vec::with_capacity(n_q * arity0);
                            for &j in first_round_unique_js {
                                let mut fiber_point =
                                    Val::GENERATOR * domain_gen.exp_u64(j as u64);
                                for _ in 0..arity0 {
                                    let step = *r_comb * fiber_point;
                                    fiber_steps.push(step);
                                    denoms.push(Challenge::ONE - step);
                                    fiber_point *= fiber_step;
                                }
                            }

                            // The queried lanes are distinct coset points, so at most one can
                            // reach `step = 1`, where the geometric sum degenerates to
                            // `gap + 1`. Substituting a unit keeps the batch inversion defined
                            // and makes that lane's numerator vanish, so the sweep below
                            // contributes nothing there and the closed form is added back —
                            // the same handling `combine_on_coset` applies on the prover side.
                            let degenerate = denoms.iter().position(|d| d.is_zero());
                            if let Some(lane) = degenerate {
                                denoms[lane] = Challenge::ONE;
                            }
                            let inv_denoms = batch_multiplicative_inverse(&denoms);

                            let mut combined = vec![Challenge::zero_vec(arity0); n_q];
                            for (&log_native_h, ro_class) in
                                native_heights.iter().zip(&expected_ro_by_class)
                            {
                                let &(r_i, gap) = coeffs_by_height
                                    .get(&log_native_h)
                                    .ok_or(StirError::InvalidProofShape)?;

                                // Within a query the lanes advance by the fixed base-field
                                // ratio `fiber_step^(gap+1)`, so the numerator sweep costs one
                                // extension exponentiation per query instead of one per lane.
                                let gap_plus_1 = (gap + 1) as u64;
                                let lane_ratio = fiber_step.exp_u64(gap_plus_1);
                                for q_idx in 0..n_q {
                                    let base = q_idx * arity0;
                                    let mut step_hi = fiber_steps[base].exp_u64(gap_plus_1);
                                    for l in 0..arity0 {
                                        combined[q_idx][l] += r_i
                                            * ro_class[q_idx][l]
                                            * (Challenge::ONE - step_hi)
                                            * inv_denoms[base + l];
                                        step_hi *= lane_ratio;
                                    }
                                }

                                if let Some(lane) = degenerate {
                                    let (q_idx, l) = (lane / arity0, lane % arity0);
                                    combined[q_idx][l] += r_i
                                        * ro_class[q_idx][l]
                                        * Challenge::from_usize(gap + 1);
                                }
                            }
                            Ok(combined)
                        }
                    }
                    }
                },
            )
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
/// Returns `[(r_i, gap_i)]` where `gap_i = d* - dᵢ` (a degree count, not log) is what
/// `eval_degree_correction`/`combine_on_coset` expect directly.
///
/// # Panics
///
/// If any `log_d` exceeds `log_d_star`. Both call sites read `log_d_star` off the head of the
/// same descending list they pass in, so this holds by construction — but the invariant lives
/// outside the function, and an unchecked `d* - dᵢ` would wrap in release and yield a wrong
/// codeword rather than a failure.
fn combine_coefficients<EF: Field>(
    r_comb: EF,
    log_d_star: usize,
    sorted_log_ds: impl Iterator<Item = usize>,
) -> Vec<(EF, usize)> {
    let d_star = 1u64 << log_d_star;
    let mut running_exp = 0u64;
    sorted_log_ds
        .map(|log_d| {
            let r_i = r_comb.exp_u64(running_exp);
            let gap = d_star
                .checked_sub(1u64 << log_d)
                .expect("classes must be sorted descending, so every dᵢ <= d*")
                as usize;
            running_exp += 1 + gap as u64;
            (r_i, gap)
        })
        .collect()
}

/// Merge one shared-LDE-height bucket's native-height classes into a single codeword on
/// their shared domain, via `Combine` (§4.5) when more than one class is present.
///
/// `reduced_openings` is keyed by `(log_shared_lde_height, log_native_height)`; this removes
/// every class at `log_shared_h`, descending by native height, and returns the natural
/// (not yet bit-reversed) combined codeword STIR should run on.
///
/// Each class's codeword is taken out of the map rather than borrowed so it can be
/// un-bit-reversed in place: at PCS scale a class spans the whole shared domain, so cloning
/// them all would double peak memory for the duration of `Combine`.
fn combined_bucket_codeword<Val, Challenge, Challenger>(
    reduced_openings: &mut alloc::collections::BTreeMap<(usize, usize), Vec<Challenge>>,
    log_shared_h: usize,
    challenger: &mut Challenger,
) -> Vec<Challenge>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val> + TwoAdicField,
    Challenger: FieldChallenger<Val>,
{
    let mut log_ds: Vec<usize> = reduced_openings
        .keys()
        .filter(|(h, _)| *h == log_shared_h)
        .map(|&(_, log_d)| log_d)
        .collect();
    log_ds.sort_unstable_by(|a, b| b.cmp(a));

    // `combine_on_coset` indexes its inputs (and produces its output) in natural order, but
    // `reduced_openings` is bit-reversed (built from the bit-reversed LDE matrices), so each
    // class's codeword is un-reversed before combining; the combined result is then already
    // in the natural order STIR expects, with no further reversal needed.
    let mut natural_ros: Vec<Vec<Challenge>> = log_ds
        .iter()
        .map(|&log_d| {
            let mut natural = reduced_openings
                .remove(&(log_shared_h, log_d))
                .expect("key came from this map");
            reverse_slice_index_bits(&mut natural);
            natural
        })
        .collect();

    if natural_ros.len() == 1 {
        return natural_ros.pop().expect("checked non-empty above");
    }

    let log_d_star = log_ds[0];
    let r_comb: Challenge = challenger.sample_algebra_element();
    let coeffs = combine_coefficients(r_comb, log_d_star, log_ds.iter().copied());

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

#[cfg(test)]
mod tests {
    use alloc::format;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_security::whir::SecurityAssumption;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;

    use super::*;

    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn combine_coefficients_matches_definition_4_11() {
        // Definition 4.11 fixes `r_1 = 1` and `r_i = r^{(i-1) + Σ_{j<i}(d* - d_j)}`. Pinning
        // the exponents to that closed form rather than to the running-sum recurrence itself
        // is what keeps the two from drifting together: a wrong `r_i` still produces a
        // low-degree combined codeword, so prover and verifier would agree on a
        // miscomputation and STIR would accept it.
        //
        // The exponents are what make class `i` occupy the consecutive power block
        // `[e_i, e_i + gap_i]`, with the blocks tiling `[0, ell - 1]` without overlap. For
        // `d_i = [8, 4, 2]` and `d* = 8` the gaps are `[0, 4, 6]`, so the blocks are
        // `[0,0] | [1,5] | [6,12]` — exponents `[0, 1, 6]` and `ell = 13`.
        let r_comb = EF::from_u64(3);
        let coeffs = combine_coefficients(r_comb, 3, [3usize, 2, 1].into_iter());

        assert_eq!(
            coeffs,
            vec![(EF::ONE, 0), (r_comb.exp_u64(1), 4), (r_comb.exp_u64(6), 6),]
        );

        // The blocks tile exactly `Σᵢ (gapᵢ + 1)`, which is the `ell` the config's Combine
        // soundness accounting is charged at.
        let ell: usize = coeffs.iter().map(|&(_, gap)| gap + 1).sum();
        assert_eq!(ell, 13);
    }

    #[test]
    #[should_panic(expected = "classes must be sorted descending")]
    fn combine_coefficients_rejects_a_class_above_d_star() {
        // `d_i > d*` would wrap the `d* - d_i` subtraction in release and yield a wrong
        // codeword rather than a failure, so the precondition is checked rather than assumed.
        let _ = combine_coefficients(EF::from_u64(3), 3, [3usize, 4].into_iter());
    }

    type TestVal = BabyBear;
    type TestPerm = Poseidon2BabyBear<16>;
    type TestHash = PaddingFreeSponge<TestPerm, 16, 8, 8>;
    type TestCompress = TruncatedPermutation<TestPerm, 2, 8, 16>;
    type TestPacked = <TestVal as Field>::Packing;
    type TestValMmcs = MerkleTreeMmcs<TestPacked, TestPacked, TestHash, TestCompress, 2, 8>;
    type TestStirMmcs = ExtensionMmcs<TestVal, EF, TestValMmcs>;
    type TestChallenger = DuplexChallenger<TestVal, TestPerm, 16, 8>;
    type TestPcs = TwoAdicStirPcs<
        TestVal,
        Radix2DitParallel<TestVal>,
        TestValMmcs,
        TestStirMmcs,
        EF,
        TestChallenger,
    >;
    type TestConfig = StirConfig<TestVal, EF, TestStirMmcs, TestChallenger>;

    /// Every value the schedule derives, in one comparable string. `StirConfig` has no
    /// `PartialEq`, and only the derived schedule matters here — the `mmcs` field is cloned
    /// straight from the shared parameters.
    fn schedule_fingerprint(config: &TestConfig) -> alloc::string::String {
        format!(
            "{:?}|{}|{}|{}|{}|{}|{}|{}|{}|{:?}|{}|{}|{:?}",
            config.soundness_type,
            config.log_starting_degree,
            config.security_level,
            config.max_pow_bits,
            config.log_blowup,
            config.log_folding_factor,
            config.log_starting_folding_factor,
            config.log_final_degree,
            config.final_queries,
            config.final_eta,
            config.final_pow_bits,
            config.final_folding_pow_bits,
            config.round_configs,
        )
    }

    /// A PCS over `TestVal`/`EF` at the given layout and soundness knobs.
    fn test_pcs_with(
        max_log_height_spread: usize,
        soundness_type: SecurityAssumption,
        security_level: usize,
    ) -> TestPcs {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
        let perm = TestPerm::new_from_rng_128(&mut rng);
        let val_mmcs = TestValMmcs::new(TestHash::new(perm.clone()), TestCompress::new(perm), 0);
        let stir = StirParameters {
            log_blowup: 1,
            log_folding_factor: 2,
            log_starting_folding_factor: 2,
            soundness_type,
            security_level,
            max_pow_bits: 0,
            mmcs: TestStirMmcs::new(val_mmcs.clone()),
        };
        TwoAdicStirPcs::new(Radix2DitParallel::default(), val_mmcs, stir)
            .with_max_log_height_spread(max_log_height_spread)
    }

    /// Group sizes of the plan for `log_native_heights`, in descending LDE height.
    fn group_sizes(plan: &GroupPlan) -> Vec<usize> {
        (0..plan.log_lde_heights.len())
            .map(|g| plan.group_of_matrix.iter().filter(|&&x| x == g).count())
            .collect()
    }

    #[test]
    fn groups_split_at_the_configured_spread() {
        let heights = [8usize, 6, 4];
        let cb = SecurityAssumption::CapacityBound;

        // Spread 2 admits 8 with 6 but not 8 with 4, so the run breaks after the second.
        let plan = test_pcs_with(2, cb, 32).plan_groups(&heights);
        assert_eq!(plan.log_lde_heights, vec![9, 5]);
        assert_eq!(plan.group_of_matrix, vec![0, 0, 1]);

        // Spread 1 admits none of these pairs.
        let plan = test_pcs_with(1, cb, 32).plan_groups(&heights);
        assert_eq!(plan.log_lde_heights, vec![9, 7, 5]);
        assert_eq!(plan.group_of_matrix, vec![0, 1, 2]);
    }

    #[test]
    fn zero_spread_is_the_per_height_class_layout() {
        // Every distinct native height on its own domain: no `Combine`, one STIR instance
        // each, and every matrix extended only by `log_blowup`.
        let heights = [8usize, 7, 6, 6];
        let plan = test_pcs_with(0, SecurityAssumption::CapacityBound, 32).plan_groups(&heights);

        assert_eq!(plan.log_lde_heights, vec![9, 8, 7]);
        assert_eq!(plan.group_of_matrix, vec![0, 1, 2, 2]);
        assert_eq!(group_sizes(&plan), vec![1, 1, 2]);
    }

    #[test]
    fn spread_above_the_committed_range_is_a_single_shared_domain() {
        let heights = [8usize, 6, 4];
        let plan = test_pcs_with(64, SecurityAssumption::CapacityBound, 32).plan_groups(&heights);

        assert_eq!(plan.log_lde_heights, vec![9]);
        assert_eq!(plan.group_of_matrix, vec![0, 0, 0]);
    }

    #[test]
    fn repeated_heights_share_one_class_and_one_group() {
        // Grouping is over *distinct* heights, so duplicates never open a new group and the
        // plan does not depend on how many matrices carry a given height.
        let cb = SecurityAssumption::CapacityBound;
        let pcs = test_pcs_with(2, cb, 32);

        let plan = pcs.plan_groups(&[8, 8, 6, 8]);
        assert_eq!(plan.log_lde_heights, vec![9]);
        assert_eq!(plan.group_of_matrix, vec![0, 0, 0, 0]);

        // Caller order does not matter either: a matrix follows its height.
        let plan = pcs.plan_groups(&[4, 8, 4, 6]);
        assert_eq!(plan.log_lde_heights, vec![9, 5]);
        assert_eq!(plan.group_of_matrix, vec![1, 0, 1, 0]);
    }

    #[test]
    fn groups_shrink_when_combine_does_not_fit() {
        // JohnsonBound at 80 bits over a 124-bit challenge field: each height configures on
        // its own, but merging the two does not. The spread cap would allow one group, so
        // this is feasibility alone deciding — an infeasible parameter set degrades into more
        // STIR instances instead of failing.
        let jb = SecurityAssumption::JohnsonBound;
        let heights = [12usize, 11];

        let pcs = test_pcs_with(8, jb, 80);
        let ell = 2 * ((1u64 << 12) + 1) - ((1u64 << 12) + (1u64 << 11));
        assert!(TestConfig::try_new(12, pcs.stir.clone()).is_ok());
        assert!(TestConfig::try_new_with_combine(12, pcs.stir.clone(), 2, ell).is_err());

        let plan = pcs.plan_groups(&heights);
        assert_eq!(plan.log_lde_heights, vec![13, 12]);
        assert_eq!(plan.group_of_matrix, vec![0, 1]);

        // The same shape at a target the merge does fit stays in one group, so the split
        // above is not just the spread cap in disguise.
        let plan = test_pcs_with(8, jb, 32).plan_groups(&heights);
        assert_eq!(plan.log_lde_heights, vec![13]);
    }

    fn test_pcs_and_params() -> (TestPcs, StirParameters<TestStirMmcs>) {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
        let perm = TestPerm::new_from_rng_128(&mut rng);
        let val_mmcs = TestValMmcs::new(TestHash::new(perm.clone()), TestCompress::new(perm), 0);
        let stir = StirParameters {
            log_blowup: 1,
            log_folding_factor: 2,
            log_starting_folding_factor: 2,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: 32,
            max_pow_bits: 0,
            mmcs: TestStirMmcs::new(val_mmcs.clone()),
        };
        (
            TwoAdicStirPcs::new(Radix2DitParallel::default(), val_mmcs, stir.clone()),
            stir,
        )
    }

    #[test]
    fn cached_configs_match_a_fresh_derivation() {
        // An under-specified cache key is silent in the worst way: a proof produced under one
        // config and checked under another. Deriving the same shapes twice through the cache
        // and comparing against an uncached derivation is what catches it — in particular that
        // two buckets sharing a degree but differing in class count do not collide.
        let (pcs, stir) = test_pcs_and_params();

        let shapes: [(usize, Option<(usize, u64)>); 4] = [
            (8, None),
            // Same degree, but merging two classes: a degree-only key would alias these.
            (8, Some((2, 194))),
            (8, Some((3, 300))),
            (6, None),
        ];

        for (log_stir_degree, combine) in shapes {
            let expected = match combine {
                Some((num_classes, ell)) => TestConfig::try_new_with_combine(
                    log_stir_degree,
                    stir.clone(),
                    num_classes,
                    ell,
                ),
                None => TestConfig::try_new(log_stir_degree, stir.clone()),
            }
            .expect("feasible shape");

            // Twice: the first call populates the entry, the second must return the same one.
            for round in 0..2 {
                let cached = pcs
                    .get_or_try_compute_stir_config(log_stir_degree, combine)
                    .expect("feasible shape");
                assert_eq!(
                    schedule_fingerprint(&cached),
                    schedule_fingerprint(&expected),
                    "deg={log_stir_degree} combine={combine:?} round={round}"
                );
            }
        }

        // One entry per distinct shape, so nothing aliased and nothing was inserted twice.
        assert_eq!(pcs.config_cache.read().len(), shapes.len());
    }

    #[test]
    fn cache_errors_are_not_memoized() {
        // A garbage claim shape must not be able to occupy a cache slot permanently.
        let (pcs, _) = test_pcs_and_params();
        assert!(
            pcs.get_or_try_compute_stir_config(8, Some((2, 1))).is_err(),
            "ell below the class count must be rejected"
        );
        assert!(pcs.config_cache.read().is_empty());
    }
}
