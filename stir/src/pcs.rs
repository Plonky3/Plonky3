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
//! per-class query-count floor). STIR commits that codeword itself and opens its round-0
//! fibers from that commitment. Each round-0 query draw is then tied to the inputs at one
//! uniformly sampled lane of its fiber: the prover opens the input LDE matrices (via
//! `InputMmcs`) at exactly those positions, one row each.
//!
//! **Verify**: reproduce the grouping from the claimed domain sizes — it is a pure function of
//! those and the PCS parameters, so no part of it travels in the proof — run
//! `verifier::verify_stir_multi_inner` on every bucket, then sample the same lanes,
//! authenticate the single input rows, rebuild the reduced opening at each position from its
//! row (replaying the same alpha-batching and `Combine`), and check it against the lane of the
//! fiber STIR authenticated. Reading one lane per draw costs one input row per query instead
//! of a whole fiber; the soundness of that binding is spelled out on `prove_buckets`.
//!
//! **Assumption**: because a lane check reaches below the round-0 fold domain, that binding
//! needs *mutual* correlated agreement for the round-0 fold — the property
//! [`SecurityAssumption::JohnsonBound`](p3_security::whir::SecurityAssumption::JohnsonBound)
//! already states, and strictly stronger than the plain correlated agreement STIR's own round
//! analysis needs. Under
//! [`SecurityAssumption::CapacityBound`](p3_security::whir::SecurityAssumption::CapacityBound),
//! whose documented assumption is only the plain variant, this PCS therefore assumes more than
//! that regime states (mutual correlated agreement up to capacity, the standard WHIR/ACFY
//! conjecture, charged the same error the crate already prices).
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
use core::ops::Deref;

use itertools::{Itertools, izip};
use p3_challenger::{
    CanObserve, CanSampleUniformBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
    SerializingChallenger32,
};
use p3_commit::{Mmcs, OpenedValues, Pcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PrimeCharacteristicRing,
    PrimeField32, TwoAdicField, batch_multiplicative_inverse,
};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixCow, RowMajorMatrixView};
use p3_matrix::interpolation::{Interpolate, compute_adjusted_weights};
use p3_maybe_rayon::prelude::*;
use p3_symmetric::CryptographicPermutation;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};
use serde::{Deserialize, Serialize};
use spin::RwLock;
use tracing::instrument;

use crate::config::{StirConfig, StirConfigError, StirParameters};
use crate::error::{ProofShapeError, StirError};
use crate::proof::StirProof;
use crate::prover::prove_stir_multi_from_codewords;
use crate::utils::{combine_on_coset, eval_degree_correction};
use crate::verifier::verify_stir_multi_inner;

/// Batched openings of one input commitment's LDE matrices at the STIR-derived query
/// positions for one LDE-height bucket.
///
/// One multi-opening proof authenticates every opened row together, so sibling digests
/// shared between the bucket's queried positions travel once.
///
/// `None` when the commitment has no matrix at this bucket's height.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Val: Serialize, InputMmcs::MultiProof: Serialize",
    deserialize = "Val: Deserialize<'de>, InputMmcs::MultiProof: Deserialize<'de>"
))]
pub struct InputOpenings<Val: Send + Sync + Clone, InputMmcs: Mmcs<Val>> {
    /// `opened_values[k][m]` is the LDE row of matrix `m` at the `k`-th queried position, in
    /// the ascending order of the bucket's deduplicated positions, which prover and verifier
    /// both derive from the transcript.
    pub opened_values: Vec<Vec<Vec<Val>>>,
    /// Compact multi-opening proof authenticating every row at once.
    pub opening_proof: InputMmcs::MultiProof,
}

/// One shared-LDE-domain group of a commitment: its own Merkle tree over the matrices whose
/// native heights the partition placed together.
///
/// Every matrix here is extended onto the same domain (sized to the group's tallest) and
/// committed one LDE row per leaf.
struct DomainGroup<Val: Send + Sync + Clone, InputMmcs: Mmcs<Val>> {
    data: InputMmcs::ProverData<RowMajorMatrix<Val>>,
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

/// One Merkle root per shared-domain group of a commitment, in descending LDE height.
///
/// The matrices of one `commit()` call are partitioned into groups of bounded height spread,
/// each extended onto its own shared domain (§7's same-domain requirement applies within a
/// group, not across the whole commitment) and committed in its own tree. A commitment whose
/// heights all fit one group therefore holds a single root.
///
/// The roots are wrapped rather than handed out as a bare `Vec` so a challenger can observe
/// the whole commitment in one call — which is what
/// `p3_uni_stark::StarkGenericConfig`'s `Challenger: CanObserve<Pcs::Commitment>` bound asks
/// for. Observing runs the root count first, so two commitments that split a given total of
/// roots differently cannot reach the same transcript state.
///
/// Dereferences to its roots, so reading them needs no unwrapping.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StirCommitment<C>(Vec<C>);

impl<C> Deref for StirCommitment<C> {
    type Target = [C];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// `StirCommitment<C>`'s `CanObserve` impl is written once per challenger backend rather than
/// as a single generic impl: a blanket `impl<Ch: CanObserve<F> + CanObserve<C>, F, C>
/// CanObserve<StirCommitment<C>> for Ch` would leave `F` unconstrained by `Ch`, which Rust
/// rejects. Each impl below observes the group count as a length prefix (so a challenger
/// cannot confuse two commitments with different group counts), then each root, in the same
/// order every backend uses.
///
/// `SerializingChallenger64` has the identical gap (no impl here) — left out under YAGNI, since
/// nothing in this crate or its dependents currently pairs `TwoAdicStirPcs` with it.
impl<F, P, C, const WIDTH: usize, const RATE: usize> CanObserve<StirCommitment<C>>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy + PrimeCharacteristicRing,
    P: CryptographicPermutation<[F; WIDTH]>,
    Self: CanObserve<C>,
{
    fn observe(&mut self, commitment: StirCommitment<C>) {
        <Self as CanObserve<F>>::observe(self, F::from_usize(commitment.0.len()));
        for root in commitment.0 {
            self.observe(root);
        }
    }
}

impl<F, Inner, C> CanObserve<StirCommitment<C>> for SerializingChallenger32<F, Inner>
where
    F: PrimeField32,
    Inner: CanObserve<u8>,
    Self: CanObserve<C>,
{
    fn observe(&mut self, commitment: StirCommitment<C>) {
        <Self as CanObserve<F>>::observe(self, F::from_usize(commitment.0.len()));
        for root in commitment.0 {
            self.observe(root);
        }
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

/// Views of the committed matrices in caller order.
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
        .map(|&(group_idx, idx)| per_group[group_idx][idx].as_view())
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
/// removes a whole STIR instance and its query-count floor. Three octaves is what that trade
/// was measured to be worth buying: the shortest member of a group spanning it pays an `8x`
/// blowup, in its DFT and in its full width in every one of the group's tree leaves, to save
/// one instance. Past it the commit side keeps doubling per octave while `Combine`'s return
/// does not, so wider spreads get their own domain.
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

    /// How wide a native-height spread this instance lets share one LDE domain.
    ///
    /// The two sides of a proof must agree on it, so both can check that they do rather than
    /// relying on having been constructed the same way.
    pub const fn max_log_height_spread(&self) -> usize {
        self.max_log_height_spread
    }

    /// Commit one tree per shared-domain group.
    ///
    /// `plan` assigns matrices to groups; `grouped[i]` is matrix `i`'s bit-reversed LDE,
    /// already extended onto the domain of its own group.
    fn commit_groups(
        &self,
        plan: &GroupPlan,
        grouped: Vec<RowMajorMatrix<Val>>,
        log_native_heights: &[usize],
    ) -> (
        StirCommitment<InputMmcs::Commitment>,
        StirProverData<Val, InputMmcs>,
    )
    where
        Val: Send + Sync + Clone,
        InputMmcs: Mmcs<Val, Commitment: Send> + Sync,
        InputMmcs::ProverData<RowMajorMatrix<Val>>: Send,
    {
        let input_mmcs = &self.input_mmcs;
        let num_groups = plan.log_lde_heights.len();
        let mut per_group: Vec<Vec<RowMajorMatrix<Val>>> = vec![Vec::new(); num_groups];
        let mut per_group_heights: Vec<Vec<usize>> = vec![Vec::new(); num_groups];
        let mut placement = Vec::with_capacity(grouped.len());

        for (matrix_idx, matrix) in grouped.into_iter().enumerate() {
            let group_idx = plan.group_of_matrix[matrix_idx];
            placement.push((group_idx, per_group[group_idx].len()));
            per_group[group_idx].push(matrix);
            per_group_heights[group_idx].push(log_native_heights[matrix_idx]);
        }

        // Groups share nothing — separate matrices, separate trees, separate prover data — so
        // they are built in parallel. Only the tallest group has enough rows to saturate the
        // pool on its own; the inner per-tree parallelism work-steals alongside this one.
        let (commitments, groups): (Vec<_>, Vec<_>) =
            izip!(per_group, per_group_heights, &plan.log_lde_heights)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(matrices, log_native_heights, &log_lde_height)| {
                    let (commitment, data) = input_mmcs.commit(matrices);
                    (
                        commitment,
                        DomainGroup {
                            data,
                            log_native_heights,
                            log_lde_height,
                        },
                    )
                })
                .unzip();

        (
            StirCommitment(commitments),
            StirProverData { groups, placement },
        )
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
    /// Growth stops at the first of three limits: the configured spread cap, which bounds the
    /// extra blowup a short matrix pays for sitting on a taller group's domain; `lowest`, the
    /// lowest height that could still join the band; and `Combine` feasibility, which is what
    /// makes an infeasible parameter set degrade into more STIR instances rather than failing.
    ///
    /// The `lowest` cap never moves an admissibility decision. A span of `s` octaves is
    /// admitted exactly when `s` is within every limit, and `s <= tallest - lowest` holds for
    /// any span that exists at all; capping only skips deriving — and caching — configs for
    /// bands no group could fill.
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
    fn combine_band_width(&self, tallest: usize, lowest: usize) -> usize {
        let log_stir_degree = self.log_stir_degree(tallest + self.stir.log_blowup);
        let max_width = self
            .max_log_height_spread
            .min(tallest)
            .min(tallest - lowest);

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
    /// groups. A group is admissible only when its whole span fits inside the band
    /// [`Self::combine_band_width`] admits below its own tallest, which is what keeps that
    /// probe conservative for whatever union of classes a bucket later pools.
    ///
    /// Among admissible partitions this takes the fewest groups — one STIR instance each, so
    /// the group count fixes the proof's shape and its query structure — and, among those, the
    /// one minimizing `Σ_g 2^(tallest of g)·|g|`: every member of a group is extended onto that
    /// group's shared domain, so a group costs its own height once per member, in both DFT
    /// work and Merkle leaf material. Filling each group greedily instead reaches the same
    /// group count but pulls short heights onto the tallest domain that will take them, which
    /// is the most expensive placement available to them. Both objectives read only the
    /// distinct heights, so the prover and the verifier derive the same partition.
    fn partition_native_heights(&self, descending: &[usize]) -> Vec<usize> {
        let Some(&lowest) = descending.last() else {
            return Vec::new();
        };
        let n = descending.len();

        let band_widths: Vec<usize> = descending
            .iter()
            .map(|&tallest| self.combine_band_width(tallest, lowest))
            .collect();

        // `best[j]` is the lexicographically smallest `(group count, cost)` covering
        // `descending[..j]`, and `start_of_last[j]` where the final group achieving it begins.
        // Singleton groups are always admissible, so every prefix is reachable.
        let mut best = vec![(usize::MAX, u128::MAX); n + 1];
        let mut start_of_last = vec![0usize; n + 1];
        best[0] = (0, 0);

        for j in 1..=n {
            for i in 0..j {
                if descending[i] - descending[j - 1] > band_widths[i] {
                    continue;
                }
                let (groups, cost) = best[i];
                let candidate = (
                    groups + 1,
                    cost + ((j - i) as u128) * (1u128 << descending[i]),
                );
                if candidate < best[j] {
                    best[j] = candidate;
                    start_of_last[j] = i;
                }
            }
        }

        let mut sizes = Vec::new();
        let mut end = n;
        while end > 0 {
            let start = start_of_last[end];
            sizes.push(end - start);
            end = start;
        }
        sizes.reverse();
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

/// One commitment's prover data alongside the opening points of each of its matrices.
type ProverDataWithPoints<'a, Val, InputMmcs, Challenge> =
    (&'a StirProverData<Val, InputMmcs>, Vec<Vec<Challenge>>);

/// Everything `open` settles before STIR runs.
struct PreparedOpen<Val, Challenge, StirMmcs, Challenger> {
    /// Claimed evaluations, already absorbed into the transcript.
    opened_values: OpenedValues<Challenge>,
    /// Distinct shared LDE heights across every commitment's groups, descending: one STIR
    /// instance ("bucket") each.
    bucket_log_heights: Vec<usize>,
    /// The derived config of each bucket's instance.
    stir_configs: Vec<Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>>>,
    /// Each bucket's initial codeword in natural order: its reduced opening, `Combine`d across
    /// native-height classes when the bucket holds more than one.
    initial_codewords: Vec<Vec<Challenge>>,
}

impl<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
    TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val, Error: Sync + Debug, Commitment: Send> + Sync,
    InputMmcs::ProverData<RowMajorMatrix<Val>>: Send,
    StirMmcs: Mmcs<Challenge>,
    Challenge: ExtensionField<Val> + TwoAdicField + BasedVectorSpace<Val>,
    Challenger: FieldChallenger<Val>
        + CanObserve<InputMmcs::Commitment>
        + CanObserve<StirMmcs::Commitment>
        + GrindingChallenger<Witness = Val>
        + CanSampleUniformBits<Val>
        + Clone,
{
    /// Evaluate every matrix at its opening points, absorb the claims, alpha-batch the
    /// quotients into one reduced opening per `(shared LDE height, native height)` class, and
    /// `Combine` each bucket's classes into the codeword STIR will prove.
    fn prepare_open(
        &self,
        commitment_data_with_opening_points: &[ProverDataWithPoints<
            '_,
            Val,
            InputMmcs,
            Challenge,
        >],
        challenger: &mut Challenger,
    ) -> PreparedOpen<Val, Challenge, StirMmcs, Challenger> {
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

        PreparedOpen {
            opened_values: all_opened_values,
            bucket_log_heights,
            stir_configs,
            initial_codewords,
        }
    }

    /// Run STIR on every bucket in lockstep, then open the input trees at one lane per
    /// round-0 query draw.
    ///
    /// STIR commits each bucket's initial codeword `f_0` itself, so its round-0 queries open
    /// that commitment's fibers. Each queried fiber is tied to the input commitments at one
    /// point: once the whole STIR transcript is settled, one uniform lane per round-0 draw is
    /// sampled and the input rows at exactly those positions are opened, and the verifier
    /// rebuilds the reduced opening from each single row and compares it with that lane of the
    /// fiber STIR authenticated.
    ///
    /// Soundness of the one-lane binding: `f_0` is committed before the round-0 fold challenge,
    /// so mutual correlated agreement makes the fibers on which the fold matches the codeword
    /// pinned by the OOD samples fibers on which `f_0` equals one codeword `c` on every lane. A
    /// draw at a uniform position `p` therefore passes only if `f_0(p) = c(p)` and, by the lane
    /// check, `ro(p) = c(p)`; a reduced opening `δ`-far from the code meets `c` on at most a
    /// `1 - δ` fraction of positions, so each draw catches it at the rate the round's query
    /// count was priced on, and the folding error is the term the round's folding grind already
    /// covers. The lane must be uniform and drawn after the commitment, and one lane per draw —
    /// not per distinct fiber — is what keeps the draws independent.
    fn prove_buckets(
        &self,
        prover_data: &[&StirProverData<Val, InputMmcs>],
        prepared: PreparedOpen<Val, Challenge, StirMmcs, Challenger>,
        challenger: &mut Challenger,
    ) -> (
        OpenedValues<Challenge>,
        StirPcsProof<Val, Challenge, InputMmcs, StirMmcs>,
    ) {
        let PreparedOpen {
            opened_values,
            bucket_log_heights,
            stir_configs,
            initial_codewords,
        } = prepared;
        let stir_config_refs: Vec<&StirConfig<Val, Challenge, StirMmcs, Challenger>> =
            stir_configs.iter().map(AsRef::as_ref).collect();

        let bucket_results = prove_stir_multi_from_codewords(
            &stir_config_refs,
            initial_codewords,
            &self.dft,
            challenger,
        );

        let bucket_proofs = bucket_log_heights
            .iter()
            .zip(&stir_configs)
            .zip(bucket_results)
            .map(|((&log_h, stir_config), (stir_proof, first_round))| {
                let log_arity0 = stir_config.log_starting_folding_factor;
                let lanes = sample_lanes::<Val, _>(challenger, first_round.draws.len(), log_arity0);
                let positions = query_positions(&first_round.draws, &lanes, log_h, log_arity0);
                // The LDE is stored bit-reversed, so natural position `p` is row `rev(p)`.
                let row_indices: Vec<usize> = positions
                    .iter()
                    .map(|&p| reverse_bits_len(p, log_h))
                    .collect();

                let input_openings: Vec<Option<InputOpenings<Val, InputMmcs>>> = prover_data
                    .iter()
                    .map(|data| {
                        // Each group has its own tree on its own domain, so a bucket reads
                        // exactly the group committed at its LDE height.
                        let group = data.group_at(log_h)?;
                        let (opened_values, opening_proof) =
                            self.input_mmcs.open_multi_batch(&row_indices, &group.data);
                        Some(InputOpenings {
                            opened_values,
                            opening_proof,
                        })
                    })
                    .collect();

                (stir_proof, input_openings)
            })
            .collect();

        (opened_values, bucket_proofs)
    }
}

impl<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val, Error: Sync + Debug, Commitment: Send> + Sync,
    InputMmcs::ProverData<RowMajorMatrix<Val>>: Send,
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
    type Commitment = StirCommitment<InputMmcs::Commitment>;
    type ProverData = StirProverData<Val, InputMmcs>;
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixCow<'a, Val>>;
    /// See `StirPcsProof`.
    type Proof = StirPcsProof<Val, Challenge, InputMmcs, StirMmcs>;
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
                self.dft
                    .coset_lde_batch(evals, extra_bits, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect();
        self.commit_groups(&plan, grouped, &log_native_heights)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let (group_idx, idx_in_group) = prover_data.placement[idx];
        let group = &prover_data.groups[group_idx];
        let lde = self.input_mmcs.get_matrices(&group.data)[idx_in_group].as_view();
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
            .bit_reverse_rows()
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

        let grouped: Vec<_> = ldes
            .into_iter()
            .zip(&log_native_heights)
            .zip(&plan.group_of_matrix)
            .map(|((lde, &log_native_height), &group_idx)| {
                let log_lde_height = plan.log_lde_heights[group_idx];
                if lde.height() == 1usize << log_lde_height {
                    lde
                } else {
                    // Recovering the polynomial and evaluating it on the wider coset is a
                    // forward transform of its coefficients, zero-padded to the target size.
                    // A second `lde` would instead read those coefficients back as evaluations
                    // on a subgroup, and extend a different polynomial.
                    let natural_lde = lde.bit_reverse_rows().to_row_major_matrix();
                    let mut coeffs = self.dft.coset_idft_batch(natural_lde, Val::GENERATOR);
                    let width = coeffs.width();
                    coeffs
                        .values
                        .truncate((1usize << log_native_height) * width);
                    coeffs
                        .values
                        .resize((1usize << log_lde_height) * width, Val::ZERO);
                    self.dft
                        .coset_dft_batch(coeffs, Val::GENERATOR)
                        .bit_reverse_rows()
                        .to_row_major_matrix()
                }
            })
            .collect();
        self.commit_groups(&plan, grouped, &log_native_heights)
    }

    #[instrument(name = "STIR PCS open", skip_all)]
    fn open(
        &self,
        commitment_data_with_opening_points: Vec<(&Self::ProverData, Vec<Vec<Challenge>>)>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        let prepared = self.prepare_open(&commitment_data_with_opening_points, challenger);
        let prover_data: Vec<&Self::ProverData> = commitment_data_with_opening_points
            .iter()
            .map(|(data, _)| *data)
            .collect();
        self.prove_buckets(&prover_data, prepared, challenger)
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
        for (commit_idx, ((commitment, _), plan)) in commitments_with_opening_points
            .iter()
            .zip(&plans)
            .enumerate()
        {
            if commitment.len() != plan.log_lde_heights.len() {
                return Err(ProofShapeError::CommitmentRootCount {
                    commitment: commit_idx,
                    expected: plan.log_lde_heights.len(),
                    got: commitment.len(),
                }
                .into());
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
            return Err(ProofShapeError::BucketCount {
                expected: bucket_log_heights.len(),
                got: proof.len(),
            }
            .into());
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
        // computing them once here (rather than inside the per-bucket, per-queried-position
        // loop below) turns an `O(n_q)` recomputation per point into `O(1)`. Keyed like the
        // prover's `reduced_openings`, by `(log_shared_lde_height,
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
        for (&log_height, (_, input_openings)) in bucket_log_heights.iter().zip(proof) {
            if input_openings.len() != commitments_with_opening_points.len() {
                return Err(ProofShapeError::InputOpeningCount {
                    log_height,
                    expected: commitments_with_opening_points.len(),
                    got: input_openings.len(),
                }
                .into());
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

        // STIR runs first: it absorbs each bucket's initial-oracle commitment, checks every
        // round, and hands back the round-0 fibers it authenticated against that commitment
        // together with the draws that selected them.
        let outputs = verify_stir_multi_inner(
            &stir_config_refs,
            &stir_proofs,
            challenger,
            None::<Vec<NoExternalFibers<Challenge, StirMmcs::Error, InputMmcs::Error>>>,
        )?;

        for (bucket, (&log_h, output)) in bucket_log_heights.iter().zip(&outputs).enumerate() {
            let stir_config = &stir_configs[bucket];
            let input_openings = &proof[bucket].1;
            let combine_info = &bucket_combine[bucket];
            let native_heights = &bucket_native_heights[bucket];
            let group_indices = &bucket_group_indices[bucket];

            let bucket_height = 1usize << log_h;
            let log_arity0 = stir_config.log_starting_folding_factor;
            let fold_height0 = bucket_height >> log_arity0;
            let domain_gen = Val::two_adic_generator(log_h);

            // One lane per round-0 draw, sampled only now that every STIR message — the
            // initial-oracle commitment above all — is in the transcript.
            let lanes =
                sample_lanes::<Val, _>(challenger, output.first_round_draws.len(), log_arity0);
            let positions = query_positions(&output.first_round_draws, &lanes, log_h, log_arity0);
            let n_q = positions.len();
            let row_indices: Vec<usize> = positions
                .iter()
                .map(|&p| reverse_bits_len(p, log_h))
                .collect();
            // Coset point of each queried position: `GENERATOR * g^p`.
            let query_points: Vec<Val> = positions
                .iter()
                .map(|&p| Val::GENERATOR * domain_gen.exp_u64(p as u64))
                .collect();

            // Distinct opening points among matrices active at this bucket. Matrices
            // typically share opening points (e.g. one STARK's `zeta`), so this list is
            // usually far shorter than the matrix count.
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

            // Every queried position needs `1 / (point - x_p)` for each distinct point in
            // `bucket_points`; all of them are inverted in one batch.
            let denom_diffs: Vec<Challenge> = query_points
                .iter()
                .flat_map(|&x| {
                    let x = Challenge::from(x);
                    bucket_points.iter().map(move |&point| point - x)
                })
                .collect();

            // Invariant: no opening point sits on a queried position.
            //   z == x  =>  z - x == 0
            //           =>  the quotient (f(z) - f(x)) / (z - x) is undefined
            //           =>  batch_multiplicative_inverse panics
            if let Some(slot) = denom_diffs.iter().position(|d| d.is_zero()) {
                let (commitment, matrix, point) = locate_opening_point(
                    &commitments_with_opening_points,
                    &bucket_points[slot % n_bp],
                );
                return Err(StirError::OpeningPointMatchesQueryPoint {
                    commitment,
                    matrix,
                    point,
                });
            }
            let inv_denoms = batch_multiplicative_inverse(&denom_diffs);

            // One accumulator per native-height class present in this bucket, indexed as
            // `native_heights` is (descending); merged after the accumulation loop.
            let mut expected_ro_by_class: Vec<Vec<Challenge>> =
                vec![Challenge::zero_vec(n_q); native_heights.len()];

            // A commitment feeds this bucket through at most one of its groups: the one
            // committed on this domain. Its other groups live in other trees at other heights
            // and belong to other buckets.
            for (commit_idx, ((commitment, domain_claims), per_commit_opening)) in
                commitments_with_opening_points
                    .iter()
                    .zip(input_openings.iter())
                    .enumerate()
            {
                let group_idx = group_indices[commit_idx];

                let Some(opening) = per_commit_opening else {
                    if group_idx.is_some() {
                        return Err(ProofShapeError::MissingInputOpening {
                            log_height: log_h,
                            commitment: commit_idx,
                        }
                        .into());
                    }
                    continue;
                };
                let Some(group_idx) = group_idx else {
                    return Err(ProofShapeError::UnexpectedInputOpening {
                        log_height: log_h,
                        commitment: commit_idx,
                    }
                    .into());
                };

                // The commitment's matrices on this domain, in the order they were committed
                // to its tree: caller order, filtered to this group.
                let group_mats: Vec<usize> = matrix_lde_heights[commit_idx]
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &h)| (h == log_h).then_some(idx))
                    .collect();

                // Pin each matrix's width to its claimed evaluation count, never to the
                // proof. Every matrix has at least one claim — the up-front check in `verify`
                // rejects otherwise before the transcript is touched.
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

                // One LDE row per leaf.
                let dimensions: Vec<p3_matrix::Dimensions> = mat_widths
                    .iter()
                    .map(|&width| p3_matrix::Dimensions {
                        height: bucket_height,
                        width,
                    })
                    .collect();

                // SHAPE CHECK: opened-row count is determined entirely by public input.
                if opening.opened_values.len() != n_q {
                    return Err(ProofShapeError::InputOpenedRowCount {
                        log_height: log_h,
                        commitment: commit_idx,
                        expected: n_q,
                        got: opening.opened_values.len(),
                    }
                    .into());
                }

                self.input_mmcs
                    .verify_multi_batch(
                        &commitment[group_idx],
                        &dimensions,
                        &row_indices,
                        &opening.opened_values,
                        &opening.opening_proof,
                    )
                    .map_err(StirError::InputError)?;

                for (q, row_vals_by_mat) in opening.opened_values.iter().enumerate() {
                    for (mat_idx, point_slots) in mat_point_slots.iter().enumerate() {
                        // `mat_idx` indexes this group's tree, and therefore the opened rows;
                        // `point_data` is keyed by the commitment's full claim order, which
                        // `group_mats` maps back to.
                        let claim_idx = group_mats[mat_idx];
                        let p_x: Challenge = row_vals_by_mat[mat_idx]
                            .iter()
                            .zip(alpha_powers.iter())
                            .map(|(&v, &ap)| ap * v)
                            .sum();

                        let ro_class = &mut expected_ro_by_class[mat_class_indices[mat_idx]];

                        for (point_idx, &bp_idx) in point_slots.iter().enumerate() {
                            let (alpha_pow_offset, y_combined) =
                                point_data[commit_idx][claim_idx][point_idx];
                            let inv_denom = inv_denoms[q * n_bp + bp_idx];

                            ro_class[q] += alpha_pow_offset * (p_x - y_combined) * inv_denom;
                        }
                    }
                }
            }

            // Merge the per-class accumulators into the bucket's expected initial codeword at
            // the queried positions, mirroring the prover's `combine_on_coset` pointwise.
            //
            // Both class counts come from `native_heights`, never from the proof. Still
            // reported rather than asserted: a verifier must not panic.
            let expected: Vec<Challenge> = match combine_info {
                None => {
                    let got = expected_ro_by_class.len();
                    let mut classes = expected_ro_by_class.into_iter();
                    match (classes.next(), classes.next()) {
                        (Some(only), None) => only,
                        _ => {
                            return Err(ProofShapeError::HeightClassCount {
                                log_height: log_h,
                                expected: 1,
                                got,
                            }
                            .into());
                        }
                    }
                }
                Some((r_comb, coeffs_by_height)) => {
                    debug_assert_eq!(expected_ro_by_class.len(), coeffs_by_height.len());
                    let mut combined = Challenge::zero_vec(n_q);
                    for (&log_native_h, ro_class) in
                        native_heights.iter().zip(&expected_ro_by_class)
                    {
                        let &(r_i, gap) = coeffs_by_height.get(&log_native_h).ok_or(
                            ProofShapeError::MissingCombineCoefficient {
                                log_height: log_h,
                                log_native_height: log_native_h,
                            },
                        )?;
                        for (acc, (&ro, &x)) in
                            combined.iter_mut().zip(ro_class.iter().zip(&query_points))
                        {
                            *acc +=
                                r_i * eval_degree_correction(ro, Challenge::from(x), *r_comb, gap);
                        }
                    }
                    combined
                }
            };

            // The committed initial oracle must agree with the reduced opening at every
            // sampled lane. STIR already authenticated these fibers against its commitment.
            for (&p, &value) in positions.iter().zip(&expected) {
                let (j, lane) = split_position(p, fold_height0);
                let fiber_idx = output
                    .first_round_indices
                    .binary_search(&j)
                    .expect("every draw is among the verifier's own unique round-0 indices");
                if output.first_round_fiber_evals[fiber_idx].get(lane).copied() != Some(value) {
                    return Err(StirError::InitialOracleMismatch {
                        log_height: log_h,
                        position: p,
                    });
                }
            }
        }

        Ok(())
    }
}

/// One entry per distinct shared LDE height across every commitment's groups (descending). A
/// commitment contributes to one entry per group it holds. Each entry holds:
/// - the STIR IOP proof for that bucket, whose initial oracle is the bucket's reduced
///   opening, committed by STIR itself;
/// - `input_openings[commit_idx]`: one shared multi-opening proof for that commitment's rows
///   at the bucket's queried positions, `None` if the commitment has no group at this
///   bucket's LDE height.
type StirPcsProof<Val, Challenge, InputMmcs, StirMmcs> = Vec<(
    StirProof<Challenge, StirMmcs, Val>,
    Vec<Option<InputOpenings<Val, InputMmcs>>>,
)>;

/// Source of an external initial oracle's fibers, for the STIR verifier's `None`: this PCS
/// has STIR commit the initial oracle itself, so no such source ever exists.
type NoExternalFibers<EF, E, IE> = fn(&[usize]) -> Result<Vec<Vec<EF>>, StirError<E, IE>>;

/// One uniformly sampled fiber lane per round-0 query draw, in draw order.
fn sample_lanes<Val, Challenger>(
    challenger: &mut Challenger,
    num_draws: usize,
    log_arity0: usize,
) -> Vec<usize>
where
    Val: TwoAdicField,
    Challenger: CanSampleUniformBits<Val>,
{
    (0..num_draws)
        .map(|_| {
            challenger
                .sample_uniform_bits::<true>(log_arity0)
                .expect("RESAMPLE = true: rejection loops internally, never errors")
        })
        .collect()
}

/// The natural-order LDE positions `j + lane * 2^(log_h - log_arity0)` of every
/// `(draw, lane)` pair, ascending and deduplicated.
///
/// Fiber `j` of the round-0 fold domain is the coset `{GENERATOR * g^(j + l * 2^(log_h -
/// log_arity0))}` for `l < 2^log_arity0`, so a lane picks one point of it.
fn query_positions(
    draws: &[usize],
    lanes: &[usize],
    log_h: usize,
    log_arity0: usize,
) -> Vec<usize> {
    debug_assert_eq!(draws.len(), lanes.len());
    let mut positions: Vec<usize> = draws
        .iter()
        .zip(lanes)
        .map(|(&j, &lane)| j + (lane << (log_h - log_arity0)))
        .collect();
    positions.sort_unstable();
    positions.dedup();
    positions
}

/// A natural-order position's `(fiber index, lane)`, inverting [`query_positions`].
const fn split_position(position: usize, fold_height0: usize) -> (usize, usize) {
    (position % fold_height0, position / fold_height0)
}

/// One commitment's claims: per matrix, its domain and its `(point, values)` pairs.
type CommitmentClaims<C, D, EF> = (C, Vec<(D, Vec<(EF, Vec<EF>)>)>);

/// Locate the claim that contributed `point`, as `(commitment, matrix, point)` indices.
///
/// Only reached on the error path, where a linear scan costs nothing.
fn locate_opening_point<C, D, EF: PartialEq>(
    commitments_with_opening_points: &[CommitmentClaims<C, D, EF>],
    point: &EF,
) -> (usize, usize, usize) {
    commitments_with_opening_points
        .iter()
        .enumerate()
        .flat_map(|(commitment, (_, domain_claims))| {
            domain_claims
                .iter()
                .enumerate()
                .flat_map(move |(matrix, (_, point_claims))| {
                    point_claims
                        .iter()
                        .enumerate()
                        .map(move |(slot, (claimed, _))| (commitment, matrix, slot, claimed))
                })
        })
        .find_map(|(commitment, matrix, slot, claimed)| {
            (claimed == point).then_some((commitment, matrix, slot))
        })
        .expect("every bucket point comes from a claim")
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
    use proptest::prelude::*;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::prover::codeword_from_coeffs;
    use crate::verifier::verify_stir_multi;

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

        // Spread 2 rules out one group over all three, so two is the fewest available. Of the
        // two-group splits it does admit, `{8} | {6, 4}` keeps the 2^6 matrix off the 2^9
        // domain and is the cheaper one to extend.
        let plan = test_pcs_with(2, cb, 32).plan_groups(&heights);
        assert_eq!(plan.log_lde_heights, vec![9, 7]);
        assert_eq!(plan.group_of_matrix, vec![0, 1, 1]);

        // Spread 1 admits none of these pairs.
        let plan = test_pcs_with(1, cb, 32).plan_groups(&heights);
        assert_eq!(plan.log_lde_heights, vec![9, 7, 5]);
        assert_eq!(plan.group_of_matrix, vec![0, 1, 2]);
    }

    #[test]
    fn a_group_spans_the_cheapest_admissible_run_not_the_widest() {
        // `[20, 17, 16, 15]` needs two groups either way — the full span is five octaves, past
        // the cap — but which two matters. Filling the tall group first puts the 2^17 matrix
        // on the 2^21 domain: a 16x blowup, and a `Combine` over two classes in the tall
        // bucket. Leaving it with the short heights costs it 2x instead, at the same group
        // count and with no `Combine` in the tall bucket at all.
        let pcs = test_pcs_with(
            DEFAULT_MAX_LOG_HEIGHT_SPREAD,
            SecurityAssumption::CapacityBound,
            16,
        );

        let plan = pcs.plan_groups(&[20, 17, 16, 15]);
        assert_eq!(plan.log_lde_heights, vec![21, 18]);
        assert_eq!(plan.group_of_matrix, vec![0, 1, 1, 1]);

        // Shapes lying wholly inside one band, or wholly outside it, have nothing to
        // redistribute: the widest run is also the cheapest once the group count is fixed.
        assert_eq!(pcs.plan_groups(&[20, 18, 17]).log_lde_heights, vec![21]);
        assert_eq!(
            pcs.plan_groups(&[20, 12, 12, 12, 12]).log_lde_heights,
            vec![21, 13]
        );
        assert_eq!(pcs.plan_groups(&[20, 10]).log_lde_heights, vec![21, 11]);
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
        assert_eq!(plan.log_lde_heights, vec![9, 7]);
        assert_eq!(plan.group_of_matrix, vec![1, 0, 1, 1]);
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

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// The layout has to be a pure function of the multiset of native heights and this
        /// PCS's parameters: that is the assumption the verifier reconstructs it under, so a
        /// failure here is a verifier disagreeing with the prover for a reason no shape check
        /// in the proof could explain. The interesting inputs are multisets — duplicates,
        /// caller order, and heights sitting on a band edge — so this is sampled rather than
        /// enumerated.
        #[test]
        fn plan_groups_partitions_the_height_multiset(
            heights in prop::collection::vec(2usize..=12, 1..8),
            max_log_height_spread in 0usize..=8,
        ) {
            let pcs = test_pcs_with(
                max_log_height_spread,
                SecurityAssumption::CapacityBound,
                32,
            );
            let plan = pcs.plan_groups(&heights);

            prop_assert_eq!(plan.group_of_matrix.len(), heights.len());
            for pair in plan.log_lde_heights.windows(2) {
                prop_assert!(pair[0] > pair[1]);
            }

            // Distinct heights of each group, read back off the assignment.
            let mut members: Vec<Vec<usize>> = vec![Vec::new(); plan.log_lde_heights.len()];
            for (&h, &g) in heights.iter().zip(&plan.group_of_matrix) {
                if !members[g].contains(&h) {
                    members[g].push(h);
                }
            }
            let mut distinct = heights.clone();
            distinct.sort_unstable();
            distinct.dedup();
            prop_assert_eq!(members.iter().map(Vec::len).sum::<usize>(), distinct.len());

            for (group, &log_lde_h) in members.iter_mut().zip(&plan.log_lde_heights) {
                prop_assert!(!group.is_empty());
                group.sort_unstable_by(|a, b| b.cmp(a));
                let (tallest, lowest) = (group[0], group[group.len() - 1]);
                // Each group sits on its own tallest member's domain, ...
                prop_assert_eq!(log_lde_h, tallest + pcs.stir.log_blowup);
                // ... and stays inside the band that member admits, which is what keeps the
                // band probe conservative for whatever union of classes a bucket pools.
                prop_assert!(tallest - lowest <= pcs.combine_band_width(tallest, lowest));
            }

            // Only the multiset decides: permuting the input moves matrices between slots but
            // not between heights, and repeating a height adds no class.
            let permuted: Vec<usize> = heights.iter().rev().copied().collect();
            let permuted_plan = pcs.plan_groups(&permuted);
            let permuted_back: Vec<usize> =
                permuted_plan.group_of_matrix.iter().rev().copied().collect();
            prop_assert_eq!(&permuted_plan.log_lde_heights, &plan.log_lde_heights);
            prop_assert_eq!(&permuted_back, &plan.group_of_matrix);

            let duplicated: Vec<usize> = heights.iter().chain(heights.iter()).copied().collect();
            let duplicated_plan = pcs.plan_groups(&duplicated);
            prop_assert_eq!(&duplicated_plan.log_lde_heights, &plan.log_lde_heights);
            prop_assert_eq!(
                &duplicated_plan.group_of_matrix[..heights.len()],
                &plan.group_of_matrix[..]
            );
        }
    }

    /// STIR parameters over the test types, at the given soundness knobs.
    fn test_params(
        soundness_type: SecurityAssumption,
        security_level: usize,
        log_blowup: usize,
        max_pow_bits: usize,
    ) -> StirParameters<TestStirMmcs> {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
        let perm = TestPerm::new_from_rng_128(&mut rng);
        let val_mmcs = TestValMmcs::new(TestHash::new(perm.clone()), TestCompress::new(perm), 0);
        StirParameters {
            log_blowup,
            log_folding_factor: 2,
            log_starting_folding_factor: 2,
            soundness_type,
            security_level,
            max_pow_bits,
            mmcs: TestStirMmcs::new(val_mmcs),
        }
    }

    #[test]
    fn band_feasibility_implies_subset_feasibility() {
        // `combine_band_width` probes the *whole* band `[tallest - w, tallest]` and then lets
        // any subset of it containing `tallest` form a group, on the argument that the full
        // band maximizes Lemma 4.13's `ell` over every subset that could form. That argument
        // is what keeps the probe conservative, and it is load bearing: if it stopped holding,
        // the probe would call a width feasible, the bucket's actual class set would be a
        // strict subset whose config comes back `Err`, and `open` would panic on the prover at
        // exactly a parameter set the fallback exists to rescue. The quantifier is "every
        // subset", so this checks every subset rather than sampling them.
        for soundness_type in [
            SecurityAssumption::CapacityBound,
            SecurityAssumption::JohnsonBound,
        ] {
            for security_level in [32usize, 64, 80] {
                for log_blowup in [1usize, 2] {
                    for max_pow_bits in [0usize, 16] {
                        let params =
                            test_params(soundness_type, security_level, log_blowup, max_pow_bits);
                        for log_d_star in [6usize, 10, 14] {
                            for w in 1..=6.min(log_d_star) {
                                let band: Vec<usize> =
                                    (0..=w).map(|i| log_d_star - i).collect::<Vec<_>>();
                                let (n, ell) = TestPcs::combine_key(&band)
                                    .expect("a band of width >= 1 holds two classes");
                                if TestConfig::try_new_with_combine(
                                    log_d_star,
                                    params.clone(),
                                    n,
                                    ell,
                                )
                                .is_err()
                                {
                                    // The band itself is infeasible, so it implies nothing.
                                    continue;
                                }

                                for mask in 0u32..(1 << w) {
                                    let mut subset = vec![log_d_star];
                                    subset.extend(
                                        (0..w)
                                            .filter(|i| mask & (1 << i) != 0)
                                            .map(|i| log_d_star - 1 - i),
                                    );
                                    let Some((sub_n, sub_ell)) = TestPcs::combine_key(&subset)
                                    else {
                                        continue;
                                    };
                                    assert!(
                                        TestConfig::try_new_with_combine(
                                            log_d_star,
                                            params.clone(),
                                            sub_n,
                                            sub_ell,
                                        )
                                        .is_ok(),
                                        "band [{}..{log_d_star}] configures but subset \
                                         {subset:?} does not, at {soundness_type:?} \
                                         security_level={security_level} \
                                         log_blowup={log_blowup} max_pow_bits={max_pow_bits}",
                                        log_d_star - w,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
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

    /// A prover that commits a perfectly low-degree codeword which is *not* the reduced
    /// opening passes every STIR round — the lane checks are the only thing standing between
    /// such a prover and acceptance. Adding a random low-degree codeword to `f_0` keeps STIR
    /// happy and must be caught by the first lane the verifier compares.
    #[test]
    fn verify_rejects_a_low_degree_initial_oracle_that_is_not_the_reduced_opening() {
        let pcs = test_pcs_with(
            DEFAULT_MAX_LOG_HEIGHT_SPREAD,
            SecurityAssumption::CapacityBound,
            32,
        );
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        let perm = TestPerm::new_from_rng_128(&mut rng);
        let mut challenger = TestChallenger::new(perm);

        let log_h = 8;
        let log_lde = log_h + pcs.stir.log_blowup;
        let domain =
            <TestPcs as Pcs<EF, TestChallenger>>::natural_domain_for_degree(&pcs, 1 << log_h);
        let mat = RowMajorMatrix::<TestVal>::rand(&mut rng, 1 << log_h, 4);
        let (commit, data) =
            <TestPcs as Pcs<EF, TestChallenger>>::commit(&pcs, vec![(domain, mat)]);
        challenger.observe(commit.clone());
        let zeta: EF = challenger.sample_algebra_element();

        let mut p_ch = challenger.clone();
        let mut prepared = pcs.prepare_open(&[(&data, vec![vec![zeta]])], &mut p_ch);
        assert_eq!(prepared.initial_codewords.len(), 1);

        // A random polynomial of the same degree bound, evaluated on the same coset in the
        // same natural order STIR reads `initial_codewords` in.
        let mut coeffs: Vec<EF> = (0..1usize << log_h).map(|_| rng.random()).collect();
        coeffs.resize(1 << log_lde, EF::ZERO);
        let low_degree = codeword_from_coeffs(&pcs.dft, coeffs, TestVal::GENERATOR, log_lde);
        for (value, extra) in prepared.initial_codewords[0].iter_mut().zip(low_degree) {
            *value += extra;
        }

        let (opened_values, proof) = pcs.prove_buckets(&[&data], prepared, &mut p_ch);

        let mut v_ch = challenger;
        let claims = vec![(
            commit,
            vec![(domain, vec![(zeta, opened_values[0][0][0].clone())])],
        )];
        let err = <TestPcs as Pcs<EF, TestChallenger>>::verify(&pcs, claims, &proof, &mut v_ch)
            .expect_err("a low-degree oracle that is not the reduced opening must be rejected");
        assert!(
            matches!(err, StirError::InitialOracleMismatch { log_height, .. } if log_height == log_lde),
            "expected InitialOracleMismatch, got {err:?}"
        );
    }

    /// A bucket with no intermediate rounds reads its initial oracle from the *final* round,
    /// at the same arity, so the lane check has to bind there too. Same corruption as above,
    /// on the schedule where the fold-domain/lane split has no intermediate round to hide in.
    #[test]
    fn verify_rejects_a_low_degree_initial_oracle_at_a_zero_round_bucket() {
        let pcs = test_pcs_with(
            DEFAULT_MAX_LOG_HEIGHT_SPREAD,
            SecurityAssumption::CapacityBound,
            16,
        );
        let mut rng = rand::rngs::SmallRng::seed_from_u64(13);
        let perm = TestPerm::new_from_rng_128(&mut rng);
        let mut challenger = TestChallenger::new(perm);

        // `log_stir_degree == log_folding_factor`, so STIR schedules no intermediate round.
        let log_h = 2;
        let log_lde = log_h + pcs.stir.log_blowup;
        let domain =
            <TestPcs as Pcs<EF, TestChallenger>>::natural_domain_for_degree(&pcs, 1 << log_h);
        let mat = RowMajorMatrix::<TestVal>::rand(&mut rng, 1 << log_h, 4);
        let (commit, data) =
            <TestPcs as Pcs<EF, TestChallenger>>::commit(&pcs, vec![(domain, mat)]);
        challenger.observe(commit.clone());
        let zeta: EF = challenger.sample_algebra_element();

        let mut p_ch = challenger.clone();
        let mut prepared = pcs.prepare_open(&[(&data, vec![vec![zeta]])], &mut p_ch);
        assert_eq!(
            prepared.stir_configs[0].num_rounds(),
            0,
            "this test exists to cover the zero-intermediate-round schedule"
        );

        let mut coeffs: Vec<EF> = (0..1usize << log_h).map(|_| rng.random()).collect();
        coeffs.resize(1 << log_lde, EF::ZERO);
        let low_degree = codeword_from_coeffs(&pcs.dft, coeffs, TestVal::GENERATOR, log_lde);
        for (value, extra) in prepared.initial_codewords[0].iter_mut().zip(low_degree) {
            *value += extra;
        }

        let (opened_values, proof) = pcs.prove_buckets(&[&data], prepared, &mut p_ch);

        let mut v_ch = challenger;
        let claims = vec![(
            commit,
            vec![(domain, vec![(zeta, opened_values[0][0][0].clone())])],
        )];
        let err = <TestPcs as Pcs<EF, TestChallenger>>::verify(&pcs, claims, &proof, &mut v_ch)
            .expect_err("a low-degree oracle that is not the reduced opening must be rejected");
        assert!(
            matches!(err, StirError::InitialOracleMismatch { log_height, .. } if log_height == log_lde),
            "expected InitialOracleMismatch, got {err:?}"
        );
    }

    /// One lane per round-0 *draw*, in draw order — never one per distinct fiber.
    ///
    /// Sampling per unique fiber would leave a repeated fiber contributing no fresh
    /// randomness, and the per-draw product the round's query count is priced on would no
    /// longer hold. The two counts only differ when a fiber repeats, so this runs a
    /// zero-intermediate-round bucket whose fold domain holds two indices and repeats are
    /// forced; the transcript state after the draws is what tells the two apart.
    #[test]
    fn lanes_are_sampled_once_per_round_zero_draw() {
        let pcs = test_pcs_with(
            DEFAULT_MAX_LOG_HEIGHT_SPREAD,
            SecurityAssumption::CapacityBound,
            16,
        );
        let mut rng = rand::rngs::SmallRng::seed_from_u64(21);
        let perm = TestPerm::new_from_rng_128(&mut rng);
        let mut base = TestChallenger::new(perm);

        let log_h = 2;
        let domain =
            <TestPcs as Pcs<EF, TestChallenger>>::natural_domain_for_degree(&pcs, 1 << log_h);
        let mat = RowMajorMatrix::<TestVal>::rand(&mut rng, 1 << log_h, 2);
        let (commit, data) =
            <TestPcs as Pcs<EF, TestChallenger>>::commit(&pcs, vec![(domain, mat)]);
        base.observe(commit);
        let zeta: EF = base.sample_algebra_element();

        // The whole prover side, lanes included.
        let mut ch_full = base.clone();
        let prepared = pcs.prepare_open(&[(&data, vec![vec![zeta]])], &mut ch_full);
        let log_arity0 = prepared.stir_configs[0].log_starting_folding_factor;
        let final_queries = prepared.stir_configs[0].final_queries;
        let _ = pcs.prove_buckets(&[&data], prepared, &mut ch_full);

        // The same transcript, stopped right after STIR so the lane draws can be replayed by
        // hand at both counts.
        let mut ch_draws = base;
        let prepared = pcs.prepare_open(&[(&data, vec![vec![zeta]])], &mut ch_draws);
        let PreparedOpen {
            stir_configs,
            initial_codewords,
            ..
        } = prepared;
        let configs: Vec<&TestConfig> = stir_configs.iter().map(AsRef::as_ref).collect();
        let results =
            prove_stir_multi_from_codewords(&configs, initial_codewords, &pcs.dft, &mut ch_draws);

        let draws = &results[0].1.draws;
        let unique = &results[0].1.unique_sorted;
        assert_eq!(
            draws.len(),
            final_queries,
            "a zero-round bucket draws its initial-oracle queries in the final round"
        );
        assert!(
            unique.len() < draws.len(),
            "the fold domain must be small enough that a fiber repeats, or the two lane \
             counts are indistinguishable"
        );

        let mut ch_per_unique = ch_draws.clone();
        let lanes = sample_lanes::<TestVal, _>(&mut ch_draws, draws.len(), log_arity0);
        assert_eq!(lanes.len(), draws.len());
        let _ = sample_lanes::<TestVal, _>(&mut ch_per_unique, unique.len(), log_arity0);

        let after_full: EF = ch_full.sample_algebra_element();
        let after_per_draw: EF = ch_draws.sample_algebra_element();
        let after_per_unique: EF = ch_per_unique.sample_algebra_element();
        assert_eq!(
            after_full, after_per_draw,
            "the implementation must draw one lane per draw"
        );
        assert_ne!(
            after_full, after_per_unique,
            "one lane per unique fiber would leave a different transcript, so this comparison \
             is what makes the assertion above meaningful"
        );

        // Two draws of one fiber with different lanes are two distinct positions and both get
        // opened; with the same lane they collapse to one opened row, but two lanes were
        // still drawn. `log_h + 1` is the LDE height, so the fold domain holds two indices.
        assert_eq!(
            query_positions(&[1, 1], &[0, 1], log_h + 1, log_arity0),
            [1, 3]
        );
        assert_eq!(
            query_positions(&[1, 1], &[1, 1], log_h + 1, log_arity0),
            [3]
        );
    }

    /// The lane the PCS compares must be the codeword position STIR actually authenticated.
    ///
    /// `query_positions` maps `(fiber j, lane l)` to `p = j + l * 2^(log_h - log_arity0)`.
    /// That is only right if it agrees with the layout the prover's fiber matrix commits and
    /// with the subgroup point the verifier folds at, so it is pinned here against a real
    /// STIR instance's own round-0 fibers rather than against itself.
    #[test]
    fn fiber_lanes_index_the_committed_codeword_positions() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(5);
        let perm = TestPerm::new_from_rng_128(&mut rng);
        let dft = Radix2DitParallel::<TestVal>::default();
        let params = test_params(SecurityAssumption::CapacityBound, 32, 1, 0);
        let config = TestConfig::try_new(6, params).expect("feasible shape");
        let log_h = config.log_starting_domain_size();
        let log_arity0 = config.log_starting_folding_factor;
        let fold_height0 = (1usize << log_h) >> log_arity0;

        let mut coeffs: Vec<EF> = (0..1usize << 6).map(|_| rng.random()).collect();
        coeffs.resize(1 << log_h, EF::ZERO);
        let codeword = codeword_from_coeffs(&dft, coeffs, TestVal::GENERATOR, log_h);

        let mut p_ch = TestChallenger::new(perm.clone());
        let results =
            prove_stir_multi_from_codewords(&[&config], vec![codeword.clone()], &dft, &mut p_ch);
        let proofs: Vec<_> = results.iter().map(|(proof, _)| proof).collect();
        let mut v_ch = TestChallenger::new(perm);
        let outputs =
            verify_stir_multi(&[&config], &proofs, &mut v_ch).expect("an honest proof verifies");

        let output = &outputs[0];
        assert!(!output.first_round_indices.is_empty());
        for (&j, fiber) in output
            .first_round_indices
            .iter()
            .zip(&output.first_round_fiber_evals)
        {
            assert_eq!(fiber.len(), 1 << log_arity0);
            for (lane, &value) in fiber.iter().enumerate() {
                let positions = query_positions(&[j], &[lane], log_h, log_arity0);
                assert_eq!(positions, [j + lane * fold_height0]);
                assert_eq!(split_position(positions[0], fold_height0), (j, lane));
                assert_eq!(value, codeword[positions[0]], "fiber {j}, lane {lane}");
            }
        }
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
