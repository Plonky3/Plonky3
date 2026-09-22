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
//! **PCS accounting**: alpha batching and all subsequent `Combine` challenges share
//! one grind and one joint error budget. Each class counts every column at every
//! requested point across the pooled commitments. Round zero's `eta` satisfies that
//! budget at STIR's own proximity radius; queries are then derived from that `eta`.
//! Later folding/query grinds and unprotected OOD/Ans checks retain separate budgets.
//! PCS schedules use floor(log2(|E|)), with a further bit reserved under Johnson to
//! upper-bound the positive terms omitted by the shared proximity-gap approximation.
//! Grouping checks Combine feasibility before commitment, when opening-point counts
//! are unknown. The actual pooled width/point counts can still make an opening
//! infeasible: `Pcs::open` returns a configuration error before touching the transcript,
//! and `Pcs::verify` also rejects. Opening never repartitions already committed matrices.
//!
//! **Extraction relation**: quotients are tested at degree `< d`, where `d` is their
//! native matrix height. Correlated agreement therefore reconstructs original
//! polynomials of degree **at most `d`**, not strictly below `d`. Opening points must
//! lie outside their matrix's shared LDE coset; this is checked independently of
//! the query positions. Agreement on more than `d` positions makes all reconstructed
//! polynomials for one column identical across its opening points. Callers must
//! account for this degree allowance in their own relation and soundness budget.
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

mod grouping;
mod open;
mod plan;
mod verify;

use alloc::borrow::Cow;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::ops::Deref;

use itertools::izip;
use p3_challenger::{
    CanObserve, CanSampleUniformBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
    SerializingChallenger32,
};
use p3_commit::{CommitmentOpening, Mmcs, OpenedValues, OpeningRequest, Pcs, UnivariateStarkPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeField32, PrimeField64, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixCow, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_symmetric::CryptographicPermutation;
use p3_util::{log2_strict_usize, reverse_bits_len};
use serde::{Deserialize, Serialize};
use spin::RwLock;
use tracing::instrument;

use self::grouping::GroupPlan;
use crate::config::{StirConfig, StirConfigError, StirOptions, StirParameters};
use crate::error::StirError;
use crate::pcs_budget::PcsBatch;
use crate::pcs_transcript::observe_commitment;
use crate::proof::StirProof;

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
    /// The shared-domain layout this data was committed under: the plan `plan_groups`
    /// produced from the committed heights, read back from the stored groups.
    fn group_plan(&self) -> GroupPlan {
        GroupPlan {
            log_lde_heights: self
                .groups
                .iter()
                .map(|group| group.log_lde_height)
                .collect(),
            group_of_matrix: self
                .placement
                .iter()
                .map(|&(group_idx, _)| group_idx)
                .collect(),
        }
    }

    /// Native log2 height of each matrix, in the order the caller committed them.
    fn log_native_heights(&self) -> Vec<usize> {
        self.placement
            .iter()
            .map(|&(group_idx, idx)| self.groups[group_idx].log_native_heights[idx])
            .collect()
    }
}

/// One Merkle root per shared-domain group of a commitment, in descending LDE height.
///
/// The matrices of one `commit()` call are partitioned into groups of bounded height spread,
/// each extended onto its own shared domain (§7's same-domain requirement applies within a
/// group, not across the whole commitment) and committed in its own tree. A commitment whose
/// heights all fit one group therefore holds a single root.
///
/// The roots are wrapped rather than handed out as a bare `Vec`.
///
/// A challenger can then observe the whole commitment in one call.
///
/// That single call is what a generic proving configuration asks its challenger for.
///
/// The absorption is seeded with the root count.
///
/// Two commitments that split a given total of roots differently stay apart.
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

/// Absorbs a commitment's roots under the commitment scheme's own transcript.
///
/// The seed fixes how many roots the commitment holds.
///
/// Two commitments that split a given total of roots differently stay apart.
///
/// So do two commitments over the same roots in the opposite order.
///
/// Every challenger backend routes here rather than writing the sequence out again.
/// A backend that absorbed a commitment differently would not fail a test: prover and verifier
/// share one challenger type, so both would be wrong together.
fn observe_stir_commitment<Ch, F, C>(challenger: &mut Ch, commitment: StirCommitment<C>)
where
    Ch: CanObserve<F> + CanObserve<C>,
    F: PrimeField64,
    C: Clone,
{
    // The typed phase owns the whole sequence: the seed, then the roots.
    observe_commitment::<Ch, F, C>(challenger, commitment.0);
}

// The shared body above is a free function rather than a blanket impl.
// A blanket impl would leave the field type constrained only by the where clause, which Rust
// rejects as an unconstrained parameter, so each backend needs its own impl regardless.
//
// A backend with no impl here cannot be paired with this commitment scheme at all.
// The 64-bit serializing challenger has none, since nothing pairs it with this scheme today.
impl<F, P, C, const WIDTH: usize, const RATE: usize> CanObserve<StirCommitment<C>>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: PrimeField64,
    P: CryptographicPermutation<[F; WIDTH]>,
    C: Clone,
    Self: CanObserve<C>,
{
    fn observe(&mut self, commitment: StirCommitment<C>) {
        observe_stir_commitment::<_, F, _>(self, commitment);
    }
}

impl<F, Inner, C> CanObserve<StirCommitment<C>> for SerializingChallenger32<F, Inner>
where
    F: PrimeField32,
    Inner: CanObserve<u8>,
    C: Clone,
    Self: CanObserve<C>,
{
    fn observe(&mut self, commitment: StirCommitment<C>) {
        observe_stir_commitment::<_, F, _>(self, commitment);
    }
}

/// Degree, Combine shape, and actual per-class alpha counts. `None` distinguishes
/// commit-time probes from opening budgets, including a one-opening singleton.
type StirConfigKey = (usize, usize, u64, Option<Vec<(usize, usize)>>);

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
    options: StirOptions,
    batch_proof_of_work_bits: usize,
    /// Maximum `h_max - h_min`, in octaves, among the native heights sharing one LDE domain.
    ///
    /// `0` puts every distinct native height on its own domain, so `Combine` never runs and
    /// each height gets its own STIR instance. A value at or above the committed spread puts
    /// everything on one domain. See [`DEFAULT_MAX_LOG_HEIGHT_SPREAD`].
    max_log_height_spread: usize,
    /// `StirConfig::try_new` runs an 80-iteration floating-point bisection per stage to
    /// derive sound round parameters. `open`/`verify` re-derive it per LDE-height bucket, and
    /// bucket shapes recur across calls and across proofs of the same statement, so caching
    /// them here avoids repeating that derivation every time. Keys include the degree,
    /// Combine shape and per-class alpha counts, and distinguish commit-time feasibility
    /// probes from actual opening schedules.
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
            options: StirOptions::default(),
            batch_proof_of_work_bits: 0,
            max_log_height_spread: DEFAULT_MAX_LOG_HEIGHT_SPREAD,
            config_cache: Arc::new(RwLock::new(alloc::collections::BTreeMap::new())),
        }
    }

    /// Set the STIR prover/proof-size tradeoffs. Both prover and verifier must agree on
    /// these options; they are not carried in the proof. Cached schedules are reset so
    /// clones configured with other options retain their own derivations.
    #[must_use]
    pub fn with_options(mut self, options: StirOptions) -> Self {
        self.options = options;
        self.config_cache = Arc::new(RwLock::new(alloc::collections::BTreeMap::new()));
        self
    }

    /// Options used to derive this PCS instance's STIR schedules.
    pub const fn options(&self) -> StirOptions {
        self.options
    }

    /// Grind after absorbing all opening claims and immediately before sampling `alpha`.
    ///
    /// Defaults to zero. Both sides must agree on the difficulty. The PCS carries one
    /// witness for the whole opening batch, regardless of its number of height buckets.
    /// This credits retries of the joint alpha/`Combine` challenge block. It can lower
    /// round zero's `eta` and query count, and admit more native heights per group.
    /// Configure it **before committing**, identically on the prover and verifier,
    /// because it affects the committed layout. Changing it detaches cached schedules.
    /// [`StirParameters::max_pow_bits`] applies only to STIR's later grinding sites.
    ///
    /// # Panics
    ///
    /// If the difficulty cannot be sampled from a base-field element and a `usize`.
    #[must_use]
    pub fn with_batch_proof_of_work_bits(mut self, bits: usize) -> Self
    where
        Val: PrimeField64,
    {
        assert!(
            bits < Val::bits().min(usize::BITS as usize),
            "invalid batching PoW difficulty"
        );
        self.batch_proof_of_work_bits = bits;
        self.config_cache = Arc::new(RwLock::new(alloc::collections::BTreeMap::new()));
        self
    }

    /// Difficulty of the PCS opening-batching grind.
    pub const fn batch_proof_of_work_bits(&self) -> usize {
        self.batch_proof_of_work_bits
    }

    /// PCS-owned grinding metadata for an opening-batching security term.
    ///
    /// The PCS internally credits this site once to its joint alpha/`Combine` error.
    /// This metadata does not include the caller's outer protocol or hash-security terms.
    pub const fn grinding_sites(&self) -> p3_security::GrindingSites {
        p3_security::GrindingSites {
            batch_combination: self.batch_proof_of_work_bits,
            ..p3_security::GrindingSites::NONE
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
    Val: TwoAdicField + PrimeField64,
    Challenge: ExtensionField<Val>,
    StirMmcs: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger<Witness = Val>,
{
    /// A commit-time Combine feasibility probe, computed and cached on first use.
    ///
    /// `combine` carries the bucket's `(num_classes, ell)` when more than one native-height
    /// class shares its domain and §7's `Combine` therefore runs, and is `None` otherwise.
    /// Opening multiplicities are unknown at commit time. This probe cannot substitute
    /// for `get_or_try_compute_pcs_config` when constructing or verifying a proof.
    fn get_or_try_compute_stir_config(
        &self,
        log_stir_degree: usize,
        combine: Option<(usize, u64)>,
    ) -> Result<Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>>, StirConfigError> {
        self.get_or_try_compute_config(log_stir_degree, combine, None)
    }

    /// Derive from every alpha contribution in this opening batch, across commitments.
    fn get_or_try_compute_pcs_config(
        &self,
        log_stir_degree: usize,
        classes: &[(usize, usize)],
    ) -> Result<Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>>, StirConfigError> {
        if classes.is_empty() {
            return Err(StirConfigError::InvalidPcsBatch);
        }
        let combine = crate::pcs_budget::combine_requirement(log_stir_degree, classes)?;
        self.get_or_try_compute_config(log_stir_degree, combine, Some(classes))
    }

    fn get_or_try_compute_config(
        &self,
        log_stir_degree: usize,
        combine: Option<(usize, u64)>,
        classes: Option<&[(usize, usize)]>,
    ) -> Result<Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>>, StirConfigError> {
        let (num_classes, ell) = combine.unwrap_or((1, 0));
        let key: StirConfigKey = (
            log_stir_degree,
            num_classes,
            ell,
            classes.map(<[_]>::to_vec),
        );

        if let Some(config) = self.config_cache.read().get(&key) {
            return Ok(config.clone());
        }

        // Derived before the write guard is taken: `spin::RwLock` does not park, so holding it
        // across the bisection would make a thread missing on *any* key busy-spin for the
        // whole derivation — under rayon, possibly while the holder is descheduled. The
        // derivation is idempotent, so a racing duplicate is harmless: the loser's `Arc` is
        // simply dropped in favour of whichever landed first.
        let config = Arc::new(StirConfig::try_new_with_pcs_batch(
            log_stir_degree,
            self.stir.clone(),
            PcsBatch {
                classes: classes.unwrap_or(&[]),
                combine,
                pow_bits: self.batch_proof_of_work_bits,
            },
            self.options,
        )?);

        let mut cache = self.config_cache.write();
        if cache.len() >= CONFIG_CACHE_CAPACITY && !cache.contains_key(&key) {
            return Ok(config);
        }
        Ok(cache.entry(key).or_insert(config).clone())
    }

    /// Log2 STIR degree of a bucket whose shared LDE domain has size `2^log_lde_height`.
    fn log_stir_degree(&self, log_lde_height: usize) -> usize {
        log_lde_height.saturating_sub(self.stir.log_blowup).max(1)
    }
}

impl<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
where
    Val: TwoAdicField + PrimeField64,
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

    /// See `StirPcsProof`.
    type Proof = StirPcsProof<Val, Challenge, InputMmcs, StirMmcs>;
    type Error = StirError<StirMmcs::Error, InputMmcs::Error>;
    type ProverError = StirConfigError;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        TwoAdicMultiplicativeCoset::new(Val::ONE, log2_strict_usize(degree)).unwrap()
    }

    #[instrument(name = "STIR PCS commit", skip_all)]
    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
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
        Ok(self.commit_groups(&plan, grouped, &log_native_heights))
    }

    #[instrument(name = "STIR PCS open", skip_all)]
    fn open(
        &self,
        commitment_data_with_opening_points: Vec<OpeningRequest<'_, Self::ProverData, Challenge>>,
        challenger: &mut Challenger,
    ) -> Result<(OpenedValues<Challenge>, Self::Proof), Self::ProverError> {
        let prepared = self.prepare_open(&commitment_data_with_opening_points, challenger)?;
        let prover_data: Vec<&Self::ProverData> = commitment_data_with_opening_points
            .iter()
            .map(
                |OpeningRequest {
                     prover_data: data, ..
                 }| *data,
            )
            .collect();
        Ok(self.prove_buckets(&prover_data, prepared, challenger))
    }

    #[instrument(name = "STIR PCS verify", skip_all)]
    fn verify(
        &self,
        commitments_with_opening_points: Vec<
            CommitmentOpening<Challenge, Self::Commitment, Self::Domain>,
        >,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        self.verify_opening(commitments_with_opening_points, proof, challenger)
    }
}

impl<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger> UnivariateStarkPcs<Challenge, Challenger>
    for TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
where
    Val: TwoAdicField + PrimeField64,
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
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixCow<'a, Val>>;

    const ZK: bool = false;

    fn log_max_trace_height(&self) -> usize {
        Val::TWO_ADICITY.saturating_sub(self.stir.log_blowup)
    }

    fn log_min_trace_height(&self) -> usize {
        // Multiplicative coset selectors are defined at every height, down to a single row.
        0
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
        let width = lde.width();
        // In bit-reversed order the first `poly_height` rows are the polynomial's values on
        // the native-size GENERATOR-shifted sub-coset. Interpolate only those rows instead of
        // the whole committed LDE and discarding the high zero coefficients afterwards.
        let native_lde = RowMajorMatrixView::new(&lde.values[..poly_height * width], width)
            .bit_reverse_rows()
            .to_row_major_matrix();
        let mut coeffs = self.dft.coset_idft_batch(native_lde, Val::GENERATOR);
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
    ) -> Result<Vec<RowMajorMatrix<Val>>, Self::ProverError> {
        let min_height = 1usize << self.stir.log_starting_folding_factor;
        Ok(evaluations
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
            .collect())
    }

    fn commit_ldes(
        &self,
        ldes: Vec<RowMajorMatrix<Val>>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
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
                    let native_height = 1usize << log_native_height;
                    let width = lde.width();
                    let mut native_lde = lde;
                    native_lde.values.truncate(native_height * width);
                    let natural_native_lde = native_lde.bit_reverse_rows().to_row_major_matrix();
                    let mut coeffs = self
                        .dft
                        .coset_idft_batch(natural_native_lde, Val::GENERATOR);
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
        Ok(self.commit_groups(&plan, grouped, &log_native_heights))
    }
}

/// One entry per distinct shared LDE height across every commitment's groups (descending). A
/// commitment contributes to one entry per group it holds. Each entry holds:
/// - the STIR IOP proof for that bucket, whose initial oracle is the bucket's reduced
///   opening, committed by STIR itself;
/// - `input_openings[commit_idx]`: one shared multi-opening proof for that commitment's rows
///   at the bucket's queried positions, `None` if the commitment has no group at this
///   bucket's LDE height.
type StirPcsBucket<Val, Challenge, InputMmcs, StirMmcs> = (
    StirProof<Challenge, StirMmcs, Val>,
    Vec<Option<InputOpenings<Val, InputMmcs>>>,
);

/// A PCS opening proof, with one optional batching witness shared by every height bucket.
///
/// The witness belongs to the PCS: standalone STIR proofs do not sample `alpha`.
/// The optional field is serialized even when grinding is disabled, so this encoding
/// differs from the former bare vector of bucket proofs.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct StirPcsProof<
    Val: Field,
    Challenge: Field,
    InputMmcs: Mmcs<Val>,
    StirMmcs: Mmcs<Challenge>,
> {
    /// Present exactly when the PCS's batching difficulty is positive.
    pub batch_pow_witness: Option<Val>,
    /// STIR proofs and input openings in descending shared LDE height.
    pub buckets: Vec<StirPcsBucket<Val, Challenge, InputMmcs, StirMmcs>>,
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

/// The bit-reversed LDE row index of each natural-order position: the LDE is stored
/// bit-reversed, so natural position `p` is row `rev(p)`.
fn positions_to_row_indices(positions: &[usize], log_h: usize) -> Vec<usize> {
    positions
        .iter()
        .map(|&p| reverse_bits_len(p, log_h))
        .collect()
}

fn opening_point_in_domain<F: TwoAdicField, EF: ExtensionField<F>>(
    point: EF,
    log_domain_size: usize,
) -> bool {
    (point * F::GENERATOR.inverse()).exp_power_of_2(log_domain_size) == EF::ONE
}

#[cfg(test)]
mod tests;
