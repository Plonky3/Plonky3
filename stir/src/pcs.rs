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

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
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
use p3_commit::{
    CommitmentOpening, MatrixOpening, Mmcs, OpenedValues, OpeningRequest, Pcs, PointOpening,
    UnivariateStarkPcs,
};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PrimeField32, PrimeField64,
    TwoAdicField, batch_multiplicative_inverse,
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

use crate::config::{StirConfig, StirConfigError, StirOptions, StirParameters};
use crate::error::{ProofShapeError, StirError};
use crate::pcs_budget::PcsBatch;
use crate::pcs_transcript::{
    OpeningProverTranscript, OpeningVerifierTranscript, StirPcsBucketShape, StirPcsOpeningShape,
    observe_claims, observe_commitment, observe_opened_values,
};
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

impl GroupPlan {
    /// Log2 LDE height of the group holding `matrix`.
    fn log_lde_height_of(&self, matrix: usize) -> usize {
        self.log_lde_heights[self.group_of_matrix[matrix]]
    }
}

/// Public metadata of one opened matrix, read identically by both sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OpenedMatrix {
    /// Log2 of the native height, which names the matrix's alpha-batching class.
    log_native_height: usize,
    /// Columns, hence alpha powers drawn per opening point.
    width: usize,
    /// Opening points the matrix is read at.
    num_points: usize,
}

/// One commitment's share of an opening: its shared-domain layout and its opened matrices.
struct OpenedCommitment {
    groups: GroupPlan,
    /// In the order the caller committed them.
    matrices: Vec<OpenedMatrix>,
}

/// Where one opened matrix sits in an opening plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MatrixSlot {
    /// Log2 of the shared LDE domain its group was committed on.
    log_lde_height: usize,
    /// Log2 of its native height, which names its class.
    log_native_height: usize,
    /// Index of its class within its bucket's `classes`.
    class: usize,
    width: usize,
    /// Power of alpha weighting its first point's quotient; each later point adds `width`.
    alpha_offset: usize,
}

impl MatrixSlot {
    /// Power of alpha weighting the quotient at the matrix's `point`-th opening point.
    const fn alpha_exponent(&self, point: usize) -> usize {
        self.alpha_offset + point * self.width
    }
}

/// One commitment's share of an opening plan.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CommitmentPlan {
    /// Groups, hence Merkle roots, the commitment's layout holds.
    num_groups: usize,
    /// One slot per matrix, caller order.
    matrices: Vec<MatrixSlot>,
}

/// One commitment's group on a bucket's shared domain.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BucketInput {
    /// Index of the group, hence of its Merkle root, within the commitment.
    group: usize,
    /// Caller-order indices of the group's matrices, in the order its tree holds them.
    matrices: Vec<usize>,
}

/// One distinct shared LDE height across the opened commitments: one STIR instance.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BucketPlan {
    log_lde_height: usize,
    /// `(log2 native height, alpha powers)` of every class on this domain, tallest first:
    /// exactly the batch shape the bucket's schedule is derived for.
    classes: Vec<(usize, usize)>,
    /// Per commitment, caller order: its group on this domain, if any.
    inputs: Vec<Option<BucketInput>>,
}

impl BucketPlan {
    /// Log2 native height of every class on this domain, tallest first.
    fn log_native_heights(&self) -> Vec<usize> {
        self.classes
            .iter()
            .map(|&(log_native_height, _)| log_native_height)
            .collect()
    }
}

/// Bucket order, classes, multiplicities and alpha offsets of one opening.
///
/// Both sides build it from public data only: committed or claimed domain sizes, and opened
/// widths and point counts. None of it travels in the proof.
///
/// Every matrix contributes a class to the bucket at its group's height, and every group
/// holds at least one matrix, so the buckets are exactly the groups' distinct heights. Callers
/// reject a matrix opened at no points before building a plan: such a matrix still names a
/// class here, while it would contribute nothing to the reduced opening.
#[derive(Clone, Debug, PartialEq, Eq)]
struct OpeningPlan {
    commitments: Vec<CommitmentPlan>,
    /// Descending LDE height: the order both sides play the buckets in.
    buckets: Vec<BucketPlan>,
}

impl OpeningPlan {
    /// Lay out one opening over the commitments' shared-domain groups.
    ///
    /// Classes are keyed by `(shared LDE height, native height)`. Alpha offsets run per class
    /// in claim order: commitment, then matrix, then point.
    ///
    /// # Errors
    ///
    /// `StirConfigError::PcsBatchMultiplicityOverflow` at the first matrix, in claim order,
    /// whose class total `width * num_points` overflows `usize`.
    fn new(commitments: &[OpenedCommitment]) -> Result<Self, StirConfigError> {
        let mut class_totals = BTreeMap::<(usize, usize), usize>::new();
        let alpha_offsets = commitments
            .iter()
            .map(|OpenedCommitment { groups, matrices }| {
                matrices
                    .iter()
                    .enumerate()
                    .map(|(m, matrix)| {
                        let key = (groups.log_lde_height_of(m), matrix.log_native_height);
                        let total = class_totals.entry(key).or_default();
                        let alpha_offset = *total;
                        *total = matrix
                            .width
                            .checked_mul(matrix.num_points)
                            .and_then(|n| total.checked_add(n))
                            .ok_or(StirConfigError::PcsBatchMultiplicityOverflow)?;
                        Ok(alpha_offset)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut bucket_heights: Vec<usize> = commitments
            .iter()
            .flat_map(|commitment| commitment.groups.log_lde_heights.iter().copied())
            .collect();
        bucket_heights.sort_unstable();
        bucket_heights.dedup();
        bucket_heights.reverse();

        let buckets: Vec<BucketPlan> = bucket_heights
            .into_iter()
            .map(|log_lde_height| BucketPlan {
                log_lde_height,
                classes: class_totals
                    .iter()
                    .rev()
                    .filter(|&(&(h, _), _)| h == log_lde_height)
                    .map(|(&(_, log_native_height), &total)| (log_native_height, total))
                    .collect(),
                inputs: commitments
                    .iter()
                    .map(|OpenedCommitment { groups, .. }| {
                        // Group LDE heights within a commitment are distinct, so at most one
                        // group sits on this domain.
                        let group = groups
                            .log_lde_heights
                            .iter()
                            .position(|&h| h == log_lde_height)?;
                        let matrices = groups
                            .group_of_matrix
                            .iter()
                            .enumerate()
                            .filter_map(|(m, &g)| (g == group).then_some(m))
                            .collect();
                        Some(BucketInput { group, matrices })
                    })
                    .collect(),
            })
            .collect();

        let commitments = commitments
            .iter()
            .zip(alpha_offsets)
            .map(
                |(OpenedCommitment { groups, matrices }, offsets)| CommitmentPlan {
                    num_groups: groups.log_lde_heights.len(),
                    matrices: matrices
                        .iter()
                        .zip(offsets)
                        .enumerate()
                        .map(|(m, (matrix, alpha_offset))| {
                            let log_lde_height = groups.log_lde_height_of(m);
                            let class = buckets
                                .iter()
                                .find(|bucket| bucket.log_lde_height == log_lde_height)
                                .and_then(|bucket| {
                                    bucket
                                        .classes
                                        .iter()
                                        .position(|&(h, _)| h == matrix.log_native_height)
                                })
                                .expect("every matrix names a class of the bucket at its height");
                            MatrixSlot {
                                log_lde_height,
                                log_native_height: matrix.log_native_height,
                                class,
                                width: matrix.width,
                                alpha_offset,
                            }
                        })
                        .collect(),
                },
            )
            .collect();

        Ok(Self {
            commitments,
            buckets,
        })
    }

    /// The bucket phase's transcript shape under per-bucket schedules in plan order.
    fn transcript_shape<Val, Challenge, StirMmcs, Challenger>(
        &self,
        configs: &[Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>>],
    ) -> StirPcsOpeningShape
    where
        Val: TwoAdicField + PrimeField64,
        Challenge: ExtensionField<Val>,
        StirMmcs: Mmcs<Challenge>,
        Challenger: FieldChallenger<Val> + GrindingChallenger<Witness = Val>,
    {
        StirPcsOpeningShape::new(
            self.buckets
                .iter()
                .zip(configs)
                .map(|(bucket, config)| {
                    StirPcsBucketShape::new(
                        bucket.log_lde_height,
                        bucket.log_native_heights(),
                        config,
                    )
                })
                .collect(),
        )
    }
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
    /// could form, and its Combine error bounds whatever union actually shows up.
    /// The batch grind is credited here because it precedes every Combine challenge.
    /// Since the width depends only on `tallest` and this PCS's parameters, every
    /// commitment independently agrees on it.
    ///
    /// This is a necessary feasibility check. Widths and opening-point counts can
    /// still make the *joint* budget infeasible. Both `open` and `verify` return
    /// configuration errors; opening checks the budget before touching the transcript.
    /// Opening never changes an already committed layout to make its budget fit.
    ///
    /// A width of `0` runs no `Combine` at all, so this always terminates: in the worst case
    /// every distinct height gets its own domain and its own STIR instance.
    ///
    /// Probes and full opening schedules use distinct entries in the same bounded cache.
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

    /// The opening schedule of one bucket, derived from its classes and cached on first use.
    fn bucket_config(
        &self,
        bucket: &BucketPlan,
    ) -> Result<Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>>, StirConfigError> {
        self.get_or_try_compute_pcs_config(
            self.log_stir_degree(bucket.log_lde_height),
            &bucket.classes,
        )
    }

    /// The opening plan the claims imply.
    ///
    /// Each commitment's layout is reproduced from its claimed domain sizes, exactly as
    /// `commit` derived it from the committed ones. A matrix's width is its first claim's
    /// value count.
    ///
    /// # Errors
    ///
    /// `StirConfigError::PcsBatchMultiplicityOverflow` when a class's alpha-power count
    /// overflows `usize`.
    fn claimed_opening_plan<C>(
        &self,
        claims: &[CommitmentOpening<Challenge, C, TwoAdicMultiplicativeCoset<Val>>],
    ) -> Result<OpeningPlan, StirConfigError> {
        let opened: Vec<OpenedCommitment> = claims
            .iter()
            .map(|CommitmentOpening { matrices, .. }| {
                let log_native_heights: Vec<usize> = matrices
                    .iter()
                    .map(|MatrixOpening { domain, .. }| log2_strict_usize(domain.size()))
                    .collect();
                OpenedCommitment {
                    groups: self.plan_groups(&log_native_heights),
                    matrices: matrices
                        .iter()
                        .zip(log_native_heights)
                        .map(
                            |(MatrixOpening { points, .. }, log_native_height)| OpenedMatrix {
                                log_native_height,
                                width: points.first().map_or(0, |p| p.values.len()),
                                num_points: points.len(),
                            },
                        )
                        .collect(),
                }
            })
            .collect();
        OpeningPlan::new(&opened)
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
    OpeningRequest<'a, StirProverData<Val, InputMmcs>, Challenge>;

/// Everything `open` settles before the bucket phase runs.
struct PreparedOpen<Val, Challenge, StirMmcs, Challenger> {
    batch_pow_witness: Option<Val>,
    /// Claimed evaluations, already absorbed into the transcript.
    opened_values: OpenedValues<Challenge>,
    /// Buckets, classes and alpha offsets of this opening: one STIR instance per bucket.
    plan: OpeningPlan,
    /// The derived config of each bucket's instance, in the plan's bucket order.
    stir_configs: Vec<Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>>>,
    /// One alpha-batched reduced opening per height class, bit-reversed and unmerged.
    ///
    /// A class is keyed by its shared LDE height and its native height.
    ///
    /// Merging a bucket's classes needs a challenge.
    ///
    /// The bucket phase is what draws it.
    reduced_openings: alloc::collections::BTreeMap<(usize, usize), Vec<Challenge>>,
}

impl<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
    TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
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
    ) -> Result<PreparedOpen<Val, Challenge, StirMmcs, Challenger>, StirConfigError> {
        // Step 1: Compute evaluations at opening points using Lagrange interpolation.
        let mats_and_points: Vec<_> = commitment_data_with_opening_points
            .iter()
            .map(
                |OpeningRequest {
                     prover_data: data,
                     points,
                 }| (lde_views(&self.input_mmcs, data), points),
            )
            .collect();

        // Each commitment's shared-domain layout, as committed.
        let group_plans: Vec<GroupPlan> = commitment_data_with_opening_points
            .iter()
            .map(
                |OpeningRequest {
                     prover_data: data, ..
                 }| data.group_plan(),
            )
            .collect();

        // The quotient must be defined throughout each matrix's *shared* domain,
        // not just its native interpolation domain or the eventual query set.
        for ((mats, points), group_plan) in mats_and_points.iter().zip(&group_plans) {
            assert_eq!(
                mats.len(),
                points.len(),
                "one point list per committed matrix is required"
            );
            for (m, (mat, points)) in mats.iter().zip(points.iter()).enumerate() {
                assert!(mat.width() > 0, "opening a matrix with zero columns");
                assert!(
                    !points.is_empty(),
                    "matrix was opened at no points; every committed matrix must be opened at least once"
                );
                let log_lde_height = group_plan.log_lde_height_of(m);
                for &point in points {
                    assert!(
                        !opening_point_in_domain::<Val, Challenge>(point, log_lde_height),
                        "opening point lies in its shared LDE domain"
                    );
                }
            }
        }

        // Price every actual class before absorbing claims, sampling, or grinding.
        let opened: Vec<OpenedCommitment> = izip!(
            commitment_data_with_opening_points,
            &mats_and_points,
            group_plans
        )
        .map(
            |(
                OpeningRequest {
                    prover_data: data, ..
                },
                (mats, points),
                groups,
            )| OpenedCommitment {
                groups,
                matrices: izip!(mats, points.iter(), data.log_native_heights())
                    .map(|(mat, points, log_native_height)| OpenedMatrix {
                        log_native_height,
                        width: mat.width(),
                        num_points: points.len(),
                    })
                    .collect(),
            },
        )
        .collect();
        let plan = OpeningPlan::new(&opened)?;
        // Schedules are derived in ascending bucket height, so an opening with several
        // infeasible buckets reports the shortest one's error.
        let mut stir_configs = plan
            .buckets
            .iter()
            .rev()
            .map(|bucket| self.bucket_config(bucket))
            .collect::<Result<Vec<_>, _>>()?;
        stir_configs.reverse();

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

        // Adjusted weights are consumed only on each matrix's native-height prefix. Track the
        // longest such prefix separately per point; inverse denominators remain full-size for
        // quotient construction below.
        let mut point_max_native_height: LinearMap<Challenge, usize> = LinearMap::new();
        for ((_, points), commitment) in mats_and_points.iter().zip(&plan.commitments) {
            for (points_for_mat, slot) in points.iter().zip(&commitment.matrices) {
                let h = 1usize << slot.log_native_height;
                for &point in points_for_mat {
                    if let Some(existing) = point_max_native_height.get_mut(&point) {
                        *existing = (*existing).max(h);
                    } else {
                        point_max_native_height.insert(point, h);
                    }
                }
            }
        }

        // Precompute adjusted barycentric weights once per opening point.
        // adjusted[i] = 1/(z - x_i) - 1/z, reused across all matrices opened at z.
        let adjusted_weights: LinearMap<Challenge, Vec<Challenge>> = inv_denoms
            .iter()
            .map(|(point, denoms)| {
                let h = *point_max_native_height.get(point).unwrap();
                (*point, compute_adjusted_weights(*point, &denoms[..h]))
            })
            .collect();

        let all_opened_values: OpenedValues<Challenge> = mats_and_points
            .iter()
            .zip(&plan.commitments)
            .map(|((mats, points), commitment)| {
                izip!(mats.iter(), points.iter(), commitment.matrices.iter())
                    .map(|(mat, points_for_mat, slot)| {
                        let h = 1usize << slot.log_native_height;
                        let (low_coset, _) = mat.split_rows(h);

                        points_for_mat
                            .iter()
                            .map(|&point| {
                                // Slice the precomputed adjusted weights to match this matrix's height.
                                // Zero-allocation hot path: straight to the SIMD dot product.
                                let adj = &adjusted_weights.get(&point).unwrap()[..h];
                                low_coset.interpolate_coset_with_precomputation(
                                    Val::GENERATOR,
                                    point,
                                    adj,
                                )
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec();

        // Bind every claimed evaluation before any challenge can depend on one.
        //
        // The description of this phase is derived from the same tree.
        //
        // It holds one container per level of it.
        //
        //     commitment -> matrix -> opening point -> one value per column
        //
        // So the grouping reaches the seed, not only the absorbed values.
        observe_opened_values::<Challenger, Val, Challenge>(challenger, &all_opened_values);

        // Step 2: Alpha-batch into one reduced-opening vector per (shared LDE domain, native
        // height) class. Every matrix in a class lives on the same physical domain (its
        // commitment's shared domain) and shares the same claimed degree, both required to
        // alpha-batch them together and, later, for `Combine` to merge classes soundly.
        // Claims are fixed before the grind. No prover message may intervene between
        // this site and `alpha`, or a retry could bypass the work just paid.
        let (alpha, batch_pow_witness) = crate::batch_transcript::prove::<Val, Challenge, _>(
            challenger,
            self.batch_proof_of_work_bits,
        );
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

        for (((mats, points), opened_vals), commitment) in mats_and_points
            .iter()
            .zip(&all_opened_values)
            .zip(&plan.commitments)
        {
            for (((mat, points_for_mat), opened_for_mat), slot) in izip!(mats.iter(), points.iter())
                .zip(opened_vals.iter())
                .zip(commitment.matrices.iter())
            {
                // A matrix opened at no points would contribute nothing to the reduced
                // opening, but the verifier still counts it as a native-height class (it reads
                // class membership off the claimed domains), so skipping it here would emit a
                // proof that cannot verify. `verify` rejects the same shape up front; this is
                // the prover-side mirror.
                assert!(
                    !points_for_mat.is_empty(),
                    "STIR PCS: matrix at native height 2^{} was opened at no \
                     points; every committed matrix must be opened at least once",
                    slot.log_native_height
                );

                let key = (slot.log_lde_height, slot.log_native_height);
                let ro = reduced_openings
                    .entry(key)
                    .or_insert_with(|| vec![Challenge::ZERO; mat.height()]);

                // Precompute alpha-batched row values for this matrix (reused per point).
                let p_x_vec: Vec<Challenge> = mat
                    .rowwise_packed_dot_product::<Challenge>(&packed_alpha_powers)
                    .collect();

                for (k, (point, ys)) in points_for_mat.iter().zip(opened_for_mat.iter()).enumerate()
                {
                    // The plan already bounded every class total, hence every exponent.
                    let alpha_pow_offset = alpha.exp_u64(slot.alpha_exponent(k) as u64);

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

        // Merging a bucket's classes needs a challenge drawn in the bucket phase.
        //
        // So the classes travel on unmerged.
        Ok(PreparedOpen {
            batch_pow_witness,
            opened_values: all_opened_values,
            plan,
            stir_configs,
            reduced_openings,
        })
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
            batch_pow_witness,
            opened_values,
            plan,
            stir_configs,
            mut reduced_openings,
        } = prepared;
        let stir_config_refs: Vec<&StirConfig<Val, Challenge, StirMmcs, Challenger>> =
            stir_configs.iter().map(AsRef::as_ref).collect();

        // One description covers the whole bucket phase.
        //
        //     merging challenges  ->  bracketed proximity test  ->  lane draws
        let shape = plan.transcript_shape(&stir_configs);
        let mut transcript =
            OpeningProverTranscript::<Challenger, Val, Challenge>::new(challenger, shape);

        // Phase 1: merge each bucket's classes into the codeword STIR runs on.
        //
        // A bucket holding one class is already its own codeword and draws nothing.
        let initial_codewords: Vec<Vec<Challenge>> = plan
            .buckets
            .iter()
            .enumerate()
            .map(|(bucket, bucket_plan)| {
                let r_comb = transcript.combination_challenge(bucket);
                combined_bucket_codeword(
                    &mut reduced_openings,
                    bucket_plan.log_lde_height,
                    &bucket_plan.log_native_heights(),
                    r_comb,
                )
            })
            .collect();

        // Phase 2: every bucket's proximity test runs in lockstep, inside the bracket.
        let bucket_results = transcript.delegate(|challenger| {
            prove_stir_multi_from_codewords(
                &stir_config_refs,
                initial_codewords,
                &self.dft,
                challenger,
            )
        });

        // Phase 3: one lane per first-round query draw, per bucket.
        //
        // Every STIR message is already in the sponge, the commitments above all.
        //
        // So no lane can be chosen to dodge a disagreement.
        let bucket_lanes: Vec<Vec<usize>> = (0..plan.buckets.len())
            .map(|bucket| transcript.lanes(bucket))
            .collect();
        transcript.finish();

        let bucket_proofs = izip!(&plan.buckets, &stir_configs, bucket_results, bucket_lanes)
            .map(|(bucket, stir_config, (stir_proof, first_round), lanes)| {
                let log_h = bucket.log_lde_height;
                let log_arity0 = stir_config.log_starting_folding_factor;
                // Both counts come from the schedule, derived independently of each other.
                //
                // The description fixes the lane count before the driver is seeded.
                //
                // The draw count comes out of the run.
                //
                // Pairing them zips two lists, which would silently truncate to the shorter.
                assert_eq!(
                    lanes.len(),
                    first_round.draws.len(),
                    "the schedule describes {} lanes but the run drew {} round-0 queries",
                    lanes.len(),
                    first_round.draws.len(),
                );
                let positions = query_positions(&first_round.draws, &lanes, log_h, log_arity0);
                let row_indices = positions_to_row_indices(&positions, log_h);

                let input_openings: Vec<Option<InputOpenings<Val, InputMmcs>>> = bucket
                    .inputs
                    .iter()
                    .zip(prover_data)
                    .map(|(input, data)| {
                        // Each group has its own tree on its own domain.
                        //
                        // So a bucket reads exactly the group committed at its LDE height.
                        let input = input.as_ref()?;
                        let (opened_values, opening_proof) = self
                            .input_mmcs
                            .open_multi_batch(&row_indices, &data.groups[input.group].data);
                        Some(InputOpenings {
                            opened_values,
                            opening_proof,
                        })
                    })
                    .collect();

                (stir_proof, input_openings)
            })
            .collect();

        (
            opened_values,
            StirPcsProof {
                batch_pow_witness,
                buckets: bucket_proofs,
            },
        )
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
        // SHAPE CHECK, before anything reaches the transcript: a matrix opened at no points
        // carries no claim to pin its width, and the prover skips it entirely — so it would
        // not create a native-height class where the verifier, reading class membership off
        // the claimed domains, still counts one. That disagreement decides whether `Combine`
        // runs and therefore whether `r_comb` is drawn, so it has to be settled before the
        // transcript can fork on it. Depends only on the public claims (mirrors
        // `fri::verifier::FriError::MatrixWithoutOpeningPoints`).
        for (commitment, domain_claims) in commitments_with_opening_points.iter().enumerate().map(
            |(
                commit_idx,
                CommitmentOpening {
                    matrices: domain_claims,
                    ..
                },
            )| (commit_idx, domain_claims),
        ) {
            for (
                matrix,
                MatrixOpening {
                    points: point_claims,
                    ..
                },
            ) in domain_claims.iter().enumerate()
            {
                if point_claims.is_empty() {
                    return Err(StirError::MatrixWithoutOpeningPoints { commitment, matrix });
                }
                let width = point_claims[0].values.len();
                for (point, claim) in point_claims.iter().enumerate() {
                    if width == 0 || claim.values.len() != width {
                        return Err(StirError::InvalidOpeningWidth {
                            commitment,
                            matrix,
                            point,
                        });
                    }
                }
            }
        }

        // Bind every claimed evaluation before any challenge can depend on one.
        //
        // The description of this phase is derived from the claims themselves.
        //
        // It holds one container per level of them.
        //
        //     commitment -> matrix -> opening point -> one value per column
        //
        // So the grouping reaches the seed, not only the absorbed values.
        //
        // The claims are the statement this call was handed, not proof fields it read.
        //
        // Their shape is the caller's to fix, and this function does not fix it.
        //
        //     the caller  ->  decides the claim tree, and constrains it
        //     here        ->  binds whatever tree it was handed
        //
        // A STARK caller builds that tree out of the proof's own opened values.
        //
        // It checks the widths against its AIR before reaching this call.
        //
        // The pre-check just above enforces only internal consistency of the tree.
        //
        // So deriving the seed from the claim layout binds the statement to the run.
        //
        // It is not STIR validating that layout on its own behalf.
        observe_claims::<Challenger, Val, Challenge, _, _>(
            challenger,
            &commitments_with_opening_points,
        );

        let alpha: Challenge = crate::batch_transcript::verify(
            challenger,
            self.batch_proof_of_work_bits,
            proof.batch_pow_witness,
        )?;
        let proof = &proof.buckets;

        // Reproduce each commitment's shared-domain layout from the claimed domain sizes,
        // exactly as `commit` derived it from the committed ones, and lay the opening out over
        // it. Nothing about the layout travels in the proof: it is a function of the claimed
        // heights, widths and point counts and this PCS's parameters, and a prover that used a
        // different one fixed different MMCS dimensions and fails the input opening check
        // below.
        let plan = self
            .claimed_opening_plan(&commitments_with_opening_points)
            .map_err(StirError::Config)?;

        // SHAPE CHECK: one Merkle root per group of the layout the claims imply.
        for (commit_idx, (CommitmentOpening { commitment, .. }, commitment_plan)) in
            commitments_with_opening_points
                .iter()
                .zip(&plan.commitments)
                .enumerate()
        {
            if commitment.len() != commitment_plan.num_groups {
                return Err(ProofShapeError::CommitmentRootCount {
                    commitment: commit_idx,
                    expected: commitment_plan.num_groups,
                    got: commitment.len(),
                }
                .into());
            }
        }

        for (commitment, (claims, commitment_plan)) in commitments_with_opening_points
            .iter()
            .zip(&plan.commitments)
            .enumerate()
        {
            for (matrix, (claim, slot)) in claims
                .matrices
                .iter()
                .zip(&commitment_plan.matrices)
                .enumerate()
            {
                for (point, opening) in claim.points.iter().enumerate() {
                    if opening_point_in_domain::<Val, Challenge>(opening.point, slot.log_lde_height)
                    {
                        return Err(StirError::OpeningPointInDomain {
                            commitment,
                            matrix,
                            point,
                        });
                    }
                }
            }
        }

        if proof.len() != plan.buckets.len() {
            return Err(ProofShapeError::BucketCount {
                expected: plan.buckets.len(),
                got: proof.len(),
            }
            .into());
        }

        let global_max_width = commitments_with_opening_points
            .iter()
            .flat_map(
                |CommitmentOpening {
                     matrices: domain_claims,
                     ..
                 }| {
                    domain_claims.iter().flat_map(
                        |MatrixOpening {
                             points: point_claims,
                             ..
                         }| {
                            point_claims
                                .iter()
                                .map(|PointOpening { values: v, .. }| v.len())
                        },
                    )
                },
            )
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
        // loop below) turns an `O(n_q)` recomputation per point into `O(1)`. The exponents
        // come from the plan, which runs them per `(log_shared_lde_height,
        // log_native_height)` class exactly as the prover's `reduced_openings` is keyed.
        let point_data: Vec<Vec<Vec<(Challenge, Challenge)>>> = commitments_with_opening_points
            .iter()
            .zip(&plan.commitments)
            .map(
                |(
                    CommitmentOpening {
                        matrices: domain_claims,
                        ..
                    },
                    commitment_plan,
                )| {
                    domain_claims
                        .iter()
                        .zip(&commitment_plan.matrices)
                        .map(
                            |(
                                MatrixOpening {
                                    points: point_claims,
                                    ..
                                },
                                slot,
                            )| {
                                point_claims
                                    .iter()
                                    .enumerate()
                                    .map(|(point, PointOpening { values: vals, .. })| {
                                        let offset =
                                            alpha.exp_u64(slot.alpha_exponent(point) as u64);

                                        let y_combined: Challenge = vals
                                            .iter()
                                            .zip(alpha_powers.iter())
                                            .map(|(&y, &ap)| y * ap)
                                            .sum();

                                        (offset, y_combined)
                                    })
                                    .collect()
                            },
                        )
                        .collect()
                },
            )
            .collect();

        // SHAPE CHECK: every bucket's input_openings has one slot per public commitment.
        // Without this, a malicious proof could omit trailing commitments — a `zip` would
        // silently drop them, their claimed values would still be observed into the
        // transcript (above), but they'd never be MMCS-opened or included in the
        // reduced-opening accumulation, so the proof would verify against a subset of the
        // public input.
        for (bucket, (_, input_openings)) in plan.buckets.iter().zip(proof) {
            if input_openings.len() != commitments_with_opening_points.len() {
                return Err(ProofShapeError::InputOpeningCount {
                    log_height: bucket.log_lde_height,
                    expected: commitments_with_opening_points.len(),
                    got: input_openings.len(),
                }
                .into());
            }
        }

        let stir_configs: Vec<Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>>> = plan
            .buckets
            .iter()
            .map(|bucket| self.bucket_config(bucket))
            .collect::<Result<Vec<_>, _>>()
            .map_err(StirError::Config)?;
        let stir_config_refs: Vec<&StirConfig<Val, Challenge, StirMmcs, Challenger>> =
            stir_configs.iter().map(AsRef::as_ref).collect();
        let stir_proofs: Vec<&StirProof<Challenge, StirMmcs, Val>> =
            proof.iter().map(|(p, _)| p).collect();

        // One description covers the whole bucket phase.
        //
        //     merging challenges  ->  bracketed proximity test  ->  lane draws
        //
        // Every count in it comes from the schedules and the claimed heights.
        //
        // None comes from the proof.
        //
        // Both sides fix the description up front.
        let shape = plan.transcript_shape(&stir_configs);
        let mut transcript =
            OpeningVerifierTranscript::<Challenger, Val, Challenge>::new(challenger, shape);

        // Phase 1: redraw each merging challenge and derive its per-class coefficients.
        //
        // Both happen at the transcript position the prover merged its classes at.
        //
        // A bucket holding one native-height class merges nothing and draws nothing.
        let bucket_combine: Vec<BucketCombine<Challenge>> = plan
            .buckets
            .iter()
            .enumerate()
            .map(|(bucket, bucket_plan)| {
                let r_comb = transcript.combination_challenge(bucket)?;
                let native_heights = bucket_plan.log_native_heights();
                // The tallest class heads the list.
                //
                // Its own degree is the target degree.
                let log_d_star = native_heights[0];
                let coeffs =
                    combine_coefficients(r_comb, log_d_star, native_heights.iter().copied());
                Some((r_comb, native_heights.into_iter().zip(coeffs).collect()))
            })
            .collect();

        // Phase 2: the proximity test runs inside the bracket.
        //
        // It absorbs each bucket's initial-oracle commitment and checks every round.
        //
        // It then hands back the round-0 fibers it authenticated against that commitment.
        //
        // The draws that selected those fibers come back alongside them.
        let outputs = match transcript.delegate(|challenger| {
            verify_stir_multi_inner(
                &stir_config_refs,
                &stir_proofs,
                challenger,
                None::<Vec<NoExternalFibers<Challenge, StirMmcs::Error, InputMmcs::Error>>>,
            )
        }) {
            Ok(outputs) => outputs,
            Err(err) => {
                // Releasing the completeness check keeps this rejection the only failure.
                transcript.abort();
                return Err(err);
            }
        };

        // Phase 3: one lane per first-round query draw, per bucket.
        //
        // Every STIR message is already in the sponge, the commitments above all.
        //
        // So no lane can be chosen to dodge a disagreement.
        let bucket_lanes: Vec<Vec<usize>> = (0..plan.buckets.len())
            .map(|bucket| transcript.lanes(bucket))
            .collect();
        transcript.finish();

        for (bucket, (bucket_plan, output)) in plan.buckets.iter().zip(&outputs).enumerate() {
            let log_h = bucket_plan.log_lde_height;
            let stir_config = &stir_configs[bucket];
            let input_openings = &proof[bucket].1;
            let combine_info = &bucket_combine[bucket];

            let bucket_height = 1usize << log_h;
            let log_arity0 = stir_config.log_starting_folding_factor;
            let fold_height0 = bucket_height >> log_arity0;
            let domain_gen = Val::two_adic_generator(log_h);

            // The lanes this bucket drew, one per first-round query draw.
            //
            // Both counts come from the schedule, derived independently of each other.
            //
            // Pairing them zips two lists, which would silently truncate to the shorter.
            let lanes = &bucket_lanes[bucket];
            assert_eq!(
                lanes.len(),
                output.first_round_draws.len(),
                "the schedule describes {} lanes but the replay drew {} round-0 queries",
                lanes.len(),
                output.first_round_draws.len(),
            );
            let positions = query_positions(&output.first_round_draws, lanes, log_h, log_arity0);
            let n_q = positions.len();
            let row_indices = positions_to_row_indices(&positions, log_h);
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
                .zip(&bucket_plan.inputs)
                .flat_map(
                    |(
                        CommitmentOpening {
                            matrices: domain_claims,
                            ..
                        },
                        input,
                    )| {
                        input
                            .iter()
                            .flat_map(|input| input.matrices.iter())
                            .map(|&idx| &domain_claims[idx])
                    },
                )
                .flat_map(
                    |MatrixOpening {
                         points: point_claims,
                         ..
                     }| {
                        point_claims.iter().map(|PointOpening { point, .. }| *point)
                    },
                )
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
            // the bucket's `classes` are (descending); merged after the accumulation loop.
            let mut expected_ro_by_class: Vec<Vec<Challenge>> =
                vec![Challenge::zero_vec(n_q); bucket_plan.classes.len()];

            // A commitment feeds this bucket through at most one of its groups: the one
            // committed on this domain. Its other groups live in other trees at other heights
            // and belong to other buckets.
            for (
                commit_idx,
                (
                    (
                        CommitmentOpening {
                            commitment,
                            matrices: domain_claims,
                        },
                        per_commit_opening,
                    ),
                    input,
                ),
            ) in commitments_with_opening_points
                .iter()
                .zip(input_openings.iter())
                .zip(&bucket_plan.inputs)
                .enumerate()
            {
                let Some(opening) = per_commit_opening else {
                    if input.is_some() {
                        return Err(ProofShapeError::MissingInputOpening {
                            log_height: log_h,
                            commitment: commit_idx,
                        }
                        .into());
                    }
                    continue;
                };
                let Some(input) = input else {
                    return Err(ProofShapeError::UnexpectedInputOpening {
                        log_height: log_h,
                        commitment: commit_idx,
                    }
                    .into());
                };

                // The commitment's matrices on this domain, in the order they were committed
                // to its tree: caller order, filtered to this group.
                let group_mats = &input.matrices;
                let slots = &plan.commitments[commit_idx].matrices;

                // Pin each matrix's width to its claimed evaluation count, never to the
                // proof. Every matrix has at least one claim — the up-front check in `verify`
                // rejects otherwise before the transcript is touched.
                let mat_widths: Vec<usize> =
                    group_mats.iter().map(|&idx| slots[idx].width).collect();

                let mat_class_indices: Vec<usize> =
                    group_mats.iter().map(|&idx| slots[idx].class).collect();
                let mat_point_slots: Vec<Vec<usize>> = group_mats
                    .iter()
                    .map(|&idx| {
                        domain_claims[idx]
                            .points
                            .iter()
                            .map(|PointOpening { point, .. }| {
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
                        &commitment[input.group],
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
            // Both class counts come from the bucket's `classes`, never from the proof. Still
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
                    for (&(log_native_h, _), ro_class) in
                        bucket_plan.classes.iter().zip(&expected_ro_by_class)
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

/// Source of an external initial oracle's fibers, for the STIR verifier's `None`: this PCS
/// has STIR commit the initial oracle itself, so no such source ever exists.
type NoExternalFibers<EF, E, IE> = fn(&[usize]) -> Result<Vec<Vec<EF>>, StirError<E, IE>>;

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

/// One commitment's claims: per matrix, its domain and its `(point, values)` pairs.
type CommitmentClaims<C, D, EF> = CommitmentOpening<EF, C, D>;

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
        .flat_map(
            |(
                commitment,
                CommitmentOpening {
                    matrices: domain_claims,
                    ..
                },
            )| {
                domain_claims.iter().enumerate().flat_map(
                    move |(
                        matrix,
                        MatrixOpening {
                            points: point_claims,
                            ..
                        },
                    )| {
                        point_claims.iter().enumerate().map(
                            move |(slot, PointOpening { point: claimed, .. })| {
                                (commitment, matrix, slot, claimed)
                            },
                        )
                    },
                )
            },
        )
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
/// # Overview
///
/// The map is keyed by `(log_shared_lde_height, log_native_height)`.
///
/// This removes every class of the given shared height.
///
/// It returns the merged codeword in natural order, not bit-reversed.
///
/// Each class is taken out of the map rather than borrowed, to un-reverse it in place.
///
/// A class spans the whole shared domain at PCS scale.
///
/// Cloning them all would double peak memory for the duration of the merge.
///
/// # Arguments
///
/// - `reduced_openings`: every class's alpha-batched reduced opening, bit-reversed.
/// - `log_shared_h`: log of the shared domain whose classes are merged.
/// - `log_native_heights`: log of every native height present, descending.
/// - `r_comb`: the merging challenge, or nothing when a single height is present.
///
/// # Panics
///
/// - When a listed class is absent from the map.
/// - When several heights are present and no merging challenge was drawn for them.
fn combined_bucket_codeword<Val, Challenge>(
    reduced_openings: &mut alloc::collections::BTreeMap<(usize, usize), Vec<Challenge>>,
    log_shared_h: usize,
    log_native_heights: &[usize],
    r_comb: Option<Challenge>,
) -> Vec<Challenge>
where
    Val: TwoAdicField + PrimeField64,
    Challenge: ExtensionField<Val> + TwoAdicField,
{
    // The merge indexes its inputs, and produces its output, in natural order.
    //
    // The reduced openings are bit-reversed.
    //
    // They were built from the bit-reversed LDE matrices.
    //
    // So each class's codeword is un-reversed on the way in.
    //
    // The merged result then sits in the natural order the proximity test expects.
    let mut natural_ros: Vec<Vec<Challenge>> = log_native_heights
        .iter()
        .map(|&log_d| {
            let mut natural = reduced_openings
                .remove(&(log_shared_h, log_d))
                .expect("every listed class was accumulated into this map");
            reverse_slice_index_bits(&mut natural);
            natural
        })
        .collect();

    // A bucket of one class is already its own codeword.
    let Some(r_comb) = r_comb else {
        assert_eq!(
            natural_ros.len(),
            1,
            "several native heights on one domain must be merged under a challenge",
        );
        return natural_ros
            .pop()
            .expect("a bucket holds at least one class");
    };

    // The tallest class heads the descending list.
    //
    // Its own degree is the target degree.
    let log_d_star = log_native_heights[0];
    let coeffs = combine_coefficients(r_comb, log_d_star, log_native_heights.iter().copied());

    let groups: Vec<(Challenge, usize, &[Challenge])> = coeffs
        .into_iter()
        .zip(&natural_ros)
        .map(|((r_i, gap), ro)| (r_i, gap, ro.as_slice()))
        .collect();
    combine_on_coset(&groups, r_comb, Val::GENERATOR, log_shared_h)
}

type MatricesAndPoints<'a, F, EF> = (Vec<RowMajorMatrixView<'a, F>>, &'a Vec<Vec<EF>>);

fn opening_point_in_domain<F: TwoAdicField, EF: ExtensionField<F>>(
    point: EF,
    log_domain_size: usize,
) -> bool {
    (point * F::GENERATOR.inverse()).exp_power_of_2(log_domain_size) == EF::ONE
}

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
mod tests;
