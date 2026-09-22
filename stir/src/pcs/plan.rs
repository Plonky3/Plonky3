//! Buckets, classes and alpha offsets of one opening, shared by the prover and the verifier.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{CommitmentOpening, MatrixOpening, Mmcs};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_util::log2_strict_usize;

use super::TwoAdicStirPcs;
use super::grouping::GroupPlan;
use crate::config::{StirConfig, StirConfigError};
use crate::pcs_transcript::{StirPcsBucketShape, StirPcsOpeningShape};

/// Public metadata of one opened matrix, read identically by both sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct OpenedMatrix {
    /// Log2 of the native height, which names the matrix's alpha-batching class.
    pub(super) log_native_height: usize,
    /// Columns, hence alpha powers drawn per opening point.
    pub(super) width: usize,
    /// Opening points the matrix is read at.
    pub(super) num_points: usize,
}

/// One commitment's share of an opening: its shared-domain layout and its opened matrices.
pub(super) struct OpenedCommitment {
    pub(super) groups: GroupPlan,
    /// In the order the caller committed them.
    pub(super) matrices: Vec<OpenedMatrix>,
}

/// Where one opened matrix sits in an opening plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MatrixSlot {
    /// Log2 of the shared LDE domain its group was committed on.
    pub(super) log_lde_height: usize,
    /// Log2 of its native height, which names its class.
    pub(super) log_native_height: usize,
    /// Index of its class within its bucket's `classes`.
    pub(super) class: usize,
    pub(super) width: usize,
    /// Power of alpha weighting its first point's quotient; each later point adds `width`.
    pub(super) alpha_offset: usize,
}

impl MatrixSlot {
    /// Power of alpha weighting the quotient at the matrix's `point`-th opening point.
    pub(super) const fn alpha_exponent(&self, point: usize) -> usize {
        self.alpha_offset + point * self.width
    }
}

/// One commitment's share of an opening plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CommitmentPlan {
    /// Groups, hence Merkle roots, the commitment's layout holds.
    pub(super) num_groups: usize,
    /// One slot per matrix, caller order.
    pub(super) matrices: Vec<MatrixSlot>,
}

/// One commitment's group on a bucket's shared domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BucketInput {
    /// Index of the group, hence of its Merkle root, within the commitment.
    pub(super) group: usize,
    /// Caller-order indices of the group's matrices, in the order its tree holds them.
    pub(super) matrices: Vec<usize>,
}

/// One distinct shared LDE height across the opened commitments: one STIR instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BucketPlan {
    pub(super) log_lde_height: usize,
    /// `(log2 native height, alpha powers)` of every class on this domain, tallest first:
    /// exactly the batch shape the bucket's schedule is derived for.
    pub(super) classes: Vec<(usize, usize)>,
    /// Per commitment, caller order: its group on this domain, if any.
    pub(super) inputs: Vec<Option<BucketInput>>,
}

impl BucketPlan {
    /// Log2 native height of every class on this domain, tallest first.
    pub(super) fn log_native_heights(&self) -> Vec<usize> {
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
pub(super) struct OpeningPlan {
    pub(super) commitments: Vec<CommitmentPlan>,
    /// Descending LDE height: the order both sides play the buckets in.
    pub(super) buckets: Vec<BucketPlan>,
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
    pub(super) fn new(commitments: &[OpenedCommitment]) -> Result<Self, StirConfigError> {
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
    pub(super) fn transcript_shape<Val, Challenge, StirMmcs, Challenger>(
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

impl<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
    TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger>
where
    Val: TwoAdicField + PrimeField64,
    Challenge: ExtensionField<Val>,
    StirMmcs: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger<Witness = Val>,
{
    /// The opening schedule of one bucket, derived from its classes and cached on first use.
    pub(super) fn bucket_config(
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
    pub(super) fn claimed_opening_plan<C>(
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
pub(super) fn combine_coefficients<EF: Field>(
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
