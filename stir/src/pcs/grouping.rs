//! Partition of one commitment's matrices into shared-LDE-domain groups.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, PrimeField64, TwoAdicField};

use super::TwoAdicStirPcs;

/// How one commitment's matrices are partitioned across shared LDE domains.
///
/// Derived identically by the prover (from the committed heights) and the verifier (from the
/// claimed domain sizes), so no part of it travels in the proof.
pub(super) struct GroupPlan {
    /// Log2 LDE height of each group, descending.
    pub(super) log_lde_heights: Vec<usize>,
    /// Group index of each matrix, in the order the caller supplied them.
    pub(super) group_of_matrix: Vec<usize>,
}

impl GroupPlan {
    /// Log2 LDE height of the group holding `matrix`.
    pub(super) fn log_lde_height_of(&self, matrix: usize) -> usize {
        self.log_lde_heights[self.group_of_matrix[matrix]]
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
    pub(super) fn combine_band_width(&self, tallest: usize, lowest: usize) -> usize {
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
    pub(super) fn plan_groups(&self, log_native_heights: &[usize]) -> GroupPlan {
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
    pub(super) fn combine_key(native_heights: &[usize]) -> Option<(usize, u64)> {
        (native_heights.len() >= 2).then(|| {
            let d_star = 1u64 << native_heights[0];
            let ell = native_heights.len() as u64 * (d_star + 1)
                - native_heights.iter().map(|&d| 1u64 << d).sum::<u64>();
            (native_heights.len(), ell)
        })
    }
}
