//! The prover's side of an opening: claims, reduced openings and the bucket phase.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, OpeningRequest};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    BasedVectorSpace, ExtensionField, PackedFieldExtension, PrimeField64, TwoAdicField,
    batch_multiplicative_inverse,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::interpolation::{Interpolate, compute_adjusted_weights};
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

use super::grouping::GroupPlan;
use super::plan::{OpenedCommitment, OpenedMatrix, OpeningPlan, combine_coefficients};
use super::{
    InputOpenings, StirPcsProof, StirProverData, TwoAdicStirPcs, opening_point_in_domain,
    positions_to_row_indices, query_positions,
};
use crate::config::{StirConfig, StirConfigError};
use crate::pcs_transcript::{OpeningProverTranscript, observe_opened_values};
use crate::prover::prove_stir_multi_from_codewords;
use crate::utils::combine_on_coset;

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

/// One commitment's prover data alongside the opening points of each of its matrices.
type ProverDataWithPoints<'a, Val, InputMmcs, Challenge> =
    OpeningRequest<'a, StirProverData<Val, InputMmcs>, Challenge>;

/// Everything `open` settles before the bucket phase runs.
pub(super) struct PreparedOpen<Val, Challenge, StirMmcs, Challenger> {
    pub(super) batch_pow_witness: Option<Val>,
    /// Claimed evaluations, already absorbed into the transcript.
    pub(super) opened_values: OpenedValues<Challenge>,
    /// Buckets, classes and alpha offsets of this opening: one STIR instance per bucket.
    pub(super) plan: OpeningPlan,
    /// The derived config of each bucket's instance, in the plan's bucket order.
    pub(super) stir_configs: Vec<Arc<StirConfig<Val, Challenge, StirMmcs, Challenger>>>,
    /// One alpha-batched reduced opening per height class, bit-reversed and unmerged.
    ///
    /// A class is keyed by its shared LDE height and its native height.
    ///
    /// Merging a bucket's classes needs a challenge.
    ///
    /// The bucket phase is what draws it.
    pub(super) reduced_openings: alloc::collections::BTreeMap<(usize, usize), Vec<Challenge>>,
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
    pub(super) fn prepare_open(
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
    pub(super) fn prove_buckets(
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
pub(super) fn combined_bucket_codeword<Val, Challenge>(
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
