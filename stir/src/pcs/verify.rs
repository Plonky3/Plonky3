//! The verifier's side of an opening.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;

use itertools::Itertools;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{CommitmentOpening, MatrixOpening, Mmcs, PointOpening};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    BasedVectorSpace, ExtensionField, PackedFieldExtension, PrimeField64, TwoAdicField,
    batch_multiplicative_inverse,
};
use p3_matrix::dense::RowMajorMatrix;

use super::plan::combine_coefficients;
use super::{
    StirCommitment, StirPcsProof, TwoAdicStirPcs, opening_point_in_domain,
    positions_to_row_indices, query_positions, split_position,
};
use crate::config::StirConfig;
use crate::error::{ProofShapeError, StirError};
use crate::pcs_transcript::{OpeningVerifierTranscript, observe_claims};
use crate::proof::StirProof;
use crate::utils::eval_degree_correction;
use crate::verifier::verify_stir_multi_inner;

/// One bucket's `Combine` state for the verifier: the sampled combination challenge and each
/// present native height's `(r_i, gap_i)` coefficients (`None` when the bucket has only one
/// class, so no `Combine` step ran).
type BucketCombine<Challenge> = Option<(
    Challenge,
    alloc::collections::BTreeMap<usize, (Challenge, usize)>,
)>;

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
    /// Check an opening proof against the claimed evaluations it is meant to open.
    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn verify_opening(
        &self,
        commitments_with_opening_points: Vec<
            CommitmentOpening<
                Challenge,
                StirCommitment<InputMmcs::Commitment>,
                TwoAdicMultiplicativeCoset<Val>,
            >,
        >,
        proof: &StirPcsProof<Val, Challenge, InputMmcs, StirMmcs>,
        challenger: &mut Challenger,
    ) -> Result<(), StirError<StirMmcs::Error, InputMmcs::Error>> {
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

/// Source of an external initial oracle's fibers, for the STIR verifier's `None`: this PCS
/// has STIR commit the initial oracle itself, so no such source ever exists.
type NoExternalFibers<EF, E, IE> = fn(&[usize]) -> Result<Vec<Vec<EF>>, StirError<E, IE>>;

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
