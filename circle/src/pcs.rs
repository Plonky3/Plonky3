use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{
    CommitmentOpening, MatrixOpening, Mmcs, OpenedValues, OpeningRequest, Pcs, PeriodicLdeTable,
    PointOpening, PolynomialSpace, UnivariateStarkPcs,
};
use p3_field::extension::ComplexExtendable;
use p3_field::{ExtensionField, Field, PrimeField64, batch_multiplicative_inverse, dot_product};
use p3_fri::verifier::FriError;
use p3_fri::{BatchMultiOpening, FriFoldingStrategy, FriParameters};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixCow};
use p3_matrix::row_index_mapped::RowIndexMappedView;
use p3_matrix::{Dimensions, Matrix};
use p3_util::log2_strict_usize;
use p3_util::zip_eq::zip_eq;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug_span, info_span};

use crate::deep_quotient::{
    VanishingParts, accumulate_deep_quotient, compute_vanishing_parts, deep_quotient_reduce_row,
    extract_lambda,
};
use crate::domain::CircleDomain;
use crate::folding::{
    CircleFriFolding, CircleFriFoldingForMmcs, fold_row_with_inv_twiddle, fold_y,
};
use crate::point::{Point, compute_lagrange_den_batched};
use crate::prover::prove;
use crate::transcript::{
    CirclePcsShape, CircleProverTranscript, CircleTranscriptFailure, CircleVerifierTranscript,
};
use crate::verifier::{validate_proof_shape, verify_queries};
use crate::{
    CfftPerm, CfftPermutable, CircleEvaluations, CircleFriProof, build_periodic_lde_table_circle,
    cfft_permute_index, cfft_permute_slice,
};

#[derive(Clone, Debug)]
pub struct CirclePcs<Val: Field, InputMmcs, FriMmcs> {
    pub mmcs: InputMmcs,
    pub fri_params: FriParameters<FriMmcs>,
    pub _phantom: PhantomData<Val>,
}

impl<Val: Field, InputMmcs, FriMmcs> CirclePcs<Val, InputMmcs, FriMmcs> {
    /// Construct the PCS with grinding before opening batching, folding, and queries
    /// as configured by `fri_params`.
    pub const fn new(mmcs: InputMmcs, fri_params: FriParameters<FriMmcs>) -> Self {
        Self {
            mmcs,
            fri_params,
            _phantom: PhantomData,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct CircleInputProof<
    Val: Field,
    Challenge: Field,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
> {
    /// One multi-opening per input commitment, each covering every query with a
    /// single shared proof.
    input_openings: Vec<BatchMultiOpening<Val, InputMmcs>>,
    /// `first_layer_siblings[query]` holds one sibling per committed height.
    first_layer_siblings: Vec<Vec<Challenge>>,
    /// One shared proof authenticating every query's first-layer row.
    first_layer_proof: FriMmcs::MultiProof,
}

#[derive(Debug, Error)]
pub enum InputError<InputMmcsError, FriMmcsError>
where
    InputMmcsError: core::fmt::Debug,
    FriMmcsError: core::fmt::Debug,
{
    #[error("input MMCS error: {0:?}")]
    InputMmcsError(InputMmcsError),
    #[error("first layer MMCS error: {0:?}")]
    FirstLayerMmcsError(FriMmcsError),
    #[error("input shape error: mismatched dimensions")]
    InputShapeError,
    /// The opening point coincides with a queried domain point.
    ///
    /// The DEEP-quotient denominator vanishes there, so the row cannot be reduced.
    #[error("opening point coincides with a query point")]
    OpeningPointMatchesQueryPoint,
    #[error(
        "batch {batch}, matrix {matrix}: opened at no points; its width cannot be authenticated"
    )]
    MatrixWithoutOpeningPoints { batch: usize, matrix: usize },
    /// The tallest claimed domain leaves no room for the first-layer fold.
    ///
    /// The bivariate layer takes one bit before FRI folds anything.
    #[error("tallest claimed domain is 2^{log_n} rows; the first-layer fold needs at least 2")]
    DomainTooSmall { log_n: usize },
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize",
    deserialize = "Witness: Deserialize<'de>"
))]
pub struct CirclePcsProof<
    Val: Field,
    Challenge: Field,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Witness,
> {
    /// Witness for the grind before the challenge batching all opening claims.
    batch_pow_witness: Witness,
    first_layer_commitment: FriMmcs::Commitment,
    lambdas: Vec<Challenge>,
    fri_proof: CircleFriProof<
        Challenge,
        FriMmcs,
        Witness,
        CircleInputProof<Val, Challenge, InputMmcs, FriMmcs>,
    >,
}

/// One commitment together with every claim made about the matrices behind it.
///
/// Shape: for each matrix, its domain, then for each opening point the claimed values.
type CommitmentWithClaims<Val, Challenge, Com> =
    CommitmentOpening<Challenge, Com, CircleDomain<Val>>;

/// Challenges one replayed Circle PCS transcript hands back.
struct ReplayedChallenges<EF> {
    /// Challenge batching every claim into one DEEP quotient.
    alpha: EF,
    /// Challenge folding the first, bivariate layer.
    bivariate_beta: EF,
    /// One folding challenge per commit-phase round.
    betas: Vec<EF>,
    /// The query indices.
    indices: Vec<usize>,
}

/// Replay the whole Circle PCS transcript against the values the proof carries.
///
/// # Overview
///
/// Every step the shape describes is played here, in one pass.
/// Nothing downstream reads a challenge before this returns.
///
/// The driver is borrowed, so the caller decides whether to finalise or abort it.
///
/// # Arguments
///
/// - `transcript`: the seeded driver, described with the shape these claims fix.
/// - `rounds`: the claims, in batch then matrix then point order.
/// - `proof`: the proof carrying every value the run absorbs.
///
/// # Errors
///
/// When a grinding witness misses the difficulty its step requires.
///
/// # Panics
///
/// When an opening point is a point at infinity of the projective line.
/// Opening points are part of the statement, supplied by the caller, never by a proof.
fn replay_transcript<Val, Challenge, InputMmcs, FriMmcs, Challenger>(
    transcript: &mut CircleVerifierTranscript<'_, Challenger, Val, Challenge>,
    rounds: &[CommitmentWithClaims<Val, Challenge, InputMmcs::Commitment>],
    proof: &CirclePcsProof<Val, Challenge, InputMmcs, FriMmcs, Val>,
) -> Result<ReplayedChallenges<Challenge>, CircleTranscriptFailure>
where
    Val: ComplexExtendable + PrimeField64,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger:
        FieldChallenger<Val> + GrindingChallenger<Witness = Val> + CanObserve<FriMmcs::Commitment>,
{
    // The claims travel flattened, so both sides walk them in one fixed order.
    let claims = rounds
        .iter()
        .flat_map(|CommitmentOpening { matrices: mats, .. }| {
            mats.iter().flat_map(
                |MatrixOpening {
                     points: points_and_values,
                     ..
                 }| {
                    points_and_values.iter().map(
                        |PointOpening {
                             point: zeta,
                             values,
                         }| {
                            (Point::from_projective_line(*zeta), values.as_slice())
                        },
                    )
                },
            )
        });
    let alpha = transcript.batch_phase(claims, proof.batch_pow_witness)?;

    let bivariate_beta = transcript.first_layer(proof.first_layer_commitment.clone());

    // One folding challenge per round, each guarded by its own grinding step.
    let betas = proof
        .fri_proof
        .commit_phase_commits
        .iter()
        .zip(&proof.fri_proof.commit_pow_witnesses)
        .map(|(commit, &witness)| transcript.commit_round(commit.clone(), witness))
        .collect::<Result<Vec<_>, _>>()?;

    // Bind the final constant, re-check the query grind, redraw the indices.
    let indices =
        transcript.query_phase(proof.fri_proof.final_poly, proof.fri_proof.pow_witness)?;

    Ok(ReplayedChallenges {
        alpha,
        bivariate_beta,
        betas,
        indices,
    })
}

impl<Val, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for CirclePcs<Val, InputMmcs, FriMmcs>
where
    Val: ComplexExtendable + PrimeField64,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger:
        FieldChallenger<Val> + GrindingChallenger<Witness = Val> + CanObserve<FriMmcs::Commitment>,
{
    type Domain = CircleDomain<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type Proof = CirclePcsProof<Val, Challenge, InputMmcs, FriMmcs, Challenger::Witness>;
    type Error = FriError<FriMmcs::Error, InputError<InputMmcs::Error, FriMmcs::Error>>;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        CircleDomain::standard(log2_strict_usize(degree))
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let ldes = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert!(
                    domain.log_n >= 2,
                    "CirclePcs cannot commit to a matrix with fewer than 4 rows.",
                    // (because we bivariate fold one bit, and fri needs one more bit)
                );
                CircleEvaluations::from_natural_order(domain, evals)
                    .extrapolate(CircleDomain::standard(
                        domain.log_n + self.fri_params.log_blowup,
                    ))
                    .to_cfft_order()
            })
            .collect_vec();
        let (comm, mmcs_data) = self.mmcs.commit(ldes);
        (comm, mmcs_data)
    }

    fn open(
        &self,
        // For each round,
        rounds: Vec<OpeningRequest<'_, Self::ProverData, Challenge>>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        // Materialize the CFFT-ordered domain points once per committed height. They are shared
        // by the Lagrange denominators and the DEEP-quotient vanishing parts below, which are in
        // turn shared by every matrix opened at the same point on the same domain.
        let mut permuted_points: BTreeMap<usize, Vec<Point<Val>>> = BTreeMap::new();
        debug_span!("materialize domain points").in_scope(|| {
            for OpeningRequest {
                prover_data: data, ..
            } in &rounds
            {
                for mat in self.mmcs.get_matrices(data) {
                    let log_height = log2_strict_usize(mat.height());
                    permuted_points.entry(log_height).or_insert_with(|| {
                        cfft_permute_slice(&CircleDomain::standard(log_height).points_vec())
                    });
                }
            }
        });

        // (log_height, point) -> Lagrange denominators.
        let mut lagrange_dens: Vec<((usize, Challenge), Vec<Challenge>)> = vec![];

        // Open matrices at points
        let values: OpenedValues<Challenge> = rounds
            .iter()
            .map(
                |OpeningRequest {
                     prover_data: data,
                     points: points_for_mats,
                 }| {
                    let mats = self.mmcs.get_matrices(data);
                    debug_assert_eq!(
                        mats.len(),
                        points_for_mats.len(),
                        "Mismatched number of matrices and points"
                    );
                    izip!(mats, points_for_mats)
                        .map(|(mat, points_for_mat)| {
                            let log_height = log2_strict_usize(mat.height());
                            // The committed polynomial has degree below the pre-blow-up domain
                            // size, so its values on a sub-twin-coset of that size determine it.
                            // The first `2^log_sub` rows of the CFFT-ordered LDE are exactly the
                            // CFFT-ordered evaluations over `CircleDomain::new(log_sub, shift)`
                            // (see `eval_at_point_on_subdomain_prefix_matches_full`), so the
                            // out-of-domain evaluation only traverses `1 / blowup` of the matrix.
                            let log_sub = log_height - self.fri_params.log_blowup;
                            let sub_height = 1 << log_sub;
                            let sub_domain = CircleDomain::new(
                                log_sub,
                                CircleDomain::<Val>::standard(log_height).shift,
                            );
                            // It was committed in cfft order.
                            let evals = CircleEvaluations::from_cfft_order(
                                sub_domain,
                                mat.split_rows(sub_height).0,
                            );

                            // Resolve the Lagrange denominators for every point up front.
                            let den_idxs = points_for_mat
                                .iter()
                                .map(|&zeta_uni| {
                                    let key = (log_height, zeta_uni);
                                    lagrange_dens
                                        .iter()
                                        .position(|(k, _)| *k == key)
                                        .unwrap_or_else(|| {
                                            let den = info_span!("compute Lagrange denominators")
                                                .in_scope(|| {
                                                    compute_lagrange_den_batched(
                                                        &permuted_points[&log_height][..sub_height],
                                                        Point::from_projective_line(zeta_uni),
                                                        log_sub,
                                                    )
                                                });
                                            lagrange_dens.push((key, den));
                                            lagrange_dens.len() - 1
                                        })
                                })
                                .collect_vec();

                            let ps_for_points: Vec<Vec<Challenge>> =
                                debug_span!("compute opened values with Lagrange interpolation")
                                    .in_scope(|| match (&points_for_mat[..], &den_idxs[..]) {
                                        // A matrix opened at two points (e.g. zeta and zeta_next)
                                        // is traversed once for both.
                                        (&[zeta_0, zeta_1], &[idx_0, idx_1]) => evals
                                            .evaluate_at_two_points_with_dens(
                                                [
                                                    Point::from_projective_line(zeta_0),
                                                    Point::from_projective_line(zeta_1),
                                                ],
                                                [&lagrange_dens[idx_0].1, &lagrange_dens[idx_1].1],
                                            )
                                            .into(),
                                        _ => izip!(points_for_mat, &den_idxs)
                                            .map(|(&zeta_uni, &den_idx)| {
                                                evals.evaluate_at_point_with_den(
                                                    Point::from_projective_line(zeta_uni),
                                                    &lagrange_dens[den_idx].1,
                                                )
                                            })
                                            .collect(),
                                    });
                            ps_for_points
                        })
                        .collect()
                },
            )
            .collect();
        drop(lagrange_dens);

        // Log-height of the tallest committed low-degree extension.
        //
        // Every number the transcript is described with follows from it.
        // The map above holds one entry per committed height, so the largest key is it.
        let (&log_max_height, _) = permuted_points
            .last_key_value()
            .expect("CirclePcs::open needs at least one committed matrix");

        let folding: CircleFriFoldingForMmcs<Val, Challenge, InputMmcs, FriMmcs> =
            CircleFriFolding(PhantomData);
        // The extra bit re-indexes the first-layer sibling inside a query index.
        let extra_query_index_bits =
            FriFoldingStrategy::<Val, Challenge>::extra_query_index_bits(&folding);

        // Describe the transcript before running it.
        //
        // Every number comes from the parameters and the claims the caller opened.
        // The verifier builds the identical description from the claims it is handed.
        let opened_widths: Vec<Vec<Vec<usize>>> = values
            .iter()
            .map(|mats| {
                mats.iter()
                    .map(|points| points.iter().map(Vec::len).collect())
                    .collect()
            })
            .collect();
        let shape = CirclePcsShape::new(
            &self.fri_params,
            opened_widths,
            log_max_height,
            extra_query_index_bits,
        );
        let mut transcript =
            CircleProverTranscript::<Challenger, Val, Challenge>::new(challenger, shape);

        // Bind every claim, grind, then draw the batching challenge.
        let (alpha, batch_pow_witness) = transcript.batch_phase(izip!(&rounds, &values).flat_map(
            |(
                OpeningRequest {
                    points: points_for_mats,
                    ..
                },
                values_for_mats,
            )| {
                izip!(points_for_mats, values_for_mats).flat_map(|(points, values)| {
                    izip!(points, values).map(|(&zeta, ps_at_zeta)| {
                        (Point::from_projective_line(zeta), ps_at_zeta.as_slice())
                    })
                })
            },
        ));

        /*
        We are reducing columns ("ro" = reduced opening) with powers of alpha:
          ro = .. + α^n c_n + α^(n+1) c_(n+1) + ..
        But we want to precompute small powers of alpha, and batch the columns. So we can do:
          ro = .. + α^n (α^0 c_n + α^1 c_(n+1) + ..) + ..
        reusing the α^0, α^1, etc., then at the end of each column batch we multiply by the α^n.
        (Due to circle stark specifics, we need 2 powers of α for each column, so actually α^(2n)).
        We store this α^(2n), the running reducing factor per log_height, and call it the "alpha offset".
        */

        // log_height -> (alpha offset, reduced openings column)
        let mut reduced_openings: BTreeMap<usize, (Challenge, Vec<Challenge>)> = BTreeMap::new();

        // (log_height, point) -> DEEP-quotient vanishing parts.
        let mut vanishing_parts: Vec<((usize, Challenge), VanishingParts<Challenge>)> = vec![];

        rounds.iter().zip(values.iter()).for_each(
            |(
                OpeningRequest {
                    prover_data: data,
                    points: points_for_mats,
                },
                values,
            )| {
                let mats = self.mmcs.get_matrices(data);
                izip!(mats, points_for_mats, values).for_each(|(mat, points_for_mat, values)| {
                    let log_height = log2_strict_usize(mat.height());
                    let log_sub = log_height - self.fri_params.log_blowup;

                    let (alpha_offset, reduced_opening_for_log_height) = reduced_openings
                        .entry(log_height)
                        .or_insert_with(|| (Challenge::ONE, Challenge::zero_vec(1 << log_height)));

                    // The lift below costs a single-column CFFT extrapolation, which is
                    // latency-bound rather than bandwidth-bound: it costs about as much as
                    // the half-traversal of a ~1000-column matrix it saves, so it only pays
                    // off for matrices substantially wider than that.
                    const LIFT_MIN_WIDTH: usize = 1024;

                    // The only pass over the matrix; it does not depend on the opening points.
                    // The reduced column lies in the pre-blow-up polynomial space, so it is
                    // determined by the trace-size subdomain prefix (committed in cfft order):
                    // reduce the prefix and lift it back with a narrow CFFT instead of
                    // traversing the full LDE.
                    let reduced_rows = if log_sub > 0 && mat.width() >= LIFT_MIN_WIDTH {
                        let sub_domain = CircleDomain::new(
                            log_sub,
                            CircleDomain::<Val>::standard(log_height).shift,
                        );
                        CircleEvaluations::from_cfft_order(
                            sub_domain,
                            mat.split_rows(1 << log_sub).0,
                        )
                        .rowwise_alpha_reduce_lifted(alpha, CircleDomain::standard(log_height))
                    } else {
                        CircleEvaluations::from_cfft_order(
                            CircleDomain::standard(log_height),
                            mat.as_view(),
                        )
                        .rowwise_alpha_reduce(alpha)
                    };
                    let alpha_pow_width = alpha.exp_u64(mat.width() as u64);

                    points_for_mat
                        .iter()
                        .zip(values.iter())
                        .for_each(|(&zeta_uni, ps_at_zeta)| {
                            let zeta = Point::from_projective_line(zeta_uni);
                            let key = (log_height, zeta_uni);
                            let vp_idx = vanishing_parts
                                .iter()
                                .position(|(k, _)| *k == key)
                                .unwrap_or_else(|| {
                                    let vp = compute_vanishing_parts(
                                        &permuted_points[&log_height],
                                        zeta,
                                    );
                                    vanishing_parts.push((key, vp));
                                    vanishing_parts.len() - 1
                                });

                            // sum_j(alpha^j * p_j[zeta]), the same for all rows.
                            let reduced_ps_at_zeta: Challenge =
                                dot_product(alpha.powers(), ps_at_zeta.iter().copied());

                            // Reduce this matrix, as a deep quotient, into the running
                            // reduction, offset by alpha_offset.
                            accumulate_deep_quotient(
                                reduced_opening_for_log_height,
                                *alpha_offset,
                                alpha_pow_width,
                                &reduced_rows,
                                &vanishing_parts[vp_idx].1,
                                reduced_ps_at_zeta,
                            );

                            // Update alpha_offset from α^i -> α^(i + 2 * width)
                            *alpha_offset *= alpha_pow_width.square();
                        });
                });
            },
        );
        drop(vanishing_parts);

        // Iterate over our reduced columns and extract lambda - the multiple of the vanishing polynomial
        // which may appear in the reduced quotient due to CFFT dimension gap.

        let mut lambdas = vec![];
        let mut log_heights = vec![];
        let first_layer_mats: Vec<RowMajorMatrix<Challenge>> = reduced_openings
            .into_iter()
            .map(|(log_height, (_, mut ro))| {
                assert!(log_height > 0);
                log_heights.push(log_height);
                let lambda = extract_lambda(&mut ro, self.fri_params.log_blowup);
                lambdas.push(lambda);
                // Prepare for first layer fold with 2 siblings per leaf.
                RowMajorMatrix::new(ro, 2)
            })
            .collect();
        debug_assert_eq!(
            log_heights.iter().max().copied(),
            Some(log_max_height),
            "the described height and the committed heights must agree"
        );

        // Commit to reduced openings at each log_height, so we can challenge a global
        // folding factor for all first layers, which we use for a "manual" (not part of p3-fri) fold.
        // This is necessary because the first layer of folding uses different twiddles, so it's easiest
        // to do it here, before p3-fri.

        let (first_layer_commitment, first_layer_data) =
            self.fri_params.mmcs.commit(first_layer_mats);
        let bivariate_beta = transcript.first_layer(first_layer_commitment.clone());

        // Fold all first layers at bivariate_beta.

        let fri_input: Vec<Vec<Challenge>> = self
            .fri_params
            .mmcs
            .get_matrices(&first_layer_data)
            .into_iter()
            .map(|m| fold_y(bivariate_beta, m))
            // Reverse, because FRI expects descending by height
            .rev()
            .collect();

        let fri_proof = prove(
            &folding,
            &self.fri_params,
            fri_input,
            &mut transcript,
            |indices| {
                // CircleFriFolder asks for an extra query index bit, so we use that here to index
                // the first layer fold.

                // Open the input (big opening, lots of columns) at every full index. Queries into
                // one committed tree share a single proof, so overlapping paths ship once.
                let input_openings = rounds
                    .iter()
                    .map(
                        |OpeningRequest {
                             prover_data: data, ..
                         }| {
                            let log_max_batch_height =
                                log2_strict_usize(self.mmcs.get_max_height(data));
                            let bits_reduced = log_max_height - log_max_batch_height;
                            let reduced_indices: Vec<usize> =
                                indices.iter().map(|&index| index >> bits_reduced).collect();
                            let (opened_values, opening_proof) =
                                self.mmcs.open_multi_batch(&reduced_indices, data);
                            BatchMultiOpening {
                                opened_values,
                                opening_proof,
                            }
                        },
                    )
                    .collect();

                // We committed to first_layer in pairs, so open the reduced index and include the sibling
                // as part of the input proof.
                let paired_indices: Vec<usize> = indices.iter().map(|&index| index >> 1).collect();
                let (first_layer_values, first_layer_proof) = self
                    .fri_params
                    .mmcs
                    .open_multi_batch(&paired_indices, &first_layer_data);
                let first_layer_siblings = izip!(indices, first_layer_values)
                    .map(|(&index, values)| {
                        izip!(&values, &log_heights)
                            .map(|(v, log_height)| {
                                let reduced_index = index >> (log_max_height - log_height);
                                let sibling_index = (reduced_index & 1) ^ 1;
                                v[sibling_index]
                            })
                            .collect()
                    })
                    .collect();
                CircleInputProof {
                    input_openings,
                    first_layer_siblings,
                    first_layer_proof,
                }
            },
        );

        // Every described step has now been played.
        transcript.finish();

        (
            values,
            CirclePcsProof {
                batch_pow_witness: batch_pow_witness.unwrap_or_default(),
                first_layer_commitment,
                lambdas,
                fri_proof,
            },
        )
    }

    fn verify(
        &self,
        // For each round:
        rounds: Vec<CommitmentOpening<Challenge, Self::Commitment, Self::Domain>>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let folding: CircleFriFoldingForMmcs<Val, Challenge, InputMmcs, FriMmcs> =
            CircleFriFolding(PhantomData);
        // The extra bit re-indexes the first-layer sibling inside a query index.
        let extra_query_index_bits =
            FriFoldingStrategy::<Val, Challenge>::extra_query_index_bits(&folding);

        // The tallest claimed low-degree extension fixes every derived number.
        //
        //     H_claim = max claimed log_n + log_blowup
        //
        // Nothing below reads a height from the proof.
        // A forged round count therefore cannot widen a query index.
        // Nor can it shorten the fold chain.
        let log_global_max_height = rounds
            .iter()
            .flat_map(|CommitmentOpening { matrices: mats, .. }| {
                mats.iter()
                    .map(|MatrixOpening { domain, .. }| domain.log_n + self.fri_params.log_blowup)
            })
            .max()
            .ok_or(FriError::NoCommittedMatrices)?;

        // The bivariate layer folds one bit before FRI folds anything.
        let Some(num_commit_rounds) =
            log_global_max_height.checked_sub(self.fri_params.log_blowup + 1)
        else {
            return Err(FriError::InputError(InputError::DomainTooSmall {
                log_n: log_global_max_height - self.fri_params.log_blowup,
            }));
        };

        // Invariant: the query-index width fits the circle group of order 2^CIRCLE_TWO_ADICITY.
        //
        //     index_bits  = H_claim - 1 + extra_query_index_bits
        //     field order = 2^CIRCLE_TWO_ADICITY - 1   (one short of the group order)
        //     => a width of CIRCLE_TWO_ADICITY bits is unsampleable
        let index_bits = log_global_max_height - 1 + extra_query_index_bits;
        if index_bits >= Val::CIRCLE_TWO_ADICITY {
            return Err(FriError::GlobalMaxHeightTooLarge {
                log_global_max_height: index_bits,
                two_adicity: Val::CIRCLE_TWO_ADICITY,
            });
        }

        // Every length the transcript is described with is checked before it is seeded.
        let log_arities =
            validate_proof_shape(&self.fri_params, &proof.fri_proof, num_commit_rounds)?;

        // Describe the transcript from the claims, exactly as the prover described it.
        let opened_widths: Vec<Vec<Vec<usize>>> = rounds
            .iter()
            .map(|CommitmentOpening { matrices: mats, .. }| {
                mats.iter()
                    .map(
                        |MatrixOpening {
                             points: points_and_values,
                             ..
                         }| {
                            points_and_values.iter().map(|p| p.values.len()).collect()
                        },
                    )
                    .collect()
            })
            .collect();
        let shape = CirclePcsShape::new(
            &self.fri_params,
            opened_widths,
            log_global_max_height,
            extra_query_index_bits,
        );
        let mut transcript =
            CircleVerifierTranscript::<Challenger, Val, Challenge>::new(challenger, shape);

        // Invariant: the driver is released on both exits of this span.
        //
        //     Ok  -> the driver is closed, which runs its completeness check
        //     Err -> that check is released, so the rejection travels alone
        //
        // Nothing between the two branches can return early.
        // No path therefore drops a driver that is still mid-pattern.
        let ReplayedChallenges {
            alpha,
            bivariate_beta,
            betas,
            indices,
        } = match replay_transcript(&mut transcript, &rounds, proof) {
            Ok(challenges) => {
                transcript.finish();
                challenges
            }
            Err(failure) => {
                transcript.abort();
                return Err(FriError::InvalidPowWitness(failure.phase()));
            }
        };

        // Per (batch, matrix) `alpha^width` and `alpha^(2*width)`, plus a shared table of
        // `alpha`'s powers up to the widest matrix. A matrix's width is fixed by the
        // verifier's own claimed evaluations (`rounds`), independent of the query, so
        // computing these once here replaces recomputing them on every (query, matrix)
        // or (query, matrix, point) inside the per-query closure below.
        let matrix_alpha_pows: Vec<Vec<(Challenge, Challenge)>> = rounds
            .iter()
            .map(|CommitmentOpening { matrices: mats, .. }| {
                mats.iter()
                    .map(
                        |MatrixOpening {
                             points: points_and_values,
                             ..
                         }| {
                            let width = points_and_values
                                .first()
                                .map_or(0, |PointOpening { values: v, .. }| v.len());
                            let alpha_pow_width = alpha.exp_u64(width as u64);
                            (alpha_pow_width, alpha_pow_width.square())
                        },
                    )
                    .collect()
            })
            .collect();
        let max_width = rounds
            .iter()
            .flat_map(|CommitmentOpening { matrices: mats, .. }| mats.iter())
            .flat_map(
                |MatrixOpening {
                     points: points_and_values,
                     ..
                 }| {
                    points_and_values
                        .iter()
                        .map(|PointOpening { values: v, .. }| v.len())
                },
            )
            .max()
            .unwrap_or(0);
        let alpha_powers: Vec<Challenge> = alpha.powers().collect_n(max_width);

        verify_queries(
            &folding,
            &self.fri_params,
            &proof.fri_proof,
            &betas,
            &indices,
            &log_arities,
            |indices, input_proof| {
                let CircleInputProof {
                    input_openings,
                    first_layer_siblings,
                    first_layer_proof,
                } = input_proof;

                // One sibling set per query, one opened-row set per query per batch.
                if first_layer_siblings.len() != indices.len() {
                    return Err(InputError::InputShapeError);
                }
                for batch_opening in input_openings {
                    if batch_opening.opened_values.len() != indices.len() {
                        return Err(InputError::InputShapeError);
                    }
                }

                // Check every input commitment's shared multi-opening once, before the
                // per-query arithmetic reads any opened value.
                for (
                    batch,
                    (
                        batch_opening,
                        CommitmentOpening {
                            commitment: batch_commit,
                            matrices: mats,
                        },
                    ),
                ) in zip_eq(input_openings, &rounds, InputError::InputShapeError)?.enumerate()
                {
                    let batch_heights: Vec<usize> = mats
                        .iter()
                        .map(|MatrixOpening { domain, .. }| {
                            domain.size() << self.fri_params.log_blowup
                        })
                        .collect_vec();
                    // The opened rows must pair one-to-one with the committed matrices.
                    for opened_values in &batch_opening.opened_values {
                        if opened_values.len() != mats.len() {
                            return Err(InputError::InputShapeError);
                        }
                    }
                    let batch_dims: Vec<Dimensions> = batch_heights
                        .iter()
                        .zip(mats)
                        .enumerate()
                        .map(
                            |(
                                matrix,
                                (
                                    &height,
                                    MatrixOpening {
                                        points: points_and_values,
                                        ..
                                    },
                                ),
                            )| {
                                // Invariant: a matrix's width is fixed by its first opening point.
                                //
                                //     >= 1 point  ->  width = number of claimed evaluations
                                //     no points   ->  reject
                                //
                                // Why reject the no-points case:
                                //   - row boundaries in the flattened leaf hash are authenticated only from claimed widths
                                //   - a matrix opened at no points claims no width
                                //   - its width could then come only from the unverified proof
                                let PointOpening { values, .. } = points_and_values.first().ok_or(
                                    InputError::MatrixWithoutOpeningPoints { batch, matrix },
                                )?;
                                Ok(Dimensions {
                                    width: values.len(),
                                    height,
                                })
                            },
                        )
                        .collect::<Result<Vec<_>, _>>()?;

                    let (dims, reduced_indices) = batch_heights
                        .iter()
                        .max()
                        .map(|x| log2_strict_usize(*x))
                        .map_or_else(
                            ||
                            // Empty batch?
                            (&[][..], vec![0; indices.len()]),
                            |log_batch_max_height| {
                                let bits_reduced = log_global_max_height - log_batch_max_height;
                                (
                                    &batch_dims[..],
                                    indices.iter().map(|&i| i >> bits_reduced).collect_vec(),
                                )
                            },
                        );

                    self.mmcs
                        .verify_multi_batch(
                            batch_commit,
                            dims,
                            &reduced_indices,
                            &batch_opening.opened_values,
                            &batch_opening.opening_proof,
                        )
                        .map_err(InputError::InputMmcsError)?;
                }

                // Per query, reduce the (now authenticated) opened rows into the FRI inputs
                // and rebuild the first-layer leaves that the shared proof will authenticate.
                let mut all_fri_inputs = Vec::with_capacity(indices.len());
                let mut fl_leaves_by_query = Vec::with_capacity(indices.len());
                let mut fl_dims: Vec<Dimensions> = Vec::new();

                for (query, &index) in indices.iter().enumerate() {
                    // log_height -> (alpha_offset, ro)
                    let mut reduced_openings = BTreeMap::new();

                    for (batch, (batch_opening, CommitmentOpening { matrices: mats, .. })) in
                        zip_eq(input_openings, &rounds, InputError::InputShapeError)?.enumerate()
                    {
                        for (
                            matrix,
                            (
                                ps_at_x,
                                MatrixOpening {
                                    domain: mat_domain,
                                    points: mat_points_and_values,
                                },
                            ),
                        ) in zip_eq(
                            &batch_opening.opened_values[query],
                            mats,
                            InputError::InputShapeError,
                        )?
                        .enumerate()
                        {
                            let log_height = mat_domain.log_n + self.fri_params.log_blowup;
                            let bits_reduced = log_global_max_height - log_height;
                            let orig_idx = cfft_permute_index(index >> bits_reduced, log_height);

                            let committed_domain = CircleDomain::standard(log_height);
                            let x = committed_domain.nth_point(orig_idx);

                            let (alpha_offset, ro) = reduced_openings
                                .entry(log_height)
                                .or_insert((Challenge::ONE, Challenge::ZERO));
                            let (alpha_pow_width, alpha_pow_width_2) =
                                matrix_alpha_pows[batch][matrix];

                            for PointOpening {
                                point: zeta_uni,
                                values: ps_at_zeta,
                            } in mat_points_and_values
                            {
                                // The claimed opening must have exactly as many
                                // values as the committed row has columns.
                                if ps_at_zeta.len() != ps_at_x.len() {
                                    return Err(InputError::InputShapeError);
                                }
                                let zeta = Point::from_projective_line(*zeta_uni);

                                // A vanishing denominator means this opening point lands on the
                                // query point; reject the proof rather than dividing by zero.
                                *ro += *alpha_offset
                                    * deep_quotient_reduce_row(
                                        alpha_pow_width,
                                        &alpha_powers,
                                        x,
                                        zeta,
                                        ps_at_x,
                                        ps_at_zeta,
                                    )
                                    .ok_or(InputError::OpeningPointMatchesQueryPoint)?;

                                *alpha_offset *= alpha_pow_width_2;
                            }
                        }
                    }

                    // Verify bivariate fold and lambda correction

                    // First pass: derive the lambda-corrected leaf values and each height's
                    // first-layer (y) twiddle, without folding yet. The fold pairs a point with
                    // its negation, so the canonical (b=0) twiddle is `p.y` (sign-flipped when
                    // this query landed on the b=1 member) - the same point `p` already computed
                    // for the lambda correction, with no separate `nth_y_twiddle` scalar
                    // multiplication. All these per-height twiddles are then inverted in a single
                    // batch instead of one inversion per height.
                    let per_height: Vec<_> = zip_eq(
                        zip_eq(
                            reduced_openings,
                            &first_layer_siblings[query],
                            InputError::InputShapeError,
                        )?,
                        &proof.lambdas,
                        InputError::InputShapeError,
                    )?
                    .map(|(((log_height, (_, ro)), &fl_sib), &lambda)| {
                        assert!(log_height > 0);

                        let orig_size = log_height - self.fri_params.log_blowup;
                        let bits_reduced = log_global_max_height - log_height;
                        let b = (index >> bits_reduced) & 1;
                        let orig_idx = cfft_permute_index(index >> bits_reduced, log_height);

                        let lde_domain = CircleDomain::standard(log_height);
                        let p: Point<Val> = lde_domain.nth_point(orig_idx);

                        let lambda_corrected = ro - lambda * p.v_n(orig_size);

                        let mut fl_values = vec![lambda_corrected; 2];
                        fl_values[b ^ 1] = fl_sib;

                        let y_twiddle = if b == 0 { p.y } else { -p.y };

                        let dims = Dimensions {
                            // First-layer leaves hold the queried value and its sibling.
                            width: 2,
                            height: 1 << (log_height - 1),
                        };

                        (log_height, y_twiddle, fl_values, dims)
                    })
                    .collect();

                    let y_twiddles_inv = batch_multiplicative_inverse(
                        &per_height.iter().map(|&(_, t, _, _)| t).collect_vec(),
                    );

                    let (mut fri_input, query_fl_dims, fl_leaves): (Vec<_>, Vec<_>, Vec<_>) =
                        per_height
                            .into_iter()
                            .zip(y_twiddles_inv)
                            .map(|((log_height, _, fl_values, dims), y_twiddle_inv)| {
                                let fri_input = (
                                    // - 1 here is because we have already folded a layer.
                                    log_height - 1,
                                    fold_row_with_inv_twiddle(
                                        y_twiddle_inv,
                                        bivariate_beta,
                                        fl_values.iter().copied(),
                                    ),
                                );
                                (fri_input, dims, fl_values)
                            })
                            .multiunzip();

                    // sort descending
                    fri_input.reverse();

                    // The committed first-layer shape is the same for every query.
                    if query == 0 {
                        fl_dims = query_fl_dims;
                    }

                    all_fri_inputs.push(fri_input);
                    fl_leaves_by_query.push(fl_leaves);
                }

                // One shared check for every query's first-layer row.
                let paired_indices = indices.iter().map(|&i| i >> 1).collect_vec();
                self.fri_params
                    .mmcs
                    .verify_multi_batch(
                        &proof.first_layer_commitment,
                        &fl_dims,
                        &paired_indices,
                        &fl_leaves_by_query,
                        first_layer_proof,
                    )
                    .map_err(InputError::FirstLayerMmcsError)?;

                Ok(all_fri_inputs)
            },
        )
    }
}

impl<Val, InputMmcs, FriMmcs, Challenge, Challenger> UnivariateStarkPcs<Challenge, Challenger>
    for CirclePcs<Val, InputMmcs, FriMmcs>
where
    Val: ComplexExtendable + PrimeField64,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger:
        FieldChallenger<Val> + GrindingChallenger<Witness = Val> + CanObserve<FriMmcs::Commitment>,
{
    type EvaluationsOnDomain<'a> = RowIndexMappedView<CfftPerm, RowMajorMatrixCow<'a, Val>>;

    const ZK: bool = false;

    fn log_max_lde_height(&self) -> usize {
        Val::CIRCLE_TWO_ADICITY - 1
    }

    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
        _num_chunks: usize,
    ) -> Vec<RowMajorMatrix<Val>> {
        evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert!(
                    domain.log_n >= 2,
                    "CirclePcs cannot commit to a matrix with fewer than 4 rows.",
                    // (because we bivariate fold one bit, and fri needs one more bit)
                );
                CircleEvaluations::from_natural_order(domain, evals)
                    .extrapolate(CircleDomain::standard(
                        domain.log_n + self.fri_params.log_blowup,
                    ))
                    .to_cfft_order()
            })
            .collect_vec()
    }

    fn commit_ldes(&self, ldes: Vec<RowMajorMatrix<Val>>) -> (Self::Commitment, Self::ProverData) {
        self.mmcs.commit(ldes)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let mat = self.mmcs.get_matrices(data)[idx].as_view();
        let committed_domain = CircleDomain::standard(log2_strict_usize(mat.height()));
        if domain == committed_domain {
            mat.as_cow().cfft_perm_rows()
        } else {
            // The committed matrix is the LDE of a polynomial of `committed_domain.log_n -
            // log_blowup` coefficients. The first `2^log_sub` CFFT-ordered rows of the LDE
            // are exactly the CFFT-ordered evaluations over the smaller `sub_domain` of that
            // size (see `eval_at_point_on_subdomain_prefix_matches_full`), so interpolating
            // that prefix instead of the full committed matrix recovers the same coefficients
            // at `1 / blowup` of the CFFT work. This also lets `domain` be smaller than the
            // committed LDE (e.g. a quotient domain when `log_blowup` exceeds the quotient
            // degree), which `extrapolate` would reject.
            let log_sub = committed_domain.log_n - self.fri_params.log_blowup;
            let sub_domain = CircleDomain::new(log_sub, committed_domain.shift);
            let coeffs =
                CircleEvaluations::from_cfft_order(sub_domain, mat.split_rows(1 << log_sub).0)
                    .interpolate();
            CircleEvaluations::evaluate(domain, coeffs)
                .to_cfft_order()
                .as_cow()
                .cfft_perm_rows()
        }
    }

    fn build_periodic_lde_table(
        &self,
        periodic_cols: &[Vec<Val>],
        trace_domain: Self::Domain,
        quotient_domain: Self::Domain,
    ) -> PeriodicLdeTable<Val> {
        build_periodic_lde_table_circle(periodic_cols, &trace_domain, &quotient_domain)
    }
}

#[cfg(test)]
mod tests {
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_commit::ExtensionMmcs;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_fri::FriParameters;
    use p3_fri::verifier::{FriError, PowPhase};
    use p3_keccak::Keccak256Hash;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_mersenne_31::Mersenne31;
    use p3_symmetric::{CompressionFunctionFromHasher, MerkleCap, SerializingHasher};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type Val = Mersenne31;
    type Challenge = BinomialExtensionField<Mersenne31, 3>;
    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
    type TestPcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
    type TestError = FriError<
        <ChallengeMmcs as Mmcs<Challenge>>::Error,
        InputError<<ValMmcs as Mmcs<Val>>::Error, <ChallengeMmcs as Mmcs<Challenge>>::Error>,
    >;

    /// The benchmark parameter preset must satisfy the constructor's own guard.
    ///
    /// It reaches that constructor from `p3-examples` and from `monolith-air`'s
    /// benchmark, so a nonzero `batch_proof_of_work_bits` there turns both into
    /// a runtime panic. Nothing else catches it: every other site in the tree
    /// builds this PCS by struct literal, which bypasses the guard entirely,
    /// and examples and benchmarks are compiled but never run by `cargo test`.
    #[test]
    fn benchmark_fri_parameters_are_accepted_by_new() {
        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs::new(field_hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        // Panics if the guard rejects these parameters.
        let _ = TestPcs::new(val_mmcs, FriParameters::new_benchmark(challenge_mmcs));
    }

    /// Build a valid Circle PCS proof for a random single-column trace.
    ///
    /// Returns all the pieces needed to verify (or re-verify after mutation):
    /// the PCS instance, hasher seed, commitment, domain, evaluation point,
    /// opened values, and the proof itself.
    ///
    /// # Fixture parameters
    ///
    /// - Trace: 2^{10} = 1024 rows, 1 column of random field elements.
    /// - FRI: testing parameters with log_blowup = 2, log_final_poly_len = 0.
    /// - Hash: Keccak-256 with a binary Merkle tree.
    #[allow(clippy::type_complexity)]
    fn setup_valid_proof(
        batch_pow_bits: usize,
    ) -> (
        TestPcs,
        ByteHash,
        <ValMmcs as Mmcs<Val>>::Commitment,
        CircleDomain<Val>,
        Challenge,
        Vec<Vec<Vec<Vec<Challenge>>>>,
        CirclePcsProof<Val, Challenge, ValMmcs, ChallengeMmcs, Val>,
    ) {
        setup_valid_proof_at(batch_pow_bits, 1, 1)
    }

    /// The same fixture at caller-chosen commit- and query-phase difficulties.
    ///
    /// Both numbers reach the transcript's description, so prover and verifier must be
    /// built from the same pair for the replay to line up.
    #[allow(clippy::type_complexity)]
    fn setup_valid_proof_at(
        batch_pow_bits: usize,
        commit_pow_bits: usize,
        query_pow_bits: usize,
    ) -> (
        TestPcs,
        ByteHash,
        <ValMmcs as Mmcs<Val>>::Commitment,
        CircleDomain<Val>,
        Challenge,
        Vec<Vec<Vec<Vec<Challenge>>>>,
        CirclePcsProof<Val, Challenge, ValMmcs, ChallengeMmcs, Val>,
    ) {
        let mut rng = SmallRng::seed_from_u64(0);

        // Build the hash stack: field hasher → compression → Merkle tree.
        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs::new(field_hash, compress, 0);

        // Wrap the value-domain Merkle tree for extension-field leaves.
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        // Minimal FRI parameters for fast test execution.
        let mut fri_params = FriParameters::new_testing(challenge_mmcs, 0);
        fri_params.batch_proof_of_work_bits = batch_pow_bits;
        fri_params.commit_proof_of_work_bits = commit_pow_bits;
        fri_params.query_proof_of_work_bits = query_pow_bits;

        let pcs = TestPcs {
            mmcs: val_mmcs,
            fri_params,
            _phantom: PhantomData,
        };

        // Generate a random trace on a circle domain of size 2^{10}.
        let log_n = 10;
        let d =
            <TestPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_n);

        let evals = RowMajorMatrix::rand(&mut rng, 1 << log_n, 1);

        // Commit to the trace and produce the Merkle root.
        let (comm, data) = <TestPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(d, evals)]);

        // Random evaluation point in the extension field.
        let zeta: Challenge = rng.random();

        // Generate the opening proof at the chosen evaluation point.
        let mut chal = Challenger::from_hasher(vec![], byte_hash);
        let (values, proof) = pcs.open(
            vec![OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut chal,
        );

        (pcs, byte_hash, comm, d, zeta, values, proof)
    }

    /// Run the PCS verifier with the given proof and return the result.
    ///
    /// This is a thin wrapper that reconstructs a fresh challenger and
    /// calls the verification routine. Tests use it to verify both valid
    /// proofs and intentionally malformed ones.
    fn try_verify(
        pcs: &TestPcs,
        byte_hash: ByteHash,
        comm: &<ValMmcs as Mmcs<Val>>::Commitment,
        d: CircleDomain<Val>,
        zeta: Challenge,
        values: &[Vec<Vec<Vec<Challenge>>>],
        proof: &CirclePcsProof<Val, Challenge, ValMmcs, ChallengeMmcs, Val>,
    ) -> Result<(), TestError> {
        // Build a fresh challenger from the same seed so the transcript
        // replays identically to what the prover produced.
        let mut chal = Challenger::from_hasher(vec![], byte_hash);
        pcs.verify(
            vec![
                (
                    comm.clone(),
                    vec![(d, vec![(zeta, values[0][0][0].clone())])],
                )
                    .into(),
            ],
            proof,
            &mut chal,
        )
    }

    #[test]
    fn circle_pcs() {
        // Smoke test: an honestly generated proof must verify successfully.
        let (pcs, byte_hash, comm, d, zeta, values, proof) = setup_valid_proof(0);
        try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof).expect("verify err");
    }

    #[test]
    fn batch_grinding_proves_and_verifies() {
        let (pcs, byte_hash, comm, d, zeta, values, proof) = setup_valid_proof(8);
        try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof).unwrap();
    }

    #[test]
    fn invalid_batch_grinding_witness_is_rejected_at_the_batch_phase() {
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(8);
        proof.batch_pow_witness = Val::ZERO;
        assert!(matches!(
            try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof),
            Err(FriError::InvalidPowWitness(
                p3_fri::verifier::PowPhase::Batch
            ))
        ));
    }

    #[test]
    fn changing_batch_grinding_difficulty_rejects_the_proof() {
        for (prover_bits, verifier_bits) in [(0, 8), (8, 0), (8, 9)] {
            let (mut pcs, byte_hash, comm, d, zeta, values, proof) = setup_valid_proof(prover_bits);
            pcs.fri_params.batch_proof_of_work_bits = verifier_bits;
            assert!(try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof).is_err());
        }
    }

    #[test]
    fn batch_grinding_binds_the_opening_claims() {
        let (pcs, byte_hash, comm, d, zeta, mut values, proof) = setup_valid_proof(8);
        values[0][0][0][0] += Challenge::ONE;
        assert!(matches!(
            try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof),
            Err(FriError::InvalidPowWitness(
                p3_fri::verifier::PowPhase::Batch
            ))
        ));
    }

    #[test]
    fn reject_matrix_without_opening_points() {
        // Invariant: every input matrix must be opened at >= 1 point.
        //
        //     no points  ->  no claimed width  ->  width would come from the proof  ->  reject
        //
        // A matrix opened at no points observes nothing into the challenger.
        // This holds identically on the proving side and the verifying side.
        //
        // Fixture state: one batch of two matrices sharing a domain.
        //   - matrix 0 is opened at one point, keeping the reduced openings non-empty
        //   - matrix 1 is opened at no points
        //
        // Flow:
        //   - both sides observe only matrix 0  ->  proof-of-work challenge matches
        //   - the query phase verifies the input opening  ->  matrix 1 rejected
        let mut rng = SmallRng::seed_from_u64(0);

        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs::new(field_hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let fri_params = FriParameters::new_testing(challenge_mmcs, 0);
        let pcs = TestPcs {
            mmcs: val_mmcs,
            fri_params,
            _phantom: PhantomData,
        };

        // One batch, two single-column matrices sharing a domain of 2^{10} rows.
        let log_n = 10;
        let d =
            <TestPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_n);
        let evals_0 = RowMajorMatrix::rand(&mut rng, 1 << log_n, 1);
        let evals_1 = RowMajorMatrix::rand(&mut rng, 1 << log_n, 1);
        let (comm, data) =
            <TestPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(d, evals_0), (d, evals_1)]);

        // Prove: open matrix 0 at one point, matrix 1 at no points.
        let zeta: Challenge = rng.random();
        let mut chal = Challenger::from_hasher(vec![], byte_hash);
        let (values, proof) = pcs.open(
            vec![OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta], vec![]],
            }],
            &mut chal,
        );

        // Verify with the same shape: matrix 1 carries no opening points.
        let mut chal = Challenger::from_hasher(vec![], byte_hash);
        let err = pcs
            .verify(
                vec![
                    (
                        comm,
                        vec![(d, vec![(zeta, values[0][0][0].clone())]), (d, vec![])],
                    )
                        .into(),
                ],
                &proof,
                &mut chal,
            )
            .expect_err("matrix without opening points must be rejected");

        // The offending matrix is identified by its batch and matrix index.
        let FriError::InputError(InputError::MatrixWithoutOpeningPoints { batch, matrix }) = err
        else {
            panic!("expected MatrixWithoutOpeningPoints, got {err:?}");
        };
        assert_eq!(batch, 0);
        assert_eq!(matrix, 1);
    }

    #[test]
    fn get_evaluations_on_domain_matches_direct_lde() {
        // `get_evaluations_on_domain` must return the committed trace on the requested
        // domain whether that domain is smaller than, equal to, or larger than the
        // committed LDE. The smaller-than case is exercised whenever `log_blowup`
        // exceeds the quotient degree (e.g. the quotient domain in uni-stark).
        let mut rng = SmallRng::seed_from_u64(1);

        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs::new(field_hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        // log_blowup = 2 makes the committed LDE larger than a quotient-sized domain.
        let mut fri_params = FriParameters::new_testing(challenge_mmcs, 0);
        fri_params.log_blowup = 2;

        let pcs = TestPcs {
            mmcs: val_mmcs,
            fri_params,
            _phantom: PhantomData,
        };

        let log_n = 8;
        let width = 3;
        let d =
            <TestPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_n);
        let evals = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_n, width);

        let (_comm, data) =
            <TestPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(d, evals.clone())]);

        // The committed LDE lives on `standard(log_n + 2)`. Walk a target domain from the
        // original degree up past the committed LDE: `log_n + 1` is the smaller-than case,
        // `log_n + 2` hits the equal fast path, and `log_n + 3` is the larger-than case.
        for target_log_n in [log_n, log_n + 1, log_n + 2, log_n + 3] {
            let target = CircleDomain::standard(target_log_n);
            let got =
                <TestPcs as UnivariateStarkPcs<Challenge, Challenger>>::get_evaluations_on_domain(
                    &pcs, &data, 0, target,
                )
                .to_row_major_matrix();

            // Ground truth: extrapolate the original trace straight onto `target`.
            let expected = CircleEvaluations::from_natural_order(d, evals.clone())
                .extrapolate(target)
                .to_natural_order()
                .to_row_major_matrix();

            assert_eq!(got, expected, "mismatch for target_log_n = {target_log_n}");
        }
    }

    #[test]
    fn reject_commit_phase_query_count_mismatch() {
        // Invariant: every commit-phase round must open every query. The round's
        // shared proof carries one sibling set per query.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(0);

        // Mutation: drop one query's siblings from round 0.
        //
        //     before: sibling_values = [s_0, s_1, ..., s_{n-1}]   (n = num_queries)
        //     after:  sibling_values = [s_0, s_1, ..., s_{n-2}]   (n - 1)
        //     → expected n, got n - 1 → error on round 0
        proof.fri_proof.commit_phase_openings[0]
            .sibling_values
            .pop();

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("expected CommitPhaseQueryCountMismatch");

        // Destructure for precise field assertions (better diagnostics than matches!).
        let FriError::CommitPhaseQueryCountMismatch {
            round,
            expected,
            got,
        } = err
        else {
            panic!("expected CommitPhaseQueryCountMismatch, got {err:?}");
        };
        assert_eq!(round, 0);
        assert_eq!(expected, pcs.fri_params.num_queries);
        assert_eq!(got, pcs.fri_params.num_queries - 1);
    }

    #[test]
    fn reject_zero_queries() {
        // Invariant: a zero-query instance performs no low-degree spot checks.
        // The per-query loop never runs.
        // Without the guard any final polynomial would verify.
        //
        // Fixture state: an honest proof built with the testing query count.
        //
        // Mutation: verify it under params with num_queries = 0.
        let (mut pcs, byte_hash, comm, d, zeta, values, proof) = setup_valid_proof(0);
        pcs.fri_params.num_queries = 0;

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("zero-query instance must be rejected");

        assert!(
            matches!(err, FriError::ZeroQueries),
            "expected ZeroQueries, got {err:?}"
        );
    }

    #[test]
    fn reject_zero_blowup() {
        // Invariant: at log_blowup = 0 every length-N word is itself a degree-<N codeword, so
        // the folding chain and final-polynomial check are satisfied by an arbitrary
        // reduced-opening word.
        //
        // Fixture state: an honest proof built with log_blowup >= 1.
        //
        // Mutation: verify it under params with log_blowup = 0.
        let (mut pcs, byte_hash, comm, d, zeta, values, proof) = setup_valid_proof(0);
        pcs.fri_params.log_blowup = 0;

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("zero-blowup instance must be rejected");

        assert!(
            matches!(err, FriError::ZeroBlowup),
            "expected ZeroBlowup, got {err:?}"
        );
    }

    #[test]
    #[should_panic(expected = "num_queries must be at least 1")]
    fn prover_rejects_zero_queries() {
        // The prover must refuse to build a vacuous proof.
        // The verifier guards the same config, so the failure is symmetric.
        let mut rng = SmallRng::seed_from_u64(0);

        // Build the hash stack: field hasher → compression → Merkle tree.
        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs::new(field_hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        // Zero queries; every other parameter is otherwise valid.
        let mut fri_params = FriParameters::new_testing(challenge_mmcs, 0);
        fri_params.num_queries = 0;

        let pcs = TestPcs {
            mmcs: val_mmcs,
            fri_params,
            _phantom: PhantomData,
        };

        // Commit to a random single-column trace of 2^{10} rows.
        let log_n = 10;
        let d =
            <TestPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_n);
        let evals = RowMajorMatrix::rand(&mut rng, 1 << log_n, 1);
        let (_comm, data) = <TestPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(d, evals)]);

        // Commit succeeds; the assert fires inside the opening (FRI prover).
        let zeta: Challenge = rng.random();
        let mut chal = Challenger::from_hasher(vec![], byte_hash);
        let _ = pcs.open(
            vec![OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut chal,
        );
    }

    #[test]
    fn reject_folding_cap_above_two() {
        // Invariant: a cap above one is refused before any proof data is read.
        //
        // Circle folding halves the domain and nothing else.
        // A larger cap once reached a bare assert deep in the query loop.
        // That aborted on an honest proof as readily as on a forged one.
        //
        // Fixture state: a valid proof, then a cap of 3 on the verifier.
        let (pcs, byte_hash, comm, d, zeta, values, proof) = setup_valid_proof(0);

        let mut wide = pcs;
        wide.fri_params.max_log_arity = 3;

        let err = try_verify(&wide, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("a cap above one must be refused");

        let FriError::UnsupportedFoldingCap { max_log_arity } = err else {
            panic!("expected UnsupportedFoldingCap, got {err:?}");
        };
        assert_eq!(max_log_arity, 3);
    }

    #[test]
    #[should_panic(expected = "log_blowup must be at least 1")]
    fn prover_rejects_zero_blowup() {
        // The prover must refuse to build a proof the verifier would accept unconditionally.
        // The verifier guards the same config, so the failure is symmetric.
        let mut rng = SmallRng::seed_from_u64(0);

        // Build the hash stack: field hasher → compression → Merkle tree.
        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs::new(field_hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        // Zero blowup; every other parameter is otherwise valid.
        let mut fri_params = FriParameters::new_testing(challenge_mmcs, 0);
        fri_params.log_blowup = 0;

        let pcs = TestPcs {
            mmcs: val_mmcs,
            fri_params,
            _phantom: PhantomData,
        };

        // Commit to a random single-column trace of 2^{10} rows.
        let log_n = 10;
        let d =
            <TestPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_n);
        let evals = RowMajorMatrix::rand(&mut rng, 1 << log_n, 1);
        let (_comm, data) = <TestPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(d, evals)]);

        // Commit succeeds; the assert fires inside the opening (FRI prover).
        let zeta: Challenge = rng.random();
        let mut chal = Challenger::from_hasher(vec![], byte_hash);
        let _ = pcs.open(
            vec![OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut chal,
        );
    }

    #[test]
    fn reject_commit_pow_witness_count_mismatch() {
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(0);
        let num_rounds = proof.fri_proof.commit_phase_commits.len();

        // Drop one witness so the per-round count falls short.
        proof.fri_proof.commit_pow_witnesses.pop();

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("expected CommitPowWitnessCountMismatch");

        let FriError::CommitPowWitnessCountMismatch { expected, got } = err else {
            panic!("expected CommitPowWitnessCountMismatch, got {err:?}");
        };
        assert_eq!(expected, num_rounds);
        assert_eq!(got, num_rounds - 1);
    }

    #[test]
    fn reject_under_reported_commit_rounds() {
        // Invariant: the commit-round count is fixed by the claimed matrix height.
        //
        //   - the height comes from the claim
        //   - the round count follows from the height
        //   - a proof carrying fewer rounds is a proof of a different shape
        //   - the fold chain and the transcript are sized by the derived count
        //
        // The verifier must reject before either of them is walked.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(0);

        // The round count the claim fixes.
        //
        // The bivariate layer takes one bit before FRI folds anything.
        //
        //     H_claim = log_n + log_blowup
        //     rounds  = H_claim - log_blowup - 1 = log_n - 1
        //
        // The blowup cancels.
        //
        // So the claimed matrix height alone fixes the count.
        let rounds = d.log_n - 1;
        assert_eq!(
            proof.fri_proof.commit_phase_commits.len(),
            rounds,
            "fixture must start round-consistent"
        );

        // Mutation: drop one commit-phase commitment so the round count falls short.
        //
        //     claim fixes:  [c_0, ..., c_{n-1}]   (n rounds)
        //     proof holds:  [c_0, ..., c_{n-2}]   (n - 1 rounds)
        //
        //     n - 1 != n  ->  rejected before the fold chain runs
        proof.fri_proof.commit_phase_commits.pop();

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("expected CommitRoundCountMismatch");

        let FriError::CommitRoundCountMismatch { expected, got } = err else {
            panic!("expected CommitRoundCountMismatch, got {err:?}");
        };
        // The verifier wants one round per bit the claimed height has to travel.
        assert_eq!(expected, rounds);
        // The proof under-reports by exactly the one round we removed.
        assert_eq!(got, rounds - 1);
    }

    #[test]
    fn reject_commit_phase_openings_count_mismatch() {
        // Invariant: the proof must carry exactly one opening set per
        // commit-phase round. Fewer (or more) than there are commitments
        // makes the proof shape invalid.
        let (pcs, byte_hash, comm, d, zeta, values, proof) = setup_valid_proof(0);

        // We need the original proof to assert against its commitment count,
        // so clone before mutating.
        let mut bad = proof.clone();

        // Mutation: remove the last round's openings.
        //
        //     commit_phase_commits:   [c_0, ..., c_{n-1}]   (n rounds)
        //     commit_phase_openings:  [o_0, ..., o_{n-2}]   (n - 1 after pop)
        //     → n != n - 1 → error
        bad.fri_proof.commit_phase_openings.pop();

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &bad)
            .expect_err("expected CommitPhaseOpeningsCountMismatch");

        let FriError::CommitPhaseOpeningsCountMismatch { expected, got } = err else {
            panic!("expected CommitPhaseOpeningsCountMismatch, got {err:?}");
        };
        assert_eq!(expected, proof.fri_proof.commit_phase_commits.len());
        assert_eq!(got, expected - 1);
    }

    #[test]
    fn reject_sibling_values_length_mismatch() {
        // Invariant: in each folding round with arity k, the prover must
        // supply exactly k - 1 sibling values (the queried evaluation is
        // the remaining one).
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(0);

        // Circle folding halves the domain, so every round folds by two.
        //
        // The verifier derives that arity.
        //
        // So the fixture states it rather than reading it back out of the proof.
        let arity = 2usize;
        let original_sibling_count =
            proof.fri_proof.commit_phase_openings[0].sibling_values[0].len();

        // Mutation: remove one sibling value from query 0, round 0.
        //
        //     arity = 2^{log_arity}, expected siblings = arity - 1
        //     before: sibling_values = [s_0, ..., s_{arity-2}]   (arity - 1 elements)
        //     after:  sibling_values = [s_0, ..., s_{arity-3}]   (arity - 2 elements)
        //     → expected arity - 1, got arity - 2 → error at round 0
        proof.fri_proof.commit_phase_openings[0].sibling_values[0].pop();

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("expected SiblingValuesLengthMismatch");

        let FriError::SiblingValuesLengthMismatch {
            round,
            expected,
            got,
        } = err
        else {
            panic!("expected SiblingValuesLengthMismatch, got {err:?}");
        };
        // Error must identify round 0 as the offender.
        assert_eq!(round, 0);
        // The verifier expects (arity - 1) siblings per folding group.
        assert_eq!(expected, arity - 1);
        // We popped one, so one fewer than the original count.
        assert_eq!(got, original_sibling_count - 1);
    }

    // Two error variants cannot be triggered through the PCS verification
    // layer because Merkle commitment checks or input-proof validation
    // fail first for any proof mutation that would reach those code paths:
    //
    // - Final fold height mismatch: requires the total folding to stop at
    //   the wrong domain size, but altering round counts also invalidates
    //   Merkle proofs.
    // - Unconsumed reduced openings: requires leftover polynomial data
    //   after folding completes, but input-proof checks reject the shape
    //   before the folding loop runs.
    //
    // Both are reachable by a malicious prover who crafts openings that
    // pass Merkle checks but have wrong structure — they serve as defense
    // in depth in the low-level verifier.

    #[test]
    fn reject_input_openings_query_count_mismatch() {
        // Invariant: the shared input openings must cover every query. The
        // first-layer siblings carry one entry per query, so dropping one
        // leaves a query without its opened row.
        //
        // The cross-query arity-schedule check this test used to perform is
        // now unrepresentable: `log_arity` lives once per round, not once per
        // query, so no two queries can disagree.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(0);

        // Mutation: drop the last query's first-layer siblings.
        //
        //     before: first_layer_siblings = [f_0, ..., f_{n-1}]   (n = num_queries)
        //     after:  first_layer_siblings = [f_0, ..., f_{n-2}]   (n - 1)
        //     → the shape gate rejects before any Merkle work
        proof.fri_proof.input_openings.first_layer_siblings.pop();

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("expected InputShapeError");

        assert!(
            matches!(err, FriError::InputError(InputError::InputShapeError)),
            "expected InputShapeError, got {err:?}"
        );
    }

    #[test]
    fn reject_tampered_commit_phase_sibling_value() {
        // Invariant: a tampered sibling cannot survive, whichever check reaches
        // it first. A sibling feeds the fold, so flipping one diverges the
        // folded constant and the final-polynomial check fires. Tampering that
        // preserves the fold is caught instead by the shared per-round Merkle
        // check, because the reconstructed row stops matching the committed
        // leaf (see `reject_tampered_commit_phase_opening_proof`). Together the
        // two paths cover both shapes of attack; the accepted set is their
        // conjunction, so the order in which they run does not widen it.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(0);

        proof.fri_proof.commit_phase_openings[0].sibling_values[0][0] += Challenge::ONE;

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("a tampered sibling value must be rejected");

        assert!(
            matches!(err, FriError::FinalPolyMismatch),
            "expected FinalPolyMismatch, got {err:?}"
        );
    }

    #[test]
    fn reject_tampered_commit_phase_opening_proof() {
        // The round's shared multiproof carries every deduplicated sibling
        // digest. Corrupting one makes the recomputed root diverge from the
        // round commitment.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(0);

        proof.fri_proof.commit_phase_openings[0]
            .opening_proof
            .sibling_hashes[0] = Default::default();

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("a tampered commit-phase digest must be rejected");

        assert!(
            matches!(err, FriError::CommitPhaseMmcsError(_)),
            "expected CommitPhaseMmcsError, got {err:?}"
        );
    }

    #[test]
    fn reject_tampered_first_layer_proof() {
        // The first-layer tree is opened once for every query through a single
        // shared multiproof; a corrupted digest there must fail before any
        // reduced opening is trusted.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(0);

        proof
            .fri_proof
            .input_openings
            .first_layer_proof
            .sibling_hashes[0] = Default::default();

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("a tampered first-layer digest must be rejected");

        assert!(
            matches!(
                err,
                FriError::InputError(InputError::FirstLayerMmcsError(_))
            ),
            "expected FirstLayerMmcsError, got {err:?}"
        );
    }

    // Soundness by mutation, one test per value the transcript absorbs.
    //
    // Every one of them moves a challenge the rest of the run depends on.
    // The rejection then surfaces at whichever check meets the new challenge first:
    //
    //     a grinding replay that no longer clears its difficulty
    //     a fold chain that no longer lands on the claimed constant
    //     a Merkle check against a redrawn query index
    //
    // Which of the three lands first is an accident of the fixture.
    // These tests assert the rejection and not the variant.

    #[test]
    fn reject_tampered_opening_point() {
        // The transcript binds the point as a circle point, not just the values at it.
        //
        //     zeta on the projective line  ->  P = ((1 - zeta^2)/(1 + zeta^2), 2 zeta/(1 + zeta^2))
        let (pcs, byte_hash, comm, d, zeta, values, proof) = setup_valid_proof(0);

        // Mutation: claim the same values at a different point.
        let moved = zeta + Challenge::ONE;

        try_verify(&pcs, byte_hash, &comm, d, moved, &values, &proof)
            .expect_err("a moved opening point must be rejected");
    }

    #[test]
    fn reject_tampered_opened_value() {
        // The claimed evaluations are the statement, and the transcript binds them.
        let (pcs, byte_hash, comm, d, zeta, mut values, proof) = setup_valid_proof(0);

        // Mutation: shift one claimed evaluation.
        values[0][0][0][0] += Challenge::ONE;

        try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("a tampered claimed evaluation must be rejected");
    }

    #[test]
    fn reject_tampered_first_layer_commitment() {
        // The bivariate challenge is drawn after this commitment, so it binds it.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(0);

        proof.first_layer_commitment = MerkleCap::new(vec![[0u8; 32]]);

        try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("a tampered first-layer commitment must be rejected");
    }

    #[test]
    fn reject_tampered_commit_phase_commitment() {
        // Each round's folding challenge is drawn after its commitment.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(0);

        proof.fri_proof.commit_phase_commits[0] = MerkleCap::new(vec![[0u8; 32]]);

        try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("a tampered commit-phase commitment must be rejected");
    }

    #[test]
    fn reject_tampered_commit_pow_witness() {
        // Grinding sits between a commitment and the challenge it protects.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(0);
        assert!(
            pcs.fri_params.commit_proof_of_work_bits > 0,
            "fixture must describe a commit-phase grinding step"
        );

        proof.fri_proof.commit_pow_witnesses[0] += Val::ONE;

        try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("a tampered commit grinding witness must be rejected");
    }

    #[test]
    fn reject_tampered_final_poly() {
        // The constant is absorbed before the query indices are drawn.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(0);

        proof.fri_proof.final_poly += Challenge::ONE;

        try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("a tampered final polynomial must be rejected");
    }

    #[test]
    fn reject_tampered_query_pow_witness() {
        // Grinding here raises the cost of searching for favourable query indices.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof(0);
        assert!(
            pcs.fri_params.query_proof_of_work_bits > 0,
            "fixture must describe a query grinding step"
        );

        proof.fri_proof.pow_witness += Val::ONE;

        try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("a tampered query grinding witness must be rejected");
    }

    #[test]
    fn reject_noncanonical_pow_witnesses_at_zero_difficulty() {
        // Invariant: at zero difficulty only a canonical-value check binds the witness.
        //
        //     bits = 0 -> check_witness returns true, the step is elided -> unbound
        //     bits > 0 -> absorbed, bits resampled                       -> grind binds
        //
        // The commit-phase mutation moves one element and leaves the length alone: the
        // length has its own rejection, and a count check is not a value check.
        let (pcs, byte_hash, comm, d, zeta, values, proof) = setup_valid_proof_at(0, 0, 0);

        // The honest prover writes zero into every slot it pays no work for.
        assert!(
            proof
                .fri_proof
                .commit_pow_witnesses
                .iter()
                .all(|w| *w == Val::ZERO)
        );
        assert_eq!(proof.fri_proof.pow_witness, Val::ZERO);

        // The untouched proof still verifies, so the mutations below are the only change.
        try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect("an ungrounded proof must verify");

        let mut mutated = proof.clone();
        mutated.fri_proof.commit_pow_witnesses[0] = Val::ONE;
        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &mutated)
            .expect_err("a rewritten commit-phase witness must be rejected");
        assert!(
            matches!(
                err,
                FriError::NonCanonicalPowWitness {
                    phase: PowPhase::CommitPhase
                }
            ),
            "expected NonCanonicalPowWitness for the commit phase, got {err:?}"
        );

        let mut mutated = proof;
        mutated.fri_proof.pow_witness = Val::ONE;
        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &mutated)
            .expect_err("a rewritten query witness must be rejected");
        assert!(
            matches!(
                err,
                FriError::NonCanonicalPowWitness {
                    phase: PowPhase::Query
                }
            ),
            "expected NonCanonicalPowWitness for the query phase, got {err:?}"
        );
    }

    #[test]
    fn reject_global_max_height_too_large() {
        // Invariant: the query-index width fits the circle group of order 2^CIRCLE_TWO_ADICITY.
        //
        //     index_bits  = claimed log_n + log_blowup - 1 + extra_query_index_bits (= 1)
        //     field order = 2^CIRCLE_TWO_ADICITY - 1   (one short of the group order)
        //     => a width of CIRCLE_TWO_ADICITY bits is unsampleable
        //
        // The width comes from the claimed statement, so the claim is what has to overflow it.
        let (pcs, byte_hash, comm, _, zeta, values, proof) = setup_valid_proof(0);

        // Mutation: claim a domain wide enough to drive the index width to the bound.
        let log_n = Val::CIRCLE_TWO_ADICITY - pcs.fri_params.log_blowup;
        let too_wide = CircleDomain::standard(log_n);

        let err = try_verify(&pcs, byte_hash, &comm, too_wide, zeta, &values, &proof)
            .expect_err("expected GlobalMaxHeightTooLarge");

        let FriError::GlobalMaxHeightTooLarge {
            log_global_max_height,
            two_adicity,
        } = err
        else {
            panic!("expected GlobalMaxHeightTooLarge, got {err:?}");
        };
        // The reported bound is the circle group two-adicity.
        assert_eq!(two_adicity, Val::CIRCLE_TWO_ADICITY);
        // The rejecting width is at least that bound.
        assert!(log_global_max_height >= two_adicity);
    }
}
