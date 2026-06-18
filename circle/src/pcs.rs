use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{
    BatchOpening, BatchOpeningRef, BuildPeriodicLdeTableFast, Mmcs, OpenedValues, Pcs,
    PeriodicLdeTable, PolynomialSpace,
};
use p3_field::extension::ComplexExtendable;
use p3_field::{ExtensionField, Field, dot_product};
use p3_fri::FriParameters;
use p3_fri::verifier::FriError;
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
use crate::folding::{CircleFriFolding, CircleFriFoldingForMmcs, fold_y, fold_y_row};
use crate::point::{Point, compute_lagrange_den_batched};
use crate::prover::prove;
use crate::verifier::verify;
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
    input_openings: Vec<BatchOpening<Val, InputMmcs>>,
    first_layer_siblings: Vec<Challenge>,
    first_layer_proof: FriMmcs::Proof,
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
    first_layer_commitment: FriMmcs::Commitment,
    lambdas: Vec<Challenge>,
    fri_proof: CircleFriProof<
        Challenge,
        FriMmcs,
        Witness,
        CircleInputProof<Val, Challenge, InputMmcs, FriMmcs>,
    >,
}

impl<Val, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for CirclePcs<Val, InputMmcs, FriMmcs>
where
    Val: ComplexExtendable,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<FriMmcs::Commitment>,
{
    type Domain = CircleDomain<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type EvaluationsOnDomain<'a> = RowIndexMappedView<CfftPerm, RowMajorMatrixCow<'a, Val>>;
    type Proof = CirclePcsProof<Val, Challenge, InputMmcs, FriMmcs, Challenger::Witness>;
    type Error = FriError<FriMmcs::Error, InputError<InputMmcs::Error, FriMmcs::Error>>;
    const ZK: bool = false;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        CircleDomain::standard(log2_strict_usize(degree))
    }

    fn log_max_lde_height(&self) -> usize {
        Val::CIRCLE_TWO_ADICITY
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
            // log_blowup` coefficients, so interpolating it leaves zeros above that index.
            // Truncating to the original coefficient count lets `domain` be smaller than the
            // committed LDE (e.g. a quotient domain when `log_blowup` exceeds the quotient
            // degree), which `extrapolate` would reject.
            let log_orig = committed_domain.log_n - self.fri_params.log_blowup;
            let mut coeffs =
                CircleEvaluations::from_cfft_order(committed_domain, mat).interpolate();
            let width = coeffs.width();
            coeffs.values.truncate((1 << log_orig) * width);
            CircleEvaluations::evaluate(domain, coeffs)
                .to_cfft_order()
                .as_cow()
                .cfft_perm_rows()
        }
    }

    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        // Materialize the CFFT-ordered domain points once per committed height. They are shared
        // by the Lagrange denominators and the DEEP-quotient vanishing parts below, which are in
        // turn shared by every matrix opened at the same point on the same domain.
        let mut permuted_points: BTreeMap<usize, Vec<Point<Val>>> = BTreeMap::new();
        debug_span!("materialize domain points").in_scope(|| {
            for (data, _) in &rounds {
                for mat in self.mmcs.get_matrices(data) {
                    let log_height = log2_strict_usize(mat.height());
                    permuted_points.entry(log_height).or_insert_with(|| {
                        cfft_permute_slice(
                            &CircleDomain::standard(log_height).points().collect_vec(),
                        )
                    });
                }
            }
        });

        // (log_height, point) -> Lagrange denominators.
        let mut lagrange_dens: Vec<((usize, Challenge), Vec<Challenge>)> = vec![];

        // Open matrices at points
        let values: OpenedValues<Challenge> = rounds
            .iter()
            .map(|(data, points_for_mats)| {
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

                        for ps_at_zeta in &ps_for_points {
                            challenger.observe_algebra_slice(ps_at_zeta);
                        }
                        ps_for_points
                    })
                    .collect()
            })
            .collect();
        drop(lagrange_dens);

        // Batch combination challenge
        let alpha: Challenge = challenger.sample_algebra_element();

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

        rounds
            .iter()
            .zip(values.iter())
            .for_each(|((data, points_for_mats), values)| {
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
            });
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
        let log_max_height = log_heights.iter().max().copied().unwrap();

        // Commit to reduced openings at each log_height, so we can challenge a global
        // folding factor for all first layers, which we use for a "manual" (not part of p3-fri) fold.
        // This is necessary because the first layer of folding uses different twiddles, so it's easiest
        // to do it here, before p3-fri.

        let (first_layer_commitment, first_layer_data) =
            self.fri_params.mmcs.commit(first_layer_mats);
        challenger.observe(first_layer_commitment.clone());
        let bivariate_beta: Challenge = challenger.sample_algebra_element();

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

        let folding: CircleFriFoldingForMmcs<Val, Challenge, InputMmcs, FriMmcs> =
            CircleFriFolding(PhantomData);

        let fri_proof = prove(&folding, &self.fri_params, fri_input, challenger, |index| {
            // CircleFriFolder asks for an extra query index bit, so we use that here to index
            // the first layer fold.

            // Open the input (big opening, lots of columns) at the full index...
            let input_openings = rounds
                .iter()
                .map(|(data, _)| {
                    let log_max_batch_height = log2_strict_usize(self.mmcs.get_max_height(data));
                    let reduced_index = index >> (log_max_height - log_max_batch_height);
                    self.mmcs.open_batch(reduced_index, data)
                })
                .collect();

            // We committed to first_layer in pairs, so open the reduced index and include the sibling
            // as part of the input proof.
            let (first_layer_values, first_layer_proof) = self
                .fri_params
                .mmcs
                .open_batch(index >> 1, &first_layer_data)
                .unpack();
            let first_layer_siblings = izip!(&first_layer_values, &log_heights)
                .map(|(v, log_height)| {
                    let reduced_index = index >> (log_max_height - log_height);
                    let sibling_index = (reduced_index & 1) ^ 1;
                    v[sibling_index]
                })
                .collect();
            CircleInputProof {
                input_openings,
                first_layer_siblings,
                first_layer_proof,
            }
        });

        (
            values,
            CirclePcsProof {
                first_layer_commitment,
                lambdas,
                fri_proof,
            },
        )
    }

    fn verify(
        &self,
        // For each round:
        rounds: Vec<(
            Self::Commitment,
            // for each matrix:
            Vec<(
                // its domain,
                Self::Domain,
                // for each point:
                Vec<(
                    // the point,
                    Challenge,
                    // values at the point
                    Vec<Challenge>,
                )>,
            )>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // Write evaluations to challenger
        for (_, round) in &rounds {
            for (_, mat) in round {
                for (_, point) in mat {
                    challenger.observe_algebra_slice(point);
                }
            }
        }

        // Batch combination challenge
        let alpha: Challenge = challenger.sample_algebra_element();
        challenger.observe(proof.first_layer_commitment.clone());
        let bivariate_beta: Challenge = challenger.sample_algebra_element();

        // +1 to account for first layer
        let log_global_max_height =
            proof.fri_proof.commit_phase_commits.len() + self.fri_params.log_blowup + 1;

        // Guard the query-phase height subtraction against an under-reported round count.
        //
        // Invariant: the proof's global height covers every claimed matrix.
        //
        //     H_proof = commit-phase round count + log_blowup + 1   (first-layer fold)
        //     H_claim = max committed log_n + log_blowup
        //
        // The query phase computes `index >> (log_global_max_height - log_height)`.
        //   - `log_height <= H_claim` holds for every matrix
        //   - `H_proof < H_claim` makes that usize subtraction underflow and the shift wrap
        // Over-reporting is caught downstream (two-adicity bound, Merkle openings), so the
        // under-report is the only case to reject here.
        let expected_log_global_max_height = rounds
            .iter()
            .flat_map(|(_, mats)| {
                mats.iter()
                    .map(|(domain, _)| domain.log_n + self.fri_params.log_blowup)
            })
            .max();
        if let Some(expected) = expected_log_global_max_height
            && log_global_max_height < expected
        {
            return Err(FriError::GlobalMaxHeightMismatch {
                expected,
                got: log_global_max_height,
            });
        }

        let folding: CircleFriFoldingForMmcs<Val, Challenge, InputMmcs, FriMmcs> =
            CircleFriFolding(PhantomData);

        verify(
            &folding,
            &self.fri_params,
            &proof.fri_proof,
            challenger,
            |index, input_proof| {
                // log_height -> (alpha_offset, ro)
                let mut reduced_openings = BTreeMap::new();

                let CircleInputProof {
                    input_openings,
                    first_layer_siblings,
                    first_layer_proof,
                } = input_proof;

                for (batch, (batch_opening, (batch_commit, mats))) in
                    zip_eq(input_openings, &rounds, InputError::InputShapeError)?.enumerate()
                {
                    let batch_heights: Vec<usize> = mats
                        .iter()
                        .map(|(domain, _)| domain.size() << self.fri_params.log_blowup)
                        .collect_vec();
                    // The opened rows must pair one-to-one with the committed matrices.
                    let batch_dims: Vec<Dimensions> = zip_eq(
                        batch_heights.iter().zip(mats),
                        &batch_opening.opened_values,
                        InputError::InputShapeError,
                    )?
                    .enumerate()
                    .map(|(matrix, ((&height, (_, points_and_values)), _))| {
                        // Invariant: a matrix's width is fixed by its first opening point.
                        //
                        //     >= 1 point  ->  width = number of claimed evaluations
                        //     no points   ->  reject
                        //
                        // Why reject the no-points case:
                        //   - row boundaries in the flattened leaf hash are authenticated only from claimed widths
                        //   - a matrix opened at no points claims no width
                        //   - its width could then come only from the unverified proof
                        let (_, values) = points_and_values
                            .first()
                            .ok_or(InputError::MatrixWithoutOpeningPoints { batch, matrix })?;
                        Ok(Dimensions {
                            width: values.len(),
                            height,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                    let (dims, idx) = batch_heights
                        .iter()
                        .max()
                        .map(|x| log2_strict_usize(*x))
                        .map_or_else(
                            ||
                            // Empty batch?
                            (&[][..], 0),
                            |log_batch_max_height| {
                                (
                                    &batch_dims[..],
                                    index >> (log_global_max_height - log_batch_max_height),
                                )
                            },
                        );

                    self.mmcs
                        .verify_batch(batch_commit, dims, idx, batch_opening.into())
                        .map_err(InputError::InputMmcsError)?;

                    for (ps_at_x, (mat_domain, mat_points_and_values)) in zip_eq(
                        &batch_opening.opened_values,
                        mats,
                        InputError::InputShapeError,
                    )? {
                        let log_height = mat_domain.log_n + self.fri_params.log_blowup;
                        let bits_reduced = log_global_max_height - log_height;
                        let orig_idx = cfft_permute_index(index >> bits_reduced, log_height);

                        let committed_domain = CircleDomain::standard(log_height);
                        let x = committed_domain.nth_point(orig_idx);

                        let (alpha_offset, ro) = reduced_openings
                            .entry(log_height)
                            .or_insert((Challenge::ONE, Challenge::ZERO));
                        let alpha_pow_width_2 = alpha.exp_u64(ps_at_x.len() as u64).square();

                        for (zeta_uni, ps_at_zeta) in mat_points_and_values {
                            // The claimed opening must have exactly as many
                            // values as the committed row has columns.
                            if ps_at_zeta.len() != ps_at_x.len() {
                                return Err(InputError::InputShapeError);
                            }
                            let zeta = Point::from_projective_line(*zeta_uni);

                            // A vanishing denominator means this opening point lands on the
                            // query point; reject the proof rather than dividing by zero.
                            *ro += *alpha_offset
                                * deep_quotient_reduce_row(alpha, x, zeta, ps_at_x, ps_at_zeta)
                                    .ok_or(InputError::OpeningPointMatchesQueryPoint)?;

                            *alpha_offset *= alpha_pow_width_2;
                        }
                    }
                }

                // Verify bivariate fold and lambda correction

                let (mut fri_input, fl_dims, fl_leaves): (Vec<_>, Vec<_>, Vec<_>) = zip_eq(
                    zip_eq(
                        reduced_openings,
                        first_layer_siblings,
                        InputError::InputShapeError,
                    )?,
                    &proof.lambdas,
                    InputError::InputShapeError,
                )?
                .map(|(((log_height, (_, ro)), &fl_sib), &lambda)| {
                    assert!(log_height > 0);

                    let orig_size = log_height - self.fri_params.log_blowup;
                    let bits_reduced = log_global_max_height - log_height;
                    let orig_idx = cfft_permute_index(index >> bits_reduced, log_height);

                    let lde_domain = CircleDomain::standard(log_height);
                    let p: Point<Val> = lde_domain.nth_point(orig_idx);

                    let lambda_corrected = ro - lambda * p.v_n(orig_size);

                    let mut fl_values = vec![lambda_corrected; 2];
                    fl_values[((index >> bits_reduced) & 1) ^ 1] = fl_sib;

                    let fri_input = (
                        // - 1 here is because we have already folded a layer.
                        log_height - 1,
                        fold_y_row(
                            index >> (bits_reduced + 1),
                            // - 1 here is log_arity.
                            log_height - 1,
                            bivariate_beta,
                            fl_values.iter().copied(),
                        ),
                    );

                    let fl_dims = Dimensions {
                        // First-layer leaves hold the queried value and its sibling.
                        width: 2,
                        height: 1 << (log_height - 1),
                    };

                    (fri_input, fl_dims, fl_values)
                })
                .multiunzip();

                // sort descending
                fri_input.reverse();

                self.fri_params
                    .mmcs
                    .verify_batch(
                        &proof.first_layer_commitment,
                        &fl_dims,
                        index >> 1,
                        BatchOpeningRef::new(&fl_leaves, first_layer_proof),
                    )
                    .map_err(InputError::FirstLayerMmcsError)?;

                Ok(fri_input)
            },
        )
    }
}

impl<Val, InputMmcs, FriMmcs> BuildPeriodicLdeTableFast for CirclePcs<Val, InputMmcs, FriMmcs>
where
    Val: ComplexExtendable,
    InputMmcs: Mmcs<Val>,
{
    type PeriodicDomain = CircleDomain<Val>;

    fn maybe_build_periodic_lde_table_fast(
        &self,
        periodic_cols: &[Vec<p3_commit::Val<Self::PeriodicDomain>>],
        trace_domain: Self::PeriodicDomain,
        quotient_domain: Self::PeriodicDomain,
    ) -> Option<PeriodicLdeTable<p3_commit::Val<Self::PeriodicDomain>>>
    where
        p3_commit::Val<Self::PeriodicDomain>: Clone,
    {
        let table = build_periodic_lde_table_circle(periodic_cols, &trace_domain, &quotient_domain);
        Some(table)
    }
}

#[cfg(test)]
mod tests {
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_commit::ExtensionMmcs;
    use p3_field::extension::BinomialExtensionField;
    use p3_fri::FriParameters;
    use p3_fri::verifier::FriError;
    use p3_keccak::Keccak256Hash;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_mersenne_31::Mersenne31;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
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
    fn setup_valid_proof() -> (
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
        let fri_params = FriParameters::new_testing(challenge_mmcs, 0);

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
        let (values, proof) = pcs.open(vec![(&data, vec![vec![zeta]])], &mut chal);

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
            vec![(
                comm.clone(),
                vec![(d, vec![(zeta, values[0][0][0].clone())])],
            )],
            proof,
            &mut chal,
        )
    }

    #[test]
    fn circle_pcs() {
        // Smoke test: an honestly generated proof must verify successfully.
        let (pcs, byte_hash, comm, d, zeta, values, proof) = setup_valid_proof();
        try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof).expect("verify err");
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
        let (values, proof) = pcs.open(vec![(&data, vec![vec![zeta], vec![]])], &mut chal);

        // Verify with the same shape: matrix 1 carries no opening points.
        let mut chal = Challenger::from_hasher(vec![], byte_hash);
        let err = pcs
            .verify(
                vec![(
                    comm,
                    vec![(d, vec![(zeta, values[0][0][0].clone())]), (d, vec![])],
                )],
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
            let got = <TestPcs as Pcs<Challenge, Challenger>>::get_evaluations_on_domain(
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
    fn reject_query_proof_count_mismatch() {
        // Invariant: the proof must contain exactly num_queries query proofs.
        // The verifier rejects if the count is wrong.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof();

        // Mutation: remove one query proof so the count falls short.
        //
        //     before: query_proofs = [q_0, q_1, ..., q_{n-1}]   (n = num_queries)
        //     after:  query_proofs = [q_0, q_1, ..., q_{n-2}]   (n - 1)
        //     → expected n, got n - 1 → error
        proof.fri_proof.query_proofs.pop();

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("expected QueryProofCountMismatch");

        // Destructure for precise field assertions (better diagnostics than matches!).
        let FriError::QueryProofCountMismatch { expected, got } = err else {
            panic!("expected QueryProofCountMismatch, got {err:?}");
        };
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
        let (mut pcs, byte_hash, comm, d, zeta, values, proof) = setup_valid_proof();
        pcs.fri_params.num_queries = 0;

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("zero-query instance must be rejected");

        assert!(
            matches!(err, FriError::ZeroQueries),
            "expected ZeroQueries, got {err:?}"
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
        let _ = pcs.open(vec![(&data, vec![vec![zeta]])], &mut chal);
    }

    #[test]
    fn reject_commit_pow_witness_count_mismatch() {
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof();
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
        // Invariant: the reported commit-round count must cover the claimed matrix height.
        //   - log_global_max_height is derived from the proof's round count
        //   - under-reporting drives it below a matrix's log_height
        //   - then `index >> (log_global_max_height - log_height)` would underflow
        // The verifier must reject before that subtraction runs.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof();

        // On an honest proof the two height derivations coincide:
        //
        //     H_claim = log_n + log_blowup                            (claimed matrix)
        //     H_proof = commit_phase_commits.len() + log_blowup + 1   (first-layer fold)
        let log_blowup = pcs.fri_params.log_blowup;
        let expected = d.log_n + log_blowup;
        let original = proof.fri_proof.commit_phase_commits.len() + log_blowup + 1;
        assert_eq!(original, expected, "fixture must start height-consistent");

        // Mutation: drop one commit-phase commitment so the round count falls short.
        //
        //     before: commit_phase_commits = [c_0, ..., c_{n-1}]   → H_proof = expected
        //     after:  commit_phase_commits = [c_0, ..., c_{n-2}]   → H_proof = expected - 1
        //     → H_proof < H_claim → GlobalMaxHeightMismatch (no underflow)
        proof.fri_proof.commit_phase_commits.pop();

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("expected GlobalMaxHeightMismatch");

        let FriError::GlobalMaxHeightMismatch { expected: exp, got } = err else {
            panic!("expected GlobalMaxHeightMismatch, got {err:?}");
        };
        // The verifier wants the height the claimed matrix demands.
        assert_eq!(exp, expected);
        // The proof under-reports by exactly the one round we removed.
        assert_eq!(got, expected - 1);
    }

    #[test]
    fn reject_query_commit_phase_openings_count_mismatch() {
        // Invariant: each query proof must carry exactly one opening per
        // commit-phase round. If a query has fewer (or more) openings than
        // there are commitments, the proof shape is invalid.
        let (pcs, byte_hash, comm, d, zeta, values, proof) = setup_valid_proof();

        // We need the original proof to assert against its commitment count,
        // so clone before mutating.
        let mut bad = proof.clone();

        // Mutation: remove the last opening from query 0.
        //
        //     commit_phase_commits:                [c_0, ..., c_{n-1}]   (n rounds)
        //     query 0 commit_phase_openings:       [o_0, ..., o_{n-2}]   (n - 1 after pop)
        //     → n != n - 1 → error on query 0
        bad.fri_proof.query_proofs[0].commit_phase_openings.pop();

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &bad)
            .expect_err("expected QueryCommitPhaseOpeningsCountMismatch");

        let FriError::QueryCommitPhaseOpeningsCountMismatch {
            query,
            expected,
            got,
        } = err
        else {
            panic!("expected QueryCommitPhaseOpeningsCountMismatch, got {err:?}");
        };
        // Error must identify query 0 as the offender.
        assert_eq!(query, 0);
        assert_eq!(expected, proof.fri_proof.commit_phase_commits.len());
        assert_eq!(got, expected - 1);
    }

    #[test]
    fn reject_sibling_values_length_mismatch() {
        // Invariant: in each folding round with arity k, the prover must
        // supply exactly k - 1 sibling values (the queried evaluation is
        // the remaining one).
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof();

        // Capture the original sibling count and arity before mutating.
        let log_arity = proof.fri_proof.query_proofs[0].commit_phase_openings[0].log_arity as usize;
        let arity = 1usize << log_arity;
        let original_sibling_count = proof.fri_proof.query_proofs[0].commit_phase_openings[0]
            .sibling_values
            .len();

        // Mutation: remove one sibling value from query 0, round 0.
        //
        //     arity = 2^{log_arity}, expected siblings = arity - 1
        //     before: sibling_values = [s_0, ..., s_{arity-2}]   (arity - 1 elements)
        //     after:  sibling_values = [s_0, ..., s_{arity-3}]   (arity - 2 elements)
        //     → expected arity - 1, got arity - 2 → error at round 0
        proof.fri_proof.query_proofs[0].commit_phase_openings[0]
            .sibling_values
            .pop();

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
    fn reject_query_log_arities_mismatch() {
        // Invariant: all query proofs must use the same per-round folding
        // arity schedule. The verifier takes the first query proof's
        // schedule as a reference and rejects any that differ.
        let (mut pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof();
        // Allow arity 2 so this mutation remains a schedule mismatch rather
        // than being rejected as an out-of-range arity.
        pcs.fri_params.max_log_arity = 2;

        // This check compares query 1 against query 0, so we need at least
        // two query proofs. With testing parameters this is always true, but
        // guard defensively.
        if proof.fri_proof.query_proofs.len() < 2 {
            return;
        }

        // Capture the reference arity schedule from query 0 before mutating.
        let reference_arities: Vec<usize> = proof.fri_proof.query_proofs[0]
            .commit_phase_openings
            .iter()
            .map(|o| o.log_arity as usize)
            .collect();

        // Mutation: bump the log_arity of query 1's first round by 1.
        //
        //     query 0 arities: [a_0, a_1, ..., a_{n-1}]       (reference)
        //     query 1 arities: [a_0 + 1, a_1, ..., a_{n-1}]   (corrupted)
        //     → schedules differ → error on query 1
        let original = proof.fri_proof.query_proofs[1].commit_phase_openings[0].log_arity;
        proof.fri_proof.query_proofs[1].commit_phase_openings[0].log_arity = original + 1;

        // Build the expected corrupted schedule for query 1.
        let mut corrupted_arities = reference_arities.clone();
        corrupted_arities[0] = original as usize + 1;

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("expected QueryLogAritiesMismatch");

        let FriError::QueryLogAritiesMismatch {
            query,
            expected,
            got,
        } = err
        else {
            panic!("expected QueryLogAritiesMismatch, got {err:?}");
        };
        // Error must identify query 1 (the first one compared against the reference).
        assert_eq!(query, 1);
        // The expected schedule is query 0's (the reference).
        assert_eq!(expected, reference_arities);
        // The got schedule is query 1's corrupted version.
        assert_eq!(got, corrupted_arities);
    }

    #[test]
    fn reject_invalid_log_arity() {
        // Invariant: each log_arity must be in 1..=max_log_arity.
        let (pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof();

        // Mutation: force an invalid zero arity in query 0, round 0.
        proof.fri_proof.query_proofs[0].commit_phase_openings[0].log_arity = 0;

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
            .expect_err("expected InvalidLogArity");

        let FriError::InvalidLogArity {
            round,
            log_arity,
            max,
        } = err
        else {
            panic!("expected InvalidLogArity, got {err:?}");
        };
        assert_eq!(round, 0);
        assert_eq!(log_arity, 0);
        assert_eq!(max, pcs.fri_params.max_log_arity);
    }

    #[test]
    fn reject_global_max_height_too_large() {
        // Invariant: the query-index width fits the circle group of order 2^CIRCLE_TWO_ADICITY.
        //
        //     field order = 2^CIRCLE_TWO_ADICITY - 1   (one short of the group order)
        //     => width of CIRCLE_TWO_ADICITY bits is unsampleable => verifier must reject
        let (mut pcs, byte_hash, comm, d, zeta, values, mut proof) = setup_valid_proof();

        // Zero both proof-of-work targets.
        // Otherwise grinding rejects the cloned witnesses before the width check runs.
        pcs.fri_params.commit_proof_of_work_bits = 0;
        pcs.fri_params.query_proof_of_work_bits = 0;

        // Mutation: clone commit-phase rounds until the width reaches the bound.
        //
        //     num_index_bits = rounds + log_blowup + extra_query_index_bits (= 1 for circle)
        //     stop once num_index_bits >= CIRCLE_TWO_ADICITY
        let extra_query_index_bits = 1;
        let commit = proof.fri_proof.commit_phase_commits[0].clone();
        let witness = proof.fri_proof.commit_pow_witnesses[0];
        while proof.fri_proof.commit_phase_commits.len()
            + pcs.fri_params.log_blowup
            + extra_query_index_bits
            < Val::CIRCLE_TWO_ADICITY
        {
            proof.fri_proof.commit_phase_commits.push(commit.clone());
            proof.fri_proof.commit_pow_witnesses.push(witness);
            // Each query proof needs one opening per round.
            for qp in &mut proof.fri_proof.query_proofs {
                let opening = qp.commit_phase_openings[0].clone();
                qp.commit_phase_openings.push(opening);
            }
        }

        let err = try_verify(&pcs, byte_hash, &comm, d, zeta, &values, &proof)
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
