use core::marker::PhantomData;

use alloc::vec;
use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs, PolynomialSpace};
use p3_field::extension::{Complex, ComplexExtendable};
use p3_field::{batch_multiplicative_inverse, ExtensionField, Field};
use p3_fri::{FriConfig, FriProof, PowersReducer};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use crate::cfft::Cfft;
use crate::deep_quotient::extract_lambda;
use crate::domain::CircleDomain;
use crate::folding::{
    circle_bitrev_permute, fold_bivariate, CircleBitrevPerm, CircleBitrevView, CircleFriFolder,
};
use crate::util::{univariate_to_point, v_n};

#[derive(Debug)]
pub struct CirclePcs<Val: Field, InputMmcs, FriMmcs> {
    pub log_blowup: usize,
    pub cfft: Cfft<Val>,
    pub mmcs: InputMmcs,
    pub fri_config: FriConfig<FriMmcs>,
}

#[derive(Debug)]
pub struct ProverData<Val, MmcsData> {
    committed_domains: Vec<CircleDomain<Val>>,
    mmcs_data: MmcsData,
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BatchOpening<Val: Field, InputMmcs: Mmcs<Val>> {
    pub(crate) opened_values: Vec<Vec<Val>>,
    pub(crate) opening_proof: <InputMmcs as Mmcs<Val>>::Proof,
}

impl<Val, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for CirclePcs<Val, InputMmcs, FriMmcs>
where
    Val: ComplexExtendable,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger: CanSample<Challenge> + GrindingChallenger + CanObserve<FriMmcs::Commitment>,
{
    type Domain = CircleDomain<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = ProverData<Val, InputMmcs::ProverData<CircleBitrevView<RowMajorMatrix<Val>>>>;
    type Proof = (
        FriProof<Challenge, FriMmcs, Challenger::Witness>,
        Vec<Vec<BatchOpening<Val, InputMmcs>>>,
    );
    type Error = ();

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        CircleDomain::standard(log2_strict_usize(degree))
    }

    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let (committed_domains, ldes): (Vec<_>, Vec<_>) = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                let committed_domain = CircleDomain::standard(domain.log_n + self.log_blowup);
                let lde = self.cfft.lde(evals, domain, committed_domain);
                let perm_lde = CircleBitrevPerm::new(lde);
                (committed_domain, perm_lde)
            })
            .unzip();
        let (comm, mmcs_data) = self.mmcs.commit(ldes);
        (
            comm,
            ProverData {
                committed_domains,
                mmcs_data,
            },
        )
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> impl Matrix<Val> + 'a {
        // TODO do this correctly
        let mat = self.mmcs.get_matrices(&data.mmcs_data)[idx];
        assert_eq!(mat.height(), 1 << domain.log_n);
        assert_eq!(domain, data.committed_domains[idx]);
        mat.inner.as_view()
    }

    #[instrument(skip_all)]
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
        // Batch combination challenge
        let alpha: Challenge = challenger.sample();

        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| (self.mmcs.get_matrices(&data.mmcs_data), points))
            .collect_vec();

        let max_width = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats)
            .map(|m| m.width())
            .max()
            .unwrap();

        let mut reduced_openings: [Option<Vec<Challenge>>; 32] = core::array::from_fn(|_| None);
        let mut num_reduced = [0; 32];

        let alpha_reducer = PowersReducer::<Val, Challenge>::new(alpha, max_width);

        let values: OpenedValues<Challenge> = rounds
            .iter()
            .map(|(data, points_for_mats)| {
                let mats = self.mmcs.get_matrices(&data.mmcs_data);
                izip!(&data.committed_domains, mats, points_for_mats)
                    .map(|(domain, permuted_mat, points_for_mat)| {
                        let mat = &permuted_mat.inner;
                        let log_height = log2_strict_usize(mat.height());
                        let reduced_opening_for_log_height: &mut Vec<Challenge> = reduced_openings
                            [log_height]
                            .get_or_insert_with(|| vec![Challenge::zero(); mat.height()]);
                        points_for_mat
                            .into_iter()
                            .map(|&zeta| {
                                let zeta_point = univariate_to_point(zeta).unwrap();

                                // todo: cache basis
                                let basis: Vec<Challenge> = domain.lagrange_basis(zeta_point);
                                let v_n_at_zeta = v_n(zeta_point.real(), log_height)
                                    - v_n(domain.shift.real(), log_height);

                                let alpha_pow_offset =
                                    alpha.exp_u64(num_reduced[log_height] as u64);
                                let alpha_pow_width = alpha.exp_u64(mat.width() as u64);
                                num_reduced[log_height] += 2 * mat.width();

                                let (lhs_nums, lhs_denoms): (Vec<_>, Vec<_>) = domain
                                    .points()
                                    .map(|x| {
                                        let x_rotate_zeta: Complex<Challenge> =
                                            x.rotate(zeta_point.conjugate());

                                        let v_gamma_re: Challenge =
                                            Challenge::one() - x_rotate_zeta.real();
                                        let v_gamma_im: Challenge = x_rotate_zeta.imag();

                                        (
                                            v_gamma_re - alpha_pow_width * v_gamma_im,
                                            v_gamma_re.square() + v_gamma_im.square(),
                                        )
                                    })
                                    .unzip();
                                let inv_lhs_denoms = batch_multiplicative_inverse(&lhs_denoms);

                                // todo: we only need half of the values to interpolate, but how?
                                let ps_at_zeta: Vec<Challenge> =
                                    info_span!("compute opened values with Lagrange interpolation")
                                        .in_scope(|| {
                                            mat.columnwise_dot_product(&basis)
                                                .into_iter()
                                                .map(|x| x * v_n_at_zeta)
                                                .collect()
                                        });

                                let alpha_pow_ps_at_zeta = alpha_reducer.reduce_ext(&ps_at_zeta);

                                info_span!(
                                    "reduce rows",
                                    log_height = log_height,
                                    width = mat.width()
                                )
                                .in_scope(|| {
                                    izip!(
                                        reduced_opening_for_log_height.par_iter_mut(),
                                        mat.rows(),
                                        lhs_nums,
                                        inv_lhs_denoms,
                                    )
                                    .for_each(
                                        |(reduced_opening, row, lhs_num, inv_lhs_denom)| {
                                            *reduced_opening += lhs_num
                                                * inv_lhs_denom
                                                * alpha_pow_offset
                                                * (alpha_reducer.reduce_base(&row.collect_vec())
                                                    - alpha_pow_ps_at_zeta);
                                        },
                                    )
                                });

                                ps_at_zeta
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // We do bivariate fold now, so can't have a singleton poly
        assert!(reduced_openings[0].is_none());
        // Do the first circle fold for all polys with the same beta

        let mut first_layer_mats = vec![];
        for i in 0..31 {
            if let Some(ro) = reduced_openings.get(i + 1).unwrap() {
                let mut ro = ro.clone();
                // todo send this
                let _lambda = extract_lambda(
                    CircleDomain::standard(i + 1 - self.log_blowup),
                    CircleDomain::standard(i + 1),
                    &mut ro,
                );
                // since we unpermuted above (.inner()) we need to permute ROs
                let ro_permuted = RowMajorMatrix::new(circle_bitrev_permute(&ro), 2);
                first_layer_mats.push(ro_permuted);
            }
        }
        let (fl_comm, fl_data) = self.fri_config.mmcs.commit(first_layer_mats);

        let bivariate_beta: Challenge = challenger.sample();

        let fri_input: [Option<Vec<Challenge>>; 32] = core::array::from_fn(|i| {
            let mut ro: Vec<Challenge> = reduced_openings.get(i + 1)?.as_ref()?.clone();
            // todo send this
            let _lambda = extract_lambda(
                CircleDomain::standard(i + 1 - self.log_blowup),
                CircleDomain::standard(i + 1),
                &mut ro,
            );
            // since we unpermuted above (.inner()) we need to permute ROs
            let ro_permuted = RowMajorMatrix::new(circle_bitrev_permute(&ro), 2);
            Some(fold_bivariate(ro_permuted, bivariate_beta))
        });

        let folder = CircleFriFolder::new(bivariate_beta);

        let (fri_proof, query_indices) =
            p3_fri::prover::prove(&self.fri_config, &folder, &fri_input, challenger);

        let query_openings = query_indices
            .into_iter()
            .map(|index| {
                rounds
                    .iter()
                    .map(|(data, _)| {
                        let (opened_values, opening_proof) =
                            self.mmcs.open_batch(index, &data.mmcs_data);
                        BatchOpening {
                            opened_values,
                            opening_proof,
                        }
                    })
                    .collect()
            })
            .collect();

        (values, (fri_proof, query_openings))
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
        let (fri_proof, query_openings) = proof;
        // Batch combination challenge
        let alpha: Challenge = challenger.sample();
        let bivariate_beta: Challenge = challenger.sample();

        let fri_challenges = p3_fri::verifier::verify_shape_and_sample_challenges(
            &self.fri_config,
            &fri_proof,
            challenger,
        )
        .unwrap();

        let log_max_height = fri_proof.commit_phase_commits.len() + self.fri_config.log_blowup;

        let reduced_openings: Vec<[Challenge; 32]> = query_openings
            .iter()
            .zip(&fri_challenges.query_indices)
            .map(|(query_opening, &index)| {
                let mut ro = [Challenge::zero(); 32];
                for (batch_opening, (batch_commit, mats)) in izip!(query_opening, &rounds) {
                    let batch_dims: Vec<Dimensions> = mats
                        .iter()
                        .map(|(domain, _)| Dimensions {
                            // todo: mmcs doesn't really need width
                            width: 0,
                            height: domain.size(),
                        })
                        .collect_vec();
                    self.mmcs.verify_batch(
                        batch_commit,
                        &batch_dims,
                        index,
                        &batch_opening.opened_values,
                        &batch_opening.opening_proof,
                    )?;
                }
                Ok(ro)
            })
            .collect::<Result<Vec<_>, InputMmcs::Error>>()
            .unwrap();

        let folder = CircleFriFolder::new(bivariate_beta);

        p3_fri::verifier::verify_challenges(
            &self.fri_config,
            &folder,
            &fri_proof,
            &fri_challenges,
            &reduced_openings,
        )
        .unwrap();

        Ok(())
    }
}
