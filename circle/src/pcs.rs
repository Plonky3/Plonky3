use core::marker::PhantomData;

use alloc::vec;
use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs, PolynomialSpace};
use p3_field::extension::{Complex, ComplexExtendable};
use p3_field::{batch_multiplicative_inverse, AbstractField, ExtensionField, Field};
use p3_fri::{FriConfig, FriFolder, FriProof, PowersReducer};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use crate::cfft::Cfft;
use crate::deep_quotient::{extract_lambda, is_low_degree};
use crate::domain::CircleDomain;
use crate::folding::{
    circle_bitrev_idx, circle_bitrev_idx_inv, circle_bitrev_permute, circle_bitrev_permute_inv,
    fold_bivariate, fold_bivariate_row, CircleBitrevPerm, CircleBitrevView, CircleFriFolder,
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
        // first layer commitment
        FriMmcs::Commitment,
        // lambdas
        Vec<Challenge>,
        // for each index
        Vec<(
            // for each round, input openings
            Vec<BatchOpening<Val, InputMmcs>>,
            // first layer opening
            BatchOpening<Challenge, FriMmcs>,
        )>,
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
        let mut lambdas = vec![];
        for i in 0..31 {
            if let Some(ro) = reduced_openings.get(i + 1).unwrap() {
                let mut ro = ro.clone();
                // todo send this
                let lambda = extract_lambda(
                    CircleDomain::standard(i + 1 - self.log_blowup),
                    CircleDomain::standard(i + 1),
                    &mut ro,
                );
                println!("orig domain size: {}", i + 1 - self.log_blowup);
                println!("lde domain size: {}", i + 1);
                lambdas.push(lambda);
                debug_assert!(is_low_degree(
                    &RowMajorMatrix::new_col(ro.clone()).flatten_to_base()
                ));
                // since we unpermuted above (.inner()) we need to permute ROs
                let ro_permuted = RowMajorMatrix::new(circle_bitrev_permute(&ro), 2);
                first_layer_mats.push(ro_permuted);
            }
        }
        let (first_layer_comm, first_layer_data) = self.fri_config.mmcs.commit(first_layer_mats);

        challenger.observe(first_layer_comm.clone());

        let bivariate_beta: Challenge = challenger.sample();
        let folder = CircleFriFolder::new(bivariate_beta);

        let mut fri_input: [Option<Vec<Challenge>>; 32] = core::array::from_fn(|_| None);

        for mat in self.fri_config.mmcs.get_matrices(&first_layer_data) {
            let v = fold_bivariate(mat.as_view(), bivariate_beta);
            let log_height = log2_strict_usize(v.len());
            fri_input[log_height] = Some(v);
        }

        let (fri_proof, query_indices) =
            p3_fri::prover::prove(&self.fri_config, &folder, &fri_input, challenger);

        println!("=== PROVE: query index {} ===", query_indices[0]);
        let idx = query_indices[0];
        for (i, ro) in reduced_openings.iter().enumerate() {
            if let Some(ro) = ro {
                let orig_idx = circle_bitrev_idx(idx << 1, log2_strict_usize(ro.len()));
                println!("ro[i={i}][orig_idx={orig_idx}] = {}", ro[orig_idx]);
            }
            if let Some(fi) = &fri_input[i] {
                println!("fri_input[i={i}][idx={idx}] = {}", fi[idx]);
            }
        }

        let query_openings = query_indices
            .into_iter()
            .map(|index| {
                let input_opening = rounds
                    .iter()
                    .map(|(data, _)| {
                        let (opened_values, opening_proof) =
                            self.mmcs.open_batch(index << 1, &data.mmcs_data);
                        BatchOpening {
                            opened_values,
                            opening_proof,
                        }
                    })
                    .collect();
                let (first_layer_values, first_layer_proof) =
                    self.fri_config.mmcs.open_batch(index, &first_layer_data);
                let first_layer_opening = BatchOpening {
                    opened_values: first_layer_values,
                    opening_proof: first_layer_proof,
                };
                (input_opening, first_layer_opening)
            })
            .collect();

        (
            values,
            (fri_proof, first_layer_comm, lambdas, query_openings),
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
        let (fri_proof, first_layer_comm, lambdas, query_openings) = proof;
        // Batch combination challenge
        let alpha: Challenge = challenger.sample();
        challenger.observe(first_layer_comm.clone());
        let bivariate_beta: Challenge = challenger.sample();

        let fri_challenges = p3_fri::verifier::verify_shape_and_sample_challenges(
            &self.fri_config,
            &fri_proof,
            challenger,
        )
        .unwrap();

        // let log_global_max_height = fri_proof.commit_phase_commits.len() + self.fri_config.log_blowup + 1;

        let alpha_reducer = PowersReducer::<Val, Challenge>::new(alpha, 1024);

        // TODO: FRI MUST sample 1 more bit!! query must be one bit higher!!

        let reduced_openings: Vec<[Challenge; 32]> = query_openings
            .iter()
            .zip(&fri_challenges.query_indices)
            .map(|((input_openings, first_layer_opening), &index)| {
                println!("=== VERIFY: query index = {}: ===", index);

                let mut ro = [Challenge::zero(); 32];
                let mut num_reduced = [0; 32];
                for (batch_opening, (batch_commit, mats)) in izip!(input_openings, &rounds) {
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
                        index << 1,
                        &batch_opening.opened_values,
                        &batch_opening.opening_proof,
                    )?;
                    for (ps_at_x, (mat_domain, mat_points_and_values)) in
                        izip!(&batch_opening.opened_values, mats)
                    {
                        let log_orig_domain_size = log2_strict_usize(mat_domain.size());
                        let log_height = log_orig_domain_size + self.fri_config.log_blowup;
                        let orig_idx = circle_bitrev_idx(index << 1, log_height);

                        let shift = Val::circle_two_adic_generator(log_orig_domain_size + 2);
                        let g = Val::circle_two_adic_generator(log_orig_domain_size + 1);
                        let x = shift * g.exp_u64(orig_idx as u64);

                        for (zeta, ps_at_zeta) in mat_points_and_values {
                            let zeta_point = univariate_to_point(*zeta).unwrap();

                            let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);
                            let alpha_pow_width = alpha.exp_u64(ps_at_x.len() as u64);
                            num_reduced[log_height] += 2 * ps_at_x.len();

                            let x_rotate_zeta: Complex<Challenge> =
                                x.rotate(zeta_point.conjugate());

                            let v_gamma_re: Challenge = Challenge::one() - x_rotate_zeta.real();
                            let v_gamma_im: Challenge = x_rotate_zeta.imag();

                            let lhs_num = v_gamma_re - alpha_pow_width * v_gamma_im;
                            let lhs_denom = v_gamma_re.square() + v_gamma_im.square();
                            let inv_lhs_denom = lhs_denom.inverse();

                            let alpha_pow_ps_at_zeta = alpha_reducer.reduce_ext(&ps_at_zeta);

                            ro[log_height] += lhs_num
                                * inv_lhs_denom
                                * alpha_pow_offset
                                * (alpha_reducer.reduce_base(&ps_at_x) - alpha_pow_ps_at_zeta);
                        }
                    }
                }

                let first_layer_dims = num_reduced
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &nr)| {
                        if nr != 0 {
                            Some(Dimensions {
                                width: 0,
                                height: 1 << i,
                            })
                        } else {
                            None
                        }
                    })
                    .collect_vec();

                self.fri_config
                    .mmcs
                    .verify_batch(
                        first_layer_comm,
                        &first_layer_dims,
                        index,
                        &first_layer_opening.opened_values,
                        &first_layer_opening.opening_proof,
                    )
                    .expect("first layer verify");

                let mut fri_input = [Challenge::zero(); 32];

                let mut first_layer_value_iter =
                    first_layer_opening.opened_values.iter().zip_eq(lambdas);
                for (log_height, &nr) in num_reduced.iter().enumerate() {
                    if nr != 0 {
                        assert!(log_height > 0);

                        let lde_size = log_height;
                        let orig_size = log_height - self.fri_config.log_blowup;

                        let (fl_values, &lambda) = first_layer_value_iter.next().unwrap();

                        let shift = Val::circle_two_adic_generator(lde_size + 1);
                        let g = Val::circle_two_adic_generator(lde_size);
                        let orig_idx = circle_bitrev_idx(index << 1, lde_size);
                        let x = shift * g.exp_u64(orig_idx as u64);

                        let v_n_at_x = v_n(x.real(), orig_size);

                        let lambda_corrected = ro[log_height] - lambda * v_n_at_x;

                        dbg!(ro[log_height], fl_values, v_n_at_x, lambda);
                        assert_eq!(lambda_corrected, fl_values[0]);

                        fri_input[log_height - 1] = fold_bivariate_row(
                            index >> 1,
                            orig_size - 1,
                            bivariate_beta,
                            fl_values.iter().cloned(),
                        );
                    }
                }

                Ok(fri_input)
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

#[cfg(test)]
mod tests {
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_commit::{ExtensionMmcs, Pcs};
    use p3_keccak::Keccak256Hash;
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_mersenne_31::Mersenne31;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn circle_pcs() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);

        type Val = Mersenne31;
        type Challenge = Mersenne31;
        // type Challenge = BinomialExtensionField<Mersenne31, 3>;

        type ByteHash = Keccak256Hash;
        type FieldHash = SerializingHasher32<ByteHash>;
        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);

        type MyCompress = CompressionFunctionFromHasher<u8, ByteHash, 2, 32>;
        let compress = MyCompress::new(byte_hash);

        type ValMmcs = FieldMerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
        let val_mmcs = ValMmcs::new(field_hash, compress);

        type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

        let fri_config = FriConfig {
            log_blowup: 1,
            num_queries: 2,
            proof_of_work_bits: 1,
            mmcs: challenge_mmcs,
        };

        type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
        let pcs = Pcs {
            log_blowup: 1,
            cfft: Cfft::default(),
            mmcs: val_mmcs,
            fri_config,
        };

        let log_n = 10;

        let d = <Pcs as p3_commit::Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &pcs,
            1 << log_n,
        );

        // let d = pcs.natural_domain_for_degree(1 << log_n);
        let evals = RowMajorMatrix::rand(&mut rng, 1 << log_n, 1);

        let (comm, data) =
            <Pcs as p3_commit::Pcs<Challenge, Challenger>>::commit(&pcs, vec![(d, evals)]);

        let zeta: Challenge = rng.gen();

        let mut chal = Challenger::from_hasher(vec![], byte_hash);
        let (values, proof) = pcs.open(vec![(&data, vec![vec![zeta]])], &mut chal);

        let mut chal = Challenger::from_hasher(vec![], byte_hash);
        pcs.verify(
            vec![(comm, vec![(d, vec![(zeta, values[0][0][0].clone())])])],
            &proof,
            &mut chal,
        )
        .expect("verify err");
    }
}
