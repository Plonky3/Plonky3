use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs, PolynomialSpace};
use p3_field::extension::{Complex, ComplexExtendable};
use p3_field::{ExtensionField, Field};
use p3_fri::{FriConfig, FriProof, PowersReducer};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use crate::cfft::Cfft;
use crate::deep_quotient::{
    deep_quotient_lhs, deep_quotient_reduce_matrix, deep_quotient_reduce_row, extract_lambda,
    is_low_degree,
};
use crate::domain::CircleDomain;
use crate::folding::{
    circle_bitrev_idx, circle_bitrev_permute, fold_bivariate, fold_bivariate_row, CircleBitrevPerm,
    CircleBitrevView, CircleFriFolder,
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

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
#[allow(clippy::type_complexity)]
pub struct CirclePcsProof<
    Val: Field,
    Challenge: Field,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Witness: Serialize + for<'de2> Deserialize<'de2>,
> {
    fri_proof: FriProof<Challenge, FriMmcs, Witness>,
    first_layer_commitment: FriMmcs::Commitment,
    lambdas: Vec<Challenge>,
    // for each query index
    query_openings: Vec<(
        // for each round, input openings
        Vec<BatchOpening<Val, InputMmcs>>,
        // first layer siblings
        Vec<Challenge>,
        // first layer proof
        FriMmcs::Proof,
    )>,
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
    type Proof = CirclePcsProof<Val, Challenge, InputMmcs, FriMmcs, Challenger::Witness>;
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

        let alpha_reducer = PowersReducer::<Val, Challenge>::new(alpha, max_width);

        // log_height -> reduced openings column
        let mut reduced_openings: BTreeMap<usize, Vec<Challenge>> = BTreeMap::new();
        // log_height -> alpha^(2 * number of columns already folded in at this height)
        let mut alpha_offsets: BTreeMap<usize, Challenge> = BTreeMap::new();

        // maybe replace this with horner to make more simple?
        let mut advance_alpha_offset = |log_height: usize, width: usize| {
            let alpha_pow_width = alpha.exp_u64(width as u64);
            let alpha_offset_ptr: &mut Challenge =
                alpha_offsets.entry(log_height).or_insert(Challenge::one());
            let alpha_offset = *alpha_offset_ptr;
            *alpha_offset_ptr *= alpha_pow_width.square();
            (alpha_offset, alpha_pow_width)
        };

        let values: OpenedValues<Challenge> = rounds
            .iter()
            .map(|(data, points_for_mats)| {
                let mats = self.mmcs.get_matrices(&data.mmcs_data);
                izip!(&data.committed_domains, mats, points_for_mats)
                    .map(|(lde_domain, permuted_mat, points_for_mat)| {
                        let mat = &permuted_mat.inner;
                        let log_height = log2_strict_usize(mat.height());
                        let reduced_opening_for_log_height: &mut Vec<Challenge> = reduced_openings
                            .entry(log_height)
                            .or_insert_with(|| vec![Challenge::zero(); mat.height()]);
                        points_for_mat
                            .iter()
                            .map(|&zeta| {
                                let zeta_point = univariate_to_point(zeta).unwrap();

                                // todo: we only need half of the values to interpolate, but how?
                                let ps_at_zeta: Vec<Challenge> =
                                    info_span!("compute opened values with Lagrange interpolation")
                                        .in_scope(|| {
                                            let basis: Vec<Challenge> =
                                                lde_domain.lagrange_basis(zeta_point);
                                            let v_n_at_zeta = lde_domain.zp_at_point(zeta);
                                            mat.columnwise_dot_product(&basis)
                                                .into_iter()
                                                .map(|x| x * v_n_at_zeta)
                                                .collect()
                                        });

                                let (alpha_offset, alpha_pow_width) =
                                    advance_alpha_offset(log_height, mat.width());

                                let mat_ros = deep_quotient_reduce_matrix(
                                    lde_domain,
                                    mat,
                                    zeta_point,
                                    &ps_at_zeta,
                                    &alpha_reducer,
                                    alpha_pow_width,
                                );

                                reduced_opening_for_log_height
                                    .par_iter_mut()
                                    .zip(mat_ros)
                                    .for_each(|(ro, mat_ro)| {
                                        *ro += alpha_offset * mat_ro;
                                    });

                                ps_at_zeta
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let mut lambdas = vec![];
        let first_layer_mats: Vec<RowMajorMatrix<Challenge>> = reduced_openings
            .into_iter()
            .map(|(log_height, mut ro)| {
                assert!(log_height > 0);
                let lambda = extract_lambda(
                    CircleDomain::standard(log_height - self.log_blowup),
                    CircleDomain::standard(log_height),
                    &mut ro,
                );
                lambdas.push(lambda);
                debug_assert!(is_low_degree(
                    &RowMajorMatrix::new_col(ro.clone()).flatten_to_base()
                ));
                // since we unpermuted above (.inner()) we need to permute ROs
                RowMajorMatrix::new(circle_bitrev_permute(&ro), 2)
            })
            .collect();

        let (first_layer_commitment, first_layer_data) =
            self.fri_config.mmcs.commit(first_layer_mats);
        challenger.observe(first_layer_commitment.clone());
        let bivariate_beta: Challenge = challenger.sample();

        let fri_input: Vec<Vec<Challenge>> = self
            .fri_config
            .mmcs
            .get_matrices(&first_layer_data)
            .into_iter()
            .map(|m| fold_bivariate(bivariate_beta, m.as_view()))
            .collect();

        let (fri_proof, query_indices) = p3_fri::prover::prove::<_, _, CircleFriFolder<Val>, _>(
            &self.fri_config,
            fri_input,
            challenger,
        );

        let query_openings = query_indices
            .into_iter()
            .map(|index| {
                let first_layer_index = challenger.sample_bits(1);
                let input_opening = rounds
                    .iter()
                    .map(|(data, _)| {
                        let (opened_values, opening_proof) = self
                            .mmcs
                            .open_batch((index << 1) | first_layer_index, &data.mmcs_data);
                        BatchOpening {
                            opened_values,
                            opening_proof,
                        }
                    })
                    .collect();
                let (first_layer_values, first_layer_proof) =
                    self.fri_config.mmcs.open_batch(index, &first_layer_data);
                let first_layer_siblings = first_layer_values
                    .iter()
                    .map(|v| v[first_layer_index ^ 1])
                    .collect();
                (input_opening, first_layer_siblings, first_layer_proof)
            })
            .collect();

        (
            values,
            CirclePcsProof {
                fri_proof,
                first_layer_commitment,
                lambdas,
                query_openings,
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
        // Batch combination challenge
        let alpha: Challenge = challenger.sample();
        challenger.observe(proof.first_layer_commitment.clone());
        let bivariate_beta: Challenge = challenger.sample();

        let fri_challenges = p3_fri::verifier::verify_shape_and_sample_challenges(
            &self.fri_config,
            &proof.fri_proof,
            challenger,
        )
        .unwrap();

        let max_width = rounds
            .iter()
            .flat_map(|(_comm, mats)| {
                mats.iter()
                    .flat_map(|(_domain, pts)| pts.iter().map(|(_pt, vals)| vals.len()))
            })
            .max()
            .unwrap();

        let alpha_reducer = PowersReducer::<Val, Challenge>::new(alpha, max_width);

        let reduced_openings: Vec<[Challenge; 32]> = proof
            .query_openings
            .iter()
            .zip(&fri_challenges.query_indices)
            .map(
                |((input_openings, first_layer_siblings, first_layer_proof), &index)| {
                    let first_layer_index = challenger.sample_bits(1);
                    let full_index = (index << 1) | first_layer_index;

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
                            full_index,
                            &batch_opening.opened_values,
                            &batch_opening.opening_proof,
                        )?;
                        for (ps_at_x, (mat_domain, mat_points_and_values)) in
                            izip!(&batch_opening.opened_values, mats)
                        {
                            let log_orig_domain_size = log2_strict_usize(mat_domain.size());
                            let log_height = log_orig_domain_size + self.fri_config.log_blowup;
                            let orig_idx = circle_bitrev_idx(full_index, log_height);

                            let lde_domain = CircleDomain::standard(mat_domain.log_n + 1);
                            let x = lde_domain.nth_point(orig_idx);

                            for (zeta, ps_at_zeta) in mat_points_and_values {
                                let zeta_point = univariate_to_point(*zeta).unwrap();

                                let alpha_pow_offset =
                                    alpha.exp_u64(num_reduced[log_height] as u64);
                                let alpha_pow_width = alpha.exp_u64(ps_at_x.len() as u64);
                                num_reduced[log_height] += 2 * ps_at_x.len();

                                let (lhs_num, lhs_denom) =
                                    deep_quotient_lhs(x, zeta_point, alpha_pow_width);
                                ro[log_height] += alpha_pow_offset
                                    * deep_quotient_reduce_row(
                                        &alpha_reducer,
                                        lhs_num,
                                        lhs_denom.inverse(),
                                        ps_at_x,
                                        alpha_reducer.reduce_ext(ps_at_zeta),
                                    );
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

                    let mut fri_input = [Challenge::zero(); 32];
                    let mut fl_value_pairs = vec![];

                    let mut first_layer_iter = first_layer_siblings.iter().zip_eq(&proof.lambdas);
                    for (log_height, &nr) in num_reduced.iter().enumerate() {
                        if nr != 0 {
                            assert!(log_height > 0);

                            let lde_size = log_height;
                            let orig_size = log_height - self.fri_config.log_blowup;

                            let (&fl_sib, &lambda) = first_layer_iter.next().unwrap();

                            let orig_idx = circle_bitrev_idx(full_index, lde_size);

                            let lde_domain = CircleDomain::standard(log_height);
                            let x: Complex<Val> = lde_domain.nth_point(orig_idx);

                            let v_n_at_x = v_n(x.real(), orig_size);

                            let lambda_corrected = ro[log_height] - lambda * v_n_at_x;

                            let mut fl_values = vec![lambda_corrected; 2];
                            fl_values[first_layer_index ^ 1] = fl_sib;
                            fl_value_pairs.push(fl_values.clone());

                            fri_input[log_height - 1] = fold_bivariate_row(
                                index >> 1,
                                orig_size - 1,
                                bivariate_beta,
                                fl_values.iter().cloned(),
                            );
                        }
                    }

                    self.fri_config
                        .mmcs
                        .verify_batch(
                            &proof.first_layer_commitment,
                            &first_layer_dims,
                            index,
                            &fl_value_pairs,
                            first_layer_proof,
                        )
                        .expect("first layer verify");

                    Ok(fri_input)
                },
            )
            .collect::<Result<Vec<_>, InputMmcs::Error>>()
            .unwrap();

        p3_fri::verifier::verify_challenges::<_, _, CircleFriFolder<Val>, _>(
            &self.fri_config,
            &proof.fri_proof,
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
