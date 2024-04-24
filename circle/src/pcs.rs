use core::marker::PhantomData;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs, PolynomialSpace};
use p3_field::extension::{Complex, ComplexExtendable, ExtensionPowersReducer};
use p3_field::{ExtensionField, Field};
use p3_fri::{FriConfig, FriProof};
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
pub struct InputProof<Val: Field, Challenge: Field, InputMmcs: Mmcs<Val>, FriMmcs: Mmcs<Challenge>>
{
    input_openings: Vec<BatchOpening<Val, InputMmcs>>,
    first_layer_siblings: Vec<Challenge>,
    first_layer_proof: FriMmcs::Proof,
}

#[derive(Serialize, Deserialize)]
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
    fri_proof:
        FriProof<Challenge, FriMmcs, Witness, InputProof<Val, Challenge, InputMmcs, FriMmcs>>,
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

        let mut alpha_reducer = ExtensionPowersReducer::<Val, Challenge>::new(alpha);

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

        let values: OpenedValues<Challenge> = rounds
            .iter()
            .map(|(data, points_for_mats)| {
                let mats = self.mmcs.get_matrices(&data.mmcs_data);
                izip!(&data.committed_domains, mats, points_for_mats)
                    .map(|(lde_domain, permuted_mat, points_for_mat)| {
                        // Get the unpermuted matrix.
                        let mat = &permuted_mat.inner;

                        let log_height = log2_strict_usize(mat.height());
                        let (alpha_offset, reduced_opening_for_log_height) =
                            reduced_openings.entry(log_height).or_insert_with(|| {
                                (Challenge::one(), vec![Challenge::zero(); mat.height()])
                            });

                        let alpha_pow_width = alpha.exp_u64(mat.width() as u64);
                        alpha_reducer.prepare_for_width(mat.width());

                        points_for_mat
                            .iter()
                            .map(|&zeta| {
                                let zeta_point = univariate_to_point(zeta).unwrap();

                                // Staying in evaluation form, we lagrange interpolate to get the value of
                                // each p at zeta.
                                // todo: we only need half of the values to interpolate, but how?
                                let ps_at_zeta: Vec<Challenge> =
                                    info_span!("compute opened values with Lagrange interpolation")
                                        .in_scope(|| {
                                            // todo: cache basis
                                            let basis: Vec<Challenge> =
                                                lde_domain.lagrange_basis(zeta_point);
                                            let v_n_at_zeta = lde_domain.zp_at_point(zeta);
                                            mat.columnwise_dot_product(&basis)
                                                .into_iter()
                                                .map(|x| x * v_n_at_zeta)
                                                .collect()
                                        });

                                // Reduce this matrix, as a deep quotient, into one column with powers of α.
                                let mat_ros = deep_quotient_reduce_matrix(
                                    lde_domain,
                                    mat,
                                    zeta_point,
                                    &ps_at_zeta,
                                    &alpha_reducer,
                                    alpha_pow_width,
                                );

                                // Fold it into our running reduction, offset by alpha_offset.
                                reduced_opening_for_log_height
                                    .par_iter_mut()
                                    .zip(mat_ros)
                                    .for_each(|(ro, mat_ro)| {
                                        *ro += *alpha_offset * mat_ro;
                                    });

                                // Update alpha_offset from α^i -> α^(i + 2 * width)
                                *alpha_offset *= alpha_pow_width.square();

                                ps_at_zeta
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // Iterate over our reduced columns and extract lambda - the multiple of the vanishing polynomial
        // which may appear in the reduced quotient due to CFFT dimension gap.

        let mut lambdas = vec![];
        let first_layer_mats: Vec<RowMajorMatrix<Challenge>> = reduced_openings
            .into_iter()
            .map(|(log_height, (_, mut ro))| {
                assert!(log_height > 0);
                // Todo: use domain methods more intelligently
                let lambda = extract_lambda(
                    CircleDomain::standard(log_height - self.log_blowup),
                    CircleDomain::standard(log_height),
                    &mut ro,
                );
                lambdas.push(lambda);
                // We have been working with reduced openings in natural order, but now we are ready
                // to start FRI, so go to circle bitrev order, and prepare for first layer fold
                // with 2 siblings per leaf.
                RowMajorMatrix::new(circle_bitrev_permute(&ro), 2)
            })
            .collect();

        // Commit to reduced openings at each log_height, so we can challenge a global
        // folding factor for all first layers, which we use for a "manual" (not part of p3-fri) fold.
        // This is necessary because the first layer of folding uses different twiddles, so it's easiest
        // to do it here, before p3-fri.

        let (first_layer_commitment, first_layer_data) =
            self.fri_config.mmcs.commit(first_layer_mats);
        challenger.observe(first_layer_commitment.clone());
        let bivariate_beta: Challenge = challenger.sample();

        // Fold all first layers at bivariate_beta.

        let fri_input: Vec<Vec<Challenge>> = self
            .fri_config
            .mmcs
            .get_matrices(&first_layer_data)
            .into_iter()
            .map(|m| fold_bivariate(bivariate_beta, m.as_view()))
            .collect();

        let g: CircleFriFolder<Val, InputProof<Val, Challenge, InputMmcs, FriMmcs>> =
            CircleFriFolder(PhantomData);

        let fri_proof =
            p3_fri::prover::prove(&g, &self.fri_config, fri_input, challenger, |index| {
                // CircleFriFolder asks for an extra query index bit, so we use that here to index
                // the first layer fold.

                // Open the input (big opening, lots of columns) at the full index...
                let input_openings = rounds
                    .iter()
                    .map(|(data, _)| {
                        let (opened_values, opening_proof) =
                            self.mmcs.open_batch(index, &data.mmcs_data);
                        BatchOpening {
                            opened_values,
                            opening_proof,
                        }
                    })
                    .collect();

                // We committed to first_layer in pairs, so open the reduced index and include the sibling
                // as part of the input proof.
                let (first_layer_values, first_layer_proof) = self
                    .fri_config
                    .mmcs
                    .open_batch(index >> 1, &first_layer_data);
                let first_layer_siblings = first_layer_values
                    .iter()
                    .map(|v| v[(index ^ 1) & 1])
                    .collect();
                InputProof {
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
        // Batch combination challenge
        let alpha: Challenge = challenger.sample();
        challenger.observe(proof.first_layer_commitment.clone());
        let bivariate_beta: Challenge = challenger.sample();

        let max_width = rounds
            .iter()
            .flat_map(|(_comm, mats)| {
                mats.iter()
                    .flat_map(|(_domain, pts)| pts.iter().map(|(_pt, vals)| vals.len()))
            })
            .max()
            .unwrap();
        let mut alpha_reducer = ExtensionPowersReducer::<Val, Challenge>::new(alpha);
        alpha_reducer.prepare_for_width(max_width);

        let g: CircleFriFolder<Val, InputProof<Val, Challenge, InputMmcs, FriMmcs>> =
            CircleFriFolder(PhantomData);

        p3_fri::verifier::verify(
            &g,
            &self.fri_config,
            &proof.fri_proof,
            challenger,
            |index, input_proof| {
                let full_index = index;
                let first_layer_index = index & 1;
                let rest_layers_index = index >> 1;

                // log_height -> (alpha_offset, ro)
                let mut reduced_openings = BTreeMap::<usize, (Challenge, Challenge)>::new();

                let InputProof {
                    input_openings,
                    first_layer_siblings,
                    first_layer_proof,
                } = input_proof;

                // TODO: refactor this!!

                for (batch_opening, (batch_commit, mats)) in izip!(input_openings, &rounds) {
                    let batch_dims: Vec<Dimensions> = mats
                        .iter()
                        .map(|(domain, _)| Dimensions {
                            // todo: mmcs doesn't really need width
                            width: 0,
                            height: domain.size(),
                        })
                        .collect_vec();

                    self.mmcs
                        .verify_batch(
                            batch_commit,
                            &batch_dims,
                            full_index,
                            &batch_opening.opened_values,
                            &batch_opening.opening_proof,
                        )
                        .expect("input mmcs");

                    for (ps_at_x, (mat_domain, mat_points_and_values)) in
                        izip!(&batch_opening.opened_values, mats)
                    {
                        let log_orig_domain_size = log2_strict_usize(mat_domain.size());
                        let log_height = log_orig_domain_size + self.fri_config.log_blowup;
                        let orig_idx = circle_bitrev_idx(full_index, log_height);

                        let lde_domain = CircleDomain::standard(mat_domain.log_n + 1);
                        let x = lde_domain.nth_point(orig_idx);

                        let (alpha_offset, ro) = reduced_openings
                            .entry(log_height)
                            .or_insert((Challenge::one(), Challenge::zero()));
                        let alpha_pow_width = alpha.exp_u64(ps_at_x.len() as u64);

                        for (zeta, ps_at_zeta) in mat_points_and_values {
                            let zeta_point = univariate_to_point(*zeta).unwrap();

                            let (lhs_num, lhs_denom) =
                                deep_quotient_lhs(x, zeta_point, alpha_pow_width);
                            *ro += *alpha_offset
                                * deep_quotient_reduce_row(
                                    &alpha_reducer,
                                    lhs_num,
                                    lhs_denom.inverse(),
                                    ps_at_x,
                                    alpha_reducer.reduce_ext(ps_at_zeta),
                                );

                            *alpha_offset *= alpha_pow_width.square();
                        }
                    }
                }

                let mut fl_value_pairs = vec![];

                let first_layer_dims = reduced_openings
                    .iter()
                    .map(|(log_height, _)| Dimensions {
                        width: 0,
                        height: 1 << log_height,
                    })
                    .collect_vec();

                let fri_input: Vec<(usize, Challenge)> =
                    izip!(reduced_openings, first_layer_siblings, &proof.lambdas)
                        .map(|((log_height, (_, ro)), &fl_sib, &lambda)| {
                            assert!(log_height > 0);

                            let lde_size = log_height;
                            let orig_size = log_height - self.fri_config.log_blowup;

                            let orig_idx = circle_bitrev_idx(full_index, lde_size);

                            let lde_domain = CircleDomain::standard(log_height);
                            let x: Complex<Val> = lde_domain.nth_point(orig_idx);

                            let v_n_at_x = v_n(x.real(), orig_size);

                            let lambda_corrected = ro - lambda * v_n_at_x;

                            let mut fl_values = vec![lambda_corrected; 2];
                            fl_values[first_layer_index ^ 1] = fl_sib;
                            fl_value_pairs.push(fl_values.clone());

                            (
                                log_height - 1,
                                fold_bivariate_row(
                                    index >> 2,
                                    orig_size - 1,
                                    bivariate_beta,
                                    fl_values.iter().cloned(),
                                ),
                            )
                        })
                        .collect();

                self.fri_config
                    .mmcs
                    .verify_batch(
                        &proof.first_layer_commitment,
                        &first_layer_dims,
                        rest_layers_index,
                        &fl_value_pairs,
                        first_layer_proof,
                    )
                    .expect("first layer verify");

                fri_input
            },
        )
        .expect("fri verify");

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
