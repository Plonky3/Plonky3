use alloc::vec::Vec;

use itertools::izip;
use p3_commit::{DirectMmcs, OpenedValues, Pcs};
use p3_field::extension::ComplexExtendable;
use p3_field::ExtensionField;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::{Matrix, MatrixRows};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::cfft::Cfft;
use crate::domain::CircleDomain;
use crate::util::{gemv_tr, univariate_to_point, v_n};

pub struct CirclePcs<Val, InputMmcs> {
    pub log_blowup: usize,
    pub cfft: Cfft<Val>,
    pub mmcs: InputMmcs,
}

pub struct ProverData<Val, MmcsData> {
    committed_domains: Vec<CircleDomain<Val>>,
    mmcs_data: MmcsData,
}

impl<Val, InputMmcs, Challenge, Challenger> Pcs<Challenge, Challenger> for CirclePcs<Val, InputMmcs>
where
    Val: ComplexExtendable,
    Challenge: ExtensionField<Val>,
    InputMmcs: 'static + for<'a> DirectMmcs<Val, Mat<'a> = RowMajorMatrixView<'a, Val>>,
{
    type Domain = CircleDomain<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = ProverData<Val, InputMmcs::ProverData>;
    type Proof = ();
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
                // bitrev for fri?
                let lde = self.cfft.lde(evals, domain, committed_domain);
                (committed_domain, lde)
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

    fn get_evaluations_on_domain(
        &self,
        data: &Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> RowMajorMatrix<Val> {
        // TODO do this correctly
        let mat = self.mmcs.get_matrices(&data.mmcs_data)[idx];
        assert_eq!(mat.height(), 1 << domain.log_n);
        assert_eq!(domain, data.committed_domains[idx]);
        mat.to_row_major_matrix()
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
        _challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        let values: OpenedValues<Challenge> = rounds
            .into_iter()
            .map(|(data, points_for_mats)| {
                let mats = self.mmcs.get_matrices(&data.mmcs_data);
                izip!(&data.committed_domains, mats, points_for_mats)
                    .map(|(domain, mat, points_for_mat)| {
                        let log_n = log2_strict_usize(mat.height());
                        points_for_mat
                            .into_iter()
                            .map(|zeta| {
                                let zeta_point = univariate_to_point(zeta);
                                let basis: Vec<Challenge> = domain.lagrange_basis(zeta_point);
                                let v_n_at_zeta =
                                    v_n(zeta_point.real(), log_n) - v_n(domain.shift.real(), log_n);
                                gemv_tr(mat, basis)
                                    .into_iter()
                                    .map(|x| x * v_n_at_zeta)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        // todo: fri prove
        (values, ())
    }

    fn verify(
        &self,
        // For each round:
        _rounds: Vec<(
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
        _proof: &Self::Proof,
        _challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // todo: fri verify
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use p3_challenger::DuplexChallenger;
    use p3_mds::mersenne31::MdsMatrixMersenne31;
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_mersenne_31::Mersenne31;
    use p3_poseidon::Poseidon;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::thread_rng;

    use super::*;

    type Val = Mersenne31;
    type Challenge = Mersenne31;
    type Perm = Poseidon<Val, MdsMatrixMersenne31, 16, 5>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs = FieldMerkleTreeMmcs<Val, u8, MyHash, MyCompress, 8>;
    type Challenger = DuplexChallenger<Val, Perm, 16>;

    #[test]
    fn circle_pcs() {
        let log_n = 3;
        let n = 1 << log_n;
        let mut rng = thread_rng();

        let cfft = Cfft::<Val>::default();

        let perm = Perm::new_from_rng(3, 22, MdsMatrixMersenne31, &mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let mmcs = ValMmcs::new(hash, compress);

        type MyPcs = CirclePcs<Val, ValMmcs>;
        let _pcs = CirclePcs {
            log_blowup: 1,
            cfft: cfft.clone(),
            mmcs,
        };

        let coeffs = RowMajorMatrix::<Val>::rand(&mut rng, n, 1);
        let _evals = cfft.icfft_batch(coeffs.clone());

        // let domain = pcs.natural_domain_for_degree(n);
        /*
        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, n);

        // let (commit, data) = pcs.commit(vec![(domain, evals)]);
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, evals)]);

        let zeta: Challenge = rng.gen();

        // let zeta_point = univariate_to_point(zeta);
        let coeffs_at_zeta = eval_circle_polys(&coeffs, univariate_to_point(zeta));

        let mut challenger = Challenger::new(perm.clone());
        let (openings, proof) = pcs.open(vec![(&data, vec![vec![zeta]])], &mut challenger);
        assert_eq!(&openings[0][0][0], &coeffs_at_zeta);
        */
    }
}
