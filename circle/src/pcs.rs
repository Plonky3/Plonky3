use core::marker::PhantomData;

use alloc::vec;
use alloc::vec::Vec;
use itertools::Itertools;
use p3_commit::{DirectMmcs, OpenedValues, Pcs, PolynomialDomain};
use p3_field::{extension::ComplexExtendable, ExtensionField};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix, MatrixRows,
};
use p3_util::log2_strict_usize;

use crate::{
    cfft::Cfft,
    domain::CircleDomain,
    twiddles::TwiddleCache,
    util::{gemv_tr, lagrange_basis, univariate_to_point, v_n},
};

pub struct CirclePcs<Val, InputMmcs> {
    pub log_blowup: usize,
    pub cfft: Cfft<Val>,
    pub mmcs: InputMmcs,
}

impl<Val, InputMmcs, Challenge, Challenger> Pcs<Challenge, Challenger> for CirclePcs<Val, InputMmcs>
where
    Val: ComplexExtendable,
    Challenge: ExtensionField<Val>,
    InputMmcs: 'static + for<'a> DirectMmcs<Val, Mat<'a> = RowMajorMatrixView<'a, Val>>,
{
    type Domain = CircleDomain<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData;
    type Proof = ();
    type Error = ();

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        CircleDomain::new(log2_strict_usize(degree))
    }

    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let ldes = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                // todo: bitrev?
                self.cfft.lde_batch(evals, self.log_blowup)
            })
            .collect_vec();
        self.mmcs.commit(ldes)
    }

    fn get_evaluations_on_domain(
        &self,
        prover_data: &Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> RowMajorMatrix<Val> {
        let mat = self.mmcs.get_matrices(prover_data)[idx];
        assert_eq!(mat.height(), 1 << domain.log_n);
        mat.to_row_major_matrix()
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
        let values: OpenedValues<Challenge> = rounds
            .into_iter()
            .map(|(data, points_for_mats)| {
                let mats = self.mmcs.get_matrices(data);
                points_for_mats
                    .into_iter()
                    .zip(mats)
                    .map(|(points_for_mat, mat)| {
                        let log_n = log2_strict_usize(mat.height());
                        points_for_mat
                            .into_iter()
                            .map(|zeta| {
                                let zeta_point = univariate_to_point(zeta);
                                let basis: Vec<Challenge> = lagrange_basis(zeta_point, log_n);
                                let v_n_at_zeta = v_n(zeta_point.real(), log_n);
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
        // todo: fri
        (values, ())
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::util::{eval_circle_polys, univariate_to_point};

    use super::*;
    use alloc::vec;
    use p3_challenger::DuplexChallenger;
    use p3_mds::mersenne31::MdsMatrixMersenne31;
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_mersenne_31::Mersenne31;
    use p3_poseidon::Poseidon;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{thread_rng, Rng};

    type Val = Mersenne31;
    type Challenge = Mersenne31;
    type Perm = Poseidon<Val, MdsMatrixMersenne31, 16, 5>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs = FieldMerkleTreeMmcs<Val, MyHash, MyCompress, 8>;
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
        let pcs = CirclePcs {
            log_blowup: 1,
            cfft: cfft.clone(),
            mmcs,
        };

        let coeffs = RowMajorMatrix::<Val>::rand(&mut rng, n, 1);
        let evals = cfft.icfft_batch(coeffs.clone());
        // let domain = pcs.natural_domain_for_degree(n);
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
    }
}
