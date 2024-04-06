use alloc::vec::Vec;

use itertools::izip;
use p3_commit::{Mmcs, OpenedValues, Pcs};
use p3_field::extension::ComplexExtendable;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::cfft::Cfft;
use crate::domain::CircleDomain;
use crate::util::{univariate_to_point, v_n};

#[derive(Debug)]
pub struct CirclePcs<Val: Field, InputMmcs> {
    pub log_blowup: usize,
    pub cfft: Cfft<Val>,
    pub mmcs: InputMmcs,
}

#[derive(Debug)]
pub struct ProverData<Val, MmcsData> {
    committed_domains: Vec<CircleDomain<Val>>,
    mmcs_data: MmcsData,
}

impl<Val, InputMmcs, Challenge, Challenger> Pcs<Challenge, Challenger> for CirclePcs<Val, InputMmcs>
where
    Val: ComplexExtendable,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
{
    type Domain = CircleDomain<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = ProverData<Val, InputMmcs::ProverData<RowMajorMatrix<Val>>>;
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
        mat.clone()
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
                                let zeta_point = univariate_to_point(zeta).unwrap();
                                let basis: Vec<Challenge> = domain.lagrange_basis(zeta_point);
                                let v_n_at_zeta =
                                    v_n(zeta_point.real(), log_n) - v_n(domain.shift.real(), log_n);
                                mat.columnwise_dot_product(&basis)
                                    // columnwise_dot_product(&mat, basis.into_iter())
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
