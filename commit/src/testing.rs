use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::CanSample;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

use crate::{OpenedValues, Pcs, PolynomialSpace, TwoAdicMultiplicativeCoset};

/// A trivial PCS: its commitment is simply the coefficients of each poly.
#[derive(Debug)]
pub struct TrivialPcs<Val: TwoAdicField, Dft: TwoAdicSubgroupDft<Val>> {
    pub dft: Dft,
    // degree bound
    pub log_n: usize,
    pub _phantom: PhantomData<Val>,
}

pub fn eval_coeffs_at_pt<F: Field, EF: ExtensionField<F>>(
    coeffs: &RowMajorMatrix<F>,
    x: EF,
) -> Vec<EF> {
    let mut acc = vec![EF::zero(); coeffs.width()];
    for r in (0..coeffs.height()).rev() {
        let row = coeffs.row_slice(r);
        for (acc_c, row_c) in acc.iter_mut().zip(row.as_ref().iter()) {
            *acc_c *= x;
            *acc_c += *row_c;
        }
    }
    acc
}

impl<Val, Dft, Challenge, Challenger> Pcs<Challenge, Challenger> for TrivialPcs<Val, Dft>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    Challenger: CanSample<Challenge>,

    Dft: TwoAdicSubgroupDft<Val>,

    Vec<Vec<Val>>: Serialize + for<'de> Deserialize<'de>,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = Vec<Vec<Val>>;
    type ProverData = Vec<RowMajorMatrix<Val>>;
    type Proof = ();
    type Error = ();

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        TwoAdicMultiplicativeCoset {
            log_n: log2_strict_usize(degree),
            shift: Val::one(),
        }
    }

    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let coeffs: Vec<_> = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                let log_domain_size = log2_strict_usize(domain.size());
                // for now, only commit on larger domain than natural
                assert!(log_domain_size >= self.log_n);
                assert_eq!(domain.size(), evals.height());
                // coset_idft_batch
                let mut coeffs = self.dft.idft_batch(evals);
                coeffs
                    .rows_mut()
                    .zip(domain.shift.inverse().powers())
                    .for_each(|(row, weight)| {
                        row.iter_mut().for_each(|coeff| {
                            *coeff *= weight;
                        })
                    });
                coeffs
            })
            .collect();
        (
            coeffs.clone().into_iter().map(|m| m.values).collect(),
            coeffs,
        )
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> impl Matrix<Val> + 'a {
        let mut coeffs = prover_data[idx].clone();
        assert!(domain.log_n >= self.log_n);
        coeffs.values.resize(
            coeffs.values.len() << (domain.log_n - self.log_n),
            Val::zero(),
        );
        self.dft.coset_dft_batch(coeffs, domain.shift)
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
        _challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        (
            rounds
                .into_iter()
                .map(|(coeffs_for_round, points_for_round)| {
                    coeffs_for_round
                        .iter()
                        .zip(points_for_round)
                        .map(|(coeffs_for_mat, points_for_mat)| {
                            points_for_mat
                                .into_iter()
                                .map(|pt| eval_coeffs_at_pt(coeffs_for_mat, pt))
                                .collect()
                        })
                        .collect()
                })
                .collect(),
            (),
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
                    Challenge,
                    // values at this point
                    Vec<Challenge>,
                )>,
            )>,
        )>,
        _proof: &Self::Proof,
        _challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        for (comm, round_opening) in rounds {
            for (coeff_vec, (domain, points_and_values)) in comm.into_iter().zip(round_opening) {
                let width = coeff_vec.len() / domain.size();
                assert_eq!(width * domain.size(), coeff_vec.len());
                let coeffs = RowMajorMatrix::new(coeff_vec, width);
                for (pt, values) in points_and_values {
                    assert_eq!(eval_coeffs_at_pt(&coeffs, pt), values);
                }
            }
        }
        Ok(())
    }
}
