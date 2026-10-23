mod multilinear;
mod pcs;

use alloc::vec::Vec;
use core::marker::PhantomData;

pub use multilinear::assert_multilinear_commit_contract;
use p3_challenger::CanSample;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use p3_util::zip_eq::zip_eq;
pub use pcs::assert_pcs_opening_contract;
use serde::{Deserialize, Serialize};

use crate::{
    CommitmentOpening, MatrixOpening, OpenedValues, OpeningRequest, Pcs, PointOpening,
    PolynomialSpace, UnivariateStarkPcs,
};

/// A trivial PCS: its commitment is simply the coefficients of each poly.
#[derive(Clone, Debug)]
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
    let mut acc = EF::zero_vec(coeffs.width());
    for r in (0..coeffs.height()).rev() {
        let row = coeffs.row_slice(r).unwrap();
        for (acc_c, row_c) in acc.iter_mut().zip(row.iter()) {
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
    type ProverError = core::convert::Infallible;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        // This panics if (and only if) `degree` is not a power of 2 or `degree`
        // > `1 << Val::TWO_ADICITY`.
        TwoAdicMultiplicativeCoset::new(Val::ONE, log2_strict_usize(degree)).unwrap()
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
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
                    .zip(domain.shift_inverse().powers())
                    .for_each(|(row, weight)| {
                        row.iter_mut().for_each(|coeff| {
                            *coeff *= weight;
                        });
                    });
                coeffs
            })
            .collect();
        Ok((
            coeffs.clone().into_iter().map(|m| m.values).collect(),
            coeffs,
        ))
    }

    fn open(
        &self,
        // For each round,
        rounds: Vec<OpeningRequest<'_, Self::ProverData, Challenge>>,
        _challenger: &mut Challenger,
    ) -> Result<(OpenedValues<Challenge>, Self::Proof), Self::ProverError> {
        Ok((
            rounds
                .into_iter()
                .map(
                    |OpeningRequest {
                         prover_data: coeffs_for_round,
                         points: points_for_round,
                     }| {
                        // ensure that each matrix corresponds to a set of opening points
                        debug_assert_eq!(coeffs_for_round.len(), points_for_round.len());
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
                    },
                )
                .collect(),
            (),
        ))
    }

    fn verify(
        &self,
        // For each round:
        rounds: Vec<CommitmentOpening<Challenge, Self::Commitment, Self::Domain>>,
        _proof: &Self::Proof,
        _challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        for CommitmentOpening {
            commitment: comm,
            matrices: round_opening,
        } in rounds
        {
            for (
                coeff_vec,
                MatrixOpening {
                    domain,
                    points: points_and_values,
                },
            ) in zip_eq(comm, round_opening, ())?
            {
                // The claimed domain fixes the height, so it must divide the coefficient count.
                let width = coeff_vec.len() / domain.size();
                if width * domain.size() != coeff_vec.len() {
                    return Err(());
                }
                let coeffs = RowMajorMatrix::new(coeff_vec, width);
                // Every claimed point must carry the committed columns' values there.
                for PointOpening { point: pt, values } in points_and_values {
                    if eval_coeffs_at_pt(&coeffs, pt) != values {
                        return Err(());
                    }
                }
            }
        }
        Ok(())
    }
}

impl<Val, Dft, Challenge, Challenger> UnivariateStarkPcs<Challenge, Challenger>
    for TrivialPcs<Val, Dft>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    Challenger: CanSample<Challenge>,
    Dft: TwoAdicSubgroupDft<Val>,
    Vec<Vec<Val>>: Serialize + for<'de> Deserialize<'de>,
{
    type EvaluationsOnDomain<'a> = Dft::Evaluations;

    const ZK: bool = false;

    fn log_max_trace_height(&self) -> usize {
        Val::TWO_ADICITY
    }

    fn log_min_trace_height(&self) -> usize {
        // Multiplicative coset selectors are defined at every height, down to a single row.
        0
    }

    fn commit_quotient(
        &self,
        quotient_domain: Self::Domain,
        quotient_evaluations: RowMajorMatrix<crate::Val<Self::Domain>>,
        num_chunks: usize,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        let quotient_sub_evaluations =
            quotient_domain.split_evals(num_chunks, quotient_evaluations);
        let quotient_sub_domains = quotient_domain.split_domains(num_chunks);

        Pcs::<Challenge, Challenger>::commit(
            self,
            quotient_sub_domains
                .into_iter()
                .zip(quotient_sub_evaluations),
        )
    }

    fn get_quotient_ldes(
        &self,
        _evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
        _num_chunks: usize,
    ) -> Result<Vec<RowMajorMatrix<crate::Val<Self::Domain>>>, Self::ProverError> {
        unimplemented!("This PCS does not support computing of LDEs");
    }

    fn commit_ldes(
        &self,
        _ldes: Vec<RowMajorMatrix<Val>>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        unimplemented!("This PCS does not support computing of LDEs");
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let mut coeffs = prover_data[idx].clone();
        // The coefficients hold one row per point of the domain they were committed over.
        // A smaller target domain would fold coefficients onto each other.
        assert!(domain.size() >= coeffs.height());
        // Zero rows extend them to one row per point of the target domain.
        // Why: the target height is the domain's size, whatever height the commit domain had.
        coeffs
            .values
            .resize(coeffs.width() * domain.size(), Val::ZERO);
        self.dft.coset_dft_batch(coeffs, domain.shift())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, default_babybear_poseidon2_16};
    use p3_challenger::DuplexChallenger;
    use p3_dft::Radix2DitParallel;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::interpolation::Interpolate;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Dft = Radix2DitParallel<F>;

    // A challenger the trivial scheme never reads.
    struct Unused;

    impl CanSample<F> for Unused {
        fn sample(&mut self) -> F {
            F::ZERO
        }
    }

    fn trivial(log_n: usize) -> TrivialPcs<F, Dft> {
        TrivialPcs {
            dft: Dft::default(),
            log_n,
            _phantom: PhantomData,
        }
    }

    #[test]
    fn evaluations_have_the_domain_size_when_committed_above_the_degree_bound() {
        // Fixture state: degree bound 2^0, one column committed over the 2 points {1, -1}.
        //
        //     p(1) = 3, p(-1) = 5   ->   p(X) = 4 - X
        let pcs = trivial(0);
        let trace = TwoAdicMultiplicativeCoset::new(F::ONE, 1).unwrap();
        let evals = RowMajorMatrix::new_col(vec![F::from_u8(3), F::from_u8(5)]);
        let (_, data) =
            <TrivialPcs<F, Dft> as Pcs<F, Unused>>::commit(&pcs, [(trace, evals)]).unwrap();

        // Read back over a coset of 4 points.
        let lde = TwoAdicMultiplicativeCoset::new(F::GENERATOR, 2).unwrap();
        let out = <TrivialPcs<F, Dft> as UnivariateStarkPcs<F, Unused>>::get_evaluations_on_domain(
            &pcs, &data, 0, lde,
        );

        // Invariant: one row per point of the target domain, holding p at that point.
        //
        // Mutation: padding by 2^(log_size - log_n) gives 2 * 2^2 = 8 rows here.
        assert_eq!(out.height(), lde.size());
        for (i, x) in lde.iter().enumerate() {
            assert_eq!(out.get(i, 0), Some(F::from_u8(4) - x), "row {i}");
        }
    }

    #[test]
    fn the_trivial_scheme_meets_the_opening_contract() {
        // Fixture state: one commitment of two matrices over the 4-point subgroup.
        //
        //     matrix 0: 4 x 2,   matrix 1: 4 x 1
        let mut rng = SmallRng::seed_from_u64(1);
        let domain = TwoAdicMultiplicativeCoset::new(F::ONE, 2).unwrap();
        let rounds = vec![vec![
            (domain, RowMajorMatrix::<F>::rand(&mut rng, 4, 2)),
            (domain, RowMajorMatrix::<F>::rand(&mut rng, 4, 1)),
        ]];
        let pcs = trivial(2);
        let challenger = DuplexChallenger::<F, _, 16, 8>::new(default_babybear_poseidon2_16());

        // Invariant: every false claim the contract tries is an error, never a panic.
        //
        // The expected values come from interpolating the evaluations, not from the scheme.
        assert_pcs_opening_contract::<_, EF, _>(&pcs, &challenger, &rounds, |r, m, z| {
            rounds[r][m].1.interpolate_coset(F::ONE, z)
        });
    }

    #[test]
    fn a_claim_of_the_wrong_width_is_an_error() {
        // Fixture state: 2 coefficients, claimed over a domain of 4 points.
        //
        //     2 = width * 4 has no solution, so the claim cannot describe this commitment.
        let pcs = trivial(0);
        let domain = TwoAdicMultiplicativeCoset::new(F::ONE, 2).unwrap();
        let claim = CommitmentOpening {
            commitment: vec![vec![F::ONE, F::TWO]],
            matrices: vec![MatrixOpening {
                domain,
                points: vec![],
            }],
        };
        let mut challenger = DuplexChallenger::<F, _, 16, 8>::new(default_babybear_poseidon2_16());

        assert!(
            <TrivialPcs<F, Dft> as Pcs<EF, _>>::verify(&pcs, vec![claim], &(), &mut challenger)
                .is_err()
        );
    }
}
