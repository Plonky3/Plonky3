//! A PCS can serve generic opening clients without implementing STARK capabilities.
use p3_baby_bear::BabyBear as F;
use p3_commit::{
    CommitmentOpening, MatrixOpening, OpenedValues, OpeningRequest, Pcs, PointOpening,
};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

struct LinearPcs;
type Domain = TwoAdicMultiplicativeCoset<F>;

// Deliberately no UnivariateStarkPcs implementation, LDE support or hiding policy.
impl Pcs<F, ()> for LinearPcs {
    type Domain = Domain;
    type Commitment = Vec<Vec<F>>;
    type ProverData = Vec<Vec<F>>;
    type Proof = ();
    type Error = ();
    type ProverError = core::convert::Infallible;

    fn natural_domain_for_degree(&self, degree: usize) -> Domain {
        assert_eq!(degree, 2);
        Domain::new(F::ONE, 1).unwrap()
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Domain, RowMajorMatrix<F>)>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        let data: Vec<_> = evaluations
            .into_iter()
            .map(|(_, matrix)| matrix.values)
            .collect();
        Ok((data.clone(), data))
    }

    fn open(
        &self,
        requests: Vec<OpeningRequest<'_, Self::ProverData, F>>,
        _: &mut (),
    ) -> Result<(OpenedValues<F>, ()), Self::ProverError> {
        Ok((
            requests
                .into_iter()
                .map(|request| {
                    request
                        .prover_data
                        .iter()
                        .zip(request.points)
                        .map(|(evals, points)| {
                            points
                                .into_iter()
                                .map(|point| evaluate(evals, point))
                                .collect()
                        })
                        .collect()
                })
                .collect(),
            (),
        ))
    }

    fn verify(
        &self,
        claims: Vec<CommitmentOpening<F, Self::Commitment, Domain>>,
        _: &(),
        _: &mut (),
    ) -> Result<(), ()> {
        for claim in claims {
            if claim.commitment.len() != claim.matrices.len() {
                return Err(());
            }
            for (evals, matrix) in claim.commitment.iter().zip(claim.matrices) {
                for opening in matrix.points {
                    if evaluate(evals, opening.point) != opening.values {
                        return Err(());
                    }
                }
            }
        }
        Ok(())
    }
}

fn evaluate(evals: &[F], point: F) -> Vec<F> {
    let half = F::TWO.inverse();
    vec![(evals[0] + evals[1]) * half + point * (evals[0] - evals[1]) * half]
}

#[test]
fn core_only_pcs_preserves_round_matrix_and_point_order() {
    let pcs = LinearPcs;
    let domain = pcs.natural_domain_for_degree(2);
    // p(x) = 3 + 2x, q(x) = 7 - x, r(x) = 11 + 4x.
    let matrix = |a: u8, b: F| RowMajorMatrix::new(vec![F::from_u8(a) + b, F::from_u8(a) - b], 1);
    let (first, first_data) = pcs
        .commit([(domain, matrix(3, F::TWO)), (domain, matrix(7, -F::ONE))])
        .unwrap();
    let (second, second_data) = pcs.commit([(domain, matrix(11, F::from_u8(4)))]).unwrap();
    let x = F::from_u8(5);
    let y = F::from_u8(9);
    let (values, proof) = pcs
        .open(
            vec![
                OpeningRequest {
                    prover_data: &first_data,
                    points: vec![vec![y, x], vec![x]],
                },
                OpeningRequest {
                    prover_data: &second_data,
                    points: vec![vec![x, y]],
                },
            ],
            &mut (),
        )
        .unwrap();
    assert_eq!(
        values,
        vec![
            vec![
                vec![vec![F::from_u8(21)], vec![F::from_u8(13)]],
                vec![vec![F::TWO]]
            ],
            vec![vec![vec![F::from_u8(31)], vec![F::from_u8(47)]]]
        ]
    );
    let point = |point, value| PointOpening {
        point,
        values: vec![F::from_u8(value)],
    };
    let mut claims = vec![
        CommitmentOpening {
            commitment: first,
            matrices: vec![
                MatrixOpening {
                    domain,
                    points: vec![point(y, 21), point(x, 13)],
                },
                MatrixOpening {
                    domain,
                    points: vec![point(x, 2)],
                },
            ],
        },
        CommitmentOpening {
            commitment: second,
            matrices: vec![MatrixOpening {
                domain,
                points: vec![point(x, 31), point(y, 47)],
            }],
        },
    ];
    assert_eq!(pcs.verify(claims.clone(), &proof, &mut ()), Ok(()));
    claims[0].matrices.swap(0, 1);
    assert_eq!(pcs.verify(claims.clone(), &proof, &mut ()), Err(()));
    claims[0].matrices.swap(0, 1);
    claims[0].matrices[0].points[0].values[0] = F::from_u8(13);
    assert_eq!(pcs.verify(claims, &proof, &mut ()), Err(()));
}
