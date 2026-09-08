//! Shared opening checks for transparent, batched univariate commitment backends.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::ExtensionField;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    CommitmentOpening, MatrixOpening, OpeningRequest, Pcs, PointOpening, PolynomialSpace, Val,
};

/// Check a backend's batched opening values, verification, transcript and claim-shape handling.
///
/// The caller supplies evaluation matrices grouped by commitment, and an independent evaluator
/// for their columns. The evaluator receives `(commitment_index, matrix_index, opening_point)`.
/// It must compute the expected values from the intended polynomials, not from PCS output.
///
/// Two points are sampled after absorbing all commitments. Their order alternates between
/// matrices, exercising the association of commitments, matrices, points and columns. After
/// honest verification, the next prover and verifier challenge must agree. Fresh copies of the
/// verifier transcript also check rejection of swapped point fields, changed first and last
/// values, a missing column and a missing matrix. Value and shape mutations may also change the
/// backend's transcript; the point swap preserves the values and their observation order.
/// Backend-specific proof mutations and serialization fixtures belong in the caller's tests.
///
/// Intended for transparent backends with a fixed matrix/column opening shape, such as FRI,
/// Circle and STIR. Hiding backends may use a different preprocessing/opening protocol.
///
/// # Panics
///
/// Panics on any failed contract check. The fixture must contain at least one commitment,
/// every commitment must contain a matrix, and at least one commitment must contain two matrices.
/// Every matrix must have at least one column and the height required by its domain. The supplied
/// transcript must yield two distinct, valid opening points for the backend, and at least one
/// matrix must have different expected values at those points.
#[allow(clippy::type_complexity)]
pub fn assert_pcs_opening_contract<P, Challenge, Challenger>(
    pcs: &P,
    challenger: &Challenger,
    rounds: &[Vec<(P::Domain, RowMajorMatrix<Val<P::Domain>>)>],
    expected: impl Fn(usize, usize, Challenge) -> Vec<Challenge>,
) where
    P: Pcs<Challenge, Challenger>,
    Challenge: ExtensionField<Val<P::Domain>>,
    Challenger: Clone + CanObserve<P::Commitment> + FieldChallenger<Val<P::Domain>>,
{
    assert!(!rounds.is_empty(), "fixture needs a commitment");
    for round in rounds {
        assert!(!round.is_empty(), "fixture needs a matrix per commitment");
        for (domain, matrix) in round {
            assert_ne!(matrix.width(), 0, "fixture needs nonempty columns");
            assert_eq!(domain.size(), matrix.height(), "fixture domain height");
        }
    }
    let multi_matrix_round = rounds
        .iter()
        .position(|round| round.len() >= 2)
        .expect("fixture needs a commitment with at least two matrices");

    let (commitments, prover_data): (Vec<_>, Vec<_>) = rounds
        .iter()
        .map(|round| pcs.commit(round.iter().cloned()))
        .unzip();
    let mut prover = challenger.clone();
    prover.observe_slice(&commitments);
    let points: [Challenge; 2] = [
        prover.sample_algebra_element(),
        prover.sample_algebra_element(),
    ];
    assert_ne!(
        points[0], points[1],
        "fixture needs distinct opening points"
    );
    let (point_dependent_round, point_dependent_matrix) = rounds
        .iter()
        .enumerate()
        .find_map(|(r, round)| {
            (0..round.len())
                .find(|&m| expected(r, m, points[0]) != expected(r, m, points[1]))
                .map(|m| (r, m))
        })
        .expect("fixture needs a point-dependent column to test opening-point association");
    let opening_points: Vec<Vec<Vec<Challenge>>> = rounds
        .iter()
        .enumerate()
        .map(|(r, round)| {
            (0..round.len())
                .map(|m| {
                    let mut points = points.to_vec();
                    if (r + m) % 2 == 1 {
                        points.reverse();
                    }
                    points
                })
                .collect()
        })
        .collect();
    let (opened, proof) = pcs.open(
        prover_data
            .iter()
            .zip(opening_points.clone())
            .map(|(prover_data, points)| OpeningRequest {
                prover_data,
                points,
            })
            .collect(),
        &mut prover,
    );
    assert_eq!(opened.len(), rounds.len(), "commitment opening count");

    let claims: Vec<_> = commitments
        .iter()
        .cloned()
        .enumerate()
        .map(|(r, commitment)| {
            assert_eq!(opened[r].len(), rounds[r].len(), "matrix opening count");
            let matrices = rounds[r]
                .iter()
                .enumerate()
                .map(|(m, (domain, matrix))| {
                    assert_eq!(opened[r][m].len(), 2, "point opening count");
                    let values = opening_points[r][m]
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(p, point)| {
                            let values = expected(r, m, point);
                            assert_eq!(values.len(), matrix.width(), "expected column count");
                            assert_eq!(opened[r][m][p], values, "opening at ({r}, {m}, {p})");
                            PointOpening { point, values }
                        })
                        .collect();
                    MatrixOpening {
                        domain: *domain,
                        points: values,
                    }
                })
                .collect();
            CommitmentOpening {
                commitment,
                matrices,
            }
        })
        .collect();

    let mut verifier = challenger.clone();
    verifier.observe_slice(&commitments);
    let verifier_points: [Challenge; 2] = [
        verifier.sample_algebra_element(),
        verifier.sample_algebra_element(),
    ];
    assert_eq!(verifier_points, points, "opening-point transcript");
    let mut opening_transcript = verifier.clone();
    pcs.verify(claims.clone(), &proof, &mut verifier)
        .expect("honest opening must verify");
    assert_eq!(
        prover.sample_algebra_element::<Challenge>(),
        verifier.sample_algebra_element::<Challenge>(),
        "prover and verifier transcript after opening"
    );

    // These backends observe values, not point fields. Keep those values in place so this
    // false claim must be rejected by the opening check rather than transcript divergence.
    let mut swapped_points = claims.clone();
    let matrix_points =
        &mut swapped_points[point_dependent_round].matrices[point_dependent_matrix].points;
    let (first, rest) = matrix_points.split_at_mut(1);
    core::mem::swap(&mut first[0].point, &mut rest[0].point);
    assert!(
        pcs.verify(swapped_points, &proof, &mut opening_transcript.clone())
            .is_err(),
        "swapped opening points with unchanged values must be rejected"
    );

    let mut wrong_value = claims.clone();
    wrong_value[0].matrices[0].points[0].values[0] += Challenge::ONE;
    assert!(
        pcs.verify(wrong_value, &proof, &mut opening_transcript.clone())
            .is_err(),
        "changed first claimed value must be rejected"
    );

    let mut wrong_last_value = claims.clone();
    let last_round = wrong_last_value.last_mut().unwrap();
    let last_matrix = last_round.matrices.last_mut().unwrap();
    let last_point = last_matrix.points.last_mut().unwrap();
    *last_point.values.last_mut().unwrap() += Challenge::ONE;
    assert!(
        pcs.verify(wrong_last_value, &proof, &mut opening_transcript.clone())
            .is_err(),
        "changed last claimed value must be rejected"
    );

    let mut missing_column = claims.clone();
    missing_column[0].matrices[0].points[0].values.pop();
    assert!(
        pcs.verify(missing_column, &proof, &mut opening_transcript.clone())
            .is_err(),
        "missing claimed column must be rejected"
    );

    let mut missing_matrix = claims;
    missing_matrix[multi_matrix_round].matrices.pop();
    assert!(
        pcs.verify(missing_matrix, &proof, &mut opening_transcript)
            .is_err(),
        "missing claimed matrix must be rejected"
    );
}
