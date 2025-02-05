use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt::Debug;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs, PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::bitrev::{BitReversableMatrix, BitReversalPerm};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_matrix::horizontally_truncated::HorizontallyTruncated;
use p3_matrix::row_index_mapped::RowIndexMappedView;
use p3_matrix::Matrix;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use tracing::instrument;

use crate::verifier::FriError;
use crate::{BatchOpening, FriConfig, FriProof, TwoAdicFriPcs};

/// A hiding FRI PCS. Both MMCSs must also be hiding; this is not enforced at compile time so it's
/// the user's responsibility to configure.
#[derive(Debug)]
pub struct HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R> {
    inner: TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>,
    num_random_codewords: usize,
    rng: RefCell<R>,
}

impl<Val, Dft, InputMmcs, FriMmcs, R> HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R> {
    pub fn new(
        dft: Dft,
        mmcs: InputMmcs,
        fri: FriConfig<FriMmcs>,
        num_random_codewords: usize,
        rng: R,
    ) -> Self {
        let inner = TwoAdicFriPcs::new(dft, mmcs, fri);
        Self {
            inner,
            num_random_codewords,
            rng: rng.into(),
        }
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger, R> Pcs<Challenge, Challenger>
    for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: TwoAdicField,
    Standard: Distribution<Val>,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
    R: Rng + Send + Sync,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type EvaluationsOnDomain<'a> = HorizontallyTruncated<
        Val,
        RowIndexMappedView<BitReversalPerm, DenseMatrix<Val, &'a [Val]>>,
    >;
    /// The first item contains the openings of the random polynomials added by this wrapper.
    /// The second item is the usual FRI proof.
    type Proof = (
        OpenedValues<Challenge>,
        FriProof<Challenge, FriMmcs, Val, Vec<BatchOpening<Val, InputMmcs>>>,
    );
    type Error = FriError<FriMmcs::Error, InputMmcs::Error>;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &self.inner, degree)
    }

    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)>,
        is_random_poly: bool,
    ) -> (Self::Commitment, Self::ProverData) {
        let randomized_evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)> = evaluations
            .into_iter()
            .map(|(domain, mat)| {
                (
                    domain,
                    add_random_cols(mat, self.num_random_codewords, &mut *self.rng.borrow_mut()),
                )
            })
            .collect();
        let h = randomized_evaluations[0].1.height();
        let w = randomized_evaluations[0].1.width();
        let ldes: Vec<_> = randomized_evaluations
            .into_iter()
            .map(|(domain, evals)| {
                let shift = Val::GENERATOR / domain.shift;
                // We do not need to randomize the randomizing poly.
                if is_random_poly {
                    assert_eq!(domain.size(), evals.height());

                    self.inner
                        .dft
                        .coset_lde_batch(evals, self.inner.fri.log_blowup, shift, None)
                        .bit_reverse_rows()
                        .to_row_major_matrix()
                } else {
                    assert_eq!(domain.size(), evals.height() * 2);

                    let random_values = vec![self.rng.borrow_mut().gen(); h * w];

                    self.inner
                        .dft
                        .coset_lde_batch(
                            evals,
                            self.inner.fri.log_blowup,
                            shift,
                            Some(&random_values),
                        )
                        .bit_reverse_rows()
                        .to_row_major_matrix()
                }
            })
            .collect();
        self.inner.mmcs.commit(ldes)
    }

    fn commit_quotient(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)>,
        cis: Vec<Val>,
    ) -> (Self::Commitment, Self::ProverData) {
        let last_chunk = evaluations.len() - 1;
        let randomized_evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)> = evaluations
            .into_iter()
            .map(|(domain, mat)| {
                (
                    domain,
                    add_random_cols(mat, self.num_random_codewords, &mut *self.rng.borrow_mut()),
                )
            })
            .collect();
        // First, add random values as described in https://eprint.iacr.org/2024/1037.pdf.
        // If we have `d` chunks, let q'_i(X) = q_i(X) + v_H_i(X) * t_i(X) where t(X) is random, for 1 <= i < d.
        // q'_d(X) = q_d(X) - v_H_d(X) c_i \sum t_i(X) where c_i is a Lagrange normalization constant.
        let h = randomized_evaluations[0].1.height();
        let w = randomized_evaluations[0].1.width();

        let all_random_values =
            vec![self.rng.borrow_mut().gen(); (randomized_evaluations.len() - 1) * h * w];
        let ldes: Vec<_> = randomized_evaluations
            .into_iter()
            .enumerate()
            .map(|(i, (domain, evals))| {
                assert_eq!(domain.size(), evals.height());
                let shift = Val::GENERATOR / domain.shift;

                // Select random values, and set the random values for the final chunk accordingly.
                let random_values = if i == last_chunk {
                    let mut added_values = Val::zero_vec(h * w);
                    for j in 0..last_chunk {
                        for k in 0..h * w {
                            added_values[k] -= all_random_values[j * h * w + k]
                                * cis[j]
                                * cis[last_chunk].inverse();
                        }
                    }
                    added_values
                } else {
                    all_random_values[i * h * w..(i + 1) * h * w].to_vec()
                };

                // Check the evaluation as the verifier would here, but on challenge = 1, to see whether it works? (and compare it with non random value)
                // Commit to the bit-reversed LDE.
                self.inner
                    .dft
                    .coset_lde_batch(
                        evals,
                        self.inner.fri.log_blowup,
                        shift,
                        Some(&random_values),
                    )
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect();

        self.inner.mmcs.commit(ldes)
    }

    fn is_zk(&self) -> bool {
        true
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let inner_evals = <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<
            Challenge,
            Challenger,
        >>::get_evaluations_on_domain(
            &self.inner, prover_data, idx, domain
        );
        let inner_width = inner_evals.width();
        // Truncate off the columns representing random codewords we added in `commit` above.
        HorizontallyTruncated::new(inner_evals, inner_width - self.num_random_codewords)
    }

    fn generate_random_vals(&self, random_len: usize) -> RowMajorMatrix<Val> {
        let random_vals = vec![self.rng.borrow_mut().gen(); random_len * Challenge::D];
        assert!(
            random_len.is_power_of_two(),
            "Provided random size for the random bacth FRI polynomial is not a power of 2: {}",
            random_len
        );
        RowMajorMatrix::new(random_vals, Challenge::D)
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
        let (mut inner_opened_values, inner_proof) = self.inner.open(rounds, challenger);

        // inner_opened_values includes opened values for the random codewords. Those should be
        // hidden from our caller, so we split them off and store them in the proof.
        let opened_values_rand = inner_opened_values
            .iter_mut()
            .map(|opened_values_for_round| {
                opened_values_for_round
                    .iter_mut()
                    .map(|opened_values_for_mat| {
                        opened_values_for_mat
                            .iter_mut()
                            .map(|opened_values_for_point| {
                                let split =
                                    opened_values_for_point.len() - self.num_random_codewords;
                                opened_values_for_point.drain(split..).collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        (inner_opened_values, (opened_values_rand, inner_proof))
    }

    fn verify(
        &self,
        // For each round:
        mut rounds: Vec<(
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
        let (opened_values_for_rand_cws, inner_proof) = proof;
        // Now we merge `opened_values_for_rand_cws` into the opened values in `rounds`, undoing
        // the split that we did in `open`, to get a complete set of opened values for the inner PCS
        // to check.
        for (round, rand_round) in rounds.iter_mut().zip(opened_values_for_rand_cws) {
            for (mat, rand_mat) in round.1.iter_mut().zip(rand_round) {
                for (point, rand_point) in mat.1.iter_mut().zip(rand_mat) {
                    point.1.extend(rand_point);
                }
            }
        }
        self.inner.verify(rounds, inner_proof, challenger)
    }
}

#[instrument(level = "debug", skip_all)]
fn add_random_cols<Val, R>(
    mat: RowMajorMatrix<Val>,
    num_random_codewords: usize,
    mut rng: R,
) -> RowMajorMatrix<Val>
where
    Val: Field,
    R: Rng + Send + Sync,
    Standard: Distribution<Val>,
{
    let old_w = mat.width();
    let new_w = old_w + num_random_codewords;
    let h = mat.height();

    let new_values = Val::zero_vec(new_w * h);
    let mut result = RowMajorMatrix::new(new_values, new_w);
    // Can be parallelized by adding par_, but there are some complications with the RNG.
    // We could just use thread_rng(), but ideally we want to keep it generic...
    result
        .rows_mut()
        .zip(mat.row_slices())
        .for_each(|(new_row, old_row)| {
            new_row[..old_w].copy_from_slice(old_row);
            new_row[old_w..].iter_mut().for_each(|v| *v = rng.gen());
        });
    result
}
