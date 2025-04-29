use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt::Debug;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpening, Mmcs, OpenedValues, Pcs, PolynomialSpace};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, Field, TwoAdicField, batch_multiplicative_inverse};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversalPerm, BitReversibleMatrix};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::horizontally_truncated::HorizontallyTruncated;
use p3_matrix::row_index_mapped::RowIndexMappedView;
use p3_util::zip_eq::zip_eq;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use tracing::{info_span, instrument};

use crate::verifier::FriError;
use crate::{FriConfig, FriProof, TwoAdicFriPcs};

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
    StandardUniform: Distribution<Val>,
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
        RowIndexMappedView<BitReversalPerm, RowMajorMatrixView<'a, Val>>,
    >;
    /// The first item contains the openings of the random polynomials added by this wrapper.
    /// The second item is the usual FRI proof.
    type Proof = (
        OpenedValues<Challenge>,
        FriProof<Challenge, FriMmcs, Val, Vec<BatchOpening<Val, InputMmcs>>>,
    );
    type Error = FriError<FriMmcs::Error, InputMmcs::Error>;

    const ZK: bool = true;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &self.inner, degree)
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let randomized_evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)> =
            info_span!("randomize polys").in_scope(|| {
                evaluations
                    .into_iter()
                    .map(|(domain, mat)| {
                        let mat_width = mat.width();
                        // Let `w` and `h` be the width and height of the original matrix. The randomized matrix should have height `2h` and width `w + num_random_codewords`.
                        // To generate it, we add `w + 2 * num_random_codewords` columns to the original matrix, then reshape it by setting the width to `w + num_random_codewords`.
                        // All columns are added on the right hand side so, after reshaping, this has the net effect of adding `num_random_codewords` random columns on the right and interleaving the original trace with random rows.

                        let mut random_evaluation = add_random_cols(
                            mat,
                            mat_width + 2 * self.num_random_codewords,
                            &mut *self.rng.borrow_mut(),
                        );
                        random_evaluation.width = mat_width + self.num_random_codewords;

                        (domain, random_evaluation)
                    })
                    .collect()
            });

        Pcs::<Challenge, Challenger>::commit(&self.inner, randomized_evaluations)
    }

    fn commit_quotient(
        &self,
        domains: Vec<Self::Domain>,
        evaluations: Vec<RowMajorMatrix<Val>>,
    ) -> (Self::Commitment, Self::ProverData) {
        // Compute the vanishing polynomial normalizing constants.
        assert_eq!(domains.len(), evaluations.len());
        if evaluations.is_empty() {
            return self.inner.mmcs.commit(vec![]);
        }
        let cis = get_zp_cis(&domains);
        let last_chunk = evaluations.len() - 1;
        let last_chunk_ci_inv = cis[last_chunk].inverse();
        let mul_coeffs = (0..last_chunk)
            .map(|i| cis[i] * last_chunk_ci_inv)
            .collect_vec();

        let mut rng = self.rng.borrow_mut();
        let randomized_evaluations: Vec<RowMajorMatrix<Val>> = evaluations
            .into_iter()
            .map(|mat| add_random_cols(mat, self.num_random_codewords, &mut *rng))
            .collect();
        // Add random values to the LDE evaluations as described in https://eprint.iacr.org/2024/1037.pdf.
        // If we have `d` chunks, let q'_i(X) = q_i(X) + v_H_i(X) * t_i(X) where t(X) is random, for 1 <= i < d.
        // q'_d(X) = q_d(X) - v_H_d(X) c_i \sum t_i(X) where c_i is a Lagrange normalization constant.
        let h = randomized_evaluations[0].height();
        let w = randomized_evaluations[0].width();
        let mut all_random_values = (0..(randomized_evaluations.len() - 1) * h * w)
            .map(|_| rng.random())
            .chain(core::iter::repeat_n(Val::ZERO, h * w))
            .collect::<Vec<_>>();

        // Set the random values for the final chunk accordingly
        for j in 0..last_chunk {
            let mul_coeff = mul_coeffs[j];
            for k in 0..h * w {
                let t = all_random_values[j * h * w + k] * mul_coeff;
                all_random_values[last_chunk * h * w + k] -= t;
            }
        }

        let ldes: Vec<_> = domains
            .into_iter()
            .zip(randomized_evaluations)
            .enumerate()
            .map(|(i, (domain, evals))| {
                assert_eq!(domain.size(), evals.height());
                let shift = Val::GENERATOR / domain.shift();
                let random_values = &all_random_values[i * h * w..(i + 1) * h * w];

                // Commit to the bit-reversed LDE.
                let mut lde_evals = self
                    .inner
                    .dft
                    .coset_lde_batch(evals, self.inner.fri.log_blowup + 1, shift)
                    .to_row_major_matrix();

                // Evaluate `v_H(X) * r(X)` over the LDE, where:
                // - `v_H` is the coset vanishing polynomial, here equal to (GENERATOR * X / domain.shift)^n - 1,
                // - and `r` is a random polynomial.
                let mut vanishing_poly_coeffs =
                    Val::zero_vec((h * w) << (self.inner.fri.log_blowup + 1));
                let p = shift.exp_u64(h as u64);
                Val::GENERATOR
                    .powers()
                    .take(h)
                    .enumerate()
                    .for_each(|(i, p_i)| {
                        for j in 0..w {
                            let mul_coeff = p_i * random_values[i * w + j];
                            vanishing_poly_coeffs[i * w + j] -= mul_coeff;
                            vanishing_poly_coeffs[(h + i) * w + j] = p * mul_coeff;
                        }
                    });
                let random_eval = self
                    .inner
                    .dft
                    .dft_batch(DenseMatrix::new(vanishing_poly_coeffs, w))
                    .to_row_major_matrix();

                // Add the quotient chunk evaluations over the LDE to the evaluations of `v_H(X) * r(X)`.
                for i in 0..h * w * (1 << (self.inner.fri.log_blowup + 1)) {
                    lde_evals.values[i] += random_eval.values[i];
                }

                lde_evals.bit_reverse_rows().to_row_major_matrix()
            })
            .collect();

        self.inner.mmcs.commit(ldes)
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
        // The unwrap is safe as inner_width - self.num_random_codewords <= inner_width.
        HorizontallyTruncated::new(inner_evals, inner_width - self.num_random_codewords).unwrap()
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
        for (round, rand_round) in zip_eq(
            rounds.iter_mut(),
            opened_values_for_rand_cws,
            FriError::InvalidProofShape,
        )? {
            for (mat, rand_mat) in
                zip_eq(round.1.iter_mut(), rand_round, FriError::InvalidProofShape)?
            {
                for (point, rand_point) in
                    zip_eq(mat.1.iter_mut(), rand_mat, FriError::InvalidProofShape)?
                {
                    point.1.extend(rand_point);
                }
            }
        }
        self.inner.verify(rounds, inner_proof, challenger)
    }

    fn get_opt_randomization_poly_commitment(
        &self,
        ext_trace_domain: Self::Domain,
    ) -> Option<(Self::Commitment, Self::ProverData)> {
        let random_vals = DenseMatrix::rand(
            &mut *self.rng.borrow_mut(),
            ext_trace_domain.size(),
            self.num_random_codewords + Challenge::DIMENSION,
        );
        let extended_domain = <Self as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            self,
            ext_trace_domain.size(),
        );
        let r_commit_and_data =
            Pcs::<Challenge, Challenger>::commit(&self.inner, [(extended_domain, random_vals)]);
        Some(r_commit_and_data)
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
    StandardUniform: Distribution<Val>,
{
    let old_w = mat.width();
    let new_w = old_w + num_random_codewords;
    let h = mat.height();

    let new_values = Val::zero_vec(new_w * h);
    let mut result = RowMajorMatrix::new(new_values, new_w);
    // Can be parallelized by adding par_, but there are some complications with the RNG.
    // We could just use rng(), but ideally we want to keep it generic...
    result
        .rows_mut()
        .zip(mat.row_slices())
        .for_each(|(new_row, old_row)| {
            new_row[..old_w].copy_from_slice(old_row);
            new_row[old_w..].iter_mut().for_each(|v| *v = rng.random());
        });
    result
}

/// Compute the normalizing constants for the Langrange selectors of the provided domains.
/// See Section 4.2 of https://eprint.iacr.org/2024/1037.pdf for more details.
fn get_zp_cis<D: PolynomialSpace>(qc_domains: &[D]) -> Vec<p3_commit::Val<D>> {
    batch_multiplicative_inverse(
        &qc_domains
            .iter()
            .enumerate()
            .map(|(i, domain)| {
                qc_domains
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, other_domain)| {
                        other_domain.vanishing_poly_at_point(domain.first_point())
                    })
                    .product()
            })
            .collect::<Vec<_>>(),
    )
}
