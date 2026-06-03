use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{
    BatchOpening, BuildPeriodicLdeTableFast, Mmcs, OpenedValues, Pcs, PolynomialSpace,
};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, TwoAdicField, batch_multiplicative_inverse};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversalPerm, BitReversibleMatrix};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixCow};
use p3_matrix::horizontally_truncated::HorizontallyTruncated;
use p3_matrix::row_index_mapped::RowIndexMappedView;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};
use spin::Mutex;
use tracing::info_span;

use crate::verifier::FriError;
use crate::{FriParameters, FriProof, TwoAdicFriPcs};

/// A hiding FRI PCS. Both MMCSs must also be hiding; this is not enforced at compile time so it's
/// the user's responsibility to configure.
#[derive(Debug)]
pub struct HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R> {
    inner: TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>,
    num_random_codewords: usize,
    rng: Mutex<R>,
}

impl<Val, Dft, InputMmcs, FriMmcs, R> Clone for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: Clone,
    Dft: Clone,
    InputMmcs: Clone,
    FriMmcs: Clone,
    R: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            num_random_codewords: self.num_random_codewords,
            rng: Mutex::new(self.rng.lock().clone()),
        }
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, R> HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R> {
    pub const fn new(
        dft: Dft,
        mmcs: InputMmcs,
        params: FriParameters<FriMmcs>,
        num_random_codewords: usize,
        rng: R,
    ) -> Self {
        let inner = TwoAdicFriPcs::new(dft, mmcs, params);
        Self {
            inner,
            num_random_codewords,
            rng: Mutex::new(rng),
        }
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger, R> Pcs<Challenge, Challenger>
    for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: TwoAdicField,
    StandardUniform: Distribution<Val>,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val, Proof: Sync, Error: Sync>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
    R: Rng + Send + Sync,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type EvaluationsOnDomain<'a> =
        HorizontallyTruncated<Val, RowIndexMappedView<BitReversalPerm, RowMajorMatrixCow<'a, Val>>>;
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

    fn log_max_lde_height(&self) -> usize {
        <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<Challenge, Challenger>>::log_max_lde_height(
            &self.inner)
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

                        let mut random_evaluation = mat.with_random_cols(
                            mat_width + 2 * self.num_random_codewords,
                            &mut *self.rng.lock(),
                        );
                        random_evaluation.width = mat_width + self.num_random_codewords;

                        (domain, random_evaluation)
                    })
                    .collect()
            });

        Pcs::<Challenge, Challenger>::commit(&self.inner, randomized_evaluations)
    }

    fn commit_preprocessing(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        // Pad values with zero columns instead of random columns.
        let padded_evals = evaluations
            .into_iter()
            .map(|(domain, mat)| {
                let mat_width = mat.width();
                // Let `w` and `h` be the width and height of the original matrix. The padded matrix should have height `2h` and width `w`.
                // To generate it, we add `w` zero columns to the original matrix, then reshape it by setting the width to `w`.
                // All columns are added on the right hand side so, after reshaping, this has the net effect of adding interleaving the original trace with zero rows.
                let mut padded_evaluation = mat.with_zero_cols(mat_width);
                padded_evaluation.width = mat_width;
                (domain, padded_evaluation)
            })
            .collect::<Vec<_>>();

        Pcs::<Challenge, Challenger>::commit(&self.inner, padded_evals)
    }

    /// Get the quotient polynomial LDEs. We first decompose the quotient polynomial into
    /// `num_chunks` many smaller polynomials each of degree `degree / num_chunks`.
    /// These quotient polynomials are then randomized as explained in Section 4.2 of
    /// <https://eprint.iacr.org/2024/1037.pdf>.
    ///
    /// ### Arguments
    /// - `quotient_domain` the domain of the quotient polynomial.
    /// - `quotient_evaluations` the evaluations of the quotient polynomial over the domain. This should be in
    ///   standard (not bit-reversed) order.
    /// - `num_chunks` the number of smaller polynomials to decompose the quotient polynomial into.
    ///
    /// # Panics
    /// This function panics if `num_chunks` is either `0` or `1`. The first case makes no logical
    /// sense and in the second case, the resulting commitment would not be hiding.
    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
        num_chunks: usize,
    ) -> Vec<RowMajorMatrix<Val>> {
        assert!(
            num_chunks > 1,
            "num_chunks must be > 1 to preserve hiding (got {num_chunks})"
        );
        let (domains, evaluations): (Vec<_>, Vec<_>) = evaluations.into_iter().unzip();
        let cis = get_zp_cis(&domains);
        let last_chunk = num_chunks - 1;
        let last_chunk_ci_inv = cis[last_chunk].inverse();
        let mul_coeffs = (0..last_chunk)
            .map(|i| cis[i] * last_chunk_ci_inv)
            .collect_vec();

        let mut rng = self.rng.lock();
        let randomized_evaluations: Vec<RowMajorMatrix<Val>> = evaluations
            .into_iter()
            .map(|mat| mat.with_random_cols(self.num_random_codewords, &mut *rng))
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

        domains
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
            .collect()
    }

    fn commit_ldes(&self, ldes: Vec<RowMajorMatrix<Val>>) -> (Self::Commitment, Self::ProverData) {
        Pcs::<Challenge, Challenger>::commit_ldes(&self.inner, ldes)
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

    fn get_evaluations_on_domain_no_random<'a>(
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

        HorizontallyTruncated::new(inner_evals, inner_width).unwrap()
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
        self.open_with_preprocessing(rounds, challenger, false)
    }

    fn open_with_preprocessing(
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
        is_preprocessing: bool,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        let (mut inner_opened_values, inner_proof) =
            self.inner
                .open_with_preprocessing(rounds, challenger, is_preprocessing);
        // inner_opened_values includes opened values for the random codewords. Those should be
        // hidden from our caller, so we split them off and store them in the proof.
        let opened_values_rand = inner_opened_values
            .iter_mut()
            .enumerate()
            .map(|(idx, opened_values_for_round)| {
                opened_values_for_round
                    .iter_mut()
                    .map(|opened_values_for_mat| {
                        opened_values_for_mat
                            .iter_mut()
                            .map(|opened_values_for_point| {
                                let num_random_codewords =
                                    if is_preprocessing && idx == <Self as Pcs<Challenge, Challenger>>::PREPROCESSED_TRACE_IDX {
                                        0
                                    } else {
                                        self.num_random_codewords
                                    };
                                let split = opened_values_for_point.len() - num_random_codewords;
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

        // Proving split each opening into a public half and a hidden half.
        // - The public half lives here.
        // - The hidden half travels beside the proof as random codewords.
        //
        // Re-joining them gives the inner verifier the full openings it committed to.
        //
        //     public (per point):  [v_0, .., v_k]
        //     hidden (per point):                [r_0, .., r_m]
        //     merged:              [v_0, .., v_k,  r_0, .., r_m]
        //
        // Invariant: the halves nest identically by round, then matrix, then point.
        // Each level's length is checked before merging.
        // A mismatch returns a precise error instead of being truncated silently.

        // Level 1: one set of random openings per round.
        if opened_values_for_rand_cws.len() != rounds.len() {
            return Err(FriError::HidingRandomOpeningRoundCountMismatch {
                expected: rounds.len(),
                got: opened_values_for_rand_cws.len(),
            });
        }
        for (round_idx, (round, rand_round)) in rounds
            .iter_mut()
            .zip(opened_values_for_rand_cws.iter())
            .enumerate()
        {
            // Level 2: one set per matrix in this round.
            if rand_round.len() != round.1.len() {
                return Err(FriError::HidingRandomOpeningMatrixCountMismatch {
                    round: round_idx,
                    expected: round.1.len(),
                    got: rand_round.len(),
                });
            }
            for (matrix_idx, (mat, rand_mat)) in
                round.1.iter_mut().zip(rand_round.iter()).enumerate()
            {
                // Level 3: one set per opening point of this matrix.
                if rand_mat.len() != mat.1.len() {
                    return Err(FriError::HidingRandomOpeningPointCountMismatch {
                        round: round_idx,
                        matrix: matrix_idx,
                        expected: mat.1.len(),
                        got: rand_mat.len(),
                    });
                }
                // Shapes agree: append the hidden values onto the public ones.
                for (point, rand_point) in mat.1.iter_mut().zip(rand_mat.iter()) {
                    point.1.extend(rand_point);
                }
            }
        }
        self.inner.verify(rounds, inner_proof, challenger)
    }

    fn get_opt_randomization_poly_commitment(
        &self,
        ext_trace_domains: impl IntoIterator<Item = Self::Domain>,
    ) -> Option<(Self::Commitment, Self::ProverData)> {
        let random_input_vals = ext_trace_domains
            .into_iter()
            .map(|domain| {
                let m = DenseMatrix::rand(
                    &mut *self.rng.lock(),
                    domain.size(),
                    self.num_random_codewords + Challenge::DIMENSION,
                );

                (domain, m)
            })
            .collect::<Vec<_>>();

        let r_commit_and_data =
            Pcs::<Challenge, Challenger>::commit(&self.inner, random_input_vals);
        Some(r_commit_and_data)
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, R> BuildPeriodicLdeTableFast
    for HidingFriPcs<Val, Dft, InputMmcs, FriMmcs, R>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val>,
{
    type PeriodicDomain = TwoAdicMultiplicativeCoset<Val>;

    fn maybe_build_periodic_lde_table_fast(
        &self,
        periodic_cols: &[Vec<p3_commit::Val<Self::PeriodicDomain>>],
        trace_domain: Self::PeriodicDomain,
        quotient_domain: Self::PeriodicDomain,
    ) -> Option<p3_commit::PeriodicLdeTable<p3_commit::Val<Self::PeriodicDomain>>>
    where
        p3_commit::Val<Self::PeriodicDomain>: Clone,
    {
        self.inner
            .maybe_build_periodic_lde_table_fast(periodic_cols, trace_domain, quotient_domain)
    }
}

/// Compute the normalizing constants for the Langrange selectors of the provided domains.
/// See Section 4.2 of <https://eprint.iacr.org/2024/1037.pdf> for more details.
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

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2Dit;
    use p3_field::Field;
    use p3_field::extension::BinomialExtensionField;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Dft = Radix2Dit<Val>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type MyPcs = HidingFriPcs<Val, Dft, ValMmcs, ChallengeMmcs, SmallRng>;

    type Commitment = <ValMmcs as Mmcs<Val>>::Commitment;
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    /// Public opening claims (the `rounds` argument): per matrix, its domain and
    /// the `(point, values)` pairs.
    type Claims = Vec<(Domain, Vec<(Challenge, Vec<Challenge>)>)>;
    type Proof = <MyPcs as Pcs<Challenge, Challenger>>::Proof;
    type TestError =
        FriError<<ChallengeMmcs as Mmcs<Challenge>>::Error, <ValMmcs as Mmcs<Val>>::Error>;

    /// Random codewords appended per matrix.
    ///
    /// Must be `> 0` so each opening splits into a public part (`rounds`) and a
    /// hidden part (`proof.0`) — the split `verify` re-merges and whose shape the
    /// new error variants guard.
    const NUM_RANDOM_CODEWORDS: usize = 2;

    /// Run a real prover roundtrip and return `(pcs, claims, proof, challenger)`
    /// ready to verify, with `challenger` advanced past the commitment.
    ///
    /// One round, one matrix, one point, so the random-opening tree nests as:
    ///
    /// ```text
    ///     proof.0          = [round_0]      // 1 round
    ///     proof.0[0]       = [matrix_0]     // 1 matrix
    ///     proof.0[0][0]    = [point_0]      // 1 point
    ///     proof.0[0][0][0] = [v_0, v_1]     // NUM_RANDOM_CODEWORDS values
    /// ```
    ///
    /// Each test perturbs one level to trip the matching count check.
    fn make_fixture() -> (MyPcs, Vec<(Commitment, Claims)>, Proof, Challenger) {
        // Fixed seeds keep the roundtrip deterministic.
        let mut rng = SmallRng::seed_from_u64(1);

        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());

        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        // Minimal sound parameters: blowup 2, binary folding, 2 queries.
        let fri_params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: challenge_mmcs,
        };

        // The wrapper owns an independently seeded RNG for its random codewords.
        let pcs = MyPcs::new(
            Dft::default(),
            val_mmcs,
            fri_params,
            NUM_RANDOM_CODEWORDS,
            SmallRng::seed_from_u64(2),
        );

        // The wrapper interleaves the trace with random rows, doubling its
        // height, so (like the zk prover) we commit against a `2 * height` domain.
        let log_degree = 3;
        let width = 4;
        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 2 << log_degree);
        let trace = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_degree, width);
        let (commitment, prover_data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain, trace)]);

        // Prover: observe, sample the point, prove.
        let mut p_challenger = Challenger::new(perm.clone());
        p_challenger.observe(&commitment);
        let zeta: Challenge = p_challenger.sample_algebra_element();
        let (opened_values, proof) =
            pcs.open(vec![(&prover_data, vec![vec![zeta]])], &mut p_challenger);

        // Verifier: replay up to the point sample so a valid proof must pass.
        let mut v_challenger = Challenger::new(perm);
        v_challenger.observe(&commitment);
        let v_zeta: Challenge = v_challenger.sample_algebra_element();
        assert_eq!(
            v_zeta, zeta,
            "prover and verifier must sample the same point"
        );

        // Public claims; the hidden values stay in `proof.0` until `verify`.
        let claims = vec![(
            commitment,
            vec![(domain, vec![(zeta, opened_values[0][0][0].clone())])],
        )];

        (pcs, claims, proof, v_challenger)
    }

    /// Verify with fully qualified syntax so the type parameters are unambiguous.
    fn run_verify(
        pcs: &MyPcs,
        claims: Vec<(Commitment, Claims)>,
        proof: &Proof,
        challenger: &mut Challenger,
    ) -> Result<(), TestError> {
        <MyPcs as Pcs<Challenge, Challenger>>::verify(pcs, claims, proof, challenger)
    }

    #[test]
    fn valid_proof_passes() {
        // Baseline: an unmodified proof verifies, so the mismatch tests below
        // start from a genuinely valid proof.
        let (pcs, claims, proof, mut challenger) = make_fixture();
        run_verify(&pcs, claims, &proof, &mut challenger)
            .expect("valid hiding proof should verify");
    }

    #[test]
    fn random_opening_round_count_mismatch() {
        let (pcs, claims, mut proof, mut challenger) = make_fixture();

        // One random-opening entry is required per public round.
        //     claims:  [round_0]          -> expected 1
        //     proof.0: [round_0, EXTRA]   -> got 2
        let expected_rounds = claims.len();
        proof.0.push(vec![]);

        let err = run_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("should reject an extra random-opening round");

        match err {
            FriError::HidingRandomOpeningRoundCountMismatch { expected, got } => {
                assert_eq!(expected, expected_rounds);
                assert_eq!(got, expected_rounds + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn random_opening_matrix_count_mismatch() {
        let (pcs, claims, mut proof, mut challenger) = make_fixture();

        // Round counts match, so the per-round matrix check fires next.
        //     claims[0]:  [matrix_0]          -> expected 1
        //     proof.0[0]: [matrix_0, EXTRA]   -> got 2
        let expected_mats = claims[0].1.len();
        proof.0[0].push(vec![]);

        let err = run_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("should reject an extra random-opening matrix");

        match err {
            FriError::HidingRandomOpeningMatrixCountMismatch {
                round,
                expected,
                got,
            } => {
                assert_eq!(round, 0);
                assert_eq!(expected, expected_mats);
                assert_eq!(got, expected_mats + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn random_opening_point_count_mismatch() {
        let (pcs, claims, mut proof, mut challenger) = make_fixture();

        // Round and matrix counts match, so the per-matrix point check fires.
        //     claims[0][0]:  [point_0]          -> expected 1
        //     proof.0[0][0]: [point_0, EXTRA]   -> got 2
        let expected_points = claims[0].1[0].1.len();
        proof.0[0][0].push(vec![]);

        let err = run_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("should reject an extra random-opening point");

        match err {
            FriError::HidingRandomOpeningPointCountMismatch {
                round,
                matrix,
                expected,
                got,
            } => {
                assert_eq!(round, 0);
                assert_eq!(matrix, 0);
                assert_eq!(expected, expected_points);
                assert_eq!(got, expected_points + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }
}
