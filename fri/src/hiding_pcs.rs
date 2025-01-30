use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt::Debug;
use p3_commit::PolynomialSpace;
use p3_matrix::bitrev::BitReversableMatrix;

use crate::verifier::FriError;
use crate::{BatchOpening, FriConfig, FriProof, TwoAdicFriPcs};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs, TwoAdicMultiplicativeCoset};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::horizontally_truncated::HorizontallyTruncated;
use p3_matrix::Matrix;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use tracing::instrument;

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
        let ldes: Vec<_> = randomized_evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height() * 2);
                let shift = Val::GENERATOR / domain.shift;
                let s = domain.shift;

                let random_values = vec![self.rng.borrow_mut().gen(); h];
                // let random_values = vec![Val::ZERO; h];

                self.inner
                    .dft
                    .coset_lde_batch_zk(evals, self.inner.fri.log_blowup, shift, s, &random_values)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect();
        self.inner.mmcs.commit(ldes)
        // <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<Challenge, Challenger>>::commit(
        //     &self.inner,
        //     randomized_evaluations,
        // )
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
        // let last_chunk = randomized_evaluations.len() - 1;
        // First, add random values as described in https://eprint.iacr.org/2024/1037.pdf.
        // If we have `d` chunks, let q'_i(X) = q_i(X) + v_H_i(X) * t_i(X) where t(X) is random, for 1 <= i < d.
        // q'_d(X) = q_d(X) - v_H_d(X) c_i \sum t_i(X) where c_i is a Lagrange normalization constant.
        let h = randomized_evaluations[0].1.height();

        let all_random_values =
            vec![self.rng.borrow_mut().gen(); (randomized_evaluations.len() - 1) * h];
        let ldes: Vec<_> = randomized_evaluations
            .into_iter()
            .enumerate()
            .map(|(i, (domain, evals))| {
                assert_eq!(domain.size(), evals.height());
                let shift = Val::GENERATOR / domain.shift;
                let s = domain.shift;

                // Select random values, and set the random values for the final chunk accordingly.
                let random_values = if i == last_chunk {
                    let mut added_values = Val::zero_vec(h);
                    for j in 0..last_chunk {
                        for k in 0..h {
                            added_values[k] -=
                                all_random_values[j * h + k] * cis[j] * cis[last_chunk].inverse();
                        }
                    }
                    added_values
                } else {
                    // all_random_values[i * h * w..(i + 1) * h * w].to_vec()
                    all_random_values[i * h..(i + 1) * h]
                        .iter()
                        .map(|v| *v)
                        .collect()
                };

                // CHeck the evaluation as the verifier would here, but on challenge = 1, to see whether it works? (and compare it with non random value)
                // Commit to the bit-reversed LDE.
                self.inner
                    .dft
                    .coset_lde_batch_zk(evals, self.inner.fri.log_blowup, shift, s, &random_values)
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
    ) -> impl Matrix<Val> + 'a {
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
        // let mut random_coeffs = vec![self.rng.borrow_mut().gen(); random_len];
        let mut random_coeffs = Val::zero_vec(random_len * (1 << self.inner.fri.log_blowup) - 1);
        assert!(
            random_len.is_power_of_two(),
            "random size incorrect {}",
            random_len
        );
        random_coeffs.push(Val::ZERO);
        let random_vals = self.inner.dft.coset_dft(random_coeffs, Val::ONE);
        RowMajorMatrix::new(random_vals, 1)
    }

    // fn open(
    //     &self,
    //     // For each round,
    //     rounds: Vec<(
    //         &Self::ProverData,
    //         // for each matrix,
    //         Vec<
    //             // points to open
    //             Vec<Challenge>,
    //         >,
    //     )>,
    //     challenger: &mut Challenger,
    // ) -> (OpenedValues<Challenge>, Self::Proof) {
    //     /*

    //     A quick rundown of the optimizations in this function:
    //     We are trying to compute sum_i alpha^i * (p(X) - y)/(X - z),
    //     for each z an opening point, y = p(z). Each p(X) is given as evaluations in bit-reversed order
    //     in the columns of the matrices. y is computed by barycentric interpolation.
    //     X and p(X) are in the base field; alpha, y and z are in the extension.
    //     The primary goal is to minimize extension multiplications.

    //     - Instead of computing all alpha^i, we just compute alpha^i for i up to the largest width
    //     of a matrix, then multiply by an "alpha offset" when accumulating.
    //           a^0 x0 + a^1 x1 + a^2 x2 + a^3 x3 + ...
    //         = a^0 ( a^0 x0 + a^1 x1 ) + a^2 ( a^0 x2 + a^1 x3 ) + ...
    //         (see `alpha_pows`, `alpha_pow_offset`, `num_reduced`)

    //     - For each unique point z, we precompute 1/(X-z) for the largest subgroup opened at this point.
    //     Since we compute it in bit-reversed order, smaller subgroups can simply truncate the vector.
    //         (see `inv_denoms`)

    //     - Then, for each matrix (with columns p_i) and opening point z, we want:
    //         for each row (corresponding to subgroup element X):
    //             reduced[X] += alpha_offset * sum_i [ alpha^i * inv_denom[X] * (p_i[X] - y[i]) ]

    //         We can factor out inv_denom, and expand what's left:
    //             reduced[X] += alpha_offset * inv_denom[X] * sum_i [ alpha^i * p_i[X] - alpha^i * y[i] ]

    //         And separate the sum:
    //             reduced[X] += alpha_offset * inv_denom[X] * [ sum_i [ alpha^i * p_i[X] ] - sum_i [ alpha^i * y[i] ] ]

    //         And now the last sum doesn't depend on X, so we can precompute that for the matrix, too.
    //         So the hot loop (that depends on both X and i) is just:
    //             sum_i [ alpha^i * p_i[X] ]

    //         with alpha^i an extension, p_i[X] a base

    //     */
    //     // Batch combination challenge
    //     let alpha: Challenge = challenger.sample_ext_element();

    //     let mats_and_points: Vec<(Vec<_>, &Vec<Vec<_>>)> = rounds
    //         .iter()
    //         .map(|(data, points)| {
    //             (
    //                 self.inner
    //                     .mmcs
    //                     .get_matrices(data)
    //                     .into_iter()
    //                     .map(|m| m.as_view())
    //                     .collect(),
    //                 points,
    //             )
    //         })
    //         .collect();
    //     let mats: Vec<_> = mats_and_points.iter().flat_map(|(mats, _)| mats).collect();

    //     let global_max_height = mats.iter().map(|m| m.height()).max().unwrap();
    //     let log_global_max_height = log2_strict_usize(global_max_height);

    //     // For each unique opening point z, we will find the largest degree bound
    //     // for that point, and precompute 1/(X - z) for the largest subgroup (in bitrev order).
    //     let inv_denoms = compute_inverse_denominators(&mats_and_points, Val::GENERATOR);

    //     let mut all_opened_values: OpenedValues<Challenge> = vec![];

    //     let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);
    //     let mut num_reduced = [0; 32];

    //     // But the last quotient is a random polynomial.
    //     // quotient index = 1
    //     let mut max_height = 0;
    //     for (mats, points) in &mats_and_points[..mats_and_points.len() - 1] {
    //         let opened_values_for_round = all_opened_values.pushed_mut(vec![]);
    //         for (mat, points_for_mat) in izip!(mats.clone(), *points) {
    //             if mat.height() > max_height {
    //                 max_height = mat.height();
    //             }
    //             let log_height = log2_strict_usize(mat.height());
    //             let reduced_opening_for_log_height = reduced_openings[log_height]
    //                 .get_or_insert_with(|| vec![Challenge::ZERO; mat.height()]);
    //             debug_assert_eq!(reduced_opening_for_log_height.len(), mat.height(), "hello");

    //             let opened_values_for_mat = opened_values_for_round.pushed_mut(vec![]);
    //             for &point in points_for_mat {
    //                 let _guard =
    //                     info_span!("reduce matrix quotient", dims = %mat.dimensions()).entered();

    //                 // Use Barycentric interpolation to evaluate the matrix at the given point.
    //                 let ys = info_span!("compute opened values with Lagrange interpolation")
    //                     .in_scope(|| {
    //                         let (low_coset, _) =
    //                             mat.split_rows(mat.height() >> self.inner.fri.log_blowup);
    //                         interpolate_coset(
    //                             &BitReversalPerm::new_view(low_coset),
    //                             Val::GENERATOR,
    //                             point,
    //                         )
    //                     });

    //                 let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);
    //                 let reduced_ys: Challenge = dot_product(alpha.powers(), ys.iter().copied());

    //                 info_span!("reduce rows").in_scope(|| {
    //                     mat.dot_ext_powers(alpha)
    //                         .zip(reduced_opening_for_log_height.par_iter_mut())
    //                         // This might be longer, but zip will truncate to smaller subgroup
    //                         // (which is ok because it's bitrev)
    //                         .zip(inv_denoms.get(&point).unwrap().par_iter())
    //                         .for_each(|((reduced_row, ro), &inv_denom)| {
    //                             *ro += alpha
    //                                 * alpha_pow_offset
    //                                 * (reduced_row - reduced_ys)
    //                                 * inv_denom
    //                         })
    //                 });

    //                 num_reduced[log_height] += mat.width();
    //                 opened_values_for_mat.push(ys);
    //             }
    //         }
    //     }
    //     // Random poly
    //     let (mats, points) = &mats_and_points[mats_and_points.len() - 1];
    //     let mat = mats[0];
    //     let log_height = log2_strict_usize(mat.height());
    //     let reduced_openings_for_log_height =
    //         reduced_openings[log_height].get_or_insert_with(|| vec![Challenge::ZERO; mat.height()]);
    //     assert!(
    //         mat.height() == max_height,
    //         "mat height mat {} max {}",
    //         mat.height(),
    //         max_height
    //     );
    //     for j in 0..reduced_openings_for_log_height.len() {
    //         reduced_openings_for_log_height[j] += mat.values[j];
    //     }

    //     // Use Barycentric interpolation to evaluate the matrix at the given point.
    //     let ys = info_span!("compute opened values with Lagrange interpolation").in_scope(|| {
    //         let (low_coset, _) = mat.split_rows(mat.height() >> self.inner.fri.log_blowup);
    //         interpolate_coset(
    //             &BitReversalPerm::new_view(low_coset),
    //             Val::GENERATOR,
    //             points[0][0],
    //         )
    //     });
    //     let opened_values_for_round = all_opened_values.pushed_mut(vec![]);
    //     let opened_values_for_mat = opened_values_for_round.pushed_mut(vec![]);
    //     opened_values_for_mat.push(ys);

    //     let fri_input = reduced_openings.into_iter().rev().flatten().collect();

    //     let g: TwoAdicFriGenericConfigForMmcs<Val, InputMmcs> =
    //         TwoAdicFriGenericConfig(PhantomData);

    //     let fri_proof = prover::prove(&g, &self.inner.fri, fri_input, challenger, |index| {
    //         rounds
    //             .iter()
    //             .map(|(data, _)| {
    //                 let log_max_height = log2_strict_usize(self.inner.mmcs.get_max_height(data));
    //                 let bits_reduced = log_global_max_height - log_max_height;
    //                 let reduced_index = index >> bits_reduced;
    //                 let (opened_values, opening_proof) =
    //                     self.inner.mmcs.open_batch(reduced_index, data);
    //                 BatchOpening {
    //                     opened_values,
    //                     opening_proof,
    //                 }
    //             })
    //             .collect()
    //     });

    //     let opened_values_rand = all_opened_values
    //         .iter_mut()
    //         .map(|opened_values_for_round| {
    //             opened_values_for_round
    //                 .iter_mut()
    //                 .map(|opened_values_for_mat| {
    //                     opened_values_for_mat
    //                         .iter_mut()
    //                         .map(|opened_values_for_point| {
    //                             let split =
    //                                 opened_values_for_point.len() - self.num_random_codewords;
    //                             opened_values_for_point.drain(split..).collect()
    //                         })
    //                         .collect()
    //                 })
    //                 .collect()
    //         })
    //         .collect();

    //     (all_opened_values, (opened_values_rand, fri_proof))
    // }

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
//         // Batch combination challenge
//         let alpha: Challenge = challenger.sample_ext_element();

//         let log_global_max_height =
//             inner_proof.commit_phase_commits.len() + self.inner.fri.log_blowup;

//         let g: TwoAdicFriGenericConfigForMmcs<Val, InputMmcs> =
//             TwoAdicFriGenericConfig(PhantomData);

//         verifier::verify(
//             &g,
//             &self.inner.fri,
//             inner_proof,
//             challenger,
//             |index, input_proof| {
//                 // TODO: separate this out into functions

//                 // log_height -> (alpha_pow, reduced_opening)
//                 let mut reduced_openings = BTreeMap::<usize, (Challenge, Challenge)>::new();

//                 for (batch_opening, (batch_commit, mats)) in
//                     izip!(input_proof, &rounds[..rounds.len() - 1])
//                 {
//                     let batch_heights: Vec<_> = mats
//                         .iter()
//                         .map(|(domain, _)| domain.size() << self.inner.fri.log_blowup)
//                         .collect();
//                     let batch_dims: Vec<_> = batch_heights
//                         .iter()
//                         // TODO: MMCS doesn't really need width; we put 0 for now.
//                         .map(|&height| Dimensions { width: 0, height })
//                         .collect();

//                     let batch_max_height = batch_heights.iter().max().expect("Empty batch?");
//                     let log_batch_max_height = log2_strict_usize(*batch_max_height);
//                     let bits_reduced = log_global_max_height - log_batch_max_height;
//                     let reduced_index = index >> bits_reduced;

//                     self.inner.mmcs.verify_batch(
//                         batch_commit,
//                         &batch_dims,
//                         reduced_index,
//                         &batch_opening.opened_values,
//                         &batch_opening.opening_proof,
//                     )?;
//                     for (mat_opening, (mat_domain, mat_points_and_values)) in
//                         izip!(&batch_opening.opened_values, mats)
//                     {
//                         let log_height =
//                             log2_strict_usize(mat_domain.size()) + self.inner.fri.log_blowup;

//                         let bits_reduced = log_global_max_height - log_height;
//                         let rev_reduced_index = reverse_bits_len(index >> bits_reduced, log_height);

//                         // todo: this can be nicer with domain methods?

//                         let x = Val::GENERATOR
//                             * Val::two_adic_generator(log_height).exp_u64(rev_reduced_index as u64);

//                         let (alpha_pow, ro) = reduced_openings
//                             .entry(log_height)
//                             .or_insert((Challenge::ONE, Challenge::ZERO));

//                         for (z, ps_at_z) in mat_points_and_values {
//                             for (&p_at_x, &p_at_z) in izip!(mat_opening, ps_at_z) {
//                                 let quotient = (-p_at_z + p_at_x) / (-*z + x);
//                                 *ro += alpha * *alpha_pow * quotient;
//                                 *alpha_pow *= alpha;
//                             }
//                         }
//                     }
//                 }
//                 let (batch_commit, mats) = &rounds[rounds.len() - 1];
//                 let (mat_domain, points_and_values) = &mats[0];
//                 let log_height = log2_strict_usize(mat_domain.size()) + self.inner.fri.log_blowup;
//                 for (i, p_at_x) in input_proof[input_proof.len() - 1].opened_values[0]
//                     .iter()
//                     .enumerate()
//                 {
//                     reduced_openings
//                         .entry(log_height)
//                         .or_insert((Challenge::ONE, Challenge::ZERO))
//                         .1 += *p_at_x;
//                 }

//                 // `reduced_openings` would have a log_height = log_blowup entry only if there was a
//                 // trace matrix of height 1. In this case the reduced opening can be skipped as it will
//                 // not be checked against any commit phase commit.
//                 if let Some((_alpha_pow, ro)) = reduced_openings.remove(&self.inner.fri.log_blowup)
//                 {
//                     debug_assert!(ro.is_zero());
//                 }

//                 // Return reduced openings descending by log_height.
//                 Ok(reduced_openings
//                     .into_iter()
//                     .rev()
//                     .map(|(log_height, (_alpha_pow, ro))| (log_height, ro))
//                     .collect())
//             },
//         )
//         .expect("fri err");

//         Ok(())
//     }
// }

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
