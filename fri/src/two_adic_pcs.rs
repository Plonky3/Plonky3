use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs, PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, dot_product, ExtensionField,
    Field, TwoAdicField,
};
use p3_interpolation::interpolate_coset;
use p3_matrix::bitrev::{BitReversableMatrix, BitReversalPerm};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits, VecExt};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use crate::verifier::{self, FriError};
use crate::{prover, FriConfig, FriGenericConfig, FriProof};

#[derive(Debug)]
pub struct TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    dft: Dft,
    mmcs: InputMmcs,
    fri: FriConfig<FriMmcs>,
    _phantom: PhantomData<Val>,
}

impl<Val, Dft, InputMmcs, FriMmcs> TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub const fn new(dft: Dft, mmcs: InputMmcs, fri: FriConfig<FriMmcs>) -> Self {
        Self {
            dft,
            mmcs,
            fri,
            _phantom: PhantomData,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct BatchOpening<Val: Field, InputMmcs: Mmcs<Val>> {
    pub opened_values: Vec<Vec<Val>>,
    pub opening_proof: <InputMmcs as Mmcs<Val>>::Proof,
}

pub struct TwoAdicFriGenericConfig<InputProof, InputError>(
    pub PhantomData<(InputProof, InputError)>,
);

pub type TwoAdicFriGenericConfigForMmcs<F, M> =
    TwoAdicFriGenericConfig<Vec<BatchOpening<F, M>>, <M as Mmcs<F>>::Error>;

impl<F: TwoAdicField, InputProof, InputError: Debug> FriGenericConfig<F>
    for TwoAdicFriGenericConfig<InputProof, InputError>
{
    type InputProof = InputProof;
    type InputError = InputError;

    fn extra_query_index_bits(&self) -> usize {
        0
    }

    fn fold_row(
        &self,
        index: usize,
        log_height: usize,
        beta: F,
        evals: impl Iterator<Item = F>,
    ) -> F {
        let arity = 2;
        let log_arity = 1;
        let (e0, e1) = evals
            .collect_tuple()
            .expect("TwoAdicFriFolder only supports arity=2");
        // If performance critical, make this API stateful to avoid this
        // This is a bit more math than is necessary, but leaving it here
        // in case we want higher arity in the future
        let subgroup_start = F::two_adic_generator(log_height + log_arity)
            .exp_u64(reverse_bits_len(index, log_height) as u64);
        let mut xs = F::two_adic_generator(log_arity)
            .shifted_powers(subgroup_start)
            .take(arity)
            .collect_vec();
        reverse_slice_index_bits(&mut xs);
        assert_eq!(log_arity, 1, "can only interpolate two points for now");
        // interpolate and evaluate at beta
        e0 + (beta - xs[0]) * (e1 - e0) / (xs[1] - xs[0])
    }

    fn fold_matrix<M: Matrix<F>>(&self, beta: F, m: M) -> Vec<F> {
        // We use the fact that
        //     p_e(x^2) = (p(x) + p(-x)) / 2
        //     p_o(x^2) = (p(x) - p(-x)) / (2 x)
        // that is,
        //     p_e(g^(2i)) = (p(g^i) + p(g^(n/2 + i))) / 2
        //     p_o(g^(2i)) = (p(g^i) - p(g^(n/2 + i))) / (2 g^i)
        // so
        //     result(g^(2i)) = p_e(g^(2i)) + beta p_o(g^(2i))
        //                    = (1/2 + beta/2 g_inv^i) p(g^i)
        //                    + (1/2 - beta/2 g_inv^i) p(g^(n/2 + i))
        let g_inv = F::two_adic_generator(log2_strict_usize(m.height()) + 1).inverse();
        let one_half = F::two().inverse();
        let half_beta = beta * one_half;

        // TODO: vectorize this (after we have packed extension fields)

        // beta/2 times successive powers of g_inv
        let mut powers = g_inv
            .shifted_powers(half_beta)
            .take(m.height())
            .collect_vec();
        reverse_slice_index_bits(&mut powers);

        m.par_rows()
            .zip(powers)
            .map(|(mut row, power)| {
                let (lo, hi) = row.next_tuple().unwrap();
                (one_half + power) * lo + (one_half - power) * hi
            })
            .collect()
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type Proof = FriProof<Challenge, FriMmcs, Val, Vec<BatchOpening<Val, InputMmcs>>>;
    type Error = FriError<FriMmcs::Error, InputMmcs::Error>;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        let log_n = log2_strict_usize(degree);
        TwoAdicMultiplicativeCoset {
            log_n,
            shift: Val::one(),
        }
    }

    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let ldes: Vec<_> = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                let shift = Val::generator() / domain.shift;
                // Commit to the bit-reversed LDE.
                self.dft
                    .coset_lde_batch(evals, self.fri.log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect();

        self.mmcs.commit(ldes)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> impl Matrix<Val> + 'a {
        // todo: handle extrapolation for LDEs we don't have
        assert_eq!(domain.shift, Val::generator());
        let lde = self.mmcs.get_matrices(prover_data)[idx];
        assert!(lde.height() >= domain.size());
        lde.split_rows(domain.size()).0.bit_reverse_rows()
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
        /*

        A quick rundown of the optimizations in this function:
        We are trying to compute sum_i alpha^i * (p(X) - y)/(X - z),
        for each z an opening point, y = p(z). Each p(X) is given as evaluations in bit-reversed order
        in the columns of the matrices. y is computed by barycentric interpolation.
        X and p(X) are in the base field; alpha, y and z are in the extension.
        The primary goal is to minimize extension multiplications.

        - Instead of computing all alpha^i, we just compute alpha^i for i up to the largest width
        of a matrix, then multiply by an "alpha offset" when accumulating.
              a^0 x0 + a^1 x1 + a^2 x2 + a^3 x3 + ...
            = a^0 ( a^0 x0 + a^1 x1 ) + a^2 ( a^0 x2 + a^1 x3 ) + ...
            (see `alpha_pows`, `alpha_pow_offset`, `num_reduced`)

        - For each unique point z, we precompute 1/(X-z) for the largest subgroup opened at this point.
        Since we compute it in bit-reversed order, smaller subgroups can simply truncate the vector.
            (see `inv_denoms`)

        - Then, for each matrix (with columns p_i) and opening point z, we want:
            for each row (corresponding to subgroup element X):
                reduced[X] += alpha_offset * sum_i [ alpha^i * inv_denom[X] * (p_i[X] - y[i]) ]

            We can factor out inv_denom, and expand what's left:
                reduced[X] += alpha_offset * inv_denom[X] * sum_i [ alpha^i * p_i[X] - alpha^i * y[i] ]

            And separate the sum:
                reduced[X] += alpha_offset * inv_denom[X] * [ sum_i [ alpha^i * p_i[X] ] - sum_i [ alpha^i * y[i] ] ]

            And now the last sum doesn't depend on X, so we can precompute that for the matrix, too.
            So the hot loop (that depends on both X and i) is just:
                sum_i [ alpha^i * p_i[X] ]

            with alpha^i an extension, p_i[X] a base

        */

        // Batch combination challenge
        let alpha: Challenge = challenger.sample_ext_element();

        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| {
                (
                    self.mmcs
                        .get_matrices(data)
                        .into_iter()
                        .map(|m| m.as_view())
                        .collect_vec(),
                    points,
                )
            })
            .collect_vec();
        let mats = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats)
            .collect_vec();

        let global_max_height = mats.iter().map(|m| m.height()).max().unwrap();
        let log_global_max_height = log2_strict_usize(global_max_height);

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(X - z) for the largest subgroup (in bitrev order).
        let inv_denoms = compute_inverse_denominators(&mats_and_points, Val::generator());

        let mut all_opened_values: OpenedValues<Challenge> = vec![];

        let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);
        let mut num_reduced = [0; 32];

        for (mats, points) in mats_and_points {
            let opened_values_for_round = all_opened_values.pushed_mut(vec![]);
            for (mat, points_for_mat) in izip!(mats, points) {
                let log_height = log2_strict_usize(mat.height());
                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| vec![Challenge::zero(); mat.height()]);
                debug_assert_eq!(reduced_opening_for_log_height.len(), mat.height());

                let opened_values_for_mat = opened_values_for_round.pushed_mut(vec![]);
                for &point in points_for_mat {
                    let _guard =
                        info_span!("reduce matrix quotient", dims = %mat.dimensions()).entered();

                    // Use Barycentric interpolation to evaluate the matrix at the given point.
                    let ys = info_span!("compute opened values with Lagrange interpolation")
                        .in_scope(|| {
                            let (low_coset, _) =
                                mat.split_rows(mat.height() >> self.fri.log_blowup);
                            interpolate_coset(
                                &BitReversalPerm::new_view(low_coset),
                                Val::generator(),
                                point,
                            )
                        });

                    let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);
                    let reduced_ys: Challenge = dot_product(alpha.powers(), ys.iter().copied());

                    info_span!("reduce rows").in_scope(|| {
                        mat.dot_ext_powers(alpha)
                            .zip(reduced_opening_for_log_height.par_iter_mut())
                            // This might be longer, but zip will truncate to smaller subgroup
                            // (which is ok because it's bitrev)
                            .zip(inv_denoms.get(&point).unwrap().par_iter())
                            .for_each(|((reduced_row, ro), &inv_denom)| {
                                *ro += alpha_pow_offset * (reduced_row - reduced_ys) * inv_denom
                            })
                    });

                    num_reduced[log_height] += mat.width();
                    opened_values_for_mat.push(ys);
                }
            }
        }

        let fri_input = reduced_openings.into_iter().rev().flatten().collect_vec();

        let g: TwoAdicFriGenericConfigForMmcs<Val, InputMmcs> =
            TwoAdicFriGenericConfig(PhantomData);

        let fri_proof = prover::prove(&g, &self.fri, fri_input, challenger, |index| {
            rounds
                .iter()
                .map(|(data, _)| {
                    let log_max_height = log2_strict_usize(self.mmcs.get_max_height(data));
                    let bits_reduced = log_global_max_height - log_max_height;
                    let reduced_index = index >> bits_reduced;
                    let (opened_values, opening_proof) = self.mmcs.open_batch(reduced_index, data);
                    BatchOpening {
                        opened_values,
                        opening_proof,
                    }
                })
                .collect()
        });

        (all_opened_values, fri_proof)
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
        // Batch combination challenge
        let alpha: Challenge = challenger.sample_ext_element();

        let log_global_max_height = proof.commit_phase_commits.len() + self.fri.log_blowup;

        let g: TwoAdicFriGenericConfigForMmcs<Val, InputMmcs> =
            TwoAdicFriGenericConfig(PhantomData);

        verifier::verify(&g, &self.fri, proof, challenger, |index, input_proof| {
            // TODO: separate this out into functions

            // log_height -> (alpha_pow, reduced_opening)
            let mut reduced_openings = BTreeMap::<usize, (Challenge, Challenge)>::new();

            for (batch_opening, (batch_commit, mats)) in izip!(input_proof, &rounds) {
                let batch_heights = mats
                    .iter()
                    .map(|(domain, _)| domain.size() << self.fri.log_blowup)
                    .collect_vec();
                let batch_dims = batch_heights
                    .iter()
                    // TODO: MMCS doesn't really need width; we put 0 for now.
                    .map(|&height| Dimensions { width: 0, height })
                    .collect_vec();

                let batch_max_height = batch_heights.iter().max().expect("Empty batch?");
                let log_batch_max_height = log2_strict_usize(*batch_max_height);
                let bits_reduced = log_global_max_height - log_batch_max_height;
                let reduced_index = index >> bits_reduced;

                self.mmcs.verify_batch(
                    batch_commit,
                    &batch_dims,
                    reduced_index,
                    &batch_opening.opened_values,
                    &batch_opening.opening_proof,
                )?;
                for (mat_opening, (mat_domain, mat_points_and_values)) in
                    izip!(&batch_opening.opened_values, mats)
                {
                    let log_height = log2_strict_usize(mat_domain.size()) + self.fri.log_blowup;

                    let bits_reduced = log_global_max_height - log_height;
                    let rev_reduced_index = reverse_bits_len(index >> bits_reduced, log_height);

                    // todo: this can be nicer with domain methods?

                    let x = Val::generator()
                        * Val::two_adic_generator(log_height).exp_u64(rev_reduced_index as u64);

                    let (alpha_pow, ro) = reduced_openings
                        .entry(log_height)
                        .or_insert((Challenge::one(), Challenge::zero()));

                    for (z, ps_at_z) in mat_points_and_values {
                        for (&p_at_x, &p_at_z) in izip!(mat_opening, ps_at_z) {
                            let quotient = (-p_at_z + p_at_x) / (-*z + x);
                            *ro += *alpha_pow * quotient;
                            *alpha_pow *= alpha;
                        }
                    }
                }
            }

            // Return reduced openings descending by log_height.
            Ok(reduced_openings
                .into_iter()
                .rev()
                .map(|(log_height, (_alpha_pow, ro))| (log_height, ro))
                .collect())
        })
        .expect("fri err");

        Ok(())
    }
}

#[instrument(skip_all)]
fn compute_inverse_denominators<F: TwoAdicField, EF: ExtensionField<F>, M: Matrix<F>>(
    mats_and_points: &[(Vec<M>, &Vec<Vec<EF>>)],
    coset_shift: F,
) -> LinearMap<EF, Vec<EF>> {
    let mut max_log_height_for_point: LinearMap<EF, usize> = LinearMap::new();
    for (mats, points) in mats_and_points {
        for (mat, points_for_mat) in izip!(mats, *points) {
            let log_height = log2_strict_usize(mat.height());
            for &z in points_for_mat {
                if let Some(lh) = max_log_height_for_point.get_mut(&z) {
                    *lh = core::cmp::max(*lh, log_height);
                } else {
                    max_log_height_for_point.insert(z, log_height);
                }
            }
        }
    }

    // Compute the largest subgroup we will use, in bitrev order.
    let max_log_height = *max_log_height_for_point.values().max().unwrap();
    let mut subgroup = cyclic_subgroup_coset_known_order(
        F::two_adic_generator(max_log_height),
        coset_shift,
        1 << max_log_height,
    )
    .collect_vec();
    reverse_slice_index_bits(&mut subgroup);

    max_log_height_for_point
        .into_iter()
        .map(|(z, log_height)| {
            (
                z,
                batch_multiplicative_inverse(
                    &subgroup[..(1 << log_height)]
                        .iter()
                        .map(|&x| EF::from_base(x) - z)
                        .collect_vec(),
                ),
            )
        })
        .collect()
}
