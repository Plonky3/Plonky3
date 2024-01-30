use core::marker::PhantomData;
use std::collections::HashMap;

use alloc::vec;
use alloc::vec::Vec;
use itertools::{izip, Itertools};
use p3_challenger::{CanSample, FieldChallenger};
use p3_commit::{DirectMmcs, OpenedValues, Pcs, UnivariatePcs, UnivariatePcsWithLde};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, AbstractExtensionField,
    AbstractField, ExtensionField, Field, TwoAdicField,
};
use p3_interpolation::interpolate_coset;
use p3_matrix::{
    bitrev::{BitReversableMatrix, BitReversedMatrixView},
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Dimensions, Matrix, MatrixRows,
};
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_slice_index_bits, VecExt};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use crate::{prover, verifier::VerificationErrorForFriConfig, FriConfig, FriProof};

pub struct TwoAdicFriPcs<FC, Val, Dft, M> {
    fri: FC,
    dft: Dft,
    mmcs: M,
    _phantom: PhantomData<Val>,
}

impl<FC, Val, Dft, M> TwoAdicFriPcs<FC, Val, Dft, M> {
    pub fn new(fri: FC, dft: Dft, mmcs: M) -> Self {
        Self {
            fri,
            dft,
            mmcs,
            _phantom: PhantomData,
        }
    }
    fn coset_shift(&self) -> Val
    where
        Val: TwoAdicField,
    {
        Val::generator()
    }
}

#[derive(Serialize, Deserialize)]
pub struct TwoAdicFriPcsProof<FC: FriConfig, Val, InputMmcsProof> {
    #[serde(bound = "")]
    pub(crate) fri_proof: FriProof<FC>,
    /// For each query, for each committed batch, query openings for that batch
    pub(crate) input_openings: Vec<Vec<InputOpening<Val, InputMmcsProof>>>,
}

#[derive(Serialize, Deserialize)]
pub struct InputOpening<Val, InputMmcsProof> {
    pub(crate) opened_values: Vec<Vec<Val>>,
    pub(crate) opening_proof: InputMmcsProof,
}

impl<FC, Val, Dft, M, In> Pcs<Val, In> for TwoAdicFriPcs<FC, Val, Dft, M>
where
    Val: TwoAdicField,
    FC: FriConfig,
    FC::Challenge: ExtensionField<Val>,
    FC::Challenger: FieldChallenger<Val>,
    Dft: TwoAdicSubgroupDft<Val>,
    M: 'static + for<'a> DirectMmcs<Val, Mat<'a> = RowMajorMatrixView<'a, Val>>,
    In: MatrixRows<Val>,
{
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = TwoAdicFriPcsProof<FC, Val, M::Proof>;
    type Error = VerificationErrorForFriConfig<FC>;

    fn commit_batches(&self, polynomials: Vec<In>) -> (Self::Commitment, Self::ProverData) {
        self.commit_shifted_batches(polynomials, Val::one())
    }
}

impl<FC, Val, Dft, M, In> UnivariatePcsWithLde<Val, FC::Challenge, In, FC::Challenger>
    for TwoAdicFriPcs<FC, Val, Dft, M>
where
    Val: TwoAdicField,
    FC: FriConfig,
    FC::Challenge: ExtensionField<Val>,
    FC::Challenger: FieldChallenger<Val>,
    Dft: TwoAdicSubgroupDft<Val>,
    M: 'static + for<'a> DirectMmcs<Val, Mat<'a> = RowMajorMatrixView<'a, Val>>,
    In: MatrixRows<Val>,
{
    type Lde<'a> = BitReversedMatrixView<M::Mat<'a>> where Self: 'a;

    fn coset_shift(&self) -> Val {
        self.coset_shift()
    }

    fn log_blowup(&self) -> usize {
        self.fri.log_blowup()
    }

    fn get_ldes<'a, 'b>(&'a self, prover_data: &'b Self::ProverData) -> Vec<Self::Lde<'b>>
    where
        'a: 'b,
    {
        // We committed to the bit-reversed LDE, so now we wrap it to return in natural order.
        self.mmcs
            .get_matrices(prover_data)
            .into_iter()
            .map(|m| BitReversedMatrixView::new(m))
            .collect()
    }

    fn commit_shifted_batches(
        &self,
        polynomials: Vec<In>,
        coset_shift: Val,
    ) -> (Self::Commitment, Self::ProverData) {
        let shift = self.coset_shift() / coset_shift;
        let ldes = info_span!("compute all coset LDEs").in_scope(|| {
            polynomials
                .into_iter()
                .map(|poly| {
                    let input = poly.to_row_major_matrix();
                    // Commit to the bit-reversed LDE.
                    self.dft
                        .coset_lde_batch(input, self.fri.log_blowup(), shift)
                        .bit_reverse_rows()
                        .to_row_major_matrix()
                })
                .collect()
        });
        self.mmcs.commit(ldes)
    }
}

impl<FC, Val, Dft, M, In> UnivariatePcs<Val, FC::Challenge, In, FC::Challenger>
    for TwoAdicFriPcs<FC, Val, Dft, M>
where
    Val: TwoAdicField,
    FC: FriConfig,
    FC::Challenge: ExtensionField<Val>,
    FC::Challenger: FieldChallenger<Val>,
    Dft: TwoAdicSubgroupDft<Val>,
    M: 'static + for<'a> DirectMmcs<Val, Mat<'a> = RowMajorMatrixView<'a, Val>>,
    In: MatrixRows<Val>,
{
    #[instrument(name = "open_multi_batches", skip_all)]
    fn open_multi_batches(
        &self,
        prover_data_and_points: &[(&Self::ProverData, &[Vec<FC::Challenge>])],
        challenger: &mut FC::Challenger,
    ) -> (OpenedValues<FC::Challenge>, Self::Proof) {
        // Batch combination challenge
        let alpha = <FC::Challenger as CanSample<FC::Challenge>>::sample(challenger);
        let coset_shift = self.coset_shift();

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
            = a^0 ( a^0 x0 + a^1 x1 ) + a^2 ( a^0 x0 + a^1 x1 ) + ...
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
                reduced[X] += alpha_offset * inv_denom[X] * sum_i [ alpha^i * p_i[X] ] - sum_i [ alpha^i * y[i] ]

            And now the last sum doesn't depend on X, so we can precompute that for the matrix, too.
            So the hot loop (that depends on both X and i) is just:
                sum_i [ alpha^i * p_i[X] ]

            with alpha^i an extension, p_i[X] a base

        */

        let mats_and_points: Vec<(Vec<_>, &[Vec<FC::Challenge>])> = prover_data_and_points
            .into_iter()
            .map(|(data, points)| (self.mmcs.get_matrices(data), *points))
            .collect();

        let max_width = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats)
            .map(|mat| mat.width())
            .max()
            .unwrap();

        let alpha_pows: Vec<FC::Challenge> = alpha.powers().take(max_width).collect();

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(X - z) for the largest subgroup (in bitrev order).
        let inv_denoms: HashMap<FC::Challenge, Vec<FC::Challenge>> =
            info_span!("invert denominators").in_scope(|| {
                let mut max_log_height_for_point = HashMap::<FC::Challenge, usize>::new();
                for (mats, points) in mats_and_points {
                    for (mat, points_for_mat) in izip!(mats, points) {
                        let log_height = log2_strict_usize(mat.height());
                        for z in points_for_mat {
                            max_log_height_for_point
                                .entry(*z)
                                .and_modify(|lh| *lh = core::cmp::max(*lh, log_height))
                                .or_insert(log_height);
                        }
                    }
                }
                let max_log_height = *max_log_height_for_point.values().max().unwrap();
                // Compute the largest subgroup we will use, in bitrev order.
                let mut subgroup = cyclic_subgroup_coset_known_order(
                    Val::two_adic_generator(max_log_height),
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
                                    .into_iter()
                                    .map(|&x| FC::Challenge::from_base(x) - z)
                                    .collect_vec(),
                            ),
                        )
                    })
                    .collect()
            });

        let mut all_opened_values: OpenedValues<FC::Challenge> = vec![];
        let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);
        let mut num_reduced = [0; 32];

        for (data, points) in prover_data_and_points {
            let mats = self.mmcs.get_matrices(data);
            let opened_values_for_round = all_opened_values.pushed_mut(vec![]);
            for (mat, points_for_mat) in izip!(mats, *points) {
                let log_height = log2_strict_usize(mat.height());
                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| vec![FC::Challenge::zero(); mat.height()]);
                debug_assert_eq!(reduced_opening_for_log_height.len(), mat.height());

                let opened_values_for_mat = opened_values_for_round.pushed_mut(vec![]);
                for &point in points_for_mat {
                    let _guard =
                        info_span!("reduce matrix quotient", dims = %mat.dimensions()).entered();

                    // Use Barycentric interpolation to evaluate the matrix at the given point.
                    let ys = info_span!("compute opened values with Lagrange interpolation")
                        .in_scope(|| {
                            let (low_coset, _) =
                                mat.split_rows(mat.height() >> self.fri.log_blowup());
                            interpolate_coset(
                                &BitReversedMatrixView::new(low_coset),
                                coset_shift,
                                point,
                            )
                        });

                    let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);

                    let sum_alpha_pows_times_neg_y: FC::Challenge = izip!(&alpha_pows, &ys)
                        .map(|(&alpha_pow, &y)| alpha_pow * -y)
                        .sum();

                    info_span!("reduce rows").in_scope(|| {
                        reduced_opening_for_log_height
                            .par_iter_mut()
                            .zip_eq(mat.par_rows())
                            .zip(inv_denoms.get(&point).unwrap())
                            .for_each(|((reduced_opening, row), &inv_denom)| {
                                let row_sum: FC::Challenge = izip!(row, &alpha_pows)
                                    .map(|(&p_at_x, &alpha_pow)| {
                                        // Hot loop is just sum of (ext*base).
                                        alpha_pow * p_at_x
                                    })
                                    .sum();
                                *reduced_opening += inv_denom
                                    * alpha_pow_offset
                                    * (sum_alpha_pows_times_neg_y + row_sum);
                            });

                        /*
                        for (row, reduced_opening, &inv_denom) in izip!(
                            mat.rows(),
                            reduced_opening_for_log_height.iter_mut(),
                            // This might be longer, but zip will truncate to smaller subgroup
                            // (which is ok because it's bitrev)
                            inv_denoms.get(&point).unwrap(),
                        ) {
                            let row_sum: FC::Challenge = izip!(row, &alpha_pows)
                                .map(|(&p_at_x, &alpha_pow)| {
                                    // Hot loop is just sum of (ext*base).
                                    alpha_pow * p_at_x
                                })
                                .sum();
                            *reduced_opening += inv_denom
                                * alpha_pow_offset
                                * (sum_alpha_pows_times_neg_y + row_sum);
                        }
                        */
                    });

                    num_reduced[log_height] += mat.width();
                    opened_values_for_mat.push(ys);
                }
            }
        }

        let (fri_proof, query_indices) = prover::prove(&self.fri, &reduced_openings, challenger);

        let input_openings = query_indices
            .into_iter()
            .map(|index| {
                prover_data_and_points
                    .iter()
                    .map(|(data, _)| {
                        let (opened_values, opening_proof) = self.mmcs.open_batch(index, data);
                        InputOpening {
                            opened_values,
                            opening_proof,
                        }
                    })
                    .collect()
            })
            .collect();

        (
            all_opened_values,
            TwoAdicFriPcsProof {
                fri_proof,
                input_openings,
            },
        )
    }

    fn verify_multi_batches(
        &self,
        commits_and_points: &[(Self::Commitment, &[Vec<FC::Challenge>])],
        dims: &[Vec<Dimensions>],
        values: OpenedValues<FC::Challenge>,
        proof: &Self::Proof,
        challenger: &mut FC::Challenger,
    ) -> Result<(), Self::Error> {
        // todo!()
        Ok(())
    }
}
