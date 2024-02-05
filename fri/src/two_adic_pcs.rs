use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, CanSample, FieldChallenger, GrindingChallenger};
use p3_commit::{DirectMmcs, Mmcs, OpenedValues, Pcs, UnivariatePcs, UnivariatePcsWithLde};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, AbstractField, ExtensionField,
    Field, PackedField, TwoAdicField,
};
use p3_interpolation::interpolate_coset;
use p3_matrix::bitrev::{BitReversableMatrix, BitReversedMatrixView};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::{Dimensions, Matrix, MatrixRows};
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits, VecExt};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use crate::verifier::{self, FriError};
use crate::{prover, FriConfig, FriProof};

/// We group all of our type bounds into this trait to reduce duplication across signatures.
pub trait TwoAdicFriPcsGenericConfig: Default {
    type Val: TwoAdicField;
    type Challenge: TwoAdicField + ExtensionField<Self::Val>;
    type Challenger: FieldChallenger<Self::Val>
        + GrindingChallenger<Witness = Self::Val>
        + CanObserve<<Self::FriMmcs as Mmcs<Self::Challenge>>::Commitment>
        + CanSample<Self::Challenge>;
    type Dft: TwoAdicSubgroupDft<Self::Val>;
    type InputMmcs: 'static
        + for<'a> DirectMmcs<Self::Val, Mat<'a> = RowMajorMatrixView<'a, Self::Val>>;
    type FriMmcs: DirectMmcs<Self::Challenge>;
}

pub struct TwoAdicFriPcsConfig<Val, Challenge, Challenger, Dft, InputMmcs, FriMmcs>(
    PhantomData<(Val, Challenge, Challenger, Dft, InputMmcs, FriMmcs)>,
);
impl<Val, Challenge, Challenger, Dft, InputMmcs, FriMmcs> Default
    for TwoAdicFriPcsConfig<Val, Challenge, Challenger, Dft, InputMmcs, FriMmcs>
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Val, Challenge, Challenger, Dft, InputMmcs, FriMmcs> TwoAdicFriPcsGenericConfig
    for TwoAdicFriPcsConfig<Val, Challenge, Challenger, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger: FieldChallenger<Val>
        + GrindingChallenger<Witness = Val>
        + CanObserve<<FriMmcs as Mmcs<Challenge>>::Commitment>
        + CanSample<Challenge>,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: 'static + for<'a> DirectMmcs<Val, Mat<'a> = RowMajorMatrixView<'a, Val>>,
    FriMmcs: DirectMmcs<Challenge>,
{
    type Val = Val;
    type Challenge = Challenge;
    type Challenger = Challenger;
    type Dft = Dft;
    type InputMmcs = InputMmcs;
    type FriMmcs = FriMmcs;
}

pub struct TwoAdicFriPcs<C: TwoAdicFriPcsGenericConfig> {
    fri: FriConfig<C::FriMmcs>,
    dft: C::Dft,
    mmcs: C::InputMmcs,
}

impl<C: TwoAdicFriPcsGenericConfig> TwoAdicFriPcs<C> {
    pub fn new(fri: FriConfig<C::FriMmcs>, dft: C::Dft, mmcs: C::InputMmcs) -> Self {
        Self { fri, dft, mmcs }
    }
}

#[derive(Debug)]
pub enum VerificationError<C: TwoAdicFriPcsGenericConfig> {
    InputMmcsError(<C::InputMmcs as Mmcs<C::Val>>::Error),
    FriError(FriError<<C::FriMmcs as Mmcs<C::Challenge>>::Error>),
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct TwoAdicFriPcsProof<C: TwoAdicFriPcsGenericConfig> {
    pub(crate) fri_proof: FriProof<C::Challenge, C::FriMmcs, C::Val>,
    /// For each query, for each committed batch, query openings for that batch
    pub(crate) query_openings: Vec<Vec<BatchOpening<C>>>,
}

#[derive(Serialize, Deserialize)]
pub struct BatchOpening<C: TwoAdicFriPcsGenericConfig> {
    pub(crate) opened_values: Vec<Vec<C::Val>>,
    pub(crate) opening_proof: <C::InputMmcs as Mmcs<C::Val>>::Proof,
}

impl<C: TwoAdicFriPcsGenericConfig, In: MatrixRows<C::Val>> Pcs<C::Val, In> for TwoAdicFriPcs<C> {
    type Commitment = <C::InputMmcs as Mmcs<C::Val>>::Commitment;
    type ProverData = <C::InputMmcs as Mmcs<C::Val>>::ProverData;
    type Proof = TwoAdicFriPcsProof<C>;
    type Error = VerificationError<C>;

    fn commit_batches(&self, polynomials: Vec<In>) -> (Self::Commitment, Self::ProverData) {
        self.commit_shifted_batches(polynomials, C::Val::one())
    }
}

impl<C: TwoAdicFriPcsGenericConfig, In: MatrixRows<C::Val>>
    UnivariatePcsWithLde<C::Val, C::Challenge, In, C::Challenger> for TwoAdicFriPcs<C>
{
    type Lde<'a> = BitReversedMatrixView<<C::InputMmcs as Mmcs<C::Val>>::Mat<'a>> where Self: 'a;

    fn coset_shift(&self) -> C::Val {
        C::Val::generator()
    }

    fn log_blowup(&self) -> usize {
        self.fri.log_blowup
    }

    fn get_ldes<'a, 'b>(&'a self, prover_data: &'b Self::ProverData) -> Vec<Self::Lde<'b>>
    where
        'a: 'b,
    {
        // We committed to the bit-reversed LDE, so now we wrap it to return in natural order.
        self.mmcs
            .get_matrices(prover_data)
            .into_iter()
            .map(BitReversedMatrixView::new)
            .collect()
    }

    fn commit_shifted_batches(
        &self,
        polynomials: Vec<In>,
        coset_shift: C::Val,
    ) -> (Self::Commitment, Self::ProverData) {
        let shift = C::Val::generator() / coset_shift;
        let ldes = info_span!("compute all coset LDEs").in_scope(|| {
            polynomials
                .into_iter()
                .map(|poly| {
                    let input = poly.to_row_major_matrix();
                    // Commit to the bit-reversed LDE.
                    self.dft
                        .coset_lde_batch(input, self.fri.log_blowup, shift)
                        .bit_reverse_rows()
                        .to_row_major_matrix()
                })
                .collect()
        });
        self.mmcs.commit(ldes)
    }
}

impl<C: TwoAdicFriPcsGenericConfig, In: MatrixRows<C::Val>>
    UnivariatePcs<C::Val, C::Challenge, In, C::Challenger> for TwoAdicFriPcs<C>
{
    #[instrument(name = "open_multi_batches", skip_all)]
    fn open_multi_batches(
        &self,
        prover_data_and_points: &[(&Self::ProverData, &[Vec<C::Challenge>])],
        challenger: &mut C::Challenger,
    ) -> (OpenedValues<C::Challenge>, Self::Proof) {
        // Batch combination challenge
        let alpha = <C::Challenger as CanSample<C::Challenge>>::sample(challenger);

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

        let mats_and_points = prover_data_and_points
            .iter()
            .map(|(data, points)| (self.mmcs.get_matrices(data), *points))
            .collect_vec();

        let max_width = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats)
            .map(|mat| mat.width())
            .max()
            .unwrap();

        let alpha_reducer = PowersReducer::<C::Val, C::Challenge>::new(alpha, max_width);

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(X - z) for the largest subgroup (in bitrev order).
        let inv_denoms = compute_inverse_denominators(&mats_and_points, C::Val::generator());

        let mut all_opened_values: OpenedValues<C::Challenge> = vec![];
        let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);
        let mut num_reduced = [0; 32];

        for (data, points) in prover_data_and_points {
            let mats = self.mmcs.get_matrices(data);
            let opened_values_for_round = all_opened_values.pushed_mut(vec![]);
            for (mat, points_for_mat) in izip!(mats, *points) {
                let log_height = log2_strict_usize(mat.height());
                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| vec![C::Challenge::zero(); mat.height()]);
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
                                &BitReversedMatrixView::new(low_coset),
                                C::Val::generator(),
                                point,
                            )
                        });

                    let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);
                    let sum_alpha_pows_times_y = alpha_reducer.reduce_ext(&ys);

                    info_span!("reduce rows").in_scope(|| {
                        reduced_opening_for_log_height
                            .par_iter_mut()
                            .zip_eq(mat.par_rows())
                            // This might be longer, but zip will truncate to smaller subgroup
                            // (which is ok because it's bitrev)
                            .zip(inv_denoms.get(&point).unwrap())
                            .for_each(|((reduced_opening, row), &inv_denom)| {
                                let row_sum = alpha_reducer.reduce_base(row);
                                *reduced_opening += inv_denom
                                    * alpha_pow_offset
                                    * (row_sum - sum_alpha_pows_times_y);
                            });
                    });

                    num_reduced[log_height] += mat.width();
                    opened_values_for_mat.push(ys);
                }
            }
        }

        let (fri_proof, query_indices) = prover::prove(&self.fri, &reduced_openings, challenger);

        let query_openings = query_indices
            .into_iter()
            .map(|index| {
                prover_data_and_points
                    .iter()
                    .map(|(data, _)| {
                        let (opened_values, opening_proof) = self.mmcs.open_batch(index, data);
                        BatchOpening {
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
                query_openings,
            },
        )
    }

    fn verify_multi_batches(
        &self,
        commits_and_points: &[(Self::Commitment, &[Vec<C::Challenge>])],
        dims: &[Vec<Dimensions>],
        values: OpenedValues<C::Challenge>,
        proof: &Self::Proof,
        challenger: &mut C::Challenger,
    ) -> Result<(), Self::Error> {
        // Batch combination challenge
        let alpha = <C::Challenger as CanSample<C::Challenge>>::sample(challenger);

        let fri_challenges =
            verifier::verify_shape_and_sample_challenges(&self.fri, &proof.fri_proof, challenger)
                .map_err(VerificationError::FriError)?;

        let log_max_height = proof.fri_proof.commit_phase_commits.len() + self.fri.log_blowup;

        let reduced_openings: Vec<[C::Challenge; 32]> = proof
            .query_openings
            .iter()
            .zip(&fri_challenges.query_indices)
            .map(|(query_opening, &index)| {
                let x = C::Val::generator()
                    * C::Val::two_adic_generator(log_max_height).exp_u64(reverse_bits_len(
                        index,
                        log_max_height,
                    )
                        as u64);

                let mut ro = [C::Challenge::zero(); 32];
                let mut alpha_pow = [C::Challenge::one(); 32];
                for (batch_opening, batch_dims, (batch_commit, batch_points), batch_at_z) in
                    izip!(query_opening, dims, commits_and_points, &values)
                {
                    self.mmcs.verify_batch(
                        batch_commit,
                        batch_dims,
                        index,
                        &batch_opening.opened_values,
                        &batch_opening.opening_proof,
                    )?;
                    for (mat_opening, mat_dims, mat_points, mat_at_z) in izip!(
                        &batch_opening.opened_values,
                        batch_dims,
                        *batch_points,
                        batch_at_z
                    ) {
                        let log_height = log2_strict_usize(mat_dims.height) + self.fri.log_blowup;

                        for (&z, ps_at_z) in izip!(mat_points, mat_at_z) {
                            for (&p_at_x, &p_at_z) in izip!(mat_opening, ps_at_z) {
                                let quotient = (-p_at_z + p_at_x) / (-z + x);
                                ro[log_height] += alpha_pow[log_height] * quotient;
                                alpha_pow[log_height] *= alpha;
                            }
                        }
                    }
                }
                Ok(ro)
            })
            .collect::<Result<Vec<_>, <C::InputMmcs as Mmcs<C::Val>>::Error>>()
            .map_err(VerificationError::InputMmcsError)?;

        verifier::verify_challenges(
            &self.fri,
            &proof.fri_proof,
            &fri_challenges,
            &reduced_openings,
        )
        .map_err(VerificationError::FriError)?;

        Ok(())
    }
}

#[instrument(skip_all)]
fn compute_inverse_denominators<F: TwoAdicField, EF: ExtensionField<F>, M: Matrix<F>>(
    mats_and_points: &[(Vec<M>, &[Vec<EF>])],
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
    let max_log_height = *max_log_height_for_point.values().max().unwrap();
    // Compute the largest subgroup we will use, in bitrev order.
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

struct PowersReducer<F: Field, EF> {
    powers: Vec<EF>,
    // If EF::D = 2 and powers is [01 23 45 67],
    // this holds [[02 46] [13 57]]
    transposed_packed: Vec<Vec<F::Packing>>,
}

impl<F: Field, EF: ExtensionField<F>> PowersReducer<F, EF> {
    fn new(base: EF, max_width: usize) -> Self {
        let powers: Vec<EF> = base
            .powers()
            .take(max_width.next_multiple_of(F::Packing::WIDTH))
            .collect();

        let transposed_packed: Vec<Vec<F::Packing>> = transpose_vec(
            (0..EF::D)
                .map(|d| {
                    F::Packing::pack_slice(
                        &powers.iter().map(|a| a.as_base_slice()[d]).collect_vec(),
                    )
                    .to_vec()
                })
                .collect(),
        );

        Self {
            powers,
            transposed_packed,
        }
    }

    // Compute sum_i base^i * x_i
    fn reduce_ext(&self, xs: &[EF]) -> EF {
        self.powers.iter().zip(xs).map(|(&pow, &x)| pow * x).sum()
    }

    // Same as `self.powers.iter().zip(xs).map(|(&pow, &x)| pow * x).sum()`
    fn reduce_base(&self, xs: &[F]) -> EF {
        let (xs_packed, xs_sfx) = F::Packing::pack_slice_with_suffix(xs);
        let mut sums = (0..EF::D).map(|_| F::Packing::zero()).collect::<Vec<_>>();
        for (&x, pows) in izip!(xs_packed, &self.transposed_packed) {
            for d in 0..EF::D {
                sums[d] += x * pows[d];
            }
        }
        let packed_sum = EF::from_base_fn(|d| sums[d].as_slice().iter().copied().sum());
        let sfx_sum = xs_sfx
            .iter()
            .zip(&self.powers[(xs_packed.len() * F::Packing::WIDTH)..])
            .map(|(&x, &pow)| pow * x)
            .sum::<EF>();
        packed_sum + sfx_sum
    }
}

fn transpose_vec<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::AbstractExtensionField;
    use rand::{thread_rng, Rng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_powers_reducer() {
        let mut rng = thread_rng();
        let alpha: EF = rng.gen();
        let n = 1000;
        let sizes = [5, 110, 512, 999, 1000];
        let r = PowersReducer::<F, EF>::new(alpha, n);

        // check reduce_ext
        for size in sizes {
            let xs: Vec<EF> = (0..size).map(|_| rng.gen()).collect();
            assert_eq!(
                r.reduce_ext(&xs),
                xs.iter()
                    .enumerate()
                    .map(|(i, &x)| alpha.exp_u64(i as u64) * x)
                    .sum()
            );
        }

        // check reduce_base
        for size in sizes {
            let xs: Vec<F> = (0..size).map(|_| rng.gen()).collect();
            assert_eq!(
                r.reduce_base(&xs),
                xs.iter()
                    .enumerate()
                    .map(|(i, &x)| alpha.exp_u64(i as u64) * EF::from_base(x))
                    .sum()
            );
        }

        // bench reduce_base
        /*
        use core::hint::black_box;
        use std::time::Instant;
        let samples = 1_000;
        for i in 0..5 {
            let xs: Vec<F> = (0..999).map(|_| rng.gen()).collect();
            let t0 = Instant::now();
            for _ in 0..samples {
                black_box(r.reduce_base_slow(black_box(&xs)));
            }
            let dt_slow = t0.elapsed();
            let t0 = Instant::now();
            for _ in 0..samples {
                black_box(r.reduce_base(black_box(&xs)));
            }
            let dt_fast = t0.elapsed();
            println!("sample {i}: slow: {dt_slow:?} fast: {dt_fast:?}");
        }
        */
    }
}
