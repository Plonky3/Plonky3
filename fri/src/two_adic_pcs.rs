//! The FRI PCS protocol over two-adic fields.
//!
//! The following implements a slight variant of the usual FRI protocol. As usual we start
//! with a polynomial `F(x)` of degree `n` given as evaluations over the coset `gH` with `|H| = 2^n`.
//!
//! Now consider the polynomial `G(x) = F(gx)`. Note that `G(x)` has the same degree as `F(x)` and
//! the evaluations of `F(x)` over `gH` are identical to the evaluations of `G(x)` over `H`.
//!
//! Hence we can reinterpret our vector of evaluations as evaluations of `G(x)` over `H` and apply
//! the standard FRI protocol to this evaluation vector. This makes it easier to apply FRI to a collection
//! of polynomials defined over different cosets as we don't need to keep track of the coset shifts. We
//! can just assume that every polynomial is defined over the subgroup of the relevant size.
//!
//! If we changed our domain construction (e.g., using multiple cosets), we would need to carefully reconsider these assumptions.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{
    CommitmentOpening, Mmcs, OpenedValues, OpeningRequest, Pcs, PeriodicLdeTable,
    UnivariateStarkPcs,
};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, PackedFieldExtension, PrimeField64, TwoAdicField, batch_multiplicative_inverse,
    dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixCow};
use p3_matrix::interpolation::{Interpolate, compute_adjusted_weights};
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};
use tracing::{debug_span, instrument};

use crate::periodic::build_periodic_lde_table_two_adic;
use crate::verifier::{self, FriError};
use crate::{
    BatchMultiOpening, FriFoldingStrategy, FriParameters, FriProof, PcsProverTranscript, PcsShape,
    PcsVerifierTranscript, prover,
};

/// A polynomial commitment scheme using FRI to generate opening proofs.
///
/// We commit to a polynomial `f` via its evaluation vectors over a coset
/// `gH` where `|H| >= 2 * deg(f)`. A value `f(z)` is opened by using a FRI
/// proof to show that the evaluations of `(f(x) - f(z))/(x - z)` over
/// `gH` are low degree.
#[derive(Clone, Debug)]
pub struct TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub(crate) dft: Dft,
    pub(crate) mmcs: InputMmcs,
    pub(crate) fri: FriParameters<FriMmcs>,
    /// Folding schedule to use instead of the derived one.
    ///
    /// Only a test sets this.
    /// It lets a test forge a schedule the verifier will not derive.
    /// Every commitment and opening then agrees with the forgery.
    #[cfg(test)]
    pub(crate) forged_fold_schedule: Option<Vec<usize>>,
    _phantom: PhantomData<Val>,
}

impl<Val, Dft, InputMmcs, FriMmcs> TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub const fn new(dft: Dft, mmcs: InputMmcs, fri: FriParameters<FriMmcs>) -> Self {
        Self {
            dft,
            mmcs,
            fri,
            #[cfg(test)]
            forged_fold_schedule: None,
            _phantom: PhantomData,
        }
    }
}

/// The Prover Data associated to a commitment to a collection of matrices
/// and a list of points to open each matrix at.
pub type ProverDataWithOpeningPoints<'a, EF, ProverData> = OpeningRequest<'a, ProverData, EF>;

// Re-exported so `p3_fri::CommitmentWithOpeningPoints` keeps naming the shape this PCS
// verifies against; it is defined in `p3-commit` alongside `Pcs` so crates that build an
// opening argument without depending on FRI can name it too.
pub use p3_commit::CommitmentWithOpeningPoints;

pub struct TwoAdicFriFolding<InputProof, InputError>(pub PhantomData<(InputProof, InputError)>);

pub type TwoAdicFriFoldingForMmcs<F, M> =
    TwoAdicFriFolding<Vec<BatchMultiOpening<F, M>>, <M as Mmcs<F>>::Error>;

impl<F: TwoAdicField, InputProof: Sync, InputError: Debug + Sync, EF: ExtensionField<F>>
    FriFoldingStrategy<F, EF> for TwoAdicFriFolding<InputProof, InputError>
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
        log_arity: usize,
        beta: EF,
        mut evals: impl Iterator<Item = EF>,
    ) -> EF {
        let arity = 1 << log_arity;

        // Compute the evaluation points in the subgroup
        let subgroup_start = F::two_adic_generator(log_height + log_arity)
            .exp_u64(reverse_bits_len(index, log_height) as u64);

        if log_arity == 1 {
            // The two interpolation points are `{s, -s}` for `s = subgroup_start`, so
            //     f(beta) = (y_0 + y_1) / 2 + (y_0 - y_1) * beta / (2 s).
            // This is the closed form used by the arity-2 kernel of `fold_matrix`.
            let (lo, hi) = evals.next_tuple().expect("Expected 2 evaluations");
            assert!(evals.next().is_none(), "Expected 2 evaluations");

            let halve_inv_power = subgroup_start.double().inverse();
            return (lo + hi).halve() + (lo - hi) * beta * halve_inv_power;
        }

        let evals: Vec<_> = evals.collect();
        assert_eq!(evals.len(), arity, "Expected {} evaluations", arity);

        let mut xs: Vec<F> = F::two_adic_generator(log_arity)
            .shifted_powers(subgroup_start)
            .take(arity)
            .collect();
        reverse_slice_index_bits(&mut xs);

        // Lagrange interpolation at beta
        lagrange_interpolate_at(&xs, &evals, beta)
    }

    #[instrument(skip_all, level = "debug")]
    fn fold_matrix<M: Matrix<EF>>(&self, beta: EF, log_arity: usize, m: M) -> Vec<EF> {
        if log_arity == 1 {
            // Optimized path for arity 2
            // We use the fact that
            //     p_e(x^2) = (p(x) + p(-x)) / 2
            //     p_o(x^2) = (p(x) - p(-x)) / (2 x)
            // that is,
            //     p_e(g^(2i)) = (p(g^i) + p(g^(n/2 + i))) / 2
            //     p_o(g^(2i)) = (p(g^i) - p(g^(n/2 + i))) / (2 g^i)
            // so
            //     result(g^(2i)) = p_e(g^(2i)) + beta p_o(g^(2i))
            //
            // As p_e, p_o will be in the extension field we want to find ways to avoid extension multiplications.
            // We should only need a single one (namely multiplication by beta).
            let g_inv = F::two_adic_generator(log2_strict_usize(m.height()) + 1).inverse();

            // As beta is in the extension field, we want to avoid multiplying by it
            // for as long as possible. Here we precompute the powers  `g_inv^i / 2` in the base field.
            let mut halve_inv_powers = g_inv.shifted_powers(F::ONE.halve()).collect_n(m.height());
            reverse_slice_index_bits(&mut halve_inv_powers);

            m.par_rows()
                .zip(halve_inv_powers)
                .map(|(mut row, halve_inv_power)| {
                    let (lo, hi) = row.next_tuple().unwrap();
                    (lo + hi).halve() + (lo - hi) * beta * halve_inv_power
                })
                .collect()
        } else {
            // Decompose arity-2^k fold into k sequential arity-2 folds.
            // This way, an arity-2^k fold with a single challenge beta is equivalent to
            // k arity-2 folds with challenges beta, beta^2, beta^4, ..., beta^{2^{k-1}}.
            //
            // For arity 4 with evaluation points {s, -s, si, -si}:
            //   Step 1 (beta):   fold pairs → g(s^2), g(-s^2) where g = f_e + beta*f_o
            //   Step 2 (beta^2): fold pair  → g(beta^2) = f(beta)

            // Every row splits into whole `(lo, hi)` pairs.
            debug_assert!(m.width().is_multiple_of(2));
            let pairs_per_row = m.width() / 2;
            let initial_height = m.height() * pairs_per_row;
            let g_inv = F::two_adic_generator(log2_strict_usize(initial_height) + 1).inverse();
            let mut halve_inv_powers = g_inv
                .shifted_powers(F::ONE.halve())
                .collect_n(initial_height);
            reverse_slice_index_bits(&mut halve_inv_powers);

            let two = F::ONE + F::ONE;

            // The first fold reads the borrowed matrix row by row, so `m` is never copied.
            let mut data = EF::zero_vec(initial_height);
            data.par_chunks_exact_mut(pairs_per_row)
                .zip(m.par_rows())
                .zip(halve_inv_powers.par_chunks_exact(pairs_per_row))
                .for_each(|((out, row), inv_powers)| {
                    for (o, (lo, hi), &halve_inv_power) in
                        izip!(out.iter_mut(), row.tuples::<(EF, EF)>(), inv_powers)
                    {
                        *o = (lo + hi).halve() + (lo - hi) * beta * halve_inv_power;
                    }
                });

            let mut current_beta = beta.square();
            let mut next_data = EF::zero_vec(initial_height / 2);

            for _ in 1..log_arity {
                let height = data.len() / 2;
                // Since j << 1 is always >= j, we never overwrite data we haven't read yet.
                for j in 0..height {
                    halve_inv_powers[j] = two * halve_inv_powers[j << 1].square();
                }
                next_data[..height]
                    .par_iter_mut()
                    .zip(data.par_chunks_exact(2))
                    .zip(&halve_inv_powers[..height])
                    .for_each(|((out, chunk), &halve_inv_power)| {
                        // chunk is guaranteed to be size 2 by par_chunks_exact
                        let lo = chunk[0];
                        let hi = chunk[1];

                        *out = (lo + hi).halve() + (lo - hi) * current_beta * halve_inv_power;
                    });
                current_beta = current_beta.square();

                // Ping-pong the two buffers; the tail of the new `data` is stale and dropped.
                core::mem::swap(&mut data, &mut next_data);
                data.truncate(height);
            }

            data
        }
    }
}

/// Lagrange interpolation: given points `(xs[i], ys[i])`, evaluate at z.
///
/// Uses the barycentric formula for efficiency when xs are roots of unity.
fn lagrange_interpolate_at<F: TwoAdicField, EF: ExtensionField<F>>(
    xs: &[F],
    ys: &[EF],
    z: EF,
) -> EF {
    debug_assert_eq!(xs.len(), ys.len());
    let n = xs.len();

    if n == 0 {
        return EF::ZERO;
    }

    // If z equals one of the interpolation points, return early.
    for i in 0..n {
        if (z - xs[i]).is_zero() {
            return ys[i];
        }
    }

    let log_n = log2_strict_usize(n);

    // All xs lie in a coset of the 2^log_n roots of unity.
    let coset_power = xs[0].exp_power_of_2(log_n);
    let weight_scale = (F::from_usize(n) * coset_power).inverse();

    // Compute (z - x_i)^{-1} as a batch inversion
    let diffs: Vec<_> = xs.iter().map(|&x| z - x).collect();
    let diff_invs = batch_multiplicative_inverse(&diffs);

    // Compute L(z) = prod_i (z - x_i)
    let l_z = diffs.iter().copied().product::<EF>();

    // Barycentric formula: sum_i (w_i * y_i / (z - x_i))
    // where w_i = 1 / prod_{j != i} (x_i - x_j) = x_i * weight_scale.
    let mut result = EF::ZERO;
    for ((&x, &y), &diff_inv) in xs.iter().zip(ys).zip(diff_invs.iter()) {
        let weight = x * weight_scale;
        result += y * weight * diff_inv;
    }
    result * l_z
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val, MultiProof: Sync, Error: Sync>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type Proof = FriProof<Challenge, FriMmcs, Val, Vec<BatchMultiOpening<Val, InputMmcs>>>;
    type Error = FriError<FriMmcs::Error, InputMmcs::Error>;

    /// Get the unique subgroup `H` of size `|H| = degree`.
    ///
    /// # Panics:
    /// This function will panic if `degree` is not a power of 2 or `degree > (1 << Val::TWO_ADICITY)`.
    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        TwoAdicMultiplicativeCoset::new(Val::ONE, log2_strict_usize(degree)).unwrap()
    }

    /// Commit to a collection of evaluation matrices.
    ///
    /// Each element of `evaluations` contains a coset `shift * H` and a matrix `mat` with `mat.height() = |H|`.
    /// Interpreting each column of `mat` as the evaluations of a polynomial `p_i(x)` over `shift * H`,
    /// this computes the evaluations of `p_i` over `gK` where `g` is the chosen generator of the multiplicative group
    /// of `Val` and `K` is the unique subgroup of order `|H| << self.fri.log_blowup`.
    ///
    /// This then outputs a Merkle commitment to these evaluations.
    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let ldes: Vec<_> = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                // coset_lde_batch converts from evaluations over `xH` to evaluations over `shift * x * K`.
                // Hence, letting `shift = g/x` the output will be evaluations over `gK` as desired.
                // When `x = g`, we could just use the standard LDE but currently this doesn't seem
                // to give a meaningful performance boost.
                let shift = Val::GENERATOR / domain.shift();
                // Compute the LDE with blowup factor fri.log_blowup.
                // We bit reverse as this is required by our implementation of the FRI protocol.
                self.dft
                    .coset_lde_batch(evals, self.fri.log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect();

        // Commit to the bit-reversed LDEs.
        self.mmcs.commit(ldes)
    }

    /// Open a batch of matrices at a collection of points.
    ///
    /// Returns the opened values along with a proof.
    ///
    /// This function assumes that all matrices correspond to evaluations over the
    /// coset `gH` where `g = Val::GENERATOR` and `H` is a subgroup of appropriate size depending on the
    /// matrix.
    fn open(
        &self,
        // For each multi-matrix commitment,
        commitment_data_with_opening_points: Vec<OpeningRequest<'_, Self::ProverData, Challenge>>,
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
            = ( a^0 x0 + a^1 x1 ) + a^2 ( a^0 x2 + a^1 x3 ) + ...
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

        // Contained in each `Self::ProverData` is a list of matrices which have been committed to.
        // We extract those matrices to be able to refer to them directly.
        let mats_and_points = commitment_data_with_opening_points
            .iter()
            .map(
                |OpeningRequest {
                     prover_data: data,
                     points,
                 }| {
                    let mats = self
                        .mmcs
                        .get_matrices(data)
                        .into_iter()
                        .map(|m| m.as_view())
                        .collect_vec();
                    debug_assert_eq!(
                        mats.len(),
                        points.len(),
                        "each matrix should have a corresponding set of evaluation points"
                    );
                    (mats, points)
                },
            )
            .collect_vec();

        // Find the maximum height and the maximum width of matrices in the batch.
        // These do not need to correspond to the same matrix.
        let (global_max_height, global_max_width) = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats.iter().map(|m| (m.height(), m.width())))
            .reduce(|(hmax, wmax), (h, w)| (hmax.max(h), wmax.max(w)))
            .expect("No Matrices Supplied?");
        let log_global_max_height = log2_strict_usize(global_max_height);

        // Get all values of the coset `gH` for the largest necessary subgroup `H`.
        // We also bit reverse which means that coset has the nice property that
        // `coset[..2^i]` contains the values of `gK` for `|K| = 2^i`.
        let coset = {
            let coset =
                TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_global_max_height).unwrap();
            let mut coset_points = coset.iter().collect();
            reverse_slice_index_bits(&mut coset_points);
            coset_points
        };

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(z - X) for the largest subgroup (in bitrev order).
        let inv_denoms = compute_inverse_denominators(&mats_and_points, &coset);

        // Precompute adjusted barycentric weights once per opening point.
        // adjusted[i] = 1/(z - x_i) - 1/z, reused across all matrices opened at z.
        let adjusted_weights: LinearMap<Challenge, Vec<Challenge>> = inv_denoms
            .iter()
            .map(|(point, denoms)| (*point, compute_adjusted_weights(*point, denoms)))
            .collect();

        // Evaluate coset representations and write openings to the challenger
        let all_opened_values = mats_and_points
            .iter()
            .map(|(mats, points)| {
                // For each collection of matrices
                izip!(mats.iter(), points.iter())
                    .map(|(mat, points_for_mat)| {
                        // Every committed matrix is assumed to be an LDE at `self.fri.log_blowup`;
                        // `commit`/`commit_ldes` enforce `height >= 1 << log_blowup`. A larger actual
                        // blowup is still sound, just slightly slower.
                        // Ideally, polynomials would be passed in with their blow-up factors known.

                        // The point of this correction is that each column of the matrix corresponds to a low degree polynomial.
                        // Hence we can save time by restricting the height of the matrix to be the minimal height which
                        // uniquely identifies the polynomial.
                        let h = mat.height() >> self.fri.log_blowup;

                        // `subgroup` and `mat` are both in bit-reversed order, so we can truncate.
                        let (low_coset, _) = mat.split_rows(h);

                        points_for_mat
                            .iter()
                            .map(|&point| {
                                let _guard =
                                    debug_span!("evaluate matrix", dims = %mat.dimensions())
                                        .entered();

                                // Use Barycentric interpolation to evaluate each column of the matrix at the given point.
                                debug_span!("compute opened values with Lagrange interpolation")
                                    .in_scope(|| {
                                        // Slice the precomputed adjusted weights to match this matrix's height.
                                        // Zero-allocation hot path: straight to the SIMD dot product.
                                        let adj = &adjusted_weights.get(&point).unwrap()[..h];
                                        low_coset.interpolate_coset_with_precomputation(
                                            Val::GENERATOR,
                                            point,
                                            adj,
                                        )
                                    })
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec();

        // Describe the transcript before running it.
        //
        // Every number comes from the parameters and the matrices this prover holds.
        // The verifier builds the identical description from the claims it is handed.
        let mut transcript = PcsProverTranscript::<Challenger, Val, Challenge>::new(
            challenger,
            PcsShape::from_opened_values(&self.fri, &all_opened_values),
        );

        // Bind every claimed evaluation, in commitment, matrix and point order.
        transcript.claimed_openings(&all_opened_values);

        // Grind, then draw the challenge that batches the claims just bound.
        //
        // Why: the witness commits the prover to those openings before it learns the challenge.
        // Hunting for a challenge that makes a false batched claim look low degree costs
        // `2^batch_proof_of_work_bits` work per candidate.
        //
        // Soundness error: at minimum `k/|EF|`, over the linear combination
        //
        //     sum_i alpha^i * (f_i(zeta) - f_i(x)) / (zeta - x)
        //
        // where `k` counts every (function, opening point) pair.
        // For a univariate STARK that is twice the trace width plus the quotient chunk count.
        //
        // This site's error grows with the batch, unlike the query phase's.
        // On a wide instance over a small field it is what limits proven soundness.
        let (alpha, batch_pow_witness) = transcript.batch_phase();
        // A zero difficulty still occupies a slot in the proof, keeping its shape fixed.
        let batch_pow_witness = batch_pow_witness.unwrap_or(Val::ZERO);

        // We precompute the packed powers of alpha as we need the same powers for each matrix.
        // The hot per-matrix reduction (`rowwise_packed_dot_product`) consumes these directly; the
        // per-opening combination below unpacks `alpha`'s powers lazily via `alpha.powers()`, so we
        // never materialize a full unpacked copy.
        let packed_alpha_powers =
            Challenge::ExtensionPacking::packed_ext_powers_capped(alpha, global_max_width)
                .collect_vec();

        // Now that we have sent the openings to the verifier, it remains to prove
        // that those openings are correct.

        // Given a low degree polynomial `f(x)` with claimed evaluation `f(zeta)`, we can check
        // that `f(zeta)` is correct by doing a low degree test on `(f(zeta) - f(x))/(zeta - x)`.
        // We will use `alpha` to batch together both different claimed openings `zeta` and
        // different polynomials `f` whose evaluation vectors have the same height.

        // TODO: If we allow different polynomials to have different blow_up factors
        // we may need to revisit this and to ensure it is safe to batch them together.

        // num_reduced records the number of (function, opening point) pairs for each `log_height`.
        // TODO: This should really be `[0; Val::TWO_ADICITY + 1]` but that runs into issues with generics.
        let mut num_reduced = [0; 33];

        // For each `log_height` from 2^1 -> 2^32, reduced_openings will contain either `None`
        // if there are no matrices of that height, or `Some(vec)` where `vec` is equal to
        // a weighted sum of `(f(zeta) - f(x))/(zeta - x)` over all `f`'s of that height and
        // for each `f`, all opening points `zeta`. The sum is weighted by powers of the challenge alpha.
        let mut reduced_openings: [_; 33] = core::array::from_fn(|_| None);

        for ((mats, points), openings_for_round) in
            mats_and_points.iter().zip(all_opened_values.iter())
        {
            for (mat, points_for_mat, openings_for_mat) in
                izip!(mats.iter(), points.iter(), openings_for_round.iter())
            {
                let _guard =
                    debug_span!("reduce matrix quotient", dims = %mat.dimensions()).entered();

                let log_height = log2_strict_usize(mat.height());

                // If this is our first matrix at this height, initialise reduced_openings to zero.
                // Otherwise, get a mutable reference to it.
                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| Challenge::zero_vec(mat.height()));
                debug_assert_eq!(reduced_opening_for_log_height.len(), mat.height());

                // Treating our matrix M as the evaluations of functions f_0, f_1, ...
                // Compute the evaluations of `Mred(x) = f_0(x) + alpha*f_1(x) + ...`
                let mat_compressed = debug_span!("compress mat").in_scope(|| {
                    // This will be reused for all points z which M is opened at so we collect into a vector.
                    mat.rowwise_packed_dot_product::<Challenge>(&packed_alpha_powers)
                        .collect::<Vec<_>>()
                });

                for (&point, openings) in points_for_mat.iter().zip(openings_for_mat) {
                    // If we have multiple matrices at the same height, we need to scale alpha to combine them.
                    // This means that reduced_openings will contain:
                    // Mred_0(x) + alpha^{M_0.width()}Mred_1(x) + alpha^{M_0.width() + M_1.width()}Mred_2(x) + ...
                    // Where M_0, M_1, ... are the matrices of the same height.
                    let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);

                    // As we have all the openings `f_i(z)`, we can combine them using `alpha`
                    // in an identical way to before to compute `Mred(z)`.
                    let reduced_openings: Challenge =
                        dot_product(alpha.powers(), openings.iter().copied());

                    mat_compressed
                        .par_iter()
                        .zip(reduced_opening_for_log_height.par_iter_mut())
                        // inv_denoms contains `1/(z - x)` for `x` in a coset `gK`.
                        // If `|K| =/= mat.height()` we actually want a subset of this
                        // corresponding to the evaluations over `gH` for `|H| = mat.height()`.
                        // As inv_denoms is bit reversed, the evaluations over `gH` are exactly
                        // the evaluations over `gK` at the indices `0..mat.height()`.
                        // So zip will truncate to the desired smaller length.
                        .zip(inv_denoms.get(&point).unwrap().par_iter())
                        // Map the function `Mred(x) -> (Mred(z) - Mred(x))/(z - x)`
                        // across the evaluation vector of `Mred(x)`. Adjust by alpha_pow_offset
                        // as needed.
                        .for_each(|((&reduced_row, ro), &inv_denom)| {
                            *ro += alpha_pow_offset * (reduced_openings - reduced_row) * inv_denom;
                        });
                    num_reduced[log_height] += mat.width();
                }
            }
        }

        // It remains to prove that all evaluation vectors in reduced_openings correspond to
        // low degree functions.
        let fri_input = reduced_openings.into_iter().rev().flatten().collect_vec();

        let folding: TwoAdicFriFoldingForMmcs<Val, InputMmcs> = TwoAdicFriFolding(PhantomData);

        // Produce the FRI proof, bracketed as a sub-protocol of this transcript.
        //
        // FRI seeds its own description from the sponge state reached here.
        let fri_proof = transcript.delegate(|challenger| {
            #[cfg(test)]
            {
                prover::prove_fri_with_schedule(
                    &folding,
                    &self.fri,
                    fri_input,
                    challenger,
                    log_global_max_height,
                    &commitment_data_with_opening_points,
                    &self.mmcs,
                    batch_pow_witness,
                    self.forged_fold_schedule.clone(),
                )
            }
            #[cfg(not(test))]
            {
                prover::prove_fri(
                    &folding,
                    &self.fri,
                    fri_input,
                    challenger,
                    log_global_max_height,
                    &commitment_data_with_opening_points,
                    &self.mmcs,
                    batch_pow_witness,
                )
            }
        });

        // Every described step has now been played.
        transcript.finish();

        (all_opened_values, fri_proof)
    }

    fn verify(
        &self,
        // For each commitment:
        commitments_with_opening_points: Vec<
            CommitmentOpening<Challenge, Self::Commitment, Self::Domain>,
        >,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // A zero grinding budget leaves the witness unread, so every field is pinned here
        // rather than by its grind.
        //
        //     bits = 0 -> prover emits zero, verifier reads nothing -> pin the field here
        //     bits > 0 -> prover grinds,     verifier resamples     -> the grind pins it
        //
        // The FRI verifier repeats the check for the two phases it owns.
        verifier::check_canonical_pow_witnesses(&self.fri, proof)?;

        // Describe the transcript from the claims, which are this verifier's own input.
        //
        // The prover built the identical description from the matrices it opened.
        let mut transcript = PcsVerifierTranscript::<Challenger, Val, Challenge>::new(
            challenger,
            PcsShape::from_claims(&self.fri, &commitments_with_opening_points),
        );

        // Replay every claimed evaluation, in commitment, matrix and point order.
        //
        // A failure poisons the driver on its way out, so dropping it here is silent.
        transcript.claimed_openings(&commitments_with_opening_points)?;

        // Replay the grind and redraw the challenge that batches the claims.
        let alpha = transcript.batch_phase(Some(proof.batch_pow_witness))?;

        let folding: TwoAdicFriFoldingForMmcs<Val, InputMmcs> = TwoAdicFriFolding(PhantomData);

        // Run the low-degree test inside the bracket, lending it the sponge.
        //
        // Invariant: the bracket closes whichever way the delegated run goes.
        //
        //     delegate  -> Begin, run, End      (End is recorded on any outcome)
        //     finish    -> every described step replayed
        //     rejection -> propagated afterwards, never across an unfinished driver
        let result = transcript.delegate(|challenger| {
            verifier::verify_fri(
                &folding,
                &self.fri,
                proof,
                challenger,
                &commitments_with_opening_points,
                &self.mmcs,
                alpha,
            )
        });

        // Every described step has now been replayed.
        transcript.finish();

        result
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> UnivariateStarkPcs<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val, MultiProof: Sync, Error: Sync>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixCow<'a, Val>>;

    const ZK: bool = false;

    fn log_max_lde_height(&self) -> usize {
        Val::TWO_ADICITY
    }

    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
        _num_chunks: usize,
    ) -> Vec<RowMajorMatrix<Val>> {
        evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                // coset_lde_batch converts from evaluations over `xH` to evaluations over `shift * x * K`.
                // Hence, letting `shift = g/x` the output will be evaluations over `gK` as desired.
                // When `x = g`, we could just use the standard LDE but currently this doesn't seem
                // to give a meaningful performance boost.
                let shift = Val::GENERATOR / domain.shift();
                // Compute the LDE with blowup factor fri.log_blowup.
                // We bit reverse as this is required by our implementation of the FRI protocol.
                self.dft
                    .coset_lde_batch(evals, self.fri.log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect()
    }

    fn commit_ldes(&self, ldes: Vec<RowMajorMatrix<Val>>) -> (Self::Commitment, Self::ProverData) {
        // Opening assumes every committed matrix is an LDE at `self.fri.log_blowup` and recovers the
        // underlying polynomial degree as `height >> log_blowup`. A matrix shorter than the blowup
        // factor would silently yield a zero-height degree and a malformed proof, so reject it here.
        let min_height = 1 << self.fri.log_blowup;
        for lde in &ldes {
            assert!(
                lde.height() >= min_height,
                "committed LDE height {} is smaller than the blowup factor {min_height}",
                lde.height()
            );
        }
        self.mmcs.commit(ldes)
    }

    /// Given the evaluations on a domain `gH`, return the evaluations on a different domain `g'K`.
    ///
    /// Arguments:
    /// - `prover_data`: The prover data containing all committed evaluation matrices.
    /// - `idx`: The index of the matrix containing the evaluations we want. These evaluations
    ///   are assumed to be over the coset `gH` where `g = Val::GENERATOR`.
    /// - `domain`: The domain `g'K` on which to get evaluations on.
    ///
    /// When `g' = g` (i.e. `Val::GENERATOR`) and `K` is a subgroup of `H`, this is a simple
    /// truncation of the bit-reversed LDE. Otherwise, we recover the polynomial coefficients
    /// from the committed LDE and re-evaluate on the requested domain.
    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let lde = self.mmcs.get_matrices(prover_data)[idx];
        if domain.shift() == Val::GENERATOR && lde.height() >= domain.size() {
            return lde.split_rows(domain.size()).0.as_cow().bit_reverse_rows();
        }

        // The committed LDE contains bit-reversed evaluations over `gH`.
        // Un-bit-reverse, coset iDFT to recover coefficients, truncate to
        // the original polynomial degree, then coset DFT onto the target domain.
        let poly_height = lde.height() >> self.fri.log_blowup;
        let lde_mat = lde.as_view().bit_reverse_rows().to_row_major_matrix();
        let mut coeffs = self.dft.coset_idft_batch(lde_mat, Val::GENERATOR);
        let width = coeffs.width();
        coeffs.values.truncate(poly_height * width);
        coeffs.values.resize(domain.size() * width, Val::ZERO);
        let result = self
            .dft
            .coset_dft_batch(coeffs, domain.shift())
            .bit_reverse_rows()
            .to_row_major_matrix();
        let result_width = result.width();

        RowMajorMatrixCow::new(Cow::Owned(result.values), result_width).bit_reverse_rows()
    }

    fn build_periodic_lde_table(
        &self,
        periodic_cols: &[Vec<Val>],
        trace_domain: Self::Domain,
        quotient_domain: Self::Domain,
    ) -> PeriodicLdeTable<Val> {
        build_periodic_lde_table_two_adic(&self.dft, periodic_cols, &trace_domain, &quotient_domain)
    }
}

/// Compute vectors of inverse denominators for each unique opening point.
///
/// Arguments:
/// - `mats_and_points` is a list of matrices and for each matrix a list of points. We assume that
///    the total number of distinct points is very small as several methods contained herein are `O(n^2)`
///    in the number of points.
/// - `coset` is the set of points `gH` where `H` a two-adic subgroup such that `|H|` is greater
///     than or equal to the largest height of any matrix in `mats_and_points`. The values
///     in `coset` must be in bit-reversed order.
///
/// For each point `z`, let `M` be the matrix of largest height which opens at `z`.
/// let `H_z` be the unique subgroup of order `M.height()`. Compute the vector of
/// `1/(z - x)` for `x` in `gH_z`.
///
/// Return a LinearMap which allows us to recover the computed vectors for each `z`.
#[instrument(skip_all)]
fn compute_inverse_denominators<F: TwoAdicField, EF: ExtensionField<F>, M: Matrix<F>>(
    mats_and_points: &[(Vec<M>, &Vec<Vec<EF>>)],
    coset: &[F],
) -> LinearMap<EF, Vec<EF>> {
    // For each `z`, find the maximal height of any matrix which we need to
    // open at `z`.
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

    // Compute the inverse denominators for each point `z`.
    max_log_height_for_point
        .into_iter()
        .map(|(z, log_height)| {
            (
                z,
                batch_multiplicative_inverse(
                    // As coset is stored in bit-reversed order,
                    // we can just take the first `2^log_height` elements.
                    &coset[..(1 << log_height)]
                        .iter()
                        .map(|&x| z - x)
                        .collect_vec(),
                ),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2Dit;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::verifier::PowPhase;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;
    type MyPcs = TwoAdicFriPcs<F, Radix2Dit<F>, ValMmcs, ChallengeMmcs>;
    type Commitment = <ValMmcs as Mmcs<F>>::Commitment;
    type Domain = TwoAdicMultiplicativeCoset<F>;
    type Claims = Vec<CommitmentWithOpeningPoints<EF, Commitment, Domain>>;
    type MyProof = <MyPcs as Pcs<EF, Challenger>>::Proof;
    type TestError = FriError<<ChallengeMmcs as Mmcs<EF>>::Error, <ValMmcs as Mmcs<F>>::Error>;

    /// Grinding difficulty the roundtrip fixture proves at.
    ///
    /// Positive so the witness is genuinely absorbed.
    /// A zero-bit check short-circuits without reading the witness at all.
    const BATCH_POW_BITS: usize = 1;

    /// Run a real prover roundtrip and return everything a verifier needs.
    ///
    /// One commitment, two matrices, one opening point each.
    ///
    /// The claimed openings therefore nest as:
    ///
    /// ```text
    ///     claims           = [commitment_0]         // 1 commitment
    ///     claims[0].1      = [matrix_0, matrix_1]   // 2 matrices
    ///     claims[0].1[i].1 = [point_0]              // 1 point each
    /// ```
    ///
    /// The challenger is advanced past the commitment, ready for `Pcs::verify`.
    fn make_pcs_fixture() -> (MyPcs, Claims, MyProof, Challenger) {
        make_pcs_fixture_at(BATCH_POW_BITS)
    }

    /// The same roundtrip at a caller-chosen batch difficulty.
    ///
    /// The difficulty reaches the transcript's description, so prover and verifier must
    /// be built from the same number for the replay to line up.
    fn make_pcs_fixture_at(batch_pow_bits: usize) -> (MyPcs, Claims, MyProof, Challenger) {
        // Fixed seed keeps the roundtrip deterministic.
        let mut rng = SmallRng::seed_from_u64(42);

        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());

        let val_mmcs = ValMmcs::new(hash.clone(), compress.clone(), 0);
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress, 0));

        // Minimal sound parameters: blowup 2, binary folding, 2 queries.
        let fri_params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            batch_proof_of_work_bits: batch_pow_bits,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: challenge_mmcs,
        };
        let pcs = MyPcs::new(Radix2Dit::default(), val_mmcs, fri_params);

        // Two matrices of different widths, so moving a value between them is observable.
        let log_degree = 3;
        let domain =
            <MyPcs as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_degree);
        let traces = [2, 3].map(|width| {
            (
                domain,
                RowMajorMatrix::<F>::rand_nonzero(&mut rng, 1 << log_degree, width),
            )
        });
        let (commitment, prover_data) = <MyPcs as Pcs<EF, Challenger>>::commit(&pcs, traces);

        // Prover: observe the commitment, sample the point, open.
        let mut p_challenger = Challenger::new(perm.clone());
        p_challenger.observe(&commitment);
        let zeta: EF = p_challenger.sample_algebra_element();
        let (opened_values, proof) = pcs.open(
            vec![(&prover_data, vec![vec![zeta], vec![zeta]]).into()],
            &mut p_challenger,
        );

        // Verifier: replay up to the point sample so a valid proof must pass.
        let mut v_challenger = Challenger::new(perm);
        v_challenger.observe(&commitment);
        let v_zeta: EF = v_challenger.sample_algebra_element();
        assert_eq!(
            v_zeta, zeta,
            "prover and verifier must sample the same point"
        );

        let claims = vec![
            (
                commitment,
                opened_values[0]
                    .iter()
                    .map(|matrix| (domain, vec![(zeta, matrix[0].clone())]))
                    .collect(),
            )
                .into(),
        ];

        (pcs, claims, proof, v_challenger)
    }

    /// Verify with fully qualified syntax so the type parameters are unambiguous.
    fn run_pcs_verify(
        pcs: &MyPcs,
        claims: Claims,
        proof: &MyProof,
        challenger: &mut Challenger,
    ) -> Result<(), TestError> {
        <MyPcs as Pcs<EF, Challenger>>::verify(pcs, claims, proof, challenger)
    }

    #[test]
    fn a_valid_opening_proof_verifies() {
        // Baseline: the mutation tests below start from a genuinely valid proof.
        let (pcs, claims, proof, mut challenger) = make_pcs_fixture();

        run_pcs_verify(&pcs, claims, &proof, &mut challenger)
            .expect("an untouched opening proof must verify");
    }

    #[test]
    fn a_perturbed_claimed_evaluation_is_rejected() {
        // Every claimed evaluation is absorbed before `alpha` is drawn.
        //
        // Moving one moves the seed of the low-degree test that follows.
        // The reduced opening the verifier rebuilds then misses the committed codeword.
        let (pcs, mut claims, proof, mut challenger) = make_pcs_fixture();
        claims[0].matrices[0].points[0].values[0] += EF::ONE;

        run_pcs_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("a perturbed claimed evaluation must be rejected");
    }

    #[test]
    fn a_claimed_evaluation_moved_between_matrices_is_rejected() {
        // Invariant: the transcript sees one step per opening, of that opening's width.
        //
        //     described:  [2, 3]
        //     supplied:   [1, 4]   -> a different seed and a different step sequence
        //
        // Both sides derive that description from their own inputs, so the two disagree.
        let (pcs, mut claims, proof, mut challenger) = make_pcs_fixture();
        let moved = claims[0].matrices[0].points[0].values.pop().unwrap();
        claims[0].matrices[1].points[0].values.push(moved);

        run_pcs_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("a reshaped set of claims must be rejected");
    }

    #[test]
    fn a_tampered_batch_grinding_witness_is_rejected() {
        // Invariant: the grind is binding, not decorative.
        //
        // The witness must satisfy the proof-of-work predicate at the difficulty
        // the run was described with, even though the proof was produced at it.
        let (pcs, claims, mut proof, mut challenger) = make_pcs_fixture();
        // Fixed invalid candidate for this transcript seed. Adding one to a valid
        // 1-bit witness can produce another valid witness.
        proof.batch_pow_witness = F::TWO;

        let err = run_pcs_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("a tampered batch grinding witness must be rejected");

        match err {
            FriError::InvalidPowWitness(PowPhase::Batch) => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn a_batch_grinding_witness_below_the_configured_difficulty_is_rejected() {
        // A verifier demanding more work than the prover did must reject, and must
        // name the batch phase rather than one of the two witnesses FRI also carries.
        let (mut pcs, claims, proof, mut challenger) = make_pcs_fixture();
        pcs.fri.batch_proof_of_work_bits = 20;

        let err = run_pcs_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("a batch grinding witness below the difficulty must be rejected");

        match err {
            FriError::InvalidPowWitness(PowPhase::Batch) => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn a_noncanonical_batch_grinding_witness_at_zero_difficulty_is_rejected() {
        // Invariant: at zero difficulty only a canonical-value check binds the witness.
        //
        //     bits = 0 -> check_witness returns true, the step is elided -> unbound
        //     bits > 0 -> absorbed, bits resampled                       -> grind binds
        let (pcs, claims, mut proof, mut challenger) = make_pcs_fixture_at(0);

        // The honest prover writes zero when it pays no work.
        assert_eq!(proof.batch_pow_witness, F::ZERO);
        proof.batch_pow_witness = F::ONE;

        let err = run_pcs_verify(&pcs, claims, &proof, &mut challenger)
            .expect_err("a rewritten batch grinding witness must be rejected");

        match err {
            FriError::NonCanonicalPowWitness {
                phase: PowPhase::Batch,
            } => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    /// `fold_row`'s arity-2 closed form must agree with the generic barycentric path.
    #[test]
    fn fold_row_arity_two_matches_lagrange_interpolation() {
        let mut rng = SmallRng::seed_from_u64(0);
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);

        for log_height in 0..6 {
            for index in 0..(1 << log_height) {
                let beta: EF = rng.random();
                let evals: [EF; 2] = [rng.random(), rng.random()];

                // The interpolation points the generic path would build for this row.
                let subgroup_start = F::two_adic_generator(log_height + 1)
                    .exp_u64(reverse_bits_len(index, log_height) as u64);
                let xs = [subgroup_start, -subgroup_start];
                let expected = lagrange_interpolate_at(&xs, &evals, beta);

                let folded = FriFoldingStrategy::<F, EF>::fold_row(
                    &folding,
                    index,
                    log_height,
                    1,
                    beta,
                    evals.into_iter(),
                );
                assert_eq!(folded, expected);
            }
        }
    }

    /// `fold_matrix` folds row `i` of its input exactly as `fold_row` folds that row on its own.
    #[test]
    fn fold_matrix_matches_fold_row() {
        let mut rng = SmallRng::seed_from_u64(1);
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);

        for log_arity in 1..4 {
            for log_height in 0..5 {
                let beta: EF = rng.random();
                let m = RowMajorMatrix::<EF>::rand(&mut rng, 1 << log_height, 1 << log_arity);

                let folded = FriFoldingStrategy::<F, EF>::fold_matrix(
                    &folding,
                    beta,
                    log_arity,
                    m.as_view(),
                );
                assert_eq!(folded.len(), m.height());

                for (index, &expected) in folded.iter().enumerate() {
                    let row = m.row(index).unwrap().into_iter();
                    let folded_row = FriFoldingStrategy::<F, EF>::fold_row(
                        &folding, index, log_height, log_arity, beta, row,
                    );
                    assert_eq!(folded_row, expected);
                }
            }
        }
    }
}
