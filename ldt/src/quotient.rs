use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::{debug_assert_eq, slice};

use itertools::{izip, Itertools};
use p3_commit::Mmcs;
use p3_field::extension::HasFrobenius;
use p3_field::{
    add_vecs, batch_multiplicative_inverse, binomial_expand, cyclic_subgroup_coset_known_order,
    eval_poly, scale_vec, Field, PackedField, TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix, MatrixRowSlices, MatrixRows};
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};

/// A wrapper around an Inner MMCS, which transforms each inner value to (inner - r(X)) / m(X),
/// where m(X) is the minimal polynomial of the opening point, and r(X) = inner mod m(X).
///
/// This technique was proposed by Liam Eagen.
/// Instead of providing the quotient (p(X) - p(alpha))/(X - alpha) in the extension field,
/// we express the value of p(X) at X=alpha as the remainder r(X) = p(X) mod m(X),
/// where m(X) is the minimal polynomial such that m(alpha) = 0,
/// and prove r(X) is correct by showing (p(X) - r(X)) is divisible by m(X).
///
/// This has the benefit that all coefficients and evaluations are performed in the base field.
///
/// Since we have the values p(alpha) = y, we can recover r(X) by interpolating
/// [(alpha,y), (Frob alpha, Frob y), (Frob^2 alpha, Frob^2 y), ..]
/// since the Galois action commutes with polynomials with coefficients over the base field.
///
/// Since there can be multiple opening points, for each matrix, this transforms an inner opened row
/// into a concatenation of rows, transformed as above, for each point.
#[derive(Clone)]
pub struct QuotientMmcs<F: Field, EF, Inner: Mmcs<F>> {
    pub(crate) inner: Inner,

    /// For each matrix, a list of claimed openings, one for each point that we open that batch of
    /// polynomials at.
    pub(crate) openings: Vec<Vec<Opening<F, EF>>>,

    // The coset shift for the inner MMCS's evals, to correct `x` in the denominator.
    pub(crate) coset_shift: F,

    // QuotientMmcs and Opening, once constructed, technically do not need to know
    // anything about the extension field. However, we keep it as a generic so that
    // we can unroll the inner loop of `compute_quotient_matrix_row` over EF::D.
    pub(crate) _phantom: PhantomData<EF>,
}

/// A claimed opening.
#[derive(Clone, Debug)]
pub(crate) struct Opening<F: Field, EF> {
    // point.minimal_poly()
    pub(crate) minpoly: Vec<F>,
    // for each column, the remainder poly r(X) = p(X) mod m(X)
    pub(crate) remainder_polys: Vec<Vec<F>>,

    // each remainder poly always has degree EF::D.
    // so, the width of this matrix is EF::D, and the height is
    // `openings.len() // F::Packing::WIDTH`.
    // this matrix is missing any remaining coefficients that don't divide
    // F::Packing::WIDTH evenly, so you have to get those from remainder_polys.
    pub(crate) r_vertically_packed: Option<RowMajorMatrix<F::Packing>>,

    pub(crate) _phantom: PhantomData<EF>,
}

impl<F: Field, EF: HasFrobenius<F>> Opening<F, EF> {
    pub(crate) fn new(point: EF, values: Vec<EF>) -> Self {
        let remainder_polys = Self::compute_remainder_polys(point, &values);
        let r_vertically_packed = vertical_pack(&remainder_polys);
        Self {
            minpoly: point.minimal_poly(),
            remainder_polys,
            r_vertically_packed,
            _phantom: PhantomData,
        }
    }
    fn compute_remainder_polys(point: EF, values: &[EF]) -> Vec<Vec<F>> {
        // compute lagrange basis for [point, Frob point, Frob^2 point, ..]
        let xs = point.galois_group();
        debug_assert_eq!(xs.len(), EF::D);
        let w = xs[1..]
            .iter()
            .map(|&xi| xs[0] - xi)
            .product::<EF>()
            .inverse();
        let l_point = scale_vec(w, binomial_expand(&xs[1..]));
        debug_assert_eq!(l_point.len(), EF::D);
        // interpolate at [(pt,value),(Frob pt, Frob alpha),..]
        let mut rs = vec![];
        for &v in values {
            let mut l_point_frob = scale_vec(v, l_point.clone());
            let mut r = l_point_frob.clone();
            for _ in 1..EF::D {
                l_point_frob.iter_mut().for_each(|c| *c = c.frobenius());
                r = add_vecs(r, l_point_frob.clone());
            }
            rs.push(
                r.into_iter()
                    .map(|c| c.as_base().expect("Extension is not algebraic?"))
                    .collect(),
            );
        }
        rs
    }
}

/// For input `[
///   [ 1, 2, 3], \ pack
///   [ 4, 5, 6], /
///   [ 7, 8, 9], \ pack
///   [10,11,12], /
///   [13,14,15],
/// ]`,
/// and F::Packing::WIDTH = 2, returns `[
///   [P(1, 4), P(2, 5), P(3, 6)],
///   [P(7,10), P(8,11), P(9,12)],
/// ]`
/// where P(..) is a packed field. Trailing values (`[13,14,15]` above) are ignored.
fn vertical_pack<F: Field>(polys: &[Vec<F>]) -> Option<RowMajorMatrix<F::Packing>> {
    let width = polys[0].len();
    let height = polys.len() / F::Packing::WIDTH;
    if height == 0 {
        return None;
    }
    Some(RowMajorMatrix::new(
        (0..height)
            .flat_map(|r| {
                (0..width)
                    .map(move |c| F::Packing::from_fn(move |i| polys[r * F::Packing::WIDTH + i][c]))
            })
            .collect(),
        width,
    ))
}

#[derive(Debug, PartialEq, Eq)]
pub enum QuotientError<InnerMmcsError> {
    InnerMmcs(InnerMmcsError),
    OriginalValueMismatch,
}

impl<F, EF, Inner> Mmcs<F> for QuotientMmcs<F, EF, Inner>
where
    F: TwoAdicField,
    EF: HasFrobenius<F>,
    Inner: Mmcs<F>,
    for<'a> Inner::Mat<'a>: MatrixRowSlices<F>,
{
    type ProverData = Inner::ProverData;
    type Commitment = Inner::Commitment;
    type Proof = Inner::Proof;
    type Error = QuotientError<Inner::Error>;
    type Mat<'a> = QuotientMatrix<F, EF, Inner::Mat<'a>> where Self: 'a;

    fn open_batch(
        &self,
        index: usize,
        prover_data: &Self::ProverData,
    ) -> (Vec<Vec<F>>, Self::Proof) {
        let (inner_values, proof) = self.inner.open_batch(index, prover_data);
        let matrix_heights = self.inner.get_matrix_heights(prover_data);
        let max_height = *matrix_heights.iter().max().unwrap();
        let log_max_height = log2_strict_usize(max_height);

        let quotients = izip!(inner_values, &self.openings, matrix_heights)
            .map(|(inner_row, openings_for_mat, height)| {
                let log_height = log2_strict_usize(height);
                let bits_reduced = log_max_height - log_height;
                let reduced_index = reverse_bits_len(index >> bits_reduced, log_height);
                let x = self.coset_shift
                    * F::two_adic_generator(log_height).exp_u64(reduced_index as u64);

                let m_invs = batch_multiplicative_inverse(
                    &openings_for_mat
                        .iter()
                        .map(|opening| eval_poly(&opening.minpoly, x))
                        .collect_vec(),
                );
                compute_quotient_matrix_row(x, openings_for_mat, &m_invs, &inner_row)
            })
            .collect();

        (quotients, proof)
    }

    fn get_matrices<'a>(&'a self, prover_data: &'a Self::ProverData) -> Vec<Self::Mat<'a>> {
        self.inner
            .get_matrices(prover_data)
            .into_iter()
            .zip(self.openings.clone())
            .map(|(inner, openings)| {
                let height = inner.height();
                let log_height = log2_strict_usize(height);
                let g = F::two_adic_generator(log_height);
                let mut subgroup =
                    cyclic_subgroup_coset_known_order(g, self.coset_shift, height).collect_vec();
                reverse_slice_index_bits(&mut subgroup);

                let denominators: Vec<F> = subgroup
                    .iter()
                    .flat_map(|&x| {
                        openings
                            .iter()
                            .map(move |opening| eval_poly(&opening.minpoly, x))
                    })
                    .collect();
                let inv_denominators = RowMajorMatrix::new(
                    batch_multiplicative_inverse(&denominators),
                    openings.len(),
                );

                QuotientMatrix {
                    inner,
                    subgroup,
                    openings,
                    inv_denominators,
                    _phantom: PhantomData,
                }
            })
            .collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        opened_quotient_values: &[Vec<F>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        // quotient = (original - r(X))/m(X)
        // original = quotient * m(X) + r(X)

        let log_max_height = dimensions
            .iter()
            .map(|dims| log2_strict_usize(dims.height))
            .max()
            .unwrap();

        let opened_original_values = izip!(opened_quotient_values, &self.openings, dimensions)
            .map(|(quotient_row, openings, dims)| {
                let log_height = log2_strict_usize(dims.height);
                let bits_reduced = log_max_height - log_height;
                let reduced_index = reverse_bits_len(index >> bits_reduced, log_height);
                let x = self.coset_shift
                    * F::two_adic_generator(log_height).exp_u64(reduced_index as u64);

                let original_width = quotient_row.len() / openings.len();
                let original_row_repeated: Vec<Vec<F>> = quotient_row
                    .chunks(original_width)
                    .zip(openings)
                    .map(|(quotient_row_chunk, opening)| {
                        quotient_row_chunk
                            .iter()
                            .zip(&opening.remainder_polys)
                            .map(|(&quotient_value, r)| {
                                quotient_value * eval_poly(&opening.minpoly, x) + eval_poly(r, x)
                            })
                            .collect_vec()
                    })
                    .collect_vec();
                get_repeated(original_row_repeated.into_iter())
                    .ok_or(QuotientError::OriginalValueMismatch)
            })
            .collect::<Result<Vec<_>, Self::Error>>()?;

        self.inner
            .verify_batch(commit, dimensions, index, &opened_original_values, proof)
            .map_err(QuotientError::InnerMmcs)
    }
}

#[derive(Clone)]
pub struct QuotientMatrix<F: Field, EF, Inner: MatrixRowSlices<F>> {
    inner: Inner,
    subgroup: Vec<F>,
    openings: Vec<Opening<F, EF>>,
    /// For each row (associated with a subgroup element `x`), for each opening point,
    /// this holds `1 / m(X)`.
    inv_denominators: RowMajorMatrix<F>,
    _phantom: PhantomData<F>,
}

impl<F: Field, EF, Inner: MatrixRowSlices<F>> Matrix<F> for QuotientMatrix<F, EF, Inner> {
    fn width(&self) -> usize {
        self.inner.width() * self.openings.len()
    }

    fn height(&self) -> usize {
        self.inner.height()
    }
}

impl<F: Field, EF: HasFrobenius<F>, Inner: MatrixRowSlices<F>> MatrixRows<F>
    for QuotientMatrix<F, EF, Inner>
{
    type Row<'a> = Vec<F> where Inner: 'a;

    #[inline]
    fn row(&self, r: usize) -> Self::Row<'_> {
        compute_quotient_matrix_row(
            self.subgroup[r],
            &self.openings,
            self.inv_denominators.row_slice(r),
            self.inner.row_slice(r),
        )
    }

    #[inline]
    fn row_vec(&self, r: usize) -> Vec<F> {
        self.row(r)
    }
}

fn compute_quotient_matrix_row<F: Field, EF: HasFrobenius<F>>(
    x: F,
    openings: &[Opening<F, EF>],
    m_invs: &[F],
    inner_row: &[F],
) -> Vec<F> {
    let mut qp_ys: Vec<F> = Vec::with_capacity(inner_row.len() * openings.len());

    // [P(1,1,1,1), P(x,x,x,x),P(x^2,x^2,x^2,x^2),..]
    let packed_x_pows = x
        .powers()
        .take(EF::D)
        .map(|x_pow| F::Packing::from(x_pow))
        .collect_vec();

    for (opening, &m_inv) in openings.iter().zip(m_invs) {
        let packed_m_inv = F::Packing::from(m_inv);
        let (packed_ys, sfx_ys) = F::Packing::pack_slice_with_suffix(inner_row);

        if let Some(r_vertically_packed) = &opening.r_vertically_packed {
            let uninit = qp_ys.spare_capacity_mut();
            assert!(uninit.len() >= packed_ys.len() * F::Packing::WIDTH);
            let packed_uninit = unsafe {
                slice::from_raw_parts_mut(
                    uninit.as_mut_ptr().cast::<MaybeUninit<F::Packing>>(),
                    packed_ys.len(),
                )
            };
            for (packed_qp_y, &packed_y, coeffs) in
                izip!(packed_uninit, packed_ys, r_vertically_packed.rows())
            {
                let mut r_at_x = coeffs[0];
                for i in 1..EF::D {
                    r_at_x += coeffs[i] * packed_x_pows[i];
                }
                let qp_y = (packed_y - r_at_x) * packed_m_inv;
                packed_qp_y.write(qp_y);
            }
            unsafe {
                qp_ys.set_len(qp_ys.len() + packed_ys.len() * F::Packing::WIDTH);
            }
        }

        sfx_ys
            .iter()
            .zip(&opening.remainder_polys[opening.remainder_polys.len() - sfx_ys.len()..])
            .for_each(|(&y, r)| qp_ys.push((y - eval_poly(r, x)) * m_inv));
    }

    qp_ys
}

/// Checks that the given iterator contains repetitions of a single item, and return that item.
fn get_repeated<T: Eq + Debug, I: Iterator<Item = T>>(mut iter: I) -> Option<T> {
    let first = iter.next().expect("get_repeated on empty iterator");
    for x in iter {
        if x != first {
            return None;
        }
    }
    Some(first)
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_blake3::Blake3;
    use p3_commit::DirectMmcs;
    use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{AbstractExtensionField, AbstractField};
    use p3_interpolation::{interpolate_coset, interpolate_subgroup};
    use p3_matrix::bitrev::{BitReversableMatrix, BitReversedMatrixView};
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
    use rand::distributions::Standard;
    use rand::prelude::Distribution;
    use rand::{thread_rng, Rng};

    use super::*;

    type F = BabyBear;
    type F4 = BinomialExtensionField<F, 4>;
    type F5 = BinomialExtensionField<F, 5>;
    type MyHash = SerializingHasher32<Blake3>;
    type MyCompress = CompressionFunctionFromHasher<F, MyHash, 2, 8>;
    type ValMmcs = FieldMerkleTreeMmcs<F, MyHash, MyCompress, 8>;

    #[test]
    fn test_remainder_polys() {
        let trace: RowMajorMatrix<F> = RowMajorMatrix::rand(&mut thread_rng(), 32, 5);
        let point: F4 = thread_rng().gen();
        let values = interpolate_subgroup(&trace, point);
        let rs = Opening::compute_remainder_polys(point, &values);
        for (r, y) in rs.into_iter().zip(values) {
            // r(alpha) = p(alpha)
            assert_eq!(
                eval_poly(&r.into_iter().map(F4::from_base).collect_vec(), point),
                y
            );
        }
    }

    fn test_quotient_mmcs_with_sizes<EF: TwoAdicField + HasFrobenius<F>>(
        num_openings: usize,
        trace_sizes: &[(usize, usize)],
    ) where
        Standard: Distribution<EF>,
    {
        let hash = MyHash::new(Blake3 {});
        let compress = MyCompress::new(hash);
        let inner = ValMmcs::new(hash, compress);

        let shift = F::generator();

        let alphas: Vec<EF> = (0..num_openings).map(|_| thread_rng().gen()).collect_vec();

        let max_height = trace_sizes.iter().map(|&(h, _)| h).max().unwrap();
        let max_height_bits = log2_strict_usize(max_height);

        let (traces, ldes, dims, openings): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = trace_sizes
            .iter()
            .map(|&(height, width)| {
                let trace = RowMajorMatrix::<F>::rand_nonzero(&mut thread_rng(), height, width);
                let lde = Radix2Dit::default()
                    .coset_lde_batch(trace.clone(), 1, shift)
                    .bit_reverse_rows();
                let dims = lde.dimensions();
                let lde_truncated =
                    RowMajorMatrix::new((0..height).flat_map(|r| lde.row(r)).collect_vec(), width);
                let openings = alphas
                    .iter()
                    .map(|&alpha| {
                        Opening::new(
                            alpha,
                            interpolate_coset(
                                &BitReversedMatrixView::new(lde_truncated.clone()),
                                shift,
                                alpha,
                            ),
                        )
                    })
                    .collect_vec();
                (trace, lde.to_row_major_matrix(), dims, openings)
            })
            .multiunzip();

        let (comm, data) = inner.commit(ldes);
        let mmcs = QuotientMmcs::<F, EF, _> {
            inner,
            openings,
            coset_shift: shift,
            _phantom: PhantomData,
        };

        let index = thread_rng().gen_range(0..max_height);
        let (opened_values, proof) = mmcs.open_batch(index, &data);
        assert_eq!(
            mmcs.verify_batch(&comm, &dims, index, &opened_values, &proof),
            Ok(())
        );
        let mut bad_opened_values = opened_values.clone();
        bad_opened_values[0][0] += thread_rng().gen::<F>();
        assert!(mmcs
            .verify_batch(&comm, &dims, index, &bad_opened_values, &proof)
            .is_err());

        let mats = mmcs.get_matrices(&data);
        for (trace, mat, opened_values_for_mat) in izip!(traces, mats, opened_values) {
            let mat = mat.clone().to_row_major_matrix();

            let height_bits = log2_strict_usize(trace.height());
            let reduced_index = index >> (max_height_bits - height_bits);

            // check that open_batch and get_matrices are consistent
            assert_eq!(mat.row_slice(reduced_index), &opened_values_for_mat);

            // check low degree
            let poly =
                Radix2Dit::default().idft_batch(mat.bit_reverse_rows().to_row_major_matrix());
            let expected_degree = trace.height() - <EF as AbstractExtensionField<F>>::D;
            assert!((expected_degree..poly.height()).all(|r| poly.row(r).all(|x| x.is_zero())));
        }
    }

    #[test]
    fn test_quotient_mmcs() {
        let sizes: &[&[(usize, usize)]] = &[
            // single matrix
            &[(16, 1)],
            &[(16, 10)],
            &[(16, 14)],
            // multi matrix, same size
            &[(16, 5), (16, 10)],
            &[(8, 10), (8, 5)],
            // multi matrix, different size
            &[(16, 10), (32, 5)],
            &[(32, 52), (8, 30)],
        ];
        for num_openings in [1, 2, 3] {
            for sizes in sizes {
                test_quotient_mmcs_with_sizes::<F4>(num_openings, sizes);
                // make sure it works when Packing::WIDTH != Extension::D
                test_quotient_mmcs_with_sizes::<F5>(num_openings, sizes);
            }
        }
    }
}
