use alloc::vec;
use alloc::vec::Vec;
use core::debug_assert_eq;
use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::{izip, Itertools};
use p3_commit::Mmcs;
use p3_field::extension::HasFrobenius;
use p3_field::{
    add_vecs, batch_multiplicative_inverse, binomial_expand, cyclic_subgroup_coset_known_order,
    eval_poly, scale_vec, Field, PackedField, TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix, MatrixRowSlices, MatrixRows};
use p3_util::log2_strict_usize;

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
pub struct QuotientMmcs<F: Field, Inner: Mmcs<F>> {
    pub(crate) inner: Inner,

    /// For each matrix, a list of claimed openings, one for each point that we open that batch of
    /// polynomials at.
    pub(crate) openings: Vec<Vec<Opening<F>>>,

    // The coset shift for the inner MMCS's evals, to correct `x` in the denominator.
    pub(crate) coset_shift: F,
}

/// A claimed opening.
#[derive(Clone, Debug)]
pub(crate) struct Opening<F: Field> {
    // point.minimal_poly()
    pub(crate) minpoly: Vec<F>,
    // for each column, the remainder poly r(X) = p(X) mod m(X)
    pub(crate) remainder_polys: Vec<Vec<F>>,
    // each remainder poly always has degree EF::D.
    // in this matrix, each row represents a coefficient of the remainder poly,
    // and the packed columns are grouped chunks of remainder_polys.
    // this matrix is missing any remaining coefficients that don't divide F::Packing::WIDTH
    // evenly, so you have to get those from remainder_polys.
    pub(crate) r_transposed_packed: Option<RowMajorMatrix<F::Packing>>,
}

impl<F: Field> Opening<F> {
    pub(crate) fn new<EF: HasFrobenius<F>>(point: EF, values: Vec<EF>) -> Self {
        let remainder_polys = Self::compute_remainder_polys(point, &values);
        let r_transposed_packed = transpose_and_pack(&remainder_polys);
        Self {
            minpoly: point.minimal_poly(),
            remainder_polys,
            r_transposed_packed,
        }
    }
    fn compute_remainder_polys<EF: HasFrobenius<F>>(point: EF, values: &[EF]) -> Vec<Vec<F>> {
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
///   [ 1, 2, 3],
///   [ 4, 5, 6],
///   [ 7, 8, 9],
///   [10,11,12],
///   [13,14,15],
/// ]`,
/// and F::Packing::WIDTH = 2, returns `[
///   [P(1,4),P(7,10)],
///   [P(2,5),P(8,11)],
///   [P(3,6),P(9,12)],
/// ]`
/// where P(..) is a packed field. Trailing values (`[13,14,15]` above) are ignored.
fn transpose_and_pack<F: Field>(polys: &[Vec<F>]) -> Option<RowMajorMatrix<F::Packing>> {
    let height = polys[0].len();
    let width = polys.len() / F::Packing::WIDTH;
    if width == 0 {
        return None;
    }
    Some(RowMajorMatrix::new(
        (0..height)
            .flat_map(|coeff_idx| {
                (0..width).map(move |packed_col| {
                    F::Packing::from_fn(move |i| {
                        polys[packed_col * F::Packing::WIDTH + i][coeff_idx]
                    })
                })
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

impl<F, Inner> Mmcs<F> for QuotientMmcs<F, Inner>
where
    F: TwoAdicField,
    Inner: Mmcs<F>,
    for<'a> Inner::Mat<'a>: MatrixRowSlices<F>,
{
    type ProverData = Inner::ProverData;
    type Commitment = Inner::Commitment;
    type Proof = Inner::Proof;
    type Error = QuotientError<Inner::Error>;
    type Mat<'a> = QuotientMatrix<F, Inner::Mat<'a>> where Self: 'a;

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
                let log2_height = log2_strict_usize(height);
                let bits_reduced = log_max_height - log2_height;
                let reduced_index = index >> bits_reduced;
                let x = self.coset_shift
                    * F::two_adic_generator(log2_height).exp_u64(reduced_index as u64);

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
                let log2_height = log2_strict_usize(height);
                let g = F::two_adic_generator(log2_height);
                let subgroup =
                    cyclic_subgroup_coset_known_order(g, self.coset_shift, height).collect_vec();

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
                let reduced_index = index >> bits_reduced;
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
pub struct QuotientMatrix<F: Field, Inner: MatrixRowSlices<F>> {
    inner: Inner,
    subgroup: Vec<F>,
    openings: Vec<Opening<F>>,
    /// For each row (associated with a subgroup element `x`), for each opening point,
    /// this holds `1 / m(X)`.
    inv_denominators: RowMajorMatrix<F>,
    _phantom: PhantomData<F>,
}

impl<F: Field, Inner: MatrixRowSlices<F>> Matrix<F> for QuotientMatrix<F, Inner> {
    fn width(&self) -> usize {
        self.inner.width() * self.openings.len()
    }

    fn height(&self) -> usize {
        self.inner.height()
    }
}

impl<F: Field, Inner: MatrixRowSlices<F>> MatrixRows<F> for QuotientMatrix<F, Inner> {
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

fn compute_quotient_matrix_row<F: Field>(
    x: F,
    openings: &[Opening<F>],
    m_invs: &[F],
    inner_row: &[F],
) -> Vec<F> {
    let mut qp_ys: Vec<F> = Vec::with_capacity(inner_row.len() * openings.len());

    // this is always EF::D.
    let r_poly_len = openings[0].remainder_polys[0].len();
    // [P(1,1,1,1),P(x,x,x,x),P(x^2,x^2,x^2,x^2),..]
    let packed_x_pows = x
        .powers()
        .take(r_poly_len)
        .map(|x_pow| F::Packing::from(x_pow))
        .collect_vec();

    for (opening, &m_inv) in openings.iter().zip(m_invs) {
        let packed_m_inv = F::Packing::from(m_inv);
        let (packed_ys, sfx_ys) = F::Packing::pack_slice_with_suffix(inner_row);

        if let Some(r_transposed_packed) = &opening.r_transposed_packed {
            // once const generic exprs are stable, this will unroll a lot nicer.
            // for now, we will not bother with threading a const D: usize through every function

            // first, we will evaluate the remainder polys at x, and put them in qp_ys_packed
            // the polynomial evaluation sum starts with the constant coefficients
            /*
            let mut packed_qp_ys = r_transposed_packed.row_slice(0).to_vec();
            for (coeff_idx, &packed_x_pow) in packed_x_pows.iter().enumerate().skip(1) {
                let coeffs = r_transposed_packed.row_slice(coeff_idx);
                for col in 0..packed_qp_ys.len() {
                    packed_qp_ys[col] += packed_x_pow * coeffs[col];
                }
            }
            for (packed_qp_y, &packed_y) in packed_qp_ys.iter_mut().zip(packed_ys) {
                // packed_qp_y currently holds r(X), but we want (p(X) - r(X))/m(X)
                *packed_qp_y = (packed_y - *packed_qp_y) * packed_m_inv;
            }
            qp_ys.extend_from_slice(F::Packing::unpack_slice(&packed_qp_ys));
            */

            // Simplest implementation
            packed_ys.iter().enumerate().for_each(|(col, &packed_y)| {
                let r_at_x: F::Packing = r_transposed_packed
                    .rows()
                    .zip(&packed_x_pows)
                    .map(|(coeffs, &x_pow)| coeffs[col] * x_pow)
                    .sum();
                let qp_y = (packed_y - r_at_x) * packed_m_inv;
                qp_ys.extend_from_slice(qp_y.as_slice());
            });

            /*
            // reserve space for packed qp_ys
            let start_idx = qp_ys.len();
            // copy the constant coefficient into qp_ys
            qp_ys.extend_from_slice(F::Packing::unpack_slice(r_transposed_packed.row_slice(0)));
            let packed_qp_ys = F::Packing::pack_slice_mut(&mut qp_ys[start_idx..]);
            for coeff_idx in 1..r_poly_len {
                let coeffs = r_transposed_packed.row_slice(coeff_idx);
                for col in 0..packed_qp_ys.len() {
                    packed_qp_ys[col] += packed_x_pows[coeff_idx] * coeffs[col];
                }
            }
            for (packed_qp_y, &packed_y) in packed_qp_ys.iter_mut().zip(packed_ys) {
                *packed_qp_y = (packed_y - *packed_qp_y) * packed_m_inv;
            }
            */
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
    use p3_interpolation::interpolate_subgroup;
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
    use rand::distributions::Standard;
    use rand::prelude::Distribution;
    use rand::{thread_rng, Rng};

    use super::*;

    type F = BabyBear;
    type F4 = BinomialExtensionField<F, 4>;
    type F5 = BinomialExtensionField<F, 4>;
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

        let alphas: Vec<EF> = (0..num_openings).map(|_| thread_rng().gen()).collect_vec();

        let max_height = trace_sizes.iter().map(|&(h, _)| h).max().unwrap();
        let max_height_bits = log2_strict_usize(max_height);

        let (traces, ldes, dims, openings): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = trace_sizes
            .iter()
            .map(|&(height, width)| {
                let trace = RowMajorMatrix::<F>::rand_nonzero(&mut thread_rng(), height, width);
                let lde = Radix2Dit.coset_lde_batch(trace.clone(), 1, F::generator());
                let dims = lde.dimensions();
                let openings = alphas
                    .iter()
                    .map(|&alpha| Opening::new(alpha, interpolate_subgroup(&trace, alpha)))
                    .collect_vec();
                (trace, lde, dims, openings)
            })
            .multiunzip();

        let (comm, data) = inner.commit(ldes);
        let mmcs = QuotientMmcs {
            inner,
            openings,
            coset_shift: F::generator(),
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
            let poly = Radix2Dit.idft_batch(mat);
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
