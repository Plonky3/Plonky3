use alloc::vec::Vec;
use core::iter;
use core::marker::PhantomData;
use core::ops::Deref;

use p3_field::{ExtensionField, Field};

use crate::Matrix;

/// Flattens a matrix of extension field elements to one of base field elements. The flattening is
/// done horizontally, resulting in a wider matrix.
#[derive(Debug)]
pub struct FlatMatrixView<F, EF, Inner>(Inner, PhantomData<(F, EF)>);

impl<F, EF, Inner> FlatMatrixView<F, EF, Inner> {
    pub const fn new(inner: Inner) -> Self {
        Self(inner, PhantomData)
    }
}

impl<F, EF, Inner> Deref for FlatMatrixView<F, EF, Inner> {
    type Target = Inner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F, EF, Inner> Matrix<F> for FlatMatrixView<F, EF, Inner>
where
    F: Field,
    EF: ExtensionField<F>,
    Inner: Matrix<EF>,
{
    fn width(&self) -> usize {
        self.0.width() * EF::DIMENSION
    }

    fn height(&self) -> usize {
        self.0.height()
    }

    fn get(&self, r: usize, c: usize) -> Option<F> {
        // The c'th base field element in a row of extension field elements is
        // at index c % EF::DIMENSION in the c / EF::DIMENSION'th extension element.
        let (r, c) = (r, c / EF::DIMENSION);
        let inner = self.0.get(r, c)?;
        Some(inner.as_basis_coefficients_slice()[c % EF::DIMENSION])
    }

    unsafe fn get_unchecked(&self, r: usize, c: usize) -> F {
        // The c'th base field element in a row of extension field elements is
        // at index c % EF::DIMENSION in the c / EF::DIMENSION'th extension element.
        let (r, c) = (r, c / EF::DIMENSION);
        let inner = unsafe {
            // Safety: The caller must ensure that r < self.height() and c < self.width().
            // Assuming this, c / EF::DIMENSION < self.0.width().
            self.0.get_unchecked(r, c)
        };
        inner.as_basis_coefficients_slice()[c % EF::DIMENSION]
    }

    fn row(
        &self,
        r: usize,
    ) -> Option<impl IntoIterator<Item = F, IntoIter = impl Iterator<Item = F> + Send + Sync>> {
        Some(FlatIter {
            inner: self.0.row(r)?.into_iter().peekable(),
            idx: 0,
            _phantom: PhantomData,
        })
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = F, IntoIter = impl Iterator<Item = F> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height().
            FlatIter {
                inner: self.0.row_unchecked(r).into_iter().peekable(),
                idx: 0,
                _phantom: PhantomData,
            }
        }
    }

    unsafe fn row_subset_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = F, IntoIter = impl Iterator<Item = F> + Send + Sync> {
        // We can skip the first start / EF::DIMENSION elements in the row.
        let len = end - start;
        let inner_start = start / EF::DIMENSION;
        unsafe {
            // Safety: The caller must ensure that r < self.height(), start <= end and end < self.width().
            FlatIter {
                inner: self
                    .0
                    // We set end to be the width of the inner matrix and use take to ensure we get the right
                    // number of elements.
                    .row_subset_unchecked(r, inner_start, self.0.width())
                    .into_iter()
                    .peekable(),
                idx: start,
                _phantom: PhantomData,
            }
            .take(len)
        }
    }

    fn row_slice(&self, r: usize) -> Option<impl Deref<Target = [F]>> {
        Some(
            self.0
                .row_slice(r)?
                .iter()
                .flat_map(|val| val.as_basis_coefficients_slice())
                .copied()
                .collect::<Vec<_>>(),
        )
    }

    unsafe fn row_slice_unchecked(&self, r: usize) -> impl Deref<Target = [F]> {
        unsafe {
            // Safety: The caller must ensure that r < self.height().
            self.0
                .row_slice_unchecked(r)
                .iter()
                .flat_map(|val| val.as_basis_coefficients_slice())
                .copied()
                .collect::<Vec<_>>()
        }
    }
}

pub struct FlatIter<F, I: Iterator> {
    inner: iter::Peekable<I>,
    idx: usize,
    _phantom: PhantomData<F>,
}

impl<F, EF, I> Iterator for FlatIter<F, I>
where
    F: Field,
    EF: ExtensionField<F>,
    I: Iterator<Item = EF>,
{
    type Item = F;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == EF::DIMENSION {
            self.idx = 0;
            self.inner.next();
        }
        let value = self.inner.peek()?.as_basis_coefficients_slice()[self.idx];
        self.idx += 1;
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_field::extension::Complex;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
    use p3_mersenne_31::Mersenne31;

    use super::*;
    use crate::dense::RowMajorMatrix;
    type F = Mersenne31;
    type EF = Complex<Mersenne31>;

    // TODO: ADD Tests

    #[test]
    fn flat_matrix() {
        let values = vec![
            EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + 10)),
            EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + 20)),
            EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + 30)),
            EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + 40)),
        ];
        let ext = RowMajorMatrix::<EF>::new(values, 2);
        let flat = FlatMatrixView::<F, EF, _>::new(ext);
        assert_eq!(
            &*flat.row_slice(0).unwrap(),
            &[10, 11, 20, 21].map(F::from_u8)
        );
        assert_eq!(
            &*flat.row_slice(1).unwrap(),
            &[30, 31, 40, 41].map(F::from_u8)
        );
    }
}
