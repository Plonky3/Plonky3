use core::iter;
use core::marker::PhantomData;

use p3_field::{ExtensionField, Field};

use crate::Matrix;

#[derive(Debug)]
pub struct FlatMatrixView<F, EF, Inner>(Inner, PhantomData<(F, EF)>);

impl<F, EF, Inner> FlatMatrixView<F, EF, Inner> {
    pub fn new(inner: Inner) -> Self {
        Self(inner, PhantomData)
    }
    pub fn inner_ref(&self) -> &Inner {
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
        self.0.width() * EF::D
    }

    fn height(&self) -> usize {
        self.0.height()
    }

    type Row<'a> = FlatIter<F, Inner::Row<'a>>
    where
        Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        FlatIter {
            inner: self.0.row(r).peekable(),
            idx: 0,
            _phantom: PhantomData,
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
        if self.idx == EF::D {
            self.idx = 0;
            self.inner.next();
        }
        let value = self.inner.peek()?.as_base_slice()[self.idx];
        self.idx += 1;
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use p3_field::{extension::Complex, AbstractExtensionField, AbstractField};
    use p3_mersenne_31::Mersenne31;

    use crate::dense::RowMajorMatrix;

    use super::*;
    type F = Mersenne31;
    type EF = Complex<Mersenne31>;

    #[test]
    fn flat_matrix() {
        let values = vec![
            EF::from_base_fn(|i| F::from_canonical_usize(0 + i)),
            EF::from_base_fn(|i| F::from_canonical_usize(10 + i)),
            EF::from_base_fn(|i| F::from_canonical_usize(20 + i)),
            EF::from_base_fn(|i| F::from_canonical_usize(30 + i)),
        ];
        let ext = RowMajorMatrix::<EF>::new(values, 2);
        let flat = FlatMatrixView::<F, EF, _>::new(ext);
        assert_eq!(
            &*flat.row_slice(0),
            &[0, 1, 10, 11].map(|i| F::from_canonical_usize(i))
        );
        assert_eq!(
            &*flat.row_slice(1),
            &[20, 21, 30, 31].map(|i| F::from_canonical_usize(i))
        );
    }
}
