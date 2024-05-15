use core::iter;
use core::marker::PhantomData;

use p3_field::{Extension, ExtensionAlgebra};

use crate::Matrix;

#[derive(Debug)]
pub struct FlatMatrixView<A, Inner>(Inner, PhantomData<A>);

impl<A: ExtensionAlgebra, Inner: Matrix<Extension<A>>> FlatMatrixView<A, Inner> {
    pub fn new(inner: Inner) -> Self {
        Self(inner, PhantomData)
    }
    pub fn inner_ref(&self) -> &Inner {
        &self.0
    }
}

impl<A: ExtensionAlgebra, Inner: Matrix<Extension<A>>> Matrix<A::Base>
    for FlatMatrixView<A, Inner>
{
    fn width(&self) -> usize {
        self.0.width() * A::D
    }

    fn height(&self) -> usize {
        self.0.height()
    }

    type Row<'a> = FlatIter<A, Inner::Row<'a>>
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

pub struct FlatIter<A, I: Iterator> {
    inner: iter::Peekable<I>,
    idx: usize,
    _phantom: PhantomData<A>,
}

impl<A, I> Iterator for FlatIter<A, I>
where
    A: ExtensionAlgebra,
    I: Iterator<Item = Extension<A>>,
{
    type Item = A::Base;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == A::D {
            self.idx = 0;
            self.inner.next();
        }
        let value = self.inner.peek()?[self.idx];
        self.idx += 1;
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_field::extension::Complex;
    use p3_field::{AbstractExtensionField, AbstractField};
    use p3_mersenne_31::Mersenne31;

    use super::*;
    use crate::dense::RowMajorMatrix;
    type F = Mersenne31;
    type EF = Complex<Mersenne31>;

    #[test]
    fn flat_matrix() {
        let values = vec![
            EF::from_base_fn(|i| F::from_canonical_usize(i + 10)),
            EF::from_base_fn(|i| F::from_canonical_usize(i + 20)),
            EF::from_base_fn(|i| F::from_canonical_usize(i + 30)),
            EF::from_base_fn(|i| F::from_canonical_usize(i + 40)),
        ];
        let ext = RowMajorMatrix::<EF>::new(values, 2);
        let flat = FlatMatrixView::<F, EF, _>::new(ext);
        assert_eq!(
            &*flat.row_slice(0),
            &[10, 11, 20, 21].map(F::from_canonical_usize)
        );
        assert_eq!(
            &*flat.row_slice(1),
            &[30, 31, 40, 41].map(F::from_canonical_usize)
        );
    }
}
