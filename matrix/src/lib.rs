//! Matrix library.

#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Debug, Display, Formatter};
use core::ops::Deref;
use itertools::{izip, Itertools};

use p3_field::{ExtensionField, Field, PackedValue};
use p3_maybe_rayon::prelude::*;

use crate::dense::RowMajorMatrix;
use crate::strided::VerticallyStridedMatrixView;

pub mod bitrev;
pub mod dense;
pub mod extension;
pub mod mul;
pub mod sparse;
pub mod stack;
pub mod strided;
pub mod util;

#[derive(Clone, Copy)]
pub struct Dimensions {
    pub width: usize,
    pub height: usize,
}

impl Debug for Dimensions {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

impl Display for Dimensions {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

pub trait Matrix<T: Send + Sync>: Send + Sync {
    fn width(&self) -> usize;
    fn height(&self) -> usize;

    fn dimensions(&self) -> Dimensions {
        Dimensions {
            width: self.width(),
            height: self.height(),
        }
    }

    fn get(&self, r: usize, c: usize) -> T {
        self.row(r).into_iter().nth(c).unwrap()
    }

    type Row<'a>: Iterator<Item = T> + Send + Sync
    where
        Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_>;

    fn rows(&self) -> impl Iterator<Item = Self::Row<'_>> {
        (0..self.height()).map(move |r| self.row(r))
    }

    fn par_rows(&self) -> impl IndexedParallelIterator<Item = Self::Row<'_>> {
        (0..self.height()).into_par_iter().map(move |r| self.row(r))
    }

    // Opaque return type implicitly captures &'_ self
    fn row_slice(&self, r: usize) -> impl Deref<Target = [T]> {
        self.row(r).into_iter().collect_vec()
    }

    fn first_row(&self) -> Self::Row<'_> {
        self.row(0)
    }

    fn last_row(&self) -> Self::Row<'_> {
        self.row(self.height() - 1)
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        RowMajorMatrix::new(
            (0..self.height()).flat_map(|r| self.row(r)).collect(),
            self.width(),
        )
    }

    fn horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> (impl Iterator<Item = P>, impl Iterator<Item = T>)
    where
        P: PackedValue<Value = T>,
        T: Clone + 'a,
    {
        let num_packed = self.width() / P::WIDTH;
        let packed = (0..num_packed).map(move |c| P::from_fn(|i| self.get(r, P::WIDTH * c + i)));
        let sfx = (num_packed * P::WIDTH..self.width()).map(move |c| self.get(r, c));
        (packed, sfx)
    }

    fn vertically_packed_row<'a, P>(&'a self, r: usize) -> impl Iterator<Item = P>
    where
        P: PackedValue<Value = T>,
    {
        (0..self.width()).map(move |c| P::from_fn(|i| self.get(r + i, c)))
    }

    fn vertically_strided(self, stride: usize, offset: usize) -> VerticallyStridedMatrixView<Self>
    where
        Self: Sized,
    {
        VerticallyStridedMatrixView {
            inner: self,
            stride,
            offset,
        }
    }

    fn columnwise_dot_product<EF>(&self, v: &[EF]) -> Vec<EF>
    where
        T: Field,
        EF: ExtensionField<T>,
    {
        self.par_rows().zip(v).par_fold_reduce(
            || vec![EF::zero(); self.width()],
            |mut acc, (row, &scale)| {
                izip!(&mut acc, row).for_each(|(a, x)| *a += scale * x);
                acc
            },
            |mut acc_l, acc_r| {
                izip!(&mut acc_l, acc_r).for_each(|(l, r)| *l += r);
                acc_l
            },
        )
    }
}
