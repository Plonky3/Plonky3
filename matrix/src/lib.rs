//! Matrix library.

#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Debug, Display, Formatter};
use core::ops::Deref;

use itertools::{izip, Itertools};
use p3_field::{
    dot_product, AbstractExtensionField, AbstractField, ExtensionField, Field, PackedValue,
};
use p3_maybe_rayon::prelude::*;
use strided::{VerticallyStridedMatrixView, VerticallyStridedRowIndexMap};
use tracing::instrument;

use crate::dense::RowMajorMatrix;

pub mod bitrev;
pub mod dense;
pub mod extension;
pub mod mul;
pub mod row_index_mapped;
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
        self.row(r).nth(c).unwrap()
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
        self.row(r).collect_vec()
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
    ) -> (
        impl Iterator<Item = P> + Send + Sync,
        impl Iterator<Item = T> + Send + Sync,
    )
    where
        P: PackedValue<Value = T>,
        T: Clone + 'a,
    {
        let num_packed = self.width() / P::WIDTH;
        let packed = (0..num_packed).map(move |c| P::from_fn(|i| self.get(r, P::WIDTH * c + i)));
        let sfx = (num_packed * P::WIDTH..self.width()).map(move |c| self.get(r, c));
        (packed, sfx)
    }

    /// Zero padded.
    fn padded_horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> impl Iterator<Item = P> + Send + Sync
    where
        P: PackedValue<Value = T>,
        T: Clone + Default + 'a,
    {
        let mut row_iter = self.row(r);
        let num_elems = self.width().div_ceil(P::WIDTH);
        // array::from_fn currently always calls in order, but it's not clear whether that's guaranteed.
        (0..num_elems).map(move |_| P::from_fn(|_| row_iter.next().unwrap_or_default()))
    }

    fn par_horizontally_packed_rows<'a, P>(
        &'a self,
    ) -> impl IndexedParallelIterator<
        Item = (
            impl Iterator<Item = P> + Send + Sync,
            impl Iterator<Item = T> + Send + Sync,
        ),
    >
    where
        P: PackedValue<Value = T>,
        T: Clone + 'a,
    {
        (0..self.height())
            .into_par_iter()
            .map(|r| self.horizontally_packed_row(r))
    }

    fn par_padded_horizontally_packed_rows<'a, P>(
        &'a self,
    ) -> impl IndexedParallelIterator<Item = impl Iterator<Item = P> + Send + Sync>
    where
        P: PackedValue<Value = T>,
        T: Clone + Default + 'a,
    {
        (0..self.height())
            .into_par_iter()
            .map(|r| self.padded_horizontally_packed_row(r))
    }

    /// Wraps at the end.
    fn vertically_packed_row<P>(&self, r: usize) -> impl Iterator<Item = P>
    where
        P: PackedValue<Value = T>,
    {
        (0..self.width()).map(move |c| P::from_fn(|i| self.get((r + i) % self.height(), c)))
    }

    fn vertically_strided(self, stride: usize, offset: usize) -> VerticallyStridedMatrixView<Self>
    where
        Self: Sized,
    {
        VerticallyStridedRowIndexMap::new_view(self, stride, offset)
    }

    /// Compute Mᵀv, aka premultiply this matrix by the given vector,
    /// aka scale each row by the corresponding entry in `v` and take the sum across rows.
    /// `v` can be a vector of extension elements.
    #[instrument(level = "debug", skip_all, fields(dims = %self.dimensions()))]
    fn columnwise_dot_product<EF>(&self, v: &[EF]) -> Vec<EF>
    where
        T: Field,
        EF: ExtensionField<T>,
    {
        let packed_width = self.width().div_ceil(T::Packing::WIDTH);

        let packed_result = self
            .par_padded_horizontally_packed_rows::<T::Packing>()
            .zip(v)
            .par_fold_reduce(
                || vec![EF::ExtensionPacking::zero(); packed_width],
                |mut acc, (row, &scale)| {
                    let scale = EF::ExtensionPacking::from_base_fn(|i| {
                        T::Packing::from(scale.as_base_slice()[i])
                    });
                    izip!(&mut acc, row).for_each(|(l, r)| *l += scale * r);
                    acc
                },
                |mut acc_l, acc_r| {
                    izip!(&mut acc_l, acc_r).for_each(|(l, r)| *l += r);
                    acc_l
                },
            );

        packed_result
            .into_iter()
            .flat_map(|p| {
                (0..T::Packing::WIDTH)
                    .map(move |i| EF::from_base_fn(|j| p.as_base_slice()[j].as_slice()[i]))
            })
            .take(self.width())
            .collect()
    }

    /// Multiply this matrix by the vector of powers of `base`, which is an extension element.
    fn dot_ext_powers<EF>(&self, base: EF) -> impl IndexedParallelIterator<Item = EF>
    where
        T: Field,
        EF: ExtensionField<T>,
    {
        let powers_packed = base
            .ext_powers_packed()
            .take(self.width().next_multiple_of(T::Packing::WIDTH))
            .collect_vec();
        self.par_padded_horizontally_packed_rows::<T::Packing>()
            .map(move |row_packed| {
                let packed_sum_of_packed: EF::ExtensionPacking =
                    dot_product(powers_packed.iter().copied(), row_packed);
                let sum_of_packed: EF = EF::from_base_fn(|i| {
                    packed_sum_of_packed.as_base_slice()[i]
                        .as_slice()
                        .iter()
                        .copied()
                        .sum()
                });
                sum_of_packed
            })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use itertools::izip;
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::AbstractField;
    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_columnwise_dot_product() {
        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;

        let m = RowMajorMatrix::<F>::rand(&mut thread_rng(), 1 << 8, 1 << 4);
        let v = RowMajorMatrix::<EF>::rand(&mut thread_rng(), 1 << 8, 1).values;

        let mut expected = vec![EF::zero(); m.width()];
        for (row, &scale) in izip!(m.rows(), &v) {
            for (l, r) in izip!(&mut expected, row) {
                *l += scale * r;
            }
        }

        assert_eq!(m.columnwise_dot_product(&v), expected);
    }
}
