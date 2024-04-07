use crate::permuted::{PermutedMatrix, RowPermutation};
use crate::Matrix;

#[derive(Debug)]
pub struct VerticallyStridedPerm {
    // Store our height
    height: usize,
    stride: usize,
    offset: usize,
}

pub type VerticallyStridedMatrixView<Inner> = PermutedMatrix<VerticallyStridedPerm, Inner>;

impl VerticallyStridedPerm {
    pub fn new_view<T: Send + Sync, Inner: Matrix<T>>(
        inner: Inner,
        stride: usize,
        offset: usize,
    ) -> VerticallyStridedMatrixView<Inner> {
        let h = inner.height();
        let full_strides = h / stride;
        let remainder = h % stride;
        let final_stride = offset < remainder;
        let height = full_strides + final_stride as usize;
        PermutedMatrix {
            perm: Self {
                height,
                stride,
                offset,
            },
            inner,
        }
    }
}

impl RowPermutation for VerticallyStridedPerm {
    fn height(&self) -> usize {
        self.height
    }
    fn permute_row_index(&self, r: usize) -> usize {
        r * self.stride + self.offset
    }
}
