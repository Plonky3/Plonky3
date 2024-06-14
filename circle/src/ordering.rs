use p3_matrix::{
    row_index_mapped::{RowIndexMap, RowIndexMappedView},
    Matrix,
};
use p3_util::{log2_strict_usize, reverse_bits_len};

pub(crate) fn cfft_index_to_natural(index: usize, log_height: usize) -> usize {
    let msb = index & 1;
    let index = reverse_bits_len(index >> 1, log_height - 1) << 1;
    if msb == 0 {
        index
    } else {
        (1 << log_height) - index - 1
    }
}

pub(crate) fn natural_index_to_cfft(index: usize, log_height: usize) -> usize {
    let (index, lsb) = (index >> 1, index & 1);
    reverse_bits_len(
        if lsb == 0 {
            index
        } else {
            (1 << log_height) - index - 1
        },
        log_height,
    )
}

pub(crate) fn natural_slice_to_cfft<T: Clone>(xs: &[T]) -> Vec<T> {
    let log_n = log2_strict_usize(xs.len());
    (0..xs.len())
        .map(|i| xs[cfft_index_to_natural(i, log_n)].clone())
        .collect()
}

pub(crate) fn cfft_slice_to_natural<T: Clone>(xs: &[T]) -> Vec<T> {
    let log_n = log2_strict_usize(xs.len());
    (0..xs.len())
        .map(|i| xs[natural_index_to_cfft(i, log_n)].clone())
        .collect()
}

pub type CfftAsNaturalView<M> = RowIndexMappedView<CfftAsNaturalPerm, M>;

#[derive(Copy, Clone)]
pub struct CfftAsNaturalPerm {
    log_height: usize,
}

impl RowIndexMap for CfftAsNaturalPerm {
    fn height(&self) -> usize {
        1 << self.log_height
    }
    fn map_row_index(&self, r: usize) -> usize {
        natural_index_to_cfft(r, self.log_height)
    }
}

impl CfftAsNaturalPerm {
    pub fn view<T: Send + Sync, M: Matrix<T>>(
        inner: M,
    ) -> RowIndexMappedView<CfftAsNaturalPerm, M> {
        RowIndexMappedView {
            index_map: CfftAsNaturalPerm {
                log_height: log2_strict_usize(inner.height()),
            },
            inner,
        }
    }
}

pub type NaturalAsCfftView<M> = RowIndexMappedView<NaturalAsCfftPerm, M>;

#[derive(Copy, Clone)]
pub struct NaturalAsCfftPerm {
    log_height: usize,
}

impl RowIndexMap for NaturalAsCfftPerm {
    fn height(&self) -> usize {
        1 << self.log_height
    }
    fn map_row_index(&self, r: usize) -> usize {
        cfft_index_to_natural(r, self.log_height)
    }
}

impl NaturalAsCfftPerm {
    pub fn view<T: Send + Sync, M: Matrix<T>>(
        inner: M,
    ) -> RowIndexMappedView<NaturalAsCfftPerm, M> {
        RowIndexMappedView {
            index_map: NaturalAsCfftPerm {
                log_height: log2_strict_usize(inner.height()),
            },
            inner,
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;

    #[test]
    fn test_cfft2nat() {
        assert_eq!(
            (0..8).map(|i| cfft_index_to_natural(i, 3)).collect_vec(),
            vec![0, 7, 4, 3, 2, 5, 6, 1],
        );
        for log_n in 1..5 {
            for i in 0..(1 << log_n) {
                assert_eq!(
                    i,
                    cfft_index_to_natural(natural_index_to_cfft(i, log_n), log_n)
                );
            }
        }
    }
}
