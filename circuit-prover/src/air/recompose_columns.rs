//! Column layout for [`super::recompose_air::RecomposeAir`] preprocessed traces.

use core::mem::{size_of, transmute};

use super::column_layout::column_indices;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RecomposePrepLaneCols<T: Copy> {
    pub output_idx: T,
    pub out_mult: T,
}

pub(crate) const RECOMPOSE_PREP_LANE_WIDTH: usize = size_of::<RecomposePrepLaneCols<u8>>();

const fn recompose_prep_lane_col_map() -> RecomposePrepLaneCols<usize> {
    let indices = column_indices::<RECOMPOSE_PREP_LANE_WIDTH>();
    unsafe {
        transmute::<[usize; RECOMPOSE_PREP_LANE_WIDTH], RecomposePrepLaneCols<usize>>(indices)
    }
}

pub(crate) const RECOMPOSE_PREP_LANE_COL_MAP: RecomposePrepLaneCols<usize> =
    recompose_prep_lane_col_map();

const _: () = assert!(
    size_of::<RecomposePrepLaneCols<usize>>() == RECOMPOSE_PREP_LANE_WIDTH * size_of::<usize>()
);
