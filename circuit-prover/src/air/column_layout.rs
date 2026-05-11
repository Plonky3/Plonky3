//! Layout helpers for `#[repr(C)]` column views.
//!
//! Column count matches `size_of::<View<u8>>()`, and [`column_indices`] builds
//! a parallel `View<usize>` of base offsets (same idea as zk_evm’s column maps).

use core::mem::{size_of, transmute};

/// `[0, 1, …, N - 1]` built at compile time.
pub const fn column_indices<const N: usize>() -> [usize; N] {
    let mut arr = [0usize; N];
    let mut i = 0;
    while i < N {
        arr[i] = i;
        i += 1;
    }
    arr
}

/// Preprocessed columns shared by [`super::const_air::ConstAir`] and
/// [`super::public_air::PublicAir`]: multiplicity then witness index.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WitnessLookupPrepCols<T: Copy> {
    pub multiplicity: T,
    pub witness_idx: T,
}

/// Base-field columns per row for witness lookup preprocessing.
pub const WITNESS_LOOKUP_PREP_LANE_WIDTH: usize = size_of::<WitnessLookupPrepCols<u8>>();

const fn witness_lookup_prep_col_map() -> WitnessLookupPrepCols<usize> {
    let indices = column_indices::<WITNESS_LOOKUP_PREP_LANE_WIDTH>();
    unsafe {
        transmute::<[usize; WITNESS_LOOKUP_PREP_LANE_WIDTH], WitnessLookupPrepCols<usize>>(indices)
    }
}

/// Offsets `(0..WITNESS_LOOKUP_PREP_LANE_WIDTH)` as named fields.
pub const WITNESS_LOOKUP_PREP_COL_MAP: WitnessLookupPrepCols<usize> = witness_lookup_prep_col_map();

const _: () = assert!(
    size_of::<WitnessLookupPrepCols<usize>>()
        == WITNESS_LOOKUP_PREP_LANE_WIDTH * size_of::<usize>()
);
