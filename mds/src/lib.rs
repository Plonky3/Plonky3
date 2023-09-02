use p3_symmetric::permutation::ArrayPermutation;

pub mod babybear;
pub mod coset_mds;
pub mod goldilocks;
pub mod mersenne31;
pub(crate) mod util;

pub trait MdsPermutation<T, const WIDTH: usize>: ArrayPermutation<T, WIDTH> {}
