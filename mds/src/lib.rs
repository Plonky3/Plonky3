use p3_symmetric::permutation::ArrayPermutation;

pub mod babybear;
pub mod coset_mds;
pub mod goldilocks;
pub mod integrated_coset_mds;
pub mod mersenne31;
pub mod util;

pub trait MdsPermutation<T: Clone, const WIDTH: usize>: ArrayPermutation<T, WIDTH> {}
