use p3_symmetric::permutation::ArrayPermutation;

pub mod babybear;
pub mod goldilocks;
pub mod mersenne31;
pub mod util;

pub trait MDSPermutation<T, const WIDTH: usize>: ArrayPermutation<T, WIDTH> {}
