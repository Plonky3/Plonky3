use p3_matrix::Matrix;

pub trait AirWindow<T: Copy> {
    type M: Matrix<T>;

    /// A window of the main trace table.
    fn main(&self) -> Self::M;
}

pub trait PairWindow<T: Copy>: AirWindow<T> {
    /// A window of the preprocessed table.
    fn preprocessed(&self) -> Self::M;
}

pub trait PermutationWindow<T: Copy>: AirWindow<T> {
    /// A window of the permutation table.
    fn permutation(&self) -> Self::M;

    fn permutation_randomness(&self) -> &[T];
}
