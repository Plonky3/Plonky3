use p3_matrix::Matrix;

pub trait AirWindow<T> {
    type M: Matrix<T>;

    /// A window of the main trace table.
    fn main(&self) -> &Self::M;
}

pub trait PairWindow<T>: AirWindow<T> {
    /// A window of the preprocessed table.
    fn preprocessed(&self) -> &Self::M;
}

pub trait AuxWindow<T>: AirWindow<T> {
    /// A window of the aux table.
    fn aux(&self) -> &Self::M;

    fn aux_randomness(&self) -> &[T];
}

pub trait PermutationWindow<T>: AirWindow<T> {
    /// A window of the permutation table.
    fn permutation(&self) -> &Self::M;

    fn permutation_randomness(&self) -> &[T];
}
