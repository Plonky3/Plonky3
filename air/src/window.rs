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

pub struct BasicAirWindow<'a, T> {
    pub(crate) main: TwoRowMatrixView<'a, T>,
}

impl<'a, T> BasicAirWindow<'a, T> {
    pub fn new(local: &'a [T], next: &'a [T]) -> Self {
        Self {
            main: TwoRowMatrixView { local, next },
        }
    }
}

impl<'a, T: Copy> AirWindow<T> for BasicAirWindow<'a, T> {
    type M = TwoRowMatrixView<'a, T>;

    fn main(&self) -> Self::M {
        self.main
    }
}

#[derive(Copy, Clone)]
pub struct TwoRowMatrixView<'a, T> {
    local: &'a [T],
    next: &'a [T],
}

impl<'a, T> Matrix<T> for TwoRowMatrixView<'a, T> {
    fn width(&self) -> usize {
        self.local.len()
    }

    fn height(&self) -> usize {
        2
    }

    fn row(&self, r: usize) -> &[T] {
        match r {
            0 => self.local,
            1 => self.next,
            _ => panic!("baby-stark only supports two-rows windows"),
        }
    }
}
