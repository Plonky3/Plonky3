use crate::AirTypes;
use p3_matrix::Matrix;

pub trait AirWindow<T: AirTypes> {
    type M: Matrix<T::Var>;

    /// A window of the main trace table.
    fn main(&self) -> Self::M;

    fn is_first_row(&self) -> T::Exp;
    fn is_last_row(&self) -> T::Exp;
    fn is_transition(&self) -> T::Exp;
}

pub trait PairWindow<T: AirTypes>: AirWindow<T> {
    /// A window of the preprocessed table.
    fn preprocessed(&self) -> Self::M;
}

pub trait PermutationWindow<T: AirTypes>: AirWindow<T> {
    /// A window of the permutation table.
    fn permutation(&self) -> Self::M;

    fn permutation_randomness(&self) -> &[T];
}

pub struct BasicAirWindow<'a, T: AirTypes> {
    pub main: TwoRowMatrixView<'a, T::Var>,
    pub is_first_row: T::Exp,
    pub is_last_row: T::Exp,
    pub is_transition: T::Exp,
}

impl<'a, T: AirTypes> AirWindow<T> for BasicAirWindow<'a, T> {
    type M = TwoRowMatrixView<'a, T::Var>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> T::Exp {
        self.is_first_row.clone()
    }

    fn is_last_row(&self) -> T::Exp {
        self.is_last_row.clone()
    }

    fn is_transition(&self) -> T::Exp {
        self.is_transition.clone()
    }
}

#[derive(Copy, Clone)]
pub struct TwoRowMatrixView<'a, T> {
    local: &'a [T],
    next: &'a [T],
}

impl<'a, T> TwoRowMatrixView<'a, T> {
    pub fn new(local: &'a [T], next: &'a [T]) -> Self {
        Self { local, next }
    }
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
            _ => panic!("BasicAirWindow only supports two-rows windows"),
        }
    }
}
