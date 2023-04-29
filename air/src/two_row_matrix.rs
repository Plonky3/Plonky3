use p3_matrix::Matrix;

#[derive(Copy, Clone)]
pub struct TwoRowMatrixView<'a, T> {
    pub local: &'a [T],
    pub next: &'a [T],
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
