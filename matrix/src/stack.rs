use crate::Matrix;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::marker::PhantomData;

#[deprecated]
pub struct VertStackN<T> {
    parts: Vec<Box<dyn Matrix<T>>>,
}

impl<T> VertStackN<T> {
    pub fn new(parts: Vec<Box<dyn Matrix<T>>>) -> Self {
        assert!(parts.len() > 0);
        let width = parts[0].width();
        for part in &parts[1..] {
            assert_eq!(part.width(), width);
        }
        Self { parts }
    }
}

pub struct VertStack2<T, First: Matrix<T>, Second: Matrix<T>> {
    first: First,
    second: Second,
    _phantom: PhantomData<T>,
}

impl<T, First: Matrix<T>, Second: Matrix<T>> VertStack2<T, First, Second> {
    pub fn new(first: First, second: Second) -> Self {
        assert_eq!(first.width(), second.width());
        Self {
            first,
            second,
            _phantom: PhantomData,
        }
    }
}

impl<T, First: Matrix<T>, Second: Matrix<T>> Matrix<T> for VertStack2<T, First, Second> {
    fn width(&self) -> usize {
        self.first.width()
    }

    fn height(&self) -> usize {
        self.first.height() + self.second.height()
    }

    fn row(&self, r: usize) -> &[T] {
        if r < self.first.height() {
            self.first.row(r)
        } else {
            self.second.row(r - self.first.height())
        }
    }
}
