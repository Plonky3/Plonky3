use hyperfield::field::Field;

pub trait AlgebraicPermutation<F: Field, const WIDTH: usize> {
    fn permute(&self, input: [F; WIDTH]) -> [F; WIDTH];
}
