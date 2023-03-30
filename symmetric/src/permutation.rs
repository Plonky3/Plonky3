use hyperfield::field::Field;

pub trait AlgebraicPermutation<F: Field, const WIDTH: usize> {
    fn permute(input: [F; WIDTH]) -> [F; WIDTH];
}
