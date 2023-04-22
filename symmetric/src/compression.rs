/// An `n`-to-1 compression function.
pub trait CompressionFunction<T, const N: usize> {
    fn compress(&self, input: &[&T; N]) -> T;
}
