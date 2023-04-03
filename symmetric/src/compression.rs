/// An `n`-to-1 compression function.
pub trait CompressionFunction<T, const N: usize> {
    fn compress(input: &[&T; N]) -> T;
}
