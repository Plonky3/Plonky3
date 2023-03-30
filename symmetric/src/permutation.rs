pub trait CryptographicPermutation<T, const WIDTH: usize> {
    fn permute(input: [T; WIDTH]) -> [T; WIDTH];
}
