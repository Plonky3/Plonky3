pub trait CryptographicPermutation<T> {
    fn permute(&self, input: T) -> T;

    fn permute_mut(&self, input: &mut T)
    where
        T: Copy,
    {
        *input = self.permute(*input);
    }
}

pub trait ArrayPermutation<T, const WIDTH: usize>: CryptographicPermutation<[T; WIDTH]> {}
