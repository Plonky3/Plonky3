pub trait CryptographicPermutation<T: Clone>: Clone {
    fn permute(&self, input: T) -> T;

    fn permute_mut(&self, input: &mut T) {
        *input = self.permute(input.clone());
    }
}

pub trait ArrayPermutation<T: Clone, const WIDTH: usize>:
    CryptographicPermutation<[T; WIDTH]>
{
}
