pub trait CryptographicPermutation<T> {
    fn permute(input: T) -> T;
}

pub trait ArrayPermutation<T, const WIDTH: usize>: CryptographicPermutation<[T; WIDTH]> {}
