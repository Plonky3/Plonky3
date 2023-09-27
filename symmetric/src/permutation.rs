/// A permutation in the mathematical sense.
pub trait Permutation<T: Clone>: Clone {
    fn permute(&self, input: T) -> T;

    fn permute_mut(&self, input: &mut T) {
        *input = self.permute(input.clone());
    }
}

/// A permutation thought to be cryptographically secure, in the sense that it is thought to be
/// difficult to distinguish (in a nontrivial way) from a random permutation.
pub trait CryptographicPermutation<T: Clone>: Permutation<T> {}
