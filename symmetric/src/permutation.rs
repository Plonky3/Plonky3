/// A permutation in the mathematical sense.
pub trait Permutation<T: Clone>: Clone + Sync {
    // The methods permute, permute_mut are defined in a circular manner
    // so you only need to implement one of them and will get the other
    // for free. If you implement neither, this will cause a run time
    // error.

    #[inline(always)]
    fn permute(&self, mut input: T) -> T {
        self.permute_mut(&mut input);
        input
    }

    fn permute_mut(&self, input: &mut T) {
        *input = self.permute(input.clone());
    }
}

/// A permutation thought to be cryptographically secure, in the sense that it is thought to be
/// difficult to distinguish (in a nontrivial way) from a random permutation.
pub trait CryptographicPermutation<T: Clone>: Permutation<T> {}
