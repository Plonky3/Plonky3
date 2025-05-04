/// An iterator which iterates two other iterators of the same length simultaneously.
///
/// Equality of the lengths of `a` abd `b` are at construction time.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct ZipEq<A, B> {
    a: A,
    b: B,
}

/// Zips two iterators but **panics** if they are not of the same length.
///
/// Similar to `itertools::zip_eq`, but we check the lengths at construction time.
pub fn zip_eq<A, AIter, B, BIter, Error>(
    a: A,
    b: B,
    err: Error,
) -> Result<ZipEq<A::IntoIter, B::IntoIter>, Error>
where
    A: IntoIterator<IntoIter = AIter>,
    AIter: ExactSizeIterator,
    B: IntoIterator<IntoIter = BIter>,
    BIter: ExactSizeIterator,
{
    let a_iter = a.into_iter();
    let b_iter = b.into_iter();
    match a_iter.len() == b_iter.len() {
        true => Ok(ZipEq {
            a: a_iter,
            b: b_iter,
        }),
        false => Err(err),
    }
}

impl<A, B> Iterator for ZipEq<A, B>
where
    A: ExactSizeIterator, // We need to use ExactSizeIterator here otherwise the size_hint() methods could differ.
    B: ExactSizeIterator,
{
    type Item = (A::Item, B::Item);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.a.next(), self.b.next()) {
            (Some(a), Some(b)) => Some((a, b)),
            (None, None) => None,
            _ => unreachable!("The iterators must have the same length."),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // self.a.size_hint() = self.b.size_hint() as a and b are ExactSizeIterators
        // and we checked that they are the same length at construction time.
        debug_assert_eq!(self.a.size_hint(), self.b.size_hint());
        self.a.size_hint()
    }
}

impl<A, B> ExactSizeIterator for ZipEq<A, B>
where
    A: ExactSizeIterator,
    B: ExactSizeIterator,
{
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn test_zip_eq_success() {
        let a = [1, 2, 3];
        let b = ['a', 'b', 'c'];

        // Expect zip_eq to succeed since both slices are length 3.
        let zipped = zip_eq(a, b, "length mismatch").unwrap();

        let result: Vec<_> = zipped.collect();

        // Expect tuples zipped together positionally.
        assert_eq!(result, vec![(1, 'a'), (2, 'b'), (3, 'c')]);
    }

    #[test]
    fn test_zip_eq_length_mismatch() {
        let a = [1, 2];
        let b = ['x', 'y', 'z'];

        // Use pattern matching instead of .unwrap_err()
        match zip_eq(a, b, "oops") {
            Err(e) => assert_eq!(e, "oops"),
            Ok(_) => panic!("expected error due to mismatched lengths"),
        }
    }

    #[test]
    fn test_zip_eq_empty_iterators() {
        let a: [i32; 0] = [];
        let b: [char; 0] = [];

        // Zipping two empty iterators should succeed and produce an empty iterator.
        let zipped = zip_eq(a, b, "mismatch").unwrap();

        let result: Vec<_> = zipped.collect();

        // The result should be an empty vector.
        assert!(result.is_empty());
    }

    #[test]
    fn test_zip_eq_size_hint() {
        let a = [10, 20];
        let b = [100, 200];

        let zipped = zip_eq(a, b, "bad").unwrap();

        // Size hint should reflect the number of items remaining.
        assert_eq!(zipped.size_hint(), (2, Some(2)));
    }

    #[test]
    fn test_zip_eq_unreachable_case() {
        let a = [1, 2];
        let b = [3, 4];

        let mut zipped = zip_eq(a, b, "fail").unwrap();

        // Manually advance past the last element
        assert_eq!(zipped.next(), Some((1, 3)));
        assert_eq!(zipped.next(), Some((2, 4)));
        assert_eq!(zipped.next(), None);

        // If one iterator somehow returns more, it would panic — unreachable by construction.
        // This line is commented because it would panic if uncommented:
        // zipped.b.next(); // ⚠️ Do not call after fully consumed
    }
}
