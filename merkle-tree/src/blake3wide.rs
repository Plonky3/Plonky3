// TODO parallelize?

use alloc::vec::Vec;

use blake3::Hasher;
use p3_symmetric::CryptographicHasher;

// A wrapper around blake3 that operates on arrays of WIDTH u8 at a time.
// This is used by the hybrid Merkle tree to efficiently hash multiple elements in parallel.
#[derive(Copy, Clone, Debug)]
pub struct Blake3Wide<const WIDTH: usize>;

impl<const WIDTH: usize> CryptographicHasher<[u8; WIDTH], [[u8; WIDTH]; 32]> for Blake3Wide<WIDTH> {
    fn hash_iter<I>(&self, input: I) -> [[u8; WIDTH]; 32]
    where
        I: IntoIterator<Item = [u8; WIDTH]>,
    {
        let mut hashers: Vec<Hasher> = (0..WIDTH).map(|_| blake3::Hasher::new()).collect();

        const BUFLEN: usize = 512;

        // Option1

        // Unzip the iterator of elements of type [u8; WIDTH] into WIDTH iterators of type u8
        let mut iters: Vec<Vec<u8>> = (0..WIDTH).map(|_| Vec::new()).collect();
        for item in input {
            for i in 0..WIDTH {
                iters[i].push(item[i]);
            }
        }

        let digests = iters
            .into_iter()
            .zip(hashers.iter_mut())
            .map(|(hasher_input, hasher)| {
                p3_util::apply_to_chunks::<BUFLEN, _, _>(hasher_input, |buf| {
                    hasher.update(buf);
                });
                hasher.finalize()
            });

        // TODO how costly is this and can one transmute instead?
        let mut output = [[0u8; WIDTH]; 32];

        for (i, digest) in digests.enumerate() {
            for (j, byte) in digest.as_bytes().into_iter().enumerate() {
                output[j][i] = *byte;
            }
        }

        output
    }
}

impl<const WIDTH: usize> CryptographicHasher<u8, [u8; 32]> for Blake3Wide<WIDTH> {
    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        const BUFLEN: usize = 512;
        let mut hasher = blake3::Hasher::new();
        p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |buf| {
            hasher.update(buf);
        });
        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use p3_blake3::Blake3;

    use super::*;

    #[test]
    fn test_blake3wide() {
        const WIDTH: usize = 4;

        let random_inputs: [[u8; WIDTH]; 100] = rand::random();

        let mut expected_output = [[0u8; WIDTH]; 32];

        (0..WIDTH).for_each(|i| {
            let hasher = Blake3 {};

            let hasher_output = hasher.hash_iter(random_inputs.iter().map(|input| input[i]));

            for (j, byte) in hasher_output.into_iter().enumerate() {
                expected_output[j][i] = byte;
            }
        });

        let hasher_wide = Blake3Wide {};

        assert_eq!(hasher_wide.hash_iter(random_inputs), expected_output);
    }
}
