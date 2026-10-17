//! Sequential batched hashing for the pruned verifier's fragmented opening rows.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::PackedValue;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

use super::MerkleTreeMmcs;
use crate::builder::rows_per_call;

impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize>
    MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>
where
    P: PackedValue,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>,
{
    /// Hash equal-width, already validated rows, concatenating matrices in their tree order.
    /// Batch hashers stage a bounded run; packed hashers gather columns directly from rows.
    pub(super) fn hash_opened_rows<R: AsRef<[P::Value]>>(
        &self,
        matrices: &[usize],
        representatives: &[usize],
        openings: &[Vec<R>],
    ) -> Vec<[PW::Value; DIGEST_ELEMS]> {
        let mut out = vec![[PW::Value::default(); DIGEST_ELEMS]; representatives.len()];
        let lanes = <H as CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>>::LANES;
        if lanes > 1 {
            let width: usize = matrices
                .iter()
                .map(|&m| openings[representatives[0]][m].as_ref().len())
                .sum();
            let rows_per_group = rows_per_call(width * size_of::<P::Value>(), lanes);
            let mut rows = Vec::new();
            for (reps, digests) in representatives
                .chunks(rows_per_group)
                .zip(out.chunks_mut(rows_per_group))
            {
                rows.clear();
                for &rep in reps {
                    for &m in matrices {
                        rows.extend_from_slice(openings[rep][m].as_ref());
                    }
                }
                self.hash.hash_many(&rows, digests);
            }
            return out;
        }

        // A scalar digest packing cannot represent all the leaf lanes. Such a configuration
        // still has a well-defined verifier and uses the scalar path, as before.
        let packed_len = if P::WIDTH == PW::WIDTH && P::WIDTH > 1 {
            representatives.len() / P::WIDTH * P::WIDTH
        } else {
            0
        };
        for (reps, digests) in representatives[..packed_len]
            .chunks_exact(P::WIDTH)
            .zip(out[..packed_len].chunks_exact_mut(P::WIDTH))
        {
            let input = matrices.iter().flat_map(|&m| {
                let width = openings[reps[0]][m].as_ref().len();
                (0..width).map(move |col| P::from_fn(|lane| openings[reps[lane]][m].as_ref()[col]))
            });
            let packed = self.hash.hash_iter(input);
            PW::unpack_into(&packed, digests);
        }
        for (&rep, digest) in representatives[packed_len..]
            .iter()
            .zip(&mut out[packed_len..])
        {
            *digest = self
                .hash
                .hash_iter_slices(matrices.iter().map(|&m| openings[rep][m].as_ref()));
        }
        out
    }

    /// Compress independent parents (including binary bridges and injection pairs padded to N).
    pub(super) fn compress_opened_groups(
        &self,
        inputs: &[[[PW::Value; DIGEST_ELEMS]; N]],
        out: &mut [[PW::Value; DIGEST_ELEMS]],
    ) {
        debug_assert_eq!(inputs.len(), out.len());
        if <C as PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>>::LANES > 1 {
            self.compress.compress_many(inputs, out);
            return;
        }
        let packed_len = if PW::WIDTH > 1 {
            inputs.len() / PW::WIDTH * PW::WIDTH
        } else {
            0
        };
        for (groups, digests) in inputs[..packed_len]
            .chunks_exact(PW::WIDTH)
            .zip(out[..packed_len].chunks_exact_mut(PW::WIDTH))
        {
            let packed =
                core::array::from_fn(|child| PW::pack_columns_fn(|lane| groups[lane][child]));
            let digest = self.compress.compress(packed);
            PW::unpack_into(&digest, digests);
        }
        for (&input, digest) in inputs[packed_len..].iter().zip(&mut out[packed_len..]) {
            *digest = self.compress.compress(input);
        }
    }
}
