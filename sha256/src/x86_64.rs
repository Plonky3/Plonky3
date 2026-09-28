//! The batched backends of x86-64, picked at run time.
//!
//! Every x86-64 build compiles all of them, whatever its target features.
//!
//! The CPU is asked once, and the answer is cached.
//! A build that already enables a backend's features skips the question for it.

use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

use crate::{Sha256, Sha256Compress, x86_64_avx512, x86_64_sha_ni};

cpufeatures::new!(has_avx512, "avx512f", "avx512bw");
cpufeatures::new!(has_sha_ni, "sha", "sse4.1");

/// Messages the widest backend advances at once.
///
/// This is the AVX-512 width on every CPU, since a caller groups messages at build time.
/// A narrower backend runs a group of 32 as whole calls of its own, so nothing is wasted.
pub(crate) const LANES: usize = x86_64_avx512::LANES;

/// A batched backend that the running CPU supports.
///
/// Only the detection below builds one, so holding a value proves its instructions exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Backend(Kind);

/// The backends, from the fastest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// 32 lanes of AVX-512, and SHA-NI streams for the last few messages when `sha_ni` holds.
    Avx512 { sha_ni: bool },
    /// Four interleaved SHA-NI streams.
    ShaNi,
    /// The scalar hasher, one message at a time.
    Scalar,
}

impl Backend {
    /// The fastest backend of the running CPU.
    ///
    /// This is the one place that ranks the backends.
    ///
    /// AVX-512 comes first, as measured on Zen 5.
    /// A CPU where four SHA-NI streams beat it would only need this order changed.
    #[inline]
    pub(crate) fn detect() -> Self {
        let sha_ni = has_sha_ni::get();
        Self(if has_avx512::get() {
            Kind::Avx512 { sha_ni }
        } else if sha_ni {
            Kind::ShaNi
        } else {
            Kind::Scalar
        })
    }

    /// Every backend the running CPU supports, from the fastest.
    #[cfg(test)]
    pub(crate) fn supported() -> alloc::vec::Vec<Self> {
        let (avx512, sha_ni) = (has_avx512::get(), has_sha_ni::get());
        [
            (avx512 && sha_ni).then_some(Kind::Avx512 { sha_ni: true }),
            avx512.then_some(Kind::Avx512 { sha_ni: false }),
            sha_ni.then_some(Kind::ShaNi),
            Some(Kind::Scalar),
        ]
        .into_iter()
        .flatten()
        .map(Self)
        .collect()
    }

    /// Hash equal-length messages laid end to end in `input`.
    pub(crate) fn hash_many(self, input: &[u8], out: &mut [[u8; 32]]) {
        match self.0 {
            // SAFETY: the detection found AVX-512F and AVX-512BW, and SHA-NI when `sha_ni` holds.
            Kind::Avx512 { sha_ni } => unsafe { x86_64_avx512::hash_many(input, out, sha_ni) },
            // SAFETY: the detection found SHA-NI and SSE4.1.
            Kind::ShaNi => unsafe { x86_64_sha_ni::hash_many(input, out) },
            Kind::Scalar => scalar_hash_many(input, out),
        }
    }

    /// Compress each 64-byte pair from the initial hash value.
    pub(crate) fn compress_many(self, inputs: &[[[u8; 32]; 2]], out: &mut [[u8; 32]]) {
        match self.0 {
            // SAFETY: as in the hash.
            Kind::Avx512 { sha_ni } => unsafe { x86_64_avx512::compress_many(inputs, out, sha_ni) },
            // SAFETY: as in the hash.
            Kind::ShaNi => unsafe { x86_64_sha_ni::compress_many(inputs, out) },
            Kind::Scalar => scalar_compress_many(inputs, out),
        }
    }
}

/// Hash equal-length messages with the fastest backend of the running CPU.
#[inline]
pub(crate) fn hash_many(input: &[u8], out: &mut [[u8; 32]]) {
    Backend::detect().hash_many(input, out);
}

/// Compress 64-byte pairs with the fastest backend of the running CPU.
#[inline]
pub(crate) fn compress_many(inputs: &[[[u8; 32]; 2]], out: &mut [[u8; 32]]) {
    Backend::detect().compress_many(inputs, out);
}

/// Hash the messages one at a time.
///
/// # Panics
///
/// Panics if the input length is not a whole multiple of the digest count.
fn scalar_hash_many(input: &[u8], out: &mut [[u8; 32]]) {
    // No digests requested means there is nothing to read from the input.
    if out.is_empty() {
        return;
    }

    // Every message has the same length, so the split is exact by contract.
    assert!(
        input.len().is_multiple_of(out.len()),
        "input length ({}) must be a whole multiple of the digest count ({})",
        input.len(),
        out.len()
    );
    let len = input.len() / out.len();
    for (index, digest) in out.iter_mut().enumerate() {
        *digest = Sha256.hash_slice(&input[index * len..][..len]);
    }
}

/// Compress the pairs one at a time.
///
/// # Panics
///
/// Panics if the input and output counts differ.
fn scalar_compress_many(inputs: &[[[u8; 32]; 2]], out: &mut [[u8; 32]]) {
    assert_eq!(
        inputs.len(),
        out.len(),
        "group count ({}) must equal the output count ({})",
        inputs.len(),
        out.len()
    );
    for (input, digest) in inputs.iter().zip(out) {
        *digest = Sha256Compress.compress(*input);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::*;
    use crate::cavp::{COUNTS, LONG_MSG, SHORT_MSG};
    use crate::tests::{
        BATCH_COUNTS, SHAPE_LENGTHS, compression_inputs, random_bytes, reference,
        spec_compress_pair,
    };

    #[test]
    fn detection_picks_the_first_supported_backend() {
        // The ranking and the list of supported backends agree.
        assert_eq!(Backend::detect(), Backend::supported()[0]);

        // The scalar backend runs anywhere, so it always closes the list.
        assert_eq!(Backend::supported().last(), Some(&Backend(Kind::Scalar)));
    }

    #[test]
    fn every_backend_matches_every_cavp_vector() {
        for backend in Backend::supported() {
            for (message, expected) in SHORT_MSG.iter().chain(LONG_MSG) {
                for count in COUNTS {
                    // The same message in every lane.
                    let input = message.repeat(count);
                    let mut out = vec![[0u8; 32]; count];
                    backend.hash_many(&input, &mut out);

                    assert!(
                        out.iter().all(|digest| digest == expected),
                        "{backend:?}, {} bytes, {count} copies",
                        message.len()
                    );
                }
            }
        }
    }

    #[test]
    fn every_backend_matches_scalar_across_block_shapes() {
        for backend in Backend::supported() {
            for len in SHAPE_LENGTHS {
                for count in BATCH_COUNTS {
                    let messages =
                        random_bytes(len * count, ((len as u64) << 32) | count as u64 | 1);

                    let mut batched = vec![[0u8; 32]; count];
                    backend.hash_many(&messages, &mut batched);

                    assert_eq!(
                        batched,
                        reference(&messages, len, count),
                        "{backend:?}, len {len}, count {count}"
                    );
                }
            }
        }
    }

    #[test]
    fn every_backend_compresses_as_the_specification() {
        for backend in Backend::supported() {
            for count in BATCH_COUNTS {
                let inputs =
                    compression_inputs(&random_bytes(count * 64, 0x5851_f42d ^ count as u64));

                let mut batched = vec![[0u8; 32]; count];
                backend.compress_many(&inputs, &mut batched);

                let expected: Vec<[u8; 32]> = inputs
                    .iter()
                    .map(|input| spec_compress_pair(*input))
                    .collect();
                assert_eq!(batched, expected, "{backend:?}, count {count}");
            }
        }
    }

    proptest! {
        #[test]
        fn every_backend_matches_scalar_on_random_batches(
            len in 0usize..=300,
            count in 1usize..=80,
            seed in any::<u64>(),
        ) {
            let messages = random_bytes(len * count, seed | 1);
            let expected = reference(&messages, len, count);

            for backend in Backend::supported() {
                let mut batched = vec![[0u8; 32]; count];
                backend.hash_many(&messages, &mut batched);
                prop_assert_eq!(&batched, &expected, "{:?}", backend);
            }
        }
    }
}
