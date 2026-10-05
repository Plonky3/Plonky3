//! The batched backends of x86-64, picked at run time.
//!
//! Every x86-64 build compiles all of them, whatever its target features.
//!
//! The CPU is asked once, and the answer is cached.
//! A build that already enables a backend's features skips the question for it.

use core::arch::x86_64::__cpuid;
use core::sync::atomic::{AtomicU8, Ordering};

use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

use crate::x86_64_avx512::Streams;
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
    /// 32 lanes of AVX-512, sharing the batch with SHA-NI streams as `streams` says.
    Avx512 { streams: Streams },
    /// Four interleaved SHA-NI streams.
    ShaNi,
    /// The scalar hasher, one message at a time.
    Scalar,
}

/// The CPU designs whose crossovers between AVX-512 and SHA-NI were measured apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Design {
    /// Any Intel CPU.
    Intel = 1,
    /// AMD family 0x19 with AVX-512, which is Zen 4, since Zen 3 has no AVX-512.
    Zen4 = 2,
    /// Any other CPU, Zen 5 included.
    Other = 3,
}

impl Design {
    /// The design of the running CPU, read from `cpuid` once and then cached.
    fn get() -> Self {
        // Zero means not read yet, and every design has a non-zero discriminant.
        static CACHE: AtomicU8 = AtomicU8::new(0);

        match CACHE.load(Ordering::Relaxed) {
            1 => return Self::Intel,
            2 => return Self::Zen4,
            3 => return Self::Other,
            _ => {}
        }

        // Leaf 0 spells the vendor across EBX, EDX and ECX, and leaf 1 holds the family in EAX.
        //
        // Every x86-64 CPU has both leaves.
        let vendor = __cpuid(0);
        let signature = __cpuid(1).eax;
        let design = Self::classify([vendor.ebx, vendor.edx, vendor.ecx], signature);

        // Racing threads compute the same answer, so a plain store is enough.
        CACHE.store(design as u8, Ordering::Relaxed);
        design
    }

    /// The design of a CPU from its `cpuid` vendor words and leaf 1 signature.
    ///
    /// It only matters on a CPU with AVX-512, so it ignores the families that have none.
    const fn classify(vendor: [u32; 3], signature: u32) -> Self {
        // "GenuineIntel" and "AuthenticAMD", as the little-endian words of EBX, EDX, ECX.
        const INTEL: [u32; 3] = [0x756e_6547, 0x4965_6e69, 0x6c65_746e];
        const AMD: [u32; 3] = [0x6874_7541, 0x6974_6e65, 0x444d_4163];

        // The family is the base field, plus the extended field when the base one saturates.
        let base = (signature >> 8) & 0xf;
        let family = if base == 0xf {
            base + ((signature >> 20) & 0xff)
        } else {
            base
        };

        if vendor[0] == INTEL[0] && vendor[1] == INTEL[1] && vendor[2] == INTEL[2] {
            Self::Intel
        } else if vendor[0] == AMD[0]
            && vendor[1] == AMD[1]
            && vendor[2] == AMD[2]
            && family == 0x19
        {
            Self::Zen4
        } else {
            Self::Other
        }
    }

    /// How a CPU of this design with AVX-512 and SHA-NI shares a batch between them.
    const fn streams(self) -> Streams {
        match self {
            Self::Intel => Streams::FromFour,
            Self::Zen4 => Streams::Zen4,
            Self::Other => Streams::FromThree,
        }
    }
}

impl Backend {
    /// The fastest backend of the running CPU.
    ///
    /// This is the one place that ranks the backends.
    ///
    /// AVX-512 comes first on every CPU that has it, for the register passes on short messages.
    /// Where SHA-NI beats it, the CPU's design hands that share of the batch to the streams.
    #[inline]
    pub(crate) fn detect() -> Self {
        let sha_ni = has_sha_ni::get();
        Self(if has_avx512::get() {
            let streams = if sha_ni {
                Design::get().streams()
            } else {
                Streams::Off
            };
            Kind::Avx512 { streams }
        } else if sha_ni {
            Kind::ShaNi
        } else {
            Kind::Scalar
        })
    }

    /// Every backend the running CPU supports, from the fastest.
    ///
    /// A CPU with SHA-NI runs the stream policy of every design, so the tests cover them all.
    #[cfg(test)]
    pub(crate) fn supported() -> alloc::vec::Vec<Self> {
        let (avx512, sha_ni) = (has_avx512::get(), has_sha_ni::get());
        let policies = [Streams::FromThree, Streams::FromFour, Streams::Zen4];
        policies
            .into_iter()
            .filter(|_| avx512 && sha_ni)
            .map(|streams| Kind::Avx512 { streams })
            .chain(avx512.then_some(Kind::Avx512 {
                streams: Streams::Off,
            }))
            .chain(sha_ni.then_some(Kind::ShaNi))
            .chain([Kind::Scalar])
            .map(Self)
            .collect()
    }

    /// Hash equal-length messages laid end to end in `input`.
    pub(crate) fn hash_many(self, input: &[u8], out: &mut [[u8; 32]]) {
        match self.0 {
            // SAFETY: the detection found AVX-512F and AVX-512BW, and SHA-NI unless `streams` is off.
            Kind::Avx512 { streams } => unsafe { x86_64_avx512::hash_many(input, out, streams) },
            // SAFETY: the detection found SHA-NI and SSE4.1.
            Kind::ShaNi => unsafe { x86_64_sha_ni::hash_many(input, out) },
            Kind::Scalar => scalar_hash_many(input, out),
        }
    }

    /// Compress each 64-byte pair from the initial hash value.
    pub(crate) fn compress_many(self, inputs: &[[[u8; 32]; 2]], out: &mut [[u8; 32]]) {
        match self.0 {
            // SAFETY: as in the hash.
            Kind::Avx512 { streams } => unsafe {
                x86_64_avx512::compress_many(inputs, out, streams);
            },
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
    fn detection_picks_a_supported_backend_of_the_fastest_kind() {
        // The ranking and the list of supported backends agree on the kind.
        let detected = Backend::detect();
        let supported = Backend::supported();
        assert!(supported.contains(&detected), "{detected:?}");
        assert_eq!(
            core::mem::discriminant(&detected.0),
            core::mem::discriminant(&supported[0].0)
        );

        // The scalar backend runs anywhere, so it always closes the list.
        assert_eq!(Backend::supported().last(), Some(&Backend(Kind::Scalar)));
    }

    #[test]
    fn each_design_is_read_from_its_cpuid_signature() {
        // Vendor words of EBX, EDX and ECX.
        let intel = *b"GenuineIntel";
        let amd = *b"AuthenticAMD";
        let words = |v: [u8; 12]| {
            let (chunks, _) = v.as_chunks::<4>();
            [0, 1, 2].map(|i| u32::from_le_bytes(chunks[i]))
        };

        // Sapphire Rapids is family 6, Zen 4 family 0x19 and Zen 5 family 0x1a.
        assert_eq!(Design::classify(words(intel), 0x0008_06f8), Design::Intel);
        assert_eq!(Design::classify(words(amd), 0x00a6_0f12), Design::Zen4);
        assert_eq!(Design::classify(words(amd), 0x00a1_0f11), Design::Zen4);
        assert_eq!(Design::classify(words(amd), 0x00b4_0f40), Design::Other);

        // An unknown vendor keeps the default crossovers whatever its family.
        assert_eq!(
            Design::classify(words(*b"HygonGenuine"), 0x00a6_0f12),
            Design::Other
        );

        // Each design maps to its own policy.
        assert_eq!(Design::Intel.streams(), Streams::FromFour);
        assert_eq!(Design::Zen4.streams(), Streams::Zen4);
        assert_eq!(Design::Other.streams(), Streams::FromThree);

        // The cached read agrees with a fresh one.
        assert_eq!(Design::get(), Design::get());
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
    fn every_backend_matches_scalar_on_long_messages() {
        // Lengths on each side of the cut where Zen 4 hands whole batches to the streams.
        //
        // Counts cover whole groups, the leftovers each policy routes apart, and a lone message.
        for backend in Backend::supported() {
            for len in [2047, 2048, 4096] {
                for count in [1, 3, 13, 24, 40, 64] {
                    let messages = random_bytes(len * count, (len as u64) << 8 | count as u64);

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
