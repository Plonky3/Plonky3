//! Every test pins this implementation to something outside it: RFC 7693's own vectors, or
//! RustCrypto's `blake2`.

use alloc::vec;
use alloc::vec::Vec;

use blake2::digest::consts::U32;
use blake2::{Blake2s, Digest};
use hex_literal::hex;
use p3_symmetric::CryptographicHasher;
use proptest::prelude::*;

use crate::{Blake2s256, DIGEST_BYTES, LANES};

/// BLAKE2s-256 of "abc", from RFC 7693 appendix B.
const ABC_DIGEST: [u8; DIGEST_BYTES] =
    hex!("508C5E8C327C14E2E1A72BA34EEB452F37458B209ED63A294D999B4C86675982");

/// The digest RustCrypto produces, which is the reference every comparison uses.
fn reference(message: &[u8]) -> [u8; DIGEST_BYTES] {
    Blake2s::<U32>::digest(message).into()
}

/// A deterministic byte stream, so a failure is the same failure next run.
fn fixture(bytes: usize) -> Vec<u8> {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    (0..bytes)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

#[test]
fn the_rfc_7693_vector_matches() {
    assert_eq!(Blake2s256::hash(b"abc"), ABC_DIGEST);
}

#[test]
fn single_messages_match_the_blake2_crate() {
    // Lengths that bracket every boundary: empty, inside one block, exactly one block, just
    // past it, and several blocks with and without a tail.
    for len in [0, 1, 55, 63, 64, 65, 127, 128, 129, 256, 1000] {
        let message = fixture(len);
        assert_eq!(
            Blake2s256::hash(&message),
            reference(&message),
            "message of {len} bytes"
        );
    }
}

#[test]
fn a_message_fed_in_pieces_hashes_the_same() {
    // The last block is compressed differently from every other, so a split that lands on a
    // block boundary is the one that would break a hasher that compresses too early.
    let message = fixture(200);
    for split in [0, 1, 63, 64, 65, 128, 199, 200] {
        let (head, tail) = message.split_at(split);
        assert_eq!(
            Blake2s256.hash_iter_slices([head, tail]),
            reference(&message),
            "split at {split}"
        );
    }
}

#[test]
fn hash_iter_matches_the_contiguous_message() {
    for len in [0, 64, 513] {
        let message = fixture(len);
        assert_eq!(
            Blake2s256.hash_iter(message.iter().copied()),
            reference(&message),
            "message of {len} bytes"
        );
    }
}

#[test]
fn batches_match_the_blake2_crate() {
    // Counts that fill a group exactly, leave a remainder, and fall short of one.
    for count in [1, 3, LANES, LANES + 1, 2 * LANES, 2 * LANES + 5] {
        for len in [0, 32, 64, 65, 200] {
            let messages = fixture(len * count);
            let mut digests = vec![[0u8; DIGEST_BYTES]; count];
            Blake2s256.hash_many(&messages, &mut digests);

            for (digest, message) in digests.iter().zip(messages.chunks_exact(len.max(1))) {
                let expected = reference(if len == 0 { &[] } else { message });
                assert_eq!(digest, &expected, "{count} messages of {len} bytes");
            }
        }
    }
}

#[test]
fn a_batch_agrees_with_hashing_one_at_a_time() {
    let count = 2 * LANES + 3;
    let len = 137;
    let messages = fixture(len * count);
    let mut digests = vec![[0u8; DIGEST_BYTES]; count];
    Blake2s256.hash_many(&messages, &mut digests);

    for (digest, message) in digests.iter().zip(messages.chunks_exact(len)) {
        assert_eq!(digest, &Blake2s256::hash(message));
    }
}

#[test]
fn an_empty_batch_reads_nothing() {
    let mut digests: [[u8; DIGEST_BYTES]; 0] = [];
    Blake2s256.hash_many(&[], &mut digests);
}

#[test]
#[should_panic(expected = "whole multiple")]
fn a_ragged_batch_is_rejected() {
    let mut digests = vec![[0u8; DIGEST_BYTES]; 3];
    Blake2s256.hash_many(&fixture(10), &mut digests);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn any_message_matches_the_blake2_crate(message in prop::collection::vec(any::<u8>(), 0..300)) {
        prop_assert_eq!(Blake2s256::hash(&message), reference(&message));
    }

    #[test]
    fn any_equal_length_batch_matches_one_at_a_time(
        len in 0usize..200,
        count in 1usize..20,
    ) {
        let messages = fixture(len * count);
        let mut digests = vec![[0u8; DIGEST_BYTES]; count];
        Blake2s256.hash_many(&messages, &mut digests);
        for (digest, message) in digests.iter().zip(messages.chunks(len.max(1))) {
            let expected = if len == 0 { Blake2s256::hash(&[]) } else { Blake2s256::hash(message) };
            prop_assert_eq!(digest, &expected);
        }
    }
}
