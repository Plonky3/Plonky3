//! SHA3-256 against the NIST CAVP byte-oriented test vectors.
//!
//! The vectors come from the SHA-3 validation suite of NIST:
//!
//! <https://csrc.nist.gov/projects/cryptographic-algorithm-validation-program/secure-hashing>
//!
//! - Short messages: every length from 0 bytes up to one full rate block of 136.
//! - Long messages: the first eight vectors, several blocks each, every one ending mid-block.
//! - Monte Carlo: 100 checkpoints of a chain of 100 000 hashes of 32-byte messages.

use alloc::vec;
use alloc::vec::Vec;

use p3_symmetric::CryptographicHasher;

use crate::Sha3_256Hash;
use crate::batch::{LANES, SHA3_DOMAIN, supported};

/// One known-answer vector: a message and its expected digest.
struct Vector {
    /// Message bytes.
    msg: Vec<u8>,
    /// Expected SHA3-256 digest.
    md: [u8; 32],
}

/// Decode a string of hexadecimal digit pairs.
fn decode_hex(hex: &str) -> Vec<u8> {
    // Two digits make one byte, high nibble first.
    let (pairs, rest) = hex.as_bytes().as_chunks::<2>();
    assert!(rest.is_empty(), "odd number of hex digits");
    pairs
        .iter()
        .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

/// Decode a 32-byte digest.
fn decode_digest(hex: &str) -> [u8; 32] {
    decode_hex(hex).try_into().unwrap()
}

/// Read the value of every `key = value` line with the given key, in file order.
fn values<'a>(rsp: &'a str, key: &'a str) -> impl Iterator<Item = &'a str> {
    rsp.lines().filter_map(move |line| {
        let (k, v) = line.split_once(" = ")?;
        (k.trim() == key).then(|| v.trim())
    })
}

/// Parse a message file into its vectors.
fn parse_messages(rsp: &str) -> Vec<Vector> {
    // Each vector is a `Len`, `Msg`, `MD` triple.
    // `Len` counts bits, and the empty message is written as a single `00` byte.
    values(rsp, "Len")
        .zip(values(rsp, "Msg"))
        .zip(values(rsp, "MD"))
        .map(|((len, msg), md)| {
            let bytes: usize = len.parse::<usize>().unwrap() / 8;
            let mut msg = decode_hex(msg);
            msg.truncate(bytes);
            Vector {
                msg,
                md: decode_digest(md),
            }
        })
        .collect()
}

/// Check one vector through the one-message path and the batched path.
fn check(vector: &Vector) {
    let len = vector.msg.len();

    // One message at a time.
    assert_eq!(Sha3_256Hash.hash_slice(&vector.msg), vector.md, "len {len}");

    // A batch of one leaves every lane but the first as padding.
    let mut one = [[0u8; 32]; 1];
    Sha3_256Hash.hash_many(&vector.msg, &mut one);
    assert_eq!(one[0], vector.md, "len {len}, batch of 1");

    // A batch one message past a full group of the widest backend fills every lane.
    // It then leaves a short group:
    //
    //     lanes:  [ m | m | ... | m ]  [ m | pad ... ]
    //              LANES                 1
    let count = LANES + 1;
    let batch = vector.msg.repeat(count);
    let mut digests = vec![[0u8; 32]; count];
    Sha3_256Hash.hash_many(&batch, &mut digests);
    assert!(
        digests.iter().all(|d| *d == vector.md),
        "len {len}, batch of {count}"
    );

    // The same batch on every backend the CPU supports.
    for kernel in supported() {
        let mut digests = vec![[0u8; 32]; count];
        kernel.hash_many(SHA3_DOMAIN, &batch, len, &mut digests);
        assert!(
            digests.iter().all(|d| *d == vector.md),
            "{kernel:?}, len {len}, batch of {count}"
        );
    }
}

#[test]
fn short_messages() {
    let vectors = parse_messages(include_str!("../testdata/SHA3_256ShortMsg.rsp"));

    // 137 vectors: lengths 0, 1, ..., 136 bytes.
    assert_eq!(vectors.len(), 137);
    vectors.iter().for_each(check);
}

#[test]
fn long_messages() {
    let vectors = parse_messages(include_str!("../testdata/SHA3_256LongMsg.rsp"));

    // 8 vectors: 273 bytes, then 137 more each time, up to 1232 bytes.
    assert_eq!(vectors.len(), 8);
    vectors.iter().for_each(check);
}

#[test]
fn monte_carlo() {
    let rsp = include_str!("../testdata/SHA3_256Monte.rsp");
    let checkpoints: Vec<[u8; 32]> = values(rsp, "MD").map(decode_digest).collect();
    assert_eq!(checkpoints.len(), 100);

    // The chain starts from the seed.
    // Each step hashes the previous 32-byte digest:
    //
    //     md_0 = seed
    //     md_i = SHA3-256(md_{i-1})
    //
    // Every 1000th digest is a checkpoint, and the next run of 1000 starts from it.
    let mut md = decode_digest(values(rsp, "Seed").next().unwrap());
    for expected in checkpoints {
        for _ in 0..1000 {
            // The batched path, one lane in use.
            let mut out = [[0u8; 32]; 1];
            Sha3_256Hash.hash_many(&md, &mut out);
            md = out[0];
        }
        assert_eq!(md, expected);
    }
}
