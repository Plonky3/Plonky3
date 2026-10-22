//! Hash gadget tests: official vectors, outputs against reference implementations, witness generators against each other, and the rows of the relation.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;

use blake2::digest::consts::U32;
use blake2::{Blake2s, Digest};
use p3_keccak::Keccak256Hash;
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

use super::Gadget;
use super::arx::{G_LANES, IV};
use super::blake2s::{Blake2sCompress, SIGMA};
use super::blake3::{Blake3Compress, MSG_PERMUTATION};
use super::keccak::KeccakF1600;
use super::sha256::Sha256Compress;
use crate::tests::satisfies;
use crate::zerocheck::MIN_LOG_N;
use crate::{
    Batch, Builder, Circuit, FlockChallenger, FlockConfig, Port, PublicPorts, Witness, prove,
    verify,
};

/// Instances per batch: two full 64-lane walks.
const LOG_N: usize = 7;

/// One instance's output, by the one-instance walk.
fn run<G: Gadget>(circuit: &Circuit, input: &G) -> G::Output {
    let words = (1 << circuit.k_log()) / 64;
    let (mut z, mut az, mut bz) = (vec![0; words], vec![0; words], vec![0; words]);
    circuit.witness_instance(&input.input_words(), &mut z, &mut az, &mut bz);
    G::decode_output(circuit.output_words(&z))
}

/// Check a batch against a reference: every output, and the two witness generators against each other.
fn check_batch<G>(rows: &[G], padding: &G, reference: impl Fn(&G) -> G::Output)
where
    G: Gadget + Sync,
    G::Output: PartialEq + Debug,
{
    let circuit = G::circuit();
    let words = (1 << circuit.k_log()) / 64;
    let w = Witness::generate(&circuit, LOG_N, rows, padding, G::write_input);
    assert!(
        G::witness(&circuit, LOG_N, rows, padding) == w,
        "the gadget's witness is the walk's"
    );
    for i in 0..1 << LOG_N {
        let row = rows.get(i).unwrap_or(padding);
        let range = i * words..(i + 1) * words;
        let out = G::decode_output(circuit.output_words(&w.z()[range.clone()]));
        assert_eq!(out, reference(row), "instance {i}");

        // The one-instance walk on lanes at both ends of each 64-lane group.
        if i % 64 == 0 || i % 64 == 63 {
            let (mut z, mut az, mut bz) = (vec![0; words], vec![0; words], vec![0; words]);
            circuit.witness_instance(&row.input_words(), &mut z, &mut az, &mut bz);
            assert_eq!(z, w.z()[range.clone()], "z of instance {i}");
            assert_eq!(az, w.az()[range.clone()], "az of instance {i}");
            assert_eq!(bz, w.bz()[range], "bz of instance {i}");
        }
    }
}

/// The rows of two instances accept their honest witness and reject every single-bit tamper tried.
fn check_rows<G: Gadget + Sync>(rows: &[G; 2]) {
    let circuit = G::circuit();
    let k = 1 << circuit.k_log();
    let (input_slot, output_slot) = (5, 64 * circuit.n_input_words() + 3);
    let (const_pos, last) = (circuit.const_pos(), circuit.useful_bits() - 1);
    let w = Witness::generate(&circuit, 1, rows, &rows[0], G::write_input);
    assert!(satisfies(&circuit, w.z()));

    // Flip the first and the last product, an output bit, an input bit, the constant, a padding bit.
    for slot in [
        const_pos + 1,
        last,
        output_slot,
        input_slot,
        const_pos,
        k - 1,
    ] {
        for instance in 0..2 {
            let mut bad = w.z().to_vec();
            let bit = instance * k + slot;
            bad[bit / 64] ^= 1 << (bit % 64);
            assert!(
                !satisfies(&circuit, &bad),
                "slot {slot} of instance {instance}"
            );
        }
    }
}

/// Little-endian words of a zero-padded block of at most 64 bytes.
fn le_block(bytes: &[u8]) -> [u32; 16] {
    let mut padded = [0u8; 64];
    padded[..bytes.len()].copy_from_slice(bytes);
    core::array::from_fn(|i| u32::from_le_bytes(padded[4 * i..4 * i + 4].try_into().unwrap()))
}

/// A fixed number of bytes from a hexadecimal string.
fn hex_array<const N: usize>(s: &str) -> [u8; N] {
    hex(s).try_into().unwrap()
}

/// Bytes from a hexadecimal string.
fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// Native BLAKE3 compression.
fn blake3_reference(x: &Blake3Compress) -> [u32; 16] {
    let mut v = [0u32; 16];
    v[..8].copy_from_slice(&x.cv);
    v[8..12].copy_from_slice(&IV[..4]);
    v[12..].copy_from_slice(&[
        x.counter as u32,
        (x.counter >> 32) as u32,
        x.block_len,
        x.flags,
    ]);
    let mut m = x.block;
    for _ in 0..7 {
        for (i, &lanes) in G_LANES.iter().enumerate() {
            g_reference(&mut v, lanes, m[2 * i], m[2 * i + 1]);
        }
        m = core::array::from_fn(|i| m[MSG_PERMUTATION[i]]);
    }
    core::array::from_fn(|i| {
        if i < 8 {
            v[i] ^ v[i + 8]
        } else {
            v[i] ^ x.cv[i - 8]
        }
    })
}

/// Native BLAKE2s compression.
fn blake2s_reference(x: &Blake2sCompress) -> [u32; 8] {
    let mut v = [0u32; 16];
    v[..8].copy_from_slice(&x.h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= x.t as u32;
    v[13] ^= (x.t >> 32) as u32;
    v[14] ^= x.f0;
    for sigma in &SIGMA {
        for (i, &lanes) in G_LANES.iter().enumerate() {
            g_reference(&mut v, lanes, x.m[sigma[2 * i]], x.m[sigma[2 * i + 1]]);
        }
    }
    core::array::from_fn(|i| x.h[i] ^ v[i] ^ v[i + 8])
}

/// The G shared by BLAKE2s and BLAKE3.
fn g_reference(v: &mut [u32; 16], [a, b, c, d]: [usize; 4], x: u32, y: u32) {
    for (m, r1, r2) in [(x, 16, 12), (y, 8, 7)] {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(m);
        v[d] = (v[d] ^ v[a]).rotate_right(r1);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(r2);
    }
}

/// SHA-256 compression by the `sha2` crate.
fn sha256_reference(x: &Sha256Compress) -> [u32; 8] {
    let mut state = x.state;
    let mut block = [0u8; 64];
    for (bytes, word) in block.as_chunks_mut::<4>().0.iter_mut().zip(x.block) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    sha2::block_api::compress256(&mut state, &[block]);
    state
}

/// Keccak-f\[1600\] by `tiny-keccak`.
fn keccak_reference(x: &KeccakF1600) -> [u64; 25] {
    let mut state = x.0;
    tiny_keccak::keccakf(&mut state);
    state
}

/// Random BLAKE3 inputs, some with edge counters and flags.
fn random_blake3(rng: &mut SmallRng) -> Blake3Compress {
    Blake3Compress {
        cv: rng.random(),
        block: rng.random(),
        counter: [0, u64::MAX, rng.next_u64()][rng.random_range(0..3)],
        block_len: [0, 64, rng.next_u32()][rng.random_range(0..3)],
        flags: [11, u32::MAX, rng.next_u32()][rng.random_range(0..3)],
    }
}

/// Random BLAKE2s inputs, some with edge counters and both finalization words.
fn random_blake2s(rng: &mut SmallRng) -> Blake2sCompress {
    Blake2sCompress {
        h: rng.random(),
        m: rng.random(),
        t: [0, u64::MAX, rng.next_u64()][rng.random_range(0..3)],
        f0: [0, u32::MAX, rng.next_u32()][rng.random_range(0..3)],
    }
}

/// Random SHA-256 inputs.
fn random_sha256(rng: &mut SmallRng) -> Sha256Compress {
    Sha256Compress {
        state: rng.random(),
        block: [[0; 16], [u32::MAX; 16], rng.random()][rng.random_range(0..3)],
    }
}

/// Random Keccak states.
fn random_keccak(rng: &mut SmallRng) -> KeccakF1600 {
    KeccakF1600(core::array::from_fn(|_| rng.next_u64()))
}

#[test]
fn blake3_matches_the_reference_on_a_batch() {
    // Fixture state: 128 instances, the last 28 taking the padding row.
    let mut rng = SmallRng::seed_from_u64(0xB3);
    let rows: Vec<_> = (0..100).map(|_| random_blake3(&mut rng)).collect();
    check_batch(&rows, &random_blake3(&mut rng), blake3_reference);
}

#[test]
fn blake2s_matches_the_reference_on_a_batch() {
    let mut rng = SmallRng::seed_from_u64(0xB2);
    let rows: Vec<_> = (0..100).map(|_| random_blake2s(&mut rng)).collect();
    check_batch(&rows, &random_blake2s(&mut rng), blake2s_reference);
}

#[test]
fn sha256_matches_the_reference_on_a_batch() {
    let mut rng = SmallRng::seed_from_u64(0x256);
    let rows: Vec<_> = (0..100).map(|_| random_sha256(&mut rng)).collect();
    check_batch(&rows, &random_sha256(&mut rng), sha256_reference);
}

#[test]
fn keccak_matches_the_reference_on_a_batch() {
    let mut rng = SmallRng::seed_from_u64(0x1600);
    let rows: Vec<_> = (0..100).map(|_| random_keccak(&mut rng)).collect();
    check_batch(&rows, &KeccakF1600::default(), keccak_reference);
}

/// BLAKE3 domain flags (BLAKE3 specification, Table 3).
const CHUNK_START: u32 = 1;
/// End of a chunk.
const CHUNK_END: u32 = 2;
/// A parent node.
const PARENT: u32 = 4;
/// The root node.
const ROOT: u32 = 8;

/// Bytes of a BLAKE3 chunk.
const CHUNK_LEN: usize = 1024;

/// Big-endian words of one 64-byte block.
fn be_block(bytes: &[u8]) -> [u32; 16] {
    core::array::from_fn(|i| u32::from_be_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()))
}

/// BLAKE2s with an optional key and an output of `outlen` bytes (RFC 7693, Section 3.3), one compression per block.
fn blake2s_hash(c: &Circuit, key: &[u8], msg: &[u8], outlen: usize) -> Vec<u8> {
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ ((key.len() as u32) << 8) ^ outlen as u32;
    // A key is padded to a whole block in front of the message.
    let mut data = Vec::new();
    if !key.is_empty() {
        data.extend_from_slice(key);
        data.resize(64, 0);
    }
    data.extend_from_slice(msg);
    let blocks = data.len().div_ceil(64).max(1);
    for i in 0..blocks {
        let last = i + 1 == blocks;
        let block = &data[(64 * i).min(data.len())..(64 * i + 64).min(data.len())];
        let x = Blake2sCompress {
            h,
            m: le_block(block),
            t: if last { data.len() } else { 64 * (i + 1) } as u64,
            f0: if last { u32::MAX } else { 0 },
        };
        h = run(c, &x);
    }
    h.iter()
        .flat_map(|w| w.to_le_bytes())
        .take(outlen)
        .collect()
}

/// The deterministic test sequence of RFC 7693, Appendix E.
fn selftest_seq(len: usize, seed: u32) -> Vec<u8> {
    let (mut a, mut b) = (0xDEAD_4BAD_u32.wrapping_mul(seed), 1u32);
    (0..len)
        .map(|_| {
            let t = a.wrapping_add(b);
            (a, b) = (b, t);
            (t >> 24) as u8
        })
        .collect()
}

/// The chaining value, or the root output, of one BLAKE3 chunk (BLAKE3 specification, Section 2.4).
fn blake3_chunk(c: &Circuit, chunk: &[u8], counter: u64, root: bool) -> [u32; 16] {
    let blocks: Vec<&[u8]> = if chunk.is_empty() {
        vec![chunk]
    } else {
        chunk.chunks(64).collect()
    };
    let mut cv = IV;
    let mut out = [0; 16];
    for (j, block) in blocks.iter().enumerate() {
        let last = j + 1 == blocks.len();
        let flags = if j == 0 { CHUNK_START } else { 0 }
            | if last { CHUNK_END } else { 0 }
            | if last && root { ROOT } else { 0 };
        out = run(
            c,
            &Blake3Compress {
                cv,
                block: le_block(block),
                counter,
                block_len: block.len() as u32,
                flags,
            },
        );
        cv.copy_from_slice(&out[..8]);
    }
    out
}

/// BLAKE3 of at most two chunks: one chunk is the root, two meet in a root parent (Section 2.5).
fn blake3_hash(c: &Circuit, msg: &[u8]) -> Vec<u8> {
    assert!(msg.len() <= 2 * CHUNK_LEN, "at most two chunks");
    let out = if msg.len() <= CHUNK_LEN {
        blake3_chunk(c, msg, 0, true)
    } else {
        let left = blake3_chunk(c, &msg[..CHUNK_LEN], 0, false);
        let right = blake3_chunk(c, &msg[CHUNK_LEN..], 1, false);
        let mut block = [0; 16];
        block[..8].copy_from_slice(&left[..8]);
        block[8..].copy_from_slice(&right[..8]);
        run(
            c,
            &Blake3Compress {
                cv: IV,
                block,
                counter: 0,
                block_len: 64,
                flags: PARENT | ROOT,
            },
        )
    };
    out[..8].iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// SHA-256 by chained compressions of the padded message (FIPS 180-4, Sections 5.1.1 and 6.2), from the IV of Section 5.3.3.
fn sha256_hash(c: &Circuit, msg: &[u8]) -> Vec<u8> {
    let mut padded = msg.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&(8 * msg.len() as u64).to_be_bytes());
    let mut state = IV;
    for block in padded.chunks(64) {
        state = run(
            c,
            &Sha256Compress {
                state,
                block: be_block(block),
            },
        );
    }
    state.iter().flat_map(|w| w.to_be_bytes()).collect()
}

#[test]
fn blake2s_matches_rfc_7693() {
    let c = Blake2sCompress::circuit();

    // RFC 7693, Appendix B: BLAKE2s-256("abc").
    assert_eq!(
        blake2s_hash(&c, &[], b"abc", 32),
        hex("508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982")
    );

    // RFC 7693, Appendix E: the self-test's hash of keyed and unkeyed hashes of every parameter set.
    let mut hashes = Vec::new();
    for outlen in [16, 20, 28, 32] {
        for inlen in [0, 3, 64, 65, 255, 1024] {
            let msg = selftest_seq(inlen, inlen as u32);
            hashes.extend(blake2s_hash(&c, &[], &msg, outlen));
            let key = selftest_seq(outlen, outlen as u32);
            hashes.extend(blake2s_hash(&c, &key, &msg, outlen));
        }
    }
    assert_eq!(
        blake2s_hash(&c, &[], &hashes, 32),
        hex("6a411f08ce25adcdfb02aba641451cec53c598b24f4fc787fbdc88797f4c1dfe")
    );

    // One and two blocks, against the crate.
    let mut rng = SmallRng::seed_from_u64(0xB2B2);
    for len in [1, 63, 64, 65, 100, 128] {
        let msg: Vec<u8> = (0..len).map(|_| rng.random()).collect();
        let expected = Blake2s::<U32>::digest(&msg);
        assert_eq!(
            blake2s_hash(&c, &[], &msg, 32),
            expected.as_slice(),
            "length {len}"
        );
    }
}

#[test]
fn blake3_matches_the_official_test_vectors() {
    // BLAKE3 repository, test_vectors/test_vectors.json: the first 32 bytes of `hash`, input byte i being i mod 251.
    let vectors = [
        (
            0,
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
        ),
        (
            1,
            "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213",
        ),
        (
            63,
            "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b",
        ),
        (
            64,
            "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98",
        ),
        (
            65,
            "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee",
        ),
        (
            1023,
            "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11",
        ),
        (
            1024,
            "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7",
        ),
        (
            1025,
            "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444",
        ),
        (
            2048,
            "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a",
        ),
    ];
    let c = Blake3Compress::circuit();
    for (len, digest) in vectors {
        let msg: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        assert_eq!(blake3_hash(&c, &msg), hex(digest), "length {len}");
    }

    // Every length of one block, against the crate.
    let mut rng = SmallRng::seed_from_u64(0xB3B3);
    for len in 0..=64 {
        let msg: Vec<u8> = (0..len).map(|_| rng.random()).collect();
        assert_eq!(
            blake3_hash(&c, &msg),
            blake3::hash(&msg).as_bytes(),
            "length {len}"
        );
    }
}

#[test]
fn sha256_matches_fips_180_4_and_cavp() {
    let c = Sha256Compress::circuit();
    let vectors: [(&[u8], &str); 8] = [
        // FIPS 180-4 examples (NIST CSRC, SHA256.pdf): one block, then two blocks.
        (
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
        // NIST CAVP SHA256ShortMsg.rsp, Len = 0, 8, 24, 440, 448 and 512 bits.
        (
            &[],
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            &[0xd3],
            "28969cdfa74a12c82f3bad960b0b000aca2ac329deea5c2328ebc6f2ba9802c1",
        ),
        (
            &[0xb4, 0x19, 0x0e],
            "dff2e73091f6c05e528896c4c831b9448653dc2ff043528f6769437bc7b975c2",
        ),
        (
            &hex_array::<55>(
                "3ebfb06db8c38d5ba037f1363e118550aad94606e26835a01af05078533cc25f2f39573c04b632f62f68c294ab31f2a3e2a1a0d8c2be51",
            ),
            "6595a2ef537a69ba8583dfbf7f5bec0ab1f93ce4c8ee1916eff44a93af5749c4",
        ),
        (
            &hex_array::<56>(
                "2d52447d1244d2ebc28650e7b05654bad35b3a68eedc7f8515306b496d75f3e73385dd1b002625024b81a02f2fd6dffb6e6d561cb7d0bd7a",
            ),
            "cfb88d6faf2de3a69d36195acec2e255e2af2b7d933997f348e09f6ce5758360",
        ),
        (
            &hex_array::<64>(
                "5a86b737eaea8ee976a0a24da63e7ed7eefad18a101c1211e2b3650c5187c2a8a650547208251f6d4237e661c7bf4c77f335390394c37fa1a9f9be836ac28509",
            ),
            "42e61e174fbb3897d6dd6cef3dd2802fe67b331953b06114a65c772859dfc1aa",
        ),
    ];
    for (msg, digest) in vectors {
        assert_eq!(sha256_hash(&c, msg), hex(digest), "message {msg:02x?}");
    }
}

/// Keccak-f\[1600\] of the zero state: XKCP, KeccakF-1600-IntermediateValues.txt, first example.
const KECCAK_OF_ZERO: [u64; 25] = [
    0xF125_8F79_40E1_DDE7,
    0x84D5_CCF9_33C0_478A,
    0xD598_261E_A65A_A9EE,
    0xBD15_4730_6F80_494D,
    0x8B28_4E05_6253_D057,
    0xFF97_A42D_7F8E_6FD4,
    0x90FE_E5A0_A446_47C4,
    0x8C5B_DA0C_D619_2E76,
    0xAD30_A6F7_1B19_059C,
    0x3093_5AB7_D08F_FC64,
    0xEB5A_A93F_2317_D635,
    0xA9A6_E626_0D71_2103,
    0x81A5_7C16_DBCF_555F,
    0x43B8_31CD_0347_C826,
    0x01F2_2F1A_11A5_569F,
    0x05E5_635A_21D9_AE61,
    0x64BE_FEF2_8CC9_70F2,
    0x6136_7095_7BC4_6611,
    0xB87C_5A55_4FD0_0ECB,
    0x8C3E_E88A_1CCF_32C8,
    0x940C_7922_AE3A_2614,
    0x1841_F924_A2C5_09E4,
    0x16F5_3526_E704_65C2,
    0x75F6_44E9_7F30_A13B,
    0xEAF1_FF7B_5CEC_A249,
];

/// Keccak-f\[1600\] applied twice to the zero state: the same file's second example.
const KECCAK_OF_KECCAK_OF_ZERO: [u64; 25] = [
    0x2D5C_954D_F96E_CB3C,
    0x6A33_2CD0_7057_B56D,
    0x093D_8D12_70D7_6B6C,
    0x8A20_D9B2_5569_D094,
    0x4F9C_4F99_E5E7_F156,
    0xF957_B9A2_DA65_FB38,
    0x8577_3DAE_1275_AF0D,
    0xFAF4_F247_C3D8_10F7,
    0x1F1B_9EE6_F79A_8759,
    0xE4FE_CC0F_EE98_B425,
    0x68CE_61B6_B9CE_68A1,
    0xDEEA_66C4_BA8F_974F,
    0x33C4_3D83_6EAF_B1F5,
    0xE006_5404_2719_DBD9,
    0x7CF8_A9F0_0983_1265,
    0xFD54_49A6_BF17_4743,
    0x97DD_AD33_D899_4B40,
    0x48EA_D5FC_5D0B_E774,
    0xE3B8_C8EE_55B7_B03C,
    0x91A0_226E_649E_42E9,
    0x900E_3129_E7BA_DD7B,
    0x202A_9EC5_FAA3_CCE8,
    0x5B34_0246_4E1C_3DB6,
    0x609F_4E62_A44C_1059,
    0x20D0_6CD2_6A8F_BF5C,
];

#[test]
fn keccak_matches_the_xkcp_intermediate_values() {
    // FIPS 202 Keccak-f[1600], lane x + 5 y at index x + 5 y as in the file.
    let c = KeccakF1600::circuit();
    let once = run(&c, &KeccakF1600::default());
    assert_eq!(once, KECCAK_OF_ZERO);
    assert_eq!(run(&c, &KeccakF1600(once)), KECCAK_OF_KECCAK_OF_ZERO);
}

/// A fresh sponge.
fn sponge() -> FlockChallenger {
    FlockChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// Every port of a gadget's circuit, inputs then outputs.
fn all_ports(circuit: &Circuit) -> Vec<Port> {
    let inputs = (0..).map(Port::Input);
    let outputs = (0..).map(Port::Output);
    inputs
        .take_while(|&p| circuit.port(p).is_some())
        .chain(outputs.take_while(|&p| circuit.port(p).is_some()))
        .collect()
}

/// A statement whose instances' input and output words come from the gadget's encoders.
fn statement<G>(batch: &Batch<'_>, rows: &[(G, G::Output)]) -> PublicPorts
where
    G: Gadget + Sync,
    G::Output: Sync,
{
    let n_in = G::INPUT_WORDS;
    PublicPorts::new(batch, &all_ports(batch.circuit()), |i, words| {
        let (row, output) = &rows[i % rows.len()];
        row.write_input(&mut words[..n_in]);
        G::write_output(output, &mut words[n_in..]);
    })
}

/// Prove and verify `rows`, repeated to `2^log_instances` instances, with every input and output public.
fn prove_public<G>(log_instances: usize, rows: &[(G, G::Output)])
where
    G: Gadget + Sync,
    G::Output: Sync,
{
    let circuit = G::circuit();
    let inputs: Vec<&G> = (0..1 << log_instances)
        .map(|i| &rows[i % rows.len()].0)
        .collect();
    let w = Witness::generate(&circuit, log_instances, &inputs, &inputs[0], |g, words| {
        g.write_input(words);
    });
    let batch = Batch::new(&circuit, log_instances).unwrap();
    let public = statement(&batch, rows);
    let config = FlockConfig::default();
    let proof = prove(&config, &batch, &public, w, &mut sponge()).unwrap();
    verify(&config, &batch, &public, &proof, &mut sponge()).unwrap();
}

/// Prove and verify the smallest batch holding one vector instance, its input and expected output public.
fn prove_vector<G>(instance: G, expected: G::Output)
where
    G: Gadget + Sync,
    G::Output: Sync,
{
    let log_instances = MIN_LOG_N.saturating_sub(G::circuit().k_log());
    prove_public(log_instances, &[(instance, expected)]);
}

#[test]
fn every_gadget_proves_an_official_vector_end_to_end() {
    // RFC 7693, Appendix B: the one compression of BLAKE2s-256("abc").
    let mut h = IV;
    h[0] ^= 0x0101_0020;
    let abc = Blake2sCompress {
        h,
        m: le_block(b"abc"),
        t: 3,
        f0: u32::MAX,
    };
    let digest = hex("508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982");
    prove_vector(
        abc,
        core::array::from_fn(|i| u32::from_le_bytes(digest[4 * i..4 * i + 4].try_into().unwrap())),
    );

    // The BLAKE3 test vector of length 64: one root compression.
    let msg: Vec<u8> = (0..64).map(|i| i as u8).collect();
    let block = Blake3Compress {
        cv: IV,
        block: le_block(&msg),
        counter: 0,
        block_len: 64,
        flags: CHUNK_START | CHUNK_END | ROOT,
    };
    let digest = hex("4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98");
    let out = blake3_reference(&block);
    let cv: [u32; 8] =
        core::array::from_fn(|i| u32::from_le_bytes(digest[4 * i..4 * i + 4].try_into().unwrap()));
    assert_eq!(out[..8], cv);
    prove_vector(block, out);

    // FIPS 180-4: the one compression of SHA-256("abc").
    let mut abc = [0u32; 16];
    abc[0] = 0x6162_6380;
    abc[15] = 24;
    let digest = hex("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    prove_vector(
        Sha256Compress {
            state: IV,
            block: abc,
        },
        core::array::from_fn(|i| u32::from_be_bytes(digest[4 * i..4 * i + 4].try_into().unwrap())),
    );

    // XKCP: Keccak-f[1600] of the zero state.
    prove_vector(KeccakF1600::default(), KECCAK_OF_ZERO);
}

#[test]
fn input_words_match_the_circuits() {
    // Invariant: each gadget's packing fills exactly its circuit's input words.
    assert_eq!(
        Blake3Compress::INPUT_WORDS,
        Blake3Compress::circuit().n_input_words()
    );
    assert_eq!(
        Blake2sCompress::INPUT_WORDS,
        Blake2sCompress::circuit().n_input_words()
    );
    assert_eq!(
        Sha256Compress::INPUT_WORDS,
        Sha256Compress::circuit().n_input_words()
    );
    assert_eq!(
        KeccakF1600::INPUT_WORDS,
        KeccakF1600::circuit().n_input_words()
    );
}

#[test]
fn every_gadget_accepts_honest_witnesses_and_rejects_flipped_bits() {
    let mut rng = SmallRng::seed_from_u64(0xCC5);
    check_rows(&[random_blake3(&mut rng), random_blake3(&mut rng)]);
    check_rows(&[random_blake2s(&mut rng), random_blake2s(&mut rng)]);
    check_rows(&[random_sha256(&mut rng), random_sha256(&mut rng)]);
    check_rows(&[random_keccak(&mut rng), random_keccak(&mut rng)]);
}

#[test]
fn product_counts_and_instance_sizes_are_pinned() {
    // The products each circuit commits, and the log of its bits per instance.
    let shape = |c: Circuit| (c.num_ands(), c.k_log());
    let blake3 = shape(Blake3Compress::circuit());
    assert!(
        blake3.0 <= 10_416,
        "BLAKE3 must beat six two-operand sums per G"
    );
    // 184 per G, less six saved by the IV operands of round 0.
    assert_eq!(blake3, (10_298, 14));
    assert_eq!(shape(Blake2sCompress::circuit()), (14_714, 14));
    assert_eq!(shape(Sha256Compress::circuit()), (22_271, 15));
    assert_eq!(shape(KeccakF1600::circuit()), (24 * 1600, 16));
}

#[test]
fn batches_of_compressions_prove_their_public_inputs_and_outputs() {
    // The statement: these inputs compress to these outputs, the outputs computed natively, not read from the witness.
    let mut rng = SmallRng::seed_from_u64(0x9B);
    let sha: Vec<_> = (0..12)
        .map(|_| {
            let x = random_sha256(&mut rng);
            (x, sha256_reference(&x))
        })
        .collect();
    prove_public(4, &sha);
    let blake3: Vec<_> = (0..12)
        .map(|_| {
            let x = random_blake3(&mut rng);
            (x, blake3_reference(&x))
        })
        .collect();
    prove_public(4, &blake3);

    // Mutation: one claimed SHA-256 output word is wrong, so the honest prover refuses it.
    let circuit = Sha256Compress::circuit();
    let batch = Batch::new(&circuit, 4).unwrap();
    let mut wrong = sha.clone();
    wrong[7].1[3] ^= 1;
    let inputs: Vec<_> = (0..16).map(|i| sha[i % sha.len()].0).collect();
    let w = Witness::generate(
        &circuit,
        4,
        &inputs,
        &inputs[0],
        Sha256Compress::write_input,
    );
    let config = FlockConfig::default();
    assert!(
        prove(
            &config,
            &batch,
            &statement(&batch, &wrong),
            w.clone(),
            &mut sponge()
        )
        .is_err()
    );

    // Mutation: an honest proof checked against the wrong outputs.
    let proof = prove(&config, &batch, &statement(&batch, &sha), w, &mut sponge()).unwrap();
    assert!(
        verify(
            &config,
            &batch,
            &statement(&batch, &wrong),
            &proof,
            &mut sponge()
        )
        .is_err()
    );
}

#[test]
fn writing_an_output_inverts_decoding_it() {
    let mut rng = SmallRng::seed_from_u64(0x0D);
    let words: Vec<u64> = (0..25).map(|_| rng.next_u64()).collect();
    let mut out = vec![0; 25];
    KeccakF1600::write_output(&KeccakF1600::decode_output(&words), &mut out);
    assert_eq!(out, words);
    Blake3Compress::write_output(&Blake3Compress::decode_output(&words[..8]), &mut out[..8]);
    assert_eq!(out[..8], words[..8]);
    Sha256Compress::write_output(&Sha256Compress::decode_output(&words[..4]), &mut out[..4]);
    assert_eq!(out[..4], words[..4]);
    Blake2sCompress::write_output(&Blake2sCompress::decode_output(&words[..4]), &mut out[..4]);
    assert_eq!(out[..4], words[..4]);
}

/// The gadget's witness against the gate walk, bit for bit, on random rows and padding.
fn check_generator<G: Gadget + Sync>(
    circuit: &Circuit,
    seed: u64,
    log_n: usize,
    n_rows: usize,
    random: impl Fn(&mut SmallRng) -> G,
) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let rows: Vec<G> = (0..n_rows).map(|_| random(&mut rng)).collect();
    let padding = random(&mut rng);
    let fast = G::witness(circuit, log_n, &rows, &padding);
    let walk = Witness::generate(circuit, log_n, &rows, &padding, G::write_input);
    assert!(
        fast == walk,
        "generator and walk differ: seed {seed}, 2^{log_n}, {n_rows} rows"
    );
}

/// Every gadget's witness against the walk at one size.
fn check_generators(seed: u64, log_n: usize, n_rows: usize) {
    static CIRCUITS: std::sync::OnceLock<[Circuit; 4]> = std::sync::OnceLock::new();
    let [b3, b2, sha, keccak] = CIRCUITS.get_or_init(|| {
        [
            Blake3Compress::circuit(),
            Blake2sCompress::circuit(),
            Sha256Compress::circuit(),
            KeccakF1600::circuit(),
        ]
    });
    check_generator(b3, seed, log_n, n_rows, random_blake3);
    check_generator(b2, seed, log_n, n_rows, random_blake2s);
    check_generator(sha, seed, log_n, n_rows, random_sha256);
    check_generator(keccak, seed, log_n, n_rows, random_keccak);
}

#[test]
fn generators_match_the_walk_at_every_size() {
    // Fixed seeds: full batches and batches mostly padding, 2^3 to 2^8 instances.
    for log_n in 3..=8 {
        for (seed, n_rows) in [(1, 1 << log_n), (2, 1), (3, (1 << log_n) - 3)] {
            check_generators(seed, log_n, n_rows);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn generators_match_the_walk_on_random_batches(seed: u64, log_n in 3usize..=8, fill in 0.0f64..=1.0) {
        // Invariant: the word-level generators commit exactly the walk's z, A z and B z.
        let n_rows = ((1usize << log_n) as f64 * fill) as usize;
        check_generators(seed, log_n, n_rows);
    }
}

/// A circuit with `G`'s shape but unrelated gates: one chain of products.
fn impostor<G: Gadget>() -> Circuit {
    let canonical = G::circuit();
    let outputs = canonical.const_pos() / 64 - canonical.n_input_words();
    let products = canonical.useful_bits() - canonical.const_pos() - 1;
    let mut c = Builder::new(&vec![64; G::INPUT_WORDS], &vec![64; outputs]);
    let x = c.input(0).to_vec();
    let mut acc = x[0];
    for i in 0..products {
        acc = c.and(acc, x[1 + i % 63]);
    }
    c.output(0, 0, acc);
    let impostor = c.finish();
    // The impostor passes every dimension check, only its gates differ.
    assert_eq!(
        (
            impostor.k_log(),
            impostor.const_pos(),
            impostor.useful_bits()
        ),
        (
            canonical.k_log(),
            canonical.const_pos(),
            canonical.useful_bits()
        )
    );
    assert_ne!(impostor.digest(), canonical.digest());
    impostor
}

#[test]
#[should_panic(expected = "the circuit is not this gadget's")]
fn the_blake3_generator_refuses_a_same_shape_circuit() {
    Blake3Compress::witness(
        &impostor::<Blake3Compress>(),
        LOG_N,
        &[],
        &Blake3Compress::default(),
    );
}

#[test]
#[should_panic(expected = "the circuit is not this gadget's")]
fn the_blake2s_generator_refuses_a_same_shape_circuit() {
    Blake2sCompress::witness(
        &impostor::<Blake2sCompress>(),
        LOG_N,
        &[],
        &Blake2sCompress::default(),
    );
}

#[test]
#[should_panic(expected = "the circuit is not this gadget's")]
fn the_keccak_generator_refuses_a_same_shape_circuit() {
    KeccakF1600::witness(
        &impostor::<KeccakF1600>(),
        LOG_N,
        &[],
        &KeccakF1600::default(),
    );
}
