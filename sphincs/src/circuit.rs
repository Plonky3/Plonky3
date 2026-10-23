//! One verification as a Boolean circuit, every SHA-256 compression wired to the next.
//!
//! The shape depends only on the parameters: every chain takes all `w - 1` steps, and digit-driven selects keep or drop each one.
//!
//! The digest's bits steer every Merkle path and set every tree and key pair address.

use alloc::vec::Vec;

use p3_flock::gadgets::sha256::{IV, compress};
use p3_flock::{Builder, Circuit, Wire, Word};

use crate::address::AddressType;
use crate::hash::{BLOCK_BYTES, DIGEST_BYTES};
use crate::params::{
    A, FORS_BYTES, H, H_PRIME, K, LEN, LEN1, LEN2, LG_W, M, N, SIGNATURE_BYTES, W, XMSS_BYTES,
};
use crate::verify::{LEAF_BYTES, MD_BYTES, TREE_BYTES};

/// Input port of the public seed.
pub(crate) const SEED_PORT: usize = 0;

/// Input port of the public root, which the message digest hashes.
pub(crate) const ROOT_PORT: usize = 1;

/// Input port of the message.
pub(crate) const MESSAGE_PORT: usize = 2;

/// Input port of the signature.
pub(crate) const SIGNATURE_PORT: usize = 3;

/// Output port of the root the signature gives.
pub(crate) const COMPUTED_ROOT_PORT: usize = 0;

/// Bytes of a signed message.
pub const MESSAGE_BYTES: usize = 32;

/// A byte string as wires, eight per byte, low bit first.
type Bytes = Vec<Wire>;

/// An address whose words are wires (FIPS 205, Section 4.2).
#[derive(Clone)]
struct AddressWires {
    /// The layer, fixed by the circuit's shape.
    layer: u8,
    /// The tree, 64 bits.
    tree: Word,
    /// The type.
    kind: AddressType,
    /// Word 5, 32 bits.
    key_pair: Word,
    /// Word 6, 32 bits.
    chain: Word,
    /// Word 7, 32 bits.
    hash: Word,
}

/// The circuit under construction, with the compressed seed block every tweakable hash starts from.
struct Verifier {
    c: Builder,
    seeded: [Word; 8],
    compressions: usize,
}

/// The circuit verifying one signature of a [`MESSAGE_BYTES`]-byte message under an empty context, and its SHA-256 compressions.
///
/// The public seed, public root and message are inputs, and the root the signature gives is the output.
pub(crate) fn circuit() -> (Circuit, usize) {
    let mut c = Builder::new(
        &[8 * N, 8 * N, 8 * MESSAGE_BYTES, 8 * SIGNATURE_BYTES],
        &[8 * N],
    );
    let [seed, root, message, signature] =
        [SEED_PORT, ROOT_PORT, MESSAGE_PORT, SIGNATURE_PORT].map(|port| c.input(port).to_vec());

    // The seed block is the same for every tweakable hash, so it is compressed once.
    let iv: [Word; 8] = IV.map(|word| c.literal(u64::from(word), 32));
    let mut block = seed.clone();
    block.extend(c.literal(0, 8 * (BLOCK_BYTES - N)).iter());
    let seeded = compress(&mut c, &iv, &block_words(&block));
    let mut v = Verifier {
        c,
        seeded,
        compressions: 1,
    };

    let (r, rest) = signature.split_at(8 * N);
    let (fors, ht) = rest.split_at(8 * FORS_BYTES);
    let digest = v.h_msg(&iv, r, &seed, &root, &message);

    // The FORS indices, tree and leaf, read from the digest as FIPS 205 Algorithm 20 does.
    let stream_bit = |p: usize| digest[8 * (p / 8) + 7 - p % 8];
    let indices: Vec<Word> = (0..K)
        .map(|i| (0..A).map(|t| stream_bit(A * i + A - 1 - t)).collect())
        .collect();
    let int_bit = |end: usize, j: usize| digest[8 * (end - 1 - j / 8) + j % 8];
    let mut tree: Vec<Wire> = (0..H - H_PRIME)
        .map(|j| int_bit(MD_BYTES + TREE_BYTES, j))
        .collect();
    let mut leaf: Vec<Wire> = (0..H_PRIME)
        .map(|j| int_bit(MD_BYTES + TREE_BYTES + LEAF_BYTES, j))
        .collect();

    let mut node = v.fors_pk(fors, &indices, &tree, &leaf);
    for (layer, xmss) in ht.chunks(8 * XMSS_BYTES).enumerate() {
        node = v.xmss_pk(xmss, &node, layer as u8, &tree, &leaf);
        let split = H_PRIME.min(tree.len());
        leaf = tree[..split].to_vec();
        tree.drain(..split);
    }
    let Verifier {
        mut c,
        compressions,
        ..
    } = v;
    for (bit, &wire) in node.iter().enumerate() {
        c.output(COMPUTED_ROOT_PORT, bit, wire);
    }
    (c.finish_compiled(), compressions)
}

impl Verifier {
    /// `H_msg(R, PK.seed, PK.root, toByte(0, 1) || toByte(0, 1) || M)` (FIPS 205, Algorithm 24 and Section 11.2.1).
    fn h_msg(
        &mut self,
        iv: &[Word; 8],
        r: &[Wire],
        seed: &[Wire],
        root: &[Wire],
        message: &[Wire],
    ) -> Bytes {
        let mut input: Bytes = [r, seed, root].concat();
        input.extend(self.c.literal(0, 16).iter());
        input.extend_from_slice(message);
        let inner = state_bytes(&self.sha256(iv.clone(), 0, &input));

        // MGF1: one compression chain per 32-bit counter.
        let mut digest = Bytes::new();
        for counter in 0..M.div_ceil(DIGEST_BYTES) {
            let mut input: Bytes = [r, seed, &inner].concat();
            input.extend(reverse_bytes(&self.c.literal(counter as u64, 32)));
            digest.extend(state_bytes(&self.sha256(iv.clone(), 0, &input)));
        }
        digest.truncate(8 * M);
        digest
    }

    /// SHA-256 from `state` after `prefix` bytes of earlier blocks, over `bytes` and the padding (FIPS 180-4, Section 5.1.1).
    fn sha256(&mut self, mut state: [Word; 8], prefix: usize, bytes: &[Wire]) -> [Word; 8] {
        let total = prefix + bytes.len() / 8;
        let mut padded = bytes.to_vec();
        padded.extend(self.c.literal(0x80, 8).iter());
        let zeros = (2 * BLOCK_BYTES - 8 - (total + 1) % BLOCK_BYTES) % BLOCK_BYTES;
        padded.extend(self.c.literal(0, 8 * zeros).iter());
        padded.extend(reverse_bytes(&self.c.literal(8 * total as u64, 64)));
        for block in padded.chunks(8 * BLOCK_BYTES) {
            state = compress(&mut self.c, &state, &block_words(block));
            self.compressions += 1;
        }
        state
    }

    /// `Trunc_n(SHA-256(PK.seed || toByte(0, 64 - n) || ADRSc || M))`, from the compressed seed block.
    fn thash(&mut self, address: &AddressWires, message: &[Wire]) -> Bytes {
        let mut bytes: Bytes = self.c.literal(u64::from(address.layer), 8).to_vec();
        bytes.extend(reverse_bytes(&address.tree));
        bytes.extend(self.c.literal(address.kind as u64, 8).iter());
        for word in [&address.key_pair, &address.chain, &address.hash] {
            bytes.extend(reverse_bytes(word));
        }
        bytes.extend_from_slice(message);
        let mut out = state_bytes(&self.sha256(self.seeded.clone(), BLOCK_BYTES, &bytes));
        out.truncate(8 * N);
        out
    }

    /// The root of a Merkle path from `node`, the bits of `index` choosing each side (FIPS 205, Algorithms 11 and 17).
    fn climb(
        &mut self,
        mut node: Bytes,
        path: &[Wire],
        index: &Word,
        address: &mut AddressWires,
    ) -> Bytes {
        for (level, sibling) in path.chunks(8 * N).enumerate() {
            let bit = index[level];
            address.chain = self.c.literal(level as u64 + 1, 32);
            address.hash = index >> (level + 1);
            let left: Bytes = node
                .iter()
                .zip(sibling)
                .map(|(&n, &s)| self.c.mux(bit, s, n))
                .collect();
            let swap = self.c.xor_word(&node, sibling);
            let right = self.c.xor_word(&left, &swap);
            let children = [left, right.to_vec()].concat();
            node = self.thash(address, &children);
        }
        node
    }

    /// The FORS public key from its signature and indices (FIPS 205, Algorithm 17).
    fn fors_pk(&mut self, sig: &[Wire], indices: &[Word], tree: &[Wire], leaf: &[Wire]) -> Bytes {
        let mut roots = Bytes::new();
        for (i, (part, index)) in sig.chunks(8 * (1 + A) * N).zip(indices).enumerate() {
            let (sk, path) = part.split_at(8 * N);
            // The leaf's index in the whole forest, `i 2^a + indices[i]`.
            let offset = self.c.literal((i as u64) << A, 32);
            let index: Word = (0..32)
                .map(|t| if t < A { index[t] } else { offset[t] })
                .collect();
            let mut address = AddressWires {
                layer: 0,
                tree: widen(tree, 64),
                kind: AddressType::ForsTree,
                key_pair: widen(leaf, 32),
                chain: self.c.literal(0, 32),
                hash: index.clone(),
            };
            let node = self.thash(&address, sk);
            roots.extend(self.climb(node, path, &index, &mut address));
        }
        let address = AddressWires {
            layer: 0,
            tree: widen(tree, 64),
            kind: AddressType::ForsRoots,
            key_pair: widen(leaf, 32),
            chain: self.c.literal(0, 32),
            hash: self.c.literal(0, 32),
        };
        self.thash(&address, &roots)
    }

    /// The XMSS root from a WOTS+ signature of `message` and a path (FIPS 205, Algorithms 8 and 11).
    fn xmss_pk(
        &mut self,
        sig: &[Wire],
        message: &[Wire],
        layer: u8,
        tree: &[Wire],
        leaf: &[Wire],
    ) -> Bytes {
        let (wots, path) = sig.split_at(8 * LEN * N);
        let one = self.c.one();

        // The message's digits, two per byte as `lg_w = 4`, high nibble first, then the checksum's.
        let mut digits: Vec<Word> = message
            .chunks(8)
            .flat_map(|byte| [Word::from(&byte[LG_W..]), Word::from(&byte[..LG_W])])
            .collect();
        let complements: Vec<Word> = digits
            .iter()
            .map(|digit| widen(&self.c.not_word(digit), LEN2 * LG_W))
            .collect();
        let operands: Vec<&[Wire]> = complements.iter().map(|w| &w[..]).collect();
        let checksum = self.c.wrapping_sum(&operands[..LEN1]);
        digits.extend(
            (0..LEN2)
                .rev()
                .map(|j| Word::from(&checksum[LG_W * j..LG_W * (j + 1)])),
        );

        // Every chain takes all its steps, and a step counts once the slot reaches the digit.
        let mut ends = Bytes::new();
        for (chain, (digit, value)) in digits.iter().zip(wots.chunks(8 * N)).enumerate() {
            let mut value = value.to_vec();
            let mut reached = Wire::ZERO;
            for slot in 0..W - 1 {
                let mut equal = one;
                for (b, &bit) in digit.iter().enumerate() {
                    let matched = if (slot >> b) & 1 == 1 {
                        bit
                    } else {
                        self.c.not(bit)
                    };
                    equal = self.c.and(equal, matched);
                }
                // At most one slot equals the digit, so the sum is an OR.
                reached = self.c.xor(reached, equal);
                let address = AddressWires {
                    layer,
                    tree: widen(tree, 64),
                    kind: AddressType::WotsHash,
                    key_pair: widen(leaf, 32),
                    chain: self.c.literal(chain as u64, 32),
                    hash: self.c.literal(slot as u64, 32),
                };
                let step = self.thash(&address, &value);
                value = value
                    .iter()
                    .zip(&step)
                    .map(|(&kept, &stepped)| self.c.mux(reached, stepped, kept))
                    .collect();
            }
            ends.extend(value);
        }
        let address = AddressWires {
            layer,
            tree: widen(tree, 64),
            kind: AddressType::WotsPk,
            key_pair: widen(leaf, 32),
            chain: self.c.literal(0, 32),
            hash: self.c.literal(0, 32),
        };
        let pk = self.thash(&address, &ends);
        let mut address = AddressWires {
            kind: AddressType::Tree,
            key_pair: self.c.literal(0, 32),
            ..address
        };
        self.climb(pk, path, &widen(leaf, 32), &mut address)
    }
}

/// Wires zero-extended to `bits`.
fn widen(wires: &[Wire], bits: usize) -> Word {
    wires
        .iter()
        .copied()
        .chain(core::iter::repeat(Wire::ZERO))
        .take(bits)
        .collect()
}

/// The bytes of a run of wires in reverse order, which turns a big-endian byte string into a word and back.
fn reverse_bytes(wires: &[Wire]) -> Bytes {
    wires.rchunks(8).flatten().copied().collect()
}

/// A 64-byte block as sixteen big-endian words.
fn block_words(block: &[Wire]) -> [Word; 16] {
    core::array::from_fn(|i| {
        reverse_bytes(&block[32 * i..32 * (i + 1)])
            .into_iter()
            .collect()
    })
}

/// A chaining value as its 32 big-endian bytes.
fn state_bytes(state: &[Word; 8]) -> Bytes {
    state.iter().flat_map(|word| reverse_bytes(word)).collect()
}
