//! Verification split into the algorithms of FIPS 205, every hash call explicit.

use crate::address::{Address, AddressType};
use crate::hash::{Hashes, h_msg};
use crate::params::{A, D, FORS_BYTES, H, H_PRIME, K, LEN, LEN1, LG_W, M, N, W, XMSS_BYTES};

/// Bytes of the FORS message digest, `ceil(k a / 8)`.
pub(crate) const MD_BYTES: usize = (K * A).div_ceil(8);

/// Bytes the tree index is read from, `ceil((h - h / d) / 8)`.
pub(crate) const TREE_BYTES: usize = (H - H_PRIME).div_ceil(8);

/// Bytes the leaf index is read from, `ceil(h / (8 d))`.
pub(crate) const LEAF_BYTES: usize = H_PRIME.div_ceil(8);

/// The root `slh_verify_internal` compares to `PK.root` (FIPS 205, Algorithm 20).
///
/// The signature has the length of the parameter set.
pub(crate) fn root_from_signature(
    seed: &[u8; N],
    root: &[u8; N],
    message: &[&[u8]],
    signature: &[u8],
) -> [u8; N] {
    let hashes = Hashes::new(seed);
    let (r, rest) = signature.split_at(N);
    let (fors, ht) = rest.split_at(FORS_BYTES);
    let r = r.try_into().expect("n bytes");
    let digest = h_msg(r, seed, root, message);
    let (indices, tree, leaf) = split_digest(&digest);
    let mut address = Address::new(0, tree, AddressType::ForsTree);
    address.key_pair = leaf;
    let fors_pk = fors_pk_from_sig(&hashes, fors, &indices, &address);
    ht_root(&hashes, &fors_pk, ht, tree, leaf)
}

/// The FORS indices, the tree index and the leaf index a message digest selects (FIPS 205, Algorithm 20, lines 7-12).
pub(crate) fn split_digest(digest: &[u8; M]) -> ([u32; K], u64, u32) {
    let (md, rest) = digest.split_at(MD_BYTES);
    let (tree, leaf) = rest.split_at(TREE_BYTES);
    let to_int = |bytes: &[u8]| bytes.iter().fold(0u64, |acc, &b| (acc << 8) | u64::from(b));
    let tree = to_int(tree) & ((1 << (H - H_PRIME)) - 1);
    let leaf = to_int(&leaf[..LEAF_BYTES]) as u32 & ((1 << H_PRIME) - 1);
    (base_2b(md, A), tree, leaf)
}

/// The first `OUT` digits of `bits` bits of a byte string, most significant first (FIPS 205, Algorithm 4).
fn base_2b<const OUT: usize>(bytes: &[u8], bits: usize) -> [u32; OUT] {
    let (mut total, mut held, mut at) = (0u64, 0, 0);
    core::array::from_fn(|_| {
        while held < bits {
            total = (total << 8) | u64::from(bytes[at]);
            (held, at) = (held + 8, at + 1);
        }
        held -= bits;
        ((total >> held) & ((1 << bits) - 1)) as u32
    })
}

/// The digits a WOTS+ signature signs: the message's, then its checksum's (FIPS 205, Algorithm 8, lines 1-7).
pub(crate) fn wots_digits(message: &[u8; N]) -> [u32; LEN] {
    let digits: [u32; LEN1] = base_2b(message, LG_W);
    let checksum: u32 = digits.iter().map(|&d| W as u32 - 1 - d).sum();

    // Shifting left to a byte boundary then reading `len_2` digits keeps the checksum's low `len_2 lg_w` bits.
    core::array::from_fn(|i| {
        if i < LEN1 {
            digits[i]
        } else {
            (checksum >> (LG_W * (LEN - 1 - i))) & (W as u32 - 1)
        }
    })
}

/// The WOTS+ public key a signature of `message` gives (FIPS 205, Algorithms 5 and 8).
///
/// The address is of type WOTS_HASH with its key pair set.
fn wots_pk_from_sig(hashes: &Hashes, sig: &[u8], message: &[u8; N], address: &Address) -> [u8; N] {
    let mut ends = [[0; N]; LEN];
    let mut step = *address;
    for (i, (end, digit)) in ends.iter_mut().zip(wots_digits(message)).enumerate() {
        *end = sig[i * N..(i + 1) * N].try_into().expect("n bytes");
        step.chain = i as u32;
        for j in digit..W as u32 - 1 {
            step.hash = j;
            *end = hashes.thash(&step, &[end]);
        }
    }
    let mut pk = Address::new(address.layer, address.tree, AddressType::WotsPk);
    pk.key_pair = address.key_pair;
    hashes.thash(&pk, &[ends.as_flattened()])
}

/// The root of a Merkle path from `node` at leaf `index` (FIPS 205, Algorithms 11 and 17).
///
/// The address carries the layer, tree, type and key pair; height and index are set per level.
fn climb(
    hashes: &Hashes,
    mut node: [u8; N],
    path: &[u8],
    index: u32,
    address: &Address,
) -> [u8; N] {
    let mut step = *address;
    for (level, sibling) in path.chunks(N).enumerate() {
        step.chain = level as u32 + 1;
        step.hash = index >> (level + 1);
        node = if (index >> level) & 1 == 0 {
            hashes.thash(&step, &[&node, sibling])
        } else {
            hashes.thash(&step, &[sibling, &node])
        };
    }
    node
}

/// The XMSS root a signature of `message` at leaf `leaf` gives (FIPS 205, Algorithm 11).
fn xmss_pk_from_sig(
    hashes: &Hashes,
    sig: &[u8],
    message: &[u8; N],
    layer: u32,
    tree: u64,
    leaf: u32,
) -> [u8; N] {
    let (wots, path) = sig.split_at(LEN * N);
    let mut address = Address::new(layer, tree, AddressType::WotsHash);
    address.key_pair = leaf;
    let node = wots_pk_from_sig(hashes, wots, message, &address);
    climb(
        hashes,
        node,
        path,
        leaf,
        &Address::new(layer, tree, AddressType::Tree),
    )
}

/// The top root of the hypertree a signature of `message` gives (FIPS 205, Algorithm 13).
fn ht_root(
    hashes: &Hashes,
    message: &[u8; N],
    sig: &[u8],
    mut tree: u64,
    mut leaf: u32,
) -> [u8; N] {
    let mut node = *message;
    for (layer, xmss) in sig.chunks(XMSS_BYTES).take(D).enumerate() {
        node = xmss_pk_from_sig(hashes, xmss, &node, layer as u32, tree, leaf);
        leaf = (tree & ((1 << H_PRIME) - 1)) as u32;
        tree >>= H_PRIME;
    }
    node
}

/// The FORS public key a signature of `indices` gives (FIPS 205, Algorithm 17).
///
/// The address is of type FORS_TREE with its key pair set.
fn fors_pk_from_sig(hashes: &Hashes, sig: &[u8], indices: &[u32; K], address: &Address) -> [u8; N] {
    let mut roots = [[0; N]; K];
    for (i, ((root, &index), tree)) in roots
        .iter_mut()
        .zip(indices)
        .zip(sig.chunks((1 + A) * N))
        .enumerate()
    {
        let (sk, path) = tree.split_at(N);
        let index = ((i as u32) << A) | index;
        let mut leaf = *address;
        leaf.hash = index;
        let node = hashes.thash(&leaf, &[sk]);
        *root = climb(hashes, node, path, index, address);
    }
    let mut pk = Address::new(address.layer, address.tree, AddressType::ForsRoots);
    pk.key_pair = address.key_pair;
    hashes.thash(&pk, &[roots.as_flattened()])
}
