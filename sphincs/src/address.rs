//! Addresses, the tweaks that separate every hash call (FIPS 205, Section 4.2).

/// The type word of an address (FIPS 205, Section 4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum AddressType {
    /// A step of a WOTS+ chain.
    WotsHash = 0,
    /// The compression of a WOTS+ public key.
    WotsPk = 1,
    /// A node of an XMSS tree.
    Tree = 2,
    /// A node of a FORS tree.
    ForsTree = 3,
    /// The compression of the FORS roots.
    ForsRoots = 4,
}

/// An address: the layer and tree in the hypertree, the type, and three type-specific words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Address {
    /// The layer of the hypertree.
    pub(crate) layer: u32,
    /// The tree within the layer.
    pub(crate) tree: u64,
    /// What the hash computes.
    pub(crate) kind: AddressType,
    /// The key pair address, word 5.
    pub(crate) key_pair: u32,
    /// The chain address or tree height, word 6.
    pub(crate) chain: u32,
    /// The hash address or tree index, word 7.
    pub(crate) hash: u32,
}

impl Address {
    /// The address of type `kind` in `tree` of `layer`, its other words zero.
    pub(crate) const fn new(layer: u32, tree: u64, kind: AddressType) -> Self {
        Self {
            layer,
            tree,
            kind,
            key_pair: 0,
            chain: 0,
            hash: 0,
        }
    }

    /// The 22-byte compressed address `ADRS[3] || ADRS[8:16] || ADRS[19] || ADRS[20:32]` (FIPS 205, Section 11.2).
    pub(crate) fn compressed(&self) -> [u8; 22] {
        let mut out = [0; 22];
        out[0] = self.layer as u8;
        out[1..9].copy_from_slice(&self.tree.to_be_bytes());
        out[9] = self.kind as u8;
        out[10..14].copy_from_slice(&self.key_pair.to_be_bytes());
        out[14..18].copy_from_slice(&self.chain.to_be_bytes());
        out[18..].copy_from_slice(&self.hash.to_be_bytes());
        out
    }
}
