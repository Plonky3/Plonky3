//! Leaf hashers: hash matrix rows with a **byte- or `u64`-oriented** primitive (Blake3, Keccak sponge,
//! …), map the fixed digest into **`[F; DIGEST_ELEMS]`**, and plug into `p3_merkle_tree::MerkleTreeMmcs`
//! together with a **separate** field-native **inner** compressor (typically [`TruncatedPermutation`]
//! over Poseidon).
//!
//! # End-to-end shape
//!
//! - **Leaves:** one digest per row, type `[F; DIGEST_ELEMS]` (or packed `[P; DIGEST_ELEMS]` when
//!   `P = F::Packing` is SIMD). The helpers here only implement **leaf** hashing; they do **not**
//!   replace Poseidon (or similar) on internal Merkle nodes.
//! - **Inner nodes:** unchanged from a pure-Poseidon MMCS, e.g.
//!   `TruncatedPermutation::<PoseidonPerm, 2, DIGEST_ELEMS, WIDTH>::new(perm)` with the same
//!   `DIGEST_ELEMS` and arity `N` as your
//!   `p3_merkle_tree::MerkleTreeMmcs<_, _, _, _, N, DIGEST_ELEMS>`.
//! - **Challenger / transcript:** still your choice; leaf and inner primitives need not match.
//!
//! Pick **`DIGEST_ELEMS`** once so leaf outputs, inner compression chunk size, and
//! `p3_merkle_tree::MerkleTreeMmcs`’s const generic agree (often `8` with a width-16 Poseidon and binary
//! `N = 2`).
//!
//! # Mapping digest bytes → field limbs
//!
//! - [`field_digest_from_le_u32_bytes`]: consecutive little-endian **`u32`** limbs, each passed to
//!   [`PrimeField::from_int`]. Requires at least **`4 * DIGEST_ELEMS`** bytes (e.g. Blake3’s 32 bytes
//!   give **eight** limbs). Natural for ~31-bit fields (BabyBear, KoalaBear, …).
//! - [`field_digest_from_le_u64_bytes`]: little-endian **`u64`** limbs; requires **`8 * DIGEST_ELEMS`**
//!   bytes and **`F: PrimeField64`** (e.g. Goldilocks). A 32-byte hash yields only **four** limbs—use a
//!   longer hash output or smaller `DIGEST_ELEMS` if you need more.
//!
//! Trailing digest bytes are ignored if the hash is longer than required.
//!
//! # Which leaf type to use (scalar `P = F` vs SIMD `P ≠ F`)
//!
//! `p3_merkle_tree::MerkleTreeMmcs` needs **`H: CryptographicHasher<F, [PW::Value; DE]>`** and
//! **`H: CryptographicHasher<P, [PW; DE]>`** with `P = F::Packing` and `PW` the digest packing type
//! (usually `PW = P` in Poseidon setups).
//!
//! - If **`F::Packing` is `F`** (scalar packing, e.g. Monty31 when NEON/AVX target features are off):
//!   both bounds are the same trait instance. Use a **single** leaf hasher:
//!   - Bytes: [`FieldLeafFromByteHash`] or, for 64-bit limbs, [`FieldLeafFromByteHashU64`].
//!   - `u64` sponge stream: [`FieldLeafFromU64Hash`].
//!   Example type:
//!   `p3_merkle_tree::MerkleTreeMmcs<F, F, FieldLeafFromByteHash<Blake3, F, 32, 8>, MyCompress, 2, 8>`.
//! - If **`P ≠ F`** (SIMD vertical packing): you need **both** scalar-row and packed-row hashing.
//!   Rust does not allow one generic wrapper to implement both without conflicting impls when `P` could
//!   equal `F`, so at a **concrete** `F` and packed type `PF` (must be **exactly** `<F as Field>::Packing`):
//!   - Invoke [`byte_hash_merkle_leaf_pair!`], [`byte_hash_merkle_leaf_pair_u64!`], or
//!     [`u64_hash_merkle_leaf_pair!`] to generate a small newtype that implements both modes.
//!   Example: `byte_hash_merkle_leaf_pair!(pub MyLeaf, Blake3, F, PackedBabyBearNeon, 32, 8);` then
//!   `p3_merkle_tree::MerkleTreeMmcs<PF, PF, MyLeaf, MyCompress, 2, 8>` with `PF = F::Packing`.
//!
//! For Keccak-style leaves over a field, combine [`SerializingHasher`](crate::SerializingHasher) with a
//! sponge implementing [`CryptographicHasher<u64, [u64; R]>`], then use [`FieldLeafFromU64Hash`] or
//! [`u64_hash_merkle_leaf_pair!`] (same `P = F` vs `P ≠ F` split as above).
//!
//! **Custom:** implement [`CryptographicHasher`] on your own newtype and call
//! [`field_digest_from_le_u32_bytes`] / [`field_digest_from_le_u64_bytes`] after your inner hash; or
//! delegate to the provided structs.
//!
//! # Examples in this repo
//!
//! Integration tests under `p3-symmetric`’s `tests/hybrid_overlap.rs` show BabyBear + Blake3 leaves +
//! Poseidon2 compression for scalar packing and for SIMD packing.

use alloc::vec::Vec;
use core::array;
use core::marker::PhantomData;

use p3_field::{Field, PackedValue, PrimeField, PrimeField64};

use crate::CryptographicHasher;

/// Map the first `4 * DIGEST_ELEMS` bytes of `bytes` into field elements (little-endian `u32` per limb).
///
/// # Panics
/// Panics if `bytes.len()` is smaller than `4 * DIGEST_ELEMS`.
#[inline]
pub fn field_digest_from_le_u32_bytes<F: PrimeField, const DIGEST_ELEMS: usize>(
    bytes: &[u8],
) -> [F; DIGEST_ELEMS] {
    assert!(
        bytes.len() >= 4 * DIGEST_ELEMS,
        "need at least {} bytes for {} field limbs",
        4 * DIGEST_ELEMS,
        DIGEST_ELEMS
    );
    array::from_fn(|i| {
        let start = 4 * i;
        F::from_int(u32::from_le_bytes(
            bytes[start..start + 4].try_into().expect("chunk size"),
        ))
    })
}

/// Map the first `8 * DIGEST_ELEMS` bytes of `bytes` into field elements (little-endian `u64` per limb).
///
/// Intended for [`PrimeField64`] fields (e.g. Goldilocks). Each limb uses `F::from_int` on the
/// little-endian `u64` word.
///
/// # Panics
/// Panics if `bytes.len()` is smaller than `8 * DIGEST_ELEMS`.
#[inline]
pub fn field_digest_from_le_u64_bytes<F: PrimeField64, const DIGEST_ELEMS: usize>(
    bytes: &[u8],
) -> [F; DIGEST_ELEMS] {
    assert!(
        bytes.len() >= 8 * DIGEST_ELEMS,
        "need at least {} bytes for {} u64 limbs",
        8 * DIGEST_ELEMS,
        DIGEST_ELEMS
    );
    array::from_fn(|i| {
        let start = 8 * i;
        F::from_int(u64::from_le_bytes(
            bytes[start..start + 8].try_into().expect("chunk size"),
        ))
    })
}

fn u64_digest_to_bytes<const N_U64: usize>(d: &[u64; N_U64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(N_U64 * 8);
    for w in d {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Implements both Merkle leaf hashing modes for a byte-oriented inner hasher at **concrete** `F` and `PF`.
///
/// # Parameters
///
/// - `$name`: newtype wrapper around the inner hasher (e.g. `HybridBlake3Leaf`).
/// - `$inner`: type implementing `CryptographicHasher<u8, [u8; BYTE_OUT]>` (e.g. `p3_blake3::Blake3`);
///   should be `Copy`.
/// - `$f` / `$pf`: scalar field and its **exact** SIMD packing `<$f as Field>::Packing` when `P ≠ F`.
/// - `$bo`: digest length in bytes (`BYTE_OUT`); must satisfy `$bo >= 4 * $de` for the default u32-limb decode.
/// - `$de`: digest width in field elements (`DIGEST_ELEMS`); must match inner Merkle compression and MMCS consts.
///
/// # Example
///
/// ```ignore
/// use p3_baby_bear::{BabyBear, PackedBabyBearNeon};
/// use p3_blake3::Blake3;
/// use p3_symmetric::byte_hash_merkle_leaf_pair;
///
/// byte_hash_merkle_leaf_pair!(pub HybridBlake3Leaf, Blake3, BabyBear, PackedBabyBearNeon, 32, 8);
/// ```
#[macro_export]
macro_rules! byte_hash_merkle_leaf_pair {
    (
        $(#[$attr:meta])*
        $vis:vis $name:ident, $inner:ty, $f:ty, $pf:ty, $bo:literal, $de:literal
    ) => {
        $(#[$attr])*
        #[derive(Clone, Copy, Debug)]
        $vis struct $name($inner);

        impl $crate::CryptographicHasher<$f, [$f; $de]> for $name {
            fn hash_iter<I>(&self, input: I) -> [$f; $de]
            where
                I: IntoIterator<Item = $f>,
            {
                $crate::FieldLeafFromByteHash::<$inner, $f, $bo, $de>::new(self.0).hash_iter(input)
            }
        }

        impl $crate::CryptographicHasher<$pf, [$pf; $de]> for $name {
            fn hash_iter<I>(&self, input: I) -> [$pf; $de]
            where
                I: IntoIterator<Item = $pf>,
            {
                $crate::FieldLeafPackedLanesFromByteHash::<$inner, $f, $pf, $bo, $de>::new(self.0)
                    .hash_iter(input)
            }
        }
    };
}

/// Same as [`byte_hash_merkle_leaf_pair`], but the inner hasher outputs **`u64` words** (e.g. a Keccak
/// sponge) instead of a byte array.
///
/// # Parameters
///
/// - `$nu`: number of `u64` words in the sponge output. The implementation flattens them to bytes and
///   decodes with [`field_digest_from_le_u32_bytes`], so you need **`8 * $nu >= 4 * $de`** bytes (same
///   constraint as [`FieldLeafFromU64Hash`]'s `const` assert).
/// - Other parameters match [`byte_hash_merkle_leaf_pair`].
#[macro_export]
macro_rules! u64_hash_merkle_leaf_pair {
    (
        $(#[$attr:meta])*
        $vis:vis $name:ident, $inner:ty, $f:ty, $pf:ty, $nu:literal, $de:literal
    ) => {
        $(#[$attr])*
        #[derive(Clone, Copy, Debug)]
        $vis struct $name($inner);

        impl $crate::CryptographicHasher<$f, [$f; $de]> for $name {
            fn hash_iter<I>(&self, input: I) -> [$f; $de]
            where
                I: IntoIterator<Item = $f>,
            {
                $crate::FieldLeafFromU64Hash::<$inner, $f, $nu, $de>::new(self.0).hash_iter(input)
            }
        }

        impl $crate::CryptographicHasher<$pf, [$pf; $de]> for $name {
            fn hash_iter<I>(&self, input: I) -> [$pf; $de]
            where
                I: IntoIterator<Item = $pf>,
            {
                $crate::FieldLeafPackedLanesFromU64Hash::<$inner, $f, $pf, $nu, $de>::new(self.0)
                    .hash_iter(input)
            }
        }
    };
}

/// Like [`byte_hash_merkle_leaf_pair`], but decodes the inner **byte** digest with
/// [`field_digest_from_le_u64_bytes`] (eight bytes per field limb, `F: PrimeField64`).
///
/// Requires `$bo >= 8 * $de` at the use site (enforced by the leaf types). Use when your field is
/// naturally 64-bit and you want the digest split into `u64` little-endian chunks rather than `u32`.
///
/// # Parameters
///
/// Same as [`byte_hash_merkle_leaf_pair`], except `$bo` must cover `8 * $de` bytes.
#[macro_export]
macro_rules! byte_hash_merkle_leaf_pair_u64 {
    (
        $(#[$attr:meta])*
        $vis:vis $name:ident, $inner:ty, $f:ty, $pf:ty, $bo:literal, $de:literal
    ) => {
        $(#[$attr])*
        #[derive(Clone, Copy, Debug)]
        $vis struct $name($inner);

        impl $crate::CryptographicHasher<$f, [$f; $de]> for $name {
            fn hash_iter<I>(&self, input: I) -> [$f; $de]
            where
                I: IntoIterator<Item = $f>,
            {
                $crate::FieldLeafFromByteHashU64::<$inner, $f, $bo, $de>::new(self.0).hash_iter(input)
            }
        }

        impl $crate::CryptographicHasher<$pf, [$pf; $de]> for $name {
            fn hash_iter<I>(&self, input: I) -> [$pf; $de]
            where
                I: IntoIterator<Item = $pf>,
            {
                $crate::FieldLeafPackedLanesFromByteHashU64::<$inner, $f, $pf, $bo, $de>::new(self.0)
                    .hash_iter(input)
            }
        }
    };
}

/// Leaf hasher: hash scalar field rows as a byte stream, then map the digest into `[F; DIGEST_ELEMS]`.
#[derive(Copy, Clone, Debug)]
pub struct FieldLeafFromByteHash<Inner, F, const BYTE_OUT: usize, const DIGEST_ELEMS: usize> {
    inner: Inner,
    _phantom: PhantomData<F>,
}

impl<Inner, F, const BYTE_OUT: usize, const DIGEST_ELEMS: usize>
    FieldLeafFromByteHash<Inner, F, BYTE_OUT, DIGEST_ELEMS>
{
    pub const fn new(inner: Inner) -> Self {
        const {
            assert!(
                BYTE_OUT >= 4 * DIGEST_ELEMS,
                "BYTE_OUT must be at least 4 * DIGEST_ELEMS for u32 limb decoding"
            );
        }
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<Inner, F, const BYTE_OUT: usize, const DIGEST_ELEMS: usize>
    CryptographicHasher<F, [F; DIGEST_ELEMS]>
    for FieldLeafFromByteHash<Inner, F, BYTE_OUT, DIGEST_ELEMS>
where
    Inner: CryptographicHasher<u8, [u8; BYTE_OUT]> + Clone,
    F: Field + PrimeField,
{
    fn hash_iter<I>(&self, input: I) -> [F; DIGEST_ELEMS]
    where
        I: IntoIterator<Item = F>,
    {
        let digest = self.inner.hash_iter(F::into_byte_stream(input));
        field_digest_from_le_u32_bytes(&digest)
    }
}

/// Packed Merkle leaves only: hashes each SIMD lane’s scalar stream independently (vertical packing).
#[derive(Copy, Clone, Debug)]
pub struct FieldLeafPackedLanesFromByteHash<
    Inner,
    F,
    PF,
    const BYTE_OUT: usize,
    const DIGEST_ELEMS: usize,
> {
    inner: Inner,
    _phantom: PhantomData<(F, PF)>,
}

impl<Inner, F, PF, const BYTE_OUT: usize, const DIGEST_ELEMS: usize>
    FieldLeafPackedLanesFromByteHash<Inner, F, PF, BYTE_OUT, DIGEST_ELEMS>
{
    pub const fn new(inner: Inner) -> Self {
        const {
            assert!(
                BYTE_OUT >= 4 * DIGEST_ELEMS,
                "BYTE_OUT must be at least 4 * DIGEST_ELEMS for u32 limb decoding"
            );
        }
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<Inner, F, PF, const BYTE_OUT: usize, const DIGEST_ELEMS: usize>
    CryptographicHasher<PF, [PF; DIGEST_ELEMS]>
    for FieldLeafPackedLanesFromByteHash<Inner, F, PF, BYTE_OUT, DIGEST_ELEMS>
where
    Inner: CryptographicHasher<u8, [u8; BYTE_OUT]> + Clone,
    F: Field + PrimeField,
    PF: PackedValue<Value = F>,
{
    fn hash_iter<I>(&self, input: I) -> [PF; DIGEST_ELEMS]
    where
        I: IntoIterator<Item = PF>,
    {
        let packed_inputs: Vec<PF> = input.into_iter().collect();
        let width = PF::WIDTH;
        let per_lane: Vec<[F; DIGEST_ELEMS]> = (0..width)
            .map(|lane| {
                let digest = self.inner.hash_iter(F::into_byte_stream(
                    packed_inputs.iter().map(|p| p.extract(lane)),
                ));
                field_digest_from_le_u32_bytes(&digest)
            })
            .collect();
        array::from_fn(|j| PF::from_fn(|lane| per_lane[lane][j]))
    }
}

/// Like [`FieldLeafFromByteHash`], but decodes the digest with [`field_digest_from_le_u64_bytes`]
/// (needs [`PrimeField64`] and a longer byte output for the same `DIGEST_ELEMS`).
#[derive(Copy, Clone, Debug)]
pub struct FieldLeafFromByteHashU64<Inner, F, const BYTE_OUT: usize, const DIGEST_ELEMS: usize> {
    inner: Inner,
    _phantom: PhantomData<F>,
}

impl<Inner, F, const BYTE_OUT: usize, const DIGEST_ELEMS: usize>
    FieldLeafFromByteHashU64<Inner, F, BYTE_OUT, DIGEST_ELEMS>
{
    pub const fn new(inner: Inner) -> Self {
        const {
            assert!(
                BYTE_OUT >= 8 * DIGEST_ELEMS,
                "BYTE_OUT must be at least 8 * DIGEST_ELEMS for u64 limb decoding"
            );
        }
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<Inner, F, const BYTE_OUT: usize, const DIGEST_ELEMS: usize>
    CryptographicHasher<F, [F; DIGEST_ELEMS]>
    for FieldLeafFromByteHashU64<Inner, F, BYTE_OUT, DIGEST_ELEMS>
where
    Inner: CryptographicHasher<u8, [u8; BYTE_OUT]> + Clone,
    F: Field + PrimeField64,
{
    fn hash_iter<I>(&self, input: I) -> [F; DIGEST_ELEMS]
    where
        I: IntoIterator<Item = F>,
    {
        let digest = self.inner.hash_iter(F::into_byte_stream(input));
        field_digest_from_le_u64_bytes(&digest)
    }
}

/// Packed Merkle leaves using [`field_digest_from_le_u64_bytes`].
#[derive(Copy, Clone, Debug)]
pub struct FieldLeafPackedLanesFromByteHashU64<
    Inner,
    F,
    PF,
    const BYTE_OUT: usize,
    const DIGEST_ELEMS: usize,
> {
    inner: Inner,
    _phantom: PhantomData<(F, PF)>,
}

impl<Inner, F, PF, const BYTE_OUT: usize, const DIGEST_ELEMS: usize>
    FieldLeafPackedLanesFromByteHashU64<Inner, F, PF, BYTE_OUT, DIGEST_ELEMS>
{
    pub const fn new(inner: Inner) -> Self {
        const {
            assert!(
                BYTE_OUT >= 8 * DIGEST_ELEMS,
                "BYTE_OUT must be at least 8 * DIGEST_ELEMS for u64 limb decoding"
            );
        }
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<Inner, F, PF, const BYTE_OUT: usize, const DIGEST_ELEMS: usize>
    CryptographicHasher<PF, [PF; DIGEST_ELEMS]>
    for FieldLeafPackedLanesFromByteHashU64<Inner, F, PF, BYTE_OUT, DIGEST_ELEMS>
where
    Inner: CryptographicHasher<u8, [u8; BYTE_OUT]> + Clone,
    F: Field + PrimeField64,
    PF: PackedValue<Value = F>,
{
    fn hash_iter<I>(&self, input: I) -> [PF; DIGEST_ELEMS]
    where
        I: IntoIterator<Item = PF>,
    {
        let packed_inputs: Vec<PF> = input.into_iter().collect();
        let width = PF::WIDTH;
        let per_lane: Vec<[F; DIGEST_ELEMS]> = (0..width)
            .map(|lane| {
                let digest = self.inner.hash_iter(F::into_byte_stream(
                    packed_inputs.iter().map(|p| p.extract(lane)),
                ));
                field_digest_from_le_u64_bytes(&digest)
            })
            .collect();
        array::from_fn(|j| PF::from_fn(|lane| per_lane[lane][j]))
    }
}

/// Leaf hasher from a `u64`-stream hash for scalar rows.
#[derive(Copy, Clone, Debug)]
pub struct FieldLeafFromU64Hash<Inner, F, const N_U64: usize, const DIGEST_ELEMS: usize> {
    inner: Inner,
    _phantom: PhantomData<F>,
}

impl<Inner, F, const N_U64: usize, const DIGEST_ELEMS: usize>
    FieldLeafFromU64Hash<Inner, F, N_U64, DIGEST_ELEMS>
{
    pub const fn new(inner: Inner) -> Self {
        const {
            assert!(
                8 * N_U64 >= 4 * DIGEST_ELEMS,
                "u64 digest does not contain enough bytes for DIGEST_ELEMS u32 limbs"
            );
        }
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<Inner, F, const N_U64: usize, const DIGEST_ELEMS: usize>
    CryptographicHasher<F, [F; DIGEST_ELEMS]>
    for FieldLeafFromU64Hash<Inner, F, N_U64, DIGEST_ELEMS>
where
    Inner: CryptographicHasher<u64, [u64; N_U64]> + Clone,
    F: Field + PrimeField,
{
    fn hash_iter<I>(&self, input: I) -> [F; DIGEST_ELEMS]
    where
        I: IntoIterator<Item = F>,
    {
        let words = self.inner.hash_iter(F::into_u64_stream(input));
        let bytes = u64_digest_to_bytes(&words);
        field_digest_from_le_u32_bytes(&bytes)
    }
}

/// Packed Merkle leaves for `u64`-stream inner hashers.
#[derive(Copy, Clone, Debug)]
pub struct FieldLeafPackedLanesFromU64Hash<
    Inner,
    F,
    PF,
    const N_U64: usize,
    const DIGEST_ELEMS: usize,
> {
    inner: Inner,
    _phantom: PhantomData<(F, PF)>,
}

impl<Inner, F, PF, const N_U64: usize, const DIGEST_ELEMS: usize>
    FieldLeafPackedLanesFromU64Hash<Inner, F, PF, N_U64, DIGEST_ELEMS>
{
    pub const fn new(inner: Inner) -> Self {
        const {
            assert!(
                8 * N_U64 >= 4 * DIGEST_ELEMS,
                "u64 digest does not contain enough bytes for DIGEST_ELEMS u32 limbs"
            );
        }
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<Inner, F, PF, const N_U64: usize, const DIGEST_ELEMS: usize>
    CryptographicHasher<PF, [PF; DIGEST_ELEMS]>
    for FieldLeafPackedLanesFromU64Hash<Inner, F, PF, N_U64, DIGEST_ELEMS>
where
    Inner: CryptographicHasher<u64, [u64; N_U64]> + Clone,
    F: Field + PrimeField,
    PF: PackedValue<Value = F>,
{
    fn hash_iter<I>(&self, input: I) -> [PF; DIGEST_ELEMS]
    where
        I: IntoIterator<Item = PF>,
    {
        let packed_inputs: Vec<PF> = input.into_iter().collect();
        let width = PF::WIDTH;
        let per_lane: Vec<[F; DIGEST_ELEMS]> = (0..width)
            .map(|lane| {
                let words = self.inner.hash_iter(F::into_u64_stream(
                    packed_inputs.iter().map(|p| p.extract(lane)),
                ));
                let bytes = u64_digest_to_bytes(&words);
                field_digest_from_le_u32_bytes(&bytes)
            })
            .collect();
        array::from_fn(|j| PF::from_fn(|lane| per_lane[lane][j]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CryptographicHasher;

    #[derive(Copy, Clone, Debug)]
    struct SumByteHasher;

    impl CryptographicHasher<u8, [u8; 32]> for SumByteHasher {
        fn hash_iter<I>(&self, input: I) -> [u8; 32]
        where
            I: IntoIterator<Item = u8>,
        {
            let s: u32 = input.into_iter().map(u32::from).sum();
            let mut out = [0u8; 32];
            out[..4].copy_from_slice(&s.to_le_bytes());
            out
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct SumU64Hasher;

    impl CryptographicHasher<u64, [u64; 4]> for SumU64Hasher {
        fn hash_iter<I>(&self, input: I) -> [u64; 4]
        where
            I: IntoIterator<Item = u64>,
        {
            let s: u64 = input.into_iter().sum();
            [s, 0, 0, 0]
        }
    }

    #[test]
    fn field_digest_from_bytes_u32_limbs() {
        use p3_field::PrimeCharacteristicRing;
        use p3_koala_bear::KoalaBear;

        type F = KoalaBear;
        let mut b = [0u8; 32];
        b[..4].copy_from_slice(&7u32.to_le_bytes());
        b[4..8].copy_from_slice(&11u32.to_le_bytes());
        let d = field_digest_from_le_u32_bytes::<F, 2>(&b);
        assert_eq!(d[0], F::from_u32(7));
        assert_eq!(d[1], F::from_u32(11));
    }

    #[test]
    fn field_digest_from_bytes_u64_limbs_goldilocks() {
        use p3_field::PrimeCharacteristicRing;
        use p3_goldilocks::Goldilocks;

        type F = Goldilocks;
        let mut b = [0u8; 64];
        b[..8].copy_from_slice(&0x0102_0304_0506_0708_u64.to_le_bytes());
        b[8..16].copy_from_slice(&0x1112_1314_1516_1718_u64.to_le_bytes());
        let d = field_digest_from_le_u64_bytes::<F, 2>(&b);
        assert_eq!(d[0], F::from_u64(0x0102_0304_0506_0708));
        assert_eq!(d[1], F::from_u64(0x1112_1314_1516_1718));
    }

    #[test]
    fn byte_hash_scalar_matches_direct_digest() {
        use p3_field::{PrimeCharacteristicRing, RawDataSerializable};
        use p3_koala_bear::KoalaBear;

        type F = KoalaBear;
        let h = FieldLeafFromByteHash::<SumByteHasher, F, 32, 2>::new(SumByteHasher);
        let d = h.hash_iter([F::from_u32(1), F::from_u32(2)]);
        let mut expected_bytes = [0u8; 32];
        let s: u32 = F::into_byte_stream([F::from_u32(1), F::from_u32(2)])
            .into_iter()
            .map(u32::from)
            .sum();
        expected_bytes[..4].copy_from_slice(&s.to_le_bytes());
        let expected = field_digest_from_le_u32_bytes::<F, 2>(&expected_bytes);
        assert_eq!(d, expected);
    }

    #[test]
    fn u64_hash_scalar_matches_direct_digest() {
        use p3_field::{PrimeCharacteristicRing, RawDataSerializable};
        use p3_koala_bear::KoalaBear;

        type F = KoalaBear;
        let h = FieldLeafFromU64Hash::<SumU64Hasher, F, 4, 2>::new(SumU64Hasher);
        let d = h.hash_iter([F::from_u32(10)]);
        let s: u64 = F::into_u64_stream([F::from_u32(10)]).into_iter().sum();
        let words = SumU64Hasher.hash_iter([s]);
        let bytes = u64_digest_to_bytes(&words);
        let expected = field_digest_from_le_u32_bytes::<F, 2>(&bytes);
        assert_eq!(d, expected);
    }
}
