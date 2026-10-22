//! The parameter set SLH-DSA-SHA2-128s (FIPS 205, Section 11, Table 2).

/// Bytes of a hash value, the security parameter `n`.
pub const N: usize = 16;

/// Height of the hypertree, `h`.
pub const H: usize = 63;

/// Layers of the hypertree, `d`.
pub const D: usize = 7;

/// Height of each XMSS tree, `h'`.
pub const H_PRIME: usize = H / D;

/// Height of each FORS tree, `a`.
pub const A: usize = 12;

/// FORS trees, `k`.
pub const K: usize = 14;

/// Bits of a Winternitz digit, `lg_w`.
pub const LG_W: usize = 4;

/// The Winternitz parameter `w`, the length of every chain plus one.
pub const W: usize = 1 << LG_W;

/// Bytes of the message digest, `m`.
pub const M: usize = 30;

/// Message digits of a WOTS+ signature, `len_1` (FIPS 205, Equation 5.1).
pub const LEN1: usize = 8 * N / LG_W;

/// Checksum digits of a WOTS+ signature, `len_2` (FIPS 205, Equation 5.3).
pub const LEN2: usize = (LEN1 * (W - 1)).ilog2() as usize / LG_W + 1;

/// Chains of a WOTS+ signature, `len`.
pub const LEN: usize = LEN1 + LEN2;

/// Bytes of the FORS part of a signature.
pub const FORS_BYTES: usize = K * (1 + A) * N;

/// Bytes of one XMSS signature: a WOTS+ signature and an authentication path.
pub const XMSS_BYTES: usize = (LEN + H_PRIME) * N;

/// Bytes of a signature (FIPS 205, Section 9.2).
pub const SIGNATURE_BYTES: usize = N + FORS_BYTES + D * XMSS_BYTES;

/// Bytes of a public key.
pub const PUBLIC_KEY_BYTES: usize = 2 * N;
