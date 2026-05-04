//! SHAKE128 sponge adapter matching the IETF draft §8.1 reference.

use alloc::vec;
use alloc::vec::Vec;

use tiny_keccak::{Hasher, Shake, Xof};

use crate::{CanObserve, CanSample};

/// SHAKE128 rate, in bytes.
const SHAKE128_RATE: usize = 168;

/// IETF protocol-identifier length, in bytes.
pub const IV_LEN: usize = 64;

/// Streaming SHAKE128 sponge with the IETF initial-block convention.
///
/// - Bytes accumulate end-to-end after a 168-byte initial block.
/// - Each squeeze re-hashes the whole accumulator from scratch.
/// - Two consecutive squeezes return overlapping bytes, matching
///   `hashlib.shake_128().copy().digest(n)`.
///
/// # Limitations
///
/// Not a true duplex sponge: there is no incremental absorb-permute-squeeze.
/// For that, use a duplex challenger over a cryptographic permutation.
#[derive(Clone, Debug)]
pub struct Shake128 {
    /// All bytes absorbed so far, including the 168-byte initial block.
    absorbed: Vec<u8>,
}

impl Shake128 {
    /// Build a sponge seeded with a 64-byte IV.
    ///
    /// The initial block is `[iv (64 bytes) | 0x00 * 104]`, exactly one rate block.
    #[must_use]
    pub fn new(iv: &[u8; IV_LEN]) -> Self {
        let mut absorbed = Vec::with_capacity(SHAKE128_RATE);
        // IV occupies the first 64 bytes.
        absorbed.extend_from_slice(iv);
        // Pad to one full rate block so the first permute call sees a fixed prefix.
        absorbed.extend(core::iter::repeat_n(0u8, SHAKE128_RATE - IV_LEN));
        Self { absorbed }
    }

    /// Append `bytes` to the absorbed input.
    pub fn absorb(&mut self, bytes: &[u8]) {
        self.absorbed.extend_from_slice(bytes);
    }

    /// Return the first `length` bytes of SHAKE128 over the accumulator.
    #[must_use]
    pub fn squeeze_into_vec(&self, length: usize) -> Vec<u8> {
        // Fresh hasher per squeeze, mirroring `state.copy().digest` from the IETF Python reference.
        let mut shake = Shake::v128();
        shake.update(&self.absorbed);
        // One contiguous output of `length` bytes equals one `digest(length)` call.
        let mut out = vec![0u8; length];
        shake.squeeze(&mut out);
        out
    }
}

impl CanObserve<u8> for Shake128 {
    fn observe(&mut self, value: u8) {
        self.absorbed.push(value);
    }

    fn observe_slice(&mut self, values: &[u8]) {
        // Bulk path skips per-byte trait dispatch.
        self.absorbed.extend_from_slice(values);
    }
}

impl CanSample<u8> for Shake128 {
    fn sample(&mut self) -> u8 {
        // No "sample one byte" primitive in the IETF reference, so route through a 1-byte squeeze.
        //
        // Bulk callers should use `squeeze_into_vec` directly to avoid the per-byte rebuild cost.
        self.squeeze_into_vec(1)[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode an ASCII hex string into raw bytes.
    ///
    /// # Panics
    ///
    /// On odd-length input or any non-hex character.
    fn hex_to_bytes(hex: &str) -> Vec<u8> {
        assert!(
            hex.len().is_multiple_of(2),
            "hex string length must be even"
        );
        let mut out = Vec::with_capacity(hex.len() / 2);
        for chunk in hex.as_bytes().chunks(2) {
            // Two ASCII nibbles per output byte.
            let hi = decode_nibble(chunk[0]);
            let lo = decode_nibble(chunk[1]);
            out.push((hi << 4) | lo);
        }
        out
    }

    /// ASCII hex digit -> integer 0..=15.
    fn decode_nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => panic!("non-hex character: {c:?}"),
        }
    }

    /// Decode a 64-byte IV from a hex string.
    fn iv_from_hex(hex: &str) -> [u8; IV_LEN] {
        let raw = hex_to_bytes(hex);
        assert!(raw.len() == IV_LEN, "IV must be exactly {IV_LEN} bytes");
        let mut iv = [0u8; IV_LEN];
        iv.copy_from_slice(&raw);
        iv
    }

    /// IV shared by the IETF unit tests below.
    fn unit_tests_iv() -> [u8; IV_LEN] {
        iv_from_hex(
            "756e69745f74657374735f6b656363616b5f697600000000000000000000000\
             00000000000000000000000000000000000000000000000000000000000000000",
        )
    }

    // Local unit behaviour.

    #[test]
    fn initial_block_layout_is_iv_then_104_zeros() {
        // Constructor leaves the buffer as [iv | 0x00 * 104].
        let iv = [0xa5u8; IV_LEN];
        let s = Shake128::new(&iv);
        assert_eq!(s.absorbed.len(), SHAKE128_RATE);
        assert_eq!(&s.absorbed[..IV_LEN], iv.as_slice());
        assert!(s.absorbed[IV_LEN..].iter().all(|&b| b == 0));
    }

    #[test]
    fn empty_squeeze_returns_empty_vec() {
        // Zero-length squeeze does not need a hasher round.
        let iv = [0u8; IV_LEN];
        let s = Shake128::new(&iv);
        let out = s.squeeze_into_vec(0);
        assert!(out.is_empty());
    }

    #[test]
    fn squeeze_is_idempotent_when_no_absorb_in_between() {
        // Two squeezes of the same length on the same accumulator agree.
        let iv = [0u8; IV_LEN];
        let mut s = Shake128::new(&iv);
        s.absorb(b"data");
        let a = s.squeeze_into_vec(32);
        let b = s.squeeze_into_vec(32);
        assert_eq!(a, b);
    }

    #[test]
    fn longer_squeeze_extends_shorter_squeeze() {
        // SHAKE128 is a stream; longer outputs are extensions of shorter ones.
        let iv = [0u8; IV_LEN];
        let mut s = Shake128::new(&iv);
        s.absorb(b"data");
        let short = s.squeeze_into_vec(16);
        let long = s.squeeze_into_vec(64);
        assert_eq!(&long[..16], short.as_slice());
    }

    // IETF Appendix A test vectors.

    #[test]
    fn ietf_a1_basic_duplex_sponge() {
        // Appendix A.1: absorb "basic duplex sponge test", squeeze 64 bytes.
        let mut s = Shake128::new(&unit_tests_iv());
        s.absorb(&hex_to_bytes(
            "6261736963206475706c65782073706f6e67652074657374",
        ));
        let out = s.squeeze_into_vec(64);

        let expected = hex_to_bytes(
            "f845c3ef4231a4d6e09c29b1eea0055842246fd57558fd7d93e1302f7\
             799dd9593d2e4d06eda72d5252ca5b2feff4b8cb324ec96673a7417cf70fa77b1898\
             991",
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn ietf_a2_absorb_empty_before_does_not_break() {
        // Appendix A.2: empty absorb after a non-empty one is a no-op.
        let mut s = Shake128::new(&unit_tests_iv());
        s.absorb(&hex_to_bytes("656d707479206d657373616765206166746572"));
        s.absorb(&hex_to_bytes(""));
        let out = s.squeeze_into_vec(64);

        let expected = hex_to_bytes(
            "3953e577d9e5d4dc7b86d1a62e881f2d1eb750ea3550fcae315854d16\
             6136ae816ca922a4c7e54d711b8721c8969598449922122768c50313f47eef35020b\
             73c",
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn ietf_a3_absorb_empty_after_does_not_break() {
        // Appendix A.3: leading empty absorb is also a no-op.
        let mut s = Shake128::new(&unit_tests_iv());
        s.absorb(&hex_to_bytes(""));
        s.absorb(&hex_to_bytes("656d707479206d657373616765206265666f7265"));
        let out = s.squeeze_into_vec(64);

        let expected = hex_to_bytes(
            "6e475edd3c400bec314d5891af570841a547c95d1a651adff9a8bfb70\
             719a79b5afde316386da13fa83525662df3c5b2367d987bf3dc4199efdb9d0612572\
             785",
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn ietf_a4_squeeze_zero_behaviour() {
        // Appendix A.4: zero-length squeezes have no observable effect.
        let mut s = Shake128::new(&unit_tests_iv());
        let _ = s.squeeze_into_vec(0);
        s.absorb(&hex_to_bytes("7a65726f2073717565657a652074657374"));
        let _ = s.squeeze_into_vec(0);
        let out = s.squeeze_into_vec(64);

        let expected = hex_to_bytes(
            "4cf7f008057b63cb615547a143f42cf793b86b239f404d2f28b3f0919\
             7d850eb029df3024ad468be5aceb2fa60e9fb7add98436236be69ddb34314ce7a905\
             f23",
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn ietf_a5_squeeze_zero_after_behaviour() {
        // Appendix A.5: leading zero-length squeeze has no effect.
        let mut s = Shake128::new(&unit_tests_iv());
        let _ = s.squeeze_into_vec(0);
        s.absorb(&hex_to_bytes("7a65726f2073717565657a65206166746572"));
        let out = s.squeeze_into_vec(64);

        let expected = hex_to_bytes(
            "bd9278e6f65cb854935b3f6b2c51ab158be8ea09744509519b8f06f0c\
             501d07c429e37f232b6f0955b620ff6226d9d02e4817b1447e7309023a3a14f73587\
             6ec",
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn ietf_a6_absorb_squeeze_absorb_consistency() {
        // Appendix A.6: squeezes never advance state; the asserted bytes
        // are SHAKE128 over the concatenation of both absorbs.
        let iv = iv_from_hex(
            "656467652d636173652d746573742d646f6d61696e2d6162736f72620000000\
             00000000000000000000000000000000000000000000000000000000000000000",
        );
        let mut s = Shake128::new(&iv);
        // First absorb + a discarded 32-byte squeeze.
        s.absorb(&hex_to_bytes("696e7465726c65617665206669727374"));
        let _ = s.squeeze_into_vec(32);
        // Second absorb + the 32-byte squeeze that the spec asserts.
        s.absorb(&hex_to_bytes("696e7465726c65617665207365636f6e64"));
        let out = s.squeeze_into_vec(32);

        let expected = hex_to_bytes(
            "4d31a75f29851f9f15cd54fa6f2335cbe07b947b9d3c28092c1ba7315\
             e295921",
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn ietf_a7_associativity_of_absorb() {
        // Appendix A.7: one big absorb equals the same bytes split across calls.
        let iv = iv_from_hex(
            "6162736f72622d6173736f6369617469766974792d646f6d61696e000000000\
             00000000000000000000000000000000000000000000000000000000000000000",
        );
        let mut s = Shake128::new(&iv);
        s.absorb(&hex_to_bytes(
            "6173736f63696174697669747920746573742066756c6c",
        ));
        let out = s.squeeze_into_vec(32);

        let expected = hex_to_bytes(
            "c0faa351141d60678dceff4f3a5760381bb335ad113958b70edf7b242\
             df01c8a",
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn ietf_a8_iv_affects_output() {
        // Appendix A.8: changing the IV flips the output bytes.
        let iv = iv_from_hex(
            "646f6d61696e2d6f6e652d646966666572732d6865726500000000000000000\
             00000000000000000000000000000000000000000000000000000000000000000",
        );
        let mut s = Shake128::new(&iv);
        s.absorb(&hex_to_bytes("697620646966666572656e63652074657374"));
        let out = s.squeeze_into_vec(32);

        let expected = hex_to_bytes(
            "7650642267cc544abf0e01ce28e2595aec4c2f5b5e5e3720ab5514496\
             37b35f2",
        );
        assert_eq!(out, expected);
    }

    /// Brute-force the byte count the IETF reference used for the multi-block KAT.
    ///
    /// The spec line-wraps the absorb hex string, so a literal copy does not
    /// preserve the length. Run once to recover the count, then disable.
    #[test]
    #[ignore = "manual: enable to discover the IETF KAT input length"]
    fn discover_ietf_a9_input_length() {
        let iv = iv_from_hex(
            "6d756c74692d626c6f636b2d6162736f72622d7465737400000000000000000\
             00000000000000000000000000000000000000000000000000000000000000000",
        );
        let expected_prefix: [u8; 8] = [0x52, 0x6d, 0x4f, 0x6c, 0xfc, 0xa2, 0x30, 0xe0];
        for n in 1..=4096usize {
            let input = vec![0xabu8; n];
            let mut s = Shake128::new(&iv);
            s.absorb(&input);
            let out = s.squeeze_into_vec(8);
            if out == expected_prefix {
                panic!("matched at n = {n}");
            }
        }
        panic!("no length matched");
    }

    #[test]
    fn ietf_a9_multiple_blocks_absorb_squeeze() {
        // Appendix A.9: absorb 600 copies of 0xab, squeeze 600 bytes.
        // The 600 figure was recovered via the brute-force test above.
        let iv = iv_from_hex(
            "6d756c74692d626c6f636b2d6162736f72622d7465737400000000000000000\
             00000000000000000000000000000000000000000000000000000000000000000",
        );
        let mut s = Shake128::new(&iv);
        let input = vec![0xabu8; 600];
        s.absorb(&input);
        let out = s.squeeze_into_vec(600);

        let expected = hex_to_bytes(
            "526d4f6cfca230e0654bf8749bddc0f4416a8a164c50f3c1b0bc1d527\
             2a88b9a524e73cafad76691a29c0e03a5255fd8fb9d778ef5a0c8c9e11e003011d25\
             6bf92dd36233e4c6c360baca0f8ac305d459adb1231a801742669efa051396e96417\
             814448b5328336d028a62dbddf24d1bb68496d27f1944eb24d4b2812d9ad4eae6c26\
             0b720c44ed2be8bfeeed3acc2640edbab987674f2cef8ceacda1e04f254170aba424\
             1dabc6364ed5afc09b58205682d5e8413bf5f9d97e9c799b97876ccd1c48d86759ad\
             e5871acc4c5d41d37f2b1843c8b6f9e0bade78342d56f9b1e8232d4c7553674d889e\
             69fe24dea31f42f0b02b70161876ceb12cc0b36868c262cbebb5e815a1eceaee97ae\
             d3402a518287c32f2f469c3a38a17afd0f0d82433acf695ae143ded9412b4e6b6144\
             bd6d4be6bb7de33c05f560480c63aa89336954f1cf5992399e6ed59d406adb4497bb\
             88aa897fd3d65646cf86e796da4f193c418a74d662f57e0e0c775386abdace02157e\
             519ba54495555145016c550ff32004981d0e34f0abe7d814ac4fe25260473ffa8746\
             0a736f20954e8d3b9f16140e79451953fe6cfc222cba6ad4f85a2e2efd6ff8f5fef6\
             5d8480e6af40baab298c4de57f30d08a5e1b4c10d123a5af7702ff26ba9a84a6fe92\
             f48391b23a7e8e8cb06deda74d1b10870611995f6bfe4df60320a0b7f2c891cad5a5\
             645ecec80868ed568591a74dafb35cabb42dae1a1085269b655db1ebf09929f63d5a\
             f775a24e43759f673b83aeefef382bc2b7bf175bb9d90e77911466ffb3b230754776\
             5cd5adc30a6b07881a88fd1511e5f8d2dcc4347c076e6c79676d8df",
        );
        assert_eq!(out, expected);
    }
}
