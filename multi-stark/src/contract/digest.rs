//! Length-prefixed absorption into one canonical byte string.

use alloc::vec::Vec;

use p3_symmetric::CryptographicHasher;

/// A canonical byte string under construction.
///
/// Every input of variable length is absorbed behind its own length.
pub(crate) struct Preimage(Vec<u8>);

impl Preimage {
    /// Start a byte string under a fixed domain label.
    pub(crate) fn new(domain: &[u8]) -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(domain.len() as u64).to_le_bytes());
        bytes.extend_from_slice(domain);
        Self(bytes)
    }

    /// Absorb a thirty-two-bit value.
    pub(crate) fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    /// Absorb a count, widened so the encoding does not depend on the host.
    pub(crate) fn usize(&mut self, value: usize) {
        self.0.extend_from_slice(&(value as u64).to_le_bytes());
    }

    /// Absorb a byte string behind its own length.
    pub(crate) fn bytes(&mut self, value: &[u8]) {
        self.usize(value.len());
        self.0.extend_from_slice(value);
    }

    /// Close the byte string and hash it with the caller's choice of hash.
    pub(crate) fn finish<H>(self, hasher: &H) -> [u8; 32]
    where
        H: CryptographicHasher<u8, [u8; 32]>,
    {
        hasher.hash_iter(self.0)
    }
}

#[cfg(test)]
mod tests {
    use p3_keccak::Keccak256Hash;

    use super::*;

    fn fingerprint(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
        let mut preimage = Preimage::new(domain);
        for part in parts {
            preimage.bytes(part);
        }
        preimage.finish(&Keccak256Hash)
    }

    #[test]
    fn splitting_one_input_differently_changes_the_fingerprint() {
        // Without the length prefixes both sequences would concatenate to the same bytes.
        assert_ne!(
            fingerprint(b"domain", &[b"ab", b"c"]),
            fingerprint(b"domain", &[b"a", b"bc"])
        );
        assert_ne!(
            fingerprint(b"domain", &[b"abc"]),
            fingerprint(b"domain", &[b"ab", b"c"])
        );
    }

    #[test]
    fn an_empty_input_is_not_the_absence_of_one() {
        assert_ne!(fingerprint(b"domain", &[]), fingerprint(b"domain", &[b""]));
    }

    #[test]
    fn the_domain_label_separates_identical_content() {
        // The label sits behind its own length, so a longer label cannot swallow the rest.
        assert_ne!(fingerprint(b"a", &[b"bc"]), fingerprint(b"ab", &[b"c"]));
    }

    #[test]
    fn the_scalar_widths_are_fixed() {
        // A count and a half-width value of the same number must not absorb the same bytes.
        let mut wide = Preimage::new(b"domain");
        wide.usize(1);
        let mut narrow = Preimage::new(b"domain");
        narrow.u32(1);
        assert_ne!(wide.finish(&Keccak256Hash), narrow.finish(&Keccak256Hash));

        let mut half = Preimage::new(b"domain");
        half.u32(1);
        assert_ne!(
            half.finish(&Keccak256Hash),
            Preimage::new(b"domain").finish(&Keccak256Hash)
        );
    }
}
