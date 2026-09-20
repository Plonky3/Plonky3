//! Length-prefixed absorption into one fingerprint.

/// A fingerprint under construction.
///
/// Every input of variable length is absorbed behind its own length.
pub(crate) struct Digest(blake3::Hasher);

impl Digest {
    /// Start a fingerprint under a fixed domain label.
    pub(crate) fn new(domain: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(domain.len() as u64).to_le_bytes());
        hasher.update(domain);
        Self(hasher)
    }

    /// Absorb one byte.
    pub(crate) fn byte(&mut self, value: u8) {
        self.0.update(&[value]);
    }

    /// Absorb a thirty-two-bit value.
    pub(crate) fn u32(&mut self, value: u32) {
        self.0.update(&value.to_le_bytes());
    }

    /// Absorb a count, widened so the encoding does not depend on the host.
    pub(crate) fn usize(&mut self, value: usize) {
        self.0.update(&(value as u64).to_le_bytes());
    }

    /// Absorb a byte string behind its own length.
    pub(crate) fn bytes(&mut self, value: &[u8]) {
        self.usize(value.len());
        self.0.update(value);
    }

    /// Close the fingerprint.
    pub(crate) fn finish(self) -> [u8; 32] {
        *self.0.finalize().as_bytes()
    }
}
