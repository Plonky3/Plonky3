//! Wires and words: the output of one gate or a structural zero, and runs of them.

use alloc::vec::Vec;
use core::ops::{Deref, Shr};

/// A wire: the output of one gate, or a structural zero that needs no gate.
///
/// Only a builder mints gate wires, so every wire names a gate of its circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Wire(Option<u32>);

impl Wire {
    /// The structural zero.
    pub const ZERO: Self = Self(None);

    /// The output of a gate.
    pub(crate) const fn gate(index: u32) -> Self {
        Self(Some(index))
    }

    /// Whether this is the structural zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0.is_none()
    }

    /// Index of the driving gate, or none for the structural zero.
    pub(crate) const fn index(self) -> Option<u32> {
        self.0
    }
}

/// A word's wires, low bit first; rotations and shifts only rename wires, so they are free.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Word(Vec<Wire>);

impl Word {
    /// The word rotated right by `r` bits.
    #[must_use]
    pub fn rotate_right(&self, r: usize) -> Self {
        let n = self.0.len();
        (0..n).map(|i| self.0[(i + r) % n]).collect()
    }

    /// The word rotated left by `r` bits.
    #[must_use]
    pub fn rotate_left(&self, r: usize) -> Self {
        self.rotate_right(self.0.len() - r % self.0.len())
    }

    /// A run of wires as 32-bit words, low word first; a partial last word is dropped.
    #[must_use]
    pub fn split_u32(wires: &[Wire]) -> Vec<Self> {
        wires
            .as_chunks::<32>()
            .0
            .iter()
            .map(|w| Self(w.to_vec()))
            .collect()
    }
}

impl Deref for Word {
    type Target = [Wire];

    fn deref(&self) -> &[Wire] {
        &self.0
    }
}

impl From<&[Wire]> for Word {
    fn from(wires: &[Wire]) -> Self {
        Self(wires.to_vec())
    }
}

impl FromIterator<Wire> for Word {
    fn from_iter<I: IntoIterator<Item = Wire>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<'a> IntoIterator for &'a Word {
    type Item = &'a Wire;
    type IntoIter = core::slice::Iter<'a, Wire>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl Shr<usize> for &Word {
    type Output = Word;

    /// The word shifted right by `r` bits, zeros shifted in.
    fn shr(self, r: usize) -> Word {
        (0..self.0.len())
            .map(|i| self.0.get(i + r).copied().unwrap_or(Wire::ZERO))
            .collect()
    }
}
