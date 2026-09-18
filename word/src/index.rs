use thiserror::Error;

/// A word's visibility segment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Segment {
    /// Values supplied as part of the public statement.
    Public,
    /// Values committed by the prover.
    Witness,
}

/// An error returned when a word position cannot be represented.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum IndexError {
    /// The position exceeds the compact 32-bit representation.
    #[error("word position {position} exceeds u32::MAX")]
    PositionTooLarge {
        /// The rejected position.
        position: usize,
    },
}

/// An opaque position in either the public or committed word segment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ValueIndex {
    segment: Segment,
    position: u32,
}

impl ValueIndex {
    /// Creates an index into the public segment.
    pub fn public(position: usize) -> Result<Self, IndexError> {
        Self::new(Segment::Public, position)
    }

    /// Creates an index into the committed witness segment.
    pub fn witness(position: usize) -> Result<Self, IndexError> {
        Self::new(Segment::Witness, position)
    }

    fn new(segment: Segment, position: usize) -> Result<Self, IndexError> {
        // A compact index bounds later setup arithmetic and proof metadata.
        let position =
            u32::try_from(position).map_err(|_| IndexError::PositionTooLarge { position })?;
        Ok(Self { segment, position })
    }

    /// Returns the segment containing the word.
    #[inline]
    pub const fn segment(self) -> Segment {
        self.segment
    }

    /// Returns the position within the selected segment.
    #[inline]
    pub const fn position(self) -> u32 {
        self.position
    }
}
