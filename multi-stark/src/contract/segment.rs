//! Public claims that let one proof pick up where another left off.
//!
//! A long execution is proved as several segments, one proof each.
//!
//! Each proof exposes the state it starts from and the state it leaves behind.
//!
//! Two segments chain when the state one leaves is the state the next starts from:
//!
//! ```text
//!     segment 0 : entry E0 ──► exit X0
//!     segment 1 : entry E1 ──► exit X1        chains when X0 = E1
//!     segment 2 : entry E2 ──► exit X2        chains when X1 = E2
//! ```
//!
//! The backend does not know what a state is.
//!
//! A statement names which public values form each side, and the claim hashes them.
//!
//! Only a verified claim chains, and only with claims of the same statement.
//!
//! What those values mean, and which statements may follow which, stays with the machine.

use alloc::vec::Vec;

use p3_field::Field;
use p3_symmetric::CryptographicHasher;
use thiserror::Error;

use crate::contract::digest::Preimage;
use crate::contract::error::DeclarationError;
use crate::contract::table::TableDeclaration;

/// Domain label of the digest that summarizes one side of a segment.
const BOUNDARY_DOMAIN: &[u8] = b"p3-backend-contract/segment-boundary/v1";

/// One public value of one table, named by position.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PublicSlot {
    /// Position of the table in declaration order.
    pub table: usize,
    /// Position of the value among that table's public values.
    pub index: usize,
}

impl PublicSlot {
    /// Name one public value of one table.
    #[must_use]
    pub const fn new(table: usize, index: usize) -> Self {
        Self { table, index }
    }
}

/// Which public values a proof exposes as the state it starts from and the state it leaves.
///
/// Both sides list the same number of slots, since one segment's exit is the next one's entry.
///
/// The order of the slots is the order the values are hashed in.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SegmentInterface {
    /// Slots whose values form the state the segment starts from.
    entry: Vec<PublicSlot>,
    /// Slots whose values form the state the segment leaves behind.
    exit: Vec<PublicSlot>,
}

impl SegmentInterface {
    /// Name the slots of both sides.
    ///
    /// Nothing is checked here, since the slots only mean something against a set of tables.
    ///
    /// The declaration that adopts the interface checks them.
    #[must_use]
    pub const fn new(entry: Vec<PublicSlot>, exit: Vec<PublicSlot>) -> Self {
        Self { entry, exit }
    }

    /// Slots whose values form the state the segment starts from.
    #[must_use]
    pub fn entry(&self) -> &[PublicSlot] {
        &self.entry
    }

    /// Slots whose values form the state the segment leaves behind.
    #[must_use]
    pub fn exit(&self) -> &[PublicSlot] {
        &self.exit
    }

    /// Refuse an interface that names a value no table declares.
    pub(super) fn validate(&self, tables: &[TableDeclaration]) -> Result<(), DeclarationError> {
        if self.entry.is_empty() {
            return Err(DeclarationError::EmptySegmentBoundary);
        }
        if self.entry.len() != self.exit.len() {
            return Err(DeclarationError::SegmentArityMismatch {
                entry: self.entry.len(),
                exit: self.exit.len(),
            });
        }

        for &slot in self.entry.iter().chain(&self.exit) {
            let declared = tables
                .get(slot.table)
                .map(|table| table.columns().public)
                .ok_or(DeclarationError::SlotOutOfRange {
                    table: slot.table,
                    index: slot.index,
                })?;
            if slot.index >= declared {
                return Err(DeclarationError::SlotOutOfRange {
                    table: slot.table,
                    index: slot.index,
                });
            }
        }
        Ok(())
    }

    /// Absorb both sides into a statement fingerprint.
    pub(super) fn absorb(&self, preimage: &mut Preimage) {
        for side in [&self.entry, &self.exit] {
            preimage.usize(side.len());
            for slot in side {
                preimage.usize(slot.table);
                preimage.usize(slot.index);
            }
        }
    }

    /// Hash the values both sides name out of every table's public values.
    ///
    /// # Errors
    ///
    /// Returns an error when the public values do not have the shape the tables declare.
    pub(super) fn claim<F, H>(
        &self,
        hasher: &H,
        statement: [u8; 32],
        tables: &[TableDeclaration],
        public_values: &[&[F]],
    ) -> Result<SegmentClaim, DeclarationError>
    where
        F: Field,
        H: CryptographicHasher<u8, [u8; 32]>,
    {
        if public_values.len() != tables.len() {
            return Err(DeclarationError::PublicValueTables {
                expected: tables.len(),
                found: public_values.len(),
            });
        }
        for (table, (declared, values)) in tables.iter().zip(public_values).enumerate() {
            let expected = declared.columns().public;
            if values.len() != expected {
                return Err(DeclarationError::PublicValueCount {
                    table,
                    expected,
                    found: values.len(),
                });
            }
        }

        // Every slot was checked against the declared counts, which the values now match.
        let side = |slots: &[PublicSlot]| {
            boundary_digest(
                hasher,
                slots
                    .iter()
                    .map(|slot| public_values[slot.table][slot.index]),
            )
        };

        Ok(SegmentClaim {
            statement,
            entry: side(&self.entry),
            exit: side(&self.exit),
        })
    }

    /// Hash the values of one side, listed in slot order.
    ///
    /// Both sides name the same number of slots, so one count serves either.
    ///
    /// # Errors
    ///
    /// Returns an error when the values are not one per slot.
    pub(super) fn boundary<F, H>(
        &self,
        hasher: &H,
        values: &[F],
    ) -> Result<[u8; 32], DeclarationError>
    where
        F: Field,
        H: CryptographicHasher<u8, [u8; 32]>,
    {
        if values.len() != self.entry.len() {
            return Err(DeclarationError::BoundaryValueCount {
                expected: self.entry.len(),
                found: values.len(),
            });
        }
        Ok(boundary_digest(hasher, values.iter().copied()))
    }
}

/// Digest of one side of a segment, over its values in slot order.
///
/// The count leads, and each value follows in its canonical encoding.
fn boundary_digest<F, H>(hasher: &H, values: impl ExactSizeIterator<Item = F>) -> [u8; 32]
where
    F: Field,
    H: CryptographicHasher<u8, [u8; 32]>,
{
    let mut preimage = Preimage::new(BOUNDARY_DOMAIN);
    preimage.usize(values.len());
    for value in values {
        let encoded = postcard::to_allocvec(&value)
            .expect("serializing a field element into memory cannot fail");
        preimage.bytes(&encoded);
    }
    preimage.finish(hasher)
}

/// The boundary one proof commits to, as two digests and the statement it was proved under.
///
/// A claim alone proves nothing: a prover computes one before any proof exists.
///
/// Only [`VerifiedSegment`] carries a claim some proof has backed.
///
/// The digests cover values only, so two statements that share a boundary encoding share digests.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SegmentClaim {
    /// Fingerprint of the statement the segment is proved under.
    statement: [u8; 32],
    /// Digest of the state the segment starts from.
    entry: [u8; 32],
    /// Digest of the state the segment leaves behind.
    exit: [u8; 32],
}

impl SegmentClaim {
    /// Fingerprint of the statement the segment is proved under.
    #[must_use]
    pub const fn statement(&self) -> [u8; 32] {
        self.statement
    }

    /// Digest of the state the segment starts from.
    #[must_use]
    pub const fn entry(&self) -> [u8; 32] {
        self.entry
    }

    /// Digest of the state the segment leaves behind.
    #[must_use]
    pub const fn exit(&self) -> [u8; 32] {
        self.exit
    }
}

/// A segment claim that a proof has just been verified against.
///
/// Only a verification that passed builds one, so a predicted claim cannot pose as one.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct VerifiedSegment {
    /// The boundary the verified proof commits to.
    claim: SegmentClaim,
}

impl VerifiedSegment {
    /// Record the claim of a proof that verified.
    pub(super) const fn new(claim: SegmentClaim) -> Self {
        Self { claim }
    }

    /// The boundary the verified proof commits to.
    #[must_use]
    pub const fn claim(&self) -> SegmentClaim {
        self.claim
    }
}

/// What a chain of verified segments proves as a whole.
///
/// Every segment was proved under the one statement recorded here.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ChainedExecution {
    /// Fingerprint of the statement every segment was proved under.
    statement: [u8; 32],
    /// Digest of the state the first segment starts from.
    entry: [u8; 32],
    /// Digest of the state the last segment leaves behind.
    exit: [u8; 32],
    /// Number of segments in the chain.
    segments: usize,
}

impl ChainedExecution {
    /// Fingerprint of the statement every segment was proved under.
    #[must_use]
    pub const fn statement(&self) -> [u8; 32] {
        self.statement
    }

    /// Digest of the state the first segment starts from.
    ///
    /// A checker compares it with the digest of the start it expects.
    #[must_use]
    pub const fn entry(&self) -> [u8; 32] {
        self.entry
    }

    /// Digest of the state the last segment leaves behind.
    #[must_use]
    pub const fn exit(&self) -> [u8; 32] {
        self.exit
    }

    /// Number of segments in the chain.
    #[must_use]
    pub const fn segments(&self) -> usize {
        self.segments
    }
}

/// Why a list of verified segments does not form one execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum ChainError {
    /// An empty list describes no execution.
    #[error("a chain needs at least one segment")]
    Empty,
    /// A segment was proved under a different statement than the first one.
    #[error("segment {segment} was proved under a different statement than segment 0")]
    MixedStatements {
        /// Position of the first segment whose statement differs.
        segment: usize,
    },
    /// A segment does not start where the one before it stopped.
    #[error("segment {segment} does not start where segment {} stopped", segment - 1)]
    Broken {
        /// Position of the segment whose entry disagrees with the previous exit.
        segment: usize,
    },
}

/// Join verified segments into one execution, in order.
///
/// Every segment must be proved under one statement.
///
/// Two statements may share a boundary encoding and still mean different machines.
///
/// A chain across statements is therefore refused rather than left to the caller to notice.
///
/// # Errors
///
/// Returns an error for an empty list.
///
/// Returns an error at the first segment proved under another statement.
///
/// Returns an error at the first segment that does not start where its predecessor stopped.
pub fn chain(segments: &[VerifiedSegment]) -> Result<ChainedExecution, ChainError> {
    let (first, last) = segments
        .first()
        .zip(segments.last())
        .ok_or(ChainError::Empty)?;
    let statement = first.claim.statement;

    for (position, pair) in segments.windows(2).enumerate() {
        let (previous, next) = (pair[0].claim, pair[1].claim);
        if next.statement != statement {
            return Err(ChainError::MixedStatements {
                segment: position + 1,
            });
        }
        if previous.exit != next.entry {
            return Err(ChainError::Broken {
                segment: position + 1,
            });
        }
    }

    Ok(ChainedExecution {
        statement,
        entry: first.claim.entry,
        exit: last.claim.exit,
        segments: segments.len(),
    })
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn under(statement: u8, entry: u8, exit: u8) -> VerifiedSegment {
        VerifiedSegment::new(SegmentClaim {
            statement: [statement; 32],
            entry: [entry; 32],
            exit: [exit; 32],
        })
    }

    fn claim(entry: u8, exit: u8) -> VerifiedSegment {
        under(0, entry, exit)
    }

    #[test]
    fn a_single_segment_is_its_own_chain() {
        assert_eq!(
            chain(&[claim(1, 2)]).unwrap(),
            ChainedExecution {
                statement: [0; 32],
                entry: [1; 32],
                exit: [2; 32],
                segments: 1,
            }
        );
    }

    #[test]
    fn segments_chain_when_each_starts_where_the_last_stopped() {
        let joined = chain(&[claim(1, 2), claim(2, 3), claim(3, 4)]).unwrap();
        assert_eq!(joined.entry(), [1; 32]);
        assert_eq!(joined.exit(), [4; 32]);
        assert_eq!(joined.segments(), 3);
    }

    #[test]
    fn segments_of_another_statement_do_not_chain() {
        // The boundaries meet, but the second segment was proved under statement 7.
        assert_eq!(
            chain(&[claim(1, 2), under(7, 2, 3), claim(3, 4)]).unwrap_err(),
            ChainError::MixedStatements { segment: 1 }
        );
        // A later segment is compared with the first, not only with its neighbour.
        assert_eq!(
            chain(&[under(7, 1, 2), under(7, 2, 3), claim(3, 4)]).unwrap_err(),
            ChainError::MixedStatements { segment: 2 }
        );
        // The chain records the one statement every segment shares.
        assert_eq!(chain(&[under(7, 1, 2)]).unwrap().statement(), [7; 32]);
    }

    #[test]
    fn the_first_break_is_named() {
        // Segment two starts at 9 where segment one stopped at 3.
        assert_eq!(
            chain(&[claim(1, 2), claim(2, 3), claim(9, 4), claim(5, 6)]).unwrap_err(),
            ChainError::Broken { segment: 2 }
        );
    }

    #[test]
    fn an_empty_list_is_refused() {
        assert_eq!(chain(&[]).unwrap_err(), ChainError::Empty);
    }

    #[test]
    fn order_matters() {
        // The same two segments reversed leave a gap at the join.
        assert!(chain(&[claim(1, 2), claim(2, 3)]).is_ok());
        assert_eq!(
            chain(&[claim(2, 3), claim(1, 2)]).unwrap_err(),
            ChainError::Broken { segment: 1 }
        );
    }

    #[test]
    fn an_interface_is_checked_against_the_tables() {
        // No table is declared, so any slot is out of range.
        let interface =
            SegmentInterface::new(vec![PublicSlot::new(0, 0)], vec![PublicSlot::new(0, 1)]);
        assert_eq!(
            interface.validate(&[]).unwrap_err(),
            DeclarationError::SlotOutOfRange { table: 0, index: 0 }
        );

        // Two slots in, one slot out, cannot chain with itself.
        let lopsided = SegmentInterface::new(
            vec![PublicSlot::new(0, 0), PublicSlot::new(0, 1)],
            vec![PublicSlot::new(0, 2)],
        );
        assert_eq!(
            lopsided.validate(&[]).unwrap_err(),
            DeclarationError::SegmentArityMismatch { entry: 2, exit: 1 }
        );

        // A boundary with no values would chain anything to anything.
        assert_eq!(
            SegmentInterface::new(vec![], vec![])
                .validate(&[])
                .unwrap_err(),
            DeclarationError::EmptySegmentBoundary
        );
    }
}
