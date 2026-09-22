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
        let digest = |side: &[PublicSlot]| {
            let mut preimage = Preimage::new(BOUNDARY_DOMAIN);
            preimage.usize(side.len());
            for slot in side {
                let value = public_values[slot.table][slot.index];
                let encoded = postcard::to_allocvec(&value)
                    .expect("serializing a field element into memory cannot fail");
                preimage.bytes(&encoded);
            }
            preimage.finish(hasher)
        };

        Ok(SegmentClaim {
            statement,
            entry: digest(&self.entry),
            exit: digest(&self.exit),
        })
    }
}

/// The boundary one proof commits to, as two digests and the statement it was proved under.
///
/// The digests cover values only.
///
/// Two statements that share a boundary encoding can therefore chain.
///
/// Whether they should is continuation policy, and the statement digest is here to decide it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SegmentClaim {
    /// Fingerprint of the statement the segment was proved under.
    statement: [u8; 32],
    /// Digest of the state the segment starts from.
    entry: [u8; 32],
    /// Digest of the state the segment leaves behind.
    exit: [u8; 32],
}

impl SegmentClaim {
    /// Fingerprint of the statement the segment was proved under.
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

/// What a chain of verified segments proves as a whole.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ChainedExecution {
    /// Digest of the state the first segment starts from.
    pub entry: [u8; 32],
    /// Digest of the state the last segment leaves behind.
    pub exit: [u8; 32],
    /// Number of segments in the chain.
    pub segments: usize,
}

/// Why a list of segment claims does not form one execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum ChainError {
    /// An empty list describes no execution.
    #[error("a chain needs at least one segment")]
    Empty,
    /// A segment does not start where the one before it stopped.
    #[error("segment {segment} does not start where segment {} stopped", segment - 1)]
    Broken {
        /// Position of the segment whose entry disagrees with the previous exit.
        segment: usize,
    },
}

/// Join claims of verified segments into one execution, in order.
///
/// Only the boundaries are compared; each claim must come from a verification that passed.
///
/// # Errors
///
/// Returns an error for an empty list, or at the first segment that does not follow its predecessor.
pub fn chain(claims: &[SegmentClaim]) -> Result<ChainedExecution, ChainError> {
    let (first, last) = claims.first().zip(claims.last()).ok_or(ChainError::Empty)?;

    if let Some(position) = claims
        .windows(2)
        .position(|pair| pair[0].exit != pair[1].entry)
    {
        return Err(ChainError::Broken {
            segment: position + 1,
        });
    }

    Ok(ChainedExecution {
        entry: first.entry,
        exit: last.exit,
        segments: claims.len(),
    })
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn claim(entry: u8, exit: u8) -> SegmentClaim {
        SegmentClaim {
            statement: [0; 32],
            entry: [entry; 32],
            exit: [exit; 32],
        }
    }

    #[test]
    fn a_single_segment_is_its_own_chain() {
        assert_eq!(
            chain(&[claim(1, 2)]).unwrap(),
            ChainedExecution {
                entry: [1; 32],
                exit: [2; 32],
                segments: 1,
            }
        );
    }

    #[test]
    fn segments_chain_when_each_starts_where_the_last_stopped() {
        let joined = chain(&[claim(1, 2), claim(2, 3), claim(3, 4)]).unwrap();
        assert_eq!(joined.entry, [1; 32]);
        assert_eq!(joined.exit, [4; 32]);
        assert_eq!(joined.segments, 3);
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
