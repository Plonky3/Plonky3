use alloc::vec::Vec;

use thiserror::Error;

use crate::constraint::{AndConstraint, IntegerMulConstraint, Operand, ZeroConstraint};
use crate::index::Segment;
use crate::word::Word;

/// A homogeneous relation family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstraintKind {
    /// An XOR operand must vanish.
    Zero,
    /// Two operands are combined bit by bit.
    And,
    /// Two words produce a two-word unsigned product.
    IntegerMul,
}

/// An operand's role within a relation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperandRole {
    /// The only operand of a zero relation.
    Value,
    /// The left input.
    Left,
    /// The right input.
    Right,
    /// The output of a bitwise relation.
    Output,
    /// The low product limb.
    Low,
    /// The high product limb.
    High,
}

/// An invalid word-level constraint system.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum SystemError {
    /// A segment length exceeds the compact address space.
    #[error("{segment:?} segment length {len} exceeds u32::MAX")]
    SegmentTooLong {
        /// The oversized segment.
        segment: Segment,
        /// The rejected length.
        len: usize,
    },
    /// A term addresses a word outside its declared segment.
    #[error(
        "{kind:?} constraint {constraint} {role:?} term {term} addresses {segment:?}[{position}], but the segment length is {len}"
    )]
    IndexOutOfBounds {
        /// The relation family.
        kind: ConstraintKind,
        /// The position within the relation family.
        constraint: usize,
        /// The operand containing the term.
        role: OperandRole,
        /// The position within the XOR operand.
        term: usize,
        /// The selected segment.
        segment: Segment,
        /// The missing word position.
        position: u32,
        /// The declared segment length.
        len: usize,
    },
}

/// A failed scalar reference verification.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum VerificationError {
    /// The public input length differs from the declared shape.
    #[error("expected {expected} public words, received {actual}")]
    PublicLength {
        /// The declared length.
        expected: usize,
        /// The supplied length.
        actual: usize,
    },
    /// The committed witness length differs from the declared shape.
    #[error("expected {expected} witness words, received {actual}")]
    WitnessLength {
        /// The declared length.
        expected: usize,
        /// The supplied length.
        actual: usize,
    },
    /// A relation is not satisfied.
    #[error("{kind:?} constraint {constraint} is not satisfied")]
    Unsatisfied {
        /// The failed relation family.
        kind: ConstraintKind,
        /// The position within the relation family.
        constraint: usize,
    },
}

/// A checked collection of word-level relations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstraintSystem<W: Word> {
    /// The declared number of public words.
    public_len: u32,
    /// The declared number of committed words.
    witness_len: u32,
    /// The expressions required to vanish.
    zero_constraints: Vec<ZeroConstraint<W>>,
    /// The bitwise product relations.
    and_constraints: Vec<AndConstraint<W>>,
    /// The full-width unsigned product relations.
    integer_mul_constraints: Vec<IntegerMulConstraint<W>>,
}

impl<W: Word> ConstraintSystem<W> {
    /// Creates a system after validating every segment address.
    pub fn new(
        public_len: usize,
        witness_len: usize,
        zero_constraints: Vec<ZeroConstraint<W>>,
        and_constraints: Vec<AndConstraint<W>>,
        integer_mul_constraints: Vec<IntegerMulConstraint<W>>,
    ) -> Result<Self, SystemError> {
        // Bound lengths before compacting them into the system shape.
        let public_len = u32::try_from(public_len).map_err(|_| SystemError::SegmentTooLong {
            segment: Segment::Public,
            len: public_len,
        })?;
        let witness_len = u32::try_from(witness_len).map_err(|_| SystemError::SegmentTooLong {
            segment: Segment::Witness,
            len: witness_len,
        })?;

        let system = Self {
            public_len,
            witness_len,
            zero_constraints,
            and_constraints,
            integer_mul_constraints,
        };
        system.validate()?;
        Ok(system)
    }

    /// Returns the number of public words.
    #[inline]
    pub const fn public_len(&self) -> usize {
        self.public_len as usize
    }

    /// Returns whether the public segment is empty.
    #[inline]
    pub const fn public_is_empty(&self) -> bool {
        self.public_len == 0
    }

    /// Returns the number of committed witness words.
    #[inline]
    pub const fn witness_len(&self) -> usize {
        self.witness_len as usize
    }

    /// Returns whether the committed witness segment is empty.
    #[inline]
    pub const fn witness_is_empty(&self) -> bool {
        self.witness_len == 0
    }

    /// Returns the zero relations.
    #[inline]
    pub fn zero_constraints(&self) -> &[ZeroConstraint<W>] {
        &self.zero_constraints
    }

    /// Returns the bitwise AND relations.
    #[inline]
    pub fn and_constraints(&self) -> &[AndConstraint<W>] {
        &self.and_constraints
    }

    /// Returns the unsigned integer multiplication relations.
    #[inline]
    pub fn integer_mul_constraints(&self) -> &[IntegerMulConstraint<W>] {
        &self.integer_mul_constraints
    }

    /// Checks all relations with the scalar reference implementation.
    pub fn verify(&self, public: &[W], witness: &[W]) -> Result<(), VerificationError> {
        // Exact lengths prevent silently proving a prefix of either segment.
        if public.len() != self.public_len() {
            return Err(VerificationError::PublicLength {
                expected: self.public_len(),
                actual: public.len(),
            });
        }
        if witness.len() != self.witness_len() {
            return Err(VerificationError::WitnessLength {
                expected: self.witness_len(),
                actual: witness.len(),
            });
        }

        // Each family has an independent protocol reduction and index space.
        for (constraint, relation) in self.zero_constraints.iter().enumerate() {
            if !relation.is_satisfied(public, witness) {
                return Err(VerificationError::Unsatisfied {
                    kind: ConstraintKind::Zero,
                    constraint,
                });
            }
        }
        for (constraint, relation) in self.and_constraints.iter().enumerate() {
            if !relation.is_satisfied(public, witness) {
                return Err(VerificationError::Unsatisfied {
                    kind: ConstraintKind::And,
                    constraint,
                });
            }
        }
        for (constraint, relation) in self.integer_mul_constraints.iter().enumerate() {
            if !relation.is_satisfied(public, witness) {
                return Err(VerificationError::Unsatisfied {
                    kind: ConstraintKind::IntegerMul,
                    constraint,
                });
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), SystemError> {
        // Validate each role separately so diagnostics identify the exact source.
        for (constraint, relation) in self.zero_constraints.iter().enumerate() {
            self.validate_operand(
                ConstraintKind::Zero,
                constraint,
                OperandRole::Value,
                relation.value(),
            )?;
        }
        for (constraint, relation) in self.and_constraints.iter().enumerate() {
            self.validate_operand(
                ConstraintKind::And,
                constraint,
                OperandRole::Left,
                relation.left(),
            )?;
            self.validate_operand(
                ConstraintKind::And,
                constraint,
                OperandRole::Right,
                relation.right(),
            )?;
            self.validate_operand(
                ConstraintKind::And,
                constraint,
                OperandRole::Output,
                relation.output(),
            )?;
        }
        for (constraint, relation) in self.integer_mul_constraints.iter().enumerate() {
            self.validate_operand(
                ConstraintKind::IntegerMul,
                constraint,
                OperandRole::Left,
                relation.left(),
            )?;
            self.validate_operand(
                ConstraintKind::IntegerMul,
                constraint,
                OperandRole::Right,
                relation.right(),
            )?;
            self.validate_operand(
                ConstraintKind::IntegerMul,
                constraint,
                OperandRole::Low,
                relation.low(),
            )?;
            self.validate_operand(
                ConstraintKind::IntegerMul,
                constraint,
                OperandRole::High,
                relation.high(),
            )?;
        }
        Ok(())
    }

    fn validate_operand(
        &self,
        kind: ConstraintKind,
        constraint: usize,
        role: OperandRole,
        operand: &Operand<W>,
    ) -> Result<(), SystemError> {
        // Every term is relative to exactly one declared segment.
        for (term, index) in operand.indices() {
            let len = match index.segment() {
                Segment::Public => self.public_len(),
                Segment::Witness => self.witness_len(),
            };
            if index.position() as usize >= len {
                return Err(SystemError::IndexOutOfBounds {
                    kind,
                    constraint,
                    role,
                    term,
                    segment: index.segment(),
                    position: index.position(),
                    len,
                });
            }
        }
        Ok(())
    }
}
