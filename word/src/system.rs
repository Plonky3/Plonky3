//! Checked statement shapes and canonical relation traversal.

use alloc::vec::Vec;

use thiserror::Error;

use crate::constraint::{AndConstraint, IntegerMulConstraint, Operand, ZeroConstraint};
use crate::index::Segment;
use crate::shift::ShiftedValue;
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
    /// A relation family exceeds the compact constraint address space.
    #[error("{kind:?} relation count {len} exceeds u32::MAX")]
    TooManyConstraints {
        /// The oversized relation family.
        kind: ConstraintKind,
        /// The rejected relation count.
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

/// A pair of word segments that does not match a checked statement shape.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[error("expected {expected} {segment:?} words, received {actual}")]
pub struct ShapeError {
    /// The segment with the wrong length.
    pub segment: Segment,
    /// The checked length.
    pub expected: usize,
    /// The supplied length.
    pub actual: usize,
}

/// One shifted word together with its position in a relation family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintTerm<'a, W: Word> {
    /// The relation family containing the term.
    kind: ConstraintKind,
    /// The term's semantic role within its relation.
    role: OperandRole,
    /// The position within the homogeneous relation family.
    constraint: u32,
    /// The shifted word consumed by the relation.
    term: &'a ShiftedValue<W>,
}

impl<'a, W: Word> ConstraintTerm<'a, W> {
    /// Returns the relation family containing the term.
    #[inline]
    pub const fn kind(self) -> ConstraintKind {
        self.kind
    }

    /// Returns the term's semantic role within its relation.
    #[inline]
    pub const fn role(self) -> OperandRole {
        self.role
    }

    /// Returns the position within the homogeneous relation family.
    #[inline]
    pub const fn constraint(self) -> u32 {
        self.constraint
    }

    /// Returns the shifted word consumed by the relation.
    #[inline]
    pub const fn term(self) -> &'a ShiftedValue<W> {
        self.term
    }
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

        // Every relation position is carried as a 32-bit protocol index.
        for (kind, len) in [
            (ConstraintKind::Zero, zero_constraints.len()),
            (ConstraintKind::And, and_constraints.len()),
            (ConstraintKind::IntegerMul, integer_mul_constraints.len()),
        ] {
            Self::validate_constraint_count(kind, len)?;
        }

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

    /// Iterates over shifted words in deterministic reduction order.
    ///
    /// - Families use zero, AND, then integer-multiplication order.
    /// - AND roles use left, right, then output order.
    /// - Multiplication roles use left, right, low, then high order.
    /// - Relation positions ascend within each role.
    /// - Terms retain their supplied order and multiplicity.
    pub fn terms(&self) -> impl Iterator<Item = ConstraintTerm<'_, W>> + Clone + '_ {
        // Zero relations contain one semantic operand.
        let zero = self
            .zero_constraints
            .iter()
            .zip(0_u32..)
            .flat_map(|(relation, constraint)| {
                relation
                    .value()
                    .terms()
                    .iter()
                    .map(move |term| ConstraintTerm {
                        kind: ConstraintKind::Zero,
                        role: OperandRole::Value,
                        constraint,
                        term,
                    })
            });

        // Role-major order groups every use made by one AND reduction column.
        let and = [OperandRole::Left, OperandRole::Right, OperandRole::Output]
            .into_iter()
            .flat_map(move |role| {
                self.and_constraints
                    .iter()
                    .zip(0_u32..)
                    .flat_map(move |(relation, constraint)| {
                        let operand = match role {
                            OperandRole::Left => relation.left(),
                            OperandRole::Right => relation.right(),
                            OperandRole::Output => relation.output(),
                            _ => unreachable!("AND relations expose only three operand roles"),
                        };
                        operand.terms().iter().map(move |term| ConstraintTerm {
                            kind: ConstraintKind::And,
                            role,
                            constraint,
                            term,
                        })
                    })
            });

        // Role-major order matches the four columns of the integer product reduction.
        let integer_mul = [
            OperandRole::Left,
            OperandRole::Right,
            OperandRole::Low,
            OperandRole::High,
        ]
        .into_iter()
        .flat_map(move |role| {
            self.integer_mul_constraints.iter().zip(0_u32..).flat_map(
                move |(relation, constraint)| {
                    let operand = match role {
                        OperandRole::Left => relation.left(),
                        OperandRole::Right => relation.right(),
                        OperandRole::Low => relation.low(),
                        OperandRole::High => relation.high(),
                        _ => unreachable!(
                            "integer multiplication relations expose only four operand roles"
                        ),
                    };
                    operand.terms().iter().map(move |term| ConstraintTerm {
                        kind: ConstraintKind::IntegerMul,
                        role,
                        constraint,
                        term,
                    })
                },
            )
        });

        // Chaining fixes one canonical order for every backend consumer.
        zero.chain(and).chain(integer_mul)
    }

    /// Checks the public and committed segment lengths against the statement shape.
    pub const fn check_shape(
        &self,
        public_len: usize,
        witness_len: usize,
    ) -> Result<(), ShapeError> {
        // Public values and committed values occupy independent index spaces.
        if public_len != self.public_len() {
            return Err(ShapeError {
                segment: Segment::Public,
                expected: self.public_len(),
                actual: public_len,
            });
        }
        if witness_len != self.witness_len() {
            return Err(ShapeError {
                segment: Segment::Witness,
                expected: self.witness_len(),
                actual: witness_len,
            });
        }
        Ok(())
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

    fn validate_constraint_count(kind: ConstraintKind, len: usize) -> Result<(), SystemError> {
        // Compact protocol references must represent every relation position.
        if u32::try_from(len).is_err() {
            return Err(SystemError::TooManyConstraints { kind, len });
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

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{ShiftedValue, ValueIndex, Word32, Word64};

    fn public(position: usize) -> ValueIndex {
        // Test positions remain inside the compact address space.
        ValueIndex::public(position).expect("test position must fit")
    }

    fn witness(position: usize) -> ValueIndex {
        // Test positions remain inside the compact address space.
        ValueIndex::witness(position).expect("test position must fit")
    }

    fn word32_operand(position: usize) -> Operand<Word32> {
        // One committed word forms the complete XOR operand.
        Operand::single(ShiftedValue::plain(witness(position)))
    }

    fn word64_operand(position: usize) -> Operand<Word64> {
        // One committed word forms the complete XOR operand.
        Operand::single(ShiftedValue::plain(witness(position)))
    }

    #[test]
    fn checked_system_accepts_all_relation_families() {
        // A public word cancels with itself in the zero relation.
        let public_term = ShiftedValue::plain(public(0));
        let zero = ZeroConstraint::new(Operand::new(vec![public_term, public_term]));

        // Witness layout: [left, right, and, product_low, product_high].
        let left = 0xfedc_ba98_7654_3210_u64;
        let right = 0x1234_5678_9abc_def0_u64;
        let product = u128::from(left) * u128::from(right);
        let words = [
            Word64::new(left),
            Word64::new(right),
            Word64::new(left & right),
            Word64::new(product as u64),
            Word64::new((product >> 64) as u64),
        ];
        let and = AndConstraint::new(word64_operand(0), word64_operand(1), word64_operand(2));
        let mul = IntegerMulConstraint::new(
            word64_operand(0),
            word64_operand(1),
            word64_operand(3),
            word64_operand(4),
        );
        let system = ConstraintSystem::new(1, 5, vec![zero], vec![and], vec![mul])
            .expect("every term is in range");

        assert_eq!(system.verify(&[Word64::new(7)], &words), Ok(()));
    }

    #[test]
    fn checked_system_rejects_out_of_bounds_terms() {
        // The only term selects the second word of a one-word witness.
        let zero = ZeroConstraint::new(word32_operand(1));
        let error = ConstraintSystem::new(0, 1, vec![zero], vec![], vec![])
            .expect_err("term exceeds the declared witness");

        assert_eq!(
            error,
            SystemError::IndexOutOfBounds {
                kind: ConstraintKind::Zero,
                constraint: 0,
                role: OperandRole::Value,
                term: 0,
                segment: Segment::Witness,
                position: 1,
                len: 1,
            }
        );
    }

    #[test]
    fn verifier_rejects_wrong_shape_and_corrupted_output() {
        // Witness layout: [left, right, and].
        let relation = AndConstraint::new(word32_operand(0), word32_operand(1), word32_operand(2));
        let system = ConstraintSystem::new(0, 3, vec![], vec![relation], vec![])
            .expect("every term is in range");

        // Missing a committed word fails before any relation is read.
        assert_eq!(
            system.verify(&[], &[Word32::new(1), Word32::new(1)]),
            Err(VerificationError::WitnessLength {
                expected: 3,
                actual: 2,
            })
        );

        // Mutation: claim zero for one AND one.
        let corrupt = [Word32::new(1), Word32::new(1), Word32::new(0)];
        assert_eq!(
            system.verify(&[], &corrupt),
            Err(VerificationError::Unsatisfied {
                kind: ConstraintKind::And,
                constraint: 0,
            })
        );
    }

    #[test]
    fn verifier_rejects_a_nonzero_zero_operand() {
        // A direct witness read must vanish to satisfy the relation.
        let zero = ZeroConstraint::new(word32_operand(0));
        let system = ConstraintSystem::new(0, 1, vec![zero], vec![], vec![])
            .expect("the witness term is in range");

        assert_eq!(
            system.verify(&[], &[Word32::new(1)]),
            Err(VerificationError::Unsatisfied {
                kind: ConstraintKind::Zero,
                constraint: 0,
            })
        );
    }

    #[test]
    fn validation_reports_public_and_nonlinear_operand_roles() {
        // Each relation family reports the exact malformed role and segment.
        let cases = [
            (
                vec![ZeroConstraint::new(Operand::single(ShiftedValue::plain(
                    public(0),
                )))],
                vec![],
                vec![],
                ConstraintKind::Zero,
                OperandRole::Value,
                Segment::Public,
                0,
            ),
            (
                vec![],
                vec![AndConstraint::new(
                    Operand::default(),
                    word32_operand(1),
                    Operand::default(),
                )],
                vec![],
                ConstraintKind::And,
                OperandRole::Right,
                Segment::Witness,
                1,
            ),
            (
                vec![],
                vec![],
                vec![IntegerMulConstraint::new(
                    word32_operand(0),
                    Operand::default(),
                    Operand::default(),
                    Operand::default(),
                )],
                ConstraintKind::IntegerMul,
                OperandRole::Left,
                Segment::Witness,
                0,
            ),
        ];

        for (zero, and, mul, kind, role, segment, position) in cases {
            let error = ConstraintSystem::<Word32>::new(0, 0, zero, and, mul)
                .expect_err("the selected segment is empty");
            assert_eq!(
                error,
                SystemError::IndexOutOfBounds {
                    kind,
                    constraint: 0,
                    role,
                    term: 0,
                    segment,
                    position,
                    len: 0,
                }
            );
        }
    }

    #[test]
    fn verifier_rejects_wrong_public_length() {
        // Exact public shape is checked before relation evaluation.
        let system = ConstraintSystem::<Word32>::new(1, 0, vec![], vec![], vec![])
            .expect("the empty relation set is valid");

        assert_eq!(
            system.verify(&[], &[]),
            Err(VerificationError::PublicLength {
                expected: 1,
                actual: 0,
            })
        );
    }

    #[test]
    fn shape_check_identifies_the_mismatched_segment() {
        // The statement declares two public words and three committed words.
        let system = ConstraintSystem::<Word32>::new(2, 3, vec![], vec![], vec![])
            .expect("the empty relation set is valid");

        // Public shape is checked before the committed segment.
        assert_eq!(
            system.check_shape(1, 3),
            Err(ShapeError {
                segment: Segment::Public,
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            system.check_shape(2, 4),
            Err(ShapeError {
                segment: Segment::Witness,
                expected: 3,
                actual: 4,
            })
        );
        assert_eq!(system.check_shape(2, 3), Ok(()));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn segment_lengths_above_u32_are_rejected() {
        // Length validation precedes allocation and relation traversal.
        let len = u32::MAX as usize + 1;
        let error = ConstraintSystem::<Word32>::new(len, 0, vec![], vec![], vec![])
            .expect_err("the public segment exceeds the compact address space");

        assert_eq!(
            error,
            SystemError::SegmentTooLong {
                segment: Segment::Public,
                len,
            }
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn relation_counts_above_u32_are_rejected() {
        // Relation positions use the same compact address width as word positions.
        let len = u32::MAX as usize + 1;

        assert_eq!(
            ConstraintSystem::<Word32>::validate_constraint_count(ConstraintKind::And, len),
            Err(SystemError::TooManyConstraints {
                kind: ConstraintKind::And,
                len,
            })
        );
    }

    #[test]
    fn term_iteration_preserves_reduction_order_and_provenance() {
        // Three distinct word positions make every role visible in the result.
        let operand = |position| Operand::single(ShiftedValue::<Word32>::plain(witness(position)));
        let system = ConstraintSystem::new(
            0,
            3,
            vec![ZeroConstraint::new(operand(0))],
            vec![AndConstraint::new(operand(0), operand(1), operand(2))],
            vec![],
        )
        .expect("every term is inside the committed segment");

        // The canonical walk is family-major and then role-major.
        let terms = system.terms().collect::<Vec<_>>();
        let provenance = terms
            .iter()
            .copied()
            .map(|term| {
                (
                    term.kind(),
                    term.role(),
                    term.constraint(),
                    term.term().index().position(),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(
            provenance,
            [
                (ConstraintKind::Zero, OperandRole::Value, 0, 0),
                (ConstraintKind::And, OperandRole::Left, 0, 0),
                (ConstraintKind::And, OperandRole::Right, 0, 1),
                (ConstraintKind::And, OperandRole::Output, 0, 2),
            ]
        );
    }
}
