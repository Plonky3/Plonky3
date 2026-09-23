//! What a proving key proves: one flat system, or components instantiated many times.
//!
//! The two forms describe exactly the same statement.
//!
//! A composition names each gadget's relations once and repeats them across instances.
//!
//! Its stored description is therefore the size of the gadgets.
//!
//! Lowering expands it into the flat form, which is the size of the whole statement.
//!
//! Segment lengths, relation counts and the compiled wiring are all read through here.
//!
//! That is why both forms reach the same transcript and the same proof.

use p3_word::{Composition, ConstraintSystem, Segment, ShapeError, Word};

use crate::{CompiledKeyLayout, KeyCompileError};

/// Number of relation families a statement can declare.
pub(crate) const FAMILIES: usize = 3;

/// Anything that fixes the two segment lengths a witness must have.
///
/// Both statement forms and a bare gadget body implement it.
///
/// A packed witness can therefore be checked against whichever one the caller holds.
pub trait StatementShape<W: Word> {
    /// Returns the number of verifier-known words.
    fn public_len(&self) -> usize;

    /// Returns the number of committed words.
    fn witness_len(&self) -> usize;

    /// Checks a pair of segment lengths against this shape.
    ///
    /// # Errors
    ///
    /// Returns an error naming the segment whose length differs.
    fn check_shape(&self, public_len: usize, witness_len: usize) -> Result<(), ShapeError> {
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
}

impl<W: Word> StatementShape<W> for ConstraintSystem<W> {
    #[inline]
    fn public_len(&self) -> usize {
        Self::public_len(self)
    }

    #[inline]
    fn witness_len(&self) -> usize {
        Self::witness_len(self)
    }
}

impl<W: Word> StatementShape<W> for Composition<W> {
    #[inline]
    fn public_len(&self) -> usize {
        Self::public_len(self)
    }

    #[inline]
    fn witness_len(&self) -> usize {
        Self::witness_len(self)
    }
}

impl<W: Word> StatementShape<W> for Statement<W> {
    #[inline]
    fn public_len(&self) -> usize {
        Self::public_len(self)
    }

    #[inline]
    fn witness_len(&self) -> usize {
        Self::witness_len(self)
    }
}

/// The relations a key proves, in either of the two equivalent forms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Statement<W: Word> {
    /// Every relation written out once.
    Flat(ConstraintSystem<W>),
    /// Components declared once and instantiated a checked number of times.
    Composed(Composition<W>),
}

impl<W: Word> From<ConstraintSystem<W>> for Statement<W> {
    fn from(value: ConstraintSystem<W>) -> Self {
        Self::Flat(value)
    }
}

impl<W: Word> From<Composition<W>> for Statement<W> {
    fn from(value: Composition<W>) -> Self {
        Self::Composed(value)
    }
}

impl<W: Word> Statement<W> {
    /// Returns the number of verifier-known words.
    #[inline]
    pub const fn public_len(&self) -> usize {
        match self {
            Self::Flat(system) => system.public_len(),
            Self::Composed(composition) => composition.public_len(),
        }
    }

    /// Returns the number of committed words.
    #[inline]
    pub const fn witness_len(&self) -> usize {
        match self {
            Self::Flat(system) => system.witness_len(),
            Self::Composed(composition) => composition.witness_len(),
        }
    }

    /// Returns the relation counts in zero, AND, then product order.
    #[inline]
    pub fn relation_counts(&self) -> [usize; FAMILIES] {
        match self {
            Self::Flat(system) => [
                system.zero_constraints().len(),
                system.and_constraints().len(),
                system.integer_mul_constraints().len(),
            ],
            Self::Composed(composition) => composition.relation_counts(),
        }
    }

    /// Returns the flat relations, when the statement is written out.
    #[inline]
    pub const fn as_system(&self) -> Option<&ConstraintSystem<W>> {
        match self {
            Self::Flat(system) => Some(system),
            Self::Composed(_) => None,
        }
    }

    /// Returns the component calls, when the statement is composed.
    #[inline]
    pub const fn as_composition(&self) -> Option<&Composition<W>> {
        match self {
            Self::Flat(_) => None,
            Self::Composed(composition) => Some(composition),
        }
    }

    /// Compiles the sparse word-to-relation wiring the reductions read.
    ///
    /// # Errors
    ///
    /// Returns an error when the statement outgrows the compact key representation.
    #[inline]
    pub fn compiled_layout(&self) -> Result<CompiledKeyLayout<W>, KeyCompileError> {
        match self {
            Self::Flat(system) => CompiledKeyLayout::new(system),
            Self::Composed(composition) => CompiledKeyLayout::compose(composition),
        }
    }
}
