//! A production ceiling on proof size and verifier work, checked before a proof is produced.

use crate::whir::error::BudgetError;
use crate::whir::shape::ProofShape;

/// A ceiling a deployment refuses to prove above.
///
/// A schedule is graded against it before any witness is touched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryWhirBudget {
    /// Most codeword positions a verifier will open.
    pub max_stir_queries: usize,

    /// Most bytes a serialized proof may occupy.
    pub max_proof_bytes: usize,

    /// Hardest grinding a prover will accept.
    pub max_grinding_bits: usize,
}

impl BinaryWhirBudget {
    /// The ceiling a deployment of this scheme is expected to hold to.
    ///
    /// Two hundred openings keep a verifier inside a few milliseconds of hashing.
    ///
    /// A quarter of a megabyte is the size at which a proof stops fitting comfortably on a wire.
    ///
    /// Twenty-four grinding bits cost a prover well under a second.
    pub const PRODUCTION: Self = Self {
        max_stir_queries: 200,
        max_proof_bytes: 256 * 1024,
        max_grinding_bits: 24,
    };

    /// Grade a schedule against the ceiling, before anything is committed.
    ///
    /// # Errors
    ///
    /// Returns an error naming the first figure the schedule exceeds.
    pub const fn check_shape(
        &self,
        shape: &ProofShape,
        base_element_bytes: usize,
        extension_element_bytes: usize,
        digest_bytes: usize,
    ) -> Result<(), BudgetError> {
        if shape.stir_queries > self.max_stir_queries {
            return Err(BudgetError::Queries {
                actual: shape.stir_queries,
                budget: self.max_stir_queries,
            });
        }
        if shape.grinding_bits > self.max_grinding_bits {
            return Err(BudgetError::Grinding {
                actual: shape.grinding_bits,
                budget: self.max_grinding_bits,
            });
        }
        let bytes = shape.max_bytes(base_element_bytes, extension_element_bytes, digest_bytes);
        if bytes > self.max_proof_bytes {
            return Err(BudgetError::Bytes {
                actual: bytes,
                budget: self.max_proof_bytes,
            });
        }
        Ok(())
    }

    /// Grade a proof that was produced, against the same ceiling.
    ///
    /// # Errors
    ///
    /// Returns an error when the serialized proof is larger than the ceiling allows.
    pub const fn check_bytes(&self, bytes: usize) -> Result<(), BudgetError> {
        if bytes > self.max_proof_bytes {
            return Err(BudgetError::Bytes {
                actual: bytes,
                budget: self.max_proof_bytes,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{BinaryWhirBudget, BudgetError, ProofShape};

    const SHAPE: ProofShape = ProofShape {
        stir_queries: 40,
        opened_base_elements: 100,
        opened_extension_elements: 200,
        merkle_digests: 300,
        sent_extension_elements: 50,
        grinding_bits: 16,
    };

    #[test]
    fn a_shape_inside_the_ceiling_is_accepted() {
        BinaryWhirBudget::PRODUCTION
            .check_shape(&SHAPE, 4, 16, 32)
            .unwrap();
    }

    #[test]
    fn each_axis_refuses_on_its_own() {
        let queries = BinaryWhirBudget {
            max_stir_queries: 39,
            ..BinaryWhirBudget::PRODUCTION
        };
        assert_eq!(
            queries.check_shape(&SHAPE, 4, 16, 32),
            Err(BudgetError::Queries {
                actual: 40,
                budget: 39
            })
        );

        let grinding = BinaryWhirBudget {
            max_grinding_bits: 15,
            ..BinaryWhirBudget::PRODUCTION
        };
        assert_eq!(
            grinding.check_shape(&SHAPE, 4, 16, 32),
            Err(BudgetError::Grinding {
                actual: 16,
                budget: 15
            })
        );

        let bytes = BinaryWhirBudget {
            max_proof_bytes: 13_999,
            ..BinaryWhirBudget::PRODUCTION
        };
        assert_eq!(
            bytes.check_shape(&SHAPE, 4, 16, 32),
            Err(BudgetError::Bytes {
                actual: 14_000,
                budget: 13_999
            })
        );
        assert_eq!(
            bytes.check_bytes(14_000),
            Err(BudgetError::Bytes {
                actual: 14_000,
                budget: 13_999
            })
        );
        assert_eq!(bytes.check_bytes(13_999), Ok(()));
    }
}
