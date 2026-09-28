//! Verifier-derived dimensions and root encoding for product-tree GKR.

use alloc::vec::Vec;

use p3_field::Field;

use super::{ProductGkrError, ProductGkrShapeError};

/// How product roots are represented in a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductGkrRootShape {
    /// Every tree carries an independent root.
    Distinct,
    /// The first two trees use one shared root value.
    FirstTwoShared,
}

impl ProductGkrRootShape {
    /// Encodes roots according to the verifier-derived statement shape.
    pub(super) fn encode<F: Field>(self, roots: &[F]) -> Result<Vec<F>, ProductGkrError> {
        match self {
            Self::Distinct => Ok(roots.to_vec()),
            Self::FirstTwoShared => {
                // Structural sharing is valid only when both represented roots agree.
                if roots[0] != roots[1] {
                    return Err(ProductGkrError::SharedRootMismatch);
                }
                let mut encoded = Vec::with_capacity(roots.len() - 1);
                encoded.push(roots[0]);
                encoded.extend_from_slice(&roots[2..]);
                Ok(encoded)
            }
        }
    }

    /// Expands roots encoded according to the verifier-derived statement shape.
    pub(super) fn decode<F: Copy>(self, encoded: &[F]) -> Vec<F> {
        match self {
            Self::Distinct => encoded.to_vec(),
            Self::FirstTwoShared => {
                // The first message represents both structurally equal roots.
                let mut roots = Vec::with_capacity(encoded.len() + 1);
                roots.extend([encoded[0], encoded[0]]);
                roots.extend_from_slice(&encoded[1..]);
                roots
            }
        }
    }
}

/// Verifier-derived dimensions of one batched product reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductGkrShape {
    /// Logical base-two height of every identity-padded tree.
    pub(super) log_height: usize,
    /// Number of product trees reduced in lockstep.
    pub(super) num_trees: usize,
    /// Encoding of the root claims.
    pub(super) root_shape: ProductGkrRootShape,
}

impl ProductGkrShape {
    /// Constructs dimensions shared by the prover and verifier.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty batch, an impossible shared root, or an oversized tree.
    pub fn new(
        log_height: usize,
        num_trees: usize,
        root_shape: ProductGkrRootShape,
    ) -> Result<Self, ProductGkrShapeError> {
        // Product inputs use one machine word as their address space.
        if log_height >= usize::BITS as usize {
            return Err(ProductGkrShapeError::HeightOverflow {
                log_height,
                maximum: usize::BITS as usize - 1,
            });
        }
        if num_trees == 0 {
            return Err(ProductGkrShapeError::NoTrees {
                num_trees,
                minimum: 1,
            });
        }
        if num_trees > usize::MAX / 4 {
            return Err(ProductGkrShapeError::TreeCountOverflow {
                num_trees,
                maximum: usize::MAX / 4,
            });
        }
        if root_shape == ProductGkrRootShape::FirstTwoShared && num_trees < 2 {
            return Err(ProductGkrShapeError::SharedRootNeedsTwoTrees {
                num_trees,
                minimum: 2,
            });
        }

        Ok(Self {
            log_height,
            num_trees,
            root_shape,
        })
    }

    /// Logical base-two height of each product tree.
    #[must_use]
    pub const fn log_height(&self) -> usize {
        self.log_height
    }

    /// Number of product trees reduced together.
    #[must_use]
    pub const fn num_trees(&self) -> usize {
        self.num_trees
    }

    /// Root encoding fixed by the statement.
    #[must_use]
    pub const fn root_shape(&self) -> ProductGkrRootShape {
        self.root_shape
    }

    /// Number of root values carried by the proof.
    pub(crate) const fn root_message_len(&self) -> usize {
        match self.root_shape {
            ProductGkrRootShape::Distinct => self.num_trees,
            ProductGkrRootShape::FirstTwoShared => self.num_trees - 1,
        }
    }

    /// Root-to-leaf arity and sumcheck-round count of each layer.
    pub(crate) fn layers(&self) -> Vec<(usize, usize)> {
        // The point starts empty at the root.
        let mut point_len = 0;
        let mut remaining = self.log_height;
        let mut layers = Vec::with_capacity(self.log_height.div_ceil(2));

        while remaining > 0 {
            // An odd tree begins with one binary level.
            // Every later layer contracts two levels at once.
            let arity: usize = if remaining == self.log_height && remaining % 2 == 1 {
                2
            } else {
                4
            };
            layers.push((arity, point_len));
            let branch_count = arity.trailing_zeros() as usize;
            point_len += branch_count;
            remaining -= branch_count;
        }

        layers
    }

    /// Evaluates the constant-one suffix after an explicit prefix.
    ///
    /// Coordinates run from the most significant address bit to the least significant bit.
    /// The result is `sum_(i >= prefix_len) eq(point, i)` over the logical Boolean cube.
    ///
    /// # Panics
    ///
    /// Panics when the point length differs from the statement height.
    /// Panics when the prefix is longer than the logical table.
    #[must_use]
    pub fn identity_padding_evaluation<F: Field>(&self, prefix_len: usize, point: &[F]) -> F {
        // The statement fixes both the logical capacity and the point dimension.
        assert_eq!(
            point.len(),
            self.log_height,
            "point dimension must match the product GKR height"
        );
        let capacity = 1usize << self.log_height;
        assert!(prefix_len <= capacity, "prefix exceeds the logical table");
        if prefix_len == capacity {
            return F::ZERO;
        }

        // Sum the address weights strictly below the binary threshold.
        let mut below = F::ZERO;
        let mut equal_prefix = F::ONE;
        for (bit_index, &coordinate) in point.iter().enumerate() {
            let shift = point.len() - 1 - bit_index;
            if (prefix_len >> shift) & 1 == 1 {
                below += equal_prefix * (F::ONE - coordinate);
                equal_prefix *= coordinate;
            } else {
                equal_prefix *= F::ONE - coordinate;
            }
        }

        F::ONE - below
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_report_rejected_values_and_bounds() {
        // An empty batch reports both the supplied count and its minimum.
        assert_eq!(
            ProductGkrShape::new(0, 0, ProductGkrRootShape::Distinct),
            Err(ProductGkrShapeError::NoTrees {
                num_trees: 0,
                minimum: 1,
            })
        );

        // Structural root sharing requires two concrete trees.
        assert_eq!(
            ProductGkrShape::new(0, 1, ProductGkrRootShape::FirstTwoShared),
            Err(ProductGkrShapeError::SharedRootNeedsTwoTrees {
                num_trees: 1,
                minimum: 2,
            })
        );

        // Four children per tree determine the largest addressable batch.
        let num_trees = usize::MAX / 4 + 1;
        assert_eq!(
            ProductGkrShape::new(0, num_trees, ProductGkrRootShape::Distinct),
            Err(ProductGkrShapeError::TreeCountOverflow {
                num_trees,
                maximum: usize::MAX / 4,
            })
        );

        // A machine word cannot address a table with its own bit width as log height.
        let log_height = usize::BITS as usize;
        assert_eq!(
            ProductGkrShape::new(log_height, 1, ProductGkrRootShape::Distinct),
            Err(ProductGkrShapeError::HeightOverflow {
                log_height,
                maximum: log_height - 1,
            })
        );
    }
}
