//! Errors returned while deriving a public bus layout.

use alloc::string::String;

use thiserror::Error;

use super::{BusExpressionLocation, UnsupportedBusAccess};
use crate::{BusDirection, ProductGkrShapeError};

/// Invalid statement shapes rejected before transcript construction.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BusPlanError {
    /// The total number of symbolic declarations overflowed.
    #[error("binary-bus declaration count overflows usize")]
    DeclarationCountOverflow,
    /// One AIR's trace height cannot be represented.
    #[error("binary-bus AIR {air} height overflows usize")]
    HeightOverflow {
        /// AIR position in statement order.
        air: usize,
    },
    /// A tuple has no payload expression.
    #[error("binary-bus AIR {air} declaration {declaration} has an empty tuple")]
    EmptyTuple {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
    },
    /// Two declarations on one named bus disagree on payload width.
    #[error("binary bus {name} has payload widths {expected} and {actual}")]
    PayloadWidthMismatch {
        /// Shared bus name.
        name: String,
        /// Width fixed by the first declaration.
        expected: usize,
        /// Width carried by the conflicting declaration.
        actual: usize,
    },
    /// The number of named domains overflowed its nonzero encoding.
    #[error("binary-bus domain count overflows usize")]
    DomainCountOverflow,
    /// Tuple slots or their power-of-two table overflowed.
    #[error("binary-bus fingerprint tuple width overflows usize")]
    TupleWidthOverflow,
    /// One direction's materialized leaf count overflowed.
    #[error("binary-bus {direction:?} leaf count overflows usize")]
    LeafCountOverflow {
        /// Side whose blocks overflowed.
        direction: BusDirection,
    },
    /// A symbolic expression reads data the terminal evaluator cannot reconstruct.
    #[error(
        "binary-bus AIR {air} declaration {declaration} {location:?} uses unsupported {access:?}"
    )]
    UnsupportedExpression {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
        /// Payload or activation expression containing the access.
        location: BusExpressionLocation,
        /// Unsupported access encountered in the expression tree.
        access: UnsupportedBusAccess,
    },
    /// The derived product-tree shape is invalid.
    #[error(transparent)]
    ProductShape(#[from] ProductGkrShapeError),
}
