//! Errors for read-only offline memory checking.

use alloc::string::String;

use thiserror::Error;

use crate::{BusNameError, ProductGkrError};

/// Invalid plans, witnesses, or reduced claims for read-only memory checking.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ReadOnlyMemoryError {
    /// The caller named the array outside the channel-name alphabet.
    #[error("read-only memory bus name is invalid: {0}")]
    InvalidBusName(#[from] BusNameError),
    /// The named array has no declarations in the enclosing bus plan.
    #[error("binary bus {name} does not exist")]
    UnknownBus {
        /// Caller-owned name used to isolate the array.
        name: String,
    },
    /// The tuple cannot hold an address and a count.
    #[error("read-only memory bus {name} has payload width {actual}, expected at least {minimum}")]
    PayloadTooNarrow {
        /// Caller-owned name used to isolate the array.
        name: String,
        /// Payload slots supplied by the bus plan.
        actual: usize,
        /// Slots required before any value components.
        minimum: usize,
    },
    /// The statement covers a different number of reads than the bus declares.
    #[error(
        "read-only memory bus {name} declares {actual} reads per direction, statement covers {expected}"
    )]
    DeclaredReadCountMismatch {
        /// Caller-owned name used to isolate the array.
        name: String,
        /// Read count supplied by the statement.
        expected: usize,
        /// Row count declared on one direction of that bus.
        actual: usize,
    },
    /// A read-only array must seed at least one address.
    #[error("read-only memory requires at least one array entry")]
    EmptyTable,
    /// Generator powers would repeat before every array address is seeded.
    #[error(
        "read-only memory has {table_len} entries, but the address-generator orbit has length {orbit_len}"
    )]
    AddressOrbitTooShort {
        /// Number of entries requiring distinct addresses.
        table_len: usize,
        /// Multiplicative order of the configured field generator.
        orbit_len: usize,
    },
    /// A forged count cycle could wrap around the generator orbit.
    #[error(
        "read-only memory has {read_len} reads, but the count-generator orbit has length {orbit_len}"
    )]
    CountOrbitTooShort {
        /// Total reads covered by the statement.
        read_len: usize,
        /// Multiplicative order of the configured field generator.
        orbit_len: usize,
    },
    /// A representable row count could span the whole count orbit of the declaring field.
    #[error(
        "read-only memory declarations need an unreachable count orbit, but the field's has length {orbit_len}"
    )]
    CountOrbitReachable {
        /// Multiplicative order of the configured field generator.
        orbit_len: usize,
    },
    /// Boundary and read factors cannot fit in one addressable product tree.
    #[error("read-only memory factor count overflows usize")]
    FactorCountOverflow,
    /// The table carries the wrong number of value columns.
    #[error("read-only memory table has {actual} value columns, expected {expected}")]
    TableWidthMismatch {
        /// Width fixed by the named bus payload.
        expected: usize,
        /// Width supplied by the witness.
        actual: usize,
    },
    /// The read witness carries the wrong number of value columns.
    #[error("read-only memory reads have {actual} value columns, expected {expected}")]
    ReadWidthMismatch {
        /// Width fixed by the named bus payload.
        expected: usize,
        /// Width supplied by the witness.
        actual: usize,
    },
    /// One table component has the wrong number of entries.
    #[error("read-only memory table column {column} has length {actual}, expected {expected}")]
    TableHeightMismatch {
        /// Position of the malformed value component.
        column: usize,
        /// Entry count fixed by the statement.
        expected: usize,
        /// Entry count supplied by the witness.
        actual: usize,
    },
    /// One read component has the wrong number of events.
    #[error("read-only memory read column {column} has length {actual}, expected {expected}")]
    ReadHeightMismatch {
        /// Position of the malformed value component.
        column: usize,
        /// Read count fixed by the statement.
        expected: usize,
        /// Read count supplied by the witness.
        actual: usize,
    },
    /// The read-address column has the wrong number of events.
    #[error("read-only memory address column has length {actual}, expected {expected}")]
    AddressHeightMismatch {
        /// Read count fixed by the statement.
        expected: usize,
        /// Read count supplied by the witness.
        actual: usize,
    },
    /// The read-count column has the wrong number of events.
    #[error("read-only memory count column has length {actual}, expected {expected}")]
    CountHeightMismatch {
        /// Read count fixed by the statement.
        expected: usize,
        /// Read count supplied by the witness.
        actual: usize,
    },
    /// The final-count column has the wrong number of entries.
    #[error("read-only memory final-count column has length {actual}, expected {expected}")]
    FinalCountHeightMismatch {
        /// Entry count fixed by the statement.
        expected: usize,
        /// Entry count supplied by the witness.
        actual: usize,
    },
    /// The materialized push and pull products differ.
    #[error("read-only memory push and pull products do not balance")]
    UnbalancedProducts,
    /// At least one read count is zero.
    #[error("read-only memory count product is zero")]
    ZeroCountProduct,
    /// A reduced output has the wrong number of roots.
    #[error("read-only memory reduction has {actual} roots, expected {expected}")]
    RootCountMismatch {
        /// Roots fixed by the three-tree statement.
        expected: usize,
        /// Roots supplied by the reduction.
        actual: usize,
    },
    /// A reduced output has the wrong number of leaf claims.
    #[error("read-only memory reduction has {actual} leaf claims, expected {expected}")]
    LeafClaimCountMismatch {
        /// Claims fixed by the three-tree statement.
        expected: usize,
        /// Claims supplied by the reduction.
        actual: usize,
    },
    /// A reduced output uses a point of the wrong dimension.
    #[error("read-only memory reduction point has dimension {actual}, expected {expected}")]
    ReductionPointDimensionMismatch {
        /// Dimension fixed by the product-tree height.
        expected: usize,
        /// Dimension supplied by the reduction.
        actual: usize,
    },
    /// The three-tree reduction itself failed.
    #[error("read-only memory product reduction failed: {0}")]
    ProductReduction(#[from] ProductGkrError),
}
