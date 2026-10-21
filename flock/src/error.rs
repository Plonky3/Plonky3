//! Why a Flock proof cannot be made or is rejected.

use thiserror::Error;

use crate::lincheck::LincheckError;
use crate::opening::OpeningError;
use crate::piop::RegionError;
use crate::ports::PortsError;
use crate::zerocheck::ZerocheckError;

/// Why a proof cannot be made.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProvingError {
    /// The commitment or its opening failed.
    #[error(transparent)]
    Opening(#[from] OpeningError),
    /// The public words do not fit the batch or the witness.
    #[error(transparent)]
    Ports(#[from] PortsError),
    /// The placement does not hold the batch.
    #[error(transparent)]
    Region(#[from] RegionError),
}

/// Why a proof is rejected.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VerificationError {
    /// The commitment or its opening failed.
    #[error(transparent)]
    Opening(#[from] OpeningError),
    /// The zerocheck failed.
    #[error(transparent)]
    Zerocheck(#[from] ZerocheckError),
    /// The lincheck failed.
    #[error(transparent)]
    Lincheck(#[from] LincheckError),
    /// The public words or their check failed.
    #[error(transparent)]
    Ports(#[from] PortsError),
    /// The placement does not hold the batch.
    #[error(transparent)]
    Region(#[from] RegionError),
}
