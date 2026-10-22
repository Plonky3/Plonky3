//! A batch: one circuit repeated on a power of two of instances, the statement every phase proves.

use thiserror::Error;

use crate::circuit::Circuit;
use crate::zerocheck::MIN_LOG_N;

/// One circuit on `2^log_instances` instances, with at least `MIN_LOG_N` witness variables.
#[derive(Clone, Copy, Debug)]
pub struct Batch<'a> {
    /// The circuit every instance runs.
    circuit: &'a Circuit,
    /// Instances, as a power of two.
    log_instances: usize,
}

impl<'a> Batch<'a> {
    /// The batch of `2^log_instances` instances of a circuit.
    ///
    /// # Errors
    ///
    /// The witness has fewer variables than the zerocheck needs, or more bits than a `usize` indexes.
    pub const fn new(circuit: &'a Circuit, log_instances: usize) -> Result<Self, BatchError> {
        let k_log = circuit.k_log();
        // Every witness bit has a `usize` index, so `2^log_n` must be representable.
        let log_n = match k_log.checked_add(log_instances) {
            Some(log_n) if log_n < usize::BITS as usize => log_n,
            _ => {
                return Err(BatchError::TooManyVariables {
                    k_log,
                    log_instances,
                });
            }
        };
        if log_n < MIN_LOG_N {
            return Err(BatchError::TooFewVariables { log_n });
        }
        Ok(Self {
            circuit,
            log_instances,
        })
    }

    /// The circuit every instance runs.
    #[must_use]
    pub const fn circuit(&self) -> &'a Circuit {
        self.circuit
    }

    /// Instances, as a power of two.
    #[must_use]
    pub const fn log_instances(&self) -> usize {
        self.log_instances
    }

    /// Witness variables: instance bits, then the bits of one instance.
    #[must_use]
    pub const fn log_n(&self) -> usize {
        self.circuit.k_log() + self.log_instances
    }
}

/// Why a batch cannot be proved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum BatchError {
    /// The witness has fewer variables than the skip and the fixed coordinates take.
    #[error("{log_n} witness variables are below the minimum of {MIN_LOG_N}")]
    TooFewVariables {
        /// The batch's witness variables.
        log_n: usize,
    },
    /// The witness has more bits than a `usize` indexes.
    #[error("2^{k_log} bits on 2^{log_instances} instances exceed the address space")]
    TooManyVariables {
        /// Bits per instance, as a power of two.
        k_log: usize,
        /// Instances, as a power of two.
        log_instances: usize,
    },
}
