use thiserror::Error;

/// A planned expression references a value absent from its evaluation view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum BusEvaluationError {
    /// A main-trace expression reads a row other than the opened row.
    #[error("binary-bus main column {column} uses unsupported row offset {offset}")]
    MainOffset {
        /// Referenced column index.
        column: usize,
        /// Unsupported row offset.
        offset: usize,
    },
    /// A preprocessed expression reads a row other than the opened row.
    #[error("binary-bus preprocessed column {column} uses unsupported row offset {offset}")]
    PreprocessedOffset {
        /// Referenced column index.
        column: usize,
        /// Unsupported row offset.
        offset: usize,
    },
    /// A periodic expression has no commitment-bound opening in the evaluation view.
    #[error("binary-bus periodic column {column} has no supplied evaluation")]
    PeriodicColumn {
        /// Referenced periodic-column index.
        column: usize,
    },
    /// A main column lies outside the supplied committed opening.
    #[error("binary-bus main column {column} is not open")]
    MainColumn {
        /// Missing column index.
        column: usize,
    },
    /// A preprocessed column lies outside the supplied committed opening.
    #[error("binary-bus preprocessed column {column} is not open")]
    PreprocessedColumn {
        /// Missing column index.
        column: usize,
    },
    /// A public value lies outside the statement's public input.
    #[error("binary-bus public value {index} is absent")]
    PublicValue {
        /// Missing public-value index.
        index: usize,
    },
    /// The named-bus index lies outside the public plan.
    #[error("binary-bus domain index {bus} is outside {num_buses} planned domains")]
    UnknownBus {
        /// Rejected domain index.
        bus: usize,
        /// Number of domains in the plan.
        num_buses: usize,
    },
    /// The interaction width disagrees with its named bus.
    #[error("binary-bus domain {bus} has payload width {actual}, expected {expected}")]
    PayloadWidth {
        /// Planned domain index.
        bus: usize,
        /// Width fixed by the plan.
        expected: usize,
        /// Width supplied by the interaction.
        actual: usize,
    },
    /// The fingerprint weight table has the wrong width.
    #[error("binary-bus fingerprint has width {actual}, expected {expected}")]
    FingerprintWidth {
        /// Power-of-two width fixed by the plan.
        expected: usize,
        /// Number of supplied weights.
        actual: usize,
    },
}
