use core::fmt;

use thiserror::Error;

use super::AirLayout;
use crate::{AirBuilder, PermutationAirBuilder, WindowAccess};

/// Builder input whose length can be checked against an [`AirLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AirLayoutInput {
    /// Current main-trace row.
    MainCurrent,
    /// Next main-trace row.
    MainNext,
    /// Current preprocessed-trace row.
    PreprocessedCurrent,
    /// Next preprocessed-trace row.
    PreprocessedNext,
    /// Public input values.
    PublicValues,
    /// Periodic column values at the current row.
    PeriodicValues,
    /// Current permutation-trace row.
    PermutationCurrent,
    /// Next permutation-trace row.
    PermutationNext,
    /// Permutation argument randomness.
    PermutationRandomness,
    /// Expected cumulated permutation values.
    PermutationValues,
}

impl fmt::Display for AirLayoutInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MainCurrent => "main().current_slice()",
            Self::MainNext => "main().next_slice()",
            Self::PreprocessedCurrent => "preprocessed().current_slice()",
            Self::PreprocessedNext => "preprocessed().next_slice()",
            Self::PublicValues => "public_values()",
            Self::PeriodicValues => "periodic_values()",
            Self::PermutationCurrent => "permutation().current_slice()",
            Self::PermutationNext => "permutation().next_slice()",
            Self::PermutationRandomness => "permutation_randomness()",
            Self::PermutationValues => "permutation_values()",
        })
    }
}

/// A builder input length that disagrees with the caller-supplied [`AirLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("AirLayout {input}: expected {expected} elements, got {actual}")]
pub struct AirLayoutMismatch {
    /// Accessor and, for a trace window, the row that disagrees.
    pub input: AirLayoutInput,
    /// Length specified by the layout.
    pub expected: usize,
    /// Length exposed by the builder.
    pub actual: usize,
}

const fn check_length(
    input: AirLayoutInput,
    expected: usize,
    actual: usize,
) -> Result<(), AirLayoutMismatch> {
    if expected == actual {
        Ok(())
    } else {
        Err(AirLayoutMismatch {
            input,
            expected,
            actual,
        })
    }
}

impl AirLayout {
    /// Check both main and preprocessed rows, public values, and periodic values against this explicit layout.
    ///
    /// Returns the first mismatch; zero widths and counts are valid.
    /// The preprocessed width comes from this layout, not an AIR declaration.
    /// Use [`Self::validate_against_permutation_builder`] to check permutation inputs too.
    ///
    /// This opt-in setup/debug check acquires each input accessor at most once and may construct or clone owned windows.
    /// It checks lengths without evaluating an AIR or materializing preprocessed or periodic data.
    ///
    /// # Panics
    ///
    /// Panics from builder or window accessors propagate, so malformed windows may panic before returning a mismatch.
    pub fn validate_against_builder(
        &self,
        builder: &impl AirBuilder,
    ) -> Result<(), AirLayoutMismatch> {
        use AirLayoutInput::*;

        let main = builder.main();
        check_length(MainCurrent, self.main_width, main.current_slice().len())?;
        check_length(MainNext, self.main_width, main.next_slice().len())?;
        let preprocessed = builder.preprocessed();
        check_length(
            PreprocessedCurrent,
            self.preprocessed_width,
            preprocessed.current_slice().len(),
        )?;
        check_length(
            PreprocessedNext,
            self.preprocessed_width,
            preprocessed.next_slice().len(),
        )?;
        check_length(
            PublicValues,
            self.num_public_values,
            builder.public_values().len(),
        )?;
        check_length(
            PeriodicValues,
            self.num_periodic_columns,
            builder.periodic_values().len(),
        )
    }

    /// Check base and permutation input lengths against this explicit layout.
    ///
    /// Runs [`Self::validate_against_builder`], then checks both permutation rows, randomness, and expected values.
    /// Returns the first mismatch; zero widths and counts are valid.
    /// Each input accessor is acquired at most once, with the same setup/debug costs as the base check.
    ///
    /// # Panics
    ///
    /// Panics from builder or window accessors propagate, so malformed windows may panic before returning a mismatch.
    pub fn validate_against_permutation_builder(
        &self,
        builder: &impl PermutationAirBuilder,
    ) -> Result<(), AirLayoutMismatch> {
        use AirLayoutInput::*;

        self.validate_against_builder(builder)?;
        let permutation = builder.permutation();
        check_length(
            PermutationCurrent,
            self.permutation_width,
            permutation.current_slice().len(),
        )?;
        check_length(
            PermutationNext,
            self.permutation_width,
            permutation.next_slice().len(),
        )?;
        check_length(
            PermutationRandomness,
            self.num_permutation_challenges,
            builder.permutation_randomness().len(),
        )?;
        check_length(
            PermutationValues,
            self.num_permutation_values,
            builder.permutation_values().len(),
        )
    }
}
