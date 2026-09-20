//! What the coverage check refuses.

/// A round the protocol runs that the budget did not bound.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Unaccounted {
    /// The configuration declared the round absent.
    ///
    /// Its bits are the ceiling and nothing was bounded.
    Waived(&'static str),

    /// The budget carries no round under that name at all.
    Unmodeled(&'static str),
}

impl Unaccounted {
    /// The round that went ungraded.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Waived(label) | Self::Unmodeled(label) => label,
        }
    }
}

impl core::fmt::Display for Unaccounted {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Waived(label) => write!(f, "round {label} runs but was waived as absent"),
            Self::Unmodeled(label) => write!(f, "round {label} runs but the budget has no term"),
        }
    }
}

impl core::error::Error for Unaccounted {}
