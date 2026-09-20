//! The rounds a compiled argument actually runs.

/// The rounds a compiled argument samples a challenge for, in any order and with repeats allowed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ChallengeSchedule<'a> {
    labels: &'a [&'static str],
}

impl<'a> ChallengeSchedule<'a> {
    /// Declares the rounds the transcript runs.
    pub const fn new(labels: &'a [&'static str]) -> Self {
        Self { labels }
    }

    /// The declared rounds.
    pub const fn labels(&self) -> &'a [&'static str] {
        self.labels
    }
}
