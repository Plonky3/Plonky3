//! Refuses a round budget that leaves one of the protocol's rounds ungraded.

pub mod audited;
pub mod error;
pub mod schedule;

pub use audited::AuditedReport;
pub use error::Unaccounted;
pub use schedule::ChallengeSchedule;
