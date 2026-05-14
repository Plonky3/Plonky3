//! Field-independent FFT-based MDS multiplication infrastructure.
//!
//! - `butterfly`: FFT-4 real-signal transforms (add/sub only)
//! - `freq_mul`: 12×12 frequency-domain circulant multiply

pub mod butterfly;
pub mod freq_mul;

pub use freq_mul::mds12_multiply_freq;
