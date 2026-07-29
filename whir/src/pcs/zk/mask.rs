//! Mask-code geometry for the HVZK-WHIR pipeline.
//!
//! Code shapes and committed mask groups, shared by the configuration, the
//! verifier replay, and the base case.

use p3_dft::Radix2Dit;
use p3_field::TwoAdicField;
use p3_zk_codes::ReedSolomonZkEncoding;
use rand::distr::{Distribution, StandardUniform};

/// Shape of one small mask code: a Reed-Solomon ZK encoding over `EF`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskCodeShape {
    /// Message length.
    pub message_len: usize,
    /// ZK randomness coefficients.
    pub randomness_len: usize,
    /// Codeword length (power of two).
    pub domain_size: usize,
}

impl MaskCodeShape {
    /// Derives the smallest power-of-two domain for the requested rate.
    #[must_use]
    pub const fn new(message_len: usize, randomness_len: usize, log_inv_rate: usize) -> Self {
        let domain_size = ((message_len + randomness_len).next_power_of_two()) << log_inv_rate;
        Self {
            message_len,
            randomness_len,
            domain_size,
        }
    }

    /// Instantiates the encoding over the extension field.
    ///
    /// Mask codewords are tiny, so a fresh radix-2 DIT per call is free.
    #[must_use]
    pub fn encoding<EF>(&self) -> ReedSolomonZkEncoding<EF, Radix2Dit<EF>>
    where
        EF: TwoAdicField,
        StandardUniform: Distribution<EF>,
    {
        ReedSolomonZkEncoding::new(
            self.randomness_len,
            self.message_len,
            self.domain_size,
            Radix2Dit::default(),
        )
    }
}

/// One committed batch of same-code mask oracles.
///
/// - Masks committed together share an evaluation domain.
/// - They stack into one matrix.
/// - A single commitment and one Merkle path per opened position cover the
///   whole group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskGroupShape {
    /// Code shared by every mask in the group.
    pub shape: MaskCodeShape,
    /// Number of stacked masks.
    pub width: usize,
}
