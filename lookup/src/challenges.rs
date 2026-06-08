//! Per-bus LogUp denominators derived from one sampled challenge pair.

use alloc::boxed::Box;
use core::iter::repeat_with;

use p3_field::Field;

/// Bus-separated LogUp challenge table built from a single `(alpha, beta)` pair.
///
/// Classical LogUp draws a fresh pair per bus.
/// This draws one pair and offsets each bus instead:
///
/// ```text
///     denominator(bus, payload) = prefix[bus] - sum_k beta^k * payload_k
///     prefix[bus]               = alpha + (bus + 1) * gamma
///     gamma                     = beta^W   (W = max message width)
/// ```
///
/// # Injectivity
///
/// - Payload terms occupy `beta^0 .. beta^(W-1)`.
/// - The bus offset sits at `beta^W`, one power above every payload term.
/// - So two messages collide only when both bus and payload agree.
///
/// # Soundness
///
/// - The pair is sampled after the main commitment, so the trace cannot adapt to it.
/// - An unbalanced bus stays unbalanced except on the fingerprint polynomial's roots.
/// - That root set is negligible over a cryptographic extension field.
pub struct Challenges<EF> {
    /// Base randomness shared by every bus.
    pub alpha: EF,
    /// Combiner weighting successive payload elements; callers raise it to the powers they need.
    pub beta: EF,
    /// Offset `alpha + (i + 1) * gamma` separating bus `i` from the rest.
    pub bus_prefix: Box<[EF]>,
}

impl<EF: Field> Challenges<EF> {
    /// Build the table from one sampled challenge pair.
    ///
    /// # Arguments
    ///
    /// - `alpha`: base randomness shared by every bus.
    /// - `beta`: combiner weighting successive payload elements.
    /// - `max_message_width`: widest payload in the batch; fixes the bus-offset power.
    /// - `num_bus_ids`: number of distinct buses to precompute offsets for.
    ///
    /// # Panics
    ///
    /// - When `max_message_width` is zero, since the bus offset would land on `beta^0` and collide with payloads.
    pub fn new(alpha: EF, beta: EF, max_message_width: usize, num_bus_ids: usize) -> Self {
        // Zero width leaves no power free for the bus offset.
        assert!(max_message_width > 0, "max_message_width must be non-zero");

        // gamma = beta^W is one power above every payload term, so the bus offset never collides.
        let gamma = beta.exp_u64(max_message_width as u64);

        // prefix[i] = alpha + (i + 1) * gamma, accumulated to skip a multiply per bus.
        let mut prefix = alpha;
        let bus_prefix: Box<[EF]> = repeat_with(|| {
            prefix += gamma;
            prefix
        })
        .take(num_bus_ids)
        .collect();

        Self {
            alpha,
            beta,
            bus_prefix,
        }
    }
}
