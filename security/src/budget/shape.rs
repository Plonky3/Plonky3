//! Inputs to the round budget: protocol parameters, instance shape, and AIR shape.
//!
//! These mirror the security-relevant subset of a protocol's runtime configuration rather than
//! wrapping it. The runtime types live in the protocol's own crate, which this one must not depend
//! on, so the caller — the only site with visibility into both sides — assembles these explicitly.

/// Protocol parameters that enter the round budget.
///
/// Every field must be bound into the protocol's Fiat-Shamir transcript, so a proof cannot be
/// graded under parameters it was not produced with.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ProtocolParams {
    /// Log2 of the LDE blowup factor; the FRI rate is `2^-log_blowup`.
    pub log_blowup: u32,
    /// Log2 of the FRI folding arity.
    pub log_folding_arity: u32,
    /// Number of FRI query repetitions.
    pub num_queries: u32,
    /// Grinding bits before query index sampling.
    pub query_pow_bits: u32,
    /// Grinding bits before the DEEP-composition challenge is sampled. Read only when
    /// [`AirShape::num_deep_terms`] is `Some`.
    pub deep_pow_bits: u32,
    /// Grinding bits before each FRI folding challenge.
    pub folding_pow_bits: u32,
    /// Grinding bits before the lookup / permutation argument's challenges are sampled.
    ///
    /// The lookup round's error grows linearly in trace length, so this is the only grinding site
    /// whose absence degrades with instance size.
    pub lookup_pow_bits: u32,
}

/// Shape of the proof instance being graded.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InstanceShape {
    /// Log2 of the largest AIR trace height in the proof.
    ///
    /// A lifted (multi-AIR) statement commits every matrix at the maximum height and tests a
    /// single FRI instance against it, with shorter AIRs opened at powers of the same
    /// out-of-domain point; every degree-dependent error therefore scales with this height, not
    /// with each AIR's own. A single-AIR statement uses its own trace height directly.
    pub log_max_height: u32,
    /// Log2 of the challenge field size, in fixed point.
    pub field_bits: u64,
    /// Collision resistance of the commitment hash, in whole bits.
    pub collision_resistance: u32,
}

impl InstanceShape {
    /// The ceiling any reported level is capped at: no argument compiled with this transcript can
    /// exceed the smaller of the challenge-field size and the commitment hash's collision
    /// resistance.
    pub const fn cap(&self) -> u64 {
        let collision = crate::fixed::from_bits(self.collision_resistance);
        if collision < self.field_bits {
            collision
        } else {
            self.field_bits
        }
    }
}

/// Shape of a LogUp-style lookup argument, aggregated over every AIR in the proof.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LookupShape {
    /// Total lookup fractions emitted per row, summed over all AIRs.
    ///
    /// Bounds the number of distinct bus messages an adversary can place in an unbalanced
    /// multiset at `fractions_per_row · 2^log_max_height`.
    pub fractions_per_row: u32,
    /// Maximum message width, i.e. the highest power of the second lookup challenge a denominator
    /// can reach.
    pub max_message_width: u32,
}

/// Shape of the AIRs being proved, aggregated over every AIR in the proof.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AirShape {
    /// Total constraints batched into the composition polynomial, plus one slot for each AIR
    /// beyond the first when a second challenge batches multiple AIRs together (`0` slots for a
    /// single-AIR statement, which needs no cross-AIR batching challenge).
    pub num_composed_constraints: u32,
    /// Maximum constraint degree over all AIRs.
    pub max_constraint_degree: u32,
    /// Maximum number of out-of-domain points referenced per committed column: `2` for an AIR
    /// that opens `local` and `next` rotations, `1` if it has no transition constraint.
    ///
    /// Read by the out-of-domain round, independent of [`Self::num_deep_terms`].
    pub max_combo: u32,
    /// Total committed columns opened by a DEEP-quotient batching argument, plus one slot per
    /// out-of-domain point, or `None` when the protocol performs no α/β column-batching reduction
    /// over its committed openings. The budget's out-of-domain round is charged regardless of this
    /// field; `None` only waives the separate DEEP-composition round.
    pub num_deep_terms: Option<u32>,
    /// Lookup argument shape. A zero `fractions_per_row` means the protocol has no lookup
    /// argument, and the round contributes no constraint on security.
    pub lookup: LookupShape,
}
