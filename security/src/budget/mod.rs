//! Conjectured round budget for a lifted (multi-AIR) STARK protocol, in fixed point.
//!
//! Fiat-Shamir security of the compiled argument is the minimum, over the protocol's rounds, of
//! that round's soundness error plus the grinding sited immediately before its challenge. This
//! module bounds each round and composes them, using [`crate::fixed`] throughout so the result is
//! reproducible bit-for-bit by a recursive verifier.
//!
//! "Conjectured" here is the standard proximity-gaps conjecture regime: list sizes are taken as
//! one up to capacity, so the algebraic rounds pay a plain Schwartz-Zippel error with no list-size
//! multiplier, and the FRI query phase pays the random-words rate of
//! [2025/2010](https://eprint.iacr.org/2025/2010) section 1.5 (see [`crate::fixed::bits_per_query`]).
//! Proven bounds for this parameter class are far lower and are not modeled here; anything
//! published from this module must be labeled conjectured.
//!
//! Every bound is stated as `error ≤ coefficient · size / |E|`, which in bits is
//! `log2|E| − log2(coefficient) − log2(size)`. Coefficients round up and the field size rounds
//! down, so each round is understated rather than overstated. The out-of-domain round is the one
//! exception to that framing: part of its size is additive rather than a multiple of the trace
//! height, so it takes the logarithm of the whole product at once. It states the same bound as
//! [`crate::deep::deep_ali_error`], the `f64` mirror of the same equation, and the two are meant
//! to be read together.
//!
//! # Folding accounting
//!
//! The folding round below sums a doubled per-round coefficient rather than searching for the
//! single worst round: `error ≤ (arity − 1) · (n + 1) / |E|` per round, worst at the first round
//! where the LDE domain `n` is largest, and doubling the coefficient absorbs the `+ 1`. This is
//! more conservative than reporting only the worst round's tight bound — by less than one bit,
//! since `2n ≥ n + 1` for `n ≥ 1` — in exchange for a coefficient that does not depend on how many
//! folding rounds actually occur.
//!
//! # DEEP-quotient batching
//!
//! [`shape::AirShape::num_deep_terms`] is `Option`: a protocol that batches its DEEP quotient
//! across every committed column via further α/β powers supplies it and gets an eighth accounted
//! round; a protocol that performs no such column-batching reduction passes `None` and the round
//! contributes no constraint on security (reported at the cap, the same convention
//! [`shape::AirShape::lookup`] uses for "no such argument"). The out-of-domain round above is
//! charged unconditionally either way — it is not what `None` waives.

pub mod report;
pub mod shape;

use report::{
    COLLISION_LABEL, COMPOSITION_LABEL, DEEP_COMPOSITION_LABEL, FOLDING_LABEL, LOOKUP_LABEL,
    OUT_OF_DOMAIN_LABEL, QUERY_LABEL,
};
pub use report::{SecurityReport, SecurityTerm};
pub use shape::{AirShape, InstanceShape, LookupShape, ProtocolParams};

use crate::fixed;

/// Grades a proof configuration, returning the per-round breakdown.
///
/// The attained level is [`SecurityReport::security_level`]; the round that binds is
/// [`SecurityReport::binding_term`].
pub const fn security_report(
    params: &ProtocolParams,
    instance: &InstanceShape,
    air: &AirShape,
) -> SecurityReport {
    let cap = instance.cap();

    // The lookup challenges are sampled once, right after the main-trace commitment, and shared
    // by every bus. Each bus denominator is affine in the first challenge with total degree at
    // most `max_message_width` in the two challenges jointly. If the bus multiset is unbalanced,
    // the balance function `Φ = Σ mₖ/Dₖ` is a nonzero rational function of the challenges, and the
    // cheat succeeds on either of two events.
    //
    // Each denominator `Dₖ = α + (bus + 1)·β^W + Σ_{j<W} β^j·payloadⱼ` has individual degree 1 in
    // `α` and `W` in `β`, so the cleared numerator `Σₖ mₖ·Π_{l≠k} D_l` over `M` denominators has
    // individual degrees `M − 1` and `W(M − 1)`. Sampling the two challenges independently and
    // uniformly, the per-variable Schwartz-Zippel union bounds a numerator root by
    // `(W + 1)(M − 1)/|E|`.
    //
    // Separately, a vanishing denominator degenerates the batched transition constraint: with
    // `Dⱼ = 0` the identity collapses to `mⱼ·Π_{s≠j} D_s = 0`, forcing `mⱼ = 0` and letting the
    // adversary drop message `j` from the bus for free. Each `Dₖ` is *monic* in `α`, so for any
    // `β` exactly one `α` annihilates it: that event costs only `M/|E|`, not `M·W/|E|`.
    //
    // Together `ε ≤ (W + 1)(M − 1)/|E| + M/|E| ≤ (W + 2)·M/|E|`.
    //
    // `M` is taken as `fractions_per_row · 2^log_max_height`, charging every AIR the maximum
    // height.
    let lookup = round(
        LOOKUP_LABEL,
        instance,
        (air.lookup.max_message_width as u64 + 2) * air.lookup.fractions_per_row as u64,
        instance.log_max_height,
        params.lookup_pow_bits,
        cap,
    );

    // Constraints are batched by powers of one challenge and the AIRs by a second, so a violated
    // constraint survives only on a root of the resulting univariate of degree below the total
    // number of batched slots. Independent of trace length.
    let composition = round(
        COMPOSITION_LABEL,
        instance,
        air.num_composed_constraints as u64,
        0,
        0,
        cap,
    );

    // The out-of-domain point is rejection-sampled outside the trace and LDE domains. A violated
    // constraint leaves the DEEP-ALI identity a nonzero polynomial in that point of degree at most
    // the larger of `max_constraint_degree · (height + max_combo − 1) + (height − 1)` (clearing the
    // shared denominator lifts each of a constraint's degree-many trace factors by one per
    // out-of-domain point referenced, and the quotient side contributes the rest) and
    // `(c + 1) · height + max_combo − 1` (the degree Plonky3's own power-of-two quotient chunking
    // induces, binding whenever `c` exceeds `max_constraint_degree`). Lifting evaluates the
    // shorter AIRs at powers of the same point, so all of them pay the maximum height.
    let out_of_domain = out_of_domain_round(
        instance,
        air.max_constraint_degree,
        air.max_combo,
        params.ood_pow_bits,
        cap,
    );

    // The DEEP quotient batches every committed column and out-of-domain point by powers of two
    // further challenges, giving a univariate whose degree is the number of batched terms.
    let deep_composition = round(
        DEEP_COMPOSITION_LABEL,
        instance,
        match air.num_deep_terms {
            Some(n) => n as u64,
            None => 0,
        },
        0,
        params.deep_pow_bits,
        cap,
    );

    // Per folding round the error is at most `(arity − 1) · (n + 1) / |E|`, worst at the first
    // round where the domain is largest. Doubling the coefficient absorbs the `+ 1`.
    let folding = round(
        FOLDING_LABEL,
        instance,
        2 * ((1u64 << params.log_folding_arity) - 1),
        instance.log_max_height + params.log_blowup,
        params.folding_pow_bits,
        cap,
    );

    // Query sampling is the only round whose error is not a Schwartz-Zippel bound: it is the
    // random-words rate compounded over independent queries.
    let query_bits = params.num_queries as u64
        * fixed::bits_per_query(params.log_blowup, instance.field_bits)
        + fixed::from_bits(params.query_pow_bits);
    let query = SecurityTerm::new(QUERY_LABEL, min(query_bits, cap));

    SecurityReport::new([
        lookup,
        composition,
        out_of_domain,
        deep_composition,
        folding,
        query,
        SecurityTerm::new(COLLISION_LABEL, cap),
    ])
}

/// Bounds one round whose error is `coefficient · 2^log_size / |E|`, crediting the grinding sited
/// before its challenge and capping at the transcript's own ceiling.
///
/// A zero coefficient means the protocol has no such round, which contributes no constraint on
/// security and is reported at the cap.
const fn round(
    label: &'static str,
    instance: &InstanceShape,
    coefficient: u64,
    log_size: u32,
    pow_bits: u32,
    cap: u64,
) -> SecurityTerm {
    if coefficient == 0 {
        return SecurityTerm::new(label, cap);
    }

    let error = fixed::ceil_log2(coefficient) + fixed::from_bits(log_size);
    let bits = if error >= instance.field_bits {
        fixed::from_bits(pow_bits)
    } else {
        instance.field_bits - error + fixed::from_bits(pow_bits)
    };

    SecurityTerm::new(label, min(bits, cap))
}

/// Bounds the out-of-domain round: error `max(d · (H + combo − 1) + (H − 1), (c + 1) · H + combo
/// − 1) / |E|`, with `H` the trace height, `d` the maximum constraint degree, `combo` the number
/// of out-of-domain points referenced per column, and `c = 2^⌈log2(max(d, 2) − 1)⌉` the number of
/// power-of-two quotient chunks Plonky3 commits (`uni_stark::symbolic::get_log_num_quotient_chunks`).
///
/// Checking the DEEP-ALI identity at the sampled point clears the common denominator
/// `Π_i (x − z_i)`, which lifts each of a constraint's `d` trace factors from degree `≤ H − 1` to
/// degree `≤ H + combo − 1`, and the quotient side adds a further `H − 1`:
///
/// ```text
/// d · (H + combo − 1)     from the d lifted factors
///           + (H − 1)     from the quotient side
/// ```
///
/// That is ethSTARK's `X^i · h_i(X^d)` split. Plonky3 instead commits the quotient as `c`
/// power-of-two chunks reconstructed with degree-`(c − 1) · H` coset selectors, giving identity
/// degree `(c + 1) · H + combo − 1`; this exceeds the first term whenever `c > d`, which is why
/// the two are combined by a maximum rather than the first alone. `c` assumes a non-`zk` quotient
/// split.
///
/// [`crate::deep::deep_ali_error`] states the same bound in `f64`. Unlike [`round`], both terms
/// are additive rather than a multiple of `H`, so the whole product is folded into the size
/// directly instead of decomposed into a coefficient and a log-size. This round is always charged
/// — there is no "no such round" case to special-case at zero.
///
/// `pow_bits` is the grinding sited immediately before the point is sampled, credited exactly as
/// [`round`] credits its own: a prover hunting for a favourable point pays `2^pow_bits` per
/// candidate. It is the only lever this round has, since both the degree and the height are fixed
/// by the statement being proved.
///
/// A degenerate degree or point count clamps to one, which keeps `size ≥ 1` and so keeps
/// [`fixed::ceil_log2`]'s zero-assert unreachable. A size too large to take the logarithm of is
/// reported at zero bits rather than wrapped, since truncating it would understate the error; the
/// grind is dropped with it rather than credited against an error this round could not bound.
const fn out_of_domain_round(
    instance: &InstanceShape,
    max_constraint_degree: u32,
    max_combo: u32,
    pow_bits: u32,
    cap: u64,
) -> SecurityTerm {
    if instance.log_max_height >= u64::BITS {
        return SecurityTerm::new(OUT_OF_DOMAIN_LABEL, 0);
    }

    let d = if max_constraint_degree == 0 {
        1
    } else {
        max_constraint_degree as u128
    };
    let combo = if max_combo == 0 { 1 } else { max_combo as u128 };
    let height = 1u128 << instance.log_max_height;
    let ethstark = d * (height + combo - 1) + (height - 1);

    // `c = 2^⌈log2(chunk_arg)⌉`, mirroring `p3_util::log2_ceil_usize` in `u128`: the smallest
    // power of two at least `chunk_arg`, found by counting the leading zeros of `chunk_arg − 1`.
    let chunk_arg = (if max_constraint_degree < 2 {
        2
    } else {
        max_constraint_degree as u128
    }) - 1;
    let log_chunks = u128::BITS - chunk_arg.saturating_sub(1).leading_zeros();
    let chunks = 1u128 << log_chunks;
    let chunked = (chunks + 1) * height + combo - 1;

    let size = if ethstark > chunked {
        ethstark
    } else {
        chunked
    };
    if size > u64::MAX as u128 {
        return SecurityTerm::new(OUT_OF_DOMAIN_LABEL, 0);
    }

    let error = fixed::ceil_log2(size as u64);
    let bits = instance.field_bits.saturating_sub(error) + fixed::from_bits(pow_bits);

    SecurityTerm::new(OUT_OF_DOMAIN_LABEL, min(bits, cap))
}

const fn min(a: u64, b: u64) -> u64 {
    if a < b { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::shape::LookupShape;

    const FIELD_BITS: u64 = 8_388_607;

    fn params() -> ProtocolParams {
        ProtocolParams {
            log_blowup: 3,
            log_folding_arity: 2,
            num_queries: 27,
            query_pow_bits: 17,
            ood_pow_bits: 0,
            deep_pow_bits: 12,
            folding_pow_bits: 4,
            lookup_pow_bits: 0,
        }
    }

    fn instance(log_max_height: u32) -> InstanceShape {
        InstanceShape {
            log_max_height,
            field_bits: FIELD_BITS,
            collision_resistance: 128,
        }
    }

    fn air() -> AirShape {
        AirShape {
            num_composed_constraints: 531,
            max_constraint_degree: 9,
            max_combo: 2,
            num_deep_terms: Some(130),
            lookup: LookupShape {
                fractions_per_row: 27,
                max_message_width: 16,
            },
        }
    }

    /// At moderate trace heights the query phase is what limits the configuration.
    #[test]
    fn query_phase_binds_at_moderate_heights() {
        let report = security_report(&params(), &instance(20), &air());
        assert_eq!(report.binding_term().label, QUERY_LABEL);
        assert_eq!(report.security_level(), 96);
    }

    /// The lookup round's error grows linearly in trace length while every other round is flat or
    /// grows only logarithmically, so past some height it becomes the bottleneck.
    #[test]
    fn lookup_round_binds_at_large_heights() {
        let report = security_report(&params(), &instance(29), &air());
        assert_eq!(report.binding_term().label, LOOKUP_LABEL);
        assert!(
            report.security_level() < 96,
            "lookup round should fall below the query-phase level at maximum trace height, got {}",
            report.security_level()
        );
    }

    /// Grinding before the lookup challenges is the direct remedy for the round above, and must
    /// be credited to it bit for bit.
    #[test]
    fn lookup_grinding_lifts_the_lookup_round() {
        let ungrounded = security_report(&params(), &instance(29), &air());
        let grinding = ProtocolParams {
            lookup_pow_bits: 8,
            ..params()
        };
        let ground = security_report(&grinding, &instance(29), &air());

        let before = ungrounded.terms()[0];
        let after = ground.terms()[0];
        assert_eq!(before.label, LOOKUP_LABEL);
        assert_eq!(after.bits, before.bits + fixed::from_bits(8));
    }

    /// Grinding before the out-of-domain point is credited to that round bit for bit, and to no
    /// other. The "no other" half is the point: the round shares its `max_constraint_degree` and
    /// `max_combo` inputs with nothing else, so a boost wired to the wrong term would surface as
    /// a neighbouring round moving instead.
    #[test]
    fn ood_grinding_lifts_only_the_out_of_domain_round() {
        let ungrounded = security_report(&params(), &instance(29), &air());
        let grinding = ProtocolParams {
            ood_pow_bits: 8,
            ..params()
        };
        let ground = security_report(&grinding, &instance(29), &air());

        for (before, after) in ungrounded.terms().iter().zip(ground.terms()) {
            assert_eq!(before.label, after.label, "term order changed");
            let expected = if before.label == OUT_OF_DOMAIN_LABEL {
                before.bits + fixed::from_bits(8)
            } else {
                before.bits
            };
            assert_eq!(after.bits, expected, "{} moved", before.label);
        }
    }

    /// No round may be reported above the transcript's own ceiling.
    #[test]
    fn every_round_is_capped_by_collision_resistance() {
        let instance = InstanceShape {
            collision_resistance: 96,
            ..instance(20)
        };
        let report = security_report(&params(), &instance, &air());
        for term in report.terms() {
            assert!(
                term.bits <= fixed::from_bits(96),
                "{} exceeds the cap",
                term.label
            );
        }
        assert_eq!(report.security_level(), 96);
    }

    /// Taller traces can never be more secure than shorter ones under the same parameters.
    #[test]
    fn security_is_monotone_in_trace_height() {
        let mut previous = u32::MAX;
        for log_height in 6..=30u32 {
            let level = security_report(&params(), &instance(log_height), &air()).security_level();
            assert!(level <= previous, "level rose at log height {log_height}");
            previous = level;
        }
    }

    /// Stronger than the level above, which is a minimum over rounds and therefore hides a round
    /// that gains bits with height behind whichever round binds. Each round on its own must be
    /// non-increasing, across the whole range the shape's `u32` height admits.
    #[test]
    fn every_round_is_monotone_in_trace_height() {
        let mut previous = [u64::MAX; report::NUM_TERMS];
        for log_height in 0..=64u32 {
            let report = security_report(&params(), &instance(log_height), &air());
            for (term, prev) in report.terms().iter().zip(previous.iter_mut()) {
                assert!(
                    term.bits <= *prev,
                    "{} rose at log height {log_height}",
                    term.label
                );
                *prev = term.bits;
            }
        }
    }

    /// Opening more out-of-domain points raises the degree the DEEP-ALI identity is tested at, so
    /// it can only cost the out-of-domain round bits.
    #[test]
    fn security_is_monotone_in_max_combo() {
        let mut previous = u64::MAX;
        for max_combo in 1..=16u32 {
            let air = AirShape { max_combo, ..air() };
            let term = security_report(&params(), &instance(6), &air).terms()[2];
            assert_eq!(term.label, OUT_OF_DOMAIN_LABEL);
            assert!(
                term.bits <= previous,
                "out-of-domain rose at max_combo {max_combo}"
            );
            previous = term.bits;
        }
    }

    /// A bigger or higher-degree AIR can never be more secure under the same parameters.
    #[test]
    fn security_is_monotone_in_air_size() {
        let small = security_report(&params(), &instance(24), &air());
        let large = AirShape {
            num_composed_constraints: 4096,
            max_constraint_degree: 9,
            max_combo: 2,
            num_deep_terms: Some(1024),
            lookup: LookupShape {
                fractions_per_row: 64,
                max_message_width: 16,
            },
        };
        let large = security_report(&params(), &instance(24), &large);
        assert!(large.security_level() <= small.security_level());
    }

    /// An AIR with no lookups pays no lookup round rather than being rejected or silently graded
    /// against a zero-message bus.
    #[test]
    fn absent_lookup_argument_contributes_no_round() {
        let air = AirShape {
            lookup: LookupShape {
                fractions_per_row: 0,
                max_message_width: 0,
            },
            ..air()
        };
        let report = security_report(&params(), &instance(29), &air);
        assert_eq!(report.terms()[0].bits, instance(29).cap());
        assert_ne!(report.binding_term().label, LOOKUP_LABEL);
    }

    /// A protocol that does not batch its DEEP quotient this way pays no such round, and the
    /// grind that would have sited it is simply unread.
    #[test]
    fn absent_deep_composition_contributes_no_round() {
        let air = AirShape {
            num_deep_terms: None,
            ..air()
        };
        let report = security_report(&params(), &instance(20), &air);
        let deep_term = report
            .terms()
            .iter()
            .find(|t| t.label == DEEP_COMPOSITION_LABEL)
            .expect("term present");
        assert_eq!(deep_term.bits, instance(20).cap());
        assert_ne!(report.binding_term().label, DEEP_COMPOSITION_LABEL);
    }
}
