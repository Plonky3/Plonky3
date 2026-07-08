//! Composite STARK soundness: AIR composition + DEEP-ALI + LDT, evaluated
//! once per proximity regime (UDR and best-`m` LDR). Generic over the LDT
//! via plain function arguments — `fri.rs`, `whir.rs`, and downstream
//! drop-in LDTs all compose with the same orchestrator.
//!
//! Extra protocol-specific error terms (lookup arguments, custom DEEP
//! variants, batched openings, …) are passed through `extras: &[ErrorBits]`
//! at every entry point and folded into the same round-by-round min.
//! Pass `&[]` when only the baseline AIR + DEEP + LDT terms apply.

use alloc::vec::Vec;

use crate::error::ErrorBits;
use crate::proximity::{list_size_ldr_m, list_size_udr};
use crate::shape::{InstanceShape, StarkAirParams};
use crate::{air, deep};

/// Bits attained in a single proximity regime, given the LDT-only error,
/// the regime's list size, and any extra protocol-specific error terms.
///
/// `ldt_error` is the round-by-round min over the LDT's commit and query
/// phases (see e.g. [`crate::fri::proven_error_udr`]). `list_size` is the
/// regime's L⁺. `extras` lets the caller fold in additional independent
/// error sources (lookup, custom DEEP, …) without dropping the orchestrator.
///
/// The result is capped at `shape.collision_resistance`: a collision in the
/// commitment hash forges the proof regardless of the algebraic bound, so
/// real security is `min(algebraic soundness, hash collision resistance)`.
pub fn proven_security_regime(
    air: &StarkAirParams,
    shape: &InstanceShape,
    list_size: f64,
    ldt_error: ErrorBits,
    extras: &[ErrorBits],
) -> ErrorBits {
    let ali = air::composition_error(air.num_constraints, list_size, shape.modulus_bits);
    let deep = deep::deep_ali_error(air, shape, list_size);
    let mut all: Vec<ErrorBits> = Vec::with_capacity(3 + extras.len());
    all.push(ali);
    all.push(deep);
    all.push(ldt_error);
    all.extend_from_slice(extras);
    let algebraic = ErrorBits::min(&all);
    ErrorBits::from_log2(algebraic.bits().min(shape.collision_resistance as f64))
}

/// Composite STARK bits in the UDR regime, with optional `extras`.
pub fn proven_security_udr(
    air: &StarkAirParams,
    shape: &InstanceShape,
    ldt_error: ErrorBits,
    extras: &[ErrorBits],
) -> ErrorBits {
    proven_security_regime(air, shape, list_size_udr(), ldt_error, extras)
}

/// Composite STARK bits in the LDR regime with explicit `m`, with
/// optional `extras`.
pub fn proven_security_ldr_m(
    air: &StarkAirParams,
    shape: &InstanceShape,
    log_blowup: usize,
    m: usize,
    ldt_error: ErrorBits,
    extras: &[ErrorBits],
) -> ErrorBits {
    proven_security_regime(
        air,
        shape,
        list_size_ldr_m(log_blowup, m),
        ldt_error,
        extras,
    )
}

/// Best of UDR and a precomputed best-`m` LDR, with optional `extras`
/// applied to both regimes. Each regime is an independent valid lower
/// bound, so the max is itself a valid (and tighter) bound on
/// round-by-round soundness.
pub fn proven_security(
    air: &StarkAirParams,
    shape: &InstanceShape,
    log_blowup: usize,
    udr_ldt_error: ErrorBits,
    ldr_best_m: usize,
    ldr_ldt_error: ErrorBits,
    extras: &[ErrorBits],
) -> ErrorBits {
    let udr = proven_security_udr(air, shape, udr_ldt_error, extras);
    let ldr = proven_security_ldr_m(air, shape, log_blowup, ldr_best_m, ldr_ldt_error, extras);
    ErrorBits::from_log2(udr.bits().max(ldr.bits()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> InstanceShape {
        InstanceShape {
            log_trace_length: 20,
            modulus_bits: 252,
            collision_resistance: 128,
        }
    }

    fn air() -> StarkAirParams {
        StarkAirParams {
            num_constraints: 1,
            max_constraint_degree: 2,
            max_combo: 2,
        }
    }

    /// Extras tighten (never loosen) the regime's bound, and a tight enough
    /// extra dominates ALI/DEEP/LDT.
    #[test]
    fn extras_tighten_proven_security_regime() {
        let air = air();
        let shape = shape();
        let ldt = ErrorBits::from_log2(80.0);

        let baseline = proven_security_regime(&air, &shape, 1.0, ldt, &[]);
        let with_loose =
            proven_security_regime(&air, &shape, 1.0, ldt, &[ErrorBits::from_log2(200.0)]);
        let with_tight =
            proven_security_regime(&air, &shape, 1.0, ldt, &[ErrorBits::from_log2(40.0)]);

        // A loose extra (200 bits) sits above every other term — bound unchanged.
        assert!((baseline.bits() - with_loose.bits()).abs() < 1e-12);
        // A tight extra (40 bits) becomes the binding term.
        assert!((with_tight.bits() - 40.0).abs() < 1e-12);
        // Monotone: extras can only tighten, never loosen.
        assert!(with_tight.bits() <= baseline.bits());
    }
}
