//! Per-round polynomial assembly for the zero-knowledge sumcheck.

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};

/// Round-invariant context for the per-round polynomial assembly.
///
/// Built once at the top of the per-round loop and shared across every call.
#[derive(Debug, Clone, Copy)]
pub(super) struct RoundContext<'a, F, EF> {
    /// Folding factor.
    pub k: usize,
    /// Mask code message length; lower bound on the output length.
    pub ell_zk: usize,
    /// Powers-of-two table, length `k + 1`.
    pub pow2: &'a [F],
    /// Combining challenge that scales the plain piece.
    pub eps: EF,
}

impl<F, EF> RoundContext<'_, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Assemble the round polynomial.
    ///
    /// Implements the formula in step 4 of the masking layer.
    ///
    /// # Round formula
    ///
    /// ```text
    ///     h_j(X) = 2^{k-j}     * s_j(X)
    ///            + 2^{k-j}     * sum_{l <  j}  s_l(g_l)
    ///            + 2^{k-j-1}   * sum_{l >  j} ( s_l(0) + s_l(1) )    zero at j = k
    ///            + e           * plain_j(X)
    /// ```
    ///
    /// Encoded in the basis `(c_0, c_1, c_inf)`.
    /// The plain piece contributes a quadratic; the mask piece contributes one slot per coefficient.
    ///
    /// # Returns
    ///
    /// The full coefficient vector for the round, of length `max(ell_zk, 3)`.
    pub(super) fn assemble(&self, state: RoundState<'_, F, EF>, plain: PlainPiece<EF>) -> Vec<EF> {
        // Output length covers the larger of the mask degree and the plain quadratic.
        let h_size = self.ell_zk.max(3);
        let mut h: Vec<EF> = EF::zero_vec(h_size);

        // Live-mask contribution lands one slot per coefficient.
        let mult_live = self.pow2[self.k - state.j];
        for (i, &c) in state.mask.iter().enumerate() {
            h[i] += mult_live * c;
        }

        // Past-mask contribution: a single scalar on the constant slot.
        let past_mask_sum: EF = state.past_mask_evals.iter().copied().sum();
        h[0] += past_mask_sum * mult_live;

        // Future-mask contribution: zero in the last round, present otherwise.
        if state.j < self.k {
            let mult_future = self.pow2[self.k - state.j - 1];
            h[0] += mult_future * state.future_endpoints;
        }

        // Plain piece, scaled by the combining challenge.
        //
        // Only the constant and leading slots receive the plain term.
        // The linear slot is dropped from the wire and the verifier rederives that coefficient from the affine identity, so writing it here would be dead work.
        h[0] += self.eps * plain.c0;
        h[2] += self.eps * plain.c_inf;

        // Affine consistency cross-check.
        //
        // ```text
        //     h(0) + h(1) = 2 * h[0] + sum_{i >= 1} h[i]
        // ```
        //
        // Per-term breakdown:
        //
        //     live   : 2^{k-j} * ( s_j(0) + s_j(1) )
        //     past   : 2^{k-j+1} * past_mask_sum         (h[0] only)
        //     future : 2^{k-j} * sum_future              (h[0] only, zero at j=k)
        //     plain  : e * ( 2 c_0 + c_inf )
        //
        // The linear coefficient is excluded from the plain term on both sides:
        // it never lands in a transmitted slot, so it cancels out of the identity.
        //
        // Anchors:
        //
        //     j = 1 -> aux_target + e * mu      (round-1 target)
        //     j > 1 -> h_{j-1}(g_{j-1})         (by induction)
        #[cfg(debug_assertions)]
        {
            let mult_past = self.pow2[self.k - state.j + 1];
            let s_j_endpoints = state.mask[0].double() + state.mask[1..].iter().copied().sum::<F>();
            // Plain contribution that reaches the transmitted slots:
            //
            //     2 * (e * c_0)  from the doubled constant slot
            //         e * c_inf  from the leading slot
            //
            // The linear slot is omitted; the verifier reconstructs it from the affine identity.
            let plain_transmitted = plain.c0.double() + plain.c_inf;
            // The first term is the combining challenge times the transmitted plain sum.
            // Clippy's nursery lint flags it as if `eps * sum` should be `eps * eps`.
            // That rewrite would silently break the identity.
            #[allow(clippy::suspicious_operation_groupings)]
            let mut expected: EF = self.eps * plain_transmitted
                + past_mask_sum * mult_past
                + mult_live * s_j_endpoints;
            if state.j < self.k {
                expected += mult_live * state.future_endpoints;
            }
            debug_assert_eq!(
                h[0].double() + h[1..].iter().copied().sum::<EF>(),
                expected,
                "round polynomial affine consistency check failed at round {}",
                state.j,
            );
        }

        h
    }
}

/// Per-round dynamic state consumed by the polynomial assembly.
#[derive(Debug, Clone, Copy)]
pub(super) struct RoundState<'a, F, EF> {
    /// One-indexed round number.
    pub j: usize,
    /// Round-`j` mask coefficient vector.
    pub mask: &'a [F],
    /// Past mask evaluations at sampled challenges.
    pub past_mask_evals: &'a [EF],
    /// Running sum of future-mask endpoint pairs.
    ///
    /// The caller must have already decremented this by the current round's endpoint pair.
    pub future_endpoints: F,
}

/// Plain-piece contribution at the current round.
///
/// Only the constant and leading coefficients of the plain quadratic reach the wire.
/// The linear coefficient is dropped from the wire and reconstructed by the verifier from the affine identity, so it is never carried here.
#[derive(Debug, Clone, Copy)]
pub(super) struct PlainPiece<EF> {
    /// Constant coefficient.
    pub c0: EF,
    /// Leading coefficient.
    pub c_inf: EF,
}

/// Drop the linear coefficient from a round polynomial and return the wire form.
///
/// The verifier reconstructs the linear coefficient from the affine identity.
/// Sending the wire instead of the full polynomial saves one field element per round at zero soundness cost.
///
/// ```text
///     full : [ c_0, c_1, c_2, ..., c_d ]
///     wire : [ c_0,      c_2, ..., c_d ]
/// ```
pub(super) fn round_poly_to_wire<EF: Copy>(h: &[EF]) -> Vec<EF> {
    // First coefficient stays.
    // Linear coefficient is dropped.
    // All higher-degree coefficients follow.
    let mut wire: Vec<EF> = Vec::with_capacity(h.len() - 1);
    wire.push(h[0]);
    wire.extend_from_slice(&h[2..]);
    wire
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::zk::test_helpers::{EF, F};

    /// Pre-builds the powers-of-two table the assembly helper expects.
    fn pow2_table(k: usize) -> Vec<F> {
        F::TWO.powers().collect_n(k + 1)
    }

    /// Plain-piece value where every coefficient is zero.
    fn zero_plain() -> PlainPiece<EF> {
        PlainPiece {
            c0: EF::ZERO,
            c_inf: EF::ZERO,
        }
    }

    #[test]
    fn round_poly_to_wire_drops_the_linear_coefficient() {
        // h    = [c_0, c_1, c_2, c_3] = [10, 20, 30, 40]
        // wire = [c_0,      c_2, c_3] = [10, 30, 40]
        let h: Vec<EF> = vec![
            EF::from_u32(10),
            EF::from_u32(20),
            EF::from_u32(30),
            EF::from_u32(40),
        ];
        let wire = round_poly_to_wire(&h);
        assert_eq!(
            wire,
            vec![EF::from_u32(10), EF::from_u32(30), EF::from_u32(40)]
        );
    }

    #[test]
    fn round_poly_to_wire_handles_minimum_length() {
        // Length-3 input: only the linear coefficient drops.
        let h: Vec<EF> = vec![EF::from_u32(1), EF::from_u32(2), EF::from_u32(3)];
        let wire = round_poly_to_wire(&h);
        assert_eq!(wire, vec![EF::from_u32(1), EF::from_u32(3)]);
    }

    #[test]
    fn assemble_output_length_matches_max_of_mask_len_and_quadratic() {
        // Invariant:
        //
        //     h_size = max(ell_zk, 3)
        //
        //     ell_zk = 2  →  3   (plain quadratic dominates)
        //     ell_zk = 5  →  5   (mask piece dominates)
        //
        // With every contribution zeroed, the full polynomial is the
        // zero vector of length h_size; checking that pins both the
        // size and the absence of stray writes.
        let k = 1;
        let pow2 = pow2_table(k);

        let mask2 = vec![F::ZERO; 2];
        let h = RoundContext {
            k,
            ell_zk: 2,
            pow2: &pow2,
            eps: EF::ZERO,
        }
        .assemble(
            RoundState {
                j: 1,
                mask: &mask2,
                past_mask_evals: &[],
                future_endpoints: F::ZERO,
            },
            zero_plain(),
        );
        assert_eq!(h, vec![EF::ZERO; 3]);

        let mask5 = vec![F::ZERO; 5];
        let h = RoundContext {
            k,
            ell_zk: 5,
            pow2: &pow2,
            eps: EF::ZERO,
        }
        .assemble(
            RoundState {
                j: 1,
                mask: &mask5,
                past_mask_evals: &[],
                future_endpoints: F::ZERO,
            },
            zero_plain(),
        );
        assert_eq!(h, vec![EF::ZERO; 5]);
    }

    #[test]
    fn assemble_live_mask_lands_at_correct_slots() {
        // Fixture:
        //
        //     k = 1, j = 1,  mult_live = 2^0 = 1
        //     mask      = [7, 11, 13, 17]
        //     past      = [],  future_endpoints = 0
        //     plain     = 0
        //     ell_zk    = 4   ⇒  h_size = max(4, 3) = 4
        //
        // Live-mask contribution lands directly on h[i] = mult_live * mask[i].
        // Every other term is zero, so the full polynomial equals the lifted mask.
        let k = 1;
        let pow2 = pow2_table(k);
        let mask = vec![
            F::from_u32(7),
            F::from_u32(11),
            F::from_u32(13),
            F::from_u32(17),
        ];

        let h = RoundContext {
            k,
            ell_zk: 4,
            pow2: &pow2,
            eps: EF::ZERO,
        }
        .assemble(
            RoundState {
                j: 1,
                mask: &mask,
                past_mask_evals: &[],
                future_endpoints: F::ZERO,
            },
            zero_plain(),
        );

        // Hand-computed expected polynomial (mult_live = 1).
        let expected = vec![
            EF::from_u32(7),
            EF::from_u32(11),
            EF::from_u32(13),
            EF::from_u32(17),
        ];
        assert_eq!(h, expected);
    }

    #[test]
    fn assemble_future_term_present_only_when_j_lt_k() {
        // Fixture:
        //
        //     k = 2
        //     ell_zk = 2  ⇒  h_size = max(2, 3) = 3
        //     future_endpoints = 999     (loud sentinel)
        //     mask, past, plain all zero
        //
        // For j = 1 < k:
        //
        //     mult_future = 2^{k - j - 1} = 2^0 = 1
        //     h[0]        = mult_future * future_endpoints = 999
        //     h[1], h[2]  = 0
        //
        // For j = 2 = k: the future term is dropped, so h = [0, 0, 0].
        let k = 2;
        let pow2 = pow2_table(k);
        let mask = vec![F::ZERO, F::ZERO];
        let ctx = RoundContext {
            k,
            ell_zk: 2,
            pow2: &pow2,
            eps: EF::ZERO,
        };

        let h_first = ctx.assemble(
            RoundState {
                j: 1,
                mask: &mask,
                past_mask_evals: &[],
                future_endpoints: F::from_u32(999),
            },
            zero_plain(),
        );
        assert_eq!(h_first, vec![EF::from_u32(999), EF::ZERO, EF::ZERO]);

        let h_last = ctx.assemble(
            RoundState {
                j: 2,
                mask: &mask,
                past_mask_evals: &[],
                future_endpoints: F::from_u32(999),
            },
            zero_plain(),
        );
        assert_eq!(h_last, vec![EF::ZERO; 3]);
    }

    #[test]
    fn assemble_past_mask_sum_lands_on_constant_slot() {
        // Fixture:
        //
        //     k = 2, j = 2,  mult_live = 2^0 = 1
        //     ell_zk = 2     ⇒  h_size = max(2, 3) = 3
        //     past = [7, 11]      (past_sum = 18)
        //     mask, plain all zero, j = k (no future term)
        //
        // The past-mask term lands solely on h[0]:
        //
        //     h[0]       = past_sum * mult_live = 18 * 1 = 18
        //     h[1], h[2] = 0
        let k = 2;
        let pow2 = pow2_table(k);
        let mask = vec![F::ZERO, F::ZERO];
        let past = vec![EF::from_u32(7), EF::from_u32(11)];

        let h = RoundContext {
            k,
            ell_zk: 2,
            pow2: &pow2,
            eps: EF::ZERO,
        }
        .assemble(
            RoundState {
                j: 2,
                mask: &mask,
                past_mask_evals: &past,
                future_endpoints: F::ZERO,
            },
            zero_plain(),
        );

        assert_eq!(h, vec![EF::from_u32(18), EF::ZERO, EF::ZERO]);
    }

    #[test]
    fn assemble_satisfies_affine_consistency() {
        // Fixture (k = j = 1 to drop the past and future terms):
        //
        //     mask        = [2, 3, 5]
        //     plain.c0    = 7
        //     plain.c_inf = 13
        //     eps         = 31
        //     ell_zk      = 3  ⇒  h_size = 3
        //     mult_live   = 2^{k - j} = 1
        //
        // Per-slot expected breakdown (the linear slot gets no plain term):
        //
        //     h[0] = mult_live * mask[0] + eps * c_0    = 1 * 2 + 31 * 7  = 219
        //     h[1] = mult_live * mask[1]                = 1 * 3          = 3
        //     h[2] = mult_live * mask[2] + eps * c_inf  = 1 * 5 + 31 * 13 = 408
        //
        // Wire identity to cross-check:
        //
        //     h(0) + h(1) = 2 * h[0] + h[1] + h[2]
        //                 = 2 * 219 + 3 + 408 = 849
        //
        //     eps * (2 c_0 + c_inf) + mult_live * ( s_j(0) + s_j(1) )
        //                 = 31 * (2*7 + 13) + 1 * (2*2 + 3 + 5)
        //                 = 31 * 27 + 12 = 849
        let k = 1;
        let pow2 = pow2_table(k);
        let mask = vec![F::from_u32(2), F::from_u32(3), F::from_u32(5)];
        let plain = PlainPiece {
            c0: EF::from_u32(7),
            c_inf: EF::from_u32(13),
        };
        let eps = EF::from_u32(31);

        let h = RoundContext {
            k,
            ell_zk: 3,
            pow2: &pow2,
            eps,
        }
        .assemble(
            RoundState {
                j: 1,
                mask: &mask,
                past_mask_evals: &[],
                future_endpoints: F::ZERO,
            },
            plain,
        );

        // Per-slot pin against the hand-computed expected polynomial.
        let expected = vec![EF::from_u32(219), EF::from_u32(3), EF::from_u32(408)];
        assert_eq!(h, expected);

        // Wire identity cross-check.
        let actual_target = h[0].double() + h[1..].iter().copied().sum::<EF>();
        let live = EF::from_u32(2).double() + EF::from_u32(3) + EF::from_u32(5);
        let plain_transmitted = plain.c0.double() + plain.c_inf;
        let expected_target = eps * plain_transmitted + live;
        assert_eq!(actual_target, expected_target);
        assert_eq!(actual_target, EF::from_u32(849));
    }
}
