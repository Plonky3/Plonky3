//! Collision resistance of a Merkle commitment, by how its nodes are built and opened.
//!
//! A proof that opens a committed position two ways forges the commitment.
//!
//! This module bounds that event for the node constructions the workspace offers.
//!
//! Every bound is in the ideal-function model and counts `q` queries to the compressions.
//!
//! A bound is reported as the security level `lambda = log2 q` at which the advantage reaches one.
//!
//! That is the convention of [`InstanceShape::collision_resistance`](crate::InstanceShape): `128` for a 256-bit plain tree.
//!
//! The paper bounds one T5 node, and a tree inherits the bound of its node (Section 7.2).
//!
//! # References
//!
//! - Dodis, Khovratovich, Mouha, Nandi. *T5: Hashing Five Inputs with Three Compression Calls*. [2021/373](https://eprint.iacr.org/2021/373)

use libm::{log2, pow};

use crate::ErrorBits;

/// How the nodes of a Merkle tree are hashed and opened.
///
/// This is the switch between the plain tree and the two T5 trees of 2021/373.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MerkleScheme {
    /// Every node is one call to an ideal compression function, opened by all its siblings.
    ///
    /// Binary and `N`-ary trees of a truncated permutation or of a hash both qualify.
    Plain,

    /// Every node is a T5 over three independent 2-to-1 compressions, opened by its four siblings.
    ///
    /// This is the opening `MerkleTreeMmcs` produces at `N = 5`.
    T5Conservative,

    /// T5 nodes, each opened by the three digests of `T5::open_aggressive`.
    ///
    /// The opening is shorter and faster to verify, and it is weaker.
    T5Aggressive,
}

/// Who builds the tree an adversary tries to open inconsistently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeBuilder {
    /// The root was computed honestly over a full message, and only openings are forged.
    ///
    /// The paper calls this full-local collision resistance.
    Honest,

    /// The adversary commits to a root of its choice and produces two conflicting openings.
    ///
    /// The paper calls this local-local collision resistance.
    ///
    /// A commitment inside a proof system is built by the prover, so this is the one that applies there.
    Adversarial,
}

/// What a bound rests on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence {
    /// A theorem in the ideal-function model.
    Proven,

    /// A conjecture the paper states, or the best known attack where the paper proves no more.
    ///
    /// It is never below the proven bound, which holds regardless.
    Conjectured,
}

impl MerkleScheme {
    /// Collision resistance of a tree of this scheme, in bits of security.
    ///
    /// # Arguments
    ///
    /// - `digest_bits`: `log2` of the digest space, `256` for 32 bytes and `8 log2 p` for eight elements of `F_p`.
    /// - `builder`: whether the root is honest or adversarial.
    /// - `evidence`: proven or conjectured.
    ///
    /// # Returns
    ///
    /// The number `lambda` of bits such that `2^lambda` queries reach advantage one.
    ///
    /// Per scheme, with `n` the digest bits:
    ///
    /// - Plain: `n / 2` either way, the birthday bound.
    /// - T5, conservative: proven `(n - log2(n^2 + 10)) / 2` (Theorem 1), conjectured `n / 2` (Section 8.1).
    /// - T5, aggressive, honest root: proven from `(n q^3 + 9 q^2) / 2^n` (Theorem 3), conjectured `(n - log2(4 n^3)) / 2` (Proposition 1).
    /// - T5, aggressive, adversarial root: proven `n / 4` (Theorem 3), conjectured `(n - 2) / 3` (Proposition 2).
    ///
    /// # Panics
    ///
    /// Panics if `digest_bits` is not a positive finite number.
    #[must_use]
    pub fn collision_bits(
        self,
        digest_bits: f64,
        builder: TreeBuilder,
        evidence: Evidence,
    ) -> ErrorBits {
        assert!(
            digest_bits.is_finite() && digest_bits > 0.0,
            "digest bits must be positive and finite, got {digest_bits}"
        );
        let n = digest_bits;

        let proven = match (self, builder) {
            (Self::Plain, _) => n / 2.0,
            (Self::T5Conservative, _) => (n - log2(n * n + 10.0)) / 2.0,
            (Self::T5Aggressive, TreeBuilder::Honest) => full_local_aggressive_proven(n),
            (Self::T5Aggressive, TreeBuilder::Adversarial) => n / 4.0,
        };
        let bits = match evidence {
            Evidence::Proven => proven,
            Evidence::Conjectured => {
                let conjectured = match (self, builder) {
                    (Self::Plain | Self::T5Conservative, _) => n / 2.0,
                    (Self::T5Aggressive, TreeBuilder::Honest) => (n - log2(4.0 * n * n * n)) / 2.0,
                    (Self::T5Aggressive, TreeBuilder::Adversarial) => (n - 2.0) / 3.0,
                };
                conjectured.max(proven)
            }
        };

        // A digest too short for any security reports no bound rather than a negative one.
        ErrorBits::from_log2(bits.max(0.0))
    }
}

/// Security level of Theorem 3's cross-collision bound, solving `n q^3 + 9 q^2 = 2^n` for `log2 q`.
///
/// The left side grows with `q`, so bisection on `lambda = log2 q` converges to the unique root.
fn full_local_aggressive_proven(n: f64) -> f64 {
    // `log2(n 2^(3 lambda) + 9 2^(2 lambda)) - n`, evaluated without overflowing the powers.
    let excess = |lambda: f64| {
        let (big, small) = (log2(n) + 3.0 * lambda, log2(9.0) + 2.0 * lambda);
        let (hi, lo) = if big >= small {
            (big, small)
        } else {
            (small, big)
        };
        hi + log2(1.0 + pow(2.0, lo - hi)) - n
    };

    // The root lies below `n / 3`, where the cubic term alone already reaches `2^n`.
    let (mut lo, mut hi) = (0.0, n / 3.0);
    if excess(lo) >= 0.0 {
        return 0.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if excess(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    const ALL_BUILDERS: [TreeBuilder; 2] = [TreeBuilder::Honest, TreeBuilder::Adversarial];

    fn bits(scheme: MerkleScheme, n: f64, builder: TreeBuilder, evidence: Evidence) -> f64 {
        scheme.collision_bits(n, builder, evidence).bits()
    }

    #[test]
    fn a_plain_tree_has_the_birthday_bound() {
        for builder in ALL_BUILDERS {
            for evidence in [Evidence::Proven, Evidence::Conjectured] {
                assert_eq!(bits(MerkleScheme::Plain, 256.0, builder, evidence), 128.0);
            }
        }
    }

    #[test]
    fn a_conservative_t5_tree_loses_the_theorem_1_factor_only_when_proven() {
        // (256 - log2(256^2 + 10)) / 2 = (256 - 16.0002) / 2, just under 120 bits.
        let proven = bits(
            MerkleScheme::T5Conservative,
            256.0,
            TreeBuilder::Adversarial,
            Evidence::Proven,
        );
        assert!((proven - 119.9999).abs() < 1e-3, "{proven}");

        // No attack beats the birthday bound, so the conjectured level is 128 bits.
        let conjectured = bits(
            MerkleScheme::T5Conservative,
            256.0,
            TreeBuilder::Adversarial,
            Evidence::Conjectured,
        );
        assert_eq!(conjectured, 128.0);
    }

    #[test]
    fn an_aggressive_t5_tree_matches_every_bound_of_the_paper() {
        let at = |builder, evidence| bits(MerkleScheme::T5Aggressive, 256.0, builder, evidence);

        // Honest root, proven: 256 q^3 dominates, so lambda = (256 - 8) / 3.
        let honest_proven = at(TreeBuilder::Honest, Evidence::Proven);
        assert!(
            (honest_proven - 248.0 / 3.0).abs() < 1e-6,
            "{honest_proven}"
        );

        // Honest root, conjectured: (256 - log2(4 * 2^24)) / 2 = (256 - 26) / 2.
        assert_eq!(at(TreeBuilder::Honest, Evidence::Conjectured), 115.0);

        // Adversarial root: n / 4 proven, (n - 2) / 3 conjectured.
        assert_eq!(at(TreeBuilder::Adversarial, Evidence::Proven), 64.0);
        assert!((at(TreeBuilder::Adversarial, Evidence::Conjectured) - 254.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn an_aggressive_tree_needs_a_wider_digest_for_128_bits_against_a_prover() {
        // Conjectured: (n - 2) / 3 >= 128 first holds at n = 386.
        let conjectured = |n| {
            bits(
                MerkleScheme::T5Aggressive,
                n,
                TreeBuilder::Adversarial,
                Evidence::Conjectured,
            )
        };
        assert!(conjectured(385.0) < 128.0);
        assert!(conjectured(386.0) >= 128.0);

        // Proven: n / 4 >= 128 first holds at n = 512.
        let proven = |n| {
            bits(
                MerkleScheme::T5Aggressive,
                n,
                TreeBuilder::Adversarial,
                Evidence::Proven,
            )
        };
        assert!(proven(511.0) < 128.0);
        assert_eq!(proven(512.0), 128.0);
    }

    #[test]
    fn a_digest_too_short_for_any_security_reports_zero_bits() {
        // Theorem 1's factor exceeds a 4-bit digest space outright.
        assert_eq!(
            bits(
                MerkleScheme::T5Conservative,
                4.0,
                TreeBuilder::Honest,
                Evidence::Proven
            ),
            0.0
        );
    }

    #[test]
    #[should_panic(expected = "digest bits must be positive and finite")]
    fn a_non_positive_digest_is_rejected() {
        let _ = MerkleScheme::Plain.collision_bits(0.0, TreeBuilder::Honest, Evidence::Proven);
    }

    proptest! {
        #[test]
        fn every_bound_orders_as_the_paper_ranks_the_schemes(n in 64.0f64..2048.0) {
            for builder in ALL_BUILDERS {
                for evidence in [Evidence::Proven, Evidence::Conjectured] {
                    let plain = bits(MerkleScheme::Plain, n, builder, evidence);
                    let conservative = bits(MerkleScheme::T5Conservative, n, builder, evidence);
                    let aggressive = bits(MerkleScheme::T5Aggressive, n, builder, evidence);

                    // Opening fewer digests never buys security.
                    prop_assert!(plain >= conservative);
                    prop_assert!(conservative >= aggressive);
                }

                // A conjecture only ever adds to what is proven.
                for scheme in [MerkleScheme::Plain, MerkleScheme::T5Conservative, MerkleScheme::T5Aggressive] {
                    prop_assert!(bits(scheme, n, builder, Evidence::Conjectured) >= bits(scheme, n, builder, Evidence::Proven));
                }
            }

            // An honest root is never easier to attack than an adversarial one.
            for evidence in [Evidence::Proven, Evidence::Conjectured] {
                prop_assert!(
                    bits(MerkleScheme::T5Aggressive, n, TreeBuilder::Honest, evidence)
                        >= bits(MerkleScheme::T5Aggressive, n, TreeBuilder::Adversarial, evidence)
                );
            }
        }

        #[test]
        fn the_theorem_3_root_solves_its_equation(n in 64.0f64..2048.0) {
            // At the reported level, n q^3 + 9 q^2 equals 2^n up to rounding.
            let lambda = full_local_aggressive_proven(n);
            let lhs = log2(n * pow(2.0, 3.0 * lambda - n) + 9.0 * pow(2.0, 2.0 * lambda - n));
            prop_assert!(lhs.abs() < 1e-9, "{lhs}");
        }
    }
}
