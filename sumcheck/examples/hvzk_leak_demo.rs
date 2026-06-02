//! Demonstration of the HVZK sumcheck leak (eprint 2026/391 Construction 6.3).
//!
//! Run with:
//!
//! ```sh
//! cargo run -p p3-sumcheck --example hvzk_leak_demo
//! ```
//!
//! # What this shows
//!
//! Construction 6.3 is zero-knowledge *only when the whole protocol lives over a
//! single field* (the paper samples masks `s_j ∈ F[X]` **and** the batching
//! challenge `ε ← F`, both over the base field — Lemma 6.4's proof relies on the
//! mask coefficients and the round-polynomial coefficients living in the same
//! field, so uniform masks fill the affine transcript subspace).
//!
//! The Plonky3 implementation keeps masks over the base field `F` (for sublinear
//! proof size) but lifts `ε` and the round challenges into the extension field
//! `EF` (for soundness over a 2³¹ base field) — see
//! `sumcheck/src/zk/prover.rs:281-284`. That hybrid is **not** the paper's
//! construction and is **not** covered by Lemma 6.4.
//!
//! This example reproduces the *exact* honest round-polynomial coefficient that
//! the prover sends and shows that it leaks the witness.
//!
//! The sent coordinate modelled here is `wire[1] = h[2]`, the `X²` coefficient,
//! assembled in `prover.rs` as (see `prover.rs:354-373`, wire layout `:405-409`):
//!
//! ```text
//!     h[2] = mult_live * s_j[2]      // live mask, mult_live = 2^{k-j} ∈ F  -> F-subspace
//!          + eps * plain_c_inf       // plain piece, eps ∈ EF, plain_c_inf ∈ EF (witness-dependent)
//! ```
//!
//! The simulator that is supposed to certify zero-knowledge draws this same
//! coordinate uniformly over the full extension field
//! (`simulator.rs:156-164`, the `i < 2` branch: `rng.random::<EF>()`).
//!
//! Because the live mask `mult_live * s_j[2]` lives entirely in the base-field
//! slot (extension coordinate 0), the *higher* extension coordinates of the
//! honest `h[2]` are exactly the higher coordinates of `eps * plain_c_inf` — a
//! deterministic, witness-dependent value the mask never touches. So:
//!
//! * the honest coordinate ranges over a single base-field coset (support `|F|`),
//! * the simulator's coordinate ranges over all of `EF` (support `|F|⁴`),
//!
//! and a verifier reads `eps * plain_c_inf` mod the base-field subspace in the
//! clear — i.e. it learns the witness-dependent `plain_c_inf` (up to one
//! coordinate). This example builds the explicit distinguisher and then shows
//! that the paper-faithful fix (an extension-field mask coefficient) closes it.

use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Base field, as used by the HVZK masks.
type F = BabyBear;
/// Degree-4 extension, as used by the sumcheck challenges (`EF` in `zk/`).
type EF = BinomialExtensionField<F, 4>;

/// Extension coordinates strictly above the base-field slot.
///
/// The honest live mask `2^{k-j} * s_j[i]` only ever populates coordinate 0,
/// so these are exactly the coordinates where an unblinded `EF` offset (the
/// witness-dependent `eps * plain` term) becomes visible to the verifier.
fn high_coords(x: EF) -> Vec<F> {
    EF::as_basis_coefficients_slice(&x)[1..].to_vec()
}

fn main() {
    // Round j = 1 of a k-round HVZK sumcheck (the leak is present in every
    // round; round 1 is the simplest to state). `mult_live = 2^{k-j}` mirrors
    // `let mult_live = pow2[k - j];` in `prover.rs`.
    let k = 4usize;
    let j = 1usize;
    let mult_live: F = F::TWO.exp_u64((k - j) as u64);

    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);

    // Verifier challenge. In the honest-verifier ZK game this is sampled
    // independently of the prover's mask coins, so we fix it across every
    // sample below. `eps ∈ EF` exactly as in `prover.rs:284`.
    let eps: EF = rng.random();

    // Witness-dependent plain leading coefficient (the X² coefficient of the
    // batched plain round polynomial). `EF`-valued and a deterministic function
    // of the committed witness — `prover.rs:324` computes it as a dot product.
    let plain_c_inf: EF = rng.random();

    // The exact honest coordinate the prover sends as `wire[1]` (`prover.rs:357,373`):
    //     h[2] = mult_live * s_j[2]  +  eps * plain_c_inf
    // `s_j[2] ∈ F`, so `mult_live * s_j[2]` lifts into the base-field subspace.
    let honest_wire1 = |s_j_2: F| EF::from(mult_live * s_j_2) + eps * plain_c_inf;

    // The value the verifier reads off in the clear: the higher extension
    // coordinates of the honest coordinate equal those of `eps * plain_c_inf`,
    // regardless of the mask.
    let leak_target = high_coords(eps * plain_c_inf);

    println!("=== HVZK sumcheck leak demo (Construction 6.3, eprint 2026/391) ===\n");
    println!("field: BabyBear, extension degree 4, modelling round j=1 of k={k}\n");

    // ---------------------------------------------------------------------
    // 1. The live mask only moves the base-field coordinate.
    // ---------------------------------------------------------------------
    let w_mask_0 = honest_wire1(F::ZERO);
    let w_mask_42 = honest_wire1(F::from_u64(42));
    println!("[1] Two honest runs that differ only in the live mask s_j[2]:");
    println!("      mask = 0  -> high coords {:?}", high_coords(w_mask_0));
    println!("      mask = 42 -> high coords {:?}", high_coords(w_mask_42));
    assert_ne!(
        w_mask_0, w_mask_42,
        "the mask must change the sent coordinate somewhere"
    );
    assert_eq!(
        high_coords(w_mask_0),
        high_coords(w_mask_42),
        "but the mask must NOT change the higher coordinates"
    );
    assert_eq!(high_coords(w_mask_0), leak_target);
    println!("      => the mask only moved coordinate 0; the higher coords are fixed.\n");

    // The fixed higher coordinates are witness-dependent (non-zero), i.e. they
    // carry information about `plain_c_inf`.
    assert!(
        leak_target.iter().any(|c| *c != F::ZERO),
        "the leaked higher coordinates are witness-dependent and non-zero"
    );

    // ---------------------------------------------------------------------
    // 2. Explicit distinguisher D(w) := [ high_coords(w) == leak_target ].
    //    Honest: accepts with probability 1. Simulator: ~ |F|^-3.
    // ---------------------------------------------------------------------
    const N: usize = 200_000;

    let mut honest_accepts = 0usize;
    for _ in 0..N {
        let s2: F = rng.random();
        if high_coords(honest_wire1(s2)) == leak_target {
            honest_accepts += 1;
        }
    }

    // Simulator draws wire[1] uniformly over EF (`simulator.rs:159`).
    let mut sim_accepts = 0usize;
    for _ in 0..N {
        let w: EF = rng.random();
        if high_coords(w) == leak_target {
            sim_accepts += 1;
        }
    }

    let honest_rate = honest_accepts as f64 / N as f64;
    let sim_rate = sim_accepts as f64 / N as f64;
    println!("[2] Distinguisher  D(w) = [ high_coords(w) == eps*plain_c_inf high coords ]:");
    println!("      honest prover  accept rate: {honest_rate:.6}  ({honest_accepts}/{N})");
    println!("      simulator      accept rate: {sim_rate:.6}  ({sim_accepts}/{N})");
    println!("      distinguishing advantage   : {:.6}", honest_rate - sim_rate);
    assert_eq!(honest_accepts, N, "honest prover always matches the leaked value");
    assert!(
        sim_accepts * 100 < N,
        "simulator spreads over EF, so it almost never matches"
    );
    println!("      => real and simulated views are trivially distinguishable.\n");

    // ---------------------------------------------------------------------
    // 3. The leaked value tracks the witness: two witnesses are separable.
    // ---------------------------------------------------------------------
    let plain_a: EF = rng.random();
    let plain_b: EF = rng.random();
    let leak_a = high_coords(eps * plain_a);
    let leak_b = high_coords(eps * plain_b);
    println!("[3] Two different witnesses (plain_c_inf = A vs B):");
    println!("      A -> leaked high coords {leak_a:?}");
    println!("      B -> leaked high coords {leak_b:?}");
    assert_ne!(leak_a, leak_b, "different witnesses leak different values");
    println!("      => the verifier reads off a witness-dependent value each round.\n");

    // ---------------------------------------------------------------------
    // 4. Paper-faithful fix: an extension-field mask coefficient.
    //    With s_j[2] ∈ EF, the live mask fully blinds h[2] and the higher
    //    coordinates become uniform over EF — matching the simulator.
    // ---------------------------------------------------------------------
    let fixed_wire1 = |s_j_2: EF| EF::from(mult_live) * s_j_2 + eps * plain_c_inf;
    let mut fixed_accepts = 0usize;
    for _ in 0..N {
        let s2: EF = rng.random();
        if high_coords(fixed_wire1(s2)) == leak_target {
            fixed_accepts += 1;
        }
    }
    let fixed_rate = fixed_accepts as f64 / N as f64;
    println!("[4] Fix: sample the mask coefficient over EF (paper-faithful blinding):");
    println!("      fixed prover   accept rate: {fixed_rate:.6}  ({fixed_accepts}/{N})");
    assert!(
        fixed_accepts * 100 < N,
        "with an EF mask the sent coordinate is uniform over EF, like the simulator"
    );
    println!("      => the distinguisher advantage collapses; the leak is closed.\n");

    println!("All assertions passed: the leak is real, and the EF-mask fix closes it.");
}
