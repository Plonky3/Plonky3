//! Instantiation sweep for ring switching: the same round-trip over a base/extension pair at
//! each end of the design space, plus the two guards that keep the construction honest about
//! its own preconditions.

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_binary_field::{BinaryChallenger, BinaryField8, BinaryField16, BinaryField128};
use p3_challenger::{DuplexChallenger, HashChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::ring_switch::{pack, packed_vars, prove_ring_switch, verify_ring_switch};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// A fresh transcript for the `BinaryField8` pairs, built identically for both sides.
const fn binary_challenger() -> BinaryChallenger<BinaryField8, HashChallenger<u8, Keccak256Hash, 32>>
{
    BinaryChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// A fresh transcript for the `BabyBear` pair, built identically for both sides.
fn baby_bear_challenger() -> DuplexChallenger<BabyBear, Poseidon2BabyBear<16>, 16, 8> {
    let mut rng = SmallRng::seed_from_u64(42);
    let perm = Poseidon2BabyBear::<16>::new_from_rng_128(&mut rng);
    DuplexChallenger::new(perm)
}

/// The production pair: `GF(2^8)` into `GF(2^128)`, `κ = 4`.
#[test]
fn binary_8_into_128() {
    type F = BinaryField8;
    type EF = BinaryField128;

    let mut rng = SmallRng::seed_from_u64(100);
    for ell in 5..=8 {
        let t = Poly::<F>::rand(&mut rng, ell);
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut rng, ell);
        let s = t.eval_base(&r);

        let mut p_chal = binary_challenger();
        let (proof, r_prime_p, s_prime_p) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);

        let mut v_chal = binary_challenger();
        let (r_prime_v, s_prime_v) =
            verify_ring_switch::<F, EF, _>(&proof, &r, s, &mut v_chal).unwrap();

        assert_eq!(r_prime_p, r_prime_v, "ell = {ell}");
        assert_eq!(s_prime_p, s_prime_v, "ell = {ell}");
        assert_eq!(s_prime_v, packed.eval_ext::<F>(&r_prime_v), "ell = {ell}");
    }
}

/// The smallest binary pair, `κ = 1` — the shape of the paper's worked example, and small
/// enough that `ell = κ` enumerates the whole reduction as an empty sumcheck.
#[test]
fn binary_8_into_16() {
    type F = BinaryField8;
    type EF = BinaryField16;

    let mut rng = SmallRng::seed_from_u64(200);
    for ell in 1..=6 {
        let t = Poly::<F>::rand(&mut rng, ell);
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut rng, ell);
        let s = t.eval_base(&r);

        let mut p_chal = binary_challenger();
        let (proof, r_prime_p, s_prime_p) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);

        let mut v_chal = binary_challenger();
        let (r_prime_v, s_prime_v) =
            verify_ring_switch::<F, EF, _>(&proof, &r, s, &mut v_chal).unwrap();

        assert_eq!(r_prime_p, r_prime_v, "ell = {ell}");
        assert_eq!(s_prime_p, s_prime_v, "ell = {ell}");
        assert_eq!(s_prime_v, packed.eval_ext::<F>(&r_prime_v), "ell = {ell}");
    }
}

/// A non-binary extension, to demonstrate the reduction is not characteristic-2 specific.
/// Degree 4 is a power of two, so `κ = 2`.
#[test]
fn baby_bear_into_degree_four() {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut rng = SmallRng::seed_from_u64(300);
    for ell in 3..=7 {
        let t = Poly::<F>::rand(&mut rng, ell);
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut rng, ell);
        let s = t.eval_base(&r);

        let mut p_chal = baby_bear_challenger();
        let (proof, r_prime_p, s_prime_p) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);

        let mut v_chal = baby_bear_challenger();
        let (r_prime_v, s_prime_v) =
            verify_ring_switch::<F, EF, _>(&proof, &r, s, &mut v_chal).unwrap();

        assert_eq!(r_prime_p, r_prime_v, "ell = {ell}");
        assert_eq!(s_prime_p, s_prime_v, "ell = {ell}");
        assert_eq!(s_prime_v, packed.eval_ext::<F>(&r_prime_v), "ell = {ell}");
    }
}

/// `packed_vars` rejects a non-power-of-two degree rather than computing nonsense.
#[test]
#[should_panic = "power-of-two extension degree"]
fn a_degree_five_extension_is_rejected() {
    let _ = packed_vars::<BabyBear, BinomialExtensionField<BabyBear, 5>>();
}
