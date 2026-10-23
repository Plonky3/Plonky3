//! The native scheme: keys, signing and verification.

use p3_xmss::{Blake3Hash, Params, SignError, Xmss};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn message(seed: u32) -> [u8; 32] {
    core::array::from_fn(|i| (seed * 32 + i as u32) as u8)
}

fn xmss(params: Params) -> Xmss<Blake3Hash> {
    Xmss::with_hash(params, Blake3Hash)
}

#[test]
fn every_active_epoch_signs_and_verifies() {
    let xmss = xmss(Params::with_lifetime(3));
    let (public, secret) = xmss.generate(&mut StdRng::seed_from_u64(1), 8);
    for epoch in 0..8 {
        let signature = xmss.sign(&secret, epoch, &message(epoch)).unwrap();
        assert_eq!(
            xmss.verify(&public, epoch, &message(epoch), &signature),
            Ok(())
        );
    }
}

#[test]
fn a_signature_binds_message_epoch_and_key() {
    let xmss = xmss(Params::with_lifetime(3));
    let mut rng = StdRng::seed_from_u64(2);
    let (public, secret) = xmss.generate(&mut rng, 8);
    let (other, _) = xmss.generate(&mut rng, 8);
    let signature = xmss.sign(&secret, 5, &message(0)).unwrap();

    assert!(xmss.verify(&public, 5, &message(1), &signature).is_err());
    assert!(xmss.verify(&public, 4, &message(0), &signature).is_err());
    assert!(xmss.verify(&other, 5, &message(0), &signature).is_err());
    // An epoch beyond the lifetime shares its low bits with epoch 5, yet fails.
    assert!(
        xmss.verify(&public, 5 + 8, &message(0), &signature)
            .is_err()
    );
}

#[test]
fn tampered_signatures_are_rejected() {
    let xmss = xmss(Params::with_lifetime(3));
    let (public, secret) = xmss.generate(&mut StdRng::seed_from_u64(3), 8);
    let signature = xmss.sign(&secret, 2, &message(7)).unwrap();

    let mut chain = signature.clone();
    chain.chains[17][3] ^= 1;
    let mut path = signature.clone();
    path.path[1][0] ^= 1;
    let mut randomness = signature.clone();
    randomness.randomness[0] ^= 1;
    let mut short = signature;
    short.path.pop();

    for tampered in [chain, path, randomness, short] {
        assert!(xmss.verify(&public, 2, &message(7), &tampered).is_err());
    }
}

#[test]
fn a_sparse_tree_signs_its_active_epochs_only() {
    // A lifetime of 2^20 epochs, of which the first four are active.
    let xmss = xmss(Params::with_lifetime(20));
    let (public, secret) = xmss.generate(&mut StdRng::seed_from_u64(4), 4);
    assert_eq!(secret.active_epochs(), 4);

    let signature = xmss.sign(&secret, 3, &message(1)).unwrap();
    assert_eq!(signature.path.len(), 20);
    assert_eq!(xmss.verify(&public, 3, &message(1), &signature), Ok(()));
    assert_eq!(
        xmss.sign(&secret, 4, &message(1)),
        Err(SignError::InactiveEpoch(4))
    );
}

#[test]
fn other_parameters_sign_and_verify() {
    // Base 8 and base 16 chains, with the target at the expected digit sum.
    for params in [Params::new(2, 3, 42, 147), Params::new(2, 4, 32, 240)] {
        let xmss = xmss(params);
        let (public, secret) = xmss.generate(&mut StdRng::seed_from_u64(5), 4);
        let signature = xmss.sign(&secret, 1, &message(2)).unwrap();
        assert_eq!(xmss.verify(&public, 1, &message(2), &signature), Ok(()));
    }
}
