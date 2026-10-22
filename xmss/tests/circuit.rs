//! Leaf circuits: their fixed shape, their proofs and their rejections.

mod common;

use common::{EF, F, Proven, test_config};
use p3_air::BaseAir;
use p3_field::PrimeCharacteristicRing;
use p3_xmss::{Claim, Params, Xmss, chip, leaf_circuit};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// `count` claims from fresh keys, each signing a message of `seed` at epoch `epoch`.
fn claims(xmss: &Xmss, seed: u64, epoch: u32, count: usize) -> Vec<Claim> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|i| {
            let (public_key, secret) = xmss.generate(&mut rng, 1 << xmss.params().log_lifetime());
            let message = core::array::from_fn(|j| F::from_u64(seed + (8 * i + j) as u64));
            let signature = xmss.sign(&secret, epoch, &message).unwrap();
            Claim {
                public_key,
                epoch,
                message,
                signature,
            }
        })
        .collect()
}

#[test]
fn two_signature_sets_give_one_verifying_key() {
    let params = Params::with_lifetime(3);
    let xmss = Xmss::new(params);

    // Different keys, messages, epochs, digits and paths, under the same parameters.
    let [one, two] = [(1, 2), (2, 7)].map(|(seed, epoch)| {
        leaf_circuit::<EF, _>(chip(), &params, &claims(&xmss, seed, epoch, 2)).unwrap()
    });
    assert_eq!(one.airs.len(), two.airs.len());
    for (a, b) in one.airs.iter().zip(&two.airs) {
        assert_eq!(a.preprocessed_trace(), b.preprocessed_trace());
    }
    assert_ne!(one.publics, two.publics);
}

#[test]
#[ignore = "proves a two-signature leaf circuit; run with --include-ignored"]
fn a_leaf_circuit_proves_and_verifies() {
    let params = Params::with_lifetime(3);
    let xmss = Xmss::new(params);
    let claims = claims(&xmss, 3, 5, 2);
    let circuit = leaf_circuit::<EF, _>(chip(), &params, &claims).unwrap();
    let expected: Vec<F> = claims.iter().flat_map(Claim::public_values).collect();
    assert_eq!(circuit.publics, expected);
    Proven::new(&test_config(), circuit);
}

#[test]
fn a_leaf_circuit_rejects_a_bad_claim() {
    let params = Params::with_lifetime(2);
    let xmss = Xmss::new(params);
    let valid = claims(&xmss, 4, 1, 1).remove(0);

    let mut chain = valid.clone();
    chain.signature.chains[0][0] += F::ONE;
    let mut path = valid.clone();
    path.signature.path[0][0] += F::ONE;
    let mut message = valid.clone();
    message.message[7] += F::ONE;
    // Same low bits as the signed epoch, but past the lifetime.
    let mut epoch = valid.clone();
    epoch.epoch += 4;

    for claim in [chain, path, message, epoch] {
        assert!(leaf_circuit::<EF, _>(chip(), &params, &[valid.clone(), claim]).is_err());
    }
}
