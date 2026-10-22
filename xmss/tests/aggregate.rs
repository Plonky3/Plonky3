//! Signatures verified in leaf circuits and aggregated into one root proof.

mod common;

use common::{EF, F, Proven, test_config};
use p3_circuit::KeyBinding;
use p3_circuit::recursion::{aggregate, public_digest};
use p3_field::PrimeCharacteristicRing;
use p3_xmss::{Claim, Params, Xmss, chip, leaf_circuit};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
#[ignore = "proves a node circuit that verifies two proofs; run with --include-ignored"]
fn four_signatures_aggregate_into_one_root() {
    let params = Params::with_lifetime(4);
    let xmss = Xmss::new(params);
    let config = test_config();
    let mut rng = StdRng::seed_from_u64(1);

    // Four signers sign one message at one epoch, two per leaf.
    let (epoch, message) = (9, [F::from_u32(42); 8]);
    let claims: Vec<Claim> = (0..4)
        .map(|_| {
            let (public_key, secret) = xmss.generate(&mut rng, 16);
            let signature = xmss.sign(&secret, epoch, &message).unwrap();
            Claim {
                public_key,
                epoch,
                message,
                signature,
            }
        })
        .collect();
    let leaves: Vec<Proven> = claims
        .chunks(2)
        .map(|pair| {
            Proven::new(
                &config,
                leaf_circuit::<EF, _>(chip(), &params, pair).unwrap(),
            )
        })
        .collect();

    // The root verifies both leaves, and its own proof verifies natively.
    let children: Vec<_> = leaves.iter().map(Proven::child).collect();
    let root = aggregate(chip(), &config, KeyBinding::Constant, &children).unwrap();
    let root = Proven::new(&config, root);

    // Whoever holds the claims recomputes the root's public digest from them alone.
    let leaf_publics: Vec<Vec<Vec<F>>> = claims
        .chunks(2)
        .map(|pair| vec![pair.iter().flat_map(Claim::public_values).collect()])
        .collect();
    let leaf_publics: Vec<&[Vec<F>]> = leaf_publics.iter().map(Vec::as_slice).collect();
    assert_eq!(
        root.public_values.concat(),
        public_digest(&chip(), &leaf_publics)
    );
}

#[test]
fn a_forged_leaf_public_value_fails_aggregation() {
    let params = Params::with_lifetime(2);
    let xmss = Xmss::new(params);
    let config = test_config();
    let (public_key, secret) = xmss.generate(&mut StdRng::seed_from_u64(2), 4);
    let message = [F::ONE; 8];
    let claim = Claim {
        public_key,
        epoch: 1,
        message,
        signature: xmss.sign(&secret, 1, &message).unwrap(),
    };
    let mut leaf = Proven::new(
        &config,
        leaf_circuit::<EF, _>(chip(), &params, &[claim]).unwrap(),
    );

    // The leaf claims another message than the one its proof shows signed.
    let publics = leaf
        .public_values
        .iter_mut()
        .find(|row| !row.is_empty())
        .unwrap();
    *publics.last_mut().unwrap() += F::ONE;
    assert!(aggregate(chip(), &config, KeyBinding::Constant, &[leaf.child()]).is_err());
}
