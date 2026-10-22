//! Signatures over keyed BLAKE3: natively, and aggregated in one binary-field proof.

use p3_flock::{FlockChallenger, FlockConfig};
use p3_keccak::Keccak256Hash;
use p3_xmss::{AggregationError, BinaryAggregator, Blake3Hash, Claim, Params, Statement, Xmss};
use rand::SeedableRng;
use rand::rngs::StdRng;

// Sixteen chains of base 4 over a lifetime of 4 epochs keep the circuit small.
const PARAMS: Params = Params::new(2, 2, 16, 24);

fn challenger() -> FlockChallenger {
    FlockChallenger::from_hasher(b"xmss-binary-test".to_vec(), Keccak256Hash)
}

fn claims(xmss: &Xmss<Blake3Hash>, seed: u64, count: usize) -> Vec<Claim<Blake3Hash>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|i| {
            let (public_key, secret) = xmss.generate(&mut rng, 4);
            let epoch = (i % 4) as u32;
            let message = [i as u8; 32];
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
fn blake3_signatures_verify_natively() {
    for params in [PARAMS, Params::with_lifetime(20)] {
        let xmss = Xmss::with_hash(params, Blake3Hash);
        let claim = claims(&xmss, 1, 1).remove(0);
        let verify = |claim: &Claim<Blake3Hash>| {
            xmss.verify(
                &claim.public_key,
                claim.epoch,
                &claim.message,
                &claim.signature,
            )
        };
        assert_eq!(verify(&claim), Ok(()));

        let mut chain = claim.clone();
        chain.signature.chains[3][0] ^= 1;
        let mut path = claim.clone();
        path.signature.path[1][15] ^= 1;
        let mut message = claim.clone();
        message.message[0] ^= 1;
        let mut epoch = claim.clone();
        epoch.epoch += 1 << params.log_lifetime();
        for tampered in [chain, path, message, epoch] {
            assert!(verify(&tampered).is_err());
        }
    }
}

#[test]
fn signatures_aggregate_into_one_binary_proof() {
    let xmss = Xmss::with_hash(PARAMS, Blake3Hash);
    let aggregator = BinaryAggregator::new(PARAMS, FlockConfig::default());

    // Three claims pad to a batch of four.
    let claims = claims(&xmss, 2, 3);
    let proof = aggregator.prove(&claims, &mut challenger()).unwrap();
    let statements: Vec<Statement<Blake3Hash>> = claims.iter().map(Claim::statement).collect();
    aggregator
        .verify(&statements, &proof, &mut challenger())
        .unwrap();

    // The proof does not hold for another message, epoch or key.
    let mut message = statements.clone();
    message[1].message[5] ^= 1;
    let mut epoch = statements.clone();
    epoch[2].epoch ^= 1;
    let mut key = statements.clone();
    key[0].public_key = statements[1].public_key;
    for statements in [message, epoch, key] {
        assert!(
            aggregator
                .verify(&statements, &proof, &mut challenger())
                .is_err()
        );
    }
}

#[test]
fn a_bad_signature_cannot_be_aggregated() {
    let xmss = Xmss::with_hash(PARAMS, Blake3Hash);
    let aggregator = BinaryAggregator::new(PARAMS, FlockConfig::default());
    let valid = claims(&xmss, 3, 2);

    let mut chain = valid.clone();
    chain[1].signature.chains[7][2] ^= 4;
    let mut path = valid.clone();
    path[0].signature.path[0][0] ^= 1;
    let mut randomness = valid;
    randomness[1].signature.randomness[3] ^= 1;
    for claims in [chain, path, randomness] {
        assert!(matches!(
            aggregator.prove(&claims, &mut challenger()),
            Err(AggregationError::Proving(_))
        ));
    }
    assert!(matches!(
        aggregator.prove(&[], &mut challenger()),
        Err(AggregationError::Empty)
    ));
}
