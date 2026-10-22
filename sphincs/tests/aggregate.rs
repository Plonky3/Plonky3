use p3_flock::{FlockChallenger, FlockConfig};
use p3_keccak::Keccak256Hash;
use p3_sphincs::{Aggregator, Claim, PUBLIC_KEY_BYTES, PublicKey, Signature, Statement};
use slh_dsa::{Sha2_128s, SigningKey};

fn challenger() -> FlockChallenger {
    FlockChallenger::from_hasher(b"sphincs-test".to_vec(), Keccak256Hash)
}

/// A claim of a deterministic key over `[tag; 32]`.
fn claim(tag: u8) -> Claim {
    let sk =
        SigningKey::<Sha2_128s>::slh_keygen_internal(&[tag; 16], &[!tag; 16], &[tag ^ 0x5A; 16]);
    let key: [u8; PUBLIC_KEY_BYTES] = sk.as_ref().to_bytes().as_slice().try_into().unwrap();
    let message = [tag; 32];
    let signature = sk
        .try_sign_with_context(&message, &[], None)
        .unwrap()
        .to_bytes();
    Claim {
        statement: Statement {
            public_key: PublicKey::from_bytes(&key),
            message,
        },
        signature: Signature::try_from(signature.as_slice()).unwrap(),
    }
}

#[test]
#[ignore = "builds a 2^27-bit circuit with 27 GB peak memory; run with --release --include-ignored"]
fn signatures_aggregate_and_tampering_fails() {
    // One aggregator for every case: the circuit alone takes several gigabytes.
    let aggregator = Aggregator::new(FlockConfig::default());
    assert_eq!(aggregator.compressions(), 3999);

    // Three claims pad to a batch of four.
    let claims: Vec<Claim> = (0..3).map(claim).collect();
    let statements: Vec<Statement> = claims.iter().map(|claim| claim.statement).collect();
    for claim in &claims {
        let Statement {
            public_key,
            message,
        } = claim.statement;
        assert_eq!(public_key.verify(&message, &[], &claim.signature), Ok(()));
    }
    let proof = aggregator.prove(&claims, &mut challenger()).unwrap();
    aggregator
        .verify(&statements, &proof, &mut challenger())
        .unwrap();

    // The proof does not hold for another message or key, or fewer statements.
    let mut message = statements.clone();
    message[1].message[5] ^= 1;
    let mut key = statements.clone();
    key[0].public_key = statements[1].public_key;
    let mut seed = statements.clone();
    seed[2].public_key.seed[0] ^= 1;
    for statements in [message, key, seed, statements[..2].to_vec()] {
        assert!(
            aggregator
                .verify(&statements, &proof, &mut challenger())
                .is_err()
        );
    }

    // A flipped bit in R, a FORS path, or a WOTS+ chain of the top layer yields no accepted proof.
    for at in [3, 100, 7856 - 704 + 5] {
        let mut tampered = claims.clone();
        let mut bytes = *tampered[1].signature.as_bytes();
        bytes[at] ^= 1;
        tampered[1].signature = Signature::try_from(&bytes[..]).unwrap();
        let accepted = aggregator
            .prove(&tampered, &mut challenger())
            .is_ok_and(|proof| {
                aggregator
                    .verify(&statements, &proof, &mut challenger())
                    .is_ok()
            });
        assert!(!accepted, "byte {at}");
    }
}
