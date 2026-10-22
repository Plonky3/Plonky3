use p3_sphincs::{PUBLIC_KEY_BYTES, PublicKey, Signature, SignatureError};
use slh_dsa::{Sha2_128s, SigningKey};

/// Bytes from hex, with `-` for the empty string.
fn hex(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// A deterministic key pair from three seeds, and its public key.
fn keypair(tag: u8) -> (SigningKey<Sha2_128s>, PublicKey) {
    let sk =
        SigningKey::<Sha2_128s>::slh_keygen_internal(&[tag; 16], &[tag ^ 1; 16], &[tag ^ 2; 16]);
    let bytes: [u8; PUBLIC_KEY_BYTES] = sk.as_ref().to_bytes().as_slice().try_into().unwrap();
    (sk, PublicKey::from_bytes(&bytes))
}

#[test]
fn acvp_sigver_vectors_verify_as_expected() {
    // NIST ACVP SLH-DSA-sigVer-FIPS205 cases for SLH-DSA-SHA2-128s, internal and pure external interfaces.
    let vectors = include_str!("data/sigver-sha2-128s.txt");
    let mut cases = 0;
    for line in vectors.lines() {
        let [id, expected, interface, pk, context, message, signature] =
            line.split(' ').collect::<Vec<_>>().try_into().unwrap();
        let pk = PublicKey::from_bytes(&hex(pk).try_into().unwrap());
        let (message, context) = (hex(message), hex(context));
        let verdict = Signature::try_from(hex(signature).as_slice()).and_then(|signature| {
            if interface == "internal" {
                pk.verify_internal(&message, &signature)
            } else {
                pk.verify(&message, &context, &signature)
            }
        });
        assert_eq!(
            verdict.is_ok(),
            expected == "valid",
            "case {id}: {verdict:?}"
        );
        cases += 1;
    }
    assert_eq!(cases, 8);
}

#[test]
fn signatures_of_the_standard_signer_verify() {
    for tag in 0..4 {
        let (sk, pk) = keypair(tag);
        let message = [tag; 32];
        for context in [&b""[..], b"context"] {
            let signature = sk.try_sign_with_context(&message, context, None).unwrap();
            let signature = Signature::try_from(signature.to_bytes().as_slice()).unwrap();
            assert_eq!(pk.verify(&message, context, &signature), Ok(()));
        }
    }
}

#[test]
fn a_tampered_signature_message_or_key_fails() {
    let (sk, pk) = keypair(7);
    let message = [3; 32];
    let bytes = sk
        .try_sign_with_context(&message, &[], None)
        .unwrap()
        .to_bytes();
    let signature = Signature::try_from(bytes.as_slice()).unwrap();
    assert_eq!(pk.verify(&message, &[], &signature), Ok(()));

    // One flipped bit in R, a FORS secret, a FORS path, a WOTS+ chain, and the last path node.
    for at in [0, 16, 40, 2928, bytes.len() - 1] {
        let mut tampered = bytes.to_vec();
        tampered[at] ^= 1;
        let tampered = Signature::try_from(tampered.as_slice()).unwrap();
        assert_eq!(
            pk.verify(&message, &[], &tampered),
            Err(SignatureError::Invalid),
            "byte {at}"
        );
    }
    assert_eq!(
        pk.verify(&[4; 32], &[], &signature),
        Err(SignatureError::Invalid)
    );
    assert_eq!(
        pk.verify(&message, b"other", &signature),
        Err(SignatureError::Invalid)
    );
    let (_, other) = keypair(8);
    assert_eq!(
        other.verify(&message, &[], &signature),
        Err(SignatureError::Invalid)
    );
    assert_eq!(
        Signature::try_from(&bytes[1..]),
        Err(SignatureError::Length(bytes.len() - 1))
    );
    assert_eq!(
        pk.verify(&message, &[0; 256], &signature),
        Err(SignatureError::Context(256))
    );
}
