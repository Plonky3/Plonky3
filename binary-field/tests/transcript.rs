use p3_binary_field::{BinaryChallenger, BinaryField128, TowerLevel};
use p3_challenger::HashChallenger;
use p3_challenger::fs::{
    Codec, DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction,
    InteractionPattern, Kind, Length, ProverState, TypeTag, Unit, VerifierState,
};
use p3_challenger::testing::Recorder;
use p3_keccak::Keccak256Hash;
use proptest::prelude::*;

type F = BinaryField128;
type Ch = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;
type Cdc = FieldToFieldCodec<F>;

fn seed(bytes: &[u8]) -> Vec<F> {
    let mut recorder = Recorder::default();
    FieldUnit::<F>::observe_bytes(&mut recorder, bytes);
    recorder.into_absorbed()
}

#[test]
fn binary_seed_preserves_bytes_and_length_in_the_tower_basis() {
    assert_ne!(seed(&[2]), seed(&[0]));
    assert_ne!(seed(&[0xaa]), seed(&[0xaa, 0]));
    assert_ne!(seed(&[]), seed(&[0]));
    assert_eq!(seed(&[1, 2, 3]), [F::from_repr(3), F::from_repr(0x030201)]);
}

#[test]
fn binary_wire_encoding_is_canonical_and_uses_every_bit() {
    for raw in [0, 1, 2, 1 << 127, u128::MAX] {
        let value = F::from_repr(raw);
        let mut bytes = Vec::new();
        <Cdc as Codec<Ch, F>>::encode(&value, &mut bytes);
        assert_eq!(bytes, raw.to_be_bytes());
        assert_eq!(<Cdc as Codec<Ch, F>>::decode(&bytes).unwrap(), value);
        assert_eq!(bytes.len(), <Cdc as Codec<Ch, F>>::wire_len());
    }
    assert!(<Cdc as Codec<Ch, F>>::decode(&[0; 15]).is_err());
}

#[test]
fn binary_type_tag_commits_to_the_coefficient_field() {
    // Both carry 128 bits, but one is a native tower element and the other
    // is decomposed into 128 GF(2) coefficients; they have different encodings.
    let native =
        Interaction::algebra::<F, F>(Hierarchy::Atomic, Kind::Message, "x", Length::Scalar);
    assert_ne!(
        native.type_tag(),
        TypeTag::Algebra {
            modulus: 2,
            degree: 128,
            basis: [0; 32],
        }
    );
    assert_eq!(
        format!("{native:#}"),
        "Atomic Message 1 x Scalar BinaryTower(128^1;cfa353fe66eeeae7600254a68073d57125abac3820fde7a3e32c84482b826469)"
    );
}

#[test]
fn binary_typed_wire_round_trip() {
    let pattern = InteractionPattern::new(vec![
        Interaction::algebra::<F, F>(Hierarchy::Atomic, Kind::Message, "x", Length::Scalar),
        Interaction::algebra::<F, F>(Hierarchy::Atomic, Kind::Challenge, "r", Length::Scalar),
    ])
    .unwrap();
    let ds = DomainSeparator::<FieldUnit<F>>::new(1, b"binary-test", pattern);
    let fresh = || Ch::from_hasher(Vec::new(), Keccak256Hash);
    let value = F::from_repr(1 << 127 | 2);
    let mut prover = ProverState::new(fresh(), &ds);
    prover.add_scalar::<F, Cdc>("x", &value);
    let challenge = prover.challenge_scalar::<F, Cdc>("r").into_inner();
    let wire = prover.finalize();
    assert_eq!(wire, value.to_repr().to_be_bytes());
    let mut verifier = VerifierState::new(fresh(), &ds, &wire);
    assert_eq!(
        verifier.next_scalar::<F, Cdc>("x").unwrap().into_inner(),
        value
    );
    assert_eq!(
        verifier.challenge_scalar::<F, Cdc>("r").into_inner(),
        challenge
    );
    verifier.finalize().unwrap();
}

proptest! {
    #[test]
    fn binary_seed_recovers_every_byte(bytes in prop::collection::vec(any::<u8>(), 0..65)) {
        let encoded = seed(&bytes);
        prop_assert_eq!(encoded[0].to_repr(), bytes.len() as u128);
        let recovered: Vec<u8> = encoded[1..].iter()
            .flat_map(|value| value.to_repr().to_le_bytes())
            .take(bytes.len()).collect();
        prop_assert_eq!(recovered, bytes);
    }

    #[test]
    fn binary_wire_round_trips(raw in any::<u128>()) {
        let mut bytes = Vec::new();
        <Cdc as Codec<Ch, F>>::encode(&F::from_repr(raw), &mut bytes);
        prop_assert_eq!(<Cdc as Codec<Ch, F>>::decode(&bytes).unwrap().to_repr(), raw);
    }
}
