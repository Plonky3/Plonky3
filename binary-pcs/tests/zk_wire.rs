#![cfg(feature = "experimental-zk")]

use std::sync::OnceLock;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_binary_field::TowerLevel;
use p3_binary_pcs::zk::{ReferenceZkAir, Val, ZkAirConfig, ZkAirProof};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::MerkleCap;
use proptest::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

struct Product;
impl BaseAir<Val> for Product {
    fn width(&self) -> usize {
        2
    }
    fn num_public_values(&self) -> usize {
        1
    }
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}
impl<AB: AirBuilder<F = Val>> Air<AB> for Product {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let public: AB::Expr = builder.public_values()[0].into();
        builder.assert_zero(main.current_slice()[0] * main.current_slice()[1] - public);
    }
}
fn air() -> ReferenceZkAir {
    ReferenceZkAir::compile(&Product, ZkAirConfig::new(1, 1).unwrap()).unwrap()
}
fn proof() -> &'static ZkAirProof {
    static PROOF: OnceLock<ZkAirProof> = OnceLock::new();
    PROOF.get_or_init(|| {
        air()
            .prove(
                RowMajorMatrix::new(vec![Val::ONE; 4], 2),
                &[Val::ONE],
                &mut StdRng::seed_from_u64(83),
            )
            .unwrap()
    })
}

#[test]
fn wire_roundtrip_preserves_every_field_and_verifies_statement() {
    let air = air();
    let encoded = air.encode_proof(proof()).unwrap();
    // M=3, q=12, batch width=3, L=16, n=64: 96+96+768+24*256.
    assert_eq!(air.encoded_proof_len(), 7104);
    assert_eq!(encoded.len(), 7104);
    let decoded = air.decode_proof(&encoded).unwrap();
    assert_eq!(
        postcard::to_allocvec(&decoded).unwrap(),
        postcard::to_allocvec(proof()).unwrap()
    );
    assert_eq!(air.encode_proof(&decoded).unwrap(), encoded);
    air.verify_bytes(&encoded, &[Val::ONE]).unwrap();
    assert!(air.verify_bytes(&encoded, &[Val::ZERO]).is_err());
    assert!(air.verify_bytes(&encoded, &[]).is_err());
}

#[test]
fn wire_rejects_every_truncated_prefix_trailing_and_oversized_data() {
    let air = air();
    let mut encoded = air.encode_proof(proof()).unwrap();
    for len in 0..encoded.len() {
        assert!(air.decode_proof(&encoded[..len]).is_err(), "prefix {len}");
    }
    encoded.push(0);
    assert!(air.decode_proof(&encoded).is_err());
    encoded.resize(1 << 20, 0);
    assert!(air.verify_bytes(&encoded, &[Val::ONE]).is_err());
}

#[test]
fn wire_rejects_foreign_magic_version_and_fixed_width_geometry() {
    let air = air();
    let encoded = air.encode_proof(proof()).unwrap();
    for offset in [0, 7] {
        let mut bytes = encoded.clone();
        bytes[offset] ^= 1;
        assert!(air.decode_proof(&bytes).is_err());
    }
    for offset in [8, 16, 24] {
        for value in [0, 1, 4, u64::MAX, 1 << 32] {
            let mut bytes = encoded.clone();
            bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            assert!(air.decode_proof(&bytes).is_err());
        }
    }
    let taller = ReferenceZkAir::compile(&Product, ZkAirConfig::new(2, 1).unwrap()).unwrap();
    assert!(taller.decode_proof(&encoded).is_err());
}

#[test]
fn wire_encoder_rejects_every_malformed_commitment_and_opening_shape() {
    let air = air();
    let mutations: &[fn(&mut ZkAirProof)] = &[
        |p| p.commitment.num_polynomials = 0,
        |p| p.commitment.num_polynomials += 1,
        |p| p.commitment.message_len += 1,
        |p| p.commitment.num_queries += 1,
        |p| p.commitment.root = MerkleCap::new(vec![[0; 32]; 2]),
        |p| p.opening.mask_root = MerkleCap::new(vec![[0; 32]; 2]),
        |p| {
            p.opening.claimed_values.pop();
        },
        |p| p.opening.claimed_values.push(Val::ZERO),
        |p| {
            p.opening.sigma.pop();
        },
        |p| p.opening.sigma.push(Val::ZERO),
        |p| {
            p.opening.blended_coefficients.pop();
        },
        |p| p.opening.blended_coefficients.push(vec![]),
        |p| {
            p.opening.blended_coefficients[0].pop();
        },
        |p| p.opening.blended_coefficients[0].push(Val::ZERO),
        |p| {
            p.opening.queries.pop();
        },
        |p| p.opening.queries.push(p.opening.queries[0].clone()),
        |p| p.opening.queries[0].h.values.clear(),
        |p| p.opening.queries[0].u.values.push(Val::ZERO),
        |p| p.opening.queries[0].h.authentication_path.clear(),
        |p| p.opening.queries[0].u.authentication_path.push([0; 32]),
    ];
    for (index, mutate) in mutations.iter().enumerate() {
        let mut malformed = proof().clone();
        mutate(&mut malformed);
        assert!(air.encode_proof(&malformed).is_err(), "mutation {index}");
    }
}

#[test]
fn wire_known_fixture_pins_little_endian_geometry_field_repr_and_row_order() {
    let air = air();
    // Construct the wire fixture independently of the encoder. Distinct high
    // bytes distinguish tower representation from integer embedding/truncation.
    let repr = u128::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 128]);
    let mut fixture = Vec::new();
    fixture.extend_from_slice(b"P3ZKAIR1\x03\0\0\0\0\0\0\0\x03\0\0\0\0\0\0\0\x0c\0\0\0\0\0\0\0");
    fixture.extend_from_slice(&[0x42; 32]);
    for delta in 0..3 {
        fixture.extend_from_slice(&(repr + delta).to_le_bytes());
    }
    fixture.extend_from_slice(&[0x24; 32]);
    for delta in 3..54 {
        fixture.extend_from_slice(&(repr + delta).to_le_bytes());
    }
    for query in 0..12u8 {
        for row in 0..2u8 {
            for column in 0..4 {
                fixture.extend_from_slice(
                    &(repr + 1000 + u128::from(query) * 8 + u128::from(row) * 4 + column)
                        .to_le_bytes(),
                );
            }
            for depth in 0..6 {
                fixture.extend_from_slice(&[query * 12 + row * 6 + depth; 32]);
            }
        }
    }
    assert_eq!(fixture.len(), 7104);
    let decoded = air.decode_proof(&fixture).unwrap();
    assert_eq!(decoded.commitment.num_polynomials, 3);
    assert_eq!(decoded.commitment.message_len, 3);
    assert_eq!(decoded.commitment.num_queries, 12);
    assert_eq!(decoded.commitment.root.as_ref(), &[[0x42; 32]]);
    assert_eq!(decoded.opening.mask_root.as_ref(), &[[0x24; 32]]);
    assert_eq!(decoded.opening.claimed_values[2].to_repr(), repr + 2);
    assert_eq!(decoded.opening.sigma[0].to_repr(), repr + 3);
    assert_eq!(
        decoded.opening.blended_coefficients[1][0].to_repr(),
        repr + 22
    );
    assert_eq!(
        decoded.opening.queries[11].u.values[2].to_repr(),
        repr + 1094
    );
    assert_eq!(decoded.opening.queries[11].u.salt.to_repr(), repr + 1095);
    assert_eq!(
        decoded.opening.queries[11].u.authentication_path[5],
        [143; 32]
    );
    assert_eq!(air.encode_proof(&decoded).unwrap(), fixture);
    assert!(air.verify_bytes(&fixture, &[Val::ONE]).is_err());
}

#[test]
fn wire_tampering_each_authenticated_component_fails_verification() {
    let air = air();
    let encoded = air.encode_proof(proof()).unwrap();
    // Commitment, claims, mask root, sigma, blend, then each h/u leaf,
    // salt and path. The header has separate rejection tests above.
    for offset in [
        32, 64, 112, 144, 192, 960, 1008, 1024, 1216, 1264, 1280, 7103,
    ] {
        let mut bytes = encoded.clone();
        bytes[offset] ^= 1;
        assert!(
            air.verify_bytes(&bytes, &[Val::ONE]).is_err(),
            "offset {offset}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn wire_arbitrary_bounded_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..16000)) {
        let air = air();
        if let Ok(decoded) = air.decode_proof(&bytes) {
            prop_assert_eq!(air.encode_proof(&decoded).unwrap(), bytes.clone());
        }
        let _ = air.verify_bytes(&bytes, &[Val::ONE]);
    }
    #[test]
    fn wire_valid_proof_single_bit_mutations_reject_without_panics(offset in 0usize..7104, bit in 0u8..8) {
        let air = air();
        let mut bytes = air.encode_proof(proof()).unwrap();
        bytes[offset] ^= 1 << bit;
        prop_assert!(air.verify_bytes(&bytes, &[Val::ONE]).is_err());
    }
}
