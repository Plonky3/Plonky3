//! Proof-to-byte-grammar conformance on real AIR proofs. No configurable RO.
use alloc::vec::Vec;

use p3_air::{AirBuilder, BaseAir, WindowAccess};
use p3_binary_field::TowerLevel;
use p3_keccak::Keccak256Hash;
use p3_symmetric::CryptographicHasher;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::super::polynomial::evaluate;
use super::*;

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
        let p: AB::Expr = builder.public_values()[0].into();
        builder.assert_zero(main.current_slice()[0] * main.current_slice()[1] - p);
    }
}

// Independently framed records: these helpers never call Challenger.
fn u(out: &mut Vec<u8>, value: usize) {
    out.push(4);
    out.extend_from_slice(&(value as u64).to_le_bytes());
}
fn b(out: &mut Vec<u8>, value: &[u8]) {
    out.push(3);
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value);
}
fn f(out: &mut Vec<u8>, value: Val) {
    out.push(1);
    out.extend_from_slice(&value.to_repr().to_le_bytes());
}
fn cap(out: &mut Vec<u8>, root: &[u8; 32]) {
    out.push(2);
    out.extend_from_slice(&1u64.to_le_bytes());
    out.extend_from_slice(root);
}
fn raw(prefix: &[u8], label: &[u8], ordinal: u64, len: usize) -> Vec<u8> {
    let mut input = b"p3-binary-zk-v2/fiat-shamir\0".to_vec();
    input.extend_from_slice(&(label.len() as u64).to_le_bytes());
    input.extend_from_slice(label);
    input.extend_from_slice(&(prefix.len() as u64).to_le_bytes());
    input.extend_from_slice(prefix);
    input.extend_from_slice(&ordinal.to_le_bytes());
    input.extend_from_slice(&(len as u64).to_le_bytes());
    (0..len.div_ceil(32))
        .flat_map(|block| {
            Keccak256Hash.hash_iter(input.iter().copied().chain((block as u32).to_le_bytes()))
        })
        .take(len)
        .collect()
}
fn draw(out: &mut Vec<u8>, label: &[u8], raw: &[u8]) {
    out.push(5);
    out.extend_from_slice(&(label.len() as u64).to_le_bytes());
    out.extend_from_slice(label);
    out.extend_from_slice(&(raw.len() as u64).to_le_bytes());
    out.extend_from_slice(raw);
}
fn commitment(out: &mut Vec<u8>, root: &[u8; 32], q: usize) {
    b(out, b"p3-binary-reference-zk-pcs-v2");
    b(out, b"commitment");
    let l = 3 + q + 1;
    let n = (4 * l).next_power_of_two();
    for value in [3, q, q + 1, l, n, 3, q, 3, 1] {
        u(out, value);
    }
    cap(out, root);
}
fn select(raw: &[u8], accepts: impl Fn(Val) -> bool) -> Val {
    raw.as_chunks::<16>()
        .0
        .iter()
        .map(|x| Val::from_repr(u128::from_le_bytes(*x)))
        .find(|&x| accepts(x))
        .unwrap()
}

// A focused, fixture-owned analysis parser. It accepts arbitrary root/claim/g
// values at their designated positions, but only this compiled statement,
// constants, geometry, order and exact lengths. This is not production parsing.
struct Records<'a>(&'a [u8]);
impl<'a> Records<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let bytes = self.0.get(..n)?;
        self.0 = &self.0[n..];
        Some(bytes)
    }
    fn literal(&mut self, value: &[u8]) -> Option<()> {
        (self.take(value.len())? == value).then_some(())
    }
    fn field(&mut self) -> Option<Val> {
        self.literal(&[1])?;
        Some(Val::from_repr(u128::from_le_bytes(
            self.take(16)?.try_into().ok()?,
        )))
    }
    fn root(&mut self) -> Option<[u8; 32]> {
        self.literal(&[2])?;
        self.literal(&1u64.to_le_bytes())?;
        self.take(32)?.try_into().ok()
    }
}

#[test]
fn canonical_air_stage_prefixes_pin_records_duplicate_root_and_candidate_binding() {
    for classical in [false, true] {
        let config = if classical {
            ZkAirConfig::classical_rom(1, super::super::ClassicalRomBudget::new(1, 0, 49).unwrap())
                .unwrap()
        } else {
            ZkAirConfig::interactive(1, 1).unwrap()
        };
        let air = ReferenceZkAir::compile(&Product, config).unwrap();
        // Literal normalized postcard AST bytes, independent of shared pointers
        // or cached user degree hints.
        assert_eq!(
            air.statement,
            [
                1, 5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 4, 0, 1, 2, 0, 0, 3, 0, 2, 2, 3, 2
            ]
        );
        let mut rng = StdRng::seed_from_u64(27);
        let proof = air
            .prove(
                RowMajorMatrix::new(vec![Val::ONE; 4], 2),
                &[Val::ONE],
                &mut rng,
            )
            .unwrap();
        let mut actual = air.transcript(&[Val::ONE]);
        air.pcs.observe_commitment(&proof.commitment, &mut actual);
        let z = air.sample_point(&mut actual).unwrap();
        air.pcs
            .verify(&proof.commitment, z, &proof.opening, &mut actual)
            .unwrap();
        assert_eq!(actual.draws.len(), 3);
        let q = if classical { 52 } else { 12 };
        let s = if classical { 11 } else { 1 };
        let mut initial = Vec::new();
        b(&mut initial, b"p3-binary-reference-zk-air-v2");
        if classical {
            for v in [1, 1, 0, 49] {
                u(&mut initial, v);
            }
        } else {
            u(&mut initial, 0);
        }
        for v in [1, s, 2, 1, 1] {
            u(&mut initial, v);
        }
        b(&mut initial, &air.statement);
        u(&mut initial, 1);
        f(&mut initial, Val::ONE);
        let mut expected = initial.clone();
        let hroot = proof.commitment.root.roots()[0];
        commitment(&mut expected, &hroot, q);
        let first_commit = expected[initial.len()..].to_vec();
        let z_raw = raw(&expected, b"air-point", 0, 256);
        assert_eq!(actual.draws[0].prefix, expected);
        assert_eq!(actual.draws[0].label, b"air-point");
        assert_eq!(actual.draws[0].ordinal, 0);
        assert_eq!(actual.draws[0].raw, z_raw);
        draw(&mut expected, b"air-point", &z_raw);
        let before_duplicate = expected.len();
        commitment(&mut expected, &hroot, q);
        let duplicate_root_offset = expected.len() - 32;
        b(&mut expected, b"opening-point-and-claims");
        let selected_point_offset = expected.len();
        let selected_z = select(&z_raw, |v| v * (v + Val::ONE) != Val::ZERO);
        assert_eq!(z, selected_z);
        f(&mut expected, selected_z);
        u(&mut expected, 3);
        for &v in &proof.opening.claimed_values {
            f(&mut expected, v);
        }
        b(&mut expected, b"uniform-mask-commitment-and-functional");
        u(&mut expected, 1);
        cap(&mut expected, &proof.opening.mask_root.roots()[0]);
        u(&mut expected, 3);
        for &v in &proof.opening.sigma {
            f(&mut expected, v);
        }
        let gamma_raw = raw(&expected, b"blend-scalar", 1, 256);
        assert_eq!(actual.draws[1].prefix, expected);
        assert_eq!(actual.draws[1].label, b"blend-scalar");
        assert_eq!(actual.draws[1].ordinal, 1);
        assert_eq!(actual.draws[1].raw, gamma_raw);
        draw(&mut expected, b"blend-scalar", &gamma_raw);
        let after_gamma = expected.len();
        b(&mut expected, b"complete-monomial-blend-before-queries");
        u(&mut expected, 3);
        for g in &proof.opening.blended_coefficients {
            u(&mut expected, 3 + q + 1);
            for &v in g {
                f(&mut expected, v);
            }
        }
        let mut label = b"query-indices".to_vec();
        label.extend_from_slice(&(q as u64).to_le_bytes());
        label.extend_from_slice(&(if classical { 8u64 } else { 6 }).to_le_bytes());
        let indices_raw = raw(&expected, &label, 2, 8 * q);
        assert_eq!(actual.draws[2].prefix, expected);
        assert_eq!(actual.draws[2].label, label);
        assert_eq!(actual.draws[2].ordinal, 2);
        assert_eq!(actual.draws[2].raw, indices_raw);

        // Recognize reachable gamma/query prefixes, retaining arbitrary claims,
        // roots and blends but requiring repeated h geometry/root and the FIRST
        // accepted z. At query stage derive gamma from its raw candidates and
        // check every low-functional equation with that selected value.
        let canonical = |prefix: &[u8], query_stage: bool| -> Option<()> {
            let mut r = Records(prefix);
            r.literal(&expected[..before_duplicate])?;
            r.literal(&first_commit)?;
            let mut marker = Vec::new();
            b(&mut marker, b"opening-point-and-claims");
            r.literal(&marker)?;
            if r.field()? != selected_z {
                return None;
            }
            let mut count = Vec::new();
            u(&mut count, 3);
            r.literal(&count)?;
            let claims = [r.field()?, r.field()?, r.field()?];
            marker.clear();
            b(&mut marker, b"uniform-mask-commitment-and-functional");
            u(&mut marker, 1);
            r.literal(&marker)?;
            r.root()?;
            r.literal(&count)?;
            let sigma = [r.field()?, r.field()?, r.field()?];
            if query_stage {
                // D records must be exactly the fixed raw stream, retained in
                // full. This fixture computes the actual RO stream independently.
                r.literal(&expected[after_gamma - (1 + 8 + 12 + 8 + 256)..after_gamma])?;
                marker.clear();
                b(&mut marker, b"complete-monomial-blend-before-queries");
                u(&mut marker, 3);
                r.literal(&marker)?;
                let gamma = select(&gamma_raw, |v| v != Val::ZERO);
                for j in 0..3 {
                    marker.clear();
                    u(&mut marker, 3 + q + 1);
                    r.literal(&marker)?;
                    let g: Vec<_> = (0..3 + q + 1).map(|_| r.field()).collect::<Option<_>>()?;
                    if evaluate(&g[..3], selected_z) != gamma * claims[j] + sigma[j] {
                        return None;
                    }
                }
            }
            r.0.is_empty().then_some(())
        };
        assert!(canonical(&actual.draws[1].prefix, false).is_some());
        assert!(canonical(&actual.draws[2].prefix, true).is_some());
        let mut corrupted = actual.draws[2].prefix.clone();
        corrupted[duplicate_root_offset] ^= 1;
        assert!(canonical(&corrupted, true).is_none());
        let mut corrupted = actual.draws[2].prefix.clone();
        corrupted[selected_point_offset + 1] ^= 1;
        assert!(canonical(&corrupted, true).is_none());
        let mut corrupted = actual.draws[2].prefix.clone();
        corrupted[after_gamma - 256] ^= 1;
        assert!(canonical(&corrupted, true).is_none());
        // g must use selected gamma, not a prover-selected later candidate.
        let selected_gamma = select(&gamma_raw, |v| v != Val::ZERO);
        let later_gamma =
            Val::from_repr(u128::from_le_bytes(gamma_raw[16..32].try_into().unwrap()));
        assert_ne!(selected_gamma, later_gamma);
        let first_g =
            after_gamma + (1 + 8 + b"complete-monomial-blend-before-queries".len()) + 9 + 9 + 1;
        let mut wrong_selection = actual.draws[2].prefix.clone();
        let wrong_constant = proof.opening.blended_coefficients[0][0]
            + (selected_gamma - later_gamma) * proof.opening.claimed_values[0];
        wrong_selection[first_g..first_g + 16]
            .copy_from_slice(&wrong_constant.to_repr().to_le_bytes());
        assert!(canonical(&wrong_selection, true).is_none());
        let mut corrupted = actual.draws[2].prefix.clone();
        let last = corrupted.len() - 16;
        corrupted[last] ^= 1;
        // Last coefficient is outside lambda: changing it is still a reachable
        // message, while the first low coefficient must fail the equation.
        assert!(canonical(&corrupted, true).is_some());
        let first_g =
            after_gamma + (1 + 8 + b"complete-monomial-blend-before-queries".len()) + 9 + 9 + 1;
        corrupted[first_g] ^= 1;
        assert!(canonical(&corrupted, true).is_none());
    }
}

#[test]
fn rejects_pcs_valid_opening_of_a_trace_with_a_false_quotient() {
    let air = ReferenceZkAir::compile(&Product, ZkAirConfig::interactive(1, 1).unwrap()).unwrap();
    let public = [Val::ZERO];
    let mut rng = StdRng::seed_from_u64(0x464f_5247_4544);
    // Every row has product one, contradicting the public product zero.
    let trace = RowMajorMatrix::new(vec![Val::ONE; 4], 2);
    assert!(matches!(
        air.prove(trace.clone(), &public, &mut rng),
        Err(ZkError::InvalidWitness(_))
    ));
    let mut polynomials: Vec<_> = (0..air.width)
        .map(|column| {
            let values: Vec<_> = (0..2).map(|row| trace.values[row * 2 + column]).collect();
            add(
                &interpolate(&values, 1),
                &scale(&air.vanishing, rng.random()),
            )
        })
        .collect();
    // Bypass exact division: the zero quotient cannot certify this trace.
    polynomials.push(vec![]);
    let width = polynomials.len();
    let mut messages = vec![Val::ZERO; air.pcs.config().message_len() * width];
    for (column, polynomial) in polynomials.iter().enumerate() {
        for (row, &coefficient) in polynomial.iter().enumerate() {
            messages[row * width + column] = coefficient;
        }
    }
    let (commitment, data) = air
        .pcs
        .commit(RowMajorMatrix::new(messages, width), &mut rng)
        .unwrap();
    let mut prover = air.transcript(&public);
    air.pcs.observe_commitment(&commitment, &mut prover);
    let z = air.sample_point(&mut prover).unwrap();
    let opening = air.pcs.open(data, z, &mut prover, &mut rng).unwrap();
    // Authenticate the complete PCS proof first: only the AIR equation rejects it.
    let mut verifier = air.transcript(&public);
    air.pcs.observe_commitment(&commitment, &mut verifier);
    assert_eq!(air.sample_point(&mut verifier).unwrap(), z);
    air.pcs
        .verify(&commitment, z, &opening, &mut verifier)
        .unwrap();
    assert_eq!(
        air.verify(
            &ZkAirProof {
                commitment,
                opening
            },
            &public
        ),
        Err(ZkError::InvalidOpening("AIR quotient identity"))
    );
}
